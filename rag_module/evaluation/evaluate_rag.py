import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError, answer_question
from rag_module.retrieval.rag_search import run_hybrid_search_debug
from rag_module.shared.offline_pipeline_report import update_offline_pipeline_report


REPORT_DIR = PROJECT_ROOT / "data_storage" / "reports"
DEFAULT_PRECISION_GATE = float(os.getenv("RAG_KPI_PRECISION_GATE", "0.75") or 0.75)
DEFAULT_HIT_GATE = float(os.getenv("RAG_KPI_HIT_GATE", "0.90") or 0.90)
FALLBACK_MARKERS = [
    "information non disponible",
    "pas pu traiter",
    "erreur",
]

EVAL_SET: List[Dict] = [
    {"question": "Quelles sont les conditions d'inscription en licence ?", "keywords": ["inscription", "licence", "dossier"], "expected_doc_types": ["inscription", "formation"]},
    {"question": "Quels documents sont requis pour une inscription administrative ?", "keywords": ["documents", "inscription", "administrative"], "expected_doc_types": ["inscription"]},
    {"question": "Comment faire la preinscription en ligne ?", "keywords": ["preinscription", "ligne", "procedure"], "expected_doc_types": ["inscription"]},
    {"question": "Quels sont les delais d'inscription pour le master ?", "keywords": ["delai", "inscription", "master"], "expected_doc_types": ["inscription", "formation"]},
    {"question": "Comment verifier les resultats d'admission ?", "keywords": ["resultat", "admission", "liste"], "expected_doc_types": ["admission", "resultats"]},
    {"question": "Quelles sont les modalites du concours d'acces ?", "keywords": ["concours", "acces", "modalites"], "expected_doc_types": ["admission"]},
    {"question": "Comment obtenir une bourse universitaire ?", "keywords": ["bourse", "conditions", "demande"], "expected_doc_types": ["bourse"]},
    {"question": "Ou trouver le calendrier pedagogique ?", "keywords": ["calendrier", "pedagogique", "semestre"], "expected_doc_types": ["calendrier"]},
    {"question": "Quelles sont les filieres disponibles en master ?", "keywords": ["filiere", "master", "formation"], "expected_doc_types": ["formation"]},
    {"question": "Comment se passe la reinscription ?", "keywords": ["reinscription", "inscription", "dossier"], "expected_doc_types": ["inscription"]},
    {"question": "Quels sont les frais d'inscription ?", "keywords": ["frais", "inscription", "paiement"], "expected_doc_types": ["inscription"]},
    {"question": "Ou consulter l'emploi du temps ?", "keywords": ["emploi du temps", "planning", "cours"], "expected_doc_types": ["calendrier"]},
    {"question": "Comment contacter le service de scolarite ?", "keywords": ["scolarite", "contact", "service"], "expected_doc_types": ["inscription"]},
    {"question": "Quels sont les criteres de selection en master ?", "keywords": ["selection", "master", "criteres"], "expected_doc_types": ["admission", "formation"]},
    {"question": "Comment retirer une attestation d'inscription ?", "keywords": ["attestation", "inscription", "retrait"], "expected_doc_types": ["inscription"]},
    {"question": "Quelles sont les etapes de candidature doctorale ?", "keywords": ["candidature", "doctorat", "etapes"], "expected_doc_types": ["admission", "formation"]},
    {"question": "Comment connaitre les dates des rattrapages ?", "keywords": ["dates", "rattrapage", "calendrier"], "expected_doc_types": ["resultats", "calendrier"]},
    {"question": "Quelles pieces sont demandees pour une equivalence ?", "keywords": ["pieces", "equivalence", "dossier"], "expected_doc_types": ["inscription"]},
    {"question": "Comment suivre l'etat de ma candidature ?", "keywords": ["candidature", "etat", "suivi"], "expected_doc_types": ["admission"]},
    {"question": "Ou trouver les annonces officielles d'admission ?", "keywords": ["annonces", "admission", "officielles"], "expected_doc_types": ["admission"]},
]


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(value: str) -> Set[str]:
    return set(re.findall(r"\b[\w']+\b", _normalize_text(value)))


def _keyword_coverage(text: str, keywords: List[str]) -> Dict[str, object]:
    normalized_text = _normalize_text(text)
    text_tokens = _tokenize(normalized_text)

    matched: List[str] = []
    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        if not normalized_keyword:
            continue
        if " " in normalized_keyword and normalized_keyword in normalized_text:
            matched.append(keyword)
            continue
        keyword_tokens = _tokenize(normalized_keyword)
        if keyword_tokens and keyword_tokens.issubset(text_tokens):
            matched.append(keyword)

    coverage = len(matched) / len(keywords) if keywords else 0.0
    return {"score": round(coverage, 4), "matched_keywords": matched}


def _doc_type_match(chunk: Dict, expected_doc_types: List[str]) -> bool:
    metadata = chunk.get("metadata", {}) or {}
    doc_type = _normalize_text(str(metadata.get("document_type") or ""))
    return bool(doc_type) and doc_type in {_normalize_text(item) for item in expected_doc_types}


def _chunk_relevance(chunk: Dict, keywords: List[str], expected_doc_types: List[str]) -> Dict[str, object]:
    text = chunk.get("text", "") or ""
    coverage = _keyword_coverage(text, keywords)
    score = float(coverage["score"])

    if expected_doc_types and _doc_type_match(chunk, expected_doc_types):
        score = min(1.0, score + 0.25)

    metadata = chunk.get("metadata", {}) or {}
    source_hint = " ".join(
        [
            str(metadata.get("file_name") or ""),
            str(metadata.get("source") or ""),
            str(metadata.get("document_type") or ""),
        ]
    )
    source_coverage = _keyword_coverage(source_hint, keywords)
    score = min(1.0, score + (0.15 if float(source_coverage["score"]) >= 0.34 else 0.0))

    return {
        "score": round(score, 4),
        "matched_keywords": coverage["matched_keywords"],
        "relevant": score >= 0.45,
    }


def _answer_relevance(answer: str, keywords: List[str], expected_doc_types: List[str]) -> Dict[str, object]:
    coverage = _keyword_coverage(answer, keywords)
    score = float(coverage["score"])
    normalized_answer = _normalize_text(answer)
    if expected_doc_types and any(_normalize_text(doc_type) in normalized_answer for doc_type in expected_doc_types):
        score = min(1.0, score + 0.15)
    return {
        "score": round(score, 4),
        "matched_keywords": coverage["matched_keywords"],
        "useful": score >= 0.34,
    }


def _stage_metrics(chunks: List[Dict], keywords: List[str], expected_doc_types: List[str], top_k: int) -> Dict[str, float]:
    selected = chunks[:top_k]
    if not selected:
        return {"hit": 0.0, "coverage": 0.0, "best": 0.0}

    scores = [_chunk_relevance(chunk, keywords, expected_doc_types) for chunk in selected]
    relevant = [item for item in scores if item["relevant"]]
    return {
        "hit": float(bool(relevant)),
        "coverage": round(mean(float(item["score"]) for item in scores), 4),
        "best": round(max(float(item["score"]) for item in scores), 4),
    }


def _retrieval_metrics(question: str, keywords: List[str], expected_doc_types: List[str], top_k: int) -> Dict:
    start = time.perf_counter()
    payload = run_hybrid_search_debug(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    final_chunks = list(payload.get("final_results", []))
    if not final_chunks:
        return {
            "precision_at_k": 0.0,
            "coverage_at_k": 0.0,
            "hit_at_k": 0,
            "dense_hit_at_k": 0,
            "bm25_hit_at_k": 0,
            "fusion_hit_at_k": 0,
            "latency_ms": round(elapsed_ms, 2),
            "retrieved": 0,
            "relevant": 0,
            "best_match_score": 0.0,
            "metadata_boost_gain": 0.0,
            "rerank_gain": 0.0,
        }

    final_scores = [_chunk_relevance(chunk, keywords, expected_doc_types) for chunk in final_chunks]
    relevant = sum(1 for item in final_scores if item["relevant"])
    precision = relevant / len(final_chunks)
    avg_coverage = mean(float(item["score"]) for item in final_scores)
    best_match = max(float(item["score"]) for item in final_scores)

    dense_stage = _stage_metrics(list(payload.get("dense_results", [])), keywords, expected_doc_types, top_k)
    bm25_stage = _stage_metrics(list(payload.get("bm25_results", [])), keywords, expected_doc_types, top_k)
    fusion_stage = _stage_metrics(list(payload.get("merged_results", [])), keywords, expected_doc_types, top_k)
    boosted_stage = _stage_metrics(list(payload.get("boosted_results", [])), keywords, expected_doc_types, top_k)
    final_stage = _stage_metrics(final_chunks, keywords, expected_doc_types, top_k)

    return {
        "precision_at_k": round(precision, 4),
        "coverage_at_k": round(avg_coverage, 4),
        "hit_at_k": int(relevant > 0),
        "dense_hit_at_k": int(dense_stage["hit"] > 0),
        "bm25_hit_at_k": int(bm25_stage["hit"] > 0),
        "fusion_hit_at_k": int(fusion_stage["hit"] > 0),
        "latency_ms": round(elapsed_ms, 2),
        "retrieved": len(final_chunks),
        "relevant": relevant,
        "best_match_score": round(best_match, 4),
        "metadata_boost_gain": round(float(boosted_stage["best"]) - float(fusion_stage["best"]), 4),
        "rerank_gain": round(float(final_stage["best"]) - float(boosted_stage["best"]), 4),
    }


def _generation_metrics(question: str, keywords: List[str], expected_doc_types: List[str]) -> Dict:
    start = time.perf_counter()
    try:
        payload = answer_question(question)
        answer = payload.get("answer", "")
        error = ""
    except (RAGIndexNotReadyError, RAGGenerationError, ValueError) as exc:
        answer = ""
        error = str(exc)
    except Exception as exc:
        answer = ""
        error = f"unexpected: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000

    lower = _normalize_text(answer)
    relevance = _answer_relevance(answer, keywords, expected_doc_types)
    useful = bool(answer.strip()) and bool(relevance["useful"]) and not any(marker in lower for marker in FALLBACK_MARKERS)
    return {
        "useful_answer": int(useful),
        "answer_relevance_score": relevance["score"],
        "answer_latency_ms": round(elapsed_ms, 2),
        "answer_preview": answer[:180],
        "matched_keywords": relevance["matched_keywords"],
        "error": error,
    }


def evaluate(top_k: int, run_generation: bool) -> Dict:
    rows = []
    for case in EVAL_SET:
        question = case["question"]
        keywords = case["keywords"]
        expected_doc_types = case.get("expected_doc_types", [])
        row = {"question": question, "keywords": keywords, "expected_doc_types": expected_doc_types}
        try:
            row.update(_retrieval_metrics(question, keywords, expected_doc_types, top_k))
        except Exception as exc:
            row.update(
                {
                    "precision_at_k": 0.0,
                    "coverage_at_k": 0.0,
                    "hit_at_k": 0,
                    "dense_hit_at_k": 0,
                    "bm25_hit_at_k": 0,
                    "fusion_hit_at_k": 0,
                    "latency_ms": 0.0,
                    "retrieved": 0,
                    "relevant": 0,
                    "best_match_score": 0.0,
                    "metadata_boost_gain": 0.0,
                    "rerank_gain": 0.0,
                    "retrieval_error": str(exc),
                }
            )
            rows.append(row)
            break

        if run_generation:
            row.update(_generation_metrics(question, keywords, expected_doc_types))

        rows.append(row)

    retrieval_latencies = [r.get("latency_ms", 0.0) for r in rows if r.get("latency_ms", 0.0) > 0]
    report = {
        "generated_at": datetime.now().isoformat(),
        "top_k": top_k,
        "questions_evaluated": len(rows),
        "summary": {
            "precision_at_k_avg": round(mean([r.get("precision_at_k", 0.0) for r in rows]), 4) if rows else 0.0,
            "coverage_at_k_avg": round(mean([r.get("coverage_at_k", 0.0) for r in rows]), 4) if rows else 0.0,
            "hit_at_k_rate": round(mean([r.get("hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "dense_hit_at_k_rate": round(mean([r.get("dense_hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "bm25_hit_at_k_rate": round(mean([r.get("bm25_hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "fusion_hit_at_k_rate": round(mean([r.get("fusion_hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "best_match_score_avg": round(mean([r.get("best_match_score", 0.0) for r in rows]), 4) if rows else 0.0,
            "metadata_boost_gain_avg": round(mean([r.get("metadata_boost_gain", 0.0) for r in rows]), 4) if rows else 0.0,
            "rerank_gain_avg": round(mean([r.get("rerank_gain", 0.0) for r in rows]), 4) if rows else 0.0,
            "retrieval_latency_ms_avg": round(mean(retrieval_latencies), 2) if retrieval_latencies else 0.0,
        },
        "rows": rows,
    }

    if run_generation:
        answer_latencies = [r.get("answer_latency_ms", 0.0) for r in rows if r.get("answer_latency_ms", 0.0) > 0]
        report["summary"]["useful_answer_rate"] = round(mean([r.get("useful_answer", 0) for r in rows]), 4) if rows else 0.0
        report["summary"]["answer_relevance_score_avg"] = round(mean([r.get("answer_relevance_score", 0.0) for r in rows]), 4) if rows else 0.0
        report["summary"]["answer_latency_ms_avg"] = round(mean(answer_latencies), 2) if answer_latencies else 0.0

    return report


def evaluate_kpi_gates(report: Dict, precision_gate: float, hit_gate: float) -> Dict:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    precision = float(summary.get("precision_at_k_avg", 0.0) or 0.0)
    hit_rate = float(summary.get("hit_at_k_rate", 0.0) or 0.0)
    passed = precision >= float(precision_gate) and hit_rate >= float(hit_gate)
    return {
        "passed": passed,
        "precision_at_k_avg": round(precision, 4),
        "hit_at_k_rate": round(hit_rate, 4),
        "precision_gate": round(float(precision_gate), 4),
        "hit_gate": round(float(hit_gate), 4),
    }


def write_report(report: Dict) -> Dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"rag_eval_{timestamp}.json"
    txt_path = REPORT_DIR / f"rag_eval_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "RAG EVALUATION",
        f"Generated at: {report['generated_at']}",
        f"Top-k: {report['top_k']}",
        f"Questions evaluated: {report['questions_evaluated']}",
        "",
        f"Precision@k (avg): {report['summary'].get('precision_at_k_avg', 0.0)}",
        f"Coverage@k (avg): {report['summary'].get('coverage_at_k_avg', 0.0)}",
        f"Dense hit@k rate: {report['summary'].get('dense_hit_at_k_rate', 0.0)}",
        f"BM25 hit@k rate: {report['summary'].get('bm25_hit_at_k_rate', 0.0)}",
        f"Fusion hit@k rate: {report['summary'].get('fusion_hit_at_k_rate', 0.0)}",
        f"Best match score (avg): {report['summary'].get('best_match_score_avg', 0.0)}",
        f"Metadata boost gain (avg): {report['summary'].get('metadata_boost_gain_avg', 0.0)}",
        f"Rerank gain (avg): {report['summary'].get('rerank_gain_avg', 0.0)}",
        f"Hit@k rate: {report['summary'].get('hit_at_k_rate', 0.0)}",
        f"Retrieval latency avg (ms): {report['summary'].get('retrieval_latency_ms_avg', 0.0)}",
    ]
    if "useful_answer_rate" in report.get("summary", {}):
        lines.extend(
            [
                f"Useful answer rate: {report['summary'].get('useful_answer_rate', 0.0)}",
                f"Answer relevance score avg: {report['summary'].get('answer_relevance_score_avg', 0.0)}",
                f"Answer latency avg (ms): {report['summary'].get('answer_latency_ms_avg', 0.0)}",
            ]
        )
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return {"json": json_path, "txt": txt_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation heuristique hybride du RAG (dense, BM25, fusion et generation).")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks recuperes pour l'evaluation.")
    parser.add_argument("--skip-generation", action="store_true", help="N'evalue que la retrieval sans generation de reponse.")
    parser.add_argument("--precision-gate", type=float, default=DEFAULT_PRECISION_GATE, help="Seuil Precision@k moyen.")
    parser.add_argument("--hit-gate", type=float, default=DEFAULT_HIT_GATE, help="Seuil Hit@k rate.")
    parser.add_argument("--enforce-kpi-gates", action="store_true", help="Retourne un code d'erreur si les gates KPI echouent.")
    args = parser.parse_args()

    top_k = max(1, args.top_k)
    report = evaluate(top_k=top_k, run_generation=not args.skip_generation)
    kpi_gates = evaluate_kpi_gates(report, precision_gate=args.precision_gate, hit_gate=args.hit_gate)
    report["kpi_gates"] = kpi_gates
    output_paths = write_report(report)
    update_offline_pipeline_report("kpi_gates", {
        "generated_at": datetime.now().isoformat(),
        "top_k": top_k,
        "kpi_gates": kpi_gates,
    })
    print(f"Evaluation terminee. JSON: {output_paths['json']}")
    print(f"Evaluation terminee. TXT : {output_paths['txt']}")
    print(
        f"KPI gates: {'PASS' if kpi_gates['passed'] else 'FAIL'} | "
        f"precision={kpi_gates['precision_at_k_avg']} (>= {kpi_gates['precision_gate']}), "
        f"hit={kpi_gates['hit_at_k_rate']} (>= {kpi_gates['hit_gate']})"
    )
    if args.enforce_kpi_gates and not kpi_gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
