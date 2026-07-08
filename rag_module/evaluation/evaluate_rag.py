import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

from rag_module.adapters.storage import DocumentStorage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError, answer_question
from rag_module.retrieval.rag_search import run_hybrid_search_debug
from api_app.services.conversation_context import build_conversation_context, update_conversation_context

REPORT_DIR = DocumentStorage().report_dir
CONTEXT_EVAL_DATASET = Path(__file__).with_name("context_eval_dataset.json")
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

DRIVE_EVAL_SET: List[Dict] = [
    {"question": "Comment obtenir mon attestation sur UC@Student ?", "keywords": ["attestation", "ucastudent"], "expected_doc_types": ["scolarite"], "expected_service": "UC@Student"},
    {"question": "Ou consulter mes notes sur UC@Student ?", "keywords": ["notes", "ucastudent"], "expected_doc_types": ["scolarite"], "expected_service": "UC@Student"},
    {"question": "Comment candidater sur PEDOC ?", "keywords": ["candidature", "pedoc"], "expected_doc_types": ["scolarite"], "expected_service": "PEDOC"},
    {"question": "A quoi sert UCAPLAT ?", "keywords": ["ucaplat", "cours", "devoirs"], "expected_doc_types": ["pedagogie_numerique"], "expected_service": "UCAPLAT"},
    {"question": "Comment deposer des devoirs sur UCAPLAT ?", "keywords": ["ucaplat", "devoirs", "deposer"], "expected_doc_types": ["pedagogie_numerique"], "expected_service": "UCAPLAT"},
    {"question": "A quoi sert le CIP ?", "keywords": ["cip", "accompagnement", "guides"], "expected_doc_types": ["pedagogie_numerique"], "expected_service": "CIP"},
    {"question": "Comment demander un conge sur PUCAStaff ?", "keywords": ["pucastaff", "conge"], "expected_doc_types": ["rh"], "expected_service": "PUCAStaff"},
    {"question": "Comment suivre l etat de mon diplome ?", "keywords": ["diplome", "suivi", "etat"], "expected_doc_types": ["scolarite"], "expected_service": "Espace Diplômes"},
    {"question": "Comment obtenir un e-diplome ?", "keywords": ["e-diplome", "diplome"], "expected_doc_types": ["scolarite"], "expected_service": "Espace Diplômes"},
    {"question": "Comment postuler a une bourse via Mobilite internationale ?", "keywords": ["mobilite", "bourse", "postuler"], "expected_doc_types": ["vie_etudiante"], "expected_service": "Mobilité internationale"},
    {"question": "Comment acceder au calcul haute performance de UCA ?", "keywords": ["hpc", "calcul", "haute performance"], "expected_doc_types": ["recherche"], "expected_service": "HPC UCA"},
    {"question": "Ou consulter les appels a projets de recherche ?", "keywords": ["appels a projets", "recherche", "projets"], "expected_doc_types": ["recherche"], "expected_service": "Appels à Projets"},
    {"question": "Ou trouver un accompagnement pour monter un projet de recherche ?", "keywords": ["accompagnement", "projet de recherche", "soutien"], "expected_doc_types": ["recherche"], "expected_service": "Soutien-Recherche"},
]

def load_benchmark_set(benchmark: str) -> List[Dict]:
    if benchmark == "context":
        return []
    file_path = Path(__file__).parent / f"{benchmark}_eval_dataset.json"
    if not file_path.exists():
        initial_data = DRIVE_EVAL_SET if benchmark == "drive" else EVAL_SET
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return initial_data
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DRIVE_EVAL_SET if benchmark == "drive" else EVAL_SET

def save_benchmark_set(benchmark: str, dataset: List[Dict]) -> None:
    if benchmark == "context":
        return
    file_path = Path(__file__).parent / f"{benchmark}_eval_dataset.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

BENCHMARK_SETS: Dict[str, List[Dict]] = {
    "generic": EVAL_SET,
    "drive": DRIVE_EVAL_SET,
}


class _EvalMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _EvalMessageStore:
    def __init__(self):
        self._items: list[_EvalMessage] = []

    def order_by(self, *args):
        return list(self._items)

    def add(self, role: str, content: str) -> None:
        self._items.append(_EvalMessage(role, content))


class _EvalConversation:
    def __init__(self):
        self.context_summary = ""
        self.context_meta: Dict[str, object] = {}
        self.messages = _EvalMessageStore()

    def save(self, update_fields=None) -> None:
        return None


class _EvalResult:
    def __init__(self, sources: List[Dict] | None = None):
        self.sources = sources or []


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


def _service_match(service_name: str, expected_service: str) -> bool:
    return bool(expected_service) and _normalize_text(service_name) == _normalize_text(expected_service)


def _rewrite_match(rewritten_query: str, expected_rewritten_query: str, expected_keywords: List[str]) -> int:
    if expected_rewritten_query:
        expected_tokens = _tokenize(expected_rewritten_query)
        rewritten_tokens = _tokenize(rewritten_query)
        if not expected_tokens:
            return 1
        overlap = len(expected_tokens.intersection(rewritten_tokens)) / len(expected_tokens)
        return int(overlap >= 0.6)
    coverage = _keyword_coverage(rewritten_query, expected_keywords)
    return int(float(coverage["score"]) >= 0.5)


def _top_result_metadata(chunks: List[Dict]) -> Dict[str, object]:
    if not chunks:
        return {
            "top1_service": "",
            "top1_source": "",
            "top1_doc_type": "",
        }

    metadata = (chunks[0] or {}).get("metadata", {}) or {}
    return {
        "top1_service": str(metadata.get("service_name") or ""),
        "top1_source": str(metadata.get("file_name") or metadata.get("source") or ""),
        "top1_doc_type": str(metadata.get("document_type") or ""),
    }


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


def _retrieval_metrics(
    question: str,
    keywords: List[str],
    expected_doc_types: List[str],
    top_k: int,
    expected_service: str = "",
) -> Dict:
    start = time.perf_counter()
    payload = run_hybrid_search_debug(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    final_chunks = list(payload.get("final_results", []))
    top1_metadata = _top_result_metadata(final_chunks)
    abstained = bool(payload.get("abstain", False))
    abstain_reason = str(payload.get("abstain_reason") or "")
    top1_service_match = int(_service_match(str(top1_metadata["top1_service"]), expected_service)) if expected_service else 0
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
            "expected_service": expected_service,
            "top1_service": "",
            "top1_source": "",
            "top1_doc_type": "",
            "service_top1_match": 0,
            "abstained": int(abstained),
            "abstain_reason": abstain_reason,
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
        "expected_service": expected_service,
        "top1_service": str(top1_metadata["top1_service"]),
        "top1_source": str(top1_metadata["top1_source"]),
        "top1_doc_type": str(top1_metadata["top1_doc_type"]),
        "service_top1_match": top1_service_match,
        "abstained": int(abstained),
        "abstain_reason": abstain_reason,
    }


def _load_context_eval_set() -> List[Dict]:
    with CONTEXT_EVAL_DATASET.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def evaluate_context(top_k: int) -> Dict:
    conversations = _load_context_eval_set()
    rows: List[Dict] = []
    for conversation_case in conversations:
        conversation = _EvalConversation()
        conversation_id = str(conversation_case.get("conversation_id") or "")
        for turn_index, turn in enumerate(conversation_case.get("turns", []), start=1):
            question = str(turn.get("question") or "")
            expected_keywords = list(turn.get("expected_keywords") or turn.get("keywords") or [])
            expected_service = str(turn.get("expected_service") or "")
            expected_intent = str(turn.get("expected_intent") or "")
            expected_rewritten = str(turn.get("expected_rewritten_query") or "")
            expected_context_needed = bool(turn.get("context_needed", False))
            expected_abstain = bool(turn.get("expected_abstain", False))

            context_payload = build_conversation_context(conversation, question)
            rewritten_query = str(context_payload.get("rewritten_question") or question)
            row = {
                "conversation_id": conversation_id,
                "title": str(conversation_case.get("title") or ""),
                "turn_index": turn_index,
                "question": question,
                "rewritten_query": rewritten_query,
                "expected_rewritten_query": expected_rewritten,
                "context_needed": int(expected_context_needed),
                "context_used": int(bool(context_payload.get("context_used"))),
                "context_used_match": int(bool(context_payload.get("context_used")) == expected_context_needed),
                "rewrite_match": _rewrite_match(rewritten_query, expected_rewritten, expected_keywords),
                "expected_service": expected_service,
                "expected_intent": expected_intent,
                "context_service": str((context_payload.get("context_meta") or {}).get("service", "")),
                "context_intent": str((context_payload.get("context_meta") or {}).get("intent", "")),
                "expected_abstain": int(expected_abstain),
            }
            try:
                row.update(_retrieval_metrics(rewritten_query, expected_keywords, [], top_k, expected_service=expected_service))
            except Exception as exc:
                row.update(
                    {
                        "precision_at_k": 0.0,
                        "coverage_at_k": 0.0,
                        "hit_at_k": 0,
                        "latency_ms": 0.0,
                        "retrieved": 0,
                        "relevant": 0,
                        "best_match_score": 0.0,
                        "service_top1_match": 0,
                        "abstained": 0,
                        "abstain_reason": "",
                        "retrieval_error": str(exc),
                    }
                )

            abstained = bool(row.get("abstained", 0)) or int(row.get("retrieved", 0) or 0) == 0
            row["abstention_correct"] = int(abstained == expected_abstain)
            rows.append(row)

            conversation.messages.add("user", question)
            conversation.messages.add("assistant", "")
            update_conversation_context(
                conversation,
                question,
                "",
                _EvalResult(sources=[{"service_name": expected_service}] if expected_service else []),
                context_payload=context_payload,
            )

    retrieval_latencies = [r.get("latency_ms", 0.0) for r in rows if r.get("latency_ms", 0.0) > 0]
    report = {
        "generated_at": datetime.now().isoformat(),
        "benchmark": "context",
        "top_k": top_k,
        "conversations_evaluated": len(conversations),
        "turns_evaluated": len(rows),
        "questions_evaluated": len(rows),
        "summary": {
            "rewrite_match_rate": round(mean([r.get("rewrite_match", 0) for r in rows]), 4) if rows else 0.0,
            "service_top1_accuracy": round(mean([r.get("service_top1_match", 0) for r in rows]), 4) if rows else 0.0,
            "hit_at_k_rate": round(mean([r.get("hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "coverage_at_k_avg": round(mean([r.get("coverage_at_k", 0.0) for r in rows]), 4) if rows else 0.0,
            "context_used_accuracy": round(mean([r.get("context_used_match", 0) for r in rows]), 4) if rows else 0.0,
            "abstention_correctness": round(mean([r.get("abstention_correct", 0) for r in rows]), 4) if rows else 0.0,
            "retrieval_latency_ms_avg": round(mean(retrieval_latencies), 2) if retrieval_latencies else 0.0,
        },
        "rows": rows,
    }
    return report


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


def evaluate(top_k: int, run_generation: bool, benchmark: str = "drive") -> Dict:
    if benchmark == "context":
        return evaluate_context(top_k=max(1, top_k))

    eval_rows = load_benchmark_set(benchmark)
    rows = []
    for case in eval_rows:
        question = case["question"]
        keywords = case["keywords"]
        expected_doc_types = case.get("expected_doc_types", [])
        expected_service = str(case.get("expected_service") or "")
        row = {
            "question": question,
            "keywords": keywords,
            "expected_doc_types": expected_doc_types,
            "expected_service": expected_service,
        }
        try:
            row.update(_retrieval_metrics(question, keywords, expected_doc_types, top_k, expected_service=expected_service))
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
                    "top1_service": "",
                    "top1_source": "",
                    "top1_doc_type": "",
                    "service_top1_match": 0,
                    "abstained": 0,
                    "abstain_reason": "",
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
        "benchmark": benchmark,
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
            "service_top1_accuracy": round(mean([r.get("service_top1_match", 0) for r in rows]), 4) if rows and any(r.get("expected_service") for r in rows) else 0.0,
            "abstention_rate": round(mean([r.get("abstained", 0) for r in rows]), 4) if rows else 0.0,
        },
        "rows": rows,
    }

    if run_generation:
        answer_latencies = [r.get("answer_latency_ms", 0.0) for r in rows if r.get("answer_latency_ms", 0.0) > 0]
        report["summary"]["useful_answer_rate"] = round(mean([r.get("useful_answer", 0) for r in rows]), 4) if rows else 0.0
        report["summary"]["answer_relevance_score_avg"] = round(mean([r.get("answer_relevance_score", 0.0) for r in rows]), 4) if rows else 0.0
        report["summary"]["answer_latency_ms_avg"] = round(mean(answer_latencies), 2) if answer_latencies else 0.0

    return report


def write_report(report: Dict) -> Dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark = str(report.get("benchmark") or "generic")
    json_path = REPORT_DIR / f"rag_eval_{benchmark}_{timestamp}.json"
    txt_path = REPORT_DIR / f"rag_eval_{benchmark}_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if benchmark == "context":
        lines = [
            "CONTEXT RAG EVALUATION",
            f"Generated at: {report['generated_at']}",
            f"Benchmark: {report.get('benchmark', 'context')}",
            f"Top-k: {report['top_k']}",
            f"Conversations evaluated: {report.get('conversations_evaluated', 0)}",
            f"Turns evaluated: {report.get('turns_evaluated', 0)}",
            "",
            f"Rewrite match rate: {report['summary'].get('rewrite_match_rate', 0.0)}",
            f"Service top1 accuracy: {report['summary'].get('service_top1_accuracy', 0.0)}",
            f"Hit@k rate: {report['summary'].get('hit_at_k_rate', 0.0)}",
            f"Coverage@k avg: {report['summary'].get('coverage_at_k_avg', 0.0)}",
            f"Context used accuracy: {report['summary'].get('context_used_accuracy', 0.0)}",
            f"Abstention correctness: {report['summary'].get('abstention_correctness', 0.0)}",
            f"Retrieval latency avg (ms): {report['summary'].get('retrieval_latency_ms_avg', 0.0)}",
        ]
        with txt_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return {"json": json_path, "txt": txt_path}

    lines = [
        "RAG EVALUATION",
        f"Generated at: {report['generated_at']}",
        f"Benchmark: {report.get('benchmark', 'generic')}",
        f"Top-k: {report['top_k']}",
        f"Questions evaluated: {report['questions_evaluated']}",
        "",
        f"Precision@k (avg): {report['summary'].get('precision_at_k_avg', 0.0)}",
        f"Coverage@k (avg): {report['summary'].get('coverage_at_k_avg', 0.0)}",
        f"Dense hit@k rate: {report['summary'].get('dense_hit_at_k_rate', 0.0)}",
        f"BM25 hit@k rate: {report['summary'].get('bm25_hit_at_k_rate', 0.0)}",
        f"Fusion hit@k rate: {report['summary'].get('fusion_hit_at_k_rate', 0.0)}",
        f"Best match score (avg): {report['summary'].get('best_match_score_avg', 0.0)}",
        f"Service top1 accuracy: {report['summary'].get('service_top1_accuracy', 0.0)}",
        f"Abstention rate: {report['summary'].get('abstention_rate', 0.0)}",
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
    parser.add_argument("--benchmark", choices=["generic", "drive", "context"], default="drive", help="Jeu d'evaluation a utiliser.")
    args = parser.parse_args()

    top_k = max(1, args.top_k)
    report = evaluate(top_k=top_k, run_generation=not args.skip_generation, benchmark=args.benchmark)
    output_paths = write_report(report)
    print(f"Evaluation terminee. JSON: {output_paths['json']}")
    print(f"Evaluation terminee. TXT : {output_paths['txt']}")


if __name__ == "__main__":
    main()
