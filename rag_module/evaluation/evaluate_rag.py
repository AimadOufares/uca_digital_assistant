import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError, answer_question
from rag_module.retrieval.rag_search import get_relevant_chunks


REPORT_DIR = PROJECT_ROOT / "data_storage" / "reports"
FALLBACK_MARKERS = [
    "information non disponible",
    "pas pu traiter",
    "erreur",
]

EVAL_SET: List[Dict] = [
    {"question": "Quelles sont les conditions d'inscription en licence ?", "keywords": ["inscription", "licence", "dossier"]},
    {"question": "Quels documents sont requis pour une inscription administrative ?", "keywords": ["documents", "inscription", "administrative"]},
    {"question": "Comment faire la preinscription en ligne ?", "keywords": ["preinscription", "ligne", "procedure"]},
    {"question": "Quels sont les delais d'inscription pour le master ?", "keywords": ["delai", "inscription", "master"]},
    {"question": "Comment verifier les resultats d'admission ?", "keywords": ["resultat", "admission", "liste"]},
    {"question": "Quelles sont les modalites du concours d'acces ?", "keywords": ["concours", "acces", "modalites"]},
    {"question": "Comment obtenir une bourse universitaire ?", "keywords": ["bourse", "conditions", "demande"]},
    {"question": "Ou trouver le calendrier pedagogique ?", "keywords": ["calendrier", "pedagogique", "semestre"]},
    {"question": "Quelles sont les filieres disponibles en master ?", "keywords": ["filiere", "master", "formation"]},
    {"question": "Comment se passe la reinscription ?", "keywords": ["reinscription", "inscription", "dossier"]},
    {"question": "Quels sont les frais d'inscription ?", "keywords": ["frais", "inscription", "paiement"]},
    {"question": "Ou consulter l'emploi du temps ?", "keywords": ["emploi du temps", "planning", "cours"]},
    {"question": "Comment contacter le service de scolarite ?", "keywords": ["scolarite", "contact", "service"]},
    {"question": "Quels sont les criteres de selection en master ?", "keywords": ["selection", "master", "criteres"]},
    {"question": "Comment retirer une attestation d'inscription ?", "keywords": ["attestation", "inscription", "retrait"]},
    {"question": "Quelles sont les etapes de candidature doctorale ?", "keywords": ["candidature", "doctorat", "etapes"]},
    {"question": "Comment connaitre les dates des rattrapages ?", "keywords": ["dates", "rattrapage", "calendrier"]},
    {"question": "Quelles pieces sont demandees pour une equivalence ?", "keywords": ["pieces", "equivalence", "dossier"]},
    {"question": "Comment suivre l'etat de ma candidature ?", "keywords": ["candidature", "etat", "suivi"]},
    {"question": "Ou trouver les annonces officielles d'admission ?", "keywords": ["annonces", "admission", "officielles"]},
]


def _contains_any(text: str, keywords: List[str]) -> bool:
    lower_text = (text or "").lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _retrieval_metrics(question: str, keywords: List[str], top_k: int) -> Dict:
    start = time.perf_counter()
    chunks = get_relevant_chunks(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not chunks:
        return {
            "precision_at_k": 0.0,
            "hit_at_k": 0,
            "latency_ms": round(elapsed_ms, 2),
            "retrieved": 0,
            "relevant": 0,
        }

    relevant = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if _contains_any(text, keywords):
            relevant += 1

    precision = relevant / len(chunks)
    return {
        "precision_at_k": round(precision, 4),
        "hit_at_k": int(relevant > 0),
        "latency_ms": round(elapsed_ms, 2),
        "retrieved": len(chunks),
        "relevant": relevant,
    }


def _generation_metrics(question: str, keywords: List[str]) -> Dict:
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

    lower = answer.lower()
    useful = bool(answer.strip()) and _contains_any(answer, keywords) and not any(
        marker in lower for marker in FALLBACK_MARKERS
    )
    return {
        "useful_answer": int(useful),
        "answer_latency_ms": round(elapsed_ms, 2),
        "answer_preview": answer[:180],
        "error": error,
    }


def evaluate(top_k: int, run_generation: bool) -> Dict:
    rows = []
    for case in EVAL_SET:
        question = case["question"]
        keywords = case["keywords"]
        row = {
            "question": question,
            "keywords": keywords,
        }
        try:
            row.update(_retrieval_metrics(question, keywords, top_k))
        except Exception as exc:
            row.update(
                {
                    "precision_at_k": 0.0,
                    "hit_at_k": 0,
                    "latency_ms": 0.0,
                    "retrieved": 0,
                    "relevant": 0,
                    "retrieval_error": str(exc),
                }
            )
            rows.append(row)
            break

        if run_generation:
            row.update(_generation_metrics(question, keywords))

        rows.append(row)

    retrieval_latencies = [r.get("latency_ms", 0.0) for r in rows if r.get("latency_ms", 0.0) > 0]
    report = {
        "generated_at": datetime.now().isoformat(),
        "top_k": top_k,
        "questions_evaluated": len(rows),
        "summary": {
            "precision_at_k_avg": round(mean([r.get("precision_at_k", 0.0) for r in rows]), 4) if rows else 0.0,
            "hit_at_k_rate": round(mean([r.get("hit_at_k", 0) for r in rows]), 4) if rows else 0.0,
            "retrieval_latency_ms_avg": round(mean(retrieval_latencies), 2) if retrieval_latencies else 0.0,
        },
        "rows": rows,
    }

    if run_generation:
        answer_latencies = [r.get("answer_latency_ms", 0.0) for r in rows if r.get("answer_latency_ms", 0.0) > 0]
        report["summary"]["useful_answer_rate"] = (
            round(mean([r.get("useful_answer", 0) for r in rows]), 4) if rows else 0.0
        )
        report["summary"]["answer_latency_ms_avg"] = (
            round(mean(answer_latencies), 2) if answer_latencies else 0.0
        )

    return report


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
        f"Hit@k rate: {report['summary'].get('hit_at_k_rate', 0.0)}",
        f"Retrieval latency avg (ms): {report['summary'].get('retrieval_latency_ms_avg', 0.0)}",
    ]
    if "useful_answer_rate" in report.get("summary", {}):
        lines.extend(
            [
                f"Useful answer rate: {report['summary'].get('useful_answer_rate', 0.0)}",
                f"Answer latency avg (ms): {report['summary'].get('answer_latency_ms_avg', 0.0)}",
            ]
        )
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return {"json": json_path, "txt": txt_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation simple du RAG (Precision@k + reponses utiles).")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks recuperes pour Precision@k.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="N'evalue que la retrieval sans generation de reponse.",
    )
    args = parser.parse_args()

    top_k = max(1, args.top_k)
    report = evaluate(top_k=top_k, run_generation=not args.skip_generation)
    output_paths = write_report(report)
    print(f"Evaluation terminee. JSON: {output_paths['json']}")
    print(f"Evaluation terminee. TXT : {output_paths['txt']}")


if __name__ == "__main__":
    main()
