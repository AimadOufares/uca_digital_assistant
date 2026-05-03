from ..contracts import AnswerResult, QuestionRequest
from ..generation.rag_engine import RAGEngine
from ..shared.runtime import get_runtime_settings


def _confidence_from_sources(sources: list[dict]) -> str:
    if not sources:
        return "faible"
    top_score = 0.0
    try:
        top_score = float(sources[0].get("score", 0.0) or 0.0)
    except Exception:
        top_score = 0.0
    if top_score >= 0.8:
        return "eleve"
    if top_score >= 0.55:
        return "moyen"
    return "faible"


def answer_question(request: QuestionRequest) -> AnswerResult:
    engine = RAGEngine()
    payload = engine.answer(request.question)
    sources = list(payload.get("sources", []))
    settings = get_runtime_settings()
    retrieval_meta = dict(payload.get("retrieval_meta", {}) or {})
    retrieval_meta.update(
        {
            "provider": settings.rag_llm_provider,
            "vector_backend": settings.rag_vector_backend,
        }
    )
    return AnswerResult(
        answer=str(payload.get("answer", "") or "").strip(),
        sources=sources,
        confidence=_confidence_from_sources(sources),
        backend=settings.rag_vector_backend,
        retrieval_meta=retrieval_meta,
    )
