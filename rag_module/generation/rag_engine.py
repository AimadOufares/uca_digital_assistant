import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from .prompt_builder import build_rag_prompt
from ..shared.env_loader import load_env_file

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

load_env_file()

logger = logging.getLogger(__name__)


class RAGIndexNotReadyError(RuntimeError):
    """Raised when FAISS index or chunks are not available."""


class RAGGenerationError(RuntimeError):
    """Raised when answer generation fails."""


DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LM_STUDIO_MODEL = ""
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 420
DEFAULT_TEMPERATURE = 0.15
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_K = 4


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return value if value >= 0 else default
    except ValueError:
        return default


def _safe_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _confidence_label_from_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "faible"

    scores: List[float] = []
    for chunk in chunks[:3]:
        confidence = _chunk_confidence(chunk)
        scores.append(float(confidence.get("score", 0.0) or 0.0))

    if not scores:
        return "faible"

    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    if top_score >= 0.82 and avg_score >= 0.72:
        return "eleve"
    if top_score >= 0.62 and avg_score >= 0.5:
        return "moyen"
    return "faible"


def _fallback_sources_section(chunks: List[Dict]) -> List[str]:
    names: List[str] = []
    for source in _normalize_sources(chunks)[:3]:
        name = str(source.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _extractive_fallback_answer(query: str, chunks: List[Dict]) -> str:
    if not chunks:
        return (
            "Reponse\n"
            "Information non disponible dans mes sources actuelles.\n\n"
            "Sources utiles\n"
            "- Aucune source pertinente disponible.\n\n"
            "Niveau de confiance: faible\n\n"
            "Points a verifier\n"
            "- Reformuler la question ou preciser la faculte, l'annee ou la procedure concernee."
        )

    lines: List[str] = []
    for chunk in chunks[:3]:
        for sentence in _safe_sentences(chunk.get("text", "")):
            if len(sentence) >= 35:
                lines.append(sentence)
            if len(lines) >= 4:
                break
        if len(lines) >= 4:
            break

    if not lines:
        response_body = (
            "J'ai trouve des documents lies a votre question, mais je n'ai pas pu "
            "produire une synthese suffisamment fiable automatiquement."
        )
    else:
        response_body = "Voici les informations les plus directement appuyees par les extraits recuperes :\n"
        response_body += "\n".join(f"- {line}" for line in lines)

    sources_lines = _fallback_sources_section(chunks)
    if not sources_lines:
        sources_block = "- Sources internes non identifiees clairement."
    else:
        sources_block = "\n".join(f"- {name}" for name in sources_lines)

    confidence_label = _confidence_label_from_chunks(chunks)
    points = (
        "- Verifier les details exacts si vous avez besoin d'une date, d'un delai ou d'une procedure complete."
        if confidence_label != "eleve"
        else "- Aucun point critique supplementaire releve a partir des extraits recuperes."
    )

    return (
        f"Reponse\n{response_body}\n\n"
        f"Sources utiles\n{sources_block}\n\n"
        f"Niveau de confiance: {confidence_label}\n\n"
        f"Points a verifier\n{points}"
    )


def _abstention_answer() -> str:
    return "Information non disponible dans mes sources actuelles."


def _generate_with_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return ""

    model = os.getenv("RAG_CHAT_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    max_tokens = _env_int("RAG_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = _env_float("RAG_TEMPERATURE", DEFAULT_TEMPERATURE)
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
    client = OpenAI(api_key=api_key, timeout=timeout)

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return (getattr(response, "output_text", "") or "").strip()
    except Exception as exc:
        logger.warning("Generation OpenAI indisponible, fallback local active: %s", exc)
        return ""


def _is_embedding_model(model_id: str) -> bool:
    value = (model_id or "").lower()
    return any(token in value for token in ("embed", "embedding", "nomic-embed"))


def _resolve_lm_studio_model(client: Any, configured_model: str) -> str:
    if configured_model:
        return configured_model
    try:
        models = client.models.list()
        items = getattr(models, "data", []) or []
        ids = [getattr(item, "id", "") for item in items if getattr(item, "id", "")]
        for model_id in ids:
            if not _is_embedding_model(model_id):
                return model_id
        return ids[0] if ids else ""
    except Exception as exc:
        logger.warning("Impossible de recuperer les modeles LM Studio (%s).", exc)
        return ""


def _generate_with_lm_studio(prompt: str) -> str:
    if OpenAI is None:
        return ""

    base_url = os.getenv("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL).strip()
    configured_model = os.getenv("RAG_LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL).strip()
    api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio").strip() or "lm-studio"
    max_tokens = _env_int("RAG_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = _env_float("RAG_TEMPERATURE", DEFAULT_TEMPERATURE)
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

    if not base_url:
        return ""

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    model = _resolve_lm_studio_model(client, configured_model)
    if not model:
        logger.warning("Aucun modele LM Studio texte n'a ete trouve.")
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return (content or "").strip()
    except Exception as exc:
        logger.warning("LM Studio indisponible (%s).", exc)
        return ""


def _generation_order() -> List[str]:
    provider = os.getenv("RAG_LLM_PROVIDER", "lmstudio").strip().lower()
    if provider in {"lmstudio", "local"}:
        return ["lmstudio", "openai"]
    if provider == "openai":
        return ["openai", "lmstudio"]
    if provider == "auto":
        return ["lmstudio", "openai"]
    return ["lmstudio", "openai"]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_rerank_score(raw: float) -> float:
    # Mapping logistique simple pour rendre les scores cross-encoder comparables [0,1].
    return 1.0 / (1.0 + math.exp(-raw / 4.0))


def _chunk_confidence(chunk: Dict) -> Dict[str, Any]:
    if "rerank_score" in chunk:
        raw = _to_float(chunk.get("rerank_score"))
        return {
            "score": _normalize_rerank_score(raw),
            "score_type": "rerank",
            "raw_score": raw,
        }

    score = _to_float(chunk.get("score"))
    score_type = chunk.get("score_type") or "vector"
    if score_type == "vector":
        score = _clamp01(score)
    return {"score": score, "score_type": score_type, "raw_score": score}


def _normalize_sources(chunks: List[Dict]) -> List[Dict]:
    by_source: Dict[str, Dict] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        raw_source = metadata.get("source") or metadata.get("file_name") or "Document inconnu"
        source_path = str(raw_source) if raw_source is not None else ""
        source_name = Path(source_path).name if source_path else "Document inconnu"
        source_key = source_path or source_name

        confidence = _chunk_confidence(chunk)
        entry = by_source.get(source_key)
        if entry is None:
            by_source[source_key] = {
                "name": source_name,
                "path": source_path,
                "score": round(confidence["score"], 4),
                "score_type": confidence["score_type"],
                "hits": 1,
            }
            continue

        entry["hits"] += 1
        if confidence["score"] > entry["score"]:
            entry["score"] = round(confidence["score"], 4)
            entry["score_type"] = confidence["score_type"]

    ordered = sorted(by_source.values(), key=lambda x: (x["score"], x["hits"]), reverse=True)
    return ordered


class RAGEngine:
    def __init__(self, retrieval_k: int = DEFAULT_RETRIEVAL_K, prompt_style: str = "auto"):
        retrieval_k_from_env = _env_int("RAG_RETRIEVAL_K", retrieval_k)
        self.retrieval_k = max(1, retrieval_k_from_env)
        env_prompt_style = os.getenv("RAG_PROMPT_STYLE", prompt_style).strip().lower()
        self.prompt_style = env_prompt_style if env_prompt_style in {"auto", "standard", "concise"} else "auto"

    def retrieve(self, query: str) -> List[Dict]:
        try:
            from ..retrieval import rag_search

            if not rag_search.is_search_backend_ready():
                raise RAGIndexNotReadyError("Index RAG introuvable. Lancez d'abord l'indexation.")
            return rag_search.get_relevant_chunks(query, top_k=self.retrieval_k)
        except FileNotFoundError as exc:
            raise RAGIndexNotReadyError("Index RAG introuvable. Lancez d'abord l'indexation.") from exc
        except RAGIndexNotReadyError:
            raise
        except Exception as exc:
            raise RAGIndexNotReadyError(
                "Le moteur de recherche RAG n'est pas pret (index ou modeles indisponibles)."
            ) from exc

    def generate(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return _abstention_answer()

        backends = _generation_order()
        prompt_style = self.prompt_style
        if prompt_style == "auto":
            prompt_style = "standard"

        prompt = build_rag_prompt(query=query, chunks=chunks, style=prompt_style)
        for backend in backends:
            answer = _generate_with_lm_studio(prompt) if backend == "lmstudio" else _generate_with_openai(prompt)
            if answer:
                return answer
        return _extractive_fallback_answer(query, chunks)

    def answer(self, query: str) -> Dict:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("La question ne peut pas etre vide.")

        chunks = self.retrieve(cleaned_query)
        if not chunks:
            return {"answer": _abstention_answer(), "sources": []}
        try:
            answer = self.generate(cleaned_query, chunks)
        except Exception as exc:
            raise RAGGenerationError("Erreur lors de la generation de reponse.") from exc

        return {"answer": answer.strip(), "sources": _normalize_sources(chunks)}


_default_engine = RAGEngine()


def answer_question(question: str) -> Dict:
    """Fonction utilitaire conservee pour compatibilite avec pipeline.py."""
    return _default_engine.answer(question)
