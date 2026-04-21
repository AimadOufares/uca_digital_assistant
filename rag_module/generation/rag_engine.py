import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompt_builder import build_rag_prompt
from ..shared.context_resolution import (
    allowed_establishments_for_resolution,
    build_clarification_message,
    build_out_of_scope_message,
    resolve_context,
)
from ..shared.env_loader import load_env_file

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

load_env_file()

logger = logging.getLogger(__name__)


class RAGIndexNotReadyError(RuntimeError):
    """Raised when the Qdrant index is not available."""


class RAGGenerationError(RuntimeError):
    """Raised when answer generation fails."""


DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LM_STUDIO_MODEL = ""
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 420
DEFAULT_TEMPERATURE = 0.15
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_RETRIEVAL_K = 4
MIN_GENERATED_ANSWER_CHARS = 40


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


def _safe_preview(text: str, max_chars: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


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


def _normalize_confidence_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"eleve", "moyen", "faible"}:
        return normalized
    return "moyen"


def _extract_primary_answer_text(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    match = re.search(
        r"Reponse\s*(.*?)(?:\n\s*Sources utiles|\n\s*Niveau de confiance|\n\s*Points a verifier|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text


def _ensure_structured_answer(answer: str, chunks: List[Dict]) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""

    primary_answer = _extract_primary_answer_text(text)
    sources = _fallback_sources_section(chunks)
    if not sources:
        sources = ["Aucune source pertinente disponible."]
    confidence_label = _confidence_label_from_chunks(chunks)

    has_reponse = bool(re.search(r"^\s*Reponse\b", text, flags=re.IGNORECASE | re.MULTILINE))
    has_sources = bool(re.search(r"^\s*Sources utiles\b", text, flags=re.IGNORECASE | re.MULTILINE))
    has_confidence = bool(re.search(r"^\s*Niveau de confiance\b", text, flags=re.IGNORECASE | re.MULTILINE))

    if has_reponse and has_sources and has_confidence:
        return text

    structured_parts = [
        "Reponse",
        primary_answer or text,
        "",
        "Sources utiles",
        *[f"- {item}" for item in sources],
        "",
        f"Niveau de confiance: {_normalize_confidence_label(confidence_label)}",
    ]
    return "\n".join(structured_parts).strip()


def _generated_answer_is_usable(answer: str) -> bool:
    text = str(answer or "").strip()
    if len(text) < MIN_GENERATED_ANSWER_CHARS:
        return False
    primary_answer = _extract_primary_answer_text(text)
    if len(primary_answer) < 20:
        return False
    if primary_answer.strip().lower() == _abstention_answer().lower():
        return False
    return True


def _generation_backend_status(
    *,
    backend: str,
    success: bool,
    model: str = "",
    latency_ms: float = 0.0,
    answer: str = "",
    error: str = "",
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "success": bool(success),
        "model": str(model or ""),
        "latency_ms": round(float(latency_ms or 0.0), 2),
        "answer": str(answer or ""),
        "answer_preview": _safe_preview(answer),
        "answer_chars": len(str(answer or "")),
        "error": str(error or ""),
    }


def _generate_with_openai(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return _generation_backend_status(backend="openai", success=False, model="", error="openai_unavailable")

    model = os.getenv("RAG_CHAT_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    max_tokens = _env_int("RAG_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = _env_float("RAG_TEMPERATURE", DEFAULT_TEMPERATURE)
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
    client = OpenAI(api_key=api_key, timeout=timeout)
    started = time.perf_counter()

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        answer = (getattr(response, "output_text", "") or "").strip()
        return _generation_backend_status(
            backend="openai",
            success=bool(answer),
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            answer=answer,
        )
    except Exception as exc:
        logger.warning("Generation OpenAI indisponible, fallback local active: %s", exc)
        return _generation_backend_status(
            backend="openai",
            success=False,
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc),
        )


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


def _generate_with_lm_studio(prompt: str) -> Dict[str, Any]:
    if OpenAI is None:
        return _generation_backend_status(backend="lmstudio", success=False, model="", error="openai_sdk_unavailable")

    base_url = os.getenv("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL).strip()
    configured_model = os.getenv("RAG_LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL).strip()
    api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio").strip() or "lm-studio"
    max_tokens = _env_int("RAG_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = _env_float("RAG_TEMPERATURE", DEFAULT_TEMPERATURE)
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

    if not base_url:
        return _generation_backend_status(backend="lmstudio", success=False, model="", error="lmstudio_base_url_missing")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    model = _resolve_lm_studio_model(client, configured_model)
    if not model:
        logger.warning("Aucun modele LM Studio texte n'a ete trouve.")
        return _generation_backend_status(backend="lmstudio", success=False, model="", error="lmstudio_model_missing")

    started = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return _generation_backend_status(
                backend="lmstudio",
                success=False,
                model=model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="lmstudio_empty_choices",
            )
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        answer = (content or "").strip()
        return _generation_backend_status(
            backend="lmstudio",
            success=bool(answer),
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            answer=answer,
        )
    except Exception as exc:
        logger.warning("LM Studio indisponible (%s).", exc)
        return _generation_backend_status(
            backend="lmstudio",
            success=False,
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc),
        )


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

    def retrieve_debug(self, query: str, resolution_context: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            from ..retrieval import rag_search

            if not rag_search.is_search_backend_ready():
                raise RAGIndexNotReadyError("Index RAG introuvable. Lancez d'abord l'indexation.")
            allowed_establishments = allowed_establishments_for_resolution(resolution_context or {})
            debug_payload = rag_search.run_hybrid_search_debug(
                query,
                top_k=self.retrieval_k,
                allowed_establishments=allowed_establishments,
            )
            return debug_payload if isinstance(debug_payload, dict) else {}
        except FileNotFoundError as exc:
            raise RAGIndexNotReadyError("Index RAG introuvable. Lancez d'abord l'indexation.") from exc
        except RAGIndexNotReadyError:
            raise
        except Exception as exc:
            raise RAGIndexNotReadyError(
                "Le moteur de recherche RAG n'est pas pret (index ou modeles indisponibles)."
            ) from exc

    def retrieve(self, query: str, resolution_context: Optional[Dict] = None) -> List[Dict]:
        retrieval_debug = self.retrieve_debug(query, resolution_context=resolution_context)
        return list(retrieval_debug.get("final_results", []))

    def _effective_prompt_style(self) -> str:
        return "standard" if self.prompt_style == "auto" else self.prompt_style

    def _build_generation_prompt(self, query: str, chunks: List[Dict], resolution_context: Optional[Dict] = None) -> Dict[str, Any]:
        prompt_style = self._effective_prompt_style()
        prompt = build_rag_prompt(
            query=query,
            chunks=chunks,
            style=prompt_style,
            resolution_context=resolution_context,
        )
        return {
            "prompt": prompt,
            "prompt_style": prompt_style,
            "prompt_chars": len(prompt),
            "chunk_count": len(chunks),
            "confidence_label": _confidence_label_from_chunks(chunks),
            "sources_preview": _fallback_sources_section(chunks),
        }

    def generate_with_debug(self, query: str, chunks: List[Dict], resolution_context: Optional[Dict] = None) -> Dict[str, Any]:
        if not chunks:
            return {
                "answer": _abstention_answer(),
                "backend": "none",
                "used_fallback": True,
                "fallback_type": "abstention",
                "backend_attempts": [],
                "prompt": {"prompt_style": self._effective_prompt_style(), "prompt_chars": 0, "chunk_count": 0},
            }

        prompt_payload = self._build_generation_prompt(query, chunks, resolution_context=resolution_context)
        prompt = str(prompt_payload.get("prompt") or "")
        backend_attempts: List[Dict[str, Any]] = []

        for backend in _generation_order():
            result = _generate_with_lm_studio(prompt) if backend == "lmstudio" else _generate_with_openai(prompt)
            backend_attempts.append(result)
            raw_candidate_answer = str(result.get("answer") or "")
            candidate_answer = _ensure_structured_answer(raw_candidate_answer, chunks)
            if result.get("success"):
                normalized_answer = candidate_answer or _ensure_structured_answer(raw_candidate_answer, chunks)
                if _generated_answer_is_usable(normalized_answer):
                    return {
                        "answer": normalized_answer,
                        "backend": backend,
                        "used_fallback": False,
                        "fallback_type": "",
                        "backend_attempts": backend_attempts,
                        "prompt": {key: value for key, value in prompt_payload.items() if key != "prompt"},
                    }

        fallback_answer = _extractive_fallback_answer(query, chunks)
        return {
            "answer": fallback_answer,
            "backend": "fallback",
            "used_fallback": True,
            "fallback_type": "extractive",
            "backend_attempts": backend_attempts,
            "prompt": {key: value for key, value in prompt_payload.items() if key != "prompt"},
        }

    def generate(self, query: str, chunks: List[Dict], resolution_context: Optional[Dict] = None) -> str:
        return str(self.generate_with_debug(query, chunks, resolution_context=resolution_context).get("answer") or "")

    def answer(self, query: str, user_establishment: Optional[str] = None) -> Dict:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("La question ne peut pas etre vide.")

        resolution_context = resolve_context(cleaned_query, user_establishment=user_establishment)
        resolution_context["allowed_establishments"] = allowed_establishments_for_resolution(resolution_context)

        if resolution_context.get("mode") == "clarification":
            return {
                "answer": build_clarification_message(resolution_context),
                "sources": [],
                "needs_clarification": True,
                "resolution": resolution_context,
            }

        if resolution_context.get("mode") == "out_of_scope":
            return {
                "answer": build_out_of_scope_message(),
                "sources": [],
                "resolution": resolution_context,
            }

        retrieval_debug = self.retrieve_debug(cleaned_query, resolution_context=resolution_context)
        chunks = list(retrieval_debug.get("final_results", []))
        if not chunks:
            return {
                "answer": _abstention_answer(),
                "sources": [],
                "resolution": resolution_context,
                "retrieval_debug": retrieval_debug,
                "generation_debug": {
                    "backend": "none",
                    "used_fallback": True,
                    "fallback_type": "abstention",
                    "backend_attempts": [],
                    "prompt": {"prompt_style": self._effective_prompt_style(), "prompt_chars": 0, "chunk_count": 0},
                },
            }
        try:
            generation_debug = self.generate_with_debug(cleaned_query, chunks, resolution_context=resolution_context)
            answer = str(generation_debug.get("answer") or "")
        except Exception as exc:
            raise RAGGenerationError("Erreur lors de la generation de reponse.") from exc

        return {
            "answer": answer.strip(),
            "sources": _normalize_sources(chunks),
            "resolution": resolution_context,
            "retrieval_debug": retrieval_debug,
            "generation_debug": generation_debug,
        }


_default_engine = RAGEngine()


def answer_question(question: str, user_establishment: Optional[str] = None) -> Dict:
    """Fonction utilitaire conservee pour compatibilite avec pipeline.py."""
    return _default_engine.answer(question, user_establishment=user_establishment)
