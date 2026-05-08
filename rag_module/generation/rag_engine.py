import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from .prompt_builder import build_rag_prompt
from ..adapters.llm_provider import LLMProviderAdapter
from ..retrieval.query_intelligence import NORMALIZED_SERVICE_ALIAS_RULES, build_query_profile
from ..shared.env_loader import load_env_file
from ..shared.metadata_policy import normalize_text
from ..shared.runtime import get_runtime_settings

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


DEFAULT_LM_STUDIO_BASE_URL = ""
DEFAULT_LM_STUDIO_MODEL = ""
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 420
DEFAULT_LM_STUDIO_MAX_TOKENS = 800
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safe_sentences(text: str) -> List[str]:
    if not text:
        return []
    prepared = re.sub(r"\s*([🔹🔸✅✔➡])\s*", r"\n\1 ", text.strip())
    prepared = re.sub(r"\s+(#{1,6}\s+)", r"\n\1", prepared)
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", prepared)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def _fallback_query_tokens(query: str) -> set[str]:
    stopwords = {
        "comment",
        "obtenir",
        "avoir",
        "mon",
        "ma",
        "mes",
        "sur",
        "dans",
        "pour",
        "avec",
        "une",
        "des",
        "les",
        "le",
        "la",
        "un",
        "du",
    }
    tokens = set(re.findall(r"\b[\w@']+\b", normalize_text(query)))
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


def _sentence_query_score(sentence: str, query_tokens: set[str]) -> int:
    if not query_tokens:
        return 0
    normalized = normalize_text(sentence)
    score = 0
    for token in query_tokens:
        if token in normalized:
            if token in {"attestation", "attestations"}:
                score += 5
            elif token in {"uc@student", "ucastudent", "student"}:
                score += 1
            else:
                score += 2
    if {"attestation", "attestations"}.intersection(query_tokens) and "demande" in normalized:
        score += 3
    return score


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
        return "Information non disponible dans mes sources actuelles."

    query_tokens = _fallback_query_tokens(query)
    candidates: List[tuple[int, int, str]] = []
    for chunk_index, chunk in enumerate(chunks[:4]):
        for sentence in _safe_sentences(chunk.get("text", "")):
            if len(sentence) < 20:
                continue
            score = _sentence_query_score(sentence, query_tokens)
            if score > 0:
                candidates.append((score, -chunk_index, sentence))

    candidates.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
    lines: List[str] = []
    seen_normalized: set[str] = set()
    for _, _, sentence in candidates:
        normalized_sentence = normalize_text(sentence)
        if normalized_sentence in seen_normalized:
            continue
        seen_normalized.add(normalized_sentence)
        lines.append(sentence)
        if len(lines) >= 4:
            break

    if not lines:
        for chunk in chunks[:3]:
            for sentence in _safe_sentences(chunk.get("text", "")):
                if len(sentence) >= 35:
                    lines.append(sentence)
                if len(lines) >= 4:
                    break
            if len(lines) >= 4:
                break

    if not lines:
        return (
            "J'ai trouve des documents lies a votre question, mais je n'ai pas pu "
            "produire une synthese suffisamment fiable automatiquement."
        )

    lead = "D'apres les informations retrouvees dans les documents UCA :"
    bullet_lines = "\n".join(f"- {line}" for line in lines[:3])
    return f"{lead}\n{bullet_lines}"


def _abstention_answer() -> str:
    return "Information non disponible dans mes sources actuelles."


def _generate_with_openai(prompt: str) -> str:
    runtime = get_runtime_settings()
    api_key = runtime.openai_api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return ""

    model = os.getenv("RAG_CHAT_MODEL", runtime.rag_chat_model).strip() or runtime.rag_chat_model or DEFAULT_OPENAI_MODEL
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

    runtime = get_runtime_settings()
    base_url = os.getenv("LM_STUDIO_BASE_URL", runtime.lm_studio_base_url or DEFAULT_LM_STUDIO_BASE_URL).strip()
    configured_model = os.getenv("RAG_LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL).strip()
    api_key = os.getenv("LM_STUDIO_API_KEY", runtime.lm_studio_api_key or "lm-studio").strip() or "lm-studio"
    max_tokens_raw = _env_int("RAG_LM_STUDIO_MAX_TOKENS", _env_int("RAG_MAX_TOKENS", DEFAULT_LM_STUDIO_MAX_TOKENS))
    # -1 = pas de limite (le LLM genere jusqu'a la fin naturelle)
    max_tokens = None if max_tokens_raw <= 0 else max_tokens_raw
    temperature = _env_float("RAG_TEMPERATURE", DEFAULT_TEMPERATURE)
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

    if not base_url:
        return ""

    # max_retries=0 : evite les ré-essais automatiques qui relancent la génération depuis zéro
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    model = _resolve_lm_studio_model(client, configured_model)
    if not model:
        logger.warning("Aucun modele LM Studio texte n'a ete trouve.")
        return ""

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return (content or "").strip()
    except Exception as exc:
        logger.warning("LM Studio indisponible (%s).", exc)
        return ""


HYDE_MAX_TOKENS = 75  # Court extrait suffit, evite une generation trop longue


def _generate_hyde_doc(prompt: str) -> str:
    """Génère un document hypothétique court pour HyDE (max 75 tokens)."""
    if OpenAI is None:
        return ""

    runtime = get_runtime_settings()
    base_url = os.getenv("LM_STUDIO_BASE_URL", runtime.lm_studio_base_url or DEFAULT_LM_STUDIO_BASE_URL).strip()
    configured_model = os.getenv("RAG_LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL).strip()
    api_key = os.getenv("LM_STUDIO_API_KEY", runtime.lm_studio_api_key or "lm-studio").strip() or "lm-studio"
    timeout = _env_float("RAG_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

    if not base_url:
        return ""

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    model = _resolve_lm_studio_model(client, configured_model)
    if not model:
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=HYDE_MAX_TOKENS,  # Limite stricte pour la vitesse
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return (content or "").strip()
    except Exception as exc:
        logger.warning("HyDE generation echouee: %s", exc)
        return ""


def _generation_order() -> List[str]:
    return LLMProviderAdapter(get_runtime_settings()).provider_order()


def _prompt_style_for_backend(configured_style: str, backend: str) -> str:
    if configured_style != "auto":
        return configured_style
    if backend == "lmstudio":
        return "compact"
    return "standard"


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
                "service_name": str(metadata.get("service_name") or "").strip(),
                "official_url": str(metadata.get("official_url") or metadata.get("url") or "").strip(),
                "title": str(metadata.get("title") or "").strip(),
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


def _reorder_context(chunks: List[Dict]) -> List[Dict]:
    """Lost in the middle context reordering."""
    if len(chunks) < 3:
        return chunks
    
    left = chunks[0::2]
    right = chunks[1::2][::-1]
    return left + right


def _chunk_matches_explicit_service(chunk: Dict, requested_services: set[str]) -> bool:
    metadata = chunk.get("metadata", {}) or {}
    service_name = normalize_text(str(metadata.get("service_name") or ""))
    official_url = normalize_text(str(metadata.get("official_url") or metadata.get("url") or ""))
    source_name = normalize_text(str(metadata.get("source") or metadata.get("file_name") or ""))
    text = normalize_text(str(chunk.get("text") or ""))
    haystack = " ".join(part for part in (service_name, official_url, source_name, text[:600]) if part)

    for requested in requested_services:
        aliases = set(NORMALIZED_SERVICE_ALIAS_RULES.get(requested, set()) or set())
        aliases.add(requested)
        if any(alias and alias in haystack for alias in aliases):
            return True
    return False


def _prioritize_explicit_service_chunks(query: str, chunks: List[Dict]) -> tuple[List[Dict], Dict[str, Any]]:
    query_profile = build_query_profile(query)
    requested_services = {normalize_text(service) for service in query_profile.get("services", []) if normalize_text(service)}
    if not requested_services:
        return chunks, {"requested_services": [], "service_filtered": False}

    matching = [chunk for chunk in chunks if _chunk_matches_explicit_service(chunk, requested_services)]
    if not matching:
        return chunks, {
            "requested_services": sorted(requested_services),
            "service_filtered": False,
            "service_match_count": 0,
        }

    return matching, {
        "requested_services": sorted(requested_services),
        "service_filtered": True,
        "service_match_count": len(matching),
        "service_original_count": len(chunks),
    }


class RAGEngine:
    def __init__(self, retrieval_k: int = DEFAULT_RETRIEVAL_K, prompt_style: str = "auto", use_hyde: bool = False):
        retrieval_k_from_env = _env_int("RAG_RETRIEVAL_K", retrieval_k)
        self.retrieval_k = max(1, retrieval_k_from_env)
        env_prompt_style = os.getenv("RAG_PROMPT_STYLE", prompt_style).strip().lower()
        self.prompt_style = env_prompt_style if env_prompt_style in {"auto", "standard", "concise", "compact"} else "auto"
        self.use_hyde = _env_bool("RAG_USE_HYDE", use_hyde)

    def _llm_is_available(self) -> bool:
        """Vérifie rapidement si un LLM est disponible avant de tenter HyDE."""
        runtime = get_runtime_settings()
        # Vérifie LM Studio
        base_url = os.getenv("LM_STUDIO_BASE_URL", runtime.lm_studio_base_url or "").strip()
        if base_url:
            try:
                import urllib.request
                req = urllib.request.Request(base_url.rstrip("/v1").rstrip("/") + "/v1/models", method="GET")
                urllib.request.urlopen(req, timeout=2)
                return True
            except Exception:
                pass
        # Vérifie OpenAI
        if runtime.openai_api_key or os.getenv("OPENAI_API_KEY", "").strip():
            return True
        return False

    def retrieve(self, query: str) -> Dict[str, Any]:
        search_query = query
        hyde_used = False
        if self.use_hyde and self._llm_is_available():
            try:
                hyde_prompt = f"En 2-3 phrases maximum, rédigez un extrait administratif répondant à : {query}"
                hypo_doc = _generate_hyde_doc(hyde_prompt)
                if hypo_doc:
                    logger.info("HyDE actif: document hypothetique genere (%d chars)", len(hypo_doc))
                    search_query = f"{query} {hypo_doc}"
                    hyde_used = True
            except Exception as e:
                logger.warning(f"HyDE error: {e}")

        try:
            from ..retrieval.rag_search import get_relevant_chunks_debug

            debug_payload = get_relevant_chunks_debug(search_query, top_k=self.retrieval_k)
            return {
                "chunks": list(debug_payload.get("final_results", [])),
                "meta": {
                    "search_query": search_query,
                    "hyde_used": hyde_used,
                    "abstain": bool(debug_payload.get("abstain", False)),
                    "abstain_reason": str(debug_payload.get("abstain_reason", "") or ""),
                    "query_profile": debug_payload.get("query_profile", {}),
                    "guardrail_diagnostics": debug_payload.get("guardrail_diagnostics", {}),
                },
            }
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
        for backend in backends:
            prompt_style = _prompt_style_for_backend(self.prompt_style, backend)
            prompt = build_rag_prompt(query=query, chunks=chunks, style=prompt_style)
            answer = _generate_with_lm_studio(prompt) if backend == "lmstudio" else _generate_with_openai(prompt)
            if answer:
                return answer
        return _extractive_fallback_answer(query, chunks)

    def answer(self, query: str) -> Dict:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("La question ne peut pas etre vide.")

        retrieval_payload = self.retrieve(cleaned_query)
        chunks = list(retrieval_payload.get("chunks", []))
        retrieval_meta = dict(retrieval_payload.get("meta", {}) or {})
        if not chunks:
            return {"answer": _abstention_answer(), "sources": [], "retrieval_meta": retrieval_meta}

        chunks, service_filter_meta = _prioritize_explicit_service_chunks(cleaned_query, chunks)
        retrieval_meta.update(service_filter_meta)
        chunks = _reorder_context(chunks)
        
        try:
            answer = self.generate(cleaned_query, chunks)
        except Exception as exc:
            raise RAGGenerationError("Erreur lors de la generation de reponse.") from exc

        retrieval_meta["context_chunk_count"] = len(chunks)
        return {
            "answer": answer.strip(),
            "sources": _normalize_sources(chunks),
            "retrieval_meta": retrieval_meta,
        }

_default_engine = RAGEngine()


def answer_question(question: str) -> Dict:
    """Fonction utilitaire conservee pour compatibilite avec pipeline.py."""
    return _default_engine.answer(question)
