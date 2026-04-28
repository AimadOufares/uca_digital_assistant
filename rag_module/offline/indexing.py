import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from sentence_transformers import SentenceTransformer

try:
    from ..shared.env_loader import load_env_file
    from ..shared.metadata_policy import normalize_text
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.env_loader import load_env_file
    from rag_module.shared.metadata_policy import normalize_text
    from rag_module.shared.runtime import get_runtime_settings


load_env_file()
RUNTIME = get_runtime_settings()

CACHE_PATH = str(RUNTIME.rag_cache_dir / "embeddings_cache.json")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
]
BATCH_SIZE = 128
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_embedding_model = None
_embedding_model_name = None

BLOCKED_QUALITY_ISSUES = {
    "empty_after_static",
    "empty_after_playwright",
    "login_wall",
    "too_generic",
    "encoding_suspect",
}
HIGH_VALUE_INTENTS = {
    "connexion",
    "mot_de_passe",
    "attestation",
    "notes",
    "reinscription",
    "candidature",
    "cours",
    "depot_document",
}
ACTIONABLE_PAGE_KINDS = {"guide", "procedure", "faq", "formulaire"}
LANDING_PAGE_KINDS = {"landing"}
CANONICAL_STOPWORDS = {
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "et",
    "ou",
    "un",
    "une",
    "pour",
    "dans",
    "avec",
    "sur",
    "par",
    "est",
    "en",
    "au",
    "aux",
}


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_int_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        values = list(raw_value)
    elif raw_value in (None, ""):
        values = []
    else:
        values = [raw_value]
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if isinstance(value, list):
            for nested in value:
                item = normalize_text(str(nested or ""))
                if item and item not in seen:
                    seen.add(item)
                    normalized.append(item)
            continue
        item = normalize_text(str(value or ""))
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _canonical_tokens(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"\b[\w'-]+\b", normalized.lower(), flags=re.UNICODE)
    filtered = [
        token
        for token in tokens
        if len(token) > 2 and token not in CANONICAL_STOPWORDS and not token.isdigit()
    ]
    return filtered or tokens


def _near_duplicate_signature(text: str) -> str:
    tokens = _canonical_tokens(text)
    if not tokens:
        return ""
    if len(tokens) <= 18:
        basis = " ".join(tokens)
    else:
        basis = " ".join(tokens[:12] + tokens[-8:])
    return get_hash(basis)


def _compute_student_relevance(metadata: Dict, text: str) -> float:
    intents = _normalize_int_list(metadata.get("intent"))
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    service_name = normalize_text(str(metadata.get("service_name") or ""))
    service_type = normalize_text(str(metadata.get("service_type") or metadata.get("document_type") or ""))
    document_category = normalize_text(str(metadata.get("document_category") or ""))
    source_priority = str(metadata.get("source_priority") or "").strip().upper()
    chunk_relevance_score = float(metadata.get("chunk_relevance_score", 0.0) or 0.0)
    source_relevance_score = float(metadata.get("source_relevance_score", 0.0) or 0.0)
    freshness_score = float(metadata.get("freshness_score", 0.0) or 0.0)
    quality_issue = normalize_text(str(metadata.get("quality_issue") or ""))

    score = 0.0
    score += min(0.25, 0.05 * len(intents))
    if any(intent in HIGH_VALUE_INTENTS for intent in intents):
        score += 0.15
    if page_kind in ACTIONABLE_PAGE_KINDS:
        score += 0.15
    if service_type in {"digital_service", "pedagogie_numerique", "scolarite"}:
        score += 0.12
    if document_category in {"digital_service", "vie_etudiante", "scolarite"}:
        score += 0.08
    if service_name in {"ucastudent", "ucaplat", "pedoc", "cip", "e-candidature", "espace diplomes"}:
        score += 0.12
    if source_priority == "A":
        score += 0.08
    elif source_priority == "B":
        score += 0.04
    score += min(0.12, chunk_relevance_score * 0.03)
    score += min(0.08, source_relevance_score * 0.015)
    score += min(0.05, freshness_score * 0.05)
    if quality_issue:
        score -= 0.2
    if len(text.split()) >= 45:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 3)


def _is_actionable_chunk(metadata: Dict) -> bool:
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    intents = _normalize_int_list(metadata.get("intent"))
    if page_kind in ACTIONABLE_PAGE_KINDS:
        return True
    return any(intent in HIGH_VALUE_INTENTS for intent in intents)


def _build_retrieval_keywords(metadata: Dict) -> List[str]:
    fields = [
        metadata.get("service_name"),
        metadata.get("service_type"),
        metadata.get("document_type"),
        metadata.get("document_category"),
        metadata.get("page_kind"),
        metadata.get("intent", []),
        metadata.get("main_actions", []),
    ]
    keywords = _normalize_int_list(fields)
    return keywords[:12]


def _build_retrieval_haystack(text: str, metadata: Dict) -> str:
    additions = [
        str(metadata.get("service_name") or ""),
        str(metadata.get("service_type") or ""),
        str(metadata.get("document_type") or ""),
        str(metadata.get("page_kind") or ""),
        " ".join(_normalize_int_list(metadata.get("intent"))),
        " ".join(_normalize_int_list(metadata.get("main_actions"))),
    ]
    prefix = normalize(" ".join(part for part in additions if part))
    if not prefix:
        return text
    return normalize(f"{prefix}\n{text}")


def should_index_chunk(text: str, metadata: Dict) -> bool:
    quality_issue = normalize_text(str(metadata.get("quality_issue") or ""))
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    student_relevance_score = float(metadata.get("student_relevance_score", 0.0) or 0.0)
    chunk_relevance_score = float(metadata.get("chunk_relevance_score", 0.0) or 0.0)
    intents = _normalize_int_list(metadata.get("intent"))

    if len(text) < 60:
        return False
    if quality_issue in BLOCKED_QUALITY_ISSUES:
        return False
    if page_kind in LANDING_PAGE_KINDS and not intents and student_relevance_score < 0.45:
        return False
    if chunk_relevance_score <= 0 and student_relevance_score < 0.35:
        return False
    return True


def enrich_index_metadata(text: str, metadata: Dict, corpus_name: str) -> Dict:
    enriched = dict(metadata)
    intents = _normalize_int_list(enriched.get("intent"))
    enriched["intent"] = intents
    enriched["corpus"] = corpus_name
    enriched["service_priority"] = str(enriched.get("source_priority") or "unknown").strip().upper() or "unknown"
    enriched["is_actionable"] = _is_actionable_chunk(enriched)
    enriched["student_relevance_score"] = _compute_student_relevance(enriched, text)
    enriched["retrieval_keywords"] = _build_retrieval_keywords(enriched)
    enriched["retrieval_haystack"] = _build_retrieval_haystack(text, enriched)
    return enriched


def get_model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_candidate_model_names() -> List[str]:
    candidates = [get_model_name(), *FALLBACK_EMBEDDING_MODELS]
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        value = (candidate or "").strip()
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def is_e5_model(model_name: str) -> bool:
    return "e5" in (model_name or "").lower()


def prepare_passage_text(text: str, model_name: str) -> str:
    normalized = normalize(text)
    if is_e5_model(model_name):
        return f"passage: {normalized}"
    return normalized


def get_cache_namespace(model_name: str) -> str:
    return f"model::{model_name}"


def load_sentence_transformer_offline(model_names: List[str]) -> tuple[SentenceTransformer, str]:
    errors: List[str] = []
    for model_name in model_names:
        try:
            model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError(
        "Aucun modele d'embedding local n'est disponible. "
        f"Modeles testes: {', '.join(model_names)}. "
        f"Details: {' | '.join(errors[:3])}"
    )


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model, _embedding_model_name
    requested_model = get_model_name()
    if _embedding_model is None:
        logger.info("Chargement du modele embedding demande: %s", requested_model)
        _embedding_model, _embedding_model_name = load_sentence_transformer_offline(get_candidate_model_names())
        if _embedding_model_name != requested_model:
            logger.warning(
                "Fallback embedding actif: modele demande '%s', modele utilise '%s'.",
                requested_model,
                _embedding_model_name,
            )
    return _embedding_model


def get_active_model_name() -> str:
    get_embedding_model()
    return _embedding_model_name or get_model_name()


def load_cache() -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(CACHE_PATH):
        return {"version": 2, "models": {}}

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        logger.warning("Cache corrompu -> reset")
        return {"version": 2, "models": {}}

    if isinstance(raw, dict) and isinstance(raw.get("models"), dict):
        return {"version": 2, "models": raw["models"]}

    if isinstance(raw, dict):
        legacy_model = get_cache_namespace(get_model_name())
        return {"version": 2, "models": {legacy_model: raw}}

    return {"version": 2, "models": {}}


def save_cache(cache: Dict[str, Dict[str, List[float]]]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)


def _processed_dir_for_corpus(corpus: str) -> Path:
    if corpus == "archive":
        return Path(RUNTIME.rag_processed_archive_dir)
    if corpus == "drive":
        return Path(RUNTIME.rag_processed_drive_dir)
    return Path(RUNTIME.rag_processed_main_dir)


def get_published_corpora() -> List[str]:
    configured = []
    for corpus in RUNTIME.rag_index_published_corpora:
        value = (corpus or "").strip().lower()
        if value in {"main", "drive"} and value not in configured:
            configured.append(value)
    return configured or ["main", "drive"]


def resolve_index_corpora(corpus: str = "published") -> List[str]:
    if corpus in {"main", "drive", "archive"}:
        return [corpus]
    if corpus in {"main_and_drive", "published"}:
        return get_published_corpora()
    return ["main"]


def load_chunks(corpus: str = "main") -> List[Dict]:
    corpora = resolve_index_corpora(corpus)
    chunks: List[Dict] = []
    seen_ids = set()
    seen_text_hashes = set()
    seen_near_duplicate_signatures = set()

    for corpus_name in corpora:
        processed_path = _processed_dir_for_corpus(corpus_name)
        for file_path in sorted(processed_path.glob("*.json")):
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                logger.warning("Erreur fichier %s: %s", file_path, exc)
                continue

            text = normalize(data.get("text", ""))
            if len(text) < 30:
                continue

            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            metadata = enrich_index_metadata(text, metadata, corpus_name)
            if not should_index_chunk(text, metadata):
                continue

            chunk_id = (
                data.get("id")
                or metadata.get("chunk_hash")
                or metadata.get("hash")
                or get_hash(text)
            )
            if chunk_id in seen_ids:
                continue

            text_hash = get_hash(text.lower())
            if text_hash in seen_text_hashes:
                continue

            near_duplicate_signature = _near_duplicate_signature(text)
            if near_duplicate_signature and near_duplicate_signature in seen_near_duplicate_signatures:
                continue

            seen_ids.add(chunk_id)
            seen_text_hashes.add(text_hash)
            if near_duplicate_signature:
                seen_near_duplicate_signatures.add(near_duplicate_signature)

            source_path = metadata.get("source", str(file_path))
            source_name = metadata.get("file_name") or Path(source_path).name
            merged_metadata = dict(metadata)
            merged_metadata.update(
                {
                    "source": source_path,
                    "file_name": source_name,
                    "hash": chunk_id,
                    "date_indexed": datetime.now(timezone.utc).isoformat(),
                }
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": merged_metadata,
                }
            )

    logger.info("%s chunks charges pour %s", len(chunks), ",".join(corpora))
    return chunks


def build_bm25_corpus(chunks: List[Dict]) -> List[Dict]:
    return [
        {
            "id": chunk.get("id"),
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}) or {},
        }
        for chunk in chunks
    ]


def embed(texts: List[str], cache: Dict[str, Dict[str, List[float]]]) -> List[List[float]]:
    model = get_embedding_model()
    model_name = get_active_model_name()
    namespace = get_cache_namespace(model_name)
    model_cache = cache.setdefault("models", {}).setdefault(namespace, {})

    embeddings: List[List[float]] = []
    new_cache = False

    for index in range(0, len(texts), BATCH_SIZE):
        batch = texts[index : index + BATCH_SIZE]
        batch_embeddings: List[List[float]] = []
        to_compute: List[str] = []
        idx_map: List[int] = []

        for position, text in enumerate(batch):
            prepared = prepare_passage_text(text, model_name)
            digest = get_hash(prepared)
            cached = model_cache.get(digest)
            if cached is not None:
                batch_embeddings.append(cached)
            else:
                batch_embeddings.append([])
                to_compute.append(prepared)
                idx_map.append(position)

        if to_compute:
            computed = model.encode(to_compute, normalize_embeddings=True)
            for position, embedding in enumerate(computed):
                batch_index = idx_map[position]
                prepared = to_compute[position]
                digest = get_hash(prepared)
                embedding_list = embedding.tolist()
                batch_embeddings[batch_index] = embedding_list
                model_cache[digest] = embedding_list
                new_cache = True

        embeddings.extend(batch_embeddings)
        logger.info("Progress embeddings: %s/%s", index + len(batch), len(texts))

    if new_cache:
        save_cache(cache)
    return embeddings
