import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

try:
    from ..shared.env_loader import load_env_file
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.env_loader import load_env_file
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


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


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

            seen_ids.add(chunk_id)
            seen_text_hashes.add(text_hash)

            source_path = metadata.get("source", str(file_path))
            source_name = metadata.get("file_name") or Path(source_path).name
            merged_metadata = dict(metadata)
            merged_metadata.update(
                {
                    "source": source_path,
                    "file_name": source_name,
                    "hash": chunk_id,
                    "date_indexed": datetime.now(timezone.utc).isoformat(),
                    "corpus": corpus_name,
                }
            )

            chunks.append({"id": chunk_id, "text": text, "metadata": merged_metadata})

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
