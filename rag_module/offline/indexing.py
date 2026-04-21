import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..offline.preparation import verify_qdrant_indexing_prerequisites
    from ..offline.qdrant_indexing import build_candidate_collection_name, build_qdrant_index
    from ..shared.env_loader import load_env_file
except ImportError:  # pragma: no cover
    from rag_module.offline.preparation import verify_qdrant_indexing_prerequisites
    from rag_module.offline.qdrant_indexing import build_candidate_collection_name, build_qdrant_index
    from rag_module.shared.env_loader import load_env_file

load_env_file()


PROCESSED_PATH = "data_storage/processed"
INDEX_PATH = "data_storage/index"
CACHE_PATH = "data_storage/cache/embeddings_cache.json"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
]
BATCH_SIZE = 32

os.makedirs(INDEX_PATH, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_embedding_model = None
_embedding_model_name = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def get_model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def embedding_model_is_strict() -> bool:
    return _env_bool("RAG_STRICT_EMBEDDING_MODEL", True)


def get_candidate_model_names() -> List[str]:
    requested = get_model_name()
    if embedding_model_is_strict():
        return [requested]

    candidates = [requested, *FALLBACK_EMBEDDING_MODELS]
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
            model = SentenceTransformer(
                model_name,
                device="cpu",
                local_files_only=True,
            )
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    for model_name in model_names:
        try:
            model = SentenceTransformer(model_name, device="cpu")
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name} (online): {exc}")

    raise RuntimeError(
        "Aucun modele d'embedding compatible n'est disponible. "
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
    temp_path = CACHE_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    os.replace(temp_path, CACHE_PATH)


def build_indexable_text(text: str, metadata: Dict) -> str:
    section_title = normalize(str(metadata.get("section_title") or ""))
    section_path = normalize(" ".join(str(part).strip() for part in metadata.get("section_path", []) if str(part).strip()))
    document_type = normalize(str(metadata.get("document_type") or ""))
    establishment = normalize(str(metadata.get("etablissement") or metadata.get("faculty") or ""))
    year = metadata.get("year")

    parts = [
        section_title,
        section_path,
        document_type,
        establishment,
        str(year).strip() if year else "",
        normalize(text),
    ]
    return "\n".join(part for part in parts if part)


def load_chunks() -> List[Dict]:
    files = sorted(Path(PROCESSED_PATH).glob("*.json"))
    chunks: List[Dict] = []
    seen_ids = set()
    seen_text_hashes = set()

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

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

            source_path = metadata.get("source", str(file))
            source_name = metadata.get("file_name") or Path(source_path).name
            merged_metadata = dict(metadata)
            merged_metadata.update(
                {
                    "source": source_path,
                    "file_name": source_name,
                    "hash": chunk_id,
                }
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "indexed_text": build_indexable_text(text, merged_metadata),
                    "metadata": merged_metadata,
                }
            )
        except Exception as exc:
            logger.warning("Erreur fichier %s: %s", file, exc)

    logger.info("%s chunks charges", len(chunks))
    return chunks


def select_chunks_for_full_rebuild(chunks: List[Dict]) -> List[Dict]:
    return list(chunks)


def embed(texts: List[str], cache: Dict[str, Dict[str, List[float]]]) -> List[List[float]]:
    model = get_embedding_model()
    model_name = get_active_model_name()
    namespace = get_cache_namespace(model_name)
    model_cache = cache.setdefault("models", {}).setdefault(namespace, {})

    embeddings: List[List[float]] = []
    new_cache = False

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        batch_emb: List[List[float]] = []
        to_compute: List[str] = []
        idx_map: List[int] = []

        for j, text in enumerate(batch):
            prepared_text = prepare_passage_text(text, model_name)
            cache_key = get_hash(prepared_text)
            cached = model_cache.get(cache_key)
            if cached is not None:
                batch_emb.append(cached)
            else:
                batch_emb.append([])
                to_compute.append(prepared_text)
                idx_map.append(j)

        if to_compute:
            computed = model.encode(to_compute, normalize_embeddings=True)
            for k, emb in enumerate(computed):
                j = idx_map[k]
                prepared_text = to_compute[k]
                cache_key = get_hash(prepared_text)
                emb_list = emb.tolist()
                batch_emb[j] = emb_list
                model_cache[cache_key] = emb_list
                new_cache = True

        embeddings.extend(batch_emb)
        logger.info("Progress embeddings: %s/%s", i + len(batch), len(texts))

    if new_cache:
        save_cache(cache)

    return embeddings


def build_index(
    chunks: List[Dict],
    target_collection_name: str = "",
    manifest_path=None,
    sparse_encoder_path=None,
    chunks_snapshot_path=None,
) -> Dict:
    verify_qdrant_indexing_prerequisites()
    chunks_to_index = select_chunks_for_full_rebuild(chunks)
    texts = [chunk.get("indexed_text", chunk["text"]) for chunk in chunks_to_index]
    if not texts:
        raise RuntimeError("Aucun texte disponible pour l'indexation.")

    cache = load_cache()
    embeddings = embed(texts, cache)
    vectors = np.array(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise RuntimeError("Aucun vecteur genere pour construire l'index.")

    dim = int(vectors.shape[1])
    dense_embeddings = vectors.tolist()

    requested_model_name = get_model_name()
    active_model_name = get_active_model_name()
    fallback_model_name = active_model_name if active_model_name != requested_model_name else ""

    effective_target = (target_collection_name or "").strip() or build_candidate_collection_name()
    manifest = build_qdrant_index(
        chunks=chunks_to_index,
        model_name=active_model_name,
        embedding_dim=dim,
        dense_vectors=dense_embeddings,
        requested_model_name=requested_model_name,
        fallback_model_name=fallback_model_name,
        target_collection_name=effective_target,
        manifest_path=manifest_path,
        sparse_encoder_path=sparse_encoder_path,
        chunks_snapshot_path=chunks_snapshot_path,
    )
    logger.info(
        "INDEX QDRANT CANDIDAT CREE AVEC SUCCES | collection=%s | modele=%s | dim=%s | chunks=%s",
        effective_target,
        active_model_name,
        dim,
        len(chunks_to_index),
    )
    return manifest


if __name__ == "__main__":
    chunks_data = load_chunks()
    if not chunks_data:
        logger.error("Aucun chunk trouve !")
    else:
        build_index(chunks_data)
