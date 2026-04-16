import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..offline.preparation import verify_indexing_prerequisites
    from ..offline.qdrant_indexing import build_qdrant_index
    from ..shared.env_loader import load_env_file
    from ..shared.index_manifest import build_manifest, load_manifest, save_manifest
    from ..shared.runtime_config import configured_vector_backend
except ImportError:  # pragma: no cover
    from rag_module.offline.preparation import verify_indexing_prerequisites
    from rag_module.offline.qdrant_indexing import build_qdrant_index
    from rag_module.shared.env_loader import load_env_file
    from rag_module.shared.index_manifest import build_manifest, load_manifest, save_manifest
    from rag_module.shared.runtime_config import configured_vector_backend

load_env_file()


PROCESSED_PATH = "data_storage/processed"
INDEX_PATH = "data_storage/index"
CACHE_PATH = "data_storage/cache/embeddings_cache.json"
INDEX_MANIFEST_PATH = os.path.join(INDEX_PATH, "index_manifest.json")
BM25_CORPUS_PATH = os.path.join(INDEX_PATH, "bm25_corpus.json")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
]
BATCH_SIZE = 32
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

os.makedirs(INDEX_PATH, exist_ok=True)
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
        # Backward compatibility with flat hash -> embedding cache.
        legacy_model = get_cache_namespace(get_model_name())
        return {"version": 2, "models": {legacy_model: raw}}

    return {"version": 2, "models": {}}


def save_cache(cache: Dict[str, Dict[str, List[float]]]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)


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
                    "date_indexed": datetime.now(timezone.utc).isoformat(),
                }
            )

            chunks.append({"id": chunk_id, "text": text, "metadata": merged_metadata})
        except Exception as exc:
            logger.warning("Erreur fichier %s: %s", file, exc)

    logger.info("%s chunks charges", len(chunks))
    return chunks


def load_index_manifest() -> Dict:
    try:
        return load_manifest(INDEX_MANIFEST_PATH)
    except Exception as exc:
        logger.warning("Manifest illisible, regeneration demandee: %s", exc)
        return {}


def save_index_manifest(manifest: Dict) -> None:
    save_manifest(INDEX_MANIFEST_PATH, manifest)


def build_bm25_corpus(chunks: List[Dict]) -> List[Dict]:
    corpus: List[Dict] = []
    for chunk in chunks:
        corpus.append(
            {
                "id": chunk.get("id"),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}) or {},
            }
        )
    return corpus


def filter_chunks_for_reindex(chunks: List[Dict], manifest: Dict) -> List[Dict]:
    previous_model = str(manifest.get("model_name") or "").strip()
    current_model = get_model_name()
    if previous_model != current_model:
        return chunks
    return chunks


def _extract_processing_policy_version(chunks: List[Dict]) -> str:
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        candidate = metadata.get("processing_policy_version")
        if candidate:
            return str(candidate)
    return "unknown"


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
            h = get_hash(prepared_text)
            cached = model_cache.get(h)
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
                h = get_hash(prepared_text)
                emb_list = emb.tolist()
                batch_emb[j] = emb_list
                model_cache[h] = emb_list
                new_cache = True

        embeddings.extend(batch_emb)
        logger.info("Progress embeddings: %s/%s", i + len(batch), len(texts))

    if new_cache:
        save_cache(cache)

    return embeddings


def build_index(chunks: List[Dict]) -> None:
    verify_indexing_prerequisites()
    chunks_to_index = filter_chunks_for_reindex(chunks, load_index_manifest())
    texts = [c["text"] for c in chunks_to_index]
    if not texts:
        raise RuntimeError("Aucun texte disponible pour l'indexation.")

    cache = load_cache()
    embeddings = embed(texts, cache)
    vectors = np.array(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise RuntimeError("Aucun vecteur genere pour construire l'index.")

    dim = int(vectors.shape[1])
    active_backend = configured_vector_backend()
    dense_embeddings = vectors.tolist()

    if active_backend == "qdrant":
        manifest = build_qdrant_index(
            chunks=chunks_to_index,
            model_name=get_active_model_name(),
            embedding_dim=dim,
            dense_vectors=dense_embeddings,
        )
        manifest["requested_model_name"] = get_model_name()
        if get_active_model_name() != get_model_name():
            manifest["fallback_model_name"] = get_active_model_name()
        save_index_manifest(manifest)
        logger.info(
            "INDEX QDRANT CREE AVEC SUCCES | modele=%s | dim=%s | chunks=%s",
            get_active_model_name(),
            dim,
            len(chunks_to_index),
        )
        return

    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(vectors)

    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))
    with open(os.path.join(INDEX_PATH, "chunks.json"), "w", encoding="utf-8") as handle:
        json.dump(chunks_to_index, handle, ensure_ascii=False, indent=2)

    bm25_corpus = build_bm25_corpus(chunks_to_index)
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as handle:
        json.dump(bm25_corpus, handle, ensure_ascii=False, indent=2)

    manifest = build_manifest(
        model_name=get_active_model_name(),
        dim=dim,
        chunk_count=len(chunks_to_index),
        policy_version=_extract_processing_policy_version(chunks_to_index),
        index_type="faiss_hnsw_dense_plus_bm25",
        vector_store="faiss",
    )
    manifest["requested_model_name"] = get_model_name()
    if get_active_model_name() != get_model_name():
        manifest["fallback_model_name"] = get_active_model_name()
    save_index_manifest(manifest)

    logger.info(
        "INDEX CREE AVEC SUCCES | modele=%s | dim=%s | chunks=%s",
        get_active_model_name(),
        dim,
        len(chunks_to_index),
    )


if __name__ == "__main__":
    chunks_data = load_chunks()
    if not chunks_data:
        logger.error("Aucun chunk trouve !")
    else:
        build_index(chunks_data)
