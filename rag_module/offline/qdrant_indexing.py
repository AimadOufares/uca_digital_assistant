import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..shared.index_manifest import build_manifest, save_manifest
from ..shared.runtime_config import qdrant_collection_name, qdrant_local_path, qdrant_url
from ..shared.sparse_vectors import build_sparse_encoder, encode_sparse_text, save_sparse_encoder

logger = logging.getLogger(__name__)


INDEX_DIR = Path("data_storage/index")
QDRANT_SPARSE_ENCODER_PATH = INDEX_DIR / "qdrant_sparse_encoder.json"
QDRANT_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
QDRANT_CHUNKS_PATH = INDEX_DIR / "chunks.json"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "lexical"
UPSERT_BATCH_SIZE = 64
SPARSE_MIN_DF = 1
SPARSE_MAX_FEATURES = 60000


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def get_qdrant_client() -> QdrantClient:
    url = qdrant_url()
    timeout = _env_int("RAG_QDRANT_TIMEOUT", 60)
    api_key = os.getenv("RAG_QDRANT_API_KEY", "").strip() or None
    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=timeout)

    local_path = qdrant_local_path()
    local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(local_path), timeout=timeout)


def _point_id(index: int) -> int:
    return int(index + 1)


def _write_chunks_snapshot(chunks: List[Dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with QDRANT_CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        collections = client.get_collections()
        items = getattr(collections, "collections", []) or []
        return any(getattr(item, "name", "") == collection_name for item in items)
    except Exception:
        return False


def build_qdrant_index(chunks: List[Dict], model_name: str, embedding_dim: int, dense_vectors: List[List[float]]) -> Dict:
    if not chunks:
        raise RuntimeError("Aucun chunk disponible pour l'indexation Qdrant.")
    if len(chunks) != len(dense_vectors):
        raise RuntimeError("Le nombre de chunks et de vecteurs denses ne correspond pas.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = get_qdrant_client()
    collection_name = qdrant_collection_name()

    sparse_encoder = build_sparse_encoder(
        (chunk.get("text", "") or "" for chunk in chunks),
        min_df=SPARSE_MIN_DF,
        max_features=SPARSE_MAX_FEATURES,
    )
    save_sparse_encoder(str(QDRANT_SPARSE_ENCODER_PATH), sparse_encoder)

    if _collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=int(embedding_dim),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams()
        },
        hnsw_config=models.HnswConfigDiff(
            m=_env_int("RAG_QDRANT_HNSW_M", 32),
            ef_construct=_env_int("RAG_QDRANT_EF_CONSTRUCT", 200),
        ),
        on_disk_payload=True,
    )

    points: List[models.PointStruct] = []
    for index, (chunk, dense_vector) in enumerate(zip(chunks, dense_vectors)):
        indices, values = encode_sparse_text(chunk.get("text", "") or "", sparse_encoder)
        payload = {
            "id": chunk.get("id") or chunk.get("metadata", {}).get("chunk_hash") or str(index),
            "text": chunk.get("text", "") or "",
            "metadata": chunk.get("metadata", {}) or {},
        }
        points.append(
            models.PointStruct(
                id=_point_id(index),
                vector={
                    DENSE_VECTOR_NAME: dense_vector,
                    SPARSE_VECTOR_NAME: models.SparseVector(indices=indices, values=values),
                },
                payload=payload,
            )
        )

        if len(points) >= UPSERT_BATCH_SIZE:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)

    _write_chunks_snapshot(chunks)
    manifest = build_manifest(
        model_name=model_name,
        dim=embedding_dim,
        chunk_count=len(chunks),
        policy_version=str((chunks[0].get("metadata", {}) or {}).get("processing_policy_version") or "unknown"),
        index_type="qdrant_hybrid_dense_sparse",
        vector_store="qdrant",
        collection_name=collection_name,
        dense_vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        sparse_encoder_path=str(QDRANT_SPARSE_ENCODER_PATH),
        sparse_encoder_type="tfidf_lexical",
        hybrid_fusion="rrf",
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_manifest(str(QDRANT_MANIFEST_PATH), manifest)
    logger.info("Index Qdrant cree | collection=%s | chunks=%s", collection_name, len(chunks))
    return manifest
