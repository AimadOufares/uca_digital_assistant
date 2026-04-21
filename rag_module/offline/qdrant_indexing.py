import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..shared.index_manifest import build_manifest, load_manifest, save_manifest, validate_manifest
from ..shared.runtime_config import (
    qdrant_active_alias_name,
    qdrant_active_chunks_snapshot_path,
    qdrant_active_manifest_path,
    qdrant_active_sparse_encoder_path,
    qdrant_candidate_manifests_dir,
    qdrant_candidate_snapshots_dir,
    qdrant_candidate_sparse_dir,
    qdrant_collection_name,
    qdrant_collection_prefix,
    qdrant_keep_previous_index,
    qdrant_local_path,
    qdrant_previous_alias_name,
    qdrant_url,
)
from ..shared.sparse_vectors import build_sparse_encoder, encode_sparse_text, save_sparse_encoder

logger = logging.getLogger(__name__)


QDRANT_SPARSE_ENCODER_PATH = qdrant_active_sparse_encoder_path()
QDRANT_MANIFEST_PATH = qdrant_active_manifest_path()
QDRANT_CHUNKS_PATH = qdrant_active_chunks_snapshot_path()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "lexical"
UPSERT_BATCH_SIZE = 64
SPARSE_MIN_DF = 3
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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug() -> str:
    return _now_utc().strftime("%Y_%m_%d_%H%M%S")


def build_candidate_collection_name(prefix: str = "") -> str:
    base = (prefix or qdrant_collection_prefix()).strip() or qdrant_collection_name()
    return f"{base}_{_timestamp_slug()}"


def candidate_manifest_path(collection_name: str) -> Path:
    return qdrant_candidate_manifests_dir() / f"{collection_name}.manifest.json"


def candidate_sparse_encoder_path(collection_name: str) -> Path:
    return qdrant_candidate_sparse_dir() / f"{collection_name}.sparse.json"


def candidate_chunks_snapshot_path(collection_name: str) -> Path:
    return qdrant_candidate_snapshots_dir() / f"{collection_name}.chunks.json"


def list_candidate_manifest_paths() -> List[Path]:
    base_dir = qdrant_candidate_manifests_dir()
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob("*.manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)


def latest_candidate_manifest_path() -> Path | None:
    manifests = list_candidate_manifest_paths()
    return manifests[0] if manifests else None


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


def _save_json_atomic(path: Path, payload: Dict | List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _write_chunks_snapshot(path: Path, chunks: List[Dict]) -> None:
    _save_json_atomic(path, chunks)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.tmp")
    shutil.copyfile(source, temp_path)
    os.replace(temp_path, destination)


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        pass

    try:
        collections = client.get_collections()
        items = getattr(collections, "collections", []) or []
        return any(getattr(item, "name", "") == collection_name for item in items)
    except Exception:
        return False


def get_qdrant_alias_map(client: QdrantClient | None = None) -> Dict[str, str]:
    active_client = client or get_qdrant_client()
    try:
        response = active_client.get_aliases()
    except Exception:
        return {}

    aliases = getattr(response, "aliases", []) or []
    mapping: Dict[str, str] = {}
    for alias in aliases:
        alias_name = str(getattr(alias, "alias_name", "") or "").strip()
        collection_name = str(getattr(alias, "collection_name", "") or "").strip()
        if alias_name and collection_name:
            mapping[alias_name] = collection_name
    return mapping


def resolve_alias_target(alias_name: str, client: QdrantClient | None = None) -> str:
    return str(get_qdrant_alias_map(client).get(alias_name, "") or "")


def _payload_index_fields() -> Iterable[tuple[str, models.PayloadSchemaType]]:
    return (
        ("metadata.faculty", models.PayloadSchemaType.KEYWORD),
        ("metadata.etablissement", models.PayloadSchemaType.KEYWORD),
        ("metadata.year", models.PayloadSchemaType.INTEGER),
        ("metadata.document_type", models.PayloadSchemaType.KEYWORD),
        ("metadata.section_title", models.PayloadSchemaType.KEYWORD),
        ("metadata.language", models.PayloadSchemaType.KEYWORD),
    )


def build_qdrant_index(
    chunks: List[Dict],
    model_name: str,
    embedding_dim: int,
    dense_vectors: List[List[float]],
    requested_model_name: str = "",
    fallback_model_name: str = "",
    target_collection_name: str = "",
    manifest_path: Path | None = None,
    sparse_encoder_path: Path | None = None,
    chunks_snapshot_path: Path | None = None,
) -> Dict:
    if not chunks:
        raise RuntimeError("Aucun chunk disponible pour l'indexation Qdrant.")
    if len(chunks) != len(dense_vectors):
        raise RuntimeError("Le nombre de chunks et de vecteurs denses ne correspond pas.")

    collection_name = (target_collection_name or qdrant_collection_name()).strip() or qdrant_collection_name()
    manifest_target = Path(manifest_path or candidate_manifest_path(collection_name))
    sparse_target = Path(sparse_encoder_path or candidate_sparse_encoder_path(collection_name))
    chunks_target = Path(chunks_snapshot_path or candidate_chunks_snapshot_path(collection_name))

    client = get_qdrant_client()

    sparse_encoder = build_sparse_encoder(
        (chunk.get("indexed_text", chunk.get("text", "")) or "" for chunk in chunks),
        min_df=SPARSE_MIN_DF,
        max_features=SPARSE_MAX_FEATURES,
    )
    save_sparse_encoder(str(sparse_target), sparse_encoder)

    if _collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=int(embedding_dim),
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=_env_int("RAG_QDRANT_HNSW_M", 48),
                    ef_construct=_env_int("RAG_QDRANT_EF_CONSTRUCT", 256),
                ),
            )
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        hnsw_config=models.HnswConfigDiff(
            m=_env_int("RAG_QDRANT_HNSW_M", 48),
            ef_construct=_env_int("RAG_QDRANT_EF_CONSTRUCT", 256),
        ),
        on_disk_payload=True,
    )

    for field_name, schema in _payload_index_fields():
        client.create_payload_index(collection_name, field_name, field_schema=schema)

    points: List[models.PointStruct] = []
    for index, (chunk, dense_vector) in enumerate(zip(chunks, dense_vectors)):
        indexed_text = chunk.get("indexed_text", chunk.get("text", "")) or ""
        indices, values = encode_sparse_text(indexed_text, sparse_encoder)
        payload = {
            "id": chunk.get("id") or chunk.get("metadata", {}).get("chunk_hash") or str(index),
            "text": chunk.get("text", "") or "",
            "indexed_text": indexed_text,
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

    _write_chunks_snapshot(chunks_target, chunks)
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
        sparse_encoder_path=str(sparse_target),
        sparse_encoder_type="tfidf_lexical",
        hybrid_fusion="rrf",
        indexed_at=_now_utc().isoformat(),
        chunks_snapshot_path=str(chunks_target),
        manifest_path=str(manifest_target),
        candidate=True,
    )
    if requested_model_name:
        manifest["requested_model_name"] = requested_model_name
    if fallback_model_name and fallback_model_name != requested_model_name:
        manifest["fallback_model_name"] = fallback_model_name

    save_manifest(str(manifest_target), manifest)
    logger.info("Index Qdrant candidat cree | collection=%s | chunks=%s", collection_name, len(chunks))
    return manifest


def publish_qdrant_index(candidate_manifest: Dict | None = None, manifest_path: str | Path | None = None) -> Dict:
    manifest = dict(candidate_manifest or load_manifest(str(manifest_path or "")))
    validate_manifest(
        manifest,
        expected_model=str(manifest.get("model_name") or ""),
        expected_vector_store="qdrant",
    )

    client = get_qdrant_client()
    candidate_collection = str(manifest.get("collection_name") or "").strip()
    if not candidate_collection:
        raise RuntimeError("Manifest candidat invalide: collection_name manquant.")
    if not _collection_exists(client, candidate_collection):
        raise RuntimeError(f"Collection Qdrant candidate introuvable: {candidate_collection}")

    current_alias = qdrant_active_alias_name()
    previous_alias = qdrant_previous_alias_name()
    current_target = resolve_alias_target(current_alias, client)
    previous_target = resolve_alias_target(previous_alias, client)

    operations: List[models.CreateAliasOperation | models.DeleteAliasOperation] = []
    if previous_target:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=previous_alias)))
    if current_target:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=current_alias)))

    keep_previous = qdrant_keep_previous_index()
    if keep_previous and current_target and current_target != candidate_collection:
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(alias_name=previous_alias, collection_name=current_target)
            )
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(alias_name=current_alias, collection_name=candidate_collection)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)

    sparse_source = Path(str(manifest.get("sparse_encoder_path") or "")).resolve()
    chunks_source = Path(str(manifest.get("chunks_snapshot_path") or "")).resolve()
    manifest_path_raw = str(manifest.get("manifest_path") or manifest_path or "").strip()
    candidate_manifest_path_value = Path(manifest_path_raw).resolve() if manifest_path_raw else None

    if sparse_source.exists():
        _copy_file_atomic(sparse_source, QDRANT_SPARSE_ENCODER_PATH)
    if chunks_source.exists():
        _copy_file_atomic(chunks_source, QDRANT_CHUNKS_PATH)

    published_manifest = dict(manifest)
    published_manifest["collection_name"] = current_alias
    published_manifest["active_alias_name"] = current_alias
    published_manifest["published_collection_name"] = candidate_collection
    published_manifest["previous_collection_name"] = current_target if current_target and current_target != candidate_collection else ""
    published_manifest["previous_alias_name"] = previous_alias if keep_previous else ""
    published_manifest["published_at"] = _now_utc().isoformat()
    published_manifest["candidate"] = False
    published_manifest["candidate_manifest_path"] = str(candidate_manifest_path_value) if candidate_manifest_path_value is not None else ""
    published_manifest["sparse_encoder_path"] = str(QDRANT_SPARSE_ENCODER_PATH)
    published_manifest["chunks_snapshot_path"] = str(QDRANT_CHUNKS_PATH)
    save_manifest(str(QDRANT_MANIFEST_PATH), published_manifest)

    logger.info(
        "Alias Qdrant publie | alias=%s -> %s | previous=%s",
        current_alias,
        candidate_collection,
        published_manifest["previous_collection_name"] or "-",
    )
    return published_manifest


def delete_qdrant_collection(collection_name: str, client: QdrantClient | None = None) -> bool:
    active_client = client or get_qdrant_client()
    target = (collection_name or "").strip()
    if not target:
        return False
    if not _collection_exists(active_client, target):
        return False
    active_client.delete_collection(collection_name=target)
    return True


def delete_candidate_artifacts(collection_name: str) -> List[str]:
    removed: List[str] = []
    for path in (
        candidate_manifest_path(collection_name),
        candidate_sparse_encoder_path(collection_name),
        candidate_chunks_snapshot_path(collection_name),
    ):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed
