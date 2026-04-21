import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client.http import models

from ..offline.indexing import get_active_model_name, get_embedding_model, is_e5_model
from ..offline.qdrant_indexing import (
    DENSE_VECTOR_NAME,
    QDRANT_MANIFEST_PATH,
    QDRANT_SPARSE_ENCODER_PATH,
    SPARSE_VECTOR_NAME,
    get_qdrant_alias_map,
    get_qdrant_client,
)
from ..shared.index_manifest import load_manifest, validate_manifest
from ..shared.runtime_config import qdrant_active_alias_name, qdrant_collection_name
from ..shared.sparse_vectors import encode_sparse_text, load_sparse_encoder

logger = logging.getLogger(__name__)


def _legacy_search_module():
    from . import rag_search as legacy

    return legacy


def _prepare_query_text(query: str) -> str:
    normalized = (query or "").strip()
    if is_e5_model(get_active_model_name()):
        return f"query: {normalized}"
    return normalized


@lru_cache(maxsize=256)
def _embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    vector = model.encode(_prepare_query_text(query), normalize_embeddings=True)
    return vector.tolist()


def load_qdrant_manifest_or_raise(manifest_override: Optional[Dict] = None) -> Dict:
    manifest = dict(manifest_override or load_manifest(str(QDRANT_MANIFEST_PATH)))
    validate_manifest(
        manifest,
        expected_model=get_active_model_name(),
        expected_vector_store="qdrant",
    )
    return manifest


def _resolve_collection_name(manifest: Dict) -> str:
    alias_name = str(manifest.get("active_alias_name") or "").strip()
    collection_name = str(manifest.get("collection_name") or "").strip()
    if alias_name:
        return alias_name
    return collection_name or qdrant_collection_name()


def _resolve_sparse_encoder_path(manifest: Dict) -> Path:
    raw_value = str(manifest.get("sparse_encoder_path") or QDRANT_SPARSE_ENCODER_PATH).strip()
    return Path(raw_value)


def qdrant_index_ready(manifest_override: Optional[Dict] = None) -> bool:
    try:
        manifest = load_qdrant_manifest_or_raise(manifest_override=manifest_override)
        client = get_qdrant_client()
        collection_name = _resolve_collection_name(manifest)
        sparse_encoder_path = _resolve_sparse_encoder_path(manifest)
        collections = client.get_collections()
        items = getattr(collections, "collections", []) or []
        names = {getattr(item, "name", "") for item in items}
        alias_map = get_qdrant_alias_map(client)
        target_name = alias_map.get(collection_name, collection_name)
        return target_name in names and sparse_encoder_path.exists()
    except Exception:
        return False


def _query_filter(query_profile: Dict, allowed_establishments: Optional[List[str]] = None) -> models.Filter | None:
    faculties = list(allowed_establishments or query_profile.get("faculties", []) or [])
    years = list(query_profile.get("years", []) or [])
    must_conditions: List[object] = []
    if faculties:
        faculty_match = models.MatchAny(any=faculties)
        must_conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="metadata.etablissement", match=faculty_match),
                    models.FieldCondition(key="metadata.faculty", match=faculty_match),
                ]
            )
        )
    if len(years) == 1:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.year",
                match=models.MatchValue(value=int(years[0])),
            )
        )
    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def _normalize_scored_points(points, score_type: str, score_field: str) -> List[Dict]:
    rows = list(getattr(points, "points", []) or points or [])
    if not rows:
        return []
    max_score = max(float(getattr(point, "score", 0.0) or 0.0) for point in rows) or 1.0
    results: List[Dict] = []
    for point in rows:
        payload = getattr(point, "payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        raw_score = float(getattr(point, "score", 0.0) or 0.0)
        entry = {
            "id": payload.get("id") or metadata.get("chunk_hash") or str(getattr(point, "id", "")),
            "text": payload.get("text", "") or "",
            "metadata": metadata,
            "score": max(0.0, min(1.0, raw_score / max_score)),
            "score_type": score_type,
            score_field: raw_score,
        }
        results.append(entry)
    return results


def _merge_query_variant_results(result_batches: List[List[Dict]], limit: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for batch in result_batches:
        for result in batch:
            metadata = result.get("metadata", {}) or {}
            chunk_id = result.get("id") or metadata.get("chunk_hash") or metadata.get("hash")
            if not chunk_id:
                continue

            current = merged.get(chunk_id)
            if current is None:
                merged[chunk_id] = dict(result)
                continue

            if float(result.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                current["score"] = float(result.get("score", 0.0) or 0.0)
                current["text"] = result.get("text", current.get("text", ""))
                current["metadata"] = metadata or current.get("metadata", {})

            for raw_field in ("vector_raw_score", "sparse_raw_score", "hybrid_raw_score"):
                if raw_field in result:
                    current[raw_field] = max(
                        float(current.get(raw_field, float("-inf")) or float("-inf")),
                        float(result.get(raw_field, 0.0) or 0.0),
                    )

    rows = list(merged.values())
    rows.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return rows[:limit]


def _time_search_call(search_fn, *args, **kwargs) -> tuple[List[Dict], float]:
    started = time.perf_counter()
    results = search_fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return results, round(elapsed_ms, 2)


def _serialize_query_filter(query_filter: models.Filter | None) -> Dict | None:
    if query_filter is None:
        return None
    try:
        if hasattr(query_filter, "model_dump"):
            return query_filter.model_dump(exclude_none=True)
        if hasattr(query_filter, "dict"):
            return query_filter.dict(exclude_none=True)  # type: ignore[attr-defined]
    except Exception:
        return {"repr": repr(query_filter)}
    return {"repr": repr(query_filter)}


def _dense_search(query: str, query_filter: models.Filter | None, limit: int, manifest: Optional[Dict] = None) -> List[Dict]:
    active_manifest = load_qdrant_manifest_or_raise(manifest_override=manifest)
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=_resolve_collection_name(active_manifest),
        query=_embed_query(query),
        using=str(active_manifest.get("dense_vector_name") or DENSE_VECTOR_NAME),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="dense", score_field="vector_raw_score")


def _sparse_query_vector(query: str, manifest: Optional[Dict] = None) -> models.SparseVector | None:
    active_manifest = load_qdrant_manifest_or_raise(manifest_override=manifest)
    encoder = load_sparse_encoder(str(_resolve_sparse_encoder_path(active_manifest)))
    indices, values = encode_sparse_text(query, encoder)
    if not indices or not values:
        return None
    return models.SparseVector(indices=indices, values=values)


def _sparse_search(query: str, query_filter: models.Filter | None, limit: int, manifest: Optional[Dict] = None) -> List[Dict]:
    active_manifest = load_qdrant_manifest_or_raise(manifest_override=manifest)
    sparse_query = _sparse_query_vector(query, manifest=active_manifest)
    if sparse_query is None:
        return []
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=_resolve_collection_name(active_manifest),
        query=sparse_query,
        using=str(active_manifest.get("sparse_vector_name") or SPARSE_VECTOR_NAME),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="sparse", score_field="sparse_raw_score")


def _fusion_search(query: str, query_filter: models.Filter | None, limit: int, manifest: Optional[Dict] = None) -> List[Dict]:
    active_manifest = load_qdrant_manifest_or_raise(manifest_override=manifest)
    sparse_query = _sparse_query_vector(query, manifest=active_manifest)
    client = get_qdrant_client()
    prefetch = [
        models.Prefetch(
            query=_embed_query(query),
            using=str(active_manifest.get("dense_vector_name") or DENSE_VECTOR_NAME),
            filter=query_filter,
            limit=limit,
        )
    ]
    if sparse_query is not None:
        prefetch.append(
            models.Prefetch(
                query=sparse_query,
                using=str(active_manifest.get("sparse_vector_name") or SPARSE_VECTOR_NAME),
                filter=query_filter,
                limit=limit,
            )
        )

    response = client.query_points(
        collection_name=_resolve_collection_name(active_manifest),
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="hybrid", score_field="hybrid_raw_score")


def _multi_query_dense_search(
    queries: List[str],
    query_filter: models.Filter | None,
    limit: int,
    manifest: Optional[Dict] = None,
) -> List[Dict]:
    batches = [_dense_search(query, query_filter=query_filter, limit=limit, manifest=manifest) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def _multi_query_sparse_search(
    queries: List[str],
    query_filter: models.Filter | None,
    limit: int,
    manifest: Optional[Dict] = None,
) -> List[Dict]:
    batches = [_sparse_search(query, query_filter=query_filter, limit=limit, manifest=manifest) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def _multi_query_fusion_search(
    queries: List[str],
    query_filter: models.Filter | None,
    limit: int,
    manifest: Optional[Dict] = None,
) -> List[Dict]:
    batches = [_fusion_search(query, query_filter=query_filter, limit=limit, manifest=manifest) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def run_qdrant_candidate_search(
    query: str,
    queries: List[str],
    query_profile: Dict,
    retrieve_k: int,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict] = None,
) -> Dict[str, object]:
    active_manifest = load_qdrant_manifest_or_raise(manifest_override=manifest_override)
    normalized_queries = [item for item in queries if (item or "").strip()]
    query_filter = _query_filter(query_profile, allowed_establishments=allowed_establishments)

    dense_results, dense_latency_ms = _time_search_call(
        _multi_query_dense_search,
        normalized_queries,
        query_filter=query_filter,
        limit=retrieve_k,
        manifest=active_manifest,
    )
    sparse_results, sparse_latency_ms = _time_search_call(
        _multi_query_sparse_search,
        normalized_queries,
        query_filter=query_filter,
        limit=retrieve_k,
        manifest=active_manifest,
    )
    fusion_results, fusion_latency_ms = _time_search_call(
        _multi_query_fusion_search,
        normalized_queries,
        query_filter=query_filter,
        limit=retrieve_k,
        manifest=active_manifest,
    )

    return {
        "query": query,
        "queries": normalized_queries,
        "retrieve_k": int(retrieve_k),
        "query_filter_applied": query_filter is not None,
        "query_filter": _serialize_query_filter(query_filter),
        "manifest": {
            "collection_name": _resolve_collection_name(active_manifest),
            "published_collection_name": str(active_manifest.get("published_collection_name") or ""),
            "active_alias_name": str(active_manifest.get("active_alias_name") or qdrant_active_alias_name()),
            "model_name": str(active_manifest.get("model_name") or ""),
            "vector_store": str(active_manifest.get("vector_store") or "qdrant"),
        },
        "latency_ms": {
            "dense": dense_latency_ms,
            "sparse": sparse_latency_ms,
            "fusion": fusion_latency_ms,
            "total": round(dense_latency_ms + sparse_latency_ms + fusion_latency_ms, 2),
        },
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "fusion_results": fusion_results,
    }


def run_qdrant_search_debug(
    raw_query: str,
    top_k: int = 5,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict] = None,
) -> Dict[str, object]:
    legacy = _legacy_search_module()
    return legacy.run_hybrid_search_debug(
        raw_query,
        top_k=top_k,
        allowed_establishments=allowed_establishments,
        manifest_override=manifest_override,
    )


def get_relevant_chunks_qdrant(
    raw_query: str,
    top_k: int = 5,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict] = None,
) -> List[Dict]:
    legacy = _legacy_search_module()
    return legacy.get_relevant_chunks(
        raw_query,
        top_k=top_k,
        allowed_establishments=allowed_establishments,
        manifest_override=manifest_override,
    )
