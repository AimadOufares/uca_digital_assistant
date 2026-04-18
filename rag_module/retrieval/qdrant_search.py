import logging
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
    get_qdrant_client,
)
from ..shared.index_manifest import load_manifest, validate_manifest
from ..shared.runtime_config import qdrant_collection_name
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


def load_qdrant_manifest_or_raise() -> Dict:
    manifest = load_manifest(str(QDRANT_MANIFEST_PATH))
    validate_manifest(
        manifest,
        expected_model=get_active_model_name(),
        expected_vector_store="qdrant",
    )
    return manifest


def qdrant_index_ready() -> bool:
    try:
        manifest = load_qdrant_manifest_or_raise()
        client = get_qdrant_client()
        collection_name = str(manifest.get("collection_name") or qdrant_collection_name())
        collections = client.get_collections()
        items = getattr(collections, "collections", []) or []
        names = {getattr(item, "name", "") for item in items}
        return collection_name in names and Path(QDRANT_SPARSE_ENCODER_PATH).exists()
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


def _dense_search(query: str, query_filter: models.Filter | None, limit: int) -> List[Dict]:
    manifest = load_qdrant_manifest_or_raise()
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=str(manifest.get("collection_name") or qdrant_collection_name()),
        query=_embed_query(query),
        using=str(manifest.get("dense_vector_name") or DENSE_VECTOR_NAME),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="dense", score_field="vector_raw_score")


def _sparse_query_vector(query: str) -> models.SparseVector | None:
    encoder = load_sparse_encoder(str(QDRANT_SPARSE_ENCODER_PATH))
    indices, values = encode_sparse_text(query, encoder)
    if not indices or not values:
        return None
    return models.SparseVector(indices=indices, values=values)


def _sparse_search(query: str, query_filter: models.Filter | None, limit: int) -> List[Dict]:
    sparse_query = _sparse_query_vector(query)
    if sparse_query is None:
        return []
    manifest = load_qdrant_manifest_or_raise()
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=str(manifest.get("collection_name") or qdrant_collection_name()),
        query=sparse_query,
        using=str(manifest.get("sparse_vector_name") or SPARSE_VECTOR_NAME),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="sparse", score_field="sparse_raw_score")


def _fusion_search(query: str, query_filter: models.Filter | None, limit: int) -> List[Dict]:
    sparse_query = _sparse_query_vector(query)
    manifest = load_qdrant_manifest_or_raise()
    client = get_qdrant_client()
    prefetch = [
        models.Prefetch(
            query=_embed_query(query),
            using=str(manifest.get("dense_vector_name") or DENSE_VECTOR_NAME),
            filter=query_filter,
            limit=limit,
        )
    ]
    if sparse_query is not None:
        prefetch.append(
            models.Prefetch(
                query=sparse_query,
                using=str(manifest.get("sparse_vector_name") or SPARSE_VECTOR_NAME),
                filter=query_filter,
                limit=limit,
            )
        )

    response = client.query_points(
        collection_name=str(manifest.get("collection_name") or qdrant_collection_name()),
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _normalize_scored_points(response, score_type="hybrid", score_field="hybrid_raw_score")


def _multi_query_dense_search(queries: List[str], query_filter: models.Filter | None, limit: int) -> List[Dict]:
    batches = [_dense_search(query, query_filter=query_filter, limit=limit) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def _multi_query_sparse_search(queries: List[str], query_filter: models.Filter | None, limit: int) -> List[Dict]:
    batches = [_sparse_search(query, query_filter=query_filter, limit=limit) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def _multi_query_fusion_search(queries: List[str], query_filter: models.Filter | None, limit: int) -> List[Dict]:
    batches = [_fusion_search(query, query_filter=query_filter, limit=limit) for query in queries]
    return _merge_query_variant_results(batches, limit=limit)


def run_qdrant_search_debug(
    raw_query: str,
    top_k: int = 5,
    allowed_establishments: Optional[List[str]] = None,
) -> Dict[str, object]:
    legacy = _legacy_search_module()
    if not raw_query or not raw_query.strip():
        return {
            "query": "",
            "dense_results": [],
            "bm25_results": [],
            "merged_results": [],
            "boosted_results": [],
            "guarded_results": [],
            "final_results": [],
            "abstain": True,
            "abstain_reason": "empty_query",
            "query_profile": {},
        }

    top_k = max(1, int(top_k))
    query = legacy.enhance_query(raw_query)
    normalized_allowed = legacy._normalize_allowed_establishments(allowed_establishments)
    query_profile = legacy.build_query_profile(query, allowed_establishments=normalized_allowed)
    query_filter = _query_filter(query_profile, allowed_establishments=normalized_allowed)
    retrieve_k = max(legacy.TOP_K_RETRIEVE * 2, top_k * 6)
    if normalized_allowed:
        retrieve_k = max(retrieve_k, 60)
    queries = legacy.generate_multi_queries(query)

    dense_results = _multi_query_dense_search(queries, query_filter=query_filter, limit=retrieve_k)
    sparse_results = _multi_query_sparse_search(queries, query_filter=query_filter, limit=retrieve_k)
    fusion_results = _multi_query_fusion_search(queries, query_filter=query_filter, limit=retrieve_k)
    merged_results = legacy.merge_dense_and_bm25(dense_results, sparse_results, top_k=retrieve_k)

    dense_by_id = {result.get("id"): result for result in dense_results}
    sparse_by_id = {result.get("id"): result for result in sparse_results}
    manual_merged_by_id = {result.get("id"): result for result in merged_results}
    ranking_seed = fusion_results or merged_results
    enriched_merged: List[Dict] = []
    for result in ranking_seed:
        item = dict(result)
        fallback_scores = manual_merged_by_id.get(item.get("id"), {})
        item["dense_score"] = float(dense_by_id.get(item.get("id"), {}).get("score", 0.0) or 0.0)
        item["bm25_score"] = float(sparse_by_id.get(item.get("id"), {}).get("score", 0.0) or 0.0)
        if "hybrid_raw_score" not in item and "hybrid_raw_score" in fallback_scores:
            item["hybrid_raw_score"] = float(fallback_scores.get("hybrid_raw_score", 0.0) or 0.0)
        if item.get("score_type") != "hybrid":
            item["score_type"] = str(fallback_scores.get("score_type") or item.get("score_type") or "hybrid")
        item["retrieval_sources"] = [
            source
            for source, lookup in (("dense", dense_by_id), ("sparse", sparse_by_id))
            if item.get("id") in lookup
        ]
        enriched_merged.append(item)

    boosted = legacy._filter_results_by_allowed_establishments(
        legacy.apply_metadata_boost(enriched_merged, query),
        normalized_allowed,
    )
    guarded, guardrail_diagnostics = legacy.apply_retrieval_guardrails(
        query,
        boosted,
        top_k=top_k,
        allowed_establishments=normalized_allowed,
    )
    reranked = legacy.rerank_chunks(query, guarded, top_k=max(top_k * 2, top_k))
    final_ranked = legacy.apply_post_rerank_guardrails(
        legacy._filter_results_by_allowed_establishments(reranked, normalized_allowed),
        query_profile=guardrail_diagnostics.get("query_profile", {}),
        top_k=top_k,
    )
    abstention = legacy.decide_retrieval_abstention(final_ranked, guardrail_diagnostics.get("query_profile", {}))
    fallback_results = legacy.select_support_fallback_results(
        legacy._filter_results_by_allowed_establishments(guarded, normalized_allowed),
        query_profile=guardrail_diagnostics.get("query_profile", {}),
        top_k=top_k,
    )
    if abstention["abstain"] and abstention["reason"] == "top_rerank_too_low" and fallback_results:
        final_results = legacy.truncate_chunks(fallback_results, legacy.MAX_CONTEXT_CHARS)
        abstention = {"abstain": False, "reason": "support_fallback"}
    else:
        final_results = [] if abstention["abstain"] else legacy.truncate_chunks(
            legacy._filter_results_by_allowed_establishments(final_ranked, normalized_allowed),
            legacy.MAX_CONTEXT_CHARS,
        )

    return {
        "query": query,
        "dense_results": dense_results,
        "bm25_results": sparse_results,
        "fusion_results": fusion_results,
        "merged_results": enriched_merged,
        "boosted_results": boosted,
        "guarded_results": guarded,
        "final_results": final_results,
        "abstain": abstention["abstain"],
        "abstain_reason": abstention["reason"],
        "query_profile": guardrail_diagnostics.get("query_profile", {}),
        "guardrail_diagnostics": guardrail_diagnostics,
    }


def get_relevant_chunks_qdrant(
    raw_query: str,
    top_k: int = 5,
    allowed_establishments: Optional[List[str]] = None,
) -> List[Dict]:
    debug_payload = run_qdrant_search_debug(
        raw_query,
        top_k=top_k,
        allowed_establishments=allowed_establishments,
    )
    return list(debug_payload.get("final_results", []))
