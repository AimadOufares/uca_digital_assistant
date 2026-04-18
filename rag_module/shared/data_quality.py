import os
from typing import Dict, List

from ..offline.backup_utils import create_backup
from .metadata_policy import detect_document_type, prepare_chunk_metadata
from .relevance_policy import compute_chunk_relevance, compute_source_relevance, should_keep_chunk


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_CHUNKS_PER_SOURCE = _env_int("RAG_MAX_CHUNKS_PER_SOURCE", 120)
DOMAIN_FILTER_ENABLED = _env_bool("RAG_DOMAIN_FILTER_ENABLED", True)


def _downsample_evenly(items: List[Dict], max_items: int) -> List[Dict]:
    if len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]

    total = len(items)
    indices = [int(i * total / max_items) for i in range(max_items)]
    return [items[index] for index in indices]


def postprocess_chunks_for_source(chunks: List[Dict], source_path: str) -> List[Dict]:
    prepared_chunks: List[Dict] = []
    joined_text_parts: List[str] = []

    for chunk in chunks:
        updated_chunk = prepare_chunk_metadata(chunk, source_path)
        if updated_chunk is None:
            continue
        prepared_chunks.append(updated_chunk)
        joined_text_parts.append((updated_chunk.get("text", "") or "").strip())

    if not prepared_chunks:
        return []

    source_document_type = detect_document_type(source_path, "\n".join(joined_text_parts[:8]))
    source_relevance_score, source_relevance_hits = compute_source_relevance(
        source_path,
        "\n".join(joined_text_parts),
        source_document_type,
    )

    cleaned: List[Dict] = []
    for chunk in prepared_chunks:
        text = (chunk.get("text", "") or "").strip()
        metadata = dict(chunk.get("metadata", {}) or {})
        document_type = str(metadata.get("document_type") or "general")
        chunk_relevance_score, chunk_relevance_hits = compute_chunk_relevance(text, document_type)

        metadata["source_relevance_score"] = source_relevance_score
        metadata["source_relevance_hits"] = source_relevance_hits
        metadata["chunk_relevance_score"] = chunk_relevance_score
        metadata["chunk_relevance_hits"] = chunk_relevance_hits

        if DOMAIN_FILTER_ENABLED and not should_keep_chunk(chunk_relevance_score):
            continue

        updated_chunk = dict(chunk)
        updated_chunk["metadata"] = metadata
        cleaned.append(updated_chunk)

    limited = _downsample_evenly(cleaned, MAX_CHUNKS_PER_SOURCE)
    total = len(limited)
    for index, chunk in enumerate(limited):
        metadata = dict(chunk.get("metadata", {}) or {})
        metadata["index"] = index
        metadata["total_chunks"] = total
        chunk["metadata"] = metadata

    return limited
