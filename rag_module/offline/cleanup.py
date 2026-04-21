import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set

from ..audit.offline_pipeline_report import load_latest_offline_pipeline_report, update_offline_pipeline_report
from ..offline.processing import load_cache
from ..offline.qdrant_indexing import (
    candidate_chunks_snapshot_path,
    candidate_manifest_path,
    candidate_sparse_encoder_path,
    delete_candidate_artifacts,
    delete_qdrant_collection,
    get_qdrant_alias_map,
    get_qdrant_client,
    list_candidate_manifest_paths,
)
from ..shared.index_manifest import load_manifest
from ..shared.runtime_config import (
    PROCESSED_DIR,
    RAW_DIR,
    index_retention_count,
    qdrant_collection_prefix,
    raw_retention_days,
)

logger = logging.getLogger(__name__)


def _protected_collection_names() -> Set[str]:
    client = get_qdrant_client()
    alias_map = get_qdrant_alias_map(client)
    protected = {name for name in alias_map.values() if name}
    latest_report = load_latest_offline_pipeline_report()
    summary = latest_report.get("summary", {}) if isinstance(latest_report, dict) else {}
    for key in ("published_collection_name", "previous_collection_name", "active_collection_name"):
        value = str(summary.get(key) or "").strip()
        if value:
            protected.add(value)
    return protected


def _cleanup_raw_files(dry_run: bool) -> Dict:
    retention_days = raw_retention_days()
    if retention_days <= 0:
        return {"retention_days": retention_days, "removed_files": []}

    threshold = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed_files: List[str] = []
    for path in RAW_DIR.glob("*"):
        if not path.is_file():
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at > threshold:
            continue
        if not dry_run:
            path.unlink()
        removed_files.append(str(path))
    return {"retention_days": retention_days, "removed_files": removed_files}


def _cleanup_processed_orphans(dry_run: bool) -> Dict:
    cache = load_cache()
    file_records = cache.get("files", {}) if isinstance(cache, dict) else {}
    referenced = {
        chunk_hash
        for record in file_records.values()
        if isinstance(record, dict)
        for chunk_hash in record.get("chunk_hashes", [])
        if isinstance(chunk_hash, str) and chunk_hash.strip()
    }
    removed_files: List[str] = []
    for path in PROCESSED_DIR.glob("*.json"):
        if path.stem in referenced:
            continue
        if not dry_run:
            path.unlink()
        removed_files.append(str(path))
    return {"removed_files": removed_files, "referenced_chunk_count": len(referenced)}


def _cleanup_old_indexes(dry_run: bool) -> Dict:
    retention = max(1, index_retention_count())
    protected = _protected_collection_names()
    prefix = qdrant_collection_prefix()
    removed_collections: List[str] = []
    removed_artifacts: List[str] = []

    manifests = list_candidate_manifest_paths()
    kept_collections: Set[str] = set()
    for path in manifests:
        manifest = load_manifest(str(path))
        collection_name = str(manifest.get("collection_name") or "").strip()
        if not collection_name:
            continue
        if collection_name in protected:
            kept_collections.add(collection_name)
            continue
        if len(kept_collections) < retention:
            kept_collections.add(collection_name)

    for path in manifests:
        manifest = load_manifest(str(path))
        collection_name = str(manifest.get("collection_name") or "").strip()
        if not collection_name:
            continue
        if collection_name in kept_collections or collection_name in protected:
            continue
        if not collection_name.startswith(prefix):
            continue
        if not dry_run:
            if delete_qdrant_collection(collection_name):
                removed_collections.append(collection_name)
            removed_artifacts.extend(delete_candidate_artifacts(collection_name))
        else:
            removed_collections.append(collection_name)
            removed_artifacts.extend(
                [
                    str(candidate_manifest_path(collection_name)),
                    str(candidate_sparse_encoder_path(collection_name)),
                    str(candidate_chunks_snapshot_path(collection_name)),
                ]
            )

    return {
        "retention_count": retention,
        "protected_collections": sorted(protected),
        "kept_candidate_collections": sorted(kept_collections),
        "removed_collections": removed_collections,
        "removed_artifacts": removed_artifacts,
    }


def cleanup_offline_artifacts(dry_run: bool = False, require_last_success: bool = True) -> Dict:
    latest_report = load_latest_offline_pipeline_report()
    if require_last_success and latest_report and str(latest_report.get("status") or "").lower() != "success":
        payload = {
            "skipped": True,
            "reason": "last_offline_run_not_successful",
            "dry_run": bool(dry_run),
        }
        update_offline_pipeline_report("cleanup", payload)
        return payload

    raw_cleanup = _cleanup_raw_files(dry_run=dry_run)
    processed_cleanup = _cleanup_processed_orphans(dry_run=dry_run)
    index_cleanup = _cleanup_old_indexes(dry_run=dry_run)
    payload = {
        "skipped": False,
        "dry_run": bool(dry_run),
        "raw": raw_cleanup,
        "processed": processed_cleanup,
        "indexes": index_cleanup,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    update_offline_pipeline_report("cleanup", payload)
    logger.info(
        "Cleanup offline termine | raw=%s | processed=%s | indexes=%s",
        len(raw_cleanup.get("removed_files", [])),
        len(processed_cleanup.get("removed_files", [])),
        len(index_cleanup.get("removed_collections", [])),
    )
    return payload
