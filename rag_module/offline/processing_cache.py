import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Set, Tuple

try:
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.runtime import get_runtime_settings


RUNTIME = get_runtime_settings()
PROCESSING_CACHE_VERSION = 3


def corpus_paths(corpus: str) -> Tuple[str, str, str]:
    if corpus == "archive":
        return (
            str(RUNTIME.rag_raw_archive_dir),
            str(RUNTIME.rag_processed_archive_dir),
            str(RUNTIME.rag_cache_dir / "file_cache_archive.json"),
        )
    if corpus == "drive":
        return (
            str(RUNTIME.rag_raw_drive_dir),
            str(RUNTIME.rag_processed_drive_dir),
            str(RUNTIME.rag_cache_dir / "file_cache_drive.json"),
        )
    return (
        str(RUNTIME.rag_raw_main_dir),
        str(RUNTIME.rag_processed_main_dir),
        str(RUNTIME.rag_cache_dir / "file_cache_main.json"),
    )


def load_raw_metadata(corpus: str) -> Dict[str, Dict]:
    raw_dir, _, _ = corpus_paths(corpus)
    metadata_path = Path(raw_dir) / ".metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_cache(cache_file: str) -> Dict:
    if not os.path.exists(cache_file):
        return {"version": PROCESSING_CACHE_VERSION, "files": {}}
    try:
        with open(cache_file, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {"version": PROCESSING_CACHE_VERSION, "files": {}}

    if isinstance(raw, dict) and "files" in raw and isinstance(raw["files"], dict):
        files = {}
        for path, entry in raw["files"].items():
            if not isinstance(entry, dict):
                continue
            files[path] = normalize_file_record(
                {
                "file_hash": entry.get("file_hash", ""),
                "chunk_hashes": list(dict.fromkeys(entry.get("chunk_hashes", []))),
                "policy_version": entry.get("policy_version", ""),
                    "status": entry.get("status", ""),
                    "error": entry.get("error", ""),
                    "processed_at": entry.get("processed_at", ""),
                    "chunks_count": entry.get("chunks_count", 0),
                    "failure_count": entry.get("failure_count", 0),
                    "last_seen_at": entry.get("last_seen_at", ""),
                    "corpus": entry.get("corpus", ""),
                }
            )
        return {"version": PROCESSING_CACHE_VERSION, "files": files}

    if isinstance(raw, dict):
        files = {}
        for path, file_hash in raw.items():
            if isinstance(path, str) and isinstance(file_hash, str):
                files[path] = normalize_file_record({"file_hash": file_hash, "chunk_hashes": [], "policy_version": ""})
        return {"version": PROCESSING_CACHE_VERSION, "files": files}
    return {"version": PROCESSING_CACHE_VERSION, "files": {}}


def save_cache(cache: Dict, cache_file: str) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache["version"] = PROCESSING_CACHE_VERSION
    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_file_record(entry: Dict) -> Dict:
    status = str(entry.get("status") or "").strip() or ("processed" if entry.get("file_hash") else "new")
    chunk_hashes = list(dict.fromkeys(entry.get("chunk_hashes", []) or []))
    return {
        "file_hash": str(entry.get("file_hash") or ""),
        "chunk_hashes": chunk_hashes,
        "policy_version": str(entry.get("policy_version") or ""),
        "status": status,
        "error": str(entry.get("error") or ""),
        "processed_at": str(entry.get("processed_at") or ""),
        "chunks_count": int(entry.get("chunks_count") or len(chunk_hashes) or 0),
        "failure_count": int(entry.get("failure_count") or 0),
        "last_seen_at": str(entry.get("last_seen_at") or ""),
        "corpus": str(entry.get("corpus") or ""),
    }


def classify_document_state(
    file_path: str,
    file_hash: str,
    record: Dict | None,
    *,
    has_all_chunks: bool,
    policy_version: str,
    max_failures: int = 3,
) -> str:
    if not record:
        return "new"
    normalized = normalize_file_record(record)
    if normalized.get("status") == "quarantine" and normalized.get("failure_count", 0) >= max_failures:
        return "quarantine"
    if normalized.get("file_hash") != file_hash:
        return "modified"
    if normalized.get("policy_version") != policy_version:
        return "modified"
    if normalized.get("status") == "skipped":
        return "skipped"
    if normalized.get("status") == "processed" and has_all_chunks:
        return "processed"
    if normalized.get("status") == "failed":
        return "retry"
    return "modified"


def mark_processed(file_hash: str, chunk_hashes: list[str], policy_version: str, corpus: str) -> Dict:
    return normalize_file_record(
        {
            "file_hash": file_hash,
            "chunk_hashes": list(dict.fromkeys(chunk_hashes)),
            "policy_version": policy_version,
            "status": "processed",
            "error": "",
            "processed_at": utc_now_iso(),
            "chunks_count": len(chunk_hashes),
            "failure_count": 0,
            "last_seen_at": utc_now_iso(),
            "corpus": corpus,
        }
    )


def mark_failed(
    previous_record: Dict | None,
    file_hash: str,
    policy_version: str,
    corpus: str,
    error: str,
    *,
    max_failures: int = 3,
) -> Dict:
    previous = normalize_file_record(previous_record or {})
    failure_count = int(previous.get("failure_count") or 0) + 1
    status = "quarantine" if failure_count >= max_failures else "failed"
    return normalize_file_record(
        {
            "file_hash": file_hash,
            "chunk_hashes": previous.get("chunk_hashes", []),
            "policy_version": policy_version,
            "status": status,
            "error": error,
            "processed_at": previous.get("processed_at", ""),
            "chunks_count": previous.get("chunks_count", 0),
            "failure_count": failure_count,
            "last_seen_at": utc_now_iso(),
            "corpus": corpus,
        }
    )


def mark_skipped(previous_record: Dict, file_hash: str, policy_version: str, corpus: str) -> Dict:
    record = normalize_file_record(previous_record)
    record["file_hash"] = file_hash
    record["policy_version"] = policy_version
    record["status"] = "processed"
    record["last_seen_at"] = utc_now_iso()
    record["corpus"] = corpus
    return record


def mark_no_chunks(file_hash: str, policy_version: str, corpus: str, reason: str = "Aucun chunk genere.") -> Dict:
    return normalize_file_record(
        {
            "file_hash": file_hash,
            "chunk_hashes": [],
            "policy_version": policy_version,
            "status": "skipped",
            "error": reason,
            "processed_at": utc_now_iso(),
            "chunks_count": 0,
            "failure_count": 0,
            "last_seen_at": utc_now_iso(),
            "corpus": corpus,
        }
    )


def chunk_refcounts(file_records: Dict[str, Dict]) -> Dict[str, int]:
    refcounts: Dict[str, int] = {}
    for record in file_records.values():
        for chunk_hash in set(record.get("chunk_hashes", [])):
            refcounts[chunk_hash] = refcounts.get(chunk_hash, 0) + 1
    return refcounts


def delete_chunk_file_if_unreferenced(
    chunk_hash: str,
    refcounts: Dict[str, int],
    seen_chunks: Set[str],
    processed_path: str,
) -> bool:
    if refcounts.get(chunk_hash, 0) > 0:
        return False
    path = os.path.join(processed_path, f"{chunk_hash}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            return False
    seen_chunks.discard(chunk_hash)
    return True
