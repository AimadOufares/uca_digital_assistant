import json
import os
from pathlib import Path
from typing import Dict, Set, Tuple

try:
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.runtime import get_runtime_settings


RUNTIME = get_runtime_settings()


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
        return {"version": 2, "files": {}}
    try:
        with open(cache_file, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {"version": 2, "files": {}}

    if isinstance(raw, dict) and "files" in raw and isinstance(raw["files"], dict):
        files = {}
        for path, entry in raw["files"].items():
            if not isinstance(entry, dict):
                continue
            files[path] = {
                "file_hash": entry.get("file_hash", ""),
                "chunk_hashes": list(dict.fromkeys(entry.get("chunk_hashes", []))),
                "policy_version": entry.get("policy_version", ""),
            }
        return {"version": 2, "files": files}

    if isinstance(raw, dict):
        files = {}
        for path, file_hash in raw.items():
            if isinstance(path, str) and isinstance(file_hash, str):
                files[path] = {"file_hash": file_hash, "chunk_hashes": [], "policy_version": ""}
        return {"version": 2, "files": files}
    return {"version": 2, "files": {}}


def save_cache(cache: Dict, cache_file: str) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


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
