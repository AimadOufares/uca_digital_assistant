from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..adapters.storage import DocumentStorage
from ..services.health import build_ready_health
from ..shared.runtime import get_runtime_settings


REPORT_PREFIXES = {
    "data_audit": "data_audit",
    "raw_quality_audit": "raw_quality_audit",
    "rag_eval": "rag_eval",
}


def _latest_report_file(storage: DocumentStorage, prefix: str) -> Optional[Path]:
    if not storage.report_dir.exists():
        return None
    files = sorted(
        storage.report_dir.glob(f"{prefix}_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _latest_report_bundle(storage: DocumentStorage, prefix: str) -> Dict[str, object]:
    report_file = _latest_report_file(storage, prefix)
    if not report_file:
        return {"prefix": prefix, "available": False}
    payload = storage.load_json(report_file)
    return {
        "prefix": prefix,
        "available": True,
        "file_name": report_file.name,
        "file_path": str(report_file),
        "updated_at": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat(),
        "report": payload,
    }


def _active_index_summary(storage: DocumentStorage) -> Dict[str, object]:
    pointer = storage.load_active_index_pointer()
    paths = storage.resolve_active_faiss_paths()
    manifest = storage.load_json(paths["manifest_file"])
    manifest_mtime = paths["manifest_file"].stat().st_mtime if paths["manifest_file"].exists() else 0.0
    return {
        "backend": str(pointer.get("backend") or manifest.get("backend") or ""),
        "build_id": str(pointer.get("build_id") or manifest.get("build_id") or ""),
        "published": bool(pointer.get("published", False)),
        "chunk_count": int(manifest.get("chunk_count", 0) or 0),
        "document_count": int(manifest.get("document_count", 0) or 0),
        "source_count": int(manifest.get("document_count", 0) or 0),
        "embedding_model": str(manifest.get("model_name") or ""),
        "requested_embedding_model": str(manifest.get("requested_model_name") or ""),
        "manifest_updated_at": datetime.fromtimestamp(manifest_mtime).isoformat() if manifest_mtime else "",
        "manifest_path": str(paths["manifest_file"]),
    }


def build_drive_sync_status(storage: DocumentStorage | None = None) -> Dict[str, object]:
    storage = storage or DocumentStorage()
    settings = get_runtime_settings()
    drive_dir = settings.rag_raw_drive_dir
    files = [path for path in drive_dir.iterdir() if path.is_file()] if drive_dir.exists() else []
    latest_doc_mtime = max((path.stat().st_mtime for path in files), default=0.0)
    latest_doc_name = ""
    if latest_doc_mtime:
        latest_doc_name = max(files, key=lambda item: item.stat().st_mtime).name

    active_paths = storage.resolve_active_faiss_paths()
    manifest_mtime = active_paths["manifest_file"].stat().st_mtime if active_paths["manifest_file"].exists() else 0.0
    pointer = storage.load_active_index_pointer()
    has_active_index = bool(pointer) and active_paths["manifest_file"].exists()

    if not files:
        status_value = "up_to_date" if has_active_index else "empty"
    elif not has_active_index or latest_doc_mtime > manifest_mtime:
        status_value = "rebuild_required"
    else:
        status_value = "up_to_date"

    return {
        "status": status_value,
        "document_count": len(files),
        "latest_document_name": latest_doc_name,
        "latest_document_updated_at": datetime.fromtimestamp(latest_doc_mtime).isoformat() if latest_doc_mtime else "",
        "active_index_updated_at": datetime.fromtimestamp(manifest_mtime).isoformat() if manifest_mtime else "",
        "has_active_index": has_active_index,
    }


def load_latest_reports() -> Dict[str, Dict]:
    storage = DocumentStorage()
    payload: Dict[str, Dict] = {}
    for key, prefix in REPORT_PREFIXES.items():
        selected_prefix = "rag_eval_drive" if key == "rag_eval" else prefix
        payload[key] = storage.latest_report(selected_prefix) or storage.latest_report(prefix) or {}
    return payload


def build_dashboard_payload() -> Dict[str, object]:
    storage = DocumentStorage()
    reports = load_latest_reports()
    ready = build_ready_health()
    settings = get_runtime_settings()
    latest_reports = {
        "data_audit": _latest_report_bundle(storage, "data_audit"),
        "raw_quality_audit": _latest_report_bundle(storage, "raw_quality_audit"),
        "rag_eval": _latest_report_bundle(storage, "rag_eval_drive")
        if _latest_report_file(storage, "rag_eval_drive")
        else _latest_report_bundle(storage, "rag_eval"),
    }
    return {
        **reports,
        "system_status": ready,
        "active_index": _active_index_summary(storage),
        "drive_sync_status": build_drive_sync_status(storage),
        "llm_config": {
            "provider": settings.rag_llm_provider,
            "chat_model": settings.rag_chat_model,
            "lm_studio_base_url": settings.lm_studio_base_url,
        },
        "latest_reports": latest_reports,
    }


def latest_report_payload(kind: str) -> Dict[str, object]:
    storage = DocumentStorage()
    normalized = str(kind or "").strip().lower()
    if normalized not in REPORT_PREFIXES:
        raise ValueError("Type de rapport non supporte.")
    selected_prefix = "rag_eval_drive" if normalized == "rag_eval" else REPORT_PREFIXES[normalized]
    bundle = _latest_report_bundle(storage, selected_prefix)
    if not bundle.get("available") and normalized == "rag_eval":
        bundle = _latest_report_bundle(storage, "rag_eval")
    return bundle
