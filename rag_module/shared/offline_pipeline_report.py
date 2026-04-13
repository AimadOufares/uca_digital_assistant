import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


SESSION_PATH = Path("data_storage/cache/offline_pipeline_session.json")
REPORTS_DIR = Path("data_storage/reports")
SESSION_TTL_SECONDS = 6 * 3600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_session_fresh(started_at: str) -> bool:
    try:
        started = datetime.fromisoformat(started_at)
    except Exception:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - started).total_seconds()
    return age <= SESSION_TTL_SECONDS


def _resolve_report_path() -> Path:
    session = _load_json(SESSION_PATH)
    path_value = str(session.get("report_path") or "").strip()
    started_at = str(session.get("started_at") or "").strip()

    if path_value and started_at and _is_session_fresh(started_at):
        path = Path(path_value)
        if path.exists():
            return path

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    new_path = REPORTS_DIR / f"offline_pipeline_report_{timestamp}.json"
    _save_json_atomic(
        SESSION_PATH,
        {
            "report_path": str(new_path),
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    )
    return new_path


def update_offline_pipeline_report(stage: str, payload: Dict) -> Path:
    stage_name = (stage or "").strip().lower() or "unknown"
    report_path = _resolve_report_path()
    report = _load_json(report_path)
    now_iso = _now_iso()

    if not report:
        report = {
            "report_type": "offline_pipeline",
            "created_at": now_iso,
            "updated_at": now_iso,
            "stages": {},
        }

    stages = report.setdefault("stages", {})
    stages[stage_name] = payload
    report["updated_at"] = now_iso

    _save_json_atomic(report_path, report)
    _save_json_atomic(
        SESSION_PATH,
        {
            "report_path": str(report_path),
            "started_at": str(_load_json(SESSION_PATH).get("started_at") or now_iso),
            "updated_at": now_iso,
        },
    )
    return report_path

