import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BACKUP_ENABLED = _env_bool("RAG_CREATE_BACKUP", True)
BACKUP_MODE = os.getenv("RAG_BACKUP_MODE", "always").strip().lower() or "always"
BACKUP_CHANGE_THRESHOLD = _env_float("RAG_BACKUP_CHANGE_THRESHOLD", 0.15)
BACKUP_MIN_DAYS = _env_int("RAG_BACKUP_MIN_DAYS", 7)
BACKUP_META_FILE = "backup_meta.json"


def _load_last_backup_time(backup_root: Path) -> Optional[datetime]:
    meta_path = backup_root / BACKUP_META_FILE
    if not meta_path.exists():
        return None
    try:
        payload = meta_path.read_text(encoding="utf-8").strip()
        if not payload:
            return None
        return datetime.fromisoformat(payload)
    except Exception:
        return None


def _save_last_backup_time(backup_root: Path, when: datetime) -> None:
    meta_path = backup_root / BACKUP_META_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(when.isoformat(), encoding="utf-8")


def _should_backup(backup_root: Path, change_ratio: Optional[float]) -> bool:
    if BACKUP_MODE == "never":
        return False
    if BACKUP_MODE == "always":
        return True
    if BACKUP_MODE != "weekly_or_high_change":
        return True

    if change_ratio is not None and float(change_ratio) >= float(BACKUP_CHANGE_THRESHOLD):
        return True

    last_backup = _load_last_backup_time(backup_root)
    if last_backup is None:
        return True
    return datetime.now() - last_backup >= timedelta(days=max(1, BACKUP_MIN_DAYS))


def create_backup(
    processed_path: str,
    cache_file: str,
    backup_root: str = "data_storage/backups",
    change_ratio: Optional[float] = None,
) -> Optional[str]:
    if not BACKUP_ENABLED:
        return None

    processed_dir = Path(processed_path)
    if not processed_dir.exists():
        return None

    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        return None

    backup_root_path = Path(backup_root)
    if not _should_backup(backup_root_path, change_ratio=change_ratio):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = backup_root_path / f"processed_{timestamp}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(processed_dir, target_dir)

    cache_path = Path(cache_file)
    if cache_path.exists():
        shutil.copy2(cache_path, target_dir / cache_path.name)

    _save_last_backup_time(backup_root_path, datetime.now())
    return str(target_dir)
