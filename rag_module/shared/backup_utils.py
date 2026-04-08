import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


BACKUP_ENABLED = _env_bool("RAG_CREATE_BACKUP", True)


def create_backup(processed_path: str, cache_file: str, backup_root: str = "data_storage/backups") -> Optional[str]:
    if not BACKUP_ENABLED:
        return None

    processed_dir = Path(processed_path)
    if not processed_dir.exists():
        return None

    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(backup_root) / f"processed_{timestamp}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(processed_dir, target_dir)

    cache_path = Path(cache_file)
    if cache_path.exists():
        shutil.copy2(cache_path, target_dir / cache_path.name)

    return str(target_dir)
