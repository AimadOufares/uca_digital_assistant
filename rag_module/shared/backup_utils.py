import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .runtime import get_runtime_settings

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
    runtime = get_runtime_settings()
    resolved_backup_root = Path(backup_root)
    if backup_root == "data_storage/backups":
        resolved_backup_root = runtime.rag_backups_dir
    base_dir = resolved_backup_root / f"processed_{timestamp}"
    target_dir = base_dir
    suffix = 1
    while target_dir.exists():
        target_dir = resolved_backup_root / f"{base_dir.name}_{suffix}"
        suffix += 1
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(processed_dir, target_dir)

    cache_path = Path(cache_file)
    if cache_path.exists():
        shutil.copy2(cache_path, target_dir / cache_path.name)

    return str(target_dir)
