import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None, override: bool = False) -> Path:
    """Load a simple .env file into os.environ without requiring python-dotenv."""

    project_root = Path(__file__).resolve().parents[2]
    target = Path(env_path) if env_path else (project_root / ".env")
    if not target.exists():
        return target

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return target
