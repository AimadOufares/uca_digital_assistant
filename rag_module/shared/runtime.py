import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

from .env_loader import load_env_file


DEFAULT_SECRET_KEY = "django-insecure-local-dev-key"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: List[str] | None = None) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_path(raw_value: str, project_root: Path, default: Path) -> Path:
    value = (raw_value or "").strip()
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    app_env: str
    django_debug: bool
    secret_key: str
    allowed_hosts: List[str]
    database_url: str
    rag_data_root: Path
    rag_raw_dir: Path
    rag_processed_dir: Path
    rag_index_dir: Path
    rag_cache_dir: Path
    rag_reports_dir: Path
    rag_backups_dir: Path
    rag_locks_dir: Path
    rag_quarantine_dir: Path
    rag_vector_backend: str
    rag_qdrant_url: str
    rag_qdrant_api_key: str
    rag_qdrant_collection_prefix: str
    rag_active_index_name: str
    rag_llm_provider: str
    lm_studio_base_url: str
    lm_studio_api_key: str
    rag_chat_model: str
    openai_api_key: str

    def ensure_directories(self) -> None:
        for path in (
            self.rag_data_root,
            self.rag_raw_dir,
            self.rag_processed_dir,
            self.rag_index_dir,
            self.rag_cache_dir,
            self.rag_reports_dir,
            self.rag_backups_dir,
            self.rag_locks_dir,
            self.rag_quarantine_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    project_root = _project_root()
    load_env_file(project_root / ".env")

    rag_data_root = _resolve_path(
        os.getenv("RAG_DATA_ROOT", ""),
        project_root,
        project_root / "data_storage",
    )
    rag_raw_dir = _resolve_path(
        os.getenv("RAG_RAW_DIR", ""),
        project_root,
        rag_data_root / "raw",
    )
    rag_processed_dir = _resolve_path(
        os.getenv("RAG_PROCESSED_DIR", ""),
        project_root,
        rag_data_root / "processed",
    )
    rag_index_dir = _resolve_path(
        os.getenv("RAG_INDEX_DIR", ""),
        project_root,
        rag_data_root / "index",
    )
    rag_cache_dir = _resolve_path(
        os.getenv("RAG_CACHE_DIR", ""),
        project_root,
        rag_data_root / "cache",
    )
    rag_reports_dir = _resolve_path(
        os.getenv("RAG_REPORTS_DIR", ""),
        project_root,
        rag_data_root / "reports",
    )
    rag_backups_dir = _resolve_path(
        os.getenv("RAG_BACKUPS_DIR", ""),
        project_root,
        rag_data_root / "backups",
    )
    rag_locks_dir = _resolve_path(
        os.getenv("RAG_LOCKS_DIR", ""),
        project_root,
        rag_data_root / "locks",
    )
    rag_quarantine_dir = _resolve_path(
        os.getenv("RAG_QUARANTINE_DIR", ""),
        project_root,
        rag_data_root / "quarantine",
    )

    settings = RuntimeSettings(
        project_root=project_root,
        app_env=_env_str("APP_ENV", "local"),
        django_debug=_env_bool("DJANGO_DEBUG", True),
        secret_key=_env_str("SECRET_KEY", DEFAULT_SECRET_KEY),
        allowed_hosts=_env_csv("ALLOWED_HOSTS", ["127.0.0.1", "localhost"]),
        database_url=_env_str("DATABASE_URL", "sqlite:///db.sqlite3"),
        rag_data_root=rag_data_root,
        rag_raw_dir=rag_raw_dir,
        rag_processed_dir=rag_processed_dir,
        rag_index_dir=rag_index_dir,
        rag_cache_dir=rag_cache_dir,
        rag_reports_dir=rag_reports_dir,
        rag_backups_dir=rag_backups_dir,
        rag_locks_dir=rag_locks_dir,
        rag_quarantine_dir=rag_quarantine_dir,
        rag_vector_backend=_env_str("RAG_VECTOR_BACKEND", "faiss").lower(),
        rag_qdrant_url=_env_str("RAG_QDRANT_URL", ""),
        rag_qdrant_api_key=_env_str("RAG_QDRANT_API_KEY", ""),
        rag_qdrant_collection_prefix=_env_str("RAG_QDRANT_COLLECTION_PREFIX", "uca_kb"),
        rag_active_index_name=_env_str("RAG_ACTIVE_INDEX_NAME", "uca_kb_active"),
        rag_llm_provider=_env_str("RAG_LLM_PROVIDER", "lmstudio").lower(),
        lm_studio_base_url=_env_str("LM_STUDIO_BASE_URL", ""),
        lm_studio_api_key=_env_str("LM_STUDIO_API_KEY", "lm-studio"),
        rag_chat_model=_env_str("RAG_CHAT_MODEL", "gpt-4o-mini"),
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
    )
    settings.ensure_directories()
    return settings


def parse_database_url(database_url: str | None = None) -> Dict[str, Dict[str, Any]]:
    settings = get_runtime_settings()
    raw = (database_url or settings.database_url or "").strip()
    if not raw:
        raw = "sqlite:///db.sqlite3"

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()

    if scheme in {"sqlite", ""}:
        if raw == "sqlite:///:memory:":
            name = ":memory:"
        else:
            path_value = unquote(parsed.path or "").lstrip("/")
            if not path_value:
                path_value = "db.sqlite3"
            db_path = Path(path_value)
            if not db_path.is_absolute():
                db_path = settings.project_root / db_path
            name = str(db_path)
        return {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": name,
            }
        }

    if scheme in {"postgres", "postgresql", "pgsql"}:
        return {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": unquote(parsed.path.lstrip("/")),
                "USER": unquote(parsed.username or ""),
                "PASSWORD": unquote(parsed.password or ""),
                "HOST": parsed.hostname or "",
                "PORT": str(parsed.port or ""),
            }
        }

    raise ValueError(f"Unsupported DATABASE_URL scheme: {scheme or 'unknown'}")
