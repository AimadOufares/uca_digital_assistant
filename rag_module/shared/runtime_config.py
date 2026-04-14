import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data_storage" / "index"


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def configured_vector_backend() -> str:
    return env_str("RAG_VECTOR_BACKEND", "qdrant").strip().lower()


def qdrant_collection_name() -> str:
    return env_str("RAG_QDRANT_COLLECTION", "uca_chunks").strip() or "uca_chunks"


def qdrant_local_path() -> Path:
    return Path(env_str("RAG_QDRANT_PATH", str(INDEX_DIR / "qdrant_local"))).resolve()


def qdrant_url() -> str:
    return env_str("RAG_QDRANT_URL", "")


def html_extractor_name() -> str:
    return env_str("RAG_HTML_EXTRACTOR", "trafilatura").strip().lower()


def document_parser_name() -> str:
    return env_str("RAG_DOCUMENT_PARSER", "docling").strip().lower()


def language_detector_name() -> str:
    return env_str("RAG_LANGUAGE_DETECTOR", "fasttext").strip().lower()


def fasttext_model_path() -> Path:
    default_path = PROJECT_ROOT / "data_storage" / "models" / "lid.176.ftz"
    return Path(env_str("RAG_FASTTEXT_MODEL_PATH", str(default_path))).resolve()
