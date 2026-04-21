import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data_storage" / "index"
RAW_DIR = PROJECT_ROOT / "data_storage" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data_storage" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data_storage" / "reports"
CACHE_DIR = PROJECT_ROOT / "data_storage" / "cache"


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


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def configured_vector_backend() -> str:
    return "qdrant"


def qdrant_collection_name() -> str:
    return env_str("RAG_QDRANT_COLLECTION", "uca_chunks").strip() or "uca_chunks"


def qdrant_collection_prefix() -> str:
    return env_str("RAG_QDRANT_COLLECTION_PREFIX", qdrant_collection_name()).strip() or qdrant_collection_name()


def qdrant_active_alias_name() -> str:
    default_name = f"{qdrant_collection_prefix()}_current"
    return env_str("RAG_QDRANT_ACTIVE_ALIAS", default_name).strip() or default_name


def qdrant_previous_alias_name() -> str:
    default_name = f"{qdrant_collection_prefix()}_previous"
    return env_str("RAG_QDRANT_PREVIOUS_ALIAS", default_name).strip() or default_name


def qdrant_keep_previous_index() -> bool:
    return env_bool("RAG_KEEP_PREVIOUS_INDEX", True)


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


def qdrant_candidates_dir() -> Path:
    return Path(env_str("RAG_INDEX_CANDIDATES_DIR", str(INDEX_DIR / "candidates"))).resolve()


def qdrant_candidate_manifests_dir() -> Path:
    return qdrant_candidates_dir() / "manifests"


def qdrant_candidate_snapshots_dir() -> Path:
    return qdrant_candidates_dir() / "snapshots"


def qdrant_candidate_sparse_dir() -> Path:
    return qdrant_candidates_dir() / "sparse"


def qdrant_active_manifest_path() -> Path:
    return INDEX_DIR / "index_manifest.json"


def qdrant_active_sparse_encoder_path() -> Path:
    return INDEX_DIR / "qdrant_sparse_encoder.json"


def qdrant_active_chunks_snapshot_path() -> Path:
    return INDEX_DIR / "chunks.json"


def offline_reports_dir() -> Path:
    return REPORTS_DIR


def offline_session_path() -> Path:
    return CACHE_DIR / "offline_pipeline_session.json"


def offline_lock_dir() -> Path:
    return PROJECT_ROOT / "data_storage" / "locks"


def offline_log_dir() -> Path:
    return PROJECT_ROOT / "data_storage" / "logs"


def raw_retention_days() -> int:
    return env_int("RAG_RAW_RETENTION_DAYS", 21)


def index_retention_count() -> int:
    return env_int("RAG_INDEX_RETENTION_COUNT", 2)


def minimum_indexed_chunks() -> int:
    return env_int("RAG_MIN_INDEXED_CHUNKS", 50)


def validation_precision_gate() -> float:
    return env_float("RAG_KPI_PRECISION_GATE", 0.75)


def validation_hit_gate() -> float:
    return env_float("RAG_KPI_HIT_GATE", 0.90)


def run_validation_eval() -> bool:
    return env_bool("RAG_VALIDATE_RETRIEVAL_EVAL", True)


def validation_top_k() -> int:
    return env_int("RAG_VALIDATION_TOP_K", 5)
