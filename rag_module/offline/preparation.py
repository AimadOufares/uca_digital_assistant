import os
from pathlib import Path
from typing import Dict, List

from ..shared.runtime_config import (
    configured_vector_backend,
    document_parser_name,
    fasttext_model_path,
    html_extractor_name,
    qdrant_local_path,
    qdrant_url,
)


def _check_import(module_name: str) -> str:
    try:
        __import__(module_name)
        return "available"
    except Exception:
        return "missing"


def verify_required_artifacts() -> Dict:
    vector_backend = configured_vector_backend()
    qdrant_target = qdrant_url() or str(qdrant_local_path())
    checks = {
        "html_extractor": {
            "selected": html_extractor_name(),
            "status": _check_import("trafilatura") if html_extractor_name() == "trafilatura" else "not_selected",
        },
        "document_parser": {
            "selected": document_parser_name(),
            "status": _check_import("docling") if document_parser_name() == "docling" else "not_selected",
        },
        "language_detector": {
            "selected": os.getenv("RAG_LANGUAGE_DETECTOR", "fasttext"),
            "status": _check_import("fasttext") if os.getenv("RAG_LANGUAGE_DETECTOR", "fasttext").strip().lower() == "fasttext" else "not_selected",
            "model_path": str(fasttext_model_path()),
            "model_exists": fasttext_model_path().exists(),
        },
        "qdrant": {
            "selected": vector_backend,
            "target": qdrant_target,
            "client_status": _check_import("qdrant_client"),
            "local_path_exists": qdrant_local_path().exists(),
        },
        "embedding_model": {
            "selected": os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3",
        },
    }

    missing: List[str] = []
    if checks["html_extractor"]["status"] == "missing":
        missing.append("trafilatura")
    if checks["document_parser"]["status"] == "missing":
        missing.append("docling")
    if checks["language_detector"]["selected"].strip().lower() == "fasttext" and not checks["language_detector"]["model_exists"]:
        missing.append("fasttext_model")
    if checks["qdrant"]["client_status"] == "missing":
        missing.append("qdrant_client")

    return {
        "ready": not missing,
        "vector_backend": vector_backend,
        "checks": checks,
        "missing": missing,
    }


def verify_indexing_prerequisites() -> None:
    report = verify_required_artifacts()
    if report["missing"]:
        missing = ", ".join(report["missing"])
        raise RuntimeError(
            "Artefacts/dependances manquants pour le pipeline RAG: "
            f"{missing}. Verifie les variables d'environnement et les modeles locaux."
        )
