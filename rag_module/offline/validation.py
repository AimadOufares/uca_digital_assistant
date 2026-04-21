import logging
from pathlib import Path
from typing import Dict

from ..audit.offline_pipeline_report import update_offline_pipeline_report
from ..evaluation.evaluate_rag import evaluate, evaluate_kpi_gates
from ..offline.qdrant_indexing import get_qdrant_alias_map, get_qdrant_client
from ..retrieval.qdrant_search import qdrant_index_ready
from ..shared.index_manifest import load_manifest, validate_manifest
from ..shared.runtime_config import minimum_indexed_chunks, validation_hit_gate, validation_precision_gate, validation_top_k

logger = logging.getLogger(__name__)


def _load_candidate_manifest(candidate_manifest: Dict | None = None, manifest_path: str | Path | None = None) -> Dict:
    if candidate_manifest:
        return dict(candidate_manifest)
    if manifest_path is None:
        return {}
    return load_manifest(str(manifest_path))


def validate_candidate_index(
    candidate_manifest: Dict | None = None,
    manifest_path: str | Path | None = None,
    run_eval: bool = True,
) -> Dict:
    manifest = _load_candidate_manifest(candidate_manifest=candidate_manifest, manifest_path=manifest_path)
    if not manifest:
        raise RuntimeError("Manifest candidat introuvable pour la validation.")

    validate_manifest(
        manifest,
        expected_model=str(manifest.get("model_name") or ""),
        expected_vector_store="qdrant",
    )

    collection_name = str(manifest.get("collection_name") or "").strip()
    sparse_encoder_path = Path(str(manifest.get("sparse_encoder_path") or "")).resolve()
    chunk_count = int(manifest.get("chunk_count") or 0)
    min_chunks = minimum_indexed_chunks()

    client = get_qdrant_client()
    alias_map = get_qdrant_alias_map(client)
    collections = client.get_collections()
    collection_names = {getattr(item, "name", "") for item in getattr(collections, "collections", []) or []}

    checks = {
        "manifest_valid": True,
        "collection_accessible": collection_name in collection_names,
        "sparse_encoder_present": sparse_encoder_path.exists(),
        "minimum_chunks": chunk_count >= min_chunks,
        "retrieval_ready": qdrant_index_ready(manifest_override=manifest),
        "kpi_gates": None,
    }
    failures = [name for name, passed in checks.items() if isinstance(passed, bool) and not passed]

    if run_eval and not failures:
        eval_report = evaluate(top_k=validation_top_k(), run_generation=False, manifest_override=manifest)
        kpi_gates = evaluate_kpi_gates(
            eval_report,
            precision_gate=validation_precision_gate(),
            hit_gate=validation_hit_gate(),
        )
        checks["kpi_gates"] = kpi_gates
        if not kpi_gates.get("passed", False):
            failures.append("kpi_gates")
    else:
        eval_report = {}

    validation_payload = {
        "candidate_collection_name": collection_name,
        "chunk_count": chunk_count,
        "minimum_required_chunks": min_chunks,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "alias_snapshot": alias_map,
    }
    if eval_report:
        validation_payload["evaluation_summary"] = eval_report.get("summary", {})

    update_offline_pipeline_report("validation", validation_payload)
    logger.info(
        "Validation index candidat | collection=%s | passed=%s | failures=%s",
        collection_name,
        not failures,
        ",".join(failures) if failures else "-",
    )
    return validation_payload
