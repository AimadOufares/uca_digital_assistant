import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..audit.offline_pipeline_report import (
    finalize_offline_pipeline_report,
    load_latest_offline_pipeline_report,
    start_offline_pipeline_report,
    update_offline_pipeline_report,
)
from ..offline.cleanup import cleanup_offline_artifacts
from ..offline.indexing import build_index, load_chunks
from ..offline.ingestion import DEFAULT_SEEDS, crawl
from ..offline.processing import preprocess_all
from ..offline.qdrant_indexing import (
    build_candidate_collection_name,
    latest_candidate_manifest_path,
    publish_qdrant_index,
)
from ..offline.validation import validate_candidate_index
from ..retrieval.rag_search import invalidate_search_cache
from ..shared.index_manifest import load_manifest
from ..shared.runtime_config import run_validation_eval

logger = logging.getLogger(__name__)


@dataclass
class OfflinePipelineOptions:
    seeds: Optional[List[str]] = None
    dry_run: bool = False
    publish: bool = False
    validate_before_publish: bool = True
    run_eval_validation: bool = True
    cleanup_after_publish: bool = True


def _current_stage_payload(stage_name: str) -> Dict:
    report = load_latest_offline_pipeline_report()
    stages = report.get("stages", {}) if isinstance(report, dict) else {}
    stage = stages.get(stage_name, {})
    return stage if isinstance(stage, dict) else {}


def _processing_summary(processing_payload: Dict) -> Dict:
    changes = processing_payload.get("changes", {}) if isinstance(processing_payload, dict) else {}
    metrics = processing_payload.get("metrics", {}) if isinstance(processing_payload, dict) else {}
    return {
        "new_files": list(changes.get("new_files", []) or []),
        "modified_files": list(changes.get("modified_files", []) or []),
        "deleted_files": list(changes.get("deleted_files", []) or []),
        "chunks_added": int(metrics.get("chunks_added", 0) or 0),
        "chunks_removed": int(metrics.get("chunks_removed", 0) or 0),
    }


def run_offline_pipeline(options: OfflinePipelineOptions | None = None) -> Dict:
    opts = options or OfflinePipelineOptions()
    seeds = list(opts.seeds or DEFAULT_SEEDS)
    report_path = start_offline_pipeline_report(
        {
            "mode": "offline_pipeline",
            "dry_run": bool(opts.dry_run),
            "publish": bool(opts.publish),
            "validate_before_publish": bool(opts.validate_before_publish),
            "cleanup_after_publish": bool(opts.cleanup_after_publish),
            "seeds_count": len(seeds),
        }
    )

    status = "success"
    summary: Dict = {
        "dry_run": bool(opts.dry_run),
        "publish_requested": bool(opts.publish),
        "report_path": str(report_path),
        "planned_collection_name": "",
        "candidate_collection_name": "",
        "published_collection_name": "",
        "previous_collection_name": "",
        "active_collection_name": "",
    }

    try:
        crawl(seeds)
        ingestion_payload = _current_stage_payload("ingestion")

        processing_payload = preprocess_all(dry_run=opts.dry_run, with_report=True) or _current_stage_payload("processing")
        summary.update(_processing_summary(processing_payload))

        if opts.dry_run:
            summary["planned_collection_name"] = build_candidate_collection_name()
            summary["ingestion_downloaded_rows_count"] = int(ingestion_payload.get("downloaded_rows_count", 0) or 0)
            finalize_offline_pipeline_report("success", summary=summary)
            return {
                "status": "success",
                "dry_run": True,
                "summary": summary,
                "stages": {"ingestion": ingestion_payload, "processing": processing_payload},
            }

        chunks = load_chunks()
        if not chunks:
            raise RuntimeError("Aucun chunk disponible apres processing. Publication annulee.")

        candidate_collection_name = build_candidate_collection_name()
        summary["planned_collection_name"] = candidate_collection_name
        manifest = build_index(chunks, target_collection_name=candidate_collection_name)
        summary["candidate_collection_name"] = str(manifest.get("collection_name") or candidate_collection_name)
        update_offline_pipeline_report(
            "indexing",
            {
                "candidate_collection_name": summary["candidate_collection_name"],
                "manifest_path": str(manifest.get("manifest_path") or ""),
                "chunk_count": int(manifest.get("chunk_count", 0) or 0),
                "model_name": str(manifest.get("model_name") or ""),
            },
        )

        validation_payload = {}
        if opts.validate_before_publish:
            validation_payload = validate_candidate_index(
                candidate_manifest=manifest,
                run_eval=bool(opts.run_eval_validation),
            )
            if not validation_payload.get("passed", False):
                status = "failed"
                summary["validation_failures"] = list(validation_payload.get("failures", []) or [])
                finalize_offline_pipeline_report(status, summary=summary)
                return {
                    "status": status,
                    "summary": summary,
                    "validation": validation_payload,
                }

        publish_payload = {}
        if opts.publish:
            publish_payload = publish_qdrant_index(candidate_manifest=manifest)
            summary["published_collection_name"] = str(publish_payload.get("published_collection_name") or "")
            summary["previous_collection_name"] = str(publish_payload.get("previous_collection_name") or "")
            summary["active_collection_name"] = str(publish_payload.get("collection_name") or "")
            update_offline_pipeline_report(
                "publish",
                {
                    "active_alias_name": str(publish_payload.get("active_alias_name") or ""),
                    "active_collection_name": summary["active_collection_name"],
                    "published_collection_name": summary["published_collection_name"],
                    "previous_collection_name": summary["previous_collection_name"],
                },
            )
            invalidate_search_cache(clear_models=True)

            if opts.cleanup_after_publish:
                cleanup_payload = cleanup_offline_artifacts(dry_run=False, require_last_success=False)
                summary["cleanup"] = cleanup_payload

        finalize_offline_pipeline_report(status, summary=summary)
        return {
            "status": status,
            "summary": summary,
            "manifest": manifest,
            "validation": validation_payload,
            "publish": publish_payload,
        }
    except Exception as exc:
        status = "failed"
        summary["error"] = str(exc)
        finalize_offline_pipeline_report(status, summary=summary)
        logger.exception("Echec du pipeline offline")
        raise


def publish_latest_candidate_index(run_validation: bool = True, run_eval: Optional[bool] = None) -> Dict:
    manifest_path = latest_candidate_manifest_path()
    if manifest_path is None:
        raise RuntimeError("Aucun manifest candidat disponible pour la publication.")

    manifest = load_manifest(str(manifest_path))
    validation_payload = {}
    if run_validation:
        validation_payload = validate_candidate_index(
            candidate_manifest=manifest,
            run_eval=run_validation_eval() if run_eval is None else bool(run_eval),
        )
        if not validation_payload.get("passed", False):
            raise RuntimeError("Validation du candidat echouee. Publication annulee.")

    published = publish_qdrant_index(candidate_manifest=manifest, manifest_path=Path(manifest_path))
    invalidate_search_cache(clear_models=True)
    update_offline_pipeline_report(
        "publish",
        {
            "active_alias_name": str(published.get("active_alias_name") or ""),
            "active_collection_name": str(published.get("collection_name") or ""),
            "published_collection_name": str(published.get("published_collection_name") or ""),
            "previous_collection_name": str(published.get("previous_collection_name") or ""),
        },
    )
    return {"validation": validation_payload, "publish": published}
