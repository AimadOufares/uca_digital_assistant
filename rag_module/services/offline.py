from typing import Dict, List, Optional

from ..adapters.vector_store import get_vector_store_adapter
from ..contracts import IndexBuildResult, IngestionJobConfig
from ..evaluation.evaluate_rag import evaluate, write_report
from ..offline.indexing import load_chunks
from ..offline.ingestion import DEFAULT_SEEDS, crawl
from ..offline.processing import preprocess_all
from ..retrieval.rag_search import invalidate_search_cache
from ..shared.runtime import get_runtime_settings


def run_ingestion(config: IngestionJobConfig | None = None) -> Dict:
    settings = get_runtime_settings()
    refresh_mode = (config.refresh_mode if config else "") or settings.app_env
    seeds = (config.seeds if config else None) or DEFAULT_SEEDS
    results = crawl(seeds)
    return {
        "status": "ok",
        "step": "ingestion",
        "seeds": seeds,
        "documents_collected": len(results),
        "refresh_mode": refresh_mode,
    }


def run_processing() -> Dict:
    preprocess_all()
    return {"status": "ok", "step": "processing"}


def run_indexing(publish: bool = False, build_id: Optional[str] = None) -> IndexBuildResult:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("Aucun chunk disponible pour l'indexation.")
    adapter = get_vector_store_adapter()
    result = adapter.build_index(chunks, build_id=build_id, publish=publish)
    invalidate_search_cache(clear_models=True)
    return result


def build_knowledge_base(
    config: IngestionJobConfig | None = None,
    publish: bool = False,
    build_id: Optional[str] = None,
) -> IndexBuildResult:
    run_ingestion(config)
    run_processing()
    return run_indexing(publish=publish, build_id=build_id)


def run_evaluation(top_k: int = 5, skip_generation: bool = False) -> Dict[str, str]:
    report = evaluate(top_k=max(1, top_k), run_generation=not skip_generation)
    paths = write_report(report)
    return {key: str(value) for key, value in paths.items()}
