from typing import Dict, Optional

from ..adapters.vector_store import get_vector_store_adapter
from ..contracts import IndexBuildResult, IngestionJobConfig
from ..evaluation.evaluate_rag import evaluate, write_report
from ..offline.indexing import load_chunks
from ..offline.ingestion_utils import crawl, default_seeds
from ..offline.processing import preprocess_all
from ..retrieval.rag_search import invalidate_search_cache
from ..shared.runtime import get_runtime_settings


def run_ingestion(config: IngestionJobConfig | None = None) -> Dict:
    settings = get_runtime_settings()
    resolved_config = config or IngestionJobConfig()
    if not resolved_config.seeds:
        resolved_config = IngestionJobConfig(
            seeds=default_seeds(resolved_config.mode, premium_only=resolved_config.premium_only),
            limits=resolved_config.limits,
            refresh_mode=resolved_config.refresh_mode,
            mode=resolved_config.mode,
            target_corpus=resolved_config.target_corpus,
            premium_only=resolved_config.premium_only,
        )
    result = crawl(resolved_config)
    result["refresh_mode"] = resolved_config.refresh_mode or settings.app_env
    return result


def run_processing(corpus: str = "all") -> Dict:
    preprocess_all(corpus=corpus)
    return {"status": "ok", "step": "processing", "corpus": corpus}


def run_indexing(
    corpus: str = "main",
    publish: bool = False,
    build_id: Optional[str] = None,
) -> IndexBuildResult:
    if corpus != "main":
        raise RuntimeError("Seul le corpus principal peut etre indexe dans cette phase.")
    chunks = load_chunks(corpus=corpus)
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
    resolved_config = config or IngestionJobConfig()
    run_ingestion(resolved_config)
    run_processing(corpus=resolved_config.target_corpus)
    return run_indexing(corpus="main", publish=publish, build_id=build_id)


def run_evaluation(top_k: int = 5, skip_generation: bool = False) -> Dict[str, str]:
    report = evaluate(top_k=max(1, top_k), run_generation=not skip_generation)
    paths = write_report(report)
    return {key: str(value) for key, value in paths.items()}
