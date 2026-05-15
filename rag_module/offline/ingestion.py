import logging
from typing import Dict, List, Optional

from ..contracts import IngestionJobConfig
from .ingestion_utils import crawl as crawl_pipeline
from .ingestion_utils import default_seeds

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = default_seeds("fast")


def crawl(seeds: Optional[List[str]] = None) -> List[Dict]:
    config = IngestionJobConfig(
        seeds=seeds,
        mode="fast",
        target_corpus="all",
    )
    result = crawl_pipeline(config)
    return list(result.get("documents", []))


if __name__ == "__main__":
    logger.info("Crawler RAG demarre")
    crawl(DEFAULT_SEEDS)
