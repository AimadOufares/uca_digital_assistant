import logging
from typing import Dict, List, Optional

from .generation.rag_engine import answer_question
from .offline.indexing import build_index, load_chunks
from .offline.ingestion import DEFAULT_SEEDS, crawl
from .offline.processing import preprocess_all
from .retrieval.rag_search import invalidate_search_cache

logger = logging.getLogger(__name__)


def run_ingestion(seeds: Optional[List[str]] = None) -> List[Dict]:
    """Etape 1: collecte des documents bruts."""
    selected_seeds = seeds or DEFAULT_SEEDS
    logger.info("Ingestion lancee avec %s seed(s).", len(selected_seeds))
    return crawl(selected_seeds)


def run_processing() -> None:
    """Etape 2: nettoyage + chunking des fichiers bruts."""
    logger.info("Processing lance.")
    preprocess_all()


def run_indexing() -> int:
    """Etape 3: creation/mise a jour de l'index hybride dense + lexical."""
    logger.info("Indexing lance.")
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("Aucun chunk disponible pour l'indexation.")
    build_index(chunks)
    invalidate_search_cache(clear_models=True)
    return len(chunks)


def build_knowledge_base(seeds: Optional[List[str]] = None) -> int:
    """
    Pipeline offline complet.
    A executer manuellement (pas a chaque question).
    """
    run_ingestion(seeds)
    run_processing()
    total_chunks = run_indexing()
    logger.info("Base de connaissances prete (%s chunks).", total_chunks)
    return total_chunks


def ask_question(question: str) -> Dict:
    """
    Flux online de chat:
    - recuperation de contexte
    - generation de reponse
    Ne lance jamais ingestion/processing/indexing.
    """
    return answer_question(question)


def run_pipeline(
    url_or_question: str,
    question: Optional[str] = None,
    rebuild: bool = False,
    seeds: Optional[List[str]] = None,
) -> Dict:
    """
    Compatibilite ascendante:
    - Nouveau mode recommande: run_pipeline("ma question")
    - Ancien mode: run_pipeline("https://seed", "ma question", rebuild=True)
    """
    if question is None:
        final_question = url_or_question
    else:
        final_question = question
        if seeds is None:
            seeds = [url_or_question]

    if rebuild:
        build_knowledge_base(seeds=seeds)

    return ask_question(final_question)


if __name__ == "__main__":
    # Build offline explicite (a lancer quand les donnees changent).
    build_knowledge_base()
