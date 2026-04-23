import logging
from typing import Dict, List, Optional

from .contracts import IngestionJobConfig, QuestionRequest
from .services.offline import build_knowledge_base as build_kb_service
from .services.offline import run_indexing as run_indexing_service
from .services.offline import run_ingestion as run_ingestion_service
from .services.offline import run_processing as run_processing_service
from .services.online import answer_question as answer_question_service

logger = logging.getLogger(__name__)


def run_ingestion(seeds: Optional[List[str]] = None) -> List[Dict]:
    """Etape 1: collecte des documents bruts."""
    logger.info("Ingestion lancee avec %s seed(s).", len(seeds or []))
    result = run_ingestion_service(IngestionJobConfig(seeds=seeds))
    return [{"documents_collected": result.get("documents_collected", 0)}]


def run_processing() -> None:
    """Etape 2: nettoyage + chunking des fichiers bruts."""
    logger.info("Processing lance.")
    run_processing_service()


def run_indexing() -> int:
    """Etape 3: creation/mise a jour de l'index hybride dense + lexical."""
    logger.info("Indexing lance.")
    result = run_indexing_service(publish=False)
    return int(result.chunk_count)


def build_knowledge_base(seeds: Optional[List[str]] = None) -> int:
    """
    Pipeline offline complet.
    A executer manuellement (pas a chaque question).
    """
    result = build_kb_service(
        config=IngestionJobConfig(seeds=seeds),
        publish=False,
    )
    logger.info("Base de connaissances prete (%s chunks).", result.chunk_count)
    return int(result.chunk_count)


def ask_question(question: str) -> Dict:
    """
    Flux online de chat:
    - recuperation de contexte
    - generation de reponse
    Ne lance jamais ingestion/processing/indexing.
    """
    result = answer_question_service(QuestionRequest(question=question))
    return {"answer": result.answer, "sources": result.sources}


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
