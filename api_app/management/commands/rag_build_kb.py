from django.core.management.base import BaseCommand

from rag_module.contracts import IngestionJobConfig
from rag_module.services.offline import build_knowledge_base


class Command(BaseCommand):
    help = "Execute ingestion + processing + indexing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            action="append",
            dest="seeds",
            default=[],
            help="URL seed additionnelle. Repetable.",
        )
        parser.add_argument(
            "--mode",
            choices=["fast", "extended"],
            default="fast",
            help="Mode d'ingestion a utiliser.",
        )
        parser.add_argument(
            "--target-corpus",
            choices=["main", "archive", "all"],
            default="all",
            help="Corpus a ingerer et traiter.",
        )
        parser.add_argument(
            "--premium-only",
            action="store_true",
            help="N'utilise que les seeds premium definies dans la cartographie.",
        )
        parser.add_argument("--publish", action="store_true", help="Publie l'index construit comme index actif.")
        parser.add_argument("--build-id", default="", help="Identifiant de build explicite.")

    def handle(self, *args, **options):
        config = IngestionJobConfig(
            seeds=options["seeds"] or None,
            mode=options["mode"],
            target_corpus=options["target_corpus"],
            premium_only=bool(options["premium_only"]),
        )
        result = build_knowledge_base(
            config=config,
            publish=bool(options["publish"]),
            build_id=options["build_id"] or None,
        )
        message = (
            f"Knowledge base construite. backend={result.backend} "
            f"build_id={result.build_id} chunks={result.chunk_count} published={result.published}"
        )
        self.stdout.write(self.style.SUCCESS(message))
