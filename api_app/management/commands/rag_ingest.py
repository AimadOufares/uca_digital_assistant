from django.core.management.base import BaseCommand

from rag_module.contracts import IngestionJobConfig
from rag_module.services.offline import run_ingestion


class Command(BaseCommand):
    help = "Execute l'etape d'ingestion RAG."

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
            help="Corpus cible de l'operation.",
        )
        parser.add_argument(
            "--premium-only",
            action="store_true",
            help="N'utilise que les seeds premium definies dans la cartographie.",
        )

    def handle(self, *args, **options):
        config = IngestionJobConfig(
            seeds=options["seeds"] or None,
            mode=options["mode"],
            target_corpus=options["target_corpus"],
            premium_only=bool(options["premium_only"]),
        )
        result = run_ingestion(config)
        self.stdout.write(
            self.style.SUCCESS(
                "Ingestion terminee: "
                f"{result['documents_collected']} documents, "
                f"main={result.get('main_count', 0)}, "
                f"archive={result.get('archive_count', 0)}, "
                f"reject={result.get('reject_count', 0)}."
            )
        )
