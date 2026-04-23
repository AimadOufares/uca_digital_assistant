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

    def handle(self, *args, **options):
        config = IngestionJobConfig(seeds=options["seeds"] or None)
        result = run_ingestion(config)
        self.stdout.write(self.style.SUCCESS(f"Ingestion terminee: {result['documents_collected']} documents."))
