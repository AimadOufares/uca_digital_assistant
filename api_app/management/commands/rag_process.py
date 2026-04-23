from django.core.management.base import BaseCommand

from rag_module.services.offline import run_processing


class Command(BaseCommand):
    help = "Execute l'etape de processing RAG."

    def add_arguments(self, parser):
        parser.add_argument(
            "--corpus",
            choices=["main", "archive", "all"],
            default="all",
            help="Corpus a traiter.",
        )

    def handle(self, *args, **options):
        result = run_processing(corpus=options["corpus"])
        self.stdout.write(
            self.style.SUCCESS(f"Processing termine: {result['step']} ({result['corpus']})")
        )
