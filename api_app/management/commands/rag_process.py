from django.core.management.base import BaseCommand

from rag_module.services.offline import run_processing


class Command(BaseCommand):
    help = "Execute l'etape de processing RAG."

    def add_arguments(self, parser):
        parser.add_argument(
            "--corpus",
            choices=["main", "archive", "drive", "all"],
            default="all",
            help="Corpus a traiter.",
        )

    def handle(self, *args, **options):
        result = run_processing(corpus=options["corpus"])
        self.stdout.write(
            self.style.SUCCESS(f"Processing termine: {result['step']} ({result['corpus']})")
        )
        for summary in result.get("corpora", []):
            self.stdout.write(
                "  "
                f"{summary.get('corpus')}: "
                f"detected={summary.get('detected', 0)}, "
                f"processed={summary.get('processed', 0)}, "
                f"skipped={summary.get('skipped_unchanged', 0)}, "
                f"no_chunks={summary.get('skipped_no_chunks', 0)}, "
                f"failed={summary.get('failed', 0)}, "
                f"quarantined={summary.get('quarantined', 0)}"
            )
