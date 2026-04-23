from django.core.management.base import BaseCommand

from rag_module.services.offline import run_processing


class Command(BaseCommand):
    help = "Execute l'etape de processing RAG."

    def handle(self, *args, **options):
        result = run_processing()
        self.stdout.write(self.style.SUCCESS(f"Processing termine: {result['step']}"))
