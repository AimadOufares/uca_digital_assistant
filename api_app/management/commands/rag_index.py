from django.core.management.base import BaseCommand

from rag_module.services.offline import run_indexing


class Command(BaseCommand):
    help = "Construit un index vectoriel pour le backend configure."

    def add_arguments(self, parser):
        parser.add_argument("--publish", action="store_true", help="Publie l'index construit comme index actif.")
        parser.add_argument("--build-id", default="", help="Identifiant de build explicite.")

    def handle(self, *args, **options):
        result = run_indexing(
            publish=bool(options["publish"]),
            build_id=options["build_id"] or None,
        )
        message = (
            f"Index {result.backend} construit. build_id={result.build_id} "
            f"chunks={result.chunk_count} published={result.published}"
        )
        self.stdout.write(self.style.SUCCESS(message))
