from django.core.management.base import BaseCommand

from rag_module.offline.orchestrator import publish_latest_candidate_index
from rag_module.offline.qdrant_indexing import latest_candidate_manifest_path


class Command(BaseCommand):
    help = "Publie le dernier index candidat valide sur l'alias Qdrant actif."

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-validation",
            action="store_true",
            help="Publie sans relancer la validation du candidat.",
        )
        parser.add_argument(
            "--skip-eval-validation",
            action="store_true",
            help="Si la validation est active, saute l'evaluation retrieval.",
        )

    def handle(self, *args, **options):
        manifest_path = latest_candidate_manifest_path()
        if manifest_path is None:
            self.stderr.write("Aucun manifest candidat disponible.")
            return

        payload = publish_latest_candidate_index(
            run_validation=not bool(options.get("skip_validation")),
            run_eval=not bool(options.get("skip_eval_validation")),
        )
        publish = payload.get("publish", {}) if isinstance(payload, dict) else {}
        self.stdout.write(self.style.SUCCESS("Publication terminee."))
        self.stdout.write(f"Candidate manifest: {manifest_path}")
        self.stdout.write(f"Active alias: {publish.get('active_alias_name') or '-'}")
        self.stdout.write(f"Published collection: {publish.get('published_collection_name') or '-'}")
        self.stdout.write(f"Previous collection: {publish.get('previous_collection_name') or '-'}")

