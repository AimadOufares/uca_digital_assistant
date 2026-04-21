from django.core.management.base import BaseCommand

from rag_module.offline.cleanup import cleanup_offline_artifacts


class Command(BaseCommand):
    help = "Nettoie les artefacts offline (raw anciens, chunks orphelins, anciens index candidats)."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Liste ce qui serait nettoye sans suppression effective.")
        parser.add_argument(
            "--force",
            action="store_true",
            help="Autorise le cleanup meme si le dernier run offline n'est pas en succes.",
        )

    def handle(self, *args, **options):
        payload = cleanup_offline_artifacts(
            dry_run=bool(options.get("dry_run")),
            require_last_success=not bool(options.get("force")),
        )
        if payload.get("skipped"):
            self.stdout.write(f"Cleanup saute: {payload.get('reason', 'unknown')}")
            return

        self.stdout.write(self.style.SUCCESS("Cleanup offline termine."))
        self.stdout.write(f"Raw removed: {len(payload.get('raw', {}).get('removed_files', []) or [])}")
        self.stdout.write(f"Processed removed: {len(payload.get('processed', {}).get('removed_files', []) or [])}")
        self.stdout.write(f"Index collections removed: {len(payload.get('indexes', {}).get('removed_collections', []) or [])}")
