from django.core.management.base import BaseCommand

from rag_module.offline.orchestrator import OfflinePipelineOptions, run_offline_pipeline
from rag_module.shared.runtime_config import run_validation_eval


class Command(BaseCommand):
    help = "Execute le pipeline offline incremental (ingestion, processing, indexing, validation et publication optionnelle)."

    def add_arguments(self, parser):
        parser.add_argument("--seed", action="append", dest="seeds", help="Ajoute une seed URL. Peut etre repete.")
        parser.add_argument("--publish", action="store_true", help="Publie l'index candidat via l'alias actif.")
        parser.add_argument("--dry-run", action="store_true", help="Calcule les changements sans ecrire sur disque ni publier.")
        parser.add_argument(
            "--skip-validation",
            action="store_true",
            help="Desactive la validation avant publication.",
        )
        parser.add_argument(
            "--skip-cleanup",
            action="store_true",
            help="Desactive le cleanup automatique apres publication.",
        )
        parser.add_argument(
            "--skip-eval-validation",
            action="store_true",
            help="Desactive l'evaluation retrieval pendant la validation.",
        )

    def handle(self, *args, **options):
        payload = run_offline_pipeline(
            OfflinePipelineOptions(
                seeds=options.get("seeds") or None,
                dry_run=bool(options.get("dry_run")),
                publish=bool(options.get("publish")),
                validate_before_publish=not bool(options.get("skip_validation")),
                run_eval_validation=(run_validation_eval() and not bool(options.get("skip_eval_validation"))),
                cleanup_after_publish=not bool(options.get("skip_cleanup")),
            )
        )
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        self.stdout.write(self.style.SUCCESS(f"Offline pipeline status: {payload.get('status', 'unknown')}"))
        self.stdout.write(f"Report: {summary.get('report_path', '-')}")
        self.stdout.write(f"Candidate collection: {summary.get('candidate_collection_name') or summary.get('planned_collection_name') or '-'}")
        self.stdout.write(f"Published collection: {summary.get('published_collection_name') or '-'}")
        self.stdout.write(
            "Files delta: "
            f"+{len(summary.get('new_files', []) or [])} "
            f"~{len(summary.get('modified_files', []) or [])} "
            f"-{len(summary.get('deleted_files', []) or [])}"
        )
        self.stdout.write(
            "Chunks delta: "
            f"+{int(summary.get('chunks_added', 0) or 0)} "
            f"-{int(summary.get('chunks_removed', 0) or 0)}"
        )

