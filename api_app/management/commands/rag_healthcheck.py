import json

from django.core.management.base import BaseCommand, CommandError

from rag_module.services.health import build_ready_health


class Command(BaseCommand):
    help = "Execute un readiness check RAG."

    def add_arguments(self, parser):
        parser.add_argument("--json", action="store_true", help="Affiche la sortie en JSON.")

    def handle(self, *args, **options):
        payload = build_ready_health()
        if options["json"]:
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(f"ready={payload.get('ready')} vector={payload.get('vector_store', {})}")
            self.stdout.write(f"llm={payload.get('llm', {}).get('state')}")
        if not payload.get("ready"):
            raise CommandError("Readiness check failed.")
