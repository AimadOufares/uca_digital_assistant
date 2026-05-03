from django.core.management.base import BaseCommand

from rag_module.services.offline import run_evaluation


class Command(BaseCommand):
    help = "Lance l'evaluation heuristique du RAG."

    def add_arguments(self, parser):
        parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks recuperes pour l'evaluation.")
        parser.add_argument(
            "--benchmark",
            choices=["generic", "drive"],
            default="drive",
            help="Jeu d'evaluation a utiliser.",
        )
        parser.add_argument(
            "--skip-generation",
            action="store_true",
            help="N'evalue que la retrieval sans generation.",
        )

    def handle(self, *args, **options):
        paths = run_evaluation(
            top_k=max(1, int(options["top_k"])),
            skip_generation=bool(options["skip_generation"]),
            benchmark=str(options["benchmark"] or "drive"),
        )
        self.stdout.write(self.style.SUCCESS(f"Evaluation JSON: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Evaluation TXT : {paths['txt']}"))
