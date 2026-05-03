"""Test final end-to-end du pipeline RAG avec les nouveaux modeles."""
import os
import sys

# Fix encodage Windows pour les emojis
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ["DJANGO_SETTINGS_MODULE"] = "core.settings"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force le mode hors-ligne
os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, os.getcwd())

from rag_module.pipeline import ask_question

questions = [
    # --- Plateformes numériques ---
    "Qu'est-ce que la plateforme CIP ?",
    "À quoi sert le portail UcaStudent ?",
    "Comment accéder à la plateforme PEDOC ?",
    "Quelles sont les fonctionnalités du portail numérique de l'UCA ?",

    # --- Services spécifiques ---
    "Comment déposer une demande de bourse en ligne ?",
    "Quels services sont disponibles pour les doctorants ?",
    "Comment contacter le service de scolarité ?",
    "Où puis-je consulter mon emploi du temps ?",

    # --- Questions formulées comme un étudiant ---
    "je veux m inscrire comment je fais",
    "c est quoi uca student",
    "j arrive pas a acceder a mon compte",
    "kif ndir inscription en ligne",

    # --- Questions hors périmètre (doit répondre qu'il ne sait pas) ---
    "Quel est le menu de la cantine universitaire ?",
    "Donne-moi la liste des professeurs de mathématiques",
]

print("=" * 60)
print("TEST FINAL DU PIPELINE RAG - MODE HORS-LIGNE")
print("=" * 60)

for i, q in enumerate(questions, 1):
    print(f"\n[Q{i}] {q}")
    print("-" * 50)
    try:
        result = ask_question(q)
        answer = result.get("answer", "Pas de réponse")
        sources = result.get("sources", [])
        print(f"Reponse : {answer[:300]}...")
        print(f"Sources  : {[s.get('name','?') for s in sources]}")
    except Exception as e:
        print(f"ERREUR: {e}")

print("\n" + "=" * 60)
print("FIN DES TESTS")
print("=" * 60)
