# rag_module/prompt_builder.py
from typing import List, Dict
import json
from datetime import datetime

def build_prompt_fr(
    query: str,
    chunks: List[Dict],
    include_sources: bool = True,
    max_context_length: int = 8000,
    temperature_hint: float = 0.3
) -> str:
    """
    Construit un prompt optimisé en français pour un RAG universitaire.
    """

    if not chunks:
        return f"""
Vous êtes un assistant d'information précis et fiable de l'Université Cadi Ayyad.

Question de l'utilisateur : {query}

Malheureusement, je n'ai trouvé aucune information pertinente dans ma base de connaissances actuelle.

Répondez poliment que l'information n'est pas disponible pour le moment et proposez à l'utilisateur de reformuler sa question ou de contacter le service concerné.

Réponse :
"""

    # Construction du contexte avec métadonnées utiles
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        source = metadata.get("file_name", "Document")
        chunk_type = "Tableau" if metadata.get("is_table") else "Texte"

        part = f"""[Document {i} - {chunk_type}]
Source : {source}
Contenu :
{text}
"""
        context_parts.append(part)

    context_text = "\n\n".join(context_parts)

    # Troncature intelligente si trop long
    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "\n\n... (contexte tronqué pour respecter les limites)"

    prompt = f"""Vous êtes un assistant administratif et académique officiel de la **Faculté des Sciences Semlalia** (Université Cadi Ayyad, Marrakech).

Votre rôle est d'aider les étudiants, parents et personnel avec des réponses précises, claires et professionnelles.

### Contexte disponible (informations vérifiées) :
{context_text}

### Question de l'utilisateur :
{query}

### Instructions strictes :
- Répondez **uniquement en français**, de manière naturelle et polie.
- Basez votre réponse **exclusivement** sur le contexte fourni ci-dessus.
- Si l'information demandée n'est pas présente dans le contexte, répondez exactement : "Information non disponible dans mes sources actuelles."
- Soyez clair, structuré et précis. Utilisez des listes numérotées ou à puces quand c'est pertinent.
- Mentionnez la source lorsque c'est utile (ex: "Selon le document d'inscription...").
- Ne faites pas d'hypothèses. Ne donnez pas de conseils juridiques ou financiers.
- Si plusieurs documents contiennent des informations complémentaires, synthétisez-les de façon cohérente.

### Réponse :
"""

    return prompt.strip()


# Version alternative plus concise (pour modèles plus petits ou réponses rapides)
def build_prompt_fr_concise(query: str, chunks: List[Dict]) -> str:
    """Version légère pour modèles rapides (gpt-4o-mini, etc.)"""
    context_text = "\n\n".join([f"- {c.get('text', '')}" for c in chunks])

    prompt = f"""Tu es un assistant utile de la Faculté Semlalia.

Contexte :
{context_text}

Question : {query}

Réponds en français, de façon claire et directe. 
Utilise uniquement les informations du contexte. 
Si tu ne sais pas, dis "Information non disponible".

Réponse :"""

    return prompt.strip()


# Fonction principale recommandée
def build_rag_prompt(
    query: str,
    chunks: List[Dict],
    style: str = "standard"  # "standard" ou "concise"
) -> str:
    if style == "concise":
        return build_prompt_fr_concise(query, chunks)
    return build_prompt_fr(query, chunks)