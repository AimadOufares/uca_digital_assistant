from typing import List, Set, Tuple

from .metadata_policy import normalize_text


TARGET_KEYWORDS = {
    "inscription",
    "preinscription",
    "pre-inscription",
    "reinscription",
    "admission",
    "admis",
    "candidature",
    "concours",
    "bourse",
    "calendrier",
    "scolarite",
    "orientation",
    "master",
    "licence",
    "doctorat",
    "filiere",
    "emploi du temps",
    "planning",
    "resultat",
    "resultats",
    "rattrapage",
    "semestre",
    "module",
    "attestation",
    "paiement",
    "frais",
    "reclamation",
    "equivalence",
    "inscription administrative",
    "registration",
    "scholarship",
    "application",
    "admissions",
    "schedule",
    "student",
    "etudiant",
    "etudiants",
    "\u0627\u0644\u062a\u0633\u062c\u064a\u0644",
    "\u0642\u0628\u0648\u0644",
    "\u0645\u0646\u062d\u0629",
    "\u0645\u0628\u0627\u0631\u0627\u0629",
    "\u0645\u0627\u0633\u062a\u0631",
    "\u0625\u062c\u0627\u0632\u0629",
}

SOURCE_HINT_KEYWORDS = {
    "etudiant",
    "etudiants",
    "student",
    "students",
    "scolarite",
    "pedagogique",
    "administratif",
    "administrative",
    "service",
    "campus",
    "formation",
    "programme",
    "procedure",
    "modalite",
    "modalites",
    "deadline",
    "dossier",
}

HIGH_SIGNAL_DOCUMENT_TYPES = {"admission", "inscription", "bourse", "calendrier", "resultats", "formation"}
MIN_CHUNK_RELEVANCE_SCORE = 1

NORMALIZED_TARGET_KEYWORDS = {normalize_text(keyword) for keyword in TARGET_KEYWORDS}
NORMALIZED_SOURCE_HINTS = {normalize_text(keyword) for keyword in SOURCE_HINT_KEYWORDS}


def keyword_hits(text: str, keyword_bank: Set[str]) -> Set[str]:
    normalized = normalize_text(text)
    return {keyword for keyword in keyword_bank if keyword and keyword in normalized}


def compute_source_relevance(source_path: str, joined_text: str, document_type: str) -> Tuple[int, List[str]]:
    signals: List[str] = []
    score = 0

    source_hits = keyword_hits(source_path, NORMALIZED_TARGET_KEYWORDS)
    text_hits = keyword_hits(joined_text[:16000], NORMALIZED_TARGET_KEYWORDS)
    hint_hits = keyword_hits(f"{source_path} {joined_text[:6000]}", NORMALIZED_SOURCE_HINTS)

    if source_hits:
        score += min(3, len(source_hits))
        signals.extend(sorted(source_hits)[:5])
    if text_hits:
        score += min(4, len(text_hits))
        signals.extend(sorted(text_hits)[:6])
    if hint_hits:
        score += 1
        signals.extend(sorted(hint_hits)[:3])
    if document_type in HIGH_SIGNAL_DOCUMENT_TYPES:
        score += 2
        signals.append(f"document_type:{document_type}")

    return score, list(dict.fromkeys(signals))


def compute_chunk_relevance(text: str, document_type: str) -> Tuple[int, List[str]]:
    score = 0
    signals: List[str] = []

    chunk_hits = keyword_hits(text, NORMALIZED_TARGET_KEYWORDS)
    hint_hits = keyword_hits(text, NORMALIZED_SOURCE_HINTS)

    if chunk_hits:
        score += min(3, len(chunk_hits))
        signals.extend(sorted(chunk_hits)[:5])
    if hint_hits:
        score += 1
        signals.extend(sorted(hint_hits)[:3])
    if document_type in HIGH_SIGNAL_DOCUMENT_TYPES and (chunk_hits or hint_hits):
        score += 1
        signals.append(f"document_type:{document_type}")

    return score, list(dict.fromkeys(signals))


def should_keep_chunk(chunk_relevance_score: int) -> bool:
    return chunk_relevance_score >= MIN_CHUNK_RELEVANCE_SCORE
