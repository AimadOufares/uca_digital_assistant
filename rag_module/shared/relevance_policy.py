from typing import Dict, List, Set, Tuple

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
    "mot de passe",
    "connexion",
    "plateforme",
    "compte universitaire",
    "ucastudent",
    "pedoc",
    "ucaplat",
    "cip",
    "e-diplome",
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
    "numerique",
    "plateforme",
    "compte",
    "authentification",
    "campus",
    "formation",
    "programme",
    "procedure",
    "modalite",
    "modalites",
    "deadline",
    "dossier",
}

HIGH_SIGNAL_DOCUMENT_TYPES = {
    "admission",
    "inscription",
    "bourse",
    "calendrier",
    "resultats",
    "formation",
    "scolarite",
    "pedagogie_numerique",
    "digital_service",
}
SERVICE_ALIASES = {
    "ucastudent": ["uc@student", "ucastudent", "uc student"],
    "ucaplat": ["ucaplat"],
    "pedoc": ["pedoc"],
    "cip": ["cip", "centre d'innovation pedagogique", "centre innovation pedagogique"],
    "e-candidature": ["e-candidature", "e candidature", "ecandidature"],
    "diplomes": ["espace diplomes", "e diplome", "diplomes.uca.ma"],
    "pucastaff": ["pucastaff"],
    "hpc": ["hpc", "hpc uca"],
}
MIN_CHUNK_RELEVANCE_SCORE = 1

NORMALIZED_TARGET_KEYWORDS = {normalize_text(keyword) for keyword in TARGET_KEYWORDS}
NORMALIZED_SOURCE_HINTS = {normalize_text(keyword) for keyword in SOURCE_HINT_KEYWORDS}
NORMALIZED_SERVICE_ALIASES = {
    key: [normalize_text(alias) for alias in aliases if normalize_text(alias)]
    for key, aliases in SERVICE_ALIASES.items()
}


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


def query_keyword_hits(query: str) -> List[str]:
    return sorted(keyword_hits(query, NORMALIZED_TARGET_KEYWORDS))


def compute_metadata_boost(metadata: Dict, query: str) -> float:
    if not metadata:
        return 0.0

    normalized_query = normalize_text(query)
    query_hits = set(query_keyword_hits(query))
    boost = 0.0

    document_type = normalize_text(str(metadata.get("document_type") or ""))
    faculty = normalize_text(str(metadata.get("faculty") or ""))
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    service_name = normalize_text(str(metadata.get("service_name") or ""))
    official_url = normalize_text(str(metadata.get("official_url") or ""))
    intents = [normalize_text(str(item)) for item in metadata.get("intent", []) or []]
    retrieval_keywords = [normalize_text(str(item)) for item in metadata.get("retrieval_keywords", []) or []]
    year = metadata.get("year")

    if document_type and document_type in normalized_query:
        boost += 0.08
    if document_type and query_hits and document_type in HIGH_SIGNAL_DOCUMENT_TYPES:
        if document_type in {"inscription", "admission", "bourse", "calendrier", "resultats", "formation"}:
            boost += 0.04

    if faculty and faculty != "unknown" and faculty in normalized_query:
        boost += 0.05

    if page_kind and page_kind in normalized_query:
        boost += 0.04
    if intents and any(intent and intent in normalized_query for intent in intents):
        boost += 0.05
    if retrieval_keywords and any(keyword and keyword in normalized_query for keyword in retrieval_keywords):
        boost += 0.05

    for canonical_service, aliases in NORMALIZED_SERVICE_ALIASES.items():
        query_mentions_service = any(alias and alias in normalized_query for alias in aliases)
        if not query_mentions_service:
            continue
        if service_name == canonical_service or canonical_service in official_url or any(
            alias and (alias in service_name or alias in official_url) for alias in aliases
        ):
            boost += 0.12
        else:
            boost -= 0.04

    if isinstance(year, int) and any(token in normalized_query for token in ("calendrier", "resultat", "resultats", "inscription")):
        if year >= 2024:
            boost += 0.03

    return max(-0.05, min(boost, 0.22))


def boost_results_with_metadata(results: List[Dict], query: str) -> List[Dict]:
    boosted: List[Dict] = []
    for result in results:
        enriched = dict(result)
        metadata = enriched.get("metadata", {}) or {}
        boost = compute_metadata_boost(metadata, query)
        base_score = float(enriched.get("score", 0.0) or 0.0)
        enriched["metadata_boost"] = round(boost, 4)
        enriched["score"] = max(0.0, min(1.0, base_score + boost))
        boosted.append(enriched)
    boosted.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return boosted
