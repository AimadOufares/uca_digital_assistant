import math
import re
from typing import Any, Dict, List, Set, Tuple

from ..shared.metadata_policy import normalize_text


QUERY_STOPWORDS = {
    "a", "au", "aux", "avec", "comment", "dans", "de", "des", "du", "en", "est", "et", "faire",
    "la", "le", "les", "ma", "mes", "mon", "ou", "pour", "quelles", "quelle", "quel", "quels",
    "qui", "sur", "un", "une", "vos", "votre",
}

QUERY_CANONICAL_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bs[' ]inscrire\b", "inscription"),
    (r"\bsinscrire\b", "inscription"),
    (r"\binscrire\b", "inscription"),
    (r"\binscription a\b", "inscription"),
    (r"\bpre inscription\b", "preinscription"),
    (r"\bbource\b", "bourse"),
    (r"\bboursse\b", "bourse"),
    (r"\bbours\b", "bourse"),
    (r"\bfac\b", "faculte"),
]

SERVICE_ALIAS_RULES: Dict[str, Set[str]] = {
    "ucastudent": {"uc@student", "ucastudent", "uc student"},
    "ucaplat": {"ucaplat"},
    "pedoc": {"pedoc"},
    "cip": {"cip", "centre d innovation pedagogique", "centre innovation pedagogique"},
    "e-candidature": {"e-candidature", "e candidature", "ecandidature"},
    "diplomes": {"espace diplomes", "espace diplomes", "e diplome", "diplomes.uca.ma"},
    "pucastaff": {"pucastaff"},
    "hpc": {"hpc", "hpc uca"},
}

QUERY_INTENT_RULES: Dict[str, Set[str]] = {
    "connexion": {"connexion", "connecter", "login", "authentification", "acces"},
    "mot_de_passe": {"mot de passe", "password", "reinitialiser", "oubli", "email academique"},
    "attestation": {"attestation", "certificat", "attestation d inscription", "attestation de scolarite"},
    "notes": {"notes", "resultats", "releve"},
    "candidature": {"candidature", "preinscription", "postuler", "dossier", "suivi"},
    "cours": {"cours", "devoir", "module", "examen en ligne", "classe virtuelle"},
}
STRICT_QUERY_INTENTS = {"mot_de_passe", "attestation"}

QUERY_TOPIC_RULES: Dict[str, Dict[str, Any]] = {
    "stage": {"keywords": {"stage", "stages", "pfe", "projet de fin d etudes", "projet de fin d'etudes", "projet fin d etudes", "convention de stage", "memoire", "internship", "internships"}, "allowed_document_types": {"stage", "formation", "general"}, "conflicts": {"bourse", "calendrier", "resultats"}},
    "inscription": {"keywords": {"inscription", "preinscription", "pre inscription", "reinscription", "inscription administrative", "s inscrire", "sinscrire", "inscrire", "scolarite", "registration"}, "allowed_document_types": {"inscription", "admission", "scolarite", "digital_service", "general"}, "conflicts": {"bourse", "resultats"}},
    "admission": {"keywords": {"admission", "admissions", "candidature", "selection", "concours", "appel a candidature", "appel a candidatures", "application"}, "allowed_document_types": {"admission", "inscription", "formation", "scolarite", "general"}, "conflicts": {"bourse", "stage"}},
    "bourse": {"keywords": {"bourse", "bourses", "scholarship", "scholarships", "allocation", "aide financiere"}, "allowed_document_types": {"bourse", "vie_etudiante", "scolarite", "digital_service", "general"}, "conflicts": {"stage", "admission", "inscription", "resultats"}},
    "calendrier": {"keywords": {"calendrier", "planning", "emploi du temps", "date", "dates", "delai", "delais", "deadline", "schedule"}, "allowed_document_types": {"calendrier", "resultats", "inscription", "scolarite", "general"}, "conflicts": {"bourse"}},
    "resultats": {"keywords": {"resultat", "resultats", "note", "notes", "deliberation", "rattrapage", "classement"}, "allowed_document_types": {"resultats", "admission", "general"}, "conflicts": {"bourse", "stage"}},
    "diplome": {"keywords": {"diplome", "diplomes", "e diplome", "e-diplome", "duplicata", "suivi du diplome", "etat du diplome", "statut du diplome"}, "allowed_document_types": {"scolarite", "digital_service", "general"}, "conflicts": set()},
    "formation": {"keywords": {"formation", "formations", "filiere", "programme", "master", "licence", "doctorat", "doctorale", "module", "modules", "cours"}, "allowed_document_types": {"formation", "admission", "pedagogie_numerique", "general"}, "conflicts": set()},
    "soutien_recherche": {"keywords": {"soutien", "accompagnement", "accompagnement projet", "monter un projet", "monter un projet de recherche", "demande de soutien", "projet de recherche"}, "allowed_document_types": {"recherche", "general"}, "conflicts": set()},
    "contact": {"keywords": {"contact", "contacts", "telephone", "email", "mail", "adresse", "service", "scolarite"}, "allowed_document_types": {"contact", "inscription", "digital_service", "scolarite", "general"}, "conflicts": set()},
    "reglement": {"keywords": {"reglement", "reglements", "reglement pedagogique", "reglements pedagogiques", "lmd", "ects", "modalite", "modalites"}, "allowed_document_types": {"reglement", "formation", "pedagogie_numerique", "general"}, "conflicts": set()},
}

LEVEL_KEYWORDS = {
    "master": {"master", "masters", "mastÃ¨re"},
    "licence": {"licence", "licences", "license"},
    "doctorat": {"doctorat", "doctorale", "doctorales", "phd", "these", "theses"},
}

NORMALIZED_QUERY_TOPIC_RULES: Dict[str, Dict[str, Any]] = {
    topic: {
        "keywords": {normalize_text(keyword) for keyword in config.get("keywords", set()) if normalize_text(keyword)},
        "allowed_document_types": {normalize_text(doc_type) for doc_type in config.get("allowed_document_types", set()) if normalize_text(doc_type)},
        "conflicts": {normalize_text(topic_name) for topic_name in config.get("conflicts", set()) if normalize_text(topic_name)},
    }
    for topic, config in QUERY_TOPIC_RULES.items()
}
NORMALIZED_LEVEL_KEYWORDS = {level: {normalize_text(keyword) for keyword in keywords if normalize_text(keyword)} for level, keywords in LEVEL_KEYWORDS.items()}
NORMALIZED_SERVICE_ALIAS_RULES: Dict[str, Set[str]] = {service: {normalize_text(alias) for alias in aliases if normalize_text(alias)} for service, aliases in SERVICE_ALIAS_RULES.items()}
NORMALIZED_QUERY_INTENT_RULES: Dict[str, Set[str]] = {intent: {normalize_text(alias) for alias in aliases if normalize_text(alias)} for intent, aliases in QUERY_INTENT_RULES.items()}
NORMALIZED_FACULTY_RULES: Dict[str, str] = {}


def _tokenize_normalized(text: str) -> Set[str]:
    return set(re.findall(r"\b[\w']+\b", normalize_text(text)))


def _extract_query_topics(normalized_query: str, query_tokens: Set[str]) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {}
    for topic, config in NORMALIZED_QUERY_TOPIC_RULES.items():
        hits: List[str] = []
        for keyword in config["keywords"]:
            if not keyword:
                continue
            if " " in keyword and keyword in normalized_query:
                hits.append(keyword)
                continue
            keyword_tokens = _tokenize_normalized(keyword)
            if keyword_tokens and keyword_tokens.issubset(query_tokens):
                hits.append(keyword)
        if hits:
            matches[topic] = sorted(set(hits))
    return matches


def _extract_query_levels(normalized_query: str, query_tokens: Set[str]) -> List[str]:
    levels: List[str] = []
    for level, keywords in NORMALIZED_LEVEL_KEYWORDS.items():
        for keyword in keywords:
            if not keyword:
                continue
            if (" " in keyword and keyword in normalized_query) or _tokenize_normalized(keyword).issubset(query_tokens):
                levels.append(level)
                break
    return levels


def _extract_query_faculties(normalized_query: str) -> List[str]:
    faculties: List[str] = []
    for token, label in NORMALIZED_FACULTY_RULES.items():
        if token and token in normalized_query and label not in faculties:
            faculties.append(label)
    return faculties


def _extract_query_services(normalized_query: str, query_tokens: Set[str]) -> List[str]:
    services: List[str] = []
    for service, aliases in NORMALIZED_SERVICE_ALIAS_RULES.items():
        for alias in aliases:
            if not alias:
                continue
            if (" " in alias and alias in normalized_query) or _tokenize_normalized(alias).issubset(query_tokens):
                services.append(service)
                break
    return services


def _extract_query_intents(normalized_query: str, query_tokens: Set[str]) -> List[str]:
    intents: List[str] = []
    for intent, aliases in NORMALIZED_QUERY_INTENT_RULES.items():
        for alias in aliases:
            if not alias:
                continue
            if (" " in alias and alias in normalized_query) or _tokenize_normalized(alias).issubset(query_tokens):
                intents.append(intent)
                break
    return intents


def _alias_matches_text(alias: str, normalized_text: str, text_tokens: Set[str]) -> bool:
    if not alias:
        return False
    if " " in alias:
        return alias in normalized_text
    alias_tokens = _tokenize_normalized(alias)
    return bool(alias_tokens) and alias_tokens.issubset(text_tokens)


def build_query_profile(query: str) -> Dict[str, Any]:
    normalized_query = normalize_text(query)
    query_tokens = _tokenize_normalized(normalized_query)
    topic_hits = _extract_query_topics(normalized_query, query_tokens)
    levels = _extract_query_levels(normalized_query, query_tokens)
    faculties = _extract_query_faculties(normalized_query)
    services = _extract_query_services(normalized_query, query_tokens)
    intents = _extract_query_intents(normalized_query, query_tokens)
    years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", normalized_query)})
    informative_tokens = sorted(
        token for token in query_tokens if token and len(token) >= 3 and token not in QUERY_STOPWORDS and not token.isdigit()
    )
    return {
        "normalized_query": normalized_query,
        "query_tokens": sorted(query_tokens),
        "informative_tokens": informative_tokens,
        "topic_hits": topic_hits,
        "primary_topics": sorted(topic_hits.keys()),
        "levels": levels,
        "faculties": faculties,
        "services": services,
        "intents": intents,
        "years": years,
        "has_strong_topic": bool(topic_hits or services),
    }


def _chunk_haystack(chunk: Dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    fields = [
        chunk.get("text", ""),
        metadata.get("source", ""),
        metadata.get("file_name", ""),
        metadata.get("document_type", ""),
        metadata.get("service_name", ""),
        metadata.get("service_type", ""),
        metadata.get("official_url", ""),
        metadata.get("page_kind", ""),
        " ".join(metadata.get("retrieval_keywords", []) or []),
        metadata.get("retrieval_haystack", ""),
    ]
    return normalize_text(" ".join(str(field or "") for field in fields))


def _chunk_topics(chunk: Dict, haystack: str) -> Dict[str, List[str]]:
    chunk_tokens = _tokenize_normalized(haystack)
    hits: Dict[str, List[str]] = {}
    for topic, config in NORMALIZED_QUERY_TOPIC_RULES.items():
        topic_hits: List[str] = []
        for keyword in config["keywords"]:
            if not keyword:
                continue
            if " " in keyword and keyword in haystack:
                topic_hits.append(keyword)
                continue
            keyword_tokens = _tokenize_normalized(keyword)
            if keyword_tokens and keyword_tokens.issubset(chunk_tokens):
                topic_hits.append(keyword)
        if topic_hits:
            hits[topic] = sorted(set(topic_hits))
    return hits


def _chunk_levels(haystack: str) -> List[str]:
    chunk_tokens = _tokenize_normalized(haystack)
    levels: List[str] = []
    for level, keywords in NORMALIZED_LEVEL_KEYWORDS.items():
        for keyword in keywords:
            if (" " in keyword and keyword in haystack) or _tokenize_normalized(keyword).issubset(chunk_tokens):
                levels.append(level)
                break
    return levels


def _normalize_rerank_score(raw: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(raw) / 4.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_chunk_thematic_match(chunk: Dict, query_profile: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {}) or {}
    haystack = _chunk_haystack(chunk)
    chunk_topics = _chunk_topics(chunk, haystack)
    chunk_tokens = _tokenize_normalized(haystack)
    text_tokens = _tokenize_normalized(str(chunk.get("text") or ""))
    metadata_phrases = [
        normalize_text(str(item))
        for item in [*(metadata.get("retrieval_keywords", []) or []), *(metadata.get("main_actions", []) or [])]
        if normalize_text(str(item))
    ]
    metadata_tokens: Set[str] = set()
    for phrase in metadata_phrases:
        metadata_tokens.update(_tokenize_normalized(phrase))
    chunk_levels = _chunk_levels(haystack)
    primary_topics = set(query_profile.get("primary_topics", []))
    levels = set(query_profile.get("levels", []))
    faculties = set(query_profile.get("faculties", []))
    query_services = set(query_profile.get("services", []))
    query_intents = set(query_profile.get("intents", []))
    years = {int(year) for year in query_profile.get("years", [])}
    informative_tokens = set(query_profile.get("informative_tokens", []))
    topic_query_hits = query_profile.get("topic_hits", {}) or {}
    metadata_doc_type = normalize_text(str(metadata.get("document_type") or ""))
    metadata_service_name = normalize_text(str(metadata.get("service_name") or ""))
    metadata_official_url = normalize_text(str(metadata.get("official_url") or ""))
    metadata_intents = {normalize_text(str(item)) for item in metadata.get("intent", []) or [] if normalize_text(str(item))}
    metadata_faculty = str(metadata.get("faculty") or "").strip().upper()
    metadata_year = metadata.get("year")
    chunk_services = {
        service
        for service, aliases in NORMALIZED_SERVICE_ALIAS_RULES.items()
        if service == metadata_service_name
        or _alias_matches_text(service, haystack, chunk_tokens)
        or any(alias and (_alias_matches_text(alias, haystack, chunk_tokens) or alias in metadata_official_url) for alias in aliases)
    }
    matched_services = sorted(query_services.intersection(chunk_services))
    matched_intents = sorted(query_intents.intersection(metadata_intents))
    matched_text_intents = sorted(
        intent
        for intent in query_intents
        if any(token == intent or token.startswith(intent) or intent.startswith(token) for token in text_tokens)
    )
    matched_topics = sorted(primary_topics.intersection(chunk_topics.keys()))
    anchor_topic_hits: Dict[str, List[str]] = {}
    for topic in matched_topics:
        exact_hits = sorted(set(topic_query_hits.get(topic, [])).intersection(chunk_topics.get(topic, [])))
        if not exact_hits:
            exact_hits = sorted({hit for hit in topic_query_hits.get(topic, []) if any(hit in phrase or phrase in hit for phrase in metadata_phrases)})
        if exact_hits:
            anchor_topic_hits[topic] = exact_hits
    conflicting_topics: Set[str] = set()
    for topic in primary_topics:
        conflicts = NORMALIZED_QUERY_TOPIC_RULES.get(topic, {}).get("conflicts", set())
        conflicting_topics.update(conflicts.intersection(chunk_topics.keys()))
    allowed_document_types: Set[str] = set()
    for topic in primary_topics:
        allowed_document_types.update(NORMALIZED_QUERY_TOPIC_RULES.get(topic, {}).get("allowed_document_types", set()))
    doc_type_match = bool(metadata_doc_type and metadata_doc_type in allowed_document_types)
    matched_text_informative_tokens = sorted(informative_tokens.intersection(text_tokens))
    matched_metadata_informative_tokens = sorted(informative_tokens.intersection(metadata_tokens))
    matched_informative_tokens = sorted(set(matched_text_informative_tokens).union(matched_metadata_informative_tokens))
    text_informative_coverage = (
        float(len(matched_text_informative_tokens)) / float(len(informative_tokens))
        if informative_tokens
        else 0.0
    )
    metadata_informative_coverage = (
        float(len(matched_metadata_informative_tokens)) / float(len(informative_tokens))
        if informative_tokens
        else 0.0
    )
    informative_coverage = max(text_informative_coverage, metadata_informative_coverage * 0.45)
    score = 0.35 if not primary_topics else 0.0
    reasons: List[str] = []
    if anchor_topic_hits:
        score += 0.48 + (0.06 * min(2, len(anchor_topic_hits) - 1))
        reasons.append("topic_anchor_match")
    elif matched_topics:
        score += 0.24 + (0.05 * min(2, len(matched_topics) - 1))
        reasons.append("topic_partial_match")
    elif doc_type_match:
        score += 0.18
        reasons.append("doc_type_match")
    elif primary_topics:
        reasons.append("topic_missing")
    if primary_topics and allowed_document_types and not doc_type_match:
        score -= 0.14
        reasons.append("doc_type_mismatch")
    if matched_services:
        score += 0.42
        reasons.append("service_match")
    elif query_services:
        score -= 0.28
        reasons.append("service_mismatch")
    is_diploma_tracking_query = "diplome" in primary_topics and bool({"suivi", "etat", "statut", "avancement"}.intersection(informative_tokens))
    if is_diploma_tracking_query:
        if metadata_service_name == normalize_text("Espace DiplÃ´mes"):
            score += 0.22
            reasons.append("diploma_tracking_service_boost")
        elif metadata_service_name == normalize_text("UC@Student"):
            score -= 0.1
            reasons.append("diploma_tracking_service_penalty")
    is_research_support_query = "soutien_recherche" in primary_topics and bool({"soutien", "accompagnement"}.intersection(informative_tokens))
    if is_research_support_query:
        if metadata_service_name == normalize_text("Soutien-Recherche"):
            score += 0.24
            reasons.append("research_support_service_boost")
        elif metadata_service_name == normalize_text("Appels Ã  Projets"):
            score -= 0.12
            reasons.append("research_support_service_penalty")
    if matched_intents:
        score += 0.2 + (0.04 * min(2, len(matched_intents) - 1))
        reasons.append("intent_match")
        if matched_text_intents:
            score += 0.14
            reasons.append("intent_text_match")
    elif query_intents:
        score -= 0.18
        reasons.append("intent_missing")
    if doc_type_match and (matched_topics or anchor_topic_hits):
        score += 0.08
    if levels:
        matched_levels = sorted(levels.intersection(chunk_levels))
        if matched_levels:
            score += 0.12
            reasons.append("level_match")
        else:
            score -= 0.1
            reasons.append("level_missing")
    else:
        matched_levels = []
    faculty_match = True
    if faculties:
        if metadata_faculty and metadata_faculty in faculties:
            score += 0.12
            reasons.append("faculty_match")
        elif metadata_faculty and metadata_faculty != "UNKNOWN":
            score -= 0.24
            reasons.append("faculty_mismatch")
            faculty_match = False
        else:
            score -= 0.05
            reasons.append("faculty_unknown")
    year_match = True
    if years:
        if isinstance(metadata_year, int) and metadata_year in years:
            score += 0.08
            reasons.append("year_match")
        elif isinstance(metadata_year, int):
            score -= 0.1
            reasons.append("year_mismatch")
            year_match = False
    if conflicting_topics:
        score -= min(0.55, 0.22 * len(conflicting_topics))
        reasons.append("topic_conflict")
    if informative_tokens:
        if informative_coverage >= 0.5:
            score += 0.18
            reasons.append("query_coverage_high")
        elif informative_coverage >= 0.25:
            score += 0.08
            reasons.append("query_coverage_medium")
        elif primary_topics:
            score -= 0.16
            reasons.append("query_coverage_low")
    if (matched_topics or anchor_topic_hits) and not conflicting_topics and metadata.get("chunk_relevance_score", 0) >= 2:
        score += 0.06
    thematic_score = _clamp01(score)
    return {
        "thematic_score": thematic_score,
        "matched_topics": matched_topics,
        "matched_services": matched_services,
        "matched_intents": matched_intents,
        "matched_text_intents": matched_text_intents,
        "anchor_topic_hits": anchor_topic_hits,
        "matched_informative_tokens": matched_informative_tokens,
        "matched_text_informative_tokens": matched_text_informative_tokens,
        "matched_metadata_informative_tokens": matched_metadata_informative_tokens,
        "informative_coverage": round(informative_coverage, 4),
        "text_informative_coverage": round(text_informative_coverage, 4),
        "metadata_informative_coverage": round(metadata_informative_coverage, 4),
        "conflicting_topics": sorted(conflicting_topics),
        "matched_levels": matched_levels,
        "chunk_topics": sorted(chunk_topics.keys()),
        "doc_type_match": doc_type_match,
        "faculty_match": faculty_match,
        "year_match": year_match,
        "reasons": reasons,
    }


def apply_retrieval_guardrails(query: str, results: List[Dict], top_k: int, top_k_retrieve: int, min_thematic_score: float, min_support_score: float, min_final_support_score: float, min_top_rerank_normalized: float, topical_mismatch_drop_threshold: float) -> Tuple[List[Dict], Dict[str, Any]]:
    query_profile = build_query_profile(query)
    guarded: List[Dict] = []
    rejected: List[Dict] = []
    for result in results:
        enriched = dict(result)
        thematic = score_chunk_thematic_match(enriched, query_profile)
        base_score = float(enriched.get("score", 0.0) or 0.0)
        support_score = _clamp01((base_score * 0.62) + (float(thematic["thematic_score"]) * 0.38))
        enriched.update(thematic)
        enriched["guardrail_base_score"] = round(base_score, 4)
        enriched["support_score"] = round(support_score, 4)
        should_drop = False
        drop_reasons: List[str] = []
        if query_profile["has_strong_topic"] and float(thematic["thematic_score"]) < min_thematic_score:
            should_drop = True
            drop_reasons.append("thematic_score_below_min")
        if support_score < min_support_score:
            should_drop = True
            drop_reasons.append("support_score_below_min")
        if query_profile["has_strong_topic"] and not thematic.get("anchor_topic_hits") and float(thematic.get("informative_coverage", 0.0) or 0.0) < 0.2:
            should_drop = True
            drop_reasons.append("anchor_topic_missing")
        if thematic["conflicting_topics"] and float(thematic["thematic_score"]) <= topical_mismatch_drop_threshold:
            should_drop = True
            drop_reasons.append("topic_conflict")
        if not thematic["faculty_match"] or not thematic["year_match"]:
            should_drop = True
            if not thematic["faculty_match"]:
                drop_reasons.append("faculty_mismatch")
            if not thematic["year_match"]:
                drop_reasons.append("year_mismatch")
        enriched["guardrail_drop_reasons"] = drop_reasons
        if should_drop:
            rejected.append(enriched)
            continue
        guarded.append(enriched)
    guarded.sort(key=lambda item: (float(item.get("support_score", 0.0)), float(item.get("score", 0.0)), float(item.get("dense_score", 0.0)), float(item.get("bm25_score", 0.0))), reverse=True)
    rejected.sort(key=lambda item: float(item.get("support_score", 0.0)), reverse=True)
    diagnostics = {
        "query_profile": query_profile,
        "guarded_count": len(guarded),
        "rejected_count": len(rejected),
        "rejection_reasons_top": [item.get("reasons", []) for item in rejected[:5]],
        "guardrail_drop_reasons_top": [item.get("guardrail_drop_reasons", []) for item in rejected[:5]],
        "top_k_requested": top_k,
        "thresholds": {
            "min_thematic_score": min_thematic_score,
            "min_support_score": min_support_score,
            "min_final_support_score": min_final_support_score,
            "min_top_rerank_normalized": min_top_rerank_normalized,
            "topical_mismatch_drop_threshold": topical_mismatch_drop_threshold,
        },
    }
    return guarded[: max(top_k_retrieve, top_k * 4)], diagnostics


def apply_post_rerank_guardrails(results: List[Dict], query_profile: Dict[str, Any], top_k: int, min_thematic_score: float, min_final_support_score: float) -> List[Dict]:
    filtered: List[Dict] = []
    for result in results:
        enriched = dict(result)
        rerank_score = float(enriched.get("rerank_score", 0.0) or 0.0)
        rerank_normalized = _normalize_rerank_score(rerank_score) if "rerank_score" in enriched else float(enriched.get("score", 0.0) or 0.0)
        thematic_score = float(enriched.get("thematic_score", 0.0) or 0.0)
        hybrid_score = float(enriched.get("score", 0.0) or 0.0)
        final_support = _clamp01((rerank_normalized * 0.56) + (hybrid_score * 0.22) + (thematic_score * 0.22))
        enriched["rerank_score_normalized"] = round(rerank_normalized, 4)
        enriched["final_support_score"] = round(final_support, 4)
        if query_profile.get("has_strong_topic") and thematic_score < min_thematic_score:
            continue
        if final_support < min_final_support_score:
            continue
        filtered.append(enriched)
    filtered.sort(key=lambda item: (float(item.get("final_support_score", 0.0)), float(item.get("rerank_score_normalized", 0.0)), float(item.get("support_score", 0.0))), reverse=True)
    return filtered[:top_k]


def decide_retrieval_abstention(results: List[Dict], query_profile: Dict[str, Any], min_thematic_score: float, min_final_support_score: float, min_top_rerank_normalized: float) -> Dict[str, Any]:
    if not results:
        return {"abstain": True, "reason": "no_supported_chunks"}
    top = results[0]
    top_final_support = float(top.get("final_support_score", top.get("support_score", 0.0)) or 0.0)
    top_rerank_normalized = float(top.get("rerank_score_normalized", top.get("score", 0.0)) or 0.0)
    top_thematic = float(top.get("thematic_score", 0.0) or 0.0)
    top_matched_intents = set(top.get("matched_intents", []) or [])
    required_intents = set(query_profile.get("intents", []) or []).intersection(STRICT_QUERY_INTENTS)
    if query_profile.get("has_strong_topic") and top_thematic < min_thematic_score:
        return {"abstain": True, "reason": "top_chunk_thematically_weak"}
    if required_intents and not top_matched_intents.intersection(required_intents):
        return {"abstain": True, "reason": "top_chunk_missing_required_intent"}
    if top_final_support < min_final_support_score:
        return {"abstain": True, "reason": "top_chunk_support_too_low"}
    if top_rerank_normalized < min_top_rerank_normalized:
        return {"abstain": True, "reason": "top_rerank_too_low"}
    conflict_count = sum(1 for item in results[:3] if item.get("conflicting_topics"))
    if query_profile.get("has_strong_topic") and conflict_count >= 2:
        return {"abstain": True, "reason": "top_results_thematically_incoherent"}
    return {"abstain": False, "reason": ""}
