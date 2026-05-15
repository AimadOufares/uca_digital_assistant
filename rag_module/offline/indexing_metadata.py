import hashlib
import re
from typing import Dict, List, Set

try:
    from ..shared.metadata_policy import normalize_text
except ImportError:  # pragma: no cover
    from rag_module.shared.metadata_policy import normalize_text


BLOCKED_QUALITY_ISSUES = {
    "empty_after_static",
    "empty_after_playwright",
    "login_wall",
    "too_generic",
    "encoding_suspect",
}
HIGH_VALUE_INTENTS = {
    "connexion",
    "mot_de_passe",
    "attestation",
    "notes",
    "reinscription",
    "candidature",
    "cours",
    "depot_document",
}
ACTIONABLE_PAGE_KINDS = {"guide", "procedure", "faq", "formulaire"}
LANDING_PAGE_KINDS = {"landing"}
CANONICAL_STOPWORDS = {
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "et",
    "ou",
    "un",
    "une",
    "pour",
    "dans",
    "avec",
    "sur",
    "par",
    "est",
    "en",
    "au",
    "aux",
}


def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_int_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        values = list(raw_value)
    elif raw_value in (None, ""):
        values = []
    else:
        values = [raw_value]
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if isinstance(value, list):
            for nested in value:
                item = normalize_text(str(nested or ""))
                if item and item not in seen:
                    seen.add(item)
                    normalized.append(item)
            continue
        item = normalize_text(str(value or ""))
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def canonical_tokens(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"\b[\w'-]+\b", normalized.lower(), flags=re.UNICODE)
    filtered = [
        token
        for token in tokens
        if len(token) > 2 and token not in CANONICAL_STOPWORDS and not token.isdigit()
    ]
    return filtered or tokens


def near_duplicate_signature(text: str) -> str:
    tokens = canonical_tokens(text)
    if not tokens:
        return ""
    if len(tokens) <= 18:
        basis = " ".join(tokens)
    else:
        basis = " ".join(tokens[:12] + tokens[-8:])
    return get_hash(basis)


def compute_student_relevance(metadata: Dict, text: str) -> float:
    intents = normalize_int_list(metadata.get("intent"))
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    service_name = normalize_text(str(metadata.get("service_name") or ""))
    service_type = normalize_text(str(metadata.get("service_type") or metadata.get("document_type") or ""))
    document_category = normalize_text(str(metadata.get("document_category") or ""))
    source_priority = str(metadata.get("source_priority") or "").strip().upper()
    chunk_relevance_score = float(metadata.get("chunk_relevance_score", 0.0) or 0.0)
    source_relevance_score = float(metadata.get("source_relevance_score", 0.0) or 0.0)
    freshness_score = float(metadata.get("freshness_score", 0.0) or 0.0)
    quality_issue = normalize_text(str(metadata.get("quality_issue") or ""))

    score = 0.0
    score += min(0.25, 0.05 * len(intents))
    if any(intent in HIGH_VALUE_INTENTS for intent in intents):
        score += 0.15
    if page_kind in ACTIONABLE_PAGE_KINDS:
        score += 0.15
    if service_type in {"digital_service", "pedagogie_numerique", "scolarite"}:
        score += 0.12
    if document_category in {"digital_service", "vie_etudiante", "scolarite"}:
        score += 0.08
    if service_name in {"ucastudent", "ucaplat", "pedoc", "cip", "e-candidature", "espace diplomes"}:
        score += 0.12
    if source_priority == "A":
        score += 0.08
    elif source_priority == "B":
        score += 0.04
    score += min(0.12, chunk_relevance_score * 0.03)
    score += min(0.08, source_relevance_score * 0.015)
    score += min(0.05, freshness_score * 0.05)
    if quality_issue:
        score -= 0.2
    if len(text.split()) >= 45:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 3)


def is_actionable_chunk(metadata: Dict) -> bool:
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    intents = normalize_int_list(metadata.get("intent"))
    if page_kind in ACTIONABLE_PAGE_KINDS:
        return True
    return any(intent in HIGH_VALUE_INTENTS for intent in intents)


def build_retrieval_keywords(metadata: Dict) -> List[str]:
    fields = [
        metadata.get("service_name"),
        metadata.get("service_type"),
        metadata.get("document_type"),
        metadata.get("document_category"),
        metadata.get("page_kind"),
        metadata.get("intent", []),
        metadata.get("main_actions", []),
        metadata.get("workflow_steps", []),
        metadata.get("official_url"),
    ]
    keywords = normalize_int_list(fields)
    return keywords[:12]


def build_retrieval_haystack(text: str, metadata: Dict) -> str:
    additions = [
        str(metadata.get("service_name") or ""),
        str(metadata.get("service_type") or ""),
        str(metadata.get("document_type") or ""),
        str(metadata.get("page_kind") or ""),
        " ".join(normalize_int_list(metadata.get("intent"))),
        " ".join(normalize_int_list(metadata.get("main_actions"))),
        " ".join(normalize_int_list(metadata.get("workflow_steps"))),
        str(metadata.get("official_url") or ""),
    ]
    prefix = normalize(" ".join(part for part in additions if part))
    if not prefix:
        return text
    return normalize(f"{prefix}\n{text}")


def should_index_chunk(text: str, metadata: Dict) -> bool:
    quality_issue = normalize_text(str(metadata.get("quality_issue") or ""))
    page_kind = normalize_text(str(metadata.get("page_kind") or ""))
    student_relevance_score = float(metadata.get("student_relevance_score", 0.0) or 0.0)
    chunk_relevance_score = float(metadata.get("chunk_relevance_score", 0.0) or 0.0)
    intents = normalize_int_list(metadata.get("intent"))

    if len(text) < 60:
        return False
    if quality_issue in BLOCKED_QUALITY_ISSUES:
        return False
    if page_kind in LANDING_PAGE_KINDS and not intents and student_relevance_score < 0.45:
        return False
    if chunk_relevance_score <= 0 and student_relevance_score < 0.35:
        return False
    return True


def enrich_index_metadata(text: str, metadata: Dict, corpus_name: str) -> Dict:
    enriched = dict(metadata)
    intents = normalize_int_list(enriched.get("intent"))
    enriched["intent"] = intents
    enriched["corpus"] = corpus_name
    enriched["service_priority"] = str(enriched.get("source_priority") or "unknown").strip().upper() or "unknown"
    enriched["is_actionable"] = is_actionable_chunk(enriched)
    enriched["student_relevance_score"] = compute_student_relevance(enriched, text)
    enriched["retrieval_keywords"] = build_retrieval_keywords(enriched)
    enriched["retrieval_haystack"] = build_retrieval_haystack(text, enriched)
    return enriched
