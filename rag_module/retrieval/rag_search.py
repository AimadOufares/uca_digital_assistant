import hashlib
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import unidecode
from sentence_transformers.cross_encoder import CrossEncoder

try:
    from ..shared.context_resolution import (
        CANONICAL_ESTABLISHMENTS,
        UCA_GLOBAL,
        detect_establishments_in_text,
        get_metadata_establishment,
        normalize_establishment_label,
    )
    from ..shared.env_loader import load_env_file
    from ..shared.metadata_policy import normalize_text
    from ..shared.relevance_policy import boost_results_with_metadata
except ImportError:  # pragma: no cover
    from rag_module.shared.context_resolution import (
        CANONICAL_ESTABLISHMENTS,
        UCA_GLOBAL,
        detect_establishments_in_text,
        get_metadata_establishment,
        normalize_establishment_label,
    )
    from rag_module.shared.env_loader import load_env_file
    from rag_module.shared.metadata_policy import normalize_text
    from rag_module.shared.relevance_policy import boost_results_with_metadata

load_env_file()


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
MAX_CONTEXT_CHARS = 2500
DENSE_WEIGHT = 0.65
SPARSE_WEIGHT = 0.35

USE_RERANK = True
USE_SPELLCHECK = False
USE_MULTI_QUERY = True
USE_ASCII_NORMALIZATION = False

MIN_GUARDRAIL_SCORE = 0.24
MIN_THEMATIC_SCORE = 0.18
MIN_SUPPORT_SCORE = 0.28
MIN_FINAL_SUPPORT_SCORE = 0.42
MIN_TOP_RERANK_NORMALIZED = 0.44
TOPICAL_MISMATCH_DROP_THRESHOLD = 0.45
RERANK_FALLBACK_SUPPORT_SCORE = 0.62
RERANK_FALLBACK_THEMATIC_SCORE = 0.20

QUERY_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "comment",
    "dans",
    "de",
    "des",
    "du",
    "en",
    "est",
    "et",
    "faire",
    "la",
    "le",
    "les",
    "ma",
    "mes",
    "mon",
    "ou",
    "pour",
    "quelles",
    "quelle",
    "quel",
    "quels",
    "qui",
    "sur",
    "un",
    "une",
    "vos",
    "votre",
}

QUERY_TOPIC_RULES: Dict[str, Dict[str, Any]] = {
    "stage": {
        "keywords": {
            "stage",
            "stages",
            "pfe",
            "projet de fin d etudes",
            "projet de fin d'etudes",
            "projet fin d etudes",
            "convention de stage",
            "memoire",
            "internship",
            "internships",
        },
        "allowed_document_types": {"stage", "formation", "general"},
        "conflicts": {"bourse", "calendrier", "resultats"},
    },
    "inscription": {
        "keywords": {
            "inscription",
            "preinscription",
            "pre inscription",
            "reinscription",
            "inscription administrative",
            "scolarite",
            "registration",
        },
        "allowed_document_types": {"inscription", "admission", "general"},
        "conflicts": {"bourse", "resultats"},
    },
    "admission": {
        "keywords": {
            "admission",
            "admissions",
            "candidature",
            "selection",
            "concours",
            "appel a candidature",
            "appel a candidatures",
            "application",
        },
        "allowed_document_types": {"admission", "inscription", "formation", "general"},
        "conflicts": {"bourse", "stage"},
    },
    "bourse": {
        "keywords": {
            "bourse",
            "bourses",
            "scholarship",
            "scholarships",
            "allocation",
            "aide financiere",
        },
        "allowed_document_types": {"bourse", "general"},
        "conflicts": {"stage", "admission", "inscription", "resultats"},
    },
    "calendrier": {
        "keywords": {
            "calendrier",
            "planning",
            "emploi du temps",
            "date",
            "dates",
            "delai",
            "delais",
            "deadline",
            "schedule",
        },
        "allowed_document_types": {"calendrier", "resultats", "inscription", "general"},
        "conflicts": {"bourse"},
    },
    "resultats": {
        "keywords": {
            "resultat",
            "resultats",
            "note",
            "notes",
            "deliberation",
            "rattrapage",
            "classement",
        },
        "allowed_document_types": {"resultats", "admission", "general"},
        "conflicts": {"bourse", "stage"},
    },
    "formation": {
        "keywords": {
            "formation",
            "formations",
            "filiere",
            "filiere",
            "programme",
            "master",
            "licence",
            "doctorat",
            "doctorale",
            "doctorat",
            "module",
            "modules",
            "cours",
        },
        "allowed_document_types": {"formation", "admission", "general"},
        "conflicts": set(),
    },
    "contact": {
        "keywords": {
            "contact",
            "contacts",
            "telephone",
            "email",
            "mail",
            "adresse",
            "service",
            "scolarite",
        },
        "allowed_document_types": {"contact", "inscription", "general"},
        "conflicts": set(),
    },
    "reglement": {
        "keywords": {
            "reglement",
            "reglements",
            "reglement pedagogique",
            "reglements pedagogiques",
            "lmd",
            "ects",
            "modalite",
            "modalites",
        },
        "allowed_document_types": {"reglement", "formation", "general"},
        "conflicts": set(),
    },
}

LEVEL_KEYWORDS = {
    "master": {"master", "masters", "mastère"},
    "licence": {"licence", "licences", "license"},
    "doctorat": {"doctorat", "doctorale", "doctorales", "phd", "these", "theses"},
}

NORMALIZED_QUERY_TOPIC_RULES: Dict[str, Dict[str, Any]] = {
    topic: {
        "keywords": {normalize_text(keyword) for keyword in config.get("keywords", set()) if normalize_text(keyword)},
        "allowed_document_types": {
            normalize_text(doc_type) for doc_type in config.get("allowed_document_types", set()) if normalize_text(doc_type)
        },
        "conflicts": {normalize_text(topic_name) for topic_name in config.get("conflicts", set()) if normalize_text(topic_name)},
    }
    for topic, config in QUERY_TOPIC_RULES.items()
}

NORMALIZED_LEVEL_KEYWORDS = {
    level: {normalize_text(keyword) for keyword in keywords if normalize_text(keyword)}
    for level, keywords in LEVEL_KEYWORDS.items()
}
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_reranker = None


def invalidate_search_cache(clear_models: bool = False) -> None:
    global _reranker

    try:
        from .qdrant_search import _embed_query  # type: ignore

        _embed_query.cache_clear()
    except Exception:
        pass
    if clear_models:
        _reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANK:
        logger.info("Chargement du reranker: %s", RERANK_MODEL)
        try:
            _reranker = CrossEncoder(
                RERANK_MODEL,
                device="cpu",
                local_files_only=True,
            )
        except TypeError:
            try:
                _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            except Exception as exc:
                logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
                _reranker = None
        except Exception as exc:
            logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
            _reranker = None
    return _reranker


def preprocess_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    if USE_ASCII_NORMALIZATION:
        query = unidecode.unidecode(query)
    return query


def correct_query(query: str) -> str:
    if not USE_SPELLCHECK:
        return query
    try:
        from spellchecker import SpellChecker

        spell = SpellChecker(language="fr")
        words = query.split()
        corrected = [spell.correction(word) or word for word in words]
        return " ".join(corrected)
    except Exception:
        return query


def enhance_query(query: str) -> str:
    return correct_query(preprocess_query(query))


def generate_multi_queries(query: str) -> List[str]:
    if not USE_MULTI_QUERY:
        return [query]

    base = enhance_query(query)
    variations = [
        base,
        base + " explication detaillee",
        base + " informations importantes",
        "comment " + base if not base.startswith(("comment", "comment faire")) else base,
    ]
    return list(dict.fromkeys(variations))


def merge_dense_and_sparse(dense_results: List[Dict], sparse_results: List[Dict], top_k: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}

    for result in dense_results:
        metadata = result.get("metadata", {}) or {}
        chunk_id = result.get("id") or metadata.get("chunk_hash") or metadata.get("hash")
        if not chunk_id:
            continue

        entry = merged.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "text": result.get("text", ""),
                "metadata": metadata,
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "score": 0.0,
                "score_type": "hybrid",
                "retrieval_sources": [],
            },
        )
        entry["dense_score"] = max(float(entry["dense_score"]), float(result.get("score", 0.0) or 0.0))
        if "dense" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("dense")

    for result in sparse_results:
        metadata = result.get("metadata", {}) or {}
        chunk_id = result.get("id") or metadata.get("chunk_hash") or metadata.get("hash")
        if not chunk_id:
            continue

        entry = merged.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "text": result.get("text", ""),
                "metadata": metadata,
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "score": 0.0,
                "score_type": "hybrid",
                "retrieval_sources": [],
            },
        )
        entry["sparse_score"] = max(float(entry["sparse_score"]), float(result.get("score", 0.0) or 0.0))
        if "sparse" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("sparse")

    merged_results: List[Dict] = []
    for entry in merged.values():
        dense_score = float(entry.get("dense_score", 0.0))
        sparse_score = float(entry.get("sparse_score", 0.0))
        entry["score"] = max(0.0, min(1.0, (dense_score * DENSE_WEIGHT) + (sparse_score * SPARSE_WEIGHT)))
        merged_results.append(entry)

    merged_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return merged_results[: max(top_k, TOP_K_RETRIEVE)]


def deduplicate_chunks(chunks_list: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for chunk in chunks_list:
        metadata = chunk.get("metadata", {}) or {}
        text = (chunk.get("text", "") or "").strip().lower()
        text_fallback = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else None
        chunk_id = chunk.get("id") or metadata.get("chunk_hash") or metadata.get("hash") or text_fallback
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk)
    return unique


def _normalize_allowed_establishments(allowed_establishments: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for item in allowed_establishments or []:
        canonical = normalize_establishment_label(item)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _filter_results_by_allowed_establishments(
    results: List[Dict],
    allowed_establishments: Optional[List[str]],
) -> List[Dict]:
    allowed = _normalize_allowed_establishments(allowed_establishments)
    if not allowed:
        return list(results)

    allowed_set = set(allowed)
    filtered: List[Dict] = []
    for result in results:
        metadata = result.get("metadata", {}) or {}
        chunk_establishment = get_metadata_establishment(metadata)
        if chunk_establishment in allowed_set:
            filtered.append(result)
    return filtered


def apply_metadata_boost(results: List[Dict], query: str) -> List[Dict]:
    return boost_results_with_metadata(results, query)


def rerank_chunks(query: str, chunks_list: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not USE_RERANK or not chunks_list:
        return chunks_list[:top_k]

    reranker = get_reranker()
    if reranker is None:
        return chunks_list[:top_k]

    pairs = [(query, chunk.get("text", "")) for chunk in chunks_list]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        logger.warning("Reranking indisponible, fallback sans rerank: %s", exc)
        return chunks_list[:top_k]

    ranked = sorted(zip(chunks_list, scores), key=lambda item: item[1], reverse=True)
    selected = []
    for chunk, rerank_score in ranked[:top_k]:
        enriched = dict(chunk)
        enriched["rerank_score"] = float(rerank_score)
        enriched["score_type"] = "rerank"
        selected.append(enriched)
    return selected


def truncate_chunks(chunks_list: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> List[Dict]:
    total = 0
    selected = []
    for chunk in chunks_list:
        text = chunk.get("text", "")
        if total + len(text) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(text)
    return selected


def _normalize_result_row(result: Dict) -> Dict:
    entry = dict(result or {})
    metadata = entry.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    entry["metadata"] = metadata
    entry["id"] = entry.get("id") or metadata.get("chunk_hash") or metadata.get("hash") or ""
    entry["text"] = str(entry.get("text", "") or "")
    entry["score"] = float(entry.get("score", 0.0) or 0.0)
    entry["dense_score"] = float(entry.get("dense_score", 0.0) or 0.0)
    entry["sparse_score"] = float(entry.get("sparse_score", 0.0) or 0.0)
    if "support_score" in entry:
        entry["support_score"] = float(entry.get("support_score", 0.0) or 0.0)
    if "final_support_score" in entry:
        entry["final_support_score"] = float(entry.get("final_support_score", 0.0) or 0.0)
    if "rerank_score" in entry:
        entry["rerank_score"] = float(entry.get("rerank_score", 0.0) or 0.0)
    if "rerank_score_normalized" in entry:
        entry["rerank_score_normalized"] = float(entry.get("rerank_score_normalized", 0.0) or 0.0)
    if "thematic_score" in entry:
        entry["thematic_score"] = float(entry.get("thematic_score", 0.0) or 0.0)
    entry["score_type"] = str(entry.get("score_type") or "unknown")
    entry["retrieval_sources"] = list(dict.fromkeys(entry.get("retrieval_sources", []) or []))
    entry["rejection_reasons"] = list(dict.fromkeys(entry.get("rejection_reasons") or entry.get("reasons") or []))
    return entry


def _normalize_result_rows(results: List[Dict]) -> List[Dict]:
    return [_normalize_result_row(item) for item in results]


def _summarize_results(results: List[Dict], top_n: int = 3) -> Dict[str, Any]:
    rows = _normalize_result_rows(results)
    sources = sorted(
        {
            source
            for row in rows
            for source in row.get("retrieval_sources", [])
            if isinstance(source, str) and source.strip()
        }
    )
    return {
        "count": len(rows),
        "best_score": round(max((float(row.get("score", 0.0) or 0.0) for row in rows), default=0.0), 4),
        "top_ids": [row.get("id", "") for row in rows[:top_n]],
        "top_scores": [round(float(row.get("score", 0.0) or 0.0), 4) for row in rows[:top_n]],
        "sources": sources,
    }


def _bucket_rejection_reasons(results: List[Dict]) -> Dict[str, int]:
    buckets: Dict[str, int] = {}
    for row in results:
        reasons = list(row.get("rejection_reasons") or row.get("reasons") or [])
        if not reasons:
            buckets["unknown"] = buckets.get("unknown", 0) + 1
            continue
        for reason in reasons:
            key = str(reason or "").strip() or "unknown"
            buckets[key] = buckets.get(key, 0) + 1
    return dict(sorted(buckets.items(), key=lambda item: (-item[1], item[0])))


def _top_result_snapshot(results: List[Dict]) -> Dict[str, Any]:
    rows = _normalize_result_rows(results)
    if not rows:
        return {}
    top = rows[0]
    metadata = top.get("metadata", {}) or {}
    return {
        "id": top.get("id", ""),
        "score": round(float(top.get("score", 0.0) or 0.0), 4),
        "support_score": round(float(top.get("support_score", 0.0) or 0.0), 4) if "support_score" in top else 0.0,
        "final_support_score": round(float(top.get("final_support_score", 0.0) or 0.0), 4) if "final_support_score" in top else 0.0,
        "score_type": str(top.get("score_type") or ""),
        "document_type": str(metadata.get("document_type") or ""),
        "establishment": get_metadata_establishment(metadata),
        "source": str(metadata.get("source") or metadata.get("file_name") or ""),
    }


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
    return [item for item in detect_establishments_in_text(normalized_query) if item != UCA_GLOBAL]


def build_query_profile(query: str, allowed_establishments: Optional[List[str]] = None) -> Dict[str, Any]:
    normalized_query = normalize_text(query)
    query_tokens = _tokenize_normalized(normalized_query)
    topic_hits = _extract_query_topics(normalized_query, query_tokens)
    levels = _extract_query_levels(normalized_query, query_tokens)
    faculties = _extract_query_faculties(normalized_query)
    allowed_specific = [
        item
        for item in _normalize_allowed_establishments(allowed_establishments)
        if item != UCA_GLOBAL and item in CANONICAL_ESTABLISHMENTS
    ]
    for establishment in allowed_specific:
        if establishment not in faculties:
            faculties.append(establishment)
    years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", normalized_query)})
    informative_tokens = sorted(
        token
        for token in query_tokens
        if token
        and len(token) >= 3
        and token not in QUERY_STOPWORDS
        and not token.isdigit()
    )

    return {
        "normalized_query": normalized_query,
        "query_tokens": sorted(query_tokens),
        "informative_tokens": informative_tokens,
        "topic_hits": topic_hits,
        "primary_topics": sorted(topic_hits.keys()),
        "levels": levels,
        "faculties": faculties,
        "years": years,
        "has_strong_topic": bool(topic_hits),
    }


def _chunk_haystack(chunk: Dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    fields = [
        chunk.get("text", ""),
        metadata.get("section_title", ""),
        " ".join(str(part).strip() for part in metadata.get("section_path", []) if str(part).strip()),
        metadata.get("source", ""),
        metadata.get("file_name", ""),
        metadata.get("document_type", ""),
        metadata.get("year", ""),
        get_metadata_establishment(metadata),
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
    chunk_levels = _chunk_levels(haystack)

    primary_topics = set(query_profile.get("primary_topics", []))
    levels = set(query_profile.get("levels", []))
    faculties = set(query_profile.get("faculties", []))
    years = {int(year) for year in query_profile.get("years", [])}
    informative_tokens = set(query_profile.get("informative_tokens", []))
    topic_query_hits = query_profile.get("topic_hits", {}) or {}
    metadata_doc_type = normalize_text(str(metadata.get("document_type") or ""))
    metadata_faculty = get_metadata_establishment(metadata).strip().upper()
    metadata_year = metadata.get("year")

    matched_topics = sorted(primary_topics.intersection(chunk_topics.keys()))
    anchor_topic_hits: Dict[str, List[str]] = {}
    for topic in matched_topics:
        exact_hits = sorted(set(topic_query_hits.get(topic, [])).intersection(chunk_topics.get(topic, [])))
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
    matched_informative_tokens = sorted(informative_tokens.intersection(chunk_tokens))
    informative_coverage = (
        float(len(matched_informative_tokens)) / float(len(informative_tokens))
        if informative_tokens
        else 0.0
    )

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
        "anchor_topic_hits": anchor_topic_hits,
        "matched_informative_tokens": matched_informative_tokens,
        "informative_coverage": round(informative_coverage, 4),
        "conflicting_topics": sorted(conflicting_topics),
        "matched_levels": matched_levels,
        "chunk_topics": sorted(chunk_topics.keys()),
        "doc_type_match": doc_type_match,
        "faculty_match": faculty_match,
        "year_match": year_match,
        "reasons": reasons,
    }


def apply_retrieval_guardrails(
    query: str,
    results: List[Dict],
    top_k: int,
    allowed_establishments: Optional[List[str]] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    query_profile = build_query_profile(query, allowed_establishments=allowed_establishments)
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
        if query_profile["has_strong_topic"] and float(thematic["thematic_score"]) < MIN_THEMATIC_SCORE:
            should_drop = True
        if support_score < MIN_SUPPORT_SCORE:
            should_drop = True
        if query_profile["has_strong_topic"] and not thematic.get("anchor_topic_hits") and float(
            thematic.get("informative_coverage", 0.0) or 0.0
        ) < 0.2:
            should_drop = True
        if thematic["conflicting_topics"] and float(thematic["thematic_score"]) <= TOPICAL_MISMATCH_DROP_THRESHOLD:
            should_drop = True
        if not thematic["faculty_match"] or not thematic["year_match"]:
            should_drop = True

        if should_drop:
            enriched["rejected"] = True
            enriched["rejection_reasons"] = list(dict.fromkeys(enriched.get("reasons", []) or []))
            rejected.append(enriched)
            continue

        enriched["rejected"] = False
        enriched["rejection_reasons"] = []
        guarded.append(enriched)

    guarded.sort(
        key=lambda item: (
            float(item.get("support_score", 0.0)),
            float(item.get("score", 0.0)),
            float(item.get("dense_score", 0.0)),
            float(item.get("sparse_score", 0.0)),
        ),
        reverse=True,
    )
    rejected.sort(key=lambda item: float(item.get("support_score", 0.0)), reverse=True)

    diagnostics = {
        "query_profile": query_profile,
        "guarded_count": len(guarded),
        "rejected_count": len(rejected),
        "rejection_reasons_top": [item.get("reasons", []) for item in rejected[:5]],
        "rejection_reason_buckets": _bucket_rejection_reasons(rejected),
        "rejected_results": rejected,
        "top_k_requested": top_k,
    }
    return guarded[: max(TOP_K_RETRIEVE, top_k * 4)], diagnostics


def apply_post_rerank_guardrails(results: List[Dict], query_profile: Dict[str, Any], top_k: int) -> List[Dict]:
    filtered: List[Dict] = []
    for result in results:
        enriched = dict(result)
        rerank_score = float(enriched.get("rerank_score", 0.0) or 0.0)
        rerank_normalized = _normalize_rerank_score(rerank_score) if "rerank_score" in enriched else float(
            enriched.get("score", 0.0) or 0.0
        )
        thematic_score = float(enriched.get("thematic_score", 0.0) or 0.0)
        hybrid_score = float(enriched.get("score", 0.0) or 0.0)
        final_support = _clamp01((rerank_normalized * 0.56) + (hybrid_score * 0.22) + (thematic_score * 0.22))

        enriched["rerank_score_normalized"] = round(rerank_normalized, 4)
        enriched["final_support_score"] = round(final_support, 4)

        if query_profile.get("has_strong_topic") and thematic_score < MIN_THEMATIC_SCORE:
            continue
        if final_support < MIN_FINAL_SUPPORT_SCORE:
            continue
        filtered.append(enriched)

    filtered.sort(
        key=lambda item: (
            float(item.get("final_support_score", 0.0)),
            float(item.get("rerank_score_normalized", 0.0)),
            float(item.get("support_score", 0.0)),
        ),
        reverse=True,
    )
    return filtered[:top_k]


def select_support_fallback_results(results: List[Dict], query_profile: Dict[str, Any], top_k: int) -> List[Dict]:
    fallback_candidates: List[Dict] = []
    for result in results:
        enriched = dict(result)
        support_score = float(enriched.get("support_score", enriched.get("score", 0.0)) or 0.0)
        thematic_score = float(enriched.get("thematic_score", 0.0) or 0.0)
        if support_score < RERANK_FALLBACK_SUPPORT_SCORE:
            continue
        if query_profile.get("has_strong_topic") and thematic_score < RERANK_FALLBACK_THEMATIC_SCORE:
            continue
        if enriched.get("conflicting_topics"):
            continue
        enriched["final_support_score"] = round(max(float(enriched.get("final_support_score", 0.0) or 0.0), support_score), 4)
        fallback_candidates.append(enriched)

    fallback_candidates.sort(
        key=lambda item: (
            float(item.get("final_support_score", 0.0)),
            float(item.get("support_score", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    return fallback_candidates[:top_k]


def decide_retrieval_abstention(results: List[Dict], query_profile: Dict[str, Any]) -> Dict[str, Any]:
    if not results:
        return {"abstain": True, "reason": "no_supported_chunks"}

    top = results[0]
    top_final_support = float(top.get("final_support_score", top.get("support_score", 0.0)) or 0.0)
    top_rerank_normalized = float(top.get("rerank_score_normalized", top.get("score", 0.0)) or 0.0)
    top_thematic = float(top.get("thematic_score", 0.0) or 0.0)

    if query_profile.get("has_strong_topic") and top_thematic < MIN_THEMATIC_SCORE:
        return {"abstain": True, "reason": "top_chunk_thematically_weak"}
    if top_final_support < MIN_FINAL_SUPPORT_SCORE:
        return {"abstain": True, "reason": "top_chunk_support_too_low"}
    if top_rerank_normalized < MIN_TOP_RERANK_NORMALIZED:
        return {"abstain": True, "reason": "top_rerank_too_low"}

    conflict_count = sum(1 for item in results[:3] if item.get("conflicting_topics"))
    if query_profile.get("has_strong_topic") and conflict_count >= 2:
        return {"abstain": True, "reason": "top_results_thematically_incoherent"}

    return {"abstain": False, "reason": ""}


def _empty_debug_payload(raw_query: str, top_k: int, allowed_establishments: Optional[List[str]] = None) -> Dict[str, object]:
    normalized_allowed = _normalize_allowed_establishments(allowed_establishments)
    return {
        "query": "",
        "raw_query": raw_query,
        "query_variants": [],
        "dense_results": [],
        "sparse_results": [],
        "fusion_results": [],
        "merged_results": [],
        "boosted_results": [],
        "guarded_results": [],
        "reranked_results": [],
        "final_ranked_results": [],
        "fallback_results": [],
        "final_results": [],
        "abstain": True,
        "abstain_reason": "empty_query",
        "query_profile": {},
        "guardrail_diagnostics": {},
        "allowed_establishments_normalized": normalized_allowed,
        "candidate_retrieval": {},
        "trace": {
            "pipeline_version": "retrieval_explicit_v1",
            "stages": {
                "prepare_query": {
                    "raw_query": raw_query,
                    "query": "",
                    "query_variants": [],
                    "allowed_establishments": normalized_allowed,
                },
            },
            "decision_summary": {
                "abstain": True,
                "abstain_reason": "empty_query",
                "fallback_used": False,
                "top_k_requested": max(1, int(top_k or TOP_K_FINAL)),
                "final_result_count": 0,
                "top_result": {},
            },
        },
    }


def _build_explicit_retrieval_pipeline(
    raw_query: str,
    top_k: int,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    if not raw_query or not raw_query.strip():
        return _empty_debug_payload(raw_query, top_k, allowed_establishments=allowed_establishments)

    from .qdrant_search import run_qdrant_candidate_search

    started_at = time.perf_counter()
    effective_top_k = max(1, int(top_k))
    query = enhance_query(raw_query)
    normalized_allowed = _normalize_allowed_establishments(allowed_establishments)
    query_profile = build_query_profile(query, allowed_establishments=normalized_allowed)
    retrieve_k = max(TOP_K_RETRIEVE * 2, effective_top_k * 6)
    if normalized_allowed:
        retrieve_k = max(retrieve_k, 60)
    queries = generate_multi_queries(query)

    candidate_payload = run_qdrant_candidate_search(
        query=query,
        queries=queries,
        query_profile=query_profile,
        retrieve_k=retrieve_k,
        allowed_establishments=normalized_allowed,
        manifest_override=manifest_override,
    )

    dense_results = _normalize_result_rows(list(candidate_payload.get("dense_results", [])))
    sparse_results = _normalize_result_rows(list(candidate_payload.get("sparse_results", [])))
    fusion_results = _normalize_result_rows(list(candidate_payload.get("fusion_results", [])))
    merged_results = _normalize_result_rows(merge_dense_and_sparse(dense_results, sparse_results, top_k=retrieve_k))

    dense_by_id = {result.get("id"): result for result in dense_results}
    sparse_by_id = {result.get("id"): result for result in sparse_results}
    merged_by_id = {result.get("id"): result for result in merged_results}
    ranking_seed = fusion_results or merged_results
    enriched_merged: List[Dict] = []
    for result in ranking_seed:
        item = _normalize_result_row(result)
        fallback_scores = merged_by_id.get(item.get("id"), {})
        item["dense_score"] = float(dense_by_id.get(item.get("id"), {}).get("score", 0.0) or 0.0)
        item["sparse_score"] = float(sparse_by_id.get(item.get("id"), {}).get("score", 0.0) or 0.0)
        if "hybrid_raw_score" not in item and "hybrid_raw_score" in fallback_scores:
            item["hybrid_raw_score"] = float(fallback_scores.get("hybrid_raw_score", 0.0) or 0.0)
        item["score_type"] = str(fallback_scores.get("score_type") or item.get("score_type") or "hybrid")
        item["retrieval_sources"] = [
            source
            for source, lookup in (("dense", dense_by_id), ("sparse", sparse_by_id))
            if item.get("id") in lookup
        ]
        enriched_merged.append(_normalize_result_row(item))

    boosted_results = _normalize_result_rows(
        _filter_results_by_allowed_establishments(apply_metadata_boost(enriched_merged, query), normalized_allowed)
    )
    guarded_results, guardrail_diagnostics = apply_retrieval_guardrails(
        query,
        boosted_results,
        top_k=effective_top_k,
        allowed_establishments=normalized_allowed,
    )
    guarded_results = _normalize_result_rows(guarded_results)
    rejected_results = _normalize_result_rows(list(guardrail_diagnostics.get("rejected_results", [])))

    reranked_results = _normalize_result_rows(rerank_chunks(query, guarded_results, top_k=max(effective_top_k * 2, effective_top_k)))
    final_ranked_results = _normalize_result_rows(
        apply_post_rerank_guardrails(
            _filter_results_by_allowed_establishments(reranked_results, normalized_allowed),
            query_profile=guardrail_diagnostics.get("query_profile", {}),
            top_k=effective_top_k,
        )
    )
    abstention = decide_retrieval_abstention(final_ranked_results, guardrail_diagnostics.get("query_profile", {}))
    fallback_results = _normalize_result_rows(
        select_support_fallback_results(
            _filter_results_by_allowed_establishments(guarded_results, normalized_allowed),
            query_profile=guardrail_diagnostics.get("query_profile", {}),
            top_k=effective_top_k,
        )
    )

    fallback_used = False
    if abstention["abstain"] and abstention["reason"] == "top_rerank_too_low" and fallback_results:
        final_results = _normalize_result_rows(truncate_chunks(fallback_results, MAX_CONTEXT_CHARS))
        abstention = {"abstain": False, "reason": "support_fallback"}
        fallback_used = True
    else:
        final_results = [] if abstention["abstain"] else _normalize_result_rows(
            truncate_chunks(
                _filter_results_by_allowed_establishments(final_ranked_results, normalized_allowed),
                MAX_CONTEXT_CHARS,
            )
        )

    reranked_ids = {item.get("id") for item in reranked_results if item.get("id")}
    final_ranked_ids = {item.get("id") for item in final_ranked_results if item.get("id")}
    post_rerank_rejected = [
        item
        for item in reranked_results
        if item.get("id") and item.get("id") in reranked_ids - final_ranked_ids
    ]

    total_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    trace = {
        "pipeline_version": "retrieval_explicit_v1",
        "stages": {
            "prepare_query": {
                "raw_query": raw_query,
                "query": query,
                "query_variants": queries,
                "allowed_establishments": normalized_allowed,
                "query_profile": query_profile,
            },
            "candidate_retrieval": {
                "backend": "qdrant",
                "retrieve_k": retrieve_k,
                "manifest": candidate_payload.get("manifest", {}),
                "query_filter_applied": bool(candidate_payload.get("query_filter_applied")),
                "query_filter": candidate_payload.get("query_filter"),
                "latency_ms": candidate_payload.get("latency_ms", {}),
                "dense": _summarize_results(dense_results),
                "sparse": _summarize_results(sparse_results),
                "fusion": _summarize_results(fusion_results),
            },
            "candidate_merge": {
                "merged": _summarize_results(enriched_merged),
            },
            "metadata_boost": {
                "boosted": _summarize_results(boosted_results),
            },
            "guardrails": {
                "guarded": _summarize_results(guarded_results),
                "rejected_count": len(rejected_results),
                "rejection_reason_buckets": _bucket_rejection_reasons(rejected_results),
            },
            "rerank": {
                "enabled": bool(USE_RERANK),
                "reranked": _summarize_results(reranked_results),
                "post_rerank_rejected_count": len(post_rerank_rejected),
            },
            "final_selection": {
                "final_ranked": _summarize_results(final_ranked_results),
                "fallback": _summarize_results(fallback_results),
                "final_results": _summarize_results(final_results),
                "latency_ms_total": total_latency_ms,
            },
        },
        "decision_summary": {
            "abstain": bool(abstention["abstain"]),
            "abstain_reason": str(abstention["reason"] or ""),
            "fallback_used": fallback_used,
            "top_k_requested": effective_top_k,
            "final_result_count": len(final_results),
            "top_result": _top_result_snapshot(final_results),
        },
    }

    return {
        "query": query,
        "raw_query": raw_query,
        "query_variants": queries,
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "fusion_results": fusion_results,
        "merged_results": enriched_merged,
        "boosted_results": boosted_results,
        "guarded_results": guarded_results,
        "reranked_results": reranked_results,
        "final_ranked_results": final_ranked_results,
        "fallback_results": fallback_results,
        "final_results": final_results,
        "abstain": bool(abstention["abstain"]),
        "abstain_reason": str(abstention["reason"] or ""),
        "query_profile": guardrail_diagnostics.get("query_profile", query_profile),
        "guardrail_diagnostics": {
            **guardrail_diagnostics,
            "rejected_results": rejected_results,
            "rejection_reason_buckets": _bucket_rejection_reasons(rejected_results),
        },
        "allowed_establishments_normalized": normalized_allowed,
        "candidate_retrieval": candidate_payload,
        "trace": trace,
    }


def is_search_backend_ready() -> bool:
    try:
        from .qdrant_search import qdrant_index_ready

        return qdrant_index_ready()
    except Exception:
        return False


def run_hybrid_search_debug(
    raw_query: str,
    top_k: int = TOP_K_FINAL,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    return _build_explicit_retrieval_pipeline(
        raw_query,
        top_k=top_k,
        allowed_establishments=allowed_establishments,
        manifest_override=manifest_override,
    )


def get_relevant_chunks(
    raw_query: str,
    top_k: int = TOP_K_FINAL,
    allowed_establishments: Optional[List[str]] = None,
    manifest_override: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    debug_payload = _build_explicit_retrieval_pipeline(
        raw_query,
        top_k=top_k,
        allowed_establishments=allowed_establishments,
        manifest_override=manifest_override,
    )
    return list(debug_payload.get("final_results", []))


if __name__ == "__main__":
    test_queries = [
        "Comment s'inscrire a Semlalia ?",
        "Quelles sont les conditions d'admission a la faculte Semlalia ?",
        "Procedure inscription universite Cadi Ayyad",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        results = get_relevant_chunks(query, top_k=5)
        for i, result in enumerate(results, 1):
            score = float(result.get("score", 0.0) or 0.0)
            source = result.get("metadata", {}).get("source", "unknown")
            print(f"\n[{i}] Score: {score:.4f} | Source: {Path(source).name}")
            text = result.get("text", "")
            print(f"    {text[:280]}..." if len(text) > 280 else text)
