import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_CHUNKS_PER_SOURCE = _env_int("RAG_MAX_CHUNKS_PER_SOURCE", 120)
DOMAIN_FILTER_ENABLED = _env_bool("RAG_DOMAIN_FILTER_ENABLED", True)
BACKUP_ENABLED = _env_bool("RAG_CREATE_BACKUP", True)
LANG_ALLOWLIST = {
    lang.strip().lower()
    for lang in os.getenv("RAG_LANG_ALLOWLIST", "fr,ar,en").split(",")
    if lang.strip()
}

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

FACULTY_RULES = {
    "fssm": "FSSM",
    "fstg": "FSTG",
    "fsjes": "FSJES",
    "flsh": "FLSH",
    "ensa": "ENSA",
    "encg": "ENCG",
    "ens": "ENS",
    "fmpm": "FMPM",
    "uca": "UCA",
}

DOCUMENT_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("admission", ["admission", "admis", "selection", "concours", "\u0642\u0628\u0648\u0644", "\u0645\u0628\u0627\u0631\u0627\u0629"]),
    ("inscription", ["inscription", "preinscription", "pre-inscription", "reinscription", "\u0627\u0644\u062a\u0633\u062c\u064a\u0644"]),
    ("bourse", ["bourse", "scholarship", "\u0645\u0646\u062d\u0629"]),
    ("calendrier", ["calendrier", "planning", "emploi du temps", "schedule"]),
    ("resultats", ["resultat", "resultats", "notes", "deliberation", "rattrapage"]),
    ("formation", ["master", "licence", "doctorat", "filiere", "programme", "\u0645\u0627\u0633\u062a\u0631", "\u0625\u062c\u0627\u0632\u0629"]),
]

HIGH_SIGNAL_DOCUMENT_TYPES = {"admission", "inscription", "bourse", "calendrier", "resultats", "formation"}
MIN_CHUNK_RELEVANCE_SCORE = 1
MIN_SOURCE_RELEVANCE_SCORE = 2


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NORMALIZED_TARGET_KEYWORDS = {_normalize_text(keyword) for keyword in TARGET_KEYWORDS}
NORMALIZED_SOURCE_HINTS = {_normalize_text(keyword) for keyword in SOURCE_HINT_KEYWORDS}
NORMALIZED_DOCUMENT_TYPE_RULES: List[Tuple[str, List[str]]] = [
    (doc_type, [_normalize_text(keyword) for keyword in keywords])
    for doc_type, keywords in DOCUMENT_TYPE_RULES
]


def create_backup(processed_path: str, cache_file: str, backup_root: str = "data_storage/backups") -> Optional[str]:
    if not BACKUP_ENABLED:
        return None

    processed_dir = Path(processed_path)
    if not processed_dir.exists():
        return None

    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(backup_root) / f"processed_{timestamp}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(processed_dir, target_dir)

    cache_path = Path(cache_file)
    if cache_path.exists():
        shutil.copy2(cache_path, target_dir / cache_path.name)

    return str(target_dir)


def _canonical_file_type(raw_file_type: str, source_path: str) -> str:
    value = (raw_file_type or "").strip().lower().lstrip(".")
    if value:
        return value
    suffix = Path(source_path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _detect_faculty(source_path: str, text: str) -> str:
    haystack = _normalize_text(f"{source_path} {text}")
    for token, label in FACULTY_RULES.items():
        if token in haystack:
            return label
    return "unknown"


def _detect_document_type(source_path: str, text: str) -> str:
    haystack = _normalize_text(f"{source_path} {text}")
    for doc_type, keywords in NORMALIZED_DOCUMENT_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return doc_type
    return "general"


def _detect_year(source_path: str, text: str) -> Optional[int]:
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{source_path} {text}")
    if not years:
        return None
    year_values = sorted({int(year) for year in years})
    return year_values[-1]


def _keyword_hits(text: str, keyword_bank: Set[str]) -> Set[str]:
    normalized = _normalize_text(text)
    return {keyword for keyword in keyword_bank if keyword and keyword in normalized}


def _source_relevance_score(source_path: str, joined_text: str, document_type: str) -> Tuple[int, List[str]]:
    signals: List[str] = []
    score = 0

    source_hits = _keyword_hits(source_path, NORMALIZED_TARGET_KEYWORDS)
    text_hits = _keyword_hits(joined_text[:16000], NORMALIZED_TARGET_KEYWORDS)
    hint_hits = _keyword_hits(f"{source_path} {joined_text[:6000]}", NORMALIZED_SOURCE_HINTS)

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


def _chunk_relevance_score(source_path: str, text: str, document_type: str) -> Tuple[int, List[str]]:
    score = 0
    signals: List[str] = []

    chunk_hits = _keyword_hits(text, NORMALIZED_TARGET_KEYWORDS)
    hint_hits = _keyword_hits(text, NORMALIZED_SOURCE_HINTS)

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


def _is_target_content(
    source_path: str,
    text: str,
    document_type: str,
    source_relevance_score: int,
) -> Tuple[bool, int, List[str]]:
    chunk_score, chunk_signals = _chunk_relevance_score(source_path, text, document_type)
    keep = chunk_score >= MIN_CHUNK_RELEVANCE_SCORE
    return keep, chunk_score, chunk_signals


def _downsample_evenly(items: List[Dict], max_items: int) -> List[Dict]:
    if len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]

    total = len(items)
    indices = [int(i * total / max_items) for i in range(max_items)]
    return [items[index] for index in indices]


def postprocess_chunks_for_source(chunks: List[Dict], source_path: str) -> List[Dict]:
    prepared_chunks: List[Dict] = []
    joined_text_parts: List[str] = []

    for chunk in chunks:
        text = (chunk.get("text", "") or "").strip()
        if not text:
            continue

        metadata = dict(chunk.get("metadata", {}) or {})
        language = (metadata.get("language", "unknown") or "unknown").lower()
        if language not in LANG_ALLOWLIST:
            continue

        file_type = _canonical_file_type(str(metadata.get("file_type", "")), source_path)
        faculty = _detect_faculty(source_path, text)
        document_type = _detect_document_type(source_path, text)
        year = _detect_year(source_path, text)

        metadata["file_type"] = file_type
        metadata["faculty"] = faculty
        metadata["document_type"] = document_type
        if year is not None:
            metadata["year"] = year
        else:
            metadata.pop("year", None)

        updated_chunk = dict(chunk)
        updated_chunk["metadata"] = metadata
        prepared_chunks.append(updated_chunk)
        joined_text_parts.append(text)

    if not prepared_chunks:
        return []

    source_document_type = _detect_document_type(source_path, "\n".join(joined_text_parts[:8]))
    source_relevance_score, source_relevance_hits = _source_relevance_score(
        source_path,
        "\n".join(joined_text_parts),
        source_document_type,
    )

    cleaned: List[Dict] = []
    for chunk in prepared_chunks:
        text = (chunk.get("text", "") or "").strip()
        metadata = dict(chunk.get("metadata", {}) or {})
        document_type = str(metadata.get("document_type") or "general")

        keep = True
        chunk_relevance_score = 0
        chunk_relevance_hits: List[str] = []
        if DOMAIN_FILTER_ENABLED:
            keep, chunk_relevance_score, chunk_relevance_hits = _is_target_content(
                source_path,
                text,
                document_type,
                source_relevance_score,
            )
        else:
            chunk_relevance_score, chunk_relevance_hits = _chunk_relevance_score(
                source_path,
                text,
                document_type,
            )

        metadata["source_relevance_score"] = source_relevance_score
        metadata["source_relevance_hits"] = source_relevance_hits
        metadata["chunk_relevance_score"] = chunk_relevance_score
        metadata["chunk_relevance_hits"] = chunk_relevance_hits

        if not keep:
            continue

        updated_chunk = dict(chunk)
        updated_chunk["metadata"] = metadata
        cleaned.append(updated_chunk)

    limited = _downsample_evenly(cleaned, MAX_CHUNKS_PER_SOURCE)
    total = len(limited)
    for index, chunk in enumerate(limited):
        metadata = dict(chunk.get("metadata", {}) or {})
        metadata["index"] = index
        metadata["total_chunks"] = total
        chunk["metadata"] = metadata

    return limited
