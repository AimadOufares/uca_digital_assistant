import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context_resolution import (
    ESTABLISHMENT_ALIASES,
    detect_primary_establishment,
    get_metadata_establishment,
)

LANG_ALLOWLIST = {
    lang.strip().lower()
    for lang in os.getenv("RAG_LANG_ALLOWLIST", "fr,ar,en").split(",")
    if lang.strip()
}

DOCUMENT_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("admission", ["admission", "admis", "selection", "concours", "\u0642\u0628\u0648\u0644", "\u0645\u0628\u0627\u0631\u0627\u0629"]),
    ("inscription", ["inscription", "preinscription", "pre-inscription", "reinscription", "\u0627\u0644\u062a\u0633\u062c\u064a\u0644"]),
    ("bourse", ["bourse", "scholarship", "\u0645\u0646\u062d\u0629"]),
    ("stage", ["stage", "stages", "pfe", "internship", "memoire", "convention de stage"]),
    ("calendrier", ["calendrier", "planning", "emploi du temps", "schedule"]),
    ("resultats", ["resultat", "resultats", "notes", "deliberation", "rattrapage"]),
    ("contact", ["contact", "contacts", "email", "mail", "telephone", "service de scolarite"]),
    ("reglement", ["reglement", "reglements", "reglement pedagogique", "reglements pedagogiques", "lmd", "ects"]),
    ("formation", ["master", "licence", "doctorat", "filiere", "programme", "\u0645\u0627\u0633\u062a\u0631", "\u0625\u062c\u0627\u0632\u0629"]),
]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


FACULTY_RULES = {
    normalize_text(alias): label
    for label, aliases in ESTABLISHMENT_ALIASES.items()
    for alias in aliases
}


NORMALIZED_DOCUMENT_TYPE_RULES: List[Tuple[str, List[str]]] = [
    (doc_type, [normalize_text(keyword) for keyword in keywords])
    for doc_type, keywords in DOCUMENT_TYPE_RULES
]


def canonical_file_type(raw_file_type: str, source_path: str) -> str:
    value = (raw_file_type or "").strip().lower().lstrip(".")
    if value:
        return value
    suffix = Path(source_path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def detect_faculty(source_path: str, text: str) -> str:
    return detect_primary_establishment(source_path, text)


def detect_document_type(source_path: str, text: str) -> str:
    haystack = normalize_text(f"{source_path} {text}")
    for doc_type, keywords in NORMALIZED_DOCUMENT_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return doc_type
    return "general"


def detect_year(source_path: str, text: str) -> Optional[int]:
    haystack = f"{source_path} {text}"
    
    academic_years = re.findall(r"\b((?:19|20)\d{2})\s*[-/]\s*((?:19|20)\d{2})\b", haystack)
    if academic_years:
        return max(max(int(y1), int(y2)) for y1, y2 in academic_years)
        
    years = re.findall(r"\b(?:19|20)\d{2}\b", haystack)
    if not years:
        return None
        
    year_values = sorted({int(year) for year in years if 1990 <= int(year) <= 2030})
    if not year_values:
        return None
    return year_values[-1]


def prepare_chunk_metadata(chunk: Dict, source_path: str) -> Optional[Dict]:
    text = (chunk.get("text", "") or "").strip()
    if not text:
        return None

    metadata = dict(chunk.get("metadata", {}) or {})
    language = (metadata.get("language", "unknown") or "unknown").lower()
    if language not in LANG_ALLOWLIST:
        return None

    section_title = str(metadata.get("section_title") or "").strip()
    section_path = " ".join(str(part).strip() for part in metadata.get("section_path", []) if str(part).strip())
    contextual_text = " ".join(part for part in [section_title, section_path, text] if part).strip()

    file_type = canonical_file_type(str(metadata.get("file_type", "")), source_path)
    faculty = detect_faculty(source_path, contextual_text)
    document_type = detect_document_type(source_path, contextual_text)
    year = detect_year(source_path, contextual_text)

    metadata["file_type"] = file_type
    metadata["etablissement"] = faculty
    metadata["faculty"] = faculty
    metadata["document_type"] = document_type
    metadata["chunk_id"] = str(metadata.get("chunk_id") or metadata.get("chunk_hash") or "")
    metadata["section_title"] = section_title
    metadata["section_path"] = [str(part).strip() for part in metadata.get("section_path", []) if str(part).strip()]
    if year is not None:
        metadata["year"] = year
    else:
        metadata.pop("year", None)
    if get_metadata_establishment(metadata) == "unknown":
        metadata["etablissement"] = "unknown"
        metadata["faculty"] = "unknown"

    updated_chunk = dict(chunk)
    updated_chunk["metadata"] = metadata
    return updated_chunk
