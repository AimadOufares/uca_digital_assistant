import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LANG_ALLOWLIST = {
    lang.strip().lower()
    for lang in os.getenv("RAG_LANG_ALLOWLIST", "fr,ar,en").split(",")
    if lang.strip()
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
    haystack = normalize_text(f"{source_path} {text}")
    for token, label in FACULTY_RULES.items():
        if token in haystack:
            return label
    return "unknown"


def detect_document_type(source_path: str, text: str) -> str:
    haystack = normalize_text(f"{source_path} {text}")
    for doc_type, keywords in NORMALIZED_DOCUMENT_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return doc_type
    return "general"


def detect_year(source_path: str, text: str) -> Optional[int]:
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{source_path} {text}")
    if not years:
        return None
    year_values = sorted({int(year) for year in years})
    return year_values[-1]


def prepare_chunk_metadata(chunk: Dict, source_path: str) -> Optional[Dict]:
    text = (chunk.get("text", "") or "").strip()
    if not text:
        return None

    metadata = dict(chunk.get("metadata", {}) or {})
    language = (metadata.get("language", "unknown") or "unknown").lower()
    if language not in LANG_ALLOWLIST:
        return None

    file_type = canonical_file_type(str(metadata.get("file_type", "")), source_path)
    faculty = detect_faculty(source_path, text)
    document_type = detect_document_type(source_path, text)
    year = detect_year(source_path, text)

    metadata["file_type"] = file_type
    metadata["faculty"] = faculty
    metadata["document_type"] = document_type
    if year is not None:
        metadata["year"] = year
    else:
        metadata.pop("year", None)

    updated_chunk = dict(chunk)
    updated_chunk["metadata"] = metadata
    return updated_chunk
