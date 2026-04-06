import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    "resultat",
    "semestre",
    "module",
    "registration",
    "scholarship",
    "application",
    "admissions",
    "التسجيل",
    "قبول",
    "منحة",
    "مباراة",
    "ماستر",
    "إجازة",
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
    ("admission", ["admission", "admis", "selection", "concours", "قبول", "مباراة"]),
    ("inscription", ["inscription", "preinscription", "reinscription", "التسجيل"]),
    ("bourse", ["bourse", "scholarship", "منحة"]),
    ("calendrier", ["calendrier", "planning", "emploi du temps", "schedule"]),
    ("resultats", ["resultat", "resultats", "notes", "deliberation"]),
    ("formation", ["master", "licence", "doctorat", "filiere", "programme"]),
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
    haystack = f"{source_path} {text}".lower()
    for token, label in FACULTY_RULES.items():
        if token in haystack:
            return label
    return "unknown"


def _detect_document_type(source_path: str, text: str) -> str:
    haystack = f"{source_path} {text}".lower()
    for doc_type, keywords in DOCUMENT_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return doc_type
    return "general"


def _detect_year(source_path: str, text: str) -> Optional[int]:
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{source_path} {text}")
    if not years:
        return None
    year_values = sorted({int(year) for year in years})
    return year_values[-1]


def _is_target_content(source_path: str, text: str) -> bool:
    haystack = f"{source_path} {text}".lower()
    return any(keyword in haystack for keyword in TARGET_KEYWORDS)


def _downsample_evenly(items: List[Dict], max_items: int) -> List[Dict]:
    if len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]

    total = len(items)
    indices = [int(i * total / max_items) for i in range(max_items)]
    return [items[index] for index in indices]


def postprocess_chunks_for_source(chunks: List[Dict], source_path: str) -> List[Dict]:
    cleaned: List[Dict] = []

    for chunk in chunks:
        text = (chunk.get("text", "") or "").strip()
        if not text:
            continue

        metadata = dict(chunk.get("metadata", {}) or {})
        language = (metadata.get("language", "unknown") or "unknown").lower()
        if language not in LANG_ALLOWLIST:
            continue

        if DOMAIN_FILTER_ENABLED and not _is_target_content(source_path, text):
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
        cleaned.append(updated_chunk)

    limited = _downsample_evenly(cleaned, MAX_CHUNKS_PER_SOURCE)
    total = len(limited)
    for index, chunk in enumerate(limited):
        metadata = dict(chunk.get("metadata", {}) or {})
        metadata["index"] = index
        metadata["total_chunks"] = total
        chunk["metadata"] = metadata

    return limited
