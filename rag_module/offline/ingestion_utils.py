# rag_module/offline/ingestion_utils.py
import hashlib
import json
import logging
import os
import re
import string
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from .structured_parser import extract_main_text

logger = logging.getLogger(__name__)

# ==============================
# CONFIGURATION
# ==============================
RAW_DATA_PATH = "data_storage/raw"
META_PATH = "data_storage/index/metadata.json"
INGESTION_STATE_PATH = "data_storage/cache/ingestion_state.json"

TIMEOUT = 20
RETRIES = 3
PLAYWRIGHT_WAIT = 2500

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


MAX_DEPTH = _env_int("RAG_MAX_DEPTH", 5)
MAX_TOTAL_URLS = _env_int("RAG_MAX_TOTAL_URLS", 40000)
INGESTION_THREADS = _env_int("RAG_INGESTION_THREADS", 8)
MAX_URLS_PER_DOMAIN = _env_int("RAG_MAX_URLS_PER_DOMAIN", 3500)
MAX_URLS_PER_SUBDOMAIN = _env_int("RAG_MAX_URLS_PER_SUBDOMAIN", 1200)
DOMAIN_PRIORITY_STEP = _env_int("RAG_DOMAIN_PRIORITY_STEP", 2)

ENABLE_SITEMAP_DISCOVERY = _env_bool("RAG_ENABLE_SITEMAP_DISCOVERY", True)
MAX_SITEMAP_URLS = _env_int("RAG_MAX_SITEMAP_URLS", 12000)
MAX_DISCOVERY_PER_SEED = _env_int("RAG_MAX_DISCOVERY_PER_SEED", 4000)
MAX_SITEMAPS_PER_SEED = _env_int("RAG_MAX_SITEMAPS_PER_SEED", 20)

INCREMENTAL_FETCH = _env_bool("RAG_INCREMENTAL_FETCH", True)
REFRESH_MODE = _env_str("RAG_REFRESH_MODE", "weekly").lower()
REFRESH_DAYS = _env_int("RAG_INCREMENTAL_REFRESH_DAYS", 7)

ENABLE_NEAR_DUP_INGESTION = _env_bool("RAG_ENABLE_NEAR_DUP_INGESTION", True)
NEAR_DUP_SIMHASH_DISTANCE = _env_int("RAG_NEAR_DUP_SIMHASH_DISTANCE", 6)
RECORD_SKIPPED_IN_METADATA = _env_bool("RAG_RECORD_SKIPPED_IN_METADATA", True)

MAX_HTML_PARSE_BYTES = 2_000_000

MIN_TEXT_WORDS = 90
MIN_TEXT_CHARS = 600
MIN_ALPHA_RATIO = 0.55
MAX_SYMBOL_RATIO = 0.30
MAX_URL_DENSITY = 0.08
MAX_MOJIBAKE_RATIO = 0.008
MIN_BINARY_BYTES = 8_000
MAX_DOWNLOAD_BYTES = 25_000_000
MIN_DOWNLOAD_QUALITY_SCORE = _env_int("RAG_MIN_DOWNLOAD_QUALITY_SCORE", 60)
INGESTION_BACKEND = _env_str("RAG_INGESTION_BACKEND", "scrapy").lower()

ALLOWED_DOMAIN_SUFFIXES = {
    # Espace officiel UCA: portail principal, etablissements, preins-*, ecampus-*,
    # Uc@Student, reins, e-candidature, bibliotheque, etc.
    "uca.ma",
    # Domaine legacy officiel de la FSTG.
    "fstg-marrakech.ac.ma",
}

ALLOWED_DOMAINS = {
    # Services externes officiels utiles aux etudiants UCA.
    "onousc.ma",
    "www.onousc.ma",
    "enssup.gov.ma",
    "www.enssup.gov.ma",
}

DEFAULT_SEEDS = [
    "https://www.uca.ma",
    "https://www.uca.ma/fr",
    "https://www.uca.ma/fr/etablissements",
    "https://www.uca.ma/fr/page/plateformes-dapprentissage",
    "https://ucastudent.uca.ma/",
    "https://reins.uca.ma/",
    "https://e-candidature.uca.ma/",
    "https://biblio-univ.uca.ma/",
    "https://fsjes.uca.ma",
    "https://flsh.uca.ma",
    "https://www.uca.ma/fssm",
    "https://www.fmpm.uca.ma/",
    "https://www.fstg-marrakech.ac.ma/",
    "https://ensa-marrakech.uca.ma/",
    "https://www.uca.ma/encg",
    "https://www.uca.ma/ens",
    "https://www.uca.ma/flam",
    "https://fps.uca.ma/",
    "https://ensas.uca.ma/",
    "https://www.ests.uca.ma/",
    "https://www.uca.ma/este",
    "https://www.estk.uca.ma/",
    "https://www.uca.ma/cuks",
    "https://www.onousc.ma/",
    "https://www.onousc.ma/Bourses",
    "https://www.onousc.ma/etudiant-marocain",
    "https://www.onousc.ma/Acces-aux-restaurants-universitaires",
    "https://www.onousc.ma/Centres-medicaux",
    "https://www.enssup.gov.ma/en/etudiant",
]

HIGH_PRIORITY_KEYWORDS = {
    "inscription",
    "preinscription",
    "pre-inscription",
    "reinscription",
    "candidature",
    "admission",
    "avis",
    "resultat",
    "resultats",
    "deliberation",
    "exam",
    "examen",
    "rattrapage",
    "emploi-du-temps",
    "emploi_du_temps",
    "emploi du temps",
    "calendrier",
    "scolarite",
    "attestation",
    "releve",
    "notes",
    "bourse",
    "frais",
    "paiement",
    "reclamation",
    "guichet",
    "e-candidature",
    "ucastudent",
    "ecampus",
    "moodle",
    "e-learning",
    "bibliotheque",
    "planning",
    "restaurant",
    "logement",
    "amo",
    "etudiant",
    "etudiants",
}

MEDIUM_PRIORITY_KEYWORDS = {
    "formation",
    "master",
    "licence",
    "doctorat",
    "filiere",
    "concours",
    "orientation",
    "reglement",
    "procedure",
    "pdf",
}

DEPRIORITIZED_KEYWORDS = {
    "recherche",
    "laboratoire",
    "laboratoires",
    "conference",
    "colloque",
    "seminaire",
    "partenariat",
    "gouvernance",
    "presidence",
    "actualite-institutionnelle",
    "recrutement",
}

HIGH_PRIORITY_PATH_HINTS = {
    "/etudiant",
    "/espace-etudiant",
    "/inscription",
    "/preinscription",
    "/reinscription",
    "/candidature",
    "/admission",
    "/avis",
    "/resultat",
    "/resultats",
    "/exam",
    "/examen",
    "/notes",
    "/planning",
    "/rattrapage",
    "/emploi-du-temps",
    "/calendrier",
    "/scolarite",
    "/attestation",
    "/bourse",
    "/ecampus",
    "/moodle",
    "/bibliotheque",
}

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "sessionid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

BLOCKED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".gif",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".png",
    ".svg",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}

SKIPPED_CONTENT_TYPE_PREFIXES = (
    "audio/",
    "image/",
    "video/",
)

EXCLUDE_PATHS = [
    "/login",
    "/admin",
    "/wp-admin",
    "/logout",
    "javascript:",
    "mailto:",
    "tel:",
    "#",
    "sessionid",
    "utm_",
    "ref=",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
URL_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
MOJIBAKE_PATTERN = re.compile(r"(Ã.|Â.|â€¦|â€™|â€œ|â€“|ï¿½)")
NORMALIZED_PRIORITY_KEYWORDS = {
    token
    for token in (
        *HIGH_PRIORITY_KEYWORDS,
        *MEDIUM_PRIORITY_KEYWORDS,
        "inscription administrative",
        "service de scolarite",
    )
    if token
}

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INGESTION_STATE_PATH), exist_ok=True)
os.makedirs("data_storage/reports", exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# THREAD SAFE PLAYWRIGHT
# ==============================
thread_local = threading.local()
playwright_resources = []
playwright_resources_lock = threading.Lock()


def get_browser():
    if not hasattr(thread_local, "browser"):
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        thread_local.playwright = pw
        thread_local.browser = browser

        with playwright_resources_lock:
            playwright_resources.append((pw, browser))
    return thread_local.browser


def cleanup_playwright_resources():
    with playwright_resources_lock:
        resources = list(playwright_resources)
        playwright_resources.clear()

    for pw, browser in resources:
        try:
            browser.close()
        except Exception as exc:
            logger.warning("Browser cleanup failed: %s", exc)

        try:
            pw.stop()
        except Exception as exc:
            logger.warning("Playwright cleanup failed: %s", exc)


def fetch_with_playwright(url):
    try:
        browser = get_browser()
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(PLAYWRIGHT_WAIT)
        content = page.content()
        context.close()
        return content.encode()
    except Exception as exc:
        logger.warning("Playwright fallback failed for %s: %s", url, exc)
        return None


# ==============================
# UTILITAIRES
# ==============================
def clean_url(url):
    parsed = urlparse(url.strip())
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
    ]
    cleaned = parsed._replace(
        query=urlencode(filtered_query, doseq=True),
        fragment="",
    )
    return urlunparse(cleaned)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json_atomic(path: str, payload) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _load_json(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def normalize_host(hostname: str) -> str:
    host = (hostname or "").strip().lower()
    if ":" in host:
        host = host.split(":", 1)[0]
    return host


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def is_allowed_domain(hostname: str) -> bool:
    host = normalize_host(hostname)

    if host in ALLOWED_DOMAINS:
        return True

    for domain in ALLOWED_DOMAIN_SUFFIXES:
        d = domain.lower()
        if host == d or host.endswith(f".{d}"):
            return True
    return False


def domain_bucket(hostname: str) -> str:
    host = _strip_www(normalize_host(hostname))
    candidates = sorted(ALLOWED_DOMAIN_SUFFIXES, key=len, reverse=True)
    for suffix in candidates:
        value = _strip_www(suffix.lower())
        if host == value or host.endswith(f".{value}"):
            return value

    for allowed in ALLOWED_DOMAINS:
        allowed_norm = _strip_www(allowed.lower())
        if host == allowed_norm:
            return allowed_norm
    return host


def subdomain_bucket(hostname: str) -> str:
    return _strip_www(normalize_host(hostname))


def compute_hash(content):
    return hashlib.md5(content).hexdigest()


def compute_text_hash(text: str) -> str:
    normalized = normalize_quality_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def looks_like_html(content: bytes) -> bool:
    sample = (content or b"")[:512].lstrip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html") or b"<body" in sample


def infer_extension(url: str, content_type: str, content: bytes = b"", content_disposition: str = "") -> str:
    lowered_type = (content_type or "").lower()
    lowered_disposition = (content_disposition or "").lower()
    path_ext = Path(urlparse(url).path).suffix.lower()
    allowed_exts = {".html", ".htm", ".pdf", ".doc", ".docx", ".txt", ".md"}

    if "filename=" in lowered_disposition:
        disposition_name = lowered_disposition.split("filename=", 1)[1].strip(" \"'")
        disposition_ext = Path(disposition_name).suffix.lower()
        if disposition_ext in allowed_exts:
            return disposition_ext

    if "text/html" in lowered_type:
        return ".html"
    if "application/pdf" in lowered_type:
        return ".pdf"
    if "application/msword" in lowered_type:
        return ".doc"
    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in lowered_type:
        return ".docx"
    if "text/plain" in lowered_type:
        return ".txt"
    if content.startswith(b"%PDF"):
        return ".pdf"
    if looks_like_html(content):
        return ".html"
    if path_ext in allowed_exts:
        return path_ext
    return ".html"


def normalize_quality_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_for_simhash(text: str) -> List[str]:
    return WORD_PATTERN.findall(normalize_quality_text(text))


def _simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    acc = [0] * 64
    for token in set(tokens):
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for bit in range(64):
            acc[bit] += 1 if ((digest >> bit) & 1) else -1
    value = 0
    for bit, score in enumerate(acc):
        if score > 0:
            value |= (1 << bit)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_text_preview(content: bytes, ext: str) -> str:
    if not content:
        return ""

    if ext in {".html", ".htm"}:
        try:
            html = content[:MAX_HTML_PARSE_BYTES].decode("utf-8", errors="replace")
            extracted = extract_main_text(html)
            if extracted.get("text"):
                return str(extracted.get("text") or "")
        except Exception:
            return ""

    if ext in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")

    return ""


def score_text_quality(text: str) -> dict:
    cleaned = unescape(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = WORD_PATTERN.findall(cleaned.lower())
    word_count = len(words)
    char_count = len(cleaned)
    non_space = sum(1 for ch in cleaned if not ch.isspace())
    alpha_count = sum(1 for ch in cleaned if ch.isalpha())
    symbol_count = sum(
        1 for ch in cleaned
        if ch in string.punctuation or (not ch.isalnum() and not ch.isspace())
    )
    url_count = len(URL_PATTERN.findall(cleaned))
    alpha_ratio = (alpha_count / non_space) if non_space else 0.0
    symbol_ratio = (symbol_count / non_space) if non_space else 0.0
    url_density = (url_count / max(word_count, 1))
    mojibake_ratio = len(MOJIBAKE_PATTERN.findall(cleaned)) / max(char_count, 1)

    return {
        "text": cleaned,
        "words": word_count,
        "chars": char_count,
        "alpha_ratio": alpha_ratio,
        "symbol_ratio": symbol_ratio,
        "url_density": url_density,
        "mojibake_ratio": mojibake_ratio,
    }


def compute_download_quality(
    url: str,
    depth: int,
    content: bytes,
    content_type: str,
    ext: str,
) -> dict:
    content_size = len(content or b"")
    lowered_type = (content_type or "").lower().strip()
    normalized_url = normalize_quality_text(url)
    keyword_hits = sorted(
        keyword
        for keyword in NORMALIZED_PRIORITY_KEYWORDS
        if normalize_quality_text(keyword) and normalize_quality_text(keyword) in normalized_url
    )

    if content_size <= 0:
        return {
            "keep": False,
            "score": 0,
            "reason": "empty_content",
            "keyword_hits": keyword_hits,
            "metrics": {},
            "preview_text": "",
            "text_content_hash": "",
        }

    if content_size > MAX_DOWNLOAD_BYTES:
        return {
            "keep": False,
            "score": 0,
            "reason": "file_too_large",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
            "text_content_hash": "",
        }

    if any(lowered_type.startswith(prefix) for prefix in SKIPPED_CONTENT_TYPE_PREFIXES):
        return {
            "keep": False,
            "score": 0,
            "reason": "unsupported_media_content_type",
            "keyword_hits": keyword_hits,
            "metrics": {"content_type": lowered_type},
            "preview_text": "",
            "text_content_hash": "",
        }

    if ext == ".doc":
        return {
            "keep": False,
            "score": 0,
            "reason": "unsupported_legacy_doc",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
            "text_content_hash": "",
        }

    if ext in {".pdf", ".doc", ".docx"}:
        score = 55
        if content_size >= MIN_BINARY_BYTES:
            score += 20
        else:
            score -= 30

        if keyword_hits:
            score += min(20, len(keyword_hits) * 5)
        elif depth >= 2:
            score -= 10

        score = max(0, min(100, int(round(score))))
        return {
            "keep": score >= MIN_DOWNLOAD_QUALITY_SCORE,
            "score": score,
            "reason": "binary_quality_gate",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
            "text_content_hash": "",
        }

    preview = extract_text_preview(content, ext)
    metrics = score_text_quality(preview)

    score = 100.0
    if metrics["words"] < MIN_TEXT_WORDS:
        score -= min(35.0, (MIN_TEXT_WORDS - metrics["words"]) * 0.5)
    if metrics["chars"] < MIN_TEXT_CHARS:
        score -= min(30.0, (MIN_TEXT_CHARS - metrics["chars"]) / 20.0)
    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        score -= (MIN_ALPHA_RATIO - metrics["alpha_ratio"]) * 90.0
    if metrics["symbol_ratio"] > MAX_SYMBOL_RATIO:
        score -= (metrics["symbol_ratio"] - MAX_SYMBOL_RATIO) * 130.0
    if metrics["url_density"] > MAX_URL_DENSITY:
        score -= min(25.0, (metrics["url_density"] - MAX_URL_DENSITY) * 250.0)
    if metrics["mojibake_ratio"] > MAX_MOJIBAKE_RATIO:
        score -= min(25.0, (metrics["mojibake_ratio"] - MAX_MOJIBAKE_RATIO) * 1200.0)

    normalized_preview = normalize_quality_text(metrics["text"][:7000])
    if keyword_hits:
        score += min(16.0, len(keyword_hits) * 4.0)
    else:
        text_hits = sum(1 for keyword in NORMALIZED_PRIORITY_KEYWORDS if normalize_quality_text(keyword) in normalized_preview)
        if text_hits:
            score += min(14.0, text_hits * 3.5)
        elif depth >= 2:
            score -= 16.0

    if "404" in metrics["text"] and metrics["words"] < 140:
        score -= 20.0

    score = max(0, min(100, int(round(score))))
    return {
        "keep": score >= MIN_DOWNLOAD_QUALITY_SCORE,
        "score": score,
        "reason": "text_quality_gate",
        "keyword_hits": keyword_hits,
        "metrics": {
            "bytes": content_size,
            "words": metrics["words"],
            "chars": metrics["chars"],
            "alpha_ratio": round(metrics["alpha_ratio"], 4),
            "symbol_ratio": round(metrics["symbol_ratio"], 4),
            "url_density": round(metrics["url_density"], 4),
            "mojibake_ratio": round(metrics["mojibake_ratio"], 5),
        },
        "preview_text": metrics["text"][:12000],
        "text_content_hash": compute_text_hash(metrics["text"][:12000]),
    }


def generate_filename(url, ext):
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = re.sub(r"[^\w\-_.]", "_", name)
    base = name[:130].rstrip("._") or "document"
    # Suffixe stable pour eviter les collisions de noms entre URLs similaires.
    suffix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{suffix}{ext}"


def save_file(content, filename):
    path = os.path.join(RAW_DATA_PATH, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def score_url(url: str, depth: int = 0) -> int:
    lowered = url.lower()
    path_ext = Path(urlparse(lowered).path).suffix.lower()
    score = 100

    if any(keyword in lowered for keyword in HIGH_PRIORITY_KEYWORDS):
        score -= 35
    if any(hint in lowered for hint in HIGH_PRIORITY_PATH_HINTS):
        score -= 20
    if any(keyword in lowered for keyword in MEDIUM_PRIORITY_KEYWORDS):
        score -= 10
    if any(keyword in lowered for keyword in DEPRIORITIZED_KEYWORDS):
        score += 30
    if path_ext == ".pdf":
        score -= 20
    elif path_ext == ".docx":
        score -= 8

    score += min(depth * 4, 16)
    return max(score, 0)


# ==============================
# FILTRAGE
# ==============================
def should_accept_url(url, base=None):
    if not url:
        return None

    if base:
        url = urljoin(base, url)

    url = clean_url(url)
    parsed = urlparse(url)
    path_ext = Path(parsed.path).suffix.lower()

    if parsed.scheme not in ["http", "https"]:
        return None

    if not is_allowed_domain(parsed.netloc):
        return None

    if path_ext in BLOCKED_EXTENSIONS:
        return None

    if any(x in url.lower() for x in EXCLUDE_PATHS):
        return None

    if len(url) > 300:
        return None

    return url


# ==============================
# EXTRACTION
# ==============================
def extract_links(content, base):
    soup = BeautifulSoup(content, "lxml")
    links = set()

    for tag in soup.find_all(["a", "iframe", "embed"]):
        href = tag.get("href") or tag.get("src")
        full = should_accept_url(href, base)
        if full:
            links.add(full)

    return list(links)

