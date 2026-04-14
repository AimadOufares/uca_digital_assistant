# rag_module/ingestion.py
import os
import re
import json
import hashlib
import logging
import queue
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
from playwright.sync_api import sync_playwright

try:
    from ..shared.html_extraction import extract_main_text
    from ..shared.offline_pipeline_report import update_offline_pipeline_report
except ImportError:  # pragma: no cover
    from rag_module.shared.html_extraction import extract_main_text
    from rag_module.shared.offline_pipeline_report import update_offline_pipeline_report

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
        }

    if content_size > MAX_DOWNLOAD_BYTES:
        return {
            "keep": False,
            "score": 0,
            "reason": "file_too_large",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
        }

    if any(lowered_type.startswith(prefix) for prefix in SKIPPED_CONTENT_TYPE_PREFIXES):
        return {
            "keep": False,
            "score": 0,
            "reason": "unsupported_media_content_type",
            "keyword_hits": keyword_hits,
            "metrics": {"content_type": lowered_type},
            "preview_text": "",
        }

    if ext == ".doc":
        return {
            "keep": False,
            "score": 0,
            "reason": "unsupported_legacy_doc",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
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


def load_ingestion_state() -> Dict:
    raw = _load_json(INGESTION_STATE_PATH)
    if isinstance(raw, dict):
        urls = raw.get("urls", {})
        if isinstance(urls, dict):
            normalized = {}
            for key, value in urls.items():
                if isinstance(key, str) and isinstance(value, dict):
                    normalized[key] = value
            return {"version": 2, "urls": normalized}
    return {"version": 2, "urls": {}}


def save_ingestion_state(state: Dict) -> None:
    _save_json_atomic(INGESTION_STATE_PATH, state)


def _update_state_entry(
    state: Dict,
    canonical_url: str,
    *,
    url: Optional[str] = None,
    domain: Optional[str] = None,
    last_status: Optional[str] = None,
    last_http_code: Optional[int] = None,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    content_hash: Optional[str] = None,
    file_path: Optional[str] = None,
    last_fetched_at: Optional[str] = None,
    quality_score: Optional[int] = None,
    quality_reason: Optional[str] = None,
    skip_reason: Optional[str] = None,
    crawl_depth: Optional[int] = None,
) -> None:
    entry = dict(state.get("urls", {}).get(canonical_url, {}))
    now_value = now_iso()
    
    # Initialize all required keys if missing
    for key in [
        "url", "canonical_url", "domain", "last_status", "last_http_code",
        "etag", "last_modified", "content_hash", "file_path",
        "first_seen_at", "last_seen_at", "last_fetched_at",
        "quality_score", "quality_reason", "skip_reason", "crawl_depth"
    ]:
        if key not in entry:
            entry[key] = None if key in ["last_http_code", "quality_score", "crawl_depth"] else ""

    if not entry.get("first_seen_at"):
        entry["first_seen_at"] = now_value
    entry["last_seen_at"] = now_value
    entry["canonical_url"] = canonical_url

    if url is not None:
        entry["url"] = url
    if domain is not None:
        entry["domain"] = domain
    if last_status is not None:
        entry["last_status"] = last_status
    if last_http_code is not None:
        entry["last_http_code"] = int(last_http_code)
    if etag is not None:
        entry["etag"] = etag
    if last_modified is not None:
        entry["last_modified"] = last_modified
    if content_hash is not None:
        entry["content_hash"] = content_hash
    if file_path is not None:
        entry["file_path"] = file_path
    if last_fetched_at is not None:
        entry["last_fetched_at"] = last_fetched_at
    if quality_score is not None:
        entry["quality_score"] = int(quality_score)
    if quality_reason is not None:
        entry["quality_reason"] = quality_reason
    if skip_reason is not None:
        entry["skip_reason"] = skip_reason
    if crawl_depth is not None:
        entry["crawl_depth"] = int(crawl_depth)

    state.setdefault("urls", {})[canonical_url] = entry


def _metadata_skip_row(url: str, canonical_url: str, depth: int, reason: str, domain: str,
                       quality_score: int = 0, quality_reason: str = "",
                       content_hash: str = "", etag: str = "", last_modified: str = "") -> Dict:
    return {
        "url": url,
        "canonical_url": canonical_url,
        "domain": domain,
        "depth": depth,
        "status": "skipped",
        "skip_reason": reason,
        "fetched_at": now_iso(),
        "download_quality_score": quality_score,
        "download_quality_reason": quality_reason,
        "content_hash": content_hash,
        "etag": etag,
        "last_modified": last_modified,
    }


def _should_refresh_weekly(previous_entry: Dict) -> bool:
    if REFRESH_MODE != "weekly":
        return True
    fetched_at = str(previous_entry.get("last_fetched_at") or "").strip()
    if not fetched_at:
        return True
    try:
        dt = datetime.fromisoformat(fetched_at)
    except Exception:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt <= (datetime.now(timezone.utc) - timedelta(days=REFRESH_DAYS))


def _conditional_headers(previous_entry: Dict) -> Dict[str, str]:
    headers = dict(HEADERS)
    if not INCREMENTAL_FETCH:
        return headers
    etag = str(previous_entry.get("etag") or "").strip()
    last_modified = str(previous_entry.get("last_modified") or "").strip()
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _extract_sitemap_locs(xml_bytes: bytes) -> List[str]:
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []

    locs: List[str] = []
    for elem in root.iter():
        if elem.tag.lower().endswith("loc") and elem.text:
            locs.append(elem.text.strip())
    return locs


def _discover_seed_urls(seed_url: str) -> List[str]:
    if not ENABLE_SITEMAP_DISCOVERY:
        return []

    parsed = urlparse(seed_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_candidates = {urljoin(root, "/sitemap.xml")}

    try:
        robots_response = requests.get(urljoin(root, "/robots.txt"), headers=HEADERS, timeout=12)
        if robots_response.status_code == 200:
            for line in robots_response.text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("sitemap:"):
                    candidate = stripped.split(":", 1)[1].strip()
                    if candidate:
                        sitemap_candidates.add(candidate)
    except Exception:
        pass

    discovered: List[str] = []
    sitemap_queue = list(sitemap_candidates)
    seen_sitemaps: Set[str] = set()
    scanned_sitemaps = 0

    while sitemap_queue and scanned_sitemaps < MAX_SITEMAPS_PER_SEED and len(discovered) < MAX_DISCOVERY_PER_SEED:
        sitemap_url = clean_url(sitemap_queue.pop(0))
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)
        scanned_sitemaps += 1

        try:
            response = requests.get(sitemap_url, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                continue
            locs = _extract_sitemap_locs(response.content[:MAX_SITEMAP_URLS * 120])
        except Exception:
            continue

        for loc in locs:
            if len(discovered) >= MAX_DISCOVERY_PER_SEED:
                break
            candidate = should_accept_url(loc)
            if not candidate:
                continue
            if candidate.endswith(".xml"):
                if candidate not in seen_sitemaps:
                    sitemap_queue.append(candidate)
                continue
            discovered.append(candidate)

    return list(dict.fromkeys(discovered))


def _load_previous_metadata_rows() -> List[Dict]:
    payload = _load_json(META_PATH)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _increment_counter(stats: Dict, key: str, amount: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + amount


# ==============================
# LOCKS MULTITHREADING
# ==============================
visited_lock = threading.Lock()
results_lock = threading.Lock()

def save_metadata_safely(results_list):
    try:
        _save_json_atomic(META_PATH, {"version": 2, "rows": results_list})
    except Exception as exc:
        logger.error("Erreur lors de la sauvegarde metadata: %s", exc)

# ==============================
# DOWNLOAD
# ==============================
def download(
    url: str,
    depth: int,
    seen_hashes: Set[str],
    near_dup_signatures: List[int],
    state: Dict,
    state_lock: threading.Lock,
    dedup_lock: threading.Lock,
    stats: Dict,
    stats_lock: threading.Lock,
) -> Optional[Dict]:
    canonical_url = clean_url(url)
    host = normalize_host(urlparse(canonical_url).netloc)
    domain = domain_bucket(host)
    subdomain = subdomain_bucket(host)

    with state_lock:
        previous_entry = dict(state.get("urls", {}).get(canonical_url, {}))
        _update_state_entry(
            state,
            canonical_url,
            url=url,
            domain=domain,
            crawl_depth=depth,
            last_status="seen",
            skip_reason="",
        )

    if not _should_refresh_weekly(previous_entry):
        with stats_lock:
            _increment_counter(stats, "refresh_skipped")
        with state_lock:
            _update_state_entry(
                state,
                canonical_url,
                url=url,
                domain=domain,
                last_status="refresh_skipped",
                last_http_code=int(previous_entry.get("last_http_code") or 0),
                skip_reason="refresh_window_not_elapsed",
                crawl_depth=depth,
            )
        if RECORD_SKIPPED_IN_METADATA:
            return _metadata_skip_row(
                url, canonical_url, depth, "refresh_window_not_elapsed", domain,
                etag=str(previous_entry.get("etag") or ""),
                last_modified=str(previous_entry.get("last_modified") or "")
            )
        return None

    for _ in range(RETRIES):
        try:
            timeout = TIMEOUT if depth < 2 else 10
            response = requests.get(url, headers=_conditional_headers(previous_entry), timeout=timeout)

            if response.status_code == 304:
                with stats_lock:
                    _increment_counter(stats, "not_modified")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="not_modified",
                        last_http_code=304,
                        skip_reason="not_modified",
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(
                        url, canonical_url, depth, "not_modified", domain,
                        etag=str(previous_entry.get("etag") or ""),
                        last_modified=str(previous_entry.get("last_modified") or "")
                    )
                return None

            if response.status_code != 200:
                with stats_lock:
                    _increment_counter(stats, "http_errors")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="http_error",
                        last_http_code=response.status_code,
                        skip_reason=f"http_{response.status_code}",
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(url, canonical_url, depth, f"http_{response.status_code}", domain)
                return None

            content = response.content
            content_type = response.headers.get("Content-Type", "")
            lowered_content_type = content_type.lower()

            is_binary_document = any(
                token in lowered_content_type
                for token in (
                    "application/pdf",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/octet-stream",
                )
            )
            if not is_binary_document and (len(content) < 1200 or b"<script" in content):
                pw_content = fetch_with_playwright(url)
                if pw_content:
                    content = pw_content
                    content_type = "text/html; charset=utf-8"

            ext = infer_extension(
                url,
                content_type,
                content=content,
                content_disposition=response.headers.get("Content-Disposition", ""),
            )
            quality = compute_download_quality(
                url=url,
                depth=depth,
                content=content,
                content_type=content_type,
                ext=ext,
            )
            if not quality["keep"]:
                with stats_lock:
                    _increment_counter(stats, "skipped_quality")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="skipped_quality",
                        last_http_code=response.status_code,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or ""),
                        skip_reason=str(quality.get("reason") or "quality_rejected"),
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(
                        url,
                        canonical_url,
                        depth,
                        str(quality.get("reason") or "quality_rejected"),
                        domain,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or "")
                    )
                return None

            content_hash = compute_hash(content)
            with dedup_lock:
                if content_hash in seen_hashes:
                    with stats_lock:
                        _increment_counter(stats, "skipped_dup_exact")
                    with state_lock:
                        _update_state_entry(
                            state,
                            canonical_url,
                            url=url,
                            domain=domain,
                            last_status="skipped_dup_exact",
                            last_http_code=response.status_code,
                            content_hash=content_hash,
                            quality_score=int(quality.get("score") or 0),
                            quality_reason=str(quality.get("reason") or ""),
                            skip_reason="duplicate_exact",
                            crawl_depth=depth,
                        )
                    if RECORD_SKIPPED_IN_METADATA:
                        return _metadata_skip_row(
                            url, canonical_url, depth, "duplicate_exact", domain,
                            quality_score=int(quality.get("score") or 0),
                            quality_reason=str(quality.get("reason") or ""),
                            content_hash=content_hash
                        )
                    return None

                if ENABLE_NEAR_DUP_INGESTION and ext in {".html", ".htm", ".txt", ".md"}:
                    preview_text = str(quality.get("preview_text") or "")
                    signature = _simhash64(_tokenize_for_simhash(preview_text[:8000]))
                    if signature:
                        for existing in near_dup_signatures:
                            if _hamming_distance(signature, existing) <= NEAR_DUP_SIMHASH_DISTANCE:
                                with stats_lock:
                                    _increment_counter(stats, "skipped_dup_near")
                                with state_lock:
                                    _update_state_entry(
                                        state,
                                        canonical_url,
                                        url=url,
                                        domain=domain,
                                        last_status="skipped_dup_near",
                                        last_http_code=response.status_code,
                                        content_hash=content_hash,
                                        quality_score=int(quality.get("score") or 0),
                                        quality_reason=str(quality.get("reason") or ""),
                                        skip_reason="duplicate_near",
                                        crawl_depth=depth,
                                    )
                                if RECORD_SKIPPED_IN_METADATA:
                                    return _metadata_skip_row(
                                        url, canonical_url, depth, "duplicate_near", domain,
                                        quality_score=int(quality.get("score") or 0),
                                        quality_reason=str(quality.get("reason") or ""),
                                        content_hash=content_hash
                                    )
                                return None
                        near_dup_signatures.append(signature)

                seen_hashes.add(content_hash)

            filename = generate_filename(url, ext)
            path = save_file(content, filename)
            fetched_at = now_iso()
            etag = str(response.headers.get("ETag") or "").strip()
            last_modified = str(response.headers.get("Last-Modified") or "").strip()

            with stats_lock:
                _increment_counter(stats, "downloaded")
                stats["per_domain"][domain] = int(stats["per_domain"].get(domain, 0)) + 1
                stats["per_subdomain"][subdomain] = int(stats["per_subdomain"].get(subdomain, 0)) + 1

            with state_lock:
                _update_state_entry(
                    state,
                    canonical_url,
                    url=url,
                    domain=domain,
                    last_status="downloaded",
                    last_http_code=response.status_code,
                    etag=etag,
                    last_modified=last_modified,
                    content_hash=content_hash,
                    file_path=path,
                    last_fetched_at=fetched_at,
                    quality_score=int(quality.get("score") or 0),
                    quality_reason=str(quality.get("reason") or ""),
                    skip_reason="",
                    crawl_depth=depth,
                )

            time.sleep(0.25)
            return {
                "url": url,
                "canonical_url": canonical_url,
                "domain": domain,
                "subdomain": subdomain,
                "file": path,
                "depth": depth,
                "hash": content_hash,
                "content_hash": content_hash,
                "content_type": content_type,
                "is_html": ext in {".html", ".htm"},
                "download_quality_score": int(quality.get("score", 0)),
                "download_quality_reason": str(quality.get("reason", "")),
                "download_keyword_hits": quality.get("keyword_hits", []),
                "download_quality_metrics": quality.get("metrics", {}),
                "etag": etag,
                "last_modified": last_modified,
                "fetched_at": fetched_at,
                "skip_reason": "",
                "status": "downloaded",
            }

        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            with stats_lock:
                _increment_counter(stats, "download_errors")
            time.sleep(1)

    with state_lock:
        _update_state_entry(
            state,
            canonical_url,
            url=url,
            domain=domain,
            last_status="download_error",
            skip_reason="download_error",
            crawl_depth=depth,
        )
    if RECORD_SKIPPED_IN_METADATA:
        return _metadata_skip_row(url, canonical_url, depth, "download_error", domain)
    return None


# ==============================
# CRAWLER
# ==============================
def _crawl_with_legacy_fetch(seeds):
    q = queue.PriorityQueue()
    visited: Set[str] = set()
    metadata_rows: List[Dict] = []
    downloaded_rows: List[Dict] = []

    state = load_ingestion_state()
    state_lock = threading.Lock()
    dedup_lock = threading.Lock()
    queue_lock = threading.Lock()
    stats_lock = threading.Lock()

    seen_hashes: Set[str] = set()
    near_dup_signatures: List[int] = []

    for entry in state.get("urls", {}).values():
        if not isinstance(entry, dict):
            continue
        content_hash = str(entry.get("content_hash") or "").strip()
        if content_hash:
            seen_hashes.add(content_hash)

    stats: Dict = {
        "started_at": now_iso(),
        "crawled": 0,
        "downloaded": 0,
        "not_modified": 0,
        "refresh_skipped": 0,
        "skipped_quality": 0,
        "skipped_dup_exact": 0,
        "skipped_dup_near": 0,
        "download_errors": 0,
        "http_errors": 0,
        "quota_skipped_domain": 0,
        "quota_skipped_subdomain": 0,
        "per_domain": {},
        "per_subdomain": {},
        "seeds_count": len(seeds or []),
    }
    queued_per_domain: Dict[str, int] = defaultdict(int)

    def put_url(url_value: str, depth_value: int) -> None:
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        with queue_lock:
            pressure = int(stats["per_domain"].get(domain, 0)) + int(queued_per_domain.get(domain, 0))
            priority = score_url(url_value, depth_value) + (pressure * DOMAIN_PRIORITY_STEP)
            q.put((priority, depth_value, url_value))
            queued_per_domain[domain] += 1

    def pop_url() -> Tuple[int, int, str]:
        priority, depth_value, url_value = q.get()
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        with queue_lock:
            queued_per_domain[domain] = max(0, int(queued_per_domain.get(domain, 0)) - 1)
        return priority, depth_value, url_value

    def can_fetch(url_value: str) -> Tuple[bool, str]:
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        subdomain = subdomain_bucket(host)
        with stats_lock:
            if int(stats["per_domain"].get(domain, 0)) >= MAX_URLS_PER_DOMAIN:
                return False, "domain_quota_exceeded"
            if int(stats["per_subdomain"].get(subdomain, 0)) >= MAX_URLS_PER_SUBDOMAIN:
                return False, "subdomain_quota_exceeded"
        return True, ""

    accepted_seeds: List[str] = []
    for seed in seeds:
        candidate = should_accept_url(seed)
        if candidate:
            accepted_seeds.append(candidate)

    discovered_urls: List[str] = []
    if ENABLE_SITEMAP_DISCOVERY:
        for seed in accepted_seeds:
            discovered_urls.extend(_discover_seed_urls(seed))
        discovered_urls = discovered_urls[:MAX_SITEMAP_URLS]

    for url in list(dict.fromkeys([*accepted_seeds, *discovered_urls])):
        with visited_lock:
            if url in visited:
                continue
            visited.add(url)
        put_url(url, 0)

    active_workers = 0
    condition = threading.Condition()

    def worker():
        nonlocal active_workers
        while True:
            with condition:
                while q.empty() and active_workers > 0 and int(stats.get("downloaded", 0)) < MAX_TOTAL_URLS:
                    condition.wait()
                
                if int(stats.get("downloaded", 0)) >= MAX_TOTAL_URLS:
                    return
                if q.empty() and active_workers == 0:
                    return
                
                _, depth, url = pop_url()
                active_workers += 1

            with stats_lock:
                _increment_counter(stats, "crawled")

            can_continue, quota_reason = can_fetch(url)
            if not can_continue:
                canonical_url = clean_url(url)
                domain = domain_bucket(urlparse(canonical_url).netloc)
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="quota_skipped",
                        skip_reason=quota_reason,
                        crawl_depth=depth,
                    )
                with stats_lock:
                    if quota_reason == "domain_quota_exceeded":
                        _increment_counter(stats, "quota_skipped_domain")
                    else:
                        _increment_counter(stats, "quota_skipped_subdomain")
                if RECORD_SKIPPED_IN_METADATA:
                    with results_lock:
                        metadata_rows.append(_metadata_skip_row(url, canonical_url, depth, quota_reason, domain))

                with condition:
                    active_workers -= 1
                    q.task_done()
                    condition.notify_all()
                continue

            # Téléchargement hors du verrou principal
            res = download(
                url=url,
                depth=depth,
                seen_hashes=seen_hashes,
                near_dup_signatures=near_dup_signatures,
                state=state,
                state_lock=state_lock,
                dedup_lock=dedup_lock,
                stats=stats,
                stats_lock=stats_lock,
            )
            
            if res:
                is_downloaded = str(res.get("status", "")) == "downloaded"
                with results_lock:
                    metadata_rows.append(res)
                    if is_downloaded:
                        downloaded_rows.append(res)
                    # Sauvegarde périodique (Checkpointing)
                    if len(downloaded_rows) % 100 == 0 and downloaded_rows:
                        save_metadata_safely(metadata_rows)
                        save_ingestion_state(state)
                        logger.info("Checkpoint: %s documents téléchargés.", len(downloaded_rows))

                if is_downloaded and depth < MAX_DEPTH and res.get("is_html"):
                    links = []
                    try:
                        with open(res["file"], "rb") as fp:
                            links = extract_links(fp.read(), url)
                    except Exception as exc:
                        logger.warning("Link extraction failed for %s: %s", res["file"], exc)

                    for link in links:
                        with visited_lock:
                            if link not in visited:
                                visited.add(link)
                                added = True
                            else:
                                added = False
                        
                        if added:
                            put_url(link, depth + 1)

            with condition:
                active_workers -= 1
                q.task_done()
                condition.notify_all()
                if int(stats.get("downloaded", 0)) % 100 == 0 and int(stats.get("downloaded", 0)) > 0:
                    logger.info("Progress: %s/%s", stats.get("downloaded", 0), MAX_TOTAL_URLS)

    # Démarrage des workers
    threads = []
    for _ in range(INGESTION_THREADS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    try:
        # Attente de la fin de tous les threads
        for t in threads:
            t.join()

        # Sauvegarde finale
        save_metadata_safely(metadata_rows)
        save_ingestion_state(state)

        report_payload = {
            "started_at": stats.get("started_at"),
            "finished_at": now_iso(),
            "settings": {
                "max_total_urls": MAX_TOTAL_URLS,
                "max_depth": MAX_DEPTH,
                "ingestion_threads": INGESTION_THREADS,
                "max_urls_per_domain": MAX_URLS_PER_DOMAIN,
                "max_urls_per_subdomain": MAX_URLS_PER_SUBDOMAIN,
                "incremental_fetch": INCREMENTAL_FETCH,
                "refresh_mode": REFRESH_MODE,
                "near_dup_enabled": ENABLE_NEAR_DUP_INGESTION,
                "near_dup_simhash_distance": NEAR_DUP_SIMHASH_DISTANCE,
                "min_download_quality_score": MIN_DOWNLOAD_QUALITY_SCORE,
                "sitemap_discovery": ENABLE_SITEMAP_DISCOVERY,
            },
            "metrics": {
                "crawled": int(stats.get("crawled", 0)),
                "downloaded": int(stats.get("downloaded", 0)),
                "skipped_quality": int(stats.get("skipped_quality", 0)),
                "skipped_dup_exact": int(stats.get("skipped_dup_exact", 0)),
                "skipped_dup_near": int(stats.get("skipped_dup_near", 0)),
                "not_modified": int(stats.get("not_modified", 0)),
                "refresh_skipped": int(stats.get("refresh_skipped", 0)),
                "download_errors": int(stats.get("download_errors", 0)),
                "http_errors": int(stats.get("http_errors", 0)),
                "quota_skipped_domain": int(stats.get("quota_skipped_domain", 0)),
                "quota_skipped_subdomain": int(stats.get("quota_skipped_subdomain", 0)),
            },
            "per_domain": dict(
                sorted(stats.get("per_domain", {}).items(), key=lambda item: item[1], reverse=True)
            ),
            "per_subdomain_top20": dict(
                sorted(stats.get("per_subdomain", {}).items(), key=lambda item: item[1], reverse=True)[:20]
            ),
            "metadata_rows_count": len(metadata_rows),
            "downloaded_rows_count": len(downloaded_rows),
        }
        report_path = str(update_offline_pipeline_report("ingestion", report_payload))
        logger.info("Ingestion report updated: %s", report_path)
        return downloaded_rows
    finally:
        cleanup_playwright_resources()


def crawl(seeds):
    backend = (INGESTION_BACKEND or "scrapy").lower()
    if backend == "scrapy":
        try:
            from .scrapy_ingestion import crawl_with_scrapy

            return crawl_with_scrapy(seeds or DEFAULT_SEEDS)
        except Exception as exc:
            logger.warning("Backend Scrapy indisponible, fallback legacy requests actif: %s", exc)
    return _crawl_with_legacy_fetch(seeds or DEFAULT_SEEDS)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("Crawler RAG demarre")
    crawl(DEFAULT_SEEDS)
