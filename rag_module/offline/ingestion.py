# rag_module/ingestion.py
import os
import re
import json
import hashlib
import logging
import queue
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ==============================
# CONFIGURATION
# ==============================
RAW_DATA_PATH = "data_storage/raw"
META_PATH = "data_storage/index/metadata.json"

TIMEOUT = 20
RETRIES = 3
PLAYWRIGHT_WAIT = 2500

MAX_DEPTH = 4
MAX_TOTAL_URLS = 800

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

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

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


def is_allowed_domain(hostname: str) -> bool:
    host = (hostname or "").lower()

    if host in ALLOWED_DOMAINS:
        return True

    for domain in ALLOWED_DOMAIN_SUFFIXES:
        d = domain.lower()
        if host == d or host.endswith(f".{d}"):
            return True
    return False


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
            return ".docx" if disposition_ext == ".doc" else disposition_ext

    if "text/html" in lowered_type:
        return ".html"
    if "application/pdf" in lowered_type:
        return ".pdf"
    if "application/msword" in lowered_type:
        return ".docx"
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


# ==============================
# LOCKS MULTITHREADING
# ==============================
seen_hashes_lock = threading.Lock()
visited_lock = threading.Lock()
results_lock = threading.Lock()

def save_metadata_safely(results_list):
    """Sauvegarde atomique pour éviter la corruption de fichier."""
    temp_path = META_PATH + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, META_PATH)
    except Exception as exc:
        logger.error("Erreur lors de la sauvegarde metadata: %s", exc)

# ==============================
# DOWNLOAD
# ==============================
def download(url, depth, seen_hashes):
    for _ in range(RETRIES):
        try:
            timeout = TIMEOUT if depth < 2 else 10
            r = requests.get(url, headers=HEADERS, timeout=timeout)

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            content = r.content
            content_type = r.headers.get("Content-Type", "")
            lowered_content_type = content_type.lower()

            # Playwright fallback intelligent
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

            file_hash = compute_hash(content)
            
            with seen_hashes_lock:
                if file_hash in seen_hashes:
                    return None
                seen_hashes.add(file_hash)

            ext = infer_extension(
                url,
                content_type,
                content=content,
                content_disposition=r.headers.get("Content-Disposition", ""),
            )
            filename = generate_filename(url, ext)
            path = save_file(content, filename)
            
            # Politesse : Pause entre les téléchargements
            time.sleep(0.5)

            return {
                "url": url,
                "file": path,
                "depth": depth,
                "hash": file_hash,
                "content_type": content_type,
                "is_html": ext in {".html", ".htm"},
            }

        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            time.sleep(1)

    return None


# ==============================
# CRAWLER
# ==============================
def crawl(seeds):
    q = queue.PriorityQueue()
    visited = set()
    results = []
    seen_hashes = set()

    for url in seeds:
        u = should_accept_url(url)
        if u:
            priority = score_url(u, 0)
            q.put((priority, 0, u))
            visited.add(u)

    active_workers = 0
    condition = threading.Condition()
    num_threads = 5

    def worker():
        nonlocal active_workers
        while True:
            with condition:
                while q.empty() and active_workers > 0 and len(results) < MAX_TOTAL_URLS:
                    condition.wait()
                
                if len(results) >= MAX_TOTAL_URLS:
                    return
                if q.empty() and active_workers == 0:
                    return
                
                _, depth, url = q.get()
                active_workers += 1

            # Téléchargement hors du verrou principal
            res = download(url, depth, seen_hashes)
            
            if res:
                with results_lock:
                    if len(results) < MAX_TOTAL_URLS:
                        results.append(res)
                        # Sauvegarde périodique (Checkpointing)
                        if len(results) % 50 == 0:
                            save_metadata_safely(results)
                            logger.info("Checkpoint: %s documents sauvegardés.", len(results))

                if depth < MAX_DEPTH:
                    links = []
                    if res.get("is_html"):
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
                            priority = score_url(link, depth + 1)
                            q.put((priority, depth + 1, link))

            with condition:
                active_workers -= 1
                q.task_done()
                condition.notify_all()
                if res and len(results) <= MAX_TOTAL_URLS:
                    logger.info("Progress: %s/%s", len(results), MAX_TOTAL_URLS)

    # Démarrage des workers
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    try:
        # Attente de la fin de tous les threads
        for t in threads:
            t.join()

        # Sauvegarde finale
        save_metadata_safely(results)
        return results
    finally:
        cleanup_playwright_resources()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("Crawler RAG demarre")
    crawl(DEFAULT_SEEDS)
