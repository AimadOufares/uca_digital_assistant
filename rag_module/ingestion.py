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
from urllib.parse import urljoin, urlparse

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

ALLOWED_DOMAINS = ["uca.ma", "fstg-marrakech.ac.ma"]

DEFAULT_SEEDS = [
    "https://www.uca.ma",
    "https://www.uca.ma/fr",
    "https://fsjes.uca.ma",
    "https://flsh.uca.ma",
    "https://www.uca.ma/fssm",
    "https://www.fmpm.uca.ma/",
    "https://www.fstg-marrakech.ac.ma/",
    "https://ensa-marrakech.uca.ma/",
    "https://www.uca.ma/encg",
    "https://www.uca.ma/ens",
    "https://www.uca.ma/flam",
]

KEYWORDS = ["formation", "master", "licence", "cours", "pdf", "recherche"]

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


def get_browser():
    if not hasattr(thread_local, "browser"):
        pw = sync_playwright().start()
        thread_local.playwright = pw
        thread_local.browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
    return thread_local.browser


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
    return url.split("?")[0].strip()


def is_allowed_domain(hostname: str) -> bool:
    host = (hostname or "").lower()
    for domain in ALLOWED_DOMAINS:
        d = domain.lower()
        if host == d or host.endswith(f".{d}"):
            return True
    return False


def compute_hash(content):
    return hashlib.md5(content).hexdigest()


def infer_extension(url: str, content_type: str) -> str:
    lowered_type = (content_type or "").lower()
    path_ext = Path(urlparse(url).path).suffix.lower()
    if "text/html" in lowered_type:
        return ".html"
    if "application/pdf" in lowered_type:
        return ".pdf"
    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in lowered_type:
        return ".docx"
    if "text/plain" in lowered_type:
        return ".txt"
    if path_ext in {".html", ".htm", ".pdf", ".docx", ".txt", ".md"}:
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


def is_relevant(url):
    return any(k in url.lower() for k in KEYWORDS)


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

    if parsed.scheme not in ["http", "https"]:
        return None

    if not is_allowed_domain(parsed.netloc):
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
            if file_hash in seen_hashes:
                return None

            seen_hashes.add(file_hash)

            ext = infer_extension(url, content_type)
            filename = generate_filename(url, ext)
            path = save_file(content, filename)

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
            priority = 0 if is_relevant(u) else 1
            q.put((priority, 0, u))
            visited.add(u)

    while not q.empty() and len(results) < MAX_TOTAL_URLS:
        _, depth, url = q.get()

        res = download(url, depth, seen_hashes)
        if not res:
            continue

        results.append(res)

        if depth >= MAX_DEPTH:
            continue

        links = []
        if res.get("is_html"):
            try:
                with open(res["file"], "rb") as fp:
                    links = extract_links(fp.read(), url)
            except Exception as exc:
                logger.warning("Link extraction failed for %s: %s", res["file"], exc)

        for link in links:
            if link not in visited:
                visited.add(link)
                priority = 0 if is_relevant(link) else 1
                q.put((priority, depth + 1, link))

        logger.info("Progress: %s/%s", len(results), MAX_TOTAL_URLS)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("Crawler RAG demarre")
    crawl(DEFAULT_SEEDS)
