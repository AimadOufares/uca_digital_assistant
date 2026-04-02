# rag_module/ingestion.py
import os
import re
import json
import hashlib
import logging
import queue
import threading
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ==============================
# CONFIGURATION
# ==============================
RAW_DATA_PATH = "data_storage/raw"
META_PATH = "data_storage/index/metadata.json"

TIMEOUT = 20
MAX_HTML_WORKERS = 5
MAX_DOC_WORKERS = 6
RETRIES = 3
PLAYWRIGHT_WAIT = 2500

MAX_DEPTH = 4
MAX_TOTAL_URLS = 800

ALLOWED_DOMAINS = ["uca.ma"]

KEYWORDS = ["formation", "master", "licence", "cours", "pdf", "recherche"]

EXCLUDE_PATHS = [
    "/login", "/admin", "/wp-admin", "/logout",
    "javascript:", "mailto:", "tel:", "#",
    "sessionid", "utm_", "ref="
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
        thread_local.browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
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
    except:
        return None

# ==============================
# UTILITAIRES
# ==============================
def clean_url(url):
    return url.split("?")[0].strip()

def compute_hash(content):
    return hashlib.md5(content).hexdigest()

def generate_filename(url):
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = re.sub(r'[^\w\-_.]', '_', name)
    return name[:150] + ".html"

def save_file(content, filename):
    path = os.path.join(RAW_DATA_PATH, filename)
    if not os.path.exists(path):
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

    if not any(d in parsed.netloc for d in ALLOWED_DOMAINS):
        return None

    if any(x in url.lower() for x in EXCLUDE_PATHS):
        return None

    if len(url) > 300:
        return None

    return url

def is_document(url):
    return url.endswith((".pdf", ".docx", ".pptx", ".xlsx"))

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
seen_hashes = set()

def download(url, depth):
    for _ in range(RETRIES):
        try:
            timeout = TIMEOUT if depth < 2 else 10
            r = requests.get(url, headers=HEADERS, timeout=timeout)

            if r.status_code != 200:
                raise Exception()

            content = r.content

            # Playwright fallback intelligent
            if len(content) < 1200 or b"<script" in content:
                pw_content = fetch_with_playwright(url)
                if pw_content:
                    content = pw_content

            file_hash = compute_hash(content)

            if file_hash in seen_hashes:
                return None

            seen_hashes.add(file_hash)

            filename = generate_filename(url)
            path = save_file(content, filename)

            return {
                "url": url,
                "file": path,
                "depth": depth,
                "hash": file_hash
            }

        except:
            time.sleep(1)

    return None

# ==============================
# CRAWLER
# ==============================
def crawl(seeds):
    q = queue.PriorityQueue()
    visited = set()
    results = []

    for url in seeds:
        u = should_accept_url(url)
        if u:
            priority = 0 if is_relevant(u) else 1
            q.put((priority, 0, u))
            visited.add(u)

    while not q.empty() and len(results) < MAX_TOTAL_URLS:
        _, depth, url = q.get()

        res = download(url, depth)
        if not res:
            continue

        results.append(res)

        if depth >= MAX_DEPTH:
            continue

        links = extract_links(open(res["file"], "rb").read(), url)

        for link in links:
            if link not in visited:
                visited.add(link)
                priority = 0 if is_relevant(link) else 1
                q.put((priority, depth + 1, link))

        print(f"📈 {len(results)}/{MAX_TOTAL_URLS}")

    # Save metadata
    with open(META_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    seeds = [
        "https://www.uca.ma",
        "https://www.uca.ma/fr",
        "https://fsjes.uca.ma",
        "https://flsh.uca.ma"
    ]

    print("🚀 CRAWLER OPTIMISÉ RAG")
    crawl(seeds)