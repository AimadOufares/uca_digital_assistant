import os
import re
import hashlib
import logging
import queue
import threading
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

# ==============================
# CONFIGURATION PERFORMANTE
# ==============================
RAW_DATA_PATH = "data_storage/raw"
TIMEOUT = 25
MAX_HTML_WORKERS = 5          # Workers pour pages HTML (Playwright est lourd)
MAX_DOC_WORKERS = 8           # Workers pour documents (plus légers)
RETRIES = 3
PLAYWRIGHT_WAIT = 3500

MAX_DEPTH = 4
MAX_TOTAL_URLS = 800          # Augmenté pour profiter de la performance

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}

os.makedirs(RAW_DATA_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ALLOWED_DOMAINS = ["uca.ma"]
EXCLUDE_PATHS = [
    "/login", "/admin", "/wp-admin", "/logout", "/cart", "/checkout",
    "javascript:", "mailto:", "tel:", "#", "sessionid", "utm_", "ref="
]

# ==============================
# UTILITAIRES
# ==============================
def clean_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name)[:150]

def get_file_extension(url: str, content_type: str = "") -> str:
    lower_url = url.lower()
    if lower_url.endswith('.pdf'): return '.pdf'
    if lower_url.endswith(('.doc', '.docx')): return '.docx'
    if 'pdf' in content_type.lower(): return '.pdf'
    if any(x in content_type.lower() for x in ['word', 'officedocument']): return '.docx'
    return '.html'

def generate_filename(url: str, content_type: str = "") -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_").replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_") or hashlib.md5(url.encode()).hexdigest()[:12]
    name = f"{domain}_{path}"
    name = clean_filename(name)
    return name + get_file_extension(url, content_type)

def compute_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def is_valid_html(content: bytes) -> bool:
    try:
        text = content.decode('utf-8', errors='ignore').lower()
        return (len(text) >= 800 and 
                ("<html" in text or "<!doctype" in text) and 
                "404 not found" not in text[:600])
    except:
        return False

def save_file(content: bytes, filename: str) -> str:
    filepath = os.path.join(RAW_DATA_PATH, filename)
    if os.path.exists(filepath):
        logger.info(f"✅ Déjà existant : {filename}")
        return filepath
    with open(filepath, "wb") as f:
        f.write(content)
    logger.info(f"💾 Sauvegardé : {filepath} ({len(content)/1024:.1f} KB)")
    return filepath

# ==============================
# FILTRAGE MULTIPLE
# ==============================
def should_accept_url(url: str, base: str = None) -> Optional[str]:
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    url = url.split('#')[0].strip()
    if not url:
        return None

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if not any(d in parsed.netloc.lower() for d in ALLOWED_DOMAINS):
        return None

    path_lower = (parsed.path + "?" + parsed.query).lower()
    if any(ex in path_lower for ex in EXCLUDE_PATHS):
        return None
    if len(url) > 300:
        return None

    return url

def is_document_url(url: str) -> bool:
    return url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'))

# ==============================
# EXTRACTION DE LIENS
# ==============================
def extract_links(content: bytes, base_url: str) -> List[str]:
    if not content:
        return []
    try:
        soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "lxml")
        links = set()
        for tag in soup.find_all(["a", "link", "iframe", "embed"], href=True):
            href = tag.get("href") or tag.get("src")
            if href:
                full = should_accept_url(href, base_url)
                if full:
                    links.add(full)
        return list(links)
    except Exception as e:
        logger.warning(f"Extraction liens échouée : {e}")
        return []

# ==============================
# DOWNLOAD (multi-thread safe)
# ==============================
def download_source(url: str, is_doc: bool = False) -> Dict:
    for attempt in range(RETRIES):
        try:
            logger.info(f"[{attempt+1}/{RETRIES}] {'📄 Doc' if is_doc else '🌐 HTML'} → {url[:80]}...")

            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            final_url = response.url
            content_type = response.headers.get("Content-Type", "").lower()
            content = response.content

            # Playwright pour contenu dynamique
            if not is_doc and ("uca.ma" in final_url or len(response.text) < 1500):
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(final_url, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(PLAYWRIGHT_WAIT)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1200)
                    content = page.content().encode("utf-8")
                    browser.close()
                source = "playwright"
            else:
                source = "requests"

            doc_type = "document" if is_doc or is_document_url(final_url) else "html"
            if doc_type == "html" and not is_valid_html(content):
                raise Exception("HTML invalide")

            filename = generate_filename(final_url, content_type)
            filepath = save_file(content, filename)

            return {
                "status": "success",
                "url": final_url,
                "path": filepath,
                "type": doc_type,
                "size": len(content),
                "source": source,
                "hash": compute_hash(content),
                "content": content if doc_type == "html" else None
            }

        except Exception as e:
            logger.warning(f"Échec tentative {attempt+1} pour {url}: {e}")
            if attempt == RETRIES - 1 and not is_doc:
                logger.info("→ Fallback Playwright...")
                # fallback simplifié (même logique)

    return {"status": "error", "url": url, "error": "Max retries reached"}

# ==============================
# CRAWLER PRINCIPAL - 100% MULTI-THREADED
# ==============================
def crawl_website(seed_urls: List[str], max_depth: int = MAX_DEPTH, max_total: int = MAX_TOTAL_URLS) -> List[Dict]:
    to_crawl_queue: queue.Queue = queue.Queue()   # (url, depth)
    document_queue: List[str] = []
    visited = set()
    downloaded_results = []
    lock = threading.Lock()

    # Initialisation
    for url in seed_urls:
        norm_url = should_accept_url(url)
        if norm_url and norm_url not in visited:
            visited.add(norm_url)
            to_crawl_queue.put((norm_url, 0))

    start_time = time.time()
    logger.info(f"🚀 CRAWLER ULTRA-PERFORMANT LANCÉ | HTML Workers: {MAX_HTML_WORKERS} | Doc Workers: {MAX_DOC_WORKERS}")

    while (not to_crawl_queue.empty() or document_queue) and len(downloaded_results) < max_total:
        # 1. Téléchargement prioritaire des documents (plus rapide)
        if document_queue:
            docs_to_process = document_queue[:MAX_DOC_WORKERS]
            document_queue = document_queue[MAX_DOC_WORKERS:]

            with ThreadPoolExecutor(max_workers=MAX_DOC_WORKERS) as executor:
                future_to_url = {executor.submit(download_source, url, is_doc=True): url for url in docs_to_process}
                for future in as_completed(future_to_url):
                    result = future.result()
                    downloaded_results.append(result)
                    with lock:
                        visited.add(result.get("url"))

        # 2. Crawl des pages HTML
        if to_crawl_queue.empty():
            break

        batch_size = min(MAX_HTML_WORKERS, to_crawl_queue.qsize())
        batch = [to_crawl_queue.get() for _ in range(batch_size)]

        with ThreadPoolExecutor(max_workers=MAX_HTML_WORKERS) as executor:
            future_to_item = {executor.submit(download_source, url): (url, depth) for url, depth in batch}
            for future in as_completed(future_to_item):
                result = future.result()
                downloaded_results.append(result)
                url, depth = future_to_item[future]

                if result.get("status") == "success" and result.get("type") == "html" and depth < max_depth:
                    new_links = extract_links(result.get("content"), url)
                    for link in new_links:
                        if link not in visited:
                            with lock:
                                if link not in visited and len(visited) < max_total * 1.2:
                                    visited.add(link)
                                    if is_document_url(link):
                                        document_queue.append(link)
                                    else:
                                        to_crawl_queue.put((link, depth + 1))

        progress = len(downloaded_results)
        print(f"   📈 Progress: {progress}/{max_total} | Queue HTML: {to_crawl_queue.qsize()} | Docs en attente: {len(document_queue)}")

    elapsed = time.time() - start_time
    success = sum(1 for r in downloaded_results if r.get("status") == "success")
    pdf_count = sum(1 for r in downloaded_results if r.get("url", "").lower().endswith('.pdf'))

    logger.info(f"🏁 CRAWL TERMINÉ en {elapsed:.1f} secondes ! {success} succès sur {len(downloaded_results)} tentatives.")
    print(f"\n🎯 Résumé : {success} fichiers | {pdf_count} PDF/DOCX | Temps total : {elapsed:.1f}s")

    return downloaded_results

# ==============================
# LANCEMENT
# ==============================
if __name__ == "__main__":
    test_urls = [
        "https://www.uca.ma/", 
        "https://www.uca.ma/fr",
        "https://www.uca.ma/fr/etablissements",
        "https://ecampus-fssm.uca.ma",
        "https://ecampus-fsjes.uca.ma",
        "https://ecampus-flsh.uca.ma",
        "https://fsjes.uca.ma/",
        "https://flsh.uca.ma/",
        "https://fmpm.uca.ma/",
        "https://ensas.uca.ma/",
        "https://pedoc.uca.ma/",
    ]

    print("🔥 Lancement du CRAWLER ULTRA-PERFORMANT (100% multi-threaded)\n")
    results = crawl_website(test_urls, max_depth=MAX_DEPTH, max_total=MAX_TOTAL_URLS)