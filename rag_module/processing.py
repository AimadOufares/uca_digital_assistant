# rag_module/processing.py
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict
import logging

from bs4 import BeautifulSoup
import pdfplumber
import docx
from tiktoken import encoding_for_model
from html import unescape
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== CONFIG =====================
RAW_PATH = "data_storage/raw"
PROCESSED_PATH = "data_storage/processed"
CACHE_FILE = "data_storage/file_cache.json"

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 80
LLM_MODEL = "gpt-4o-mini"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================== GLOBAL TOKENIZER =====================
ENCODER = encoding_for_model(LLM_MODEL)

# ===================== UTILITAIRES =====================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def hash_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ===================== EXTRACTION =====================
def extract_text_html(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            tag.decompose()
        main = soup.find(['main','article']) or soup.body or soup
        text = ""
        for tag in main.find_all(['h1','h2','h3','p','li']):
            text += tag.get_text(separator=' ') + "\n"
        return text
    except Exception as e:
        logger.error(f"Erreur extraction HTML {file_path}: {e}")
        return ""

def extract_text_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
    except Exception as e:
        logger.error(f"Erreur extraction PDF {file_path}: {e}")
    return text

def extract_text_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Erreur extraction DOCX {file_path}: {e}")
        return ""

def extract_text_plain(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Erreur extraction texte brut {file_path}: {e}")
        return ""

# ===================== SEMANTIC CHUNKING =====================
def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text)

def split_into_chunks(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    if not text.strip():
        return []

    sentences = split_sentences(text)
    chunks = []
    current = ""

    for s in sentences:
        candidate = (current + " " + s).strip()
        if len(ENCODER.encode(candidate)) <= chunk_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    # Ajouter overlap
    final_chunks = []
    for i, c in enumerate(chunks):
        final_chunks.append(c)
        if i < len(chunks) - 1:
            overlap_chunk = " ".join(ENCODER.decode(
                ENCODER.encode(c)[-overlap:] + ENCODER.encode(chunks[i+1])[:overlap]
            ))
            final_chunks.append(overlap_chunk)
    # Filtrer chunks trop petits
    return [c.strip() for c in final_chunks if len(c.strip().split()) > 5]

# ===================== PRÉTRAITEMENT FICHIER =====================
def preprocess_file(file_path: str) -> List[Dict]:
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()

    extractors = {
        ".html": extract_text_html,
        ".htm": extract_text_html,
        ".pdf": extract_text_pdf,
        ".docx": extract_text_docx,
        ".txt": extract_text_plain,
        ".md": extract_text_plain,
    }

    extractor = extractors.get(ext)
    if not extractor:
        logger.warning(f"Format non supporté : {ext} → {file_path}")
        return []

    text = extractor(file_path)
    text = clean_text(text)
    if len(text) < 50:
        logger.info(f"Fichier ignoré (trop court) : {file_path}")
        return []

    chunks = split_into_chunks(text)
    result = []
    file_hash = hash_file(file_path)

    for idx, chunk in enumerate(chunks):
        chunk_hash = hash_text(chunk)
        try:
            language = detect(chunk)
        except:
            language = "unknown"

        metadata = {
            "source": file_path,
            "source_hash": file_hash,
            "type": ext[1:],
            "chunk_hash": chunk_hash,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "date_ingestion": datetime.now(timezone.utc).isoformat(),
            "chunk_tokens": len(ENCODER.encode(chunk)),
            "language": language,
            "length_words": len(chunk.split()),
        }

        result.append({"text": chunk, "metadata": metadata})

    logger.info(f"{len(chunks)} chunks générés depuis {file_path}")
    return result

# ===================== CACHE =====================
def load_cache() -> Dict[str,str]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str,str]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ===================== PRÉTRAITEMENT GLOBAL =====================
def preprocess_all(raw_path: str = RAW_PATH, processed_path: str = PROCESSED_PATH, skip_existing: bool = True, max_workers: int = 6) -> None:
    os.makedirs(processed_path, exist_ok=True)
    cache = load_cache()
    seen_chunk_hashes = set()
    processed_count = 0

    file_list = []
    for root, _, files in os.walk(raw_path):
        for file in files:
            if file.startswith('.') or file.startswith('~') or file.endswith('.tmp'):
                continue
            file_list.append(os.path.join(root, file))

    logger.info(f"Démarrage du prétraitement RAG - {len(file_list)} fichiers détectés")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(preprocess_file, f): f for f in file_list}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                file_chunks = future.result()
                file_hash = hash_file(file_path)

                # Skip si fichier inchangé
                if skip_existing and file_path in cache and cache[file_path] == file_hash:
                    continue
                cache[file_path] = file_hash

                for chunk in file_chunks:
                    chash = chunk['metadata']['chunk_hash']
                    if chash in seen_chunk_hashes:
                        continue
                    seen_chunk_hashes.add(chash)

                    out_path = os.path.join(processed_path, f"{chash}.json")
                    if skip_existing and os.path.exists(out_path):
                        continue
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    processed_count += 1

            except Exception as e:
                logger.error(f"Erreur critique sur {file_path}: {e}", exc_info=True)

    save_cache(cache)
    logger.info(f"Prétraitement terminé ! {processed_count} nouveaux chunks sauvegardés dans {processed_path}")

# ===================== POINT D’ENTRÉE =====================
if __name__ == "__main__":
    preprocess_all()