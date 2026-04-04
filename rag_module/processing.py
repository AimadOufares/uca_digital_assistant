# rag_module/processing.py
import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

import pdfplumber
import docx
from bs4 import BeautifulSoup
from tiktoken import encoding_for_model
from html import unescape
from langdetect import detect, LangDetectException
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== CONFIG =====================
RAW_PATH = "data_storage/raw"
PROCESSED_PATH = "data_storage/processed"
CACHE_FILE = "data_storage/cache/file_cache.json"

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 80
MIN_WORDS = 8
MIN_QUALITY_SCORE = 25          # Nouveau : seuil de qualité

LLM_MODEL = "gpt-4o-mini"

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== TOKENIZER =====================
ENCODER = encoding_for_model(LLM_MODEL)

# ===================== UTILS =====================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = unescape(text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]+', '', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_detect_lang(text: str) -> str:
    if len(text.split()) < 30:
        return "unknown"
    try:
        return detect(text[:1000])
    except LangDetectException:
        return "unknown"


def quality_score(text: str) -> int:
    """Score simple pour évaluer la qualité d’un chunk"""
    if not text:
        return 0
    length = len(text.split())
    punctuation = len(re.findall(r'[.!?]', text))
    return length + punctuation * 2


# ===================== EXTRACTION =====================
def extract_text_html(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        main = soup.find(["main", "article"]) or soup.body or soup
        texts = [
            tag.get_text(" ", strip=True)
            for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"])
            if tag.get_text(strip=True)
        ]
        return "\n\n".join(texts)
    except Exception as e:
        logger.warning(f"HTML extraction error {path}: {e}")
        return ""


def extract_text_pdf(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Texte principal
                t = page.extract_text()
                if t:
                    text_parts.append(t)
                
                # Tables
                tables = page.extract_tables()
                for table in tables:
                    if table and any(any(cell for cell in row) for row in table):
                        md_table = "\n".join(
                            [" | ".join(str(cell) if cell is not None else "" for cell in row) 
                             for row in table]
                        )
                        text_parts.append(f"\n[TABLE_PAGE_{page_num}]\n{md_table}\n[/TABLE]\n")
    except Exception as e:
        logger.warning(f"PDF extraction error {path}: {e}")
    return "\n\n".join(text_parts)


def extract_text_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.warning(f"DOCX extraction error {path}: {e}")
        return ""


def extract_text_plain(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Plain text extraction error {path}: {e}")
        return ""


# ===================== CHUNKING =====================
def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text)


def recursive_chunk(text: str, chunk_size: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    text = clean_text(text)
    if len(ENCODER.encode(text)) <= chunk_size:
        return [text] if len(text.split()) >= MIN_WORDS else []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        sentences = split_sentences(para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            sent_tokens = len(ENCODER.encode(sent))

            # Si on dépasse la taille, on sauvegarde le chunk actuel
            if current_tokens + sent_tokens > chunk_size and current:
                chunk_text = " ".join(current)
                if len(chunk_text.split()) >= MIN_WORDS and quality_score(chunk_text) >= MIN_QUALITY_SCORE:
                    chunks.append(chunk_text)
                
                # Overlap intelligent basé sur tokens
                overlap = []
                tokens_acc = 0
                for s in reversed(current):
                    t_len = len(ENCODER.encode(s))
                    if tokens_acc + t_len > overlap_tokens:
                        break
                    overlap.insert(0, s)
                    tokens_acc += t_len
                
                current = overlap
                current_tokens = len(ENCODER.encode(" ".join(current)))

            current.append(sent)
            current_tokens += sent_tokens

    # Dernier chunk
    if current:
        chunk_text = " ".join(current)
        if len(chunk_text.split()) >= MIN_WORDS and quality_score(chunk_text) >= MIN_QUALITY_SCORE:
            chunks.append(chunk_text)

    return chunks


# ===================== PROCESS FILE =====================
def preprocess_file(file_path: str) -> List[Dict]:
    """Version simplifiée : on calcule le hash à l'intérieur"""
    ext = Path(file_path).suffix.lower()
    extractors = {
        ".html": extract_text_html,
        ".pdf": extract_text_pdf,
        ".docx": extract_text_docx,
        ".txt": extract_text_plain,
        ".md": extract_text_plain,
    }

    if ext not in extractors:
        logger.warning(f"Format non supporté : {file_path}")
        return []

    raw_text = extractors[ext](file_path)
    if not raw_text or len(raw_text.split()) < 50:
        return []

    cleaned_text = clean_text(raw_text)
    if len(cleaned_text) < 100:
        return []

    doc_language = safe_detect_lang(cleaned_text)
    chunks = recursive_chunk(cleaned_text)
    file_name = Path(file_path).name
    file_hash = hash_file(file_path)   # Calculé ici une seule fois

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "text": chunk,
            "text_normalized": chunk.lower(),
            "quality": quality_score(chunk),
            "metadata": {
                "source": str(file_path),
                "source_hash": file_hash,
                "chunk_hash": hash_text(chunk),
                "index": i,
                "total_chunks": len(chunks),
                "tokens": len(ENCODER.encode(chunk)),
                "language": doc_language,
                "file_name": file_name,
                "file_type": ext,
                "is_table": ("[TABLE" in chunk) or ("TABLE_PAGE_" in chunk),
                "date_processed": datetime.now(timezone.utc).isoformat()
            }
        })
    return results


# ===================== CACHE =====================
def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "files" in raw and isinstance(raw["files"], dict):
                files = {}
                for path, entry in raw["files"].items():
                    if not isinstance(entry, dict):
                        continue
                    files[path] = {
                        "file_hash": entry.get("file_hash", ""),
                        "chunk_hashes": list(dict.fromkeys(entry.get("chunk_hashes", []))),
                    }
                return {"version": 2, "files": files}

            # Compatibilite ancien format: {path: file_hash}
            if isinstance(raw, dict):
                files = {}
                for path, file_hash in raw.items():
                    if isinstance(path, str) and isinstance(file_hash, str):
                        files[path] = {"file_hash": file_hash, "chunk_hashes": []}
                return {"version": 2, "files": files}
        except Exception:
            return {"version": 2, "files": {}}
    return {"version": 2, "files": {}}


def save_cache(cache: Dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _chunk_refcounts(file_records: Dict[str, Dict]) -> Dict[str, int]:
    refcounts: Dict[str, int] = {}
    for record in file_records.values():
        hashes = set(record.get("chunk_hashes", []))
        for ch in hashes:
            refcounts[ch] = refcounts.get(ch, 0) + 1
    return refcounts


def _delete_chunk_file_if_unreferenced(
    chunk_hash: str,
    refcounts: Dict[str, int],
    seen_chunks: set,
) -> bool:
    if refcounts.get(chunk_hash, 0) > 0:
        return False
    path = os.path.join(PROCESSED_PATH, f"{chunk_hash}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            return False
    seen_chunks.discard(chunk_hash)
    return True


# ===================== MAIN =====================
def preprocess_all():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    cache = load_cache()
    file_records: Dict[str, Dict] = cache.get("files", {})
    seen_chunks = {Path(f).stem for f in os.listdir(PROCESSED_PATH) if f.endswith(".json")}
    refcounts = _chunk_refcounts(file_records)

    files = [
        os.path.join(root, f)
        for root, _, fs in os.walk(RAW_PATH)
        for f in fs if not f.startswith(".")
    ]

    logger.info(f"{len(files)} fichiers détectés dans {RAW_PATH}")

    # Nettoyage des entrées de cache dont le fichier source n'existe plus.
    deleted_sources = [p for p in list(file_records.keys()) if not os.path.exists(p)]
    removed_chunks = 0
    for path in deleted_sources:
        record = file_records.pop(path, {})
        for ch in set(record.get("chunk_hashes", [])):
            if ch in refcounts:
                refcounts[ch] -= 1
                if refcounts[ch] <= 0:
                    refcounts.pop(ch, None)
            if _delete_chunk_file_if_unreferenced(ch, refcounts, seen_chunks):
                removed_chunks += 1
    if deleted_sources:
        logger.info(f"Nettoyage sources supprimées: {len(deleted_sources)} source(s), {removed_chunks} chunk(s) supprimé(s).")

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_path = {}
        for f in files:
            file_hash = hash_file(f)
            record = file_records.get(f, {})
            old_hashes = record.get("chunk_hashes", [])
            has_all_chunks = all(
                os.path.exists(os.path.join(PROCESSED_PATH, f"{ch}.json")) for ch in old_hashes
            )
            if record.get("file_hash") == file_hash and has_all_chunks:
                logger.info(f"Skip (inchangé) → {Path(f).name}")
                continue
            future_to_path[executor.submit(preprocess_file, f)] = (f, file_hash)

        for future in as_completed(future_to_path):
            path, file_hash = future_to_path[future]
            try:
                chunks = future.result()
                previous_hashes = set(file_records.get(path, {}).get("chunk_hashes", []))
                new_hashes = [chunk["metadata"]["chunk_hash"] for chunk in chunks]
                new_hashes_set = set(new_hashes)

                saved_count = 0
                # Déréférence les anciens chunks de cette source, puis supprime les orphelins.
                for old_ch in previous_hashes:
                    if old_ch in refcounts:
                        refcounts[old_ch] -= 1
                        if refcounts[old_ch] <= 0:
                            refcounts.pop(old_ch, None)
                    if old_ch not in new_hashes_set:
                        _delete_chunk_file_if_unreferenced(old_ch, refcounts, seen_chunks)

                for chunk in chunks:
                    ch_hash = chunk["metadata"]["chunk_hash"]
                    if ch_hash in seen_chunks:
                        continue
                    seen_chunks.add(ch_hash)

                    out_path = os.path.join(PROCESSED_PATH, f"{ch_hash}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    saved_count += 1

                # Référencement des nouveaux chunks pour cette source.
                for ch in new_hashes_set:
                    refcounts[ch] = refcounts.get(ch, 0) + 1

                file_records[path] = {
                    "file_hash": file_hash,
                    "chunk_hashes": list(dict.fromkeys(new_hashes)),
                }

                logger.info(f"✅ Traité : {Path(path).name} → {len(chunks)} chunks ({saved_count} sauvegardés)")

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {path}: {e}")

    save_cache({"version": 2, "files": file_records})
    logger.info("🎉 Processing terminé avec succès !")


if __name__ == "__main__":
    preprocess_all()
