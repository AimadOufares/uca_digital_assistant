# rag_module/processing.py
import os
import re
import json
import hashlib
import logging
import string
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import pdfplumber
import docx
from bs4 import BeautifulSoup
from tiktoken import encoding_for_model
from html import unescape
from langdetect import detect_langs, LangDetectException
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ..shared.data_quality import create_backup, postprocess_chunks_for_source
except ImportError:  # pragma: no cover
    from rag_module.shared.data_quality import create_backup, postprocess_chunks_for_source

# ===================== CONFIG =====================
RAW_PATH = "data_storage/raw"
PROCESSED_PATH = "data_storage/processed"
CACHE_FILE = "data_storage/cache/file_cache.json"
PROCESSING_POLICY_VERSION = "v4_strict_quality_2026_04_05_r3_sourcehash"

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 80
MIN_WORDS = 8
MIN_DOC_WORDS = 120
MIN_DOC_CHARS = 700
MIN_QUALITY_SCORE = 55
MIN_ALPHA_RATIO = 0.60
MAX_DIGIT_RATIO = 0.30
MAX_SYMBOL_RATIO = 0.22
MIN_UNIQUE_TOKEN_RATIO = 0.30
MIN_LANG_CONFIDENCE = 0.85
MAX_URLS_PER_CHUNK = 1
MAX_REPEAT_CHAR_RUN = 6
ALLOWED_LANGUAGES = {"fr", "ar", "en"}

NOISE_LINE_REGEXES = [
    r"^\s*(menu|home|accueil|contact|connexion|login|logout|search|rechercher)\s*$",
    r"^\s*(mentions legales|politique de confidentialite|privacy policy|cookie policy)\s*$",
    r"^\s*(suivez[- ]?nous|follow us|facebook|instagram|linkedin|youtube|twitter)\s*$",
    r"^\s*(tous droits reserves|all rights reserved|copyright)\s*$",
    r"^\s*(\d+\s*){1,4}$",
]

NOISE_PHRASES = (
    "accepter les cookies",
    "manage cookies",
    "mot de passe oublie",
    "forgot password",
    "subscribe to newsletter",
    "inscrivez-vous a la newsletter",
)

LLM_MODEL = "gpt-4o-mini"

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== TOKENIZER =====================
ENCODER = encoding_for_model(LLM_MODEL)


# ===================== UTILS =====================
def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def _looks_like_url_or_path(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if re.search(r"https?://|www\.", lower):
        return True
    if "/" in lower and len(lower.split()) <= 3 and len(lower) < 120:
        return True
    return False


def _is_noise_line(line: str) -> bool:
    candidate = line.strip()
    if not candidate:
        return True
    if len(candidate) <= 2:
        return True

    for pattern in NOISE_LINE_REGEXES:
        if re.match(pattern, candidate, flags=re.IGNORECASE):
            return True

    lower = candidate.lower()
    if any(phrase in lower for phrase in NOISE_PHRASES):
        return True
    if _looks_like_url_or_path(candidate):
        return True

    non_space_len = sum(1 for ch in candidate if not ch.isspace())
    if non_space_len == 0:
        return True

    alpha_count = sum(1 for ch in candidate if ch.isalpha())
    symbol_count = sum(1 for ch in candidate if not ch.isalnum() and not ch.isspace())
    if _safe_ratio(alpha_count, non_space_len) < 0.25:
        return True
    if _safe_ratio(symbol_count, non_space_len) > 0.45:
        return True
    return False


def _text_metrics(text: str) -> Dict[str, float]:
    tokens = _tokenize_words(text)
    words = len(tokens)
    unique_ratio = _safe_ratio(len(set(tokens)), words)

    non_space_len = sum(1 for ch in text if not ch.isspace())
    alpha_count = sum(1 for ch in text if ch.isalpha())
    digit_count = sum(1 for ch in text if ch.isdigit())
    symbol_count = sum(
        1
        for ch in text
        if ch in string.punctuation or (not ch.isalnum() and not ch.isspace())
    )
    sentence_count = len(
        [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(_tokenize_words(s)) >= 3]
    )
    url_count = len(re.findall(r"https?://|www\.", text.lower()))
    repeated_run = bool(re.search(rf"(.)\1{{{MAX_REPEAT_CHAR_RUN},}}", text))

    return {
        "words": float(words),
        "unique_ratio": unique_ratio,
        "alpha_ratio": _safe_ratio(alpha_count, non_space_len),
        "digit_ratio": _safe_ratio(digit_count, non_space_len),
        "symbol_ratio": _safe_ratio(symbol_count, non_space_len),
        "sentence_count": float(sentence_count),
        "url_count": float(url_count),
        "repeated_run": 1.0 if repeated_run else 0.0,
    }


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = unescape(text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]+", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    previous_line = ""
    seen_line_hashes = set()

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip(" -|\t")
        if _is_noise_line(line):
            continue

        lowered = line.lower()
        line_hash = hashlib.md5(lowered.encode("utf-8")).hexdigest()

        if lowered == previous_line:
            continue
        if line_hash in seen_line_hashes and len(line.split()) < 7:
            continue

        cleaned_lines.append(line)
        seen_line_hashes.add(line_hash)
        previous_line = lowered

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_detect_lang(text: str) -> Tuple[str, float]:
    if len(text.split()) < 30:
        return "unknown", 0.0
    try:
        candidates = detect_langs(text[:1500])
        if not candidates:
            return "unknown", 0.0

        top = candidates[0]
        lang = getattr(top, "lang", "unknown") or "unknown"
        confidence = float(getattr(top, "prob", 0.0) or 0.0)

        if confidence < MIN_LANG_CONFIDENCE:
            return "unknown", confidence
        return lang, confidence
    except LangDetectException:
        return "unknown", 0.0


def quality_score(text: str) -> int:
    """Heuristic quality score [0,100] for a chunk."""
    if not text:
        return 0

    metrics = _text_metrics(text)
    words = metrics["words"]
    score = 0.0

    score += min(words, 180.0) * 0.22
    score += min(metrics["sentence_count"], 12.0) * 2.8
    score += min(metrics["unique_ratio"], 1.0) * 24.0
    score += min(metrics["alpha_ratio"], 1.0) * 20.0

    score -= metrics["digit_ratio"] * 30.0
    score -= metrics["symbol_ratio"] * 35.0
    score -= max(0.0, metrics["url_count"] - 1.0) * 8.0
    if metrics["repeated_run"] > 0:
        score -= 15.0

    return int(max(0.0, min(100.0, round(score))))


def is_high_quality_chunk(text: str) -> bool:
    metrics = _text_metrics(text)
    if metrics["words"] < MIN_WORDS:
        return False
    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        return False
    if metrics["digit_ratio"] > MAX_DIGIT_RATIO:
        return False
    if metrics["symbol_ratio"] > MAX_SYMBOL_RATIO:
        return False
    if metrics["unique_ratio"] < MIN_UNIQUE_TOKEN_RATIO:
        return False
    if metrics["url_count"] > MAX_URLS_PER_CHUNK:
        return False
    return quality_score(text) >= MIN_QUALITY_SCORE


def _is_high_quality_document(text: str) -> bool:
    if len(text) < MIN_DOC_CHARS:
        return False
    if len(_tokenize_words(text)) < MIN_DOC_WORDS:
        return False

    metrics = _text_metrics(text[: min(len(text), 6000)])
    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        return False
    if metrics["digit_ratio"] > MAX_DIGIT_RATIO:
        return False
    if metrics["symbol_ratio"] > MAX_SYMBOL_RATIO:
        return False
    if metrics["unique_ratio"] < MIN_UNIQUE_TOKEN_RATIO:
        return False
    if quality_score(text[: min(len(text), 6000)]) < MIN_QUALITY_SCORE:
        return False
    return True


def _deduplicate_chunk_texts(chunks: List[str]) -> List[str]:
    unique_chunks: List[str] = []
    seen = set()

    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk.strip().lower())
        if not normalized:
            continue
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique_chunks.append(chunk)

    return unique_chunks


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
        logger.warning("HTML extraction error %s: %s", path, e)
        return ""


def extract_text_pdf(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if t:
                    text_parts.append(t)

                tables = page.extract_tables()
                for table in tables:
                    if table and any(any(cell for cell in row) for row in table):
                        md_table = "\n".join(
                            [" | ".join(str(cell) if cell is not None else "" for cell in row)
                             for row in table]
                        )
                        text_parts.append(f"\n[TABLE_PAGE_{page_num}]\n{md_table}\n[/TABLE]\n")
    except Exception as e:
        logger.warning("PDF extraction error %s: %s", path, e)
    return "\n\n".join(text_parts)


def extract_text_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.warning("DOCX extraction error %s: %s", path, e)
        return ""


def extract_text_plain(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Plain text extraction error %s: %s", path, e)
        return ""


# ===================== CHUNKING =====================
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip() and not _is_noise_line(s)]


def recursive_chunk(text: str, chunk_size: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    if len(ENCODER.encode(text)) <= chunk_size:
        return [text] if is_high_quality_chunk(text) else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        sentences = split_sentences(para)
        for sent in sentences:
            sent_tokens = len(ENCODER.encode(sent))

            # If the sentence itself is huge, split hard by token windows.
            if sent_tokens > chunk_size:
                words = sent.split()
                window = []
                for word in words:
                    candidate = " ".join(window + [word])
                    if len(ENCODER.encode(candidate)) <= chunk_size:
                        window.append(word)
                    else:
                        long_chunk = " ".join(window).strip()
                        if is_high_quality_chunk(long_chunk):
                            chunks.append(long_chunk)
                        window = [word]
                long_chunk = " ".join(window).strip()
                if is_high_quality_chunk(long_chunk):
                    chunks.append(long_chunk)
                continue

            if current_tokens + sent_tokens > chunk_size and current:
                chunk_text = " ".join(current).strip()
                if is_high_quality_chunk(chunk_text):
                    chunks.append(chunk_text)

                overlap: List[str] = []
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

    if current:
        chunk_text = " ".join(current).strip()
        if is_high_quality_chunk(chunk_text):
            chunks.append(chunk_text)

    return _deduplicate_chunk_texts(chunks)


# ===================== PROCESS FILE =====================
def preprocess_file(file_path: str) -> List[Dict]:
    ext = Path(file_path).suffix.lower()
    extractors = {
        ".html": extract_text_html,
        ".pdf": extract_text_pdf,
        ".docx": extract_text_docx,
        ".txt": extract_text_plain,
        ".md": extract_text_plain,
    }

    if ext not in extractors:
        logger.warning("Unsupported format: %s", file_path)
        return []

    raw_text = extractors[ext](file_path)
    if not raw_text:
        return []

    cleaned_text = clean_text(raw_text)
    if not _is_high_quality_document(cleaned_text):
        logger.info("Skip low-quality document: %s", Path(file_path).name)
        return []

    doc_language, lang_confidence = safe_detect_lang(cleaned_text)
    if doc_language == "unknown" or doc_language not in ALLOWED_LANGUAGES:
        logger.info("Skip uncertain language document: %s", Path(file_path).name)
        return []

    chunks = recursive_chunk(cleaned_text)
    if not chunks:
        return []

    file_name = Path(file_path).name
    file_hash = hash_file(file_path)

    results = []
    for i, chunk in enumerate(chunks):
        q_score = quality_score(chunk)
        metrics = _text_metrics(chunk)
        results.append({
            "text": chunk,
            "text_normalized": chunk.lower(),
            "quality": q_score,
            "metadata": {
                "source": str(file_path),
                "source_hash": file_hash,
                "chunk_hash": hash_text(f"{file_hash}:{i}:{chunk}"),
                "index": i,
                "total_chunks": len(chunks),
                "tokens": len(ENCODER.encode(chunk)),
                "language": doc_language,
                "language_confidence": round(lang_confidence, 4),
                "file_name": file_name,
                "file_type": ext,
                "is_table": ("[TABLE" in chunk) or ("TABLE_PAGE_" in chunk),
                "quality_score": q_score,
                "quality_alpha_ratio": round(metrics["alpha_ratio"], 4),
                "quality_unique_ratio": round(metrics["unique_ratio"], 4),
                "date_processed": datetime.now(timezone.utc).isoformat(),
            }
        })

    return postprocess_chunks_for_source(results, file_path)


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
                        "policy_version": entry.get("policy_version", ""),
                    }
                return {"version": 2, "files": files}

            # Backward compatibility: {path: file_hash}
            if isinstance(raw, dict):
                files = {}
                for path, file_hash in raw.items():
                    if isinstance(path, str) and isinstance(file_hash, str):
                        files[path] = {"file_hash": file_hash, "chunk_hashes": [], "policy_version": ""}
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
    backup_dir = create_backup(PROCESSED_PATH, CACHE_FILE)
    if backup_dir:
        logger.info("Backup created before cleanup: %s", backup_dir)

    cache = load_cache()
    file_records: Dict[str, Dict] = cache.get("files", {})
    seen_chunks = {Path(f).stem for f in os.listdir(PROCESSED_PATH) if f.endswith(".json")}
    refcounts = _chunk_refcounts(file_records)

    files = [
        os.path.join(root, f)
        for root, _, fs in os.walk(RAW_PATH)
        for f in fs if not f.startswith(".")
    ]

    logger.info("%s files detected in %s", len(files), RAW_PATH)

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
        logger.info(
            "Cleanup removed sources: %s source(s), %s chunk(s).",
            len(deleted_sources),
            removed_chunks,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        updated_chunk_hashes = set()
        future_to_path = {}
        for f in files:
            file_hash = hash_file(f)
            record = file_records.get(f, {})
            old_hashes = record.get("chunk_hashes", [])
            has_all_chunks = all(
                os.path.exists(os.path.join(PROCESSED_PATH, f"{ch}.json")) for ch in old_hashes
            )
            if (
                record.get("file_hash") == file_hash
                and has_all_chunks
                and record.get("policy_version") == PROCESSING_POLICY_VERSION
            ):
                logger.info("Skip unchanged -> %s", Path(f).name)
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
                overwritten_count = 0
                for old_ch in previous_hashes:
                    if old_ch in refcounts:
                        refcounts[old_ch] -= 1
                        if refcounts[old_ch] <= 0:
                            refcounts.pop(old_ch, None)
                    if old_ch not in new_hashes_set:
                        _delete_chunk_file_if_unreferenced(old_ch, refcounts, seen_chunks)

                for chunk in chunks:
                    ch_hash = chunk["metadata"]["chunk_hash"]
                    if ch_hash in updated_chunk_hashes:
                        continue
                    out_path = os.path.join(PROCESSED_PATH, f"{ch_hash}.json")
                    existed = os.path.exists(out_path)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    if existed:
                        overwritten_count += 1
                    else:
                        saved_count += 1
                    seen_chunks.add(ch_hash)
                    updated_chunk_hashes.add(ch_hash)

                for ch in new_hashes_set:
                    refcounts[ch] = refcounts.get(ch, 0) + 1

                file_records[path] = {
                    "file_hash": file_hash,
                    "chunk_hashes": list(dict.fromkeys(new_hashes)),
                    "policy_version": PROCESSING_POLICY_VERSION,
                }

                logger.info(
                    "Processed: %s -> %s chunks (%s saved, %s overwritten)",
                    Path(path).name,
                    len(chunks),
                    saved_count,
                    overwritten_count,
                )

            except Exception as e:
                logger.error("Processing error for %s: %s", path, e)

    save_cache({"version": 2, "files": file_records})
    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    preprocess_all()
