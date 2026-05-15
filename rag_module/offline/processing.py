# rag_module/processing.py
import hashlib
import json
import logging
import os
import re
import string
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Tuple

from tiktoken import encoding_for_model

try:
    from .document_extractors import build_extractors, extract_text_docx
    from .language_detection import detect_language
    from .processing_cache import (
        chunk_refcounts as pc_chunk_refcounts,
        corpus_paths as pc_corpus_paths,
        delete_chunk_file_if_unreferenced as pc_delete_chunk_file_if_unreferenced,
        classify_document_state as pc_classify_document_state,
        load_cache as pc_load_cache,
        load_raw_metadata as pc_load_raw_metadata,
        mark_failed as pc_mark_failed,
        mark_no_chunks as pc_mark_no_chunks,
        mark_processed as pc_mark_processed,
        mark_skipped as pc_mark_skipped,
        save_cache as pc_save_cache,
    )
    from .text_quality import (
        clean_text as tq_clean_text,
        deduplicate_chunk_texts as tq_deduplicate_chunk_texts,
        is_high_quality_chunk as tq_is_high_quality_chunk,
        is_high_quality_document as tq_is_high_quality_document,
        quality_score as tq_quality_score,
        split_sentences as tq_split_sentences,
        text_metrics as tq_text_metrics,
        tokenize_words as tq_tokenize_words,
    )
    from ..shared.data_quality import create_backup, postprocess_chunks_for_source
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.offline.document_extractors import build_extractors, extract_text_docx
    from rag_module.offline.language_detection import detect_language
    from rag_module.offline.processing_cache import (
        chunk_refcounts as pc_chunk_refcounts,
        corpus_paths as pc_corpus_paths,
        delete_chunk_file_if_unreferenced as pc_delete_chunk_file_if_unreferenced,
        classify_document_state as pc_classify_document_state,
        load_cache as pc_load_cache,
        load_raw_metadata as pc_load_raw_metadata,
        mark_failed as pc_mark_failed,
        mark_no_chunks as pc_mark_no_chunks,
        mark_processed as pc_mark_processed,
        mark_skipped as pc_mark_skipped,
        save_cache as pc_save_cache,
    )
    from rag_module.offline.text_quality import (
        clean_text as tq_clean_text,
        deduplicate_chunk_texts as tq_deduplicate_chunk_texts,
        is_high_quality_chunk as tq_is_high_quality_chunk,
        is_high_quality_document as tq_is_high_quality_document,
        quality_score as tq_quality_score,
        split_sentences as tq_split_sentences,
        text_metrics as tq_text_metrics,
        tokenize_words as tq_tokenize_words,
    )
    from rag_module.shared.data_quality import create_backup, postprocess_chunks_for_source
    from rag_module.shared.runtime import get_runtime_settings


RUNTIME = get_runtime_settings()
PROCESSING_POLICY_VERSION = "v12_drive_service_profiles_v5"

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
MIN_LANG_CONFIDENCE = 0.70
MAX_URLS_PER_CHUNK = 1
MAX_REPEAT_CHAR_RUN = 6
ALLOWED_LANGUAGES = {"fr", "ar", "en"}

CORPUS_POLICIES = {
    "main": {
        "min_words": MIN_WORDS,
        "min_doc_words": MIN_DOC_WORDS,
        "min_doc_chars": MIN_DOC_CHARS,
        "min_quality_score": MIN_QUALITY_SCORE,
        "min_alpha_ratio": MIN_ALPHA_RATIO,
        "max_digit_ratio": MAX_DIGIT_RATIO,
        "max_symbol_ratio": MAX_SYMBOL_RATIO,
        "min_unique_ratio": MIN_UNIQUE_TOKEN_RATIO,
        "max_urls_per_chunk": MAX_URLS_PER_CHUNK,
        "preserve_short_lines": False,
    },
    "archive": {
        "min_words": 6,
        "min_doc_words": 80,
        "min_doc_chars": 400,
        "min_quality_score": 42,
        "min_alpha_ratio": 0.50,
        "max_digit_ratio": 0.35,
        "max_symbol_ratio": 0.28,
        "min_unique_ratio": 0.22,
        "max_urls_per_chunk": 2,
        "preserve_short_lines": False,
    },
    "drive": {
        "min_words": MIN_WORDS,
        "min_doc_words": 90,
        "min_doc_chars": 500,
        "min_quality_score": 45,
        "min_alpha_ratio": 0.52,
        "max_digit_ratio": MAX_DIGIT_RATIO,
        "max_symbol_ratio": 0.32,
        "min_unique_ratio": 0.18,
        "max_urls_per_chunk": 6,
        "preserve_short_lines": True,
    },
}

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
EXTRACTORS = build_extractors(settings=RUNTIME)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ENCODER = encoding_for_model(LLM_MODEL)
MOJIBAKE_TOKENS = ("Ã", "â€™", "â€“", "â€œ", "â€", "ðŸ", "Â")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _tokenize_words(text: str) -> List[str]:
    return tq_tokenize_words(text)


def _mojibake_score(text: str) -> int:
    return sum(text.count(token) for token in MOJIBAKE_TOKENS)


def _repair_mojibake(text: str) -> str:
    if not text or _mojibake_score(text) == 0:
        return text

    best_text = text
    best_score = _mojibake_score(text)
    for encoding in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(encoding, errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            continue
        if not candidate:
            continue
        candidate_score = _mojibake_score(candidate)
        if candidate_score < best_score and len(candidate.strip()) >= max(10, int(len(text.strip()) * 0.6)):
            best_text = candidate
            best_score = candidate_score
    return best_text


def _corpus_policy(corpus: str) -> Dict[str, float]:
    return CORPUS_POLICIES.get(corpus, CORPUS_POLICIES["main"])


def _looks_like_url_or_path(line: str, corpus: str = "main") -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if re.search(r"https?://|www\.", lower):
        return corpus != "drive"
    if lower.startswith("mailto:"):
        return corpus != "drive"
    if re.match(r"^[a-z]:\\", lower):
        return True
    if "/" in lower and len(lower.split()) <= 3 and len(lower) < 120:
        return True
    return False


def _is_noise_line(line: str, corpus: str = "main") -> bool:
    candidate = line.strip()
    if not candidate:
        return True

    policy = _corpus_policy(corpus)
    preserve_short_lines = bool(policy.get("preserve_short_lines"))

    if len(candidate) <= 2 and not preserve_short_lines:
        return True

    for pattern in NOISE_LINE_REGEXES:
        if re.match(pattern, candidate, flags=re.IGNORECASE):
            return True

    lower = candidate.lower()
    if any(phrase in lower for phrase in NOISE_PHRASES):
        return True
    if _looks_like_url_or_path(candidate, corpus=corpus):
        return True

    non_space_len = sum(1 for ch in candidate if not ch.isspace())
    if non_space_len == 0:
        return True

    alpha_count = sum(1 for ch in candidate if ch.isalpha())
    symbol_count = sum(1 for ch in candidate if not ch.isalnum() and not ch.isspace())
    if not preserve_short_lines and _safe_ratio(alpha_count, non_space_len) < 0.25:
        return True
    if _safe_ratio(symbol_count, non_space_len) > 0.45:
        return True
    return False


def _text_metrics(text: str) -> Dict[str, float]:
    return tq_text_metrics(text)


def clean_text(text: str, corpus: str = "main") -> str:
    return tq_clean_text(text, corpus=corpus)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_detect_lang(text: str) -> Tuple[str, float]:
    lang, confidence = detect_language(text, settings=RUNTIME)
    if confidence < MIN_LANG_CONFIDENCE:
        return "unknown", confidence
    return lang, confidence


def quality_score(text: str) -> int:
    return tq_quality_score(text)


def _corpus_paths(corpus: str) -> Tuple[str, str, str]:
    return pc_corpus_paths(corpus)


def _load_raw_metadata(corpus: str) -> Dict[str, Dict]:
    return pc_load_raw_metadata(corpus)


def is_high_quality_chunk(text: str, corpus: str = "main") -> bool:
    return tq_is_high_quality_chunk(text, corpus=corpus)


def _is_high_quality_document(text: str, corpus: str = "main") -> bool:
    return tq_is_high_quality_document(text, corpus=corpus)


def _deduplicate_chunk_texts(chunks: List[str]) -> List[str]:
    return tq_deduplicate_chunk_texts(chunks)


def extract_text_html(path: str) -> str:
    return EXTRACTORS[".html"](path)


def extract_text_pdf(path: str) -> str:
    return EXTRACTORS[".pdf"](path)


def extract_text_plain(path: str) -> str:
    return EXTRACTORS[".txt"](path)


def split_sentences(text: str) -> List[str]:
    return tq_split_sentences(text)


def semantic_chunk(
    text: str,
    corpus: str = "main",
    chunk_size: int = CHUNK_TOKENS,
    overlap_size: int = OVERLAP_TOKENS,
) -> List[str]:
    text = clean_text(text, corpus=corpus)
    if not text:
        return []

    if len(ENCODER.encode(text)) <= chunk_size:
        return [text] if is_high_quality_chunk(text, corpus=corpus) else []

    sections = re.split(r"(?=\n#+\s+)", text)
    if len(sections) == 1:
        sections = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_tokens = len(ENCODER.encode(section))
        if section_tokens > chunk_size:
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]
            for paragraph in paragraphs:
                paragraph_tokens = len(ENCODER.encode(paragraph))
                if paragraph_tokens > chunk_size:
                    for sentence in split_sentences(paragraph):
                        sentence_tokens = len(ENCODER.encode(sentence))
                        if sentence_tokens > chunk_size:
                            chunks.append(sentence)
                            continue
                        if current_tokens + sentence_tokens > chunk_size and current:
                            chunk_text = "\n".join(current).strip()
                            if is_high_quality_chunk(chunk_text, corpus=corpus):
                                chunks.append(chunk_text)
                            current = [sentence]
                            current_tokens = sentence_tokens
                        else:
                            current.append(sentence)
                            current_tokens += sentence_tokens
                    continue

                if current_tokens + paragraph_tokens > chunk_size and current:
                    chunk_text = "\n".join(current).strip()
                    if is_high_quality_chunk(chunk_text, corpus=corpus):
                        chunks.append(chunk_text)
                    current = [paragraph]
                    current_tokens = paragraph_tokens
                else:
                    current.append(paragraph)
                    current_tokens += paragraph_tokens
            continue

        if current_tokens + section_tokens > chunk_size and current:
            chunk_text = "\n\n".join(current).strip()
            if is_high_quality_chunk(chunk_text, corpus=corpus):
                chunks.append(chunk_text)

            overlap: List[str] = []
            tokens_acc = 0
            for item in reversed(current):
                token_len = len(ENCODER.encode(item))
                if tokens_acc + token_len > overlap_size:
                    break
                overlap.insert(0, item)
                tokens_acc += token_len

            current = overlap + [section]
            current_tokens = sum(len(ENCODER.encode(item)) for item in current)
        else:
            current.append(section)
            current_tokens += section_tokens

    if current:
        chunk_text = "\n\n".join(current).strip()
        if is_high_quality_chunk(chunk_text, corpus=corpus):
            chunks.append(chunk_text)

    return _deduplicate_chunk_texts(chunks)


def preprocess_file(file_path: str, corpus: str = "main", raw_metadata: Dict | None = None) -> List[Dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".doc":
        logger.warning("Legacy Word format unsupported without conversion: %s", file_path)
        return []

    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        logger.warning("Unsupported format: %s", file_path)
        return []

    raw_text = extractor(file_path)
    if not raw_text:
        return []

    cleaned_text = clean_text(raw_text, corpus=corpus)
    if not _is_high_quality_document(cleaned_text, corpus=corpus):
        logger.info("Skip low-quality document: %s", Path(file_path).name)
        return []

    doc_language, lang_confidence = safe_detect_lang(cleaned_text)
    if doc_language == "unknown" or doc_language not in ALLOWED_LANGUAGES:
        logger.info("Skip uncertain language document: %s", Path(file_path).name)
        return []

    chunks = semantic_chunk(cleaned_text, corpus=corpus)
    if not chunks:
        return []

    file_name = Path(file_path).name
    file_hash = hash_file(file_path)

    results = []
    for index, chunk in enumerate(chunks):
        q_score = quality_score(chunk)
        metrics = _text_metrics(chunk)
        results.append(
            {
                "text": chunk,
                "text_normalized": chunk.lower(),
                "quality": q_score,
                "metadata": {
                    "source": str(file_path),
                    "source_hash": file_hash,
                    "chunk_hash": hash_text(f"{file_hash}:{index}:{chunk}"),
                    "index": index,
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
                    "corpus": corpus,
                    "processing_policy_version": PROCESSING_POLICY_VERSION,
                },
            }
        )

    if raw_metadata:
        for result in results:
            result["metadata"].update(raw_metadata)
            result["metadata"]["corpus"] = corpus
            result["metadata"]["processing_policy_version"] = PROCESSING_POLICY_VERSION

    return postprocess_chunks_for_source(results, file_path, corpus=corpus)


def load_cache(cache_file: str) -> Dict:
    return pc_load_cache(cache_file)


def save_cache(cache: Dict, cache_file: str) -> None:
    pc_save_cache(cache, cache_file)


def _chunk_refcounts(file_records: Dict[str, Dict]) -> Dict[str, int]:
    return pc_chunk_refcounts(file_records)


def _delete_chunk_file_if_unreferenced(
    chunk_hash: str,
    refcounts: Dict[str, int],
    seen_chunks: set,
    processed_path: str,
) -> bool:
    return pc_delete_chunk_file_if_unreferenced(chunk_hash, refcounts, seen_chunks, processed_path)


def _preprocess_corpus(corpus: str) -> Dict:
    raw_path, processed_path, cache_file = _corpus_paths(corpus)
    raw_metadata = _load_raw_metadata(corpus)
    os.makedirs(processed_path, exist_ok=True)
    summary = {
        "corpus": corpus,
        "raw_path": raw_path,
        "processed_path": processed_path,
        "detected": 0,
        "processed": 0,
        "skipped_unchanged": 0,
        "skipped_no_chunks": 0,
        "failed": 0,
        "quarantined": 0,
        "deleted_sources": 0,
        "removed_chunks": 0,
        "new": 0,
        "modified": 0,
        "retried": 0,
    }

    backup_dir = create_backup(processed_path, cache_file)
    if backup_dir:
        logger.info("Backup created before cleanup: %s", backup_dir)

    cache = load_cache(cache_file)
    file_records: Dict[str, Dict] = cache.get("files", {})
    seen_chunks = {Path(name).stem for name in os.listdir(processed_path) if name.endswith(".json")}
    refcounts = _chunk_refcounts(file_records)

    files = [
        os.path.join(root, file_name)
        for root, _, file_names in os.walk(raw_path)
        for file_name in file_names
        if not file_name.startswith(".")
    ]

    logger.info("%s files detected in %s [%s]", len(files), raw_path, corpus)

    deleted_sources = [path for path in list(file_records.keys()) if not os.path.exists(path)]
    removed_chunks = 0
    for path in deleted_sources:
        record = file_records.pop(path, {})
        for chunk_hash in set(record.get("chunk_hashes", [])):
            if chunk_hash in refcounts:
                refcounts[chunk_hash] -= 1
                if refcounts[chunk_hash] <= 0:
                    refcounts.pop(chunk_hash, None)
            if _delete_chunk_file_if_unreferenced(chunk_hash, refcounts, seen_chunks, processed_path):
                removed_chunks += 1
    if deleted_sources:
        summary["deleted_sources"] = len(deleted_sources)
        summary["removed_chunks"] = removed_chunks
        logger.info(
            "Cleanup removed sources: %s source(s), %s chunk(s).",
            len(deleted_sources),
            removed_chunks,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        updated_chunk_hashes = set()
        future_to_path = {}

        for file_path in files:
            summary["detected"] += 1
            file_hash = hash_file(file_path)
            record = file_records.get(file_path, {})
            old_hashes = record.get("chunk_hashes", [])
            has_all_chunks = all(
                os.path.exists(os.path.join(processed_path, f"{chunk_hash}.json"))
                for chunk_hash in old_hashes
            )
            state = pc_classify_document_state(
                file_path,
                file_hash,
                record,
                has_all_chunks=has_all_chunks,
                policy_version=PROCESSING_POLICY_VERSION,
            )
            if state in {"processed", "skipped"}:
                file_records[file_path] = pc_mark_skipped(record, file_hash, PROCESSING_POLICY_VERSION, corpus)
                summary["skipped_unchanged"] += 1
                logger.info("Skip unchanged -> %s", Path(file_path).name)
                continue
            if state == "quarantine":
                summary["quarantined"] += 1
                logger.warning("Skip quarantined -> %s", Path(file_path).name)
                continue
            if state == "new":
                summary["new"] += 1
            elif state == "modified":
                summary["modified"] += 1
            elif state == "retry":
                summary["retried"] += 1

            future = executor.submit(preprocess_file, file_path, corpus, raw_metadata.get(file_path, {}))
            future_to_path[future] = (file_path, file_hash)

        for future in as_completed(future_to_path):
            file_path, file_hash = future_to_path[future]
            try:
                chunks = future.result()
            except Exception as exc:
                logger.error("Processing error for %s: %s", file_path, exc)
                file_records[file_path] = pc_mark_failed(
                    file_records.get(file_path, {}),
                    file_hash,
                    PROCESSING_POLICY_VERSION,
                    corpus,
                    str(exc),
                )
                if file_records[file_path].get("status") == "quarantine":
                    summary["quarantined"] += 1
                else:
                    summary["failed"] += 1
                continue

            previous_hashes = set(file_records.get(file_path, {}).get("chunk_hashes", []))
            new_hashes = [chunk["metadata"]["chunk_hash"] for chunk in chunks]
            new_hashes_set = set(new_hashes)

            saved_count = 0
            overwritten_count = 0
            for old_hash in previous_hashes:
                if old_hash in refcounts:
                    refcounts[old_hash] -= 1
                    if refcounts[old_hash] <= 0:
                        refcounts.pop(old_hash, None)
                if old_hash not in new_hashes_set:
                    _delete_chunk_file_if_unreferenced(old_hash, refcounts, seen_chunks, processed_path)

            if not chunks:
                file_records[file_path] = pc_mark_no_chunks(
                    file_hash,
                    PROCESSING_POLICY_VERSION,
                    corpus,
                    reason="Aucun chunk genere: document vide, faible qualite, langue non supportee ou format ignore.",
                )
                summary["skipped_no_chunks"] += 1
                logger.info("Skipped no chunks [%s]: %s", corpus, Path(file_path).name)
                continue

            for chunk in chunks:
                chunk_hash = chunk["metadata"]["chunk_hash"]
                if chunk_hash in updated_chunk_hashes:
                    continue
                out_path = os.path.join(processed_path, f"{chunk_hash}.json")
                existed = os.path.exists(out_path)
                with open(out_path, "w", encoding="utf-8") as handle:
                    json.dump(chunk, handle, ensure_ascii=False, indent=2)
                if existed:
                    overwritten_count += 1
                else:
                    saved_count += 1
                seen_chunks.add(chunk_hash)
                updated_chunk_hashes.add(chunk_hash)

            for chunk_hash in new_hashes_set:
                refcounts[chunk_hash] = refcounts.get(chunk_hash, 0) + 1

            file_records[file_path] = pc_mark_processed(
                file_hash,
                list(dict.fromkeys(new_hashes)),
                PROCESSING_POLICY_VERSION,
                corpus,
            )
            summary["processed"] += 1

            logger.info(
                "Processed [%s]: %s -> %s chunks (%s saved, %s overwritten)",
                corpus,
                Path(file_path).name,
                len(chunks),
                saved_count,
                overwritten_count,
            )

    save_cache({"version": 2, "files": file_records}, cache_file)
    logger.info("Processing completed successfully for corpus=%s. Summary=%s", corpus, summary)
    return summary


def preprocess_all(corpus: str = "all") -> Dict:
    if corpus in {"main", "archive", "drive"}:
        return {"status": "ok", "corpora": [_preprocess_corpus(corpus)]}
    summaries = []
    for selected in ("main", "archive", "drive"):
        summaries.append(_preprocess_corpus(selected))
    return {"status": "ok", "corpora": summaries}


if __name__ == "__main__":
    preprocess_all()
