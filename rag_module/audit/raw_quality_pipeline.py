import argparse
import hashlib
import json
import os
import re
import shutil
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import docx
import pdfplumber
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from langdetect import LangDetectException, detect_langs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data_storage" / "raw"
REPORTS_DIR = PROJECT_ROOT / "data_storage" / "reports"
QUARANTINE_DIR = PROJECT_ROOT / "data_storage" / "quarantine"
BACKUPS_DIR = PROJECT_ROOT / "data_storage" / "backups"

ALLOWED_LANGS = {"fr", "ar", "en"}
MIN_LANG_CONFIDENCE = 0.85
MIN_DOC_WORDS = 120
MIN_DOC_CHARS = 700
MIN_ALPHA_RATIO = 0.60
MAX_SYMBOL_RATIO = 0.22
MAX_DIGIT_RATIO = 0.30
MIN_LEXICAL_DIVERSITY = 0.30
MAX_URLS_PER_CHUNK = 1
MIN_CHUNK_QUALITY_SCORE = 55
NEAR_DUP_THRESHOLD = 0.92

SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf", ".docx", ".txt", ".md"}
MOJIBAKE_PATTERN = re.compile(r"(Ã.|Â.|â€|â€™|â€œ|â€“|�)")
URL_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
TOPIC_PATTERN = re.compile(
    r"(inscription|admission|bourse|calendrier|scolarit|preinscription|reinscription|"
    r"candidature|concours|frais d[' ]inscription|emploi du temps|attestation|"
    r"scholarship|registration|admissions|enrollment|application deadline|"
    r"منحة|التسجيل|القبول|المباراة|الترشيح|التمدرس|الجدول الزمني)",
    flags=re.IGNORECASE,
)
NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*(menu|accueil|home|contact|connexion|login|logout)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(privacy policy|politique de confidentialite|cookie policy|cookies?)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(facebook|instagram|linkedin|youtube|twitter|suivez[- ]?nous|follow us)\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*(mentions legales|all rights reserved|tous droits reserves|copyright)\s*$", flags=re.IGNORECASE),
]


@dataclass
class DocumentAudit:
    path: str
    extension: str
    size_bytes: int
    domain: str
    extracted_chars: int
    words: int
    alpha_ratio: float
    symbol_ratio: float
    digit_ratio: float
    lexical_diversity: float
    url_count: int
    lang: str
    lang_confidence: float
    quality_score: int
    text_hash: str
    normalized_hash: str
    simhash64: int
    noise_line_ratio: float
    mojibake_ratio: float
    is_topic_relevant: bool
    flags: List[str]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]+", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_words(text: str) -> List[str]:
    return WORD_PATTERN.findall(text.lower())


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def text_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize_words(text)
    words = len(tokens)
    unique_ratio = safe_ratio(len(set(tokens)), words)
    non_space = sum(1 for ch in text if not ch.isspace())
    alpha = sum(1 for ch in text if ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    symbols = sum(
        1
        for ch in text
        if ch in string.punctuation or (not ch.isalnum() and not ch.isspace())
    )
    urls = len(URL_PATTERN.findall(text))
    return {
        "words": float(words),
        "chars": float(len(text)),
        "alpha_ratio": safe_ratio(alpha, non_space),
        "digit_ratio": safe_ratio(digits, non_space),
        "symbol_ratio": safe_ratio(symbols, non_space),
        "lexical_diversity": unique_ratio,
        "url_count": float(urls),
    }


def detect_language(text: str) -> Tuple[str, float]:
    if len(tokenize_words(text)) < 30:
        return "unknown", 0.0
    try:
        candidates = detect_langs(text[:2200])
    except LangDetectException:
        return "unknown", 0.0
    if not candidates:
        return "unknown", 0.0
    top = candidates[0]
    return (getattr(top, "lang", "unknown") or "unknown"), float(getattr(top, "prob", 0.0) or 0.0)


def extract_domain_from_filename(path: Path) -> str:
    match = re.match(r"^([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:_|$)", path.name)
    if match:
        return match.group(1).lower()
    return "unknown"


def compute_noise_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    noisy = 0
    for line in lines:
        line_lower = line.lower()
        if len(line_lower) <= 2:
            noisy += 1
            continue
        if URL_PATTERN.search(line_lower):
            noisy += 1
            continue
        if any(pattern.match(line_lower) for pattern in NOISE_LINE_PATTERNS):
            noisy += 1
    return safe_ratio(noisy, len(lines))


def compute_quality_score(metrics: Dict[str, float], lang: str, lang_conf: float, noise_ratio: float, mojibake_ratio: float) -> int:
    score = 100.0
    words = metrics["words"]
    chars = metrics["chars"]

    if words < MIN_DOC_WORDS:
        score -= min(25.0, (MIN_DOC_WORDS - words) * 0.25)
    if chars < MIN_DOC_CHARS:
        score -= min(20.0, (MIN_DOC_CHARS - chars) / 50.0)
    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        score -= (MIN_ALPHA_RATIO - metrics["alpha_ratio"]) * 120.0
    if metrics["symbol_ratio"] > MAX_SYMBOL_RATIO:
        score -= (metrics["symbol_ratio"] - MAX_SYMBOL_RATIO) * 140.0
    if metrics["digit_ratio"] > MAX_DIGIT_RATIO:
        score -= (metrics["digit_ratio"] - MAX_DIGIT_RATIO) * 100.0
    if metrics["lexical_diversity"] < MIN_LEXICAL_DIVERSITY:
        score -= (MIN_LEXICAL_DIVERSITY - metrics["lexical_diversity"]) * 120.0
    if metrics["url_count"] > MAX_URLS_PER_CHUNK:
        score -= (metrics["url_count"] - MAX_URLS_PER_CHUNK) * 4.0

    if lang not in ALLOWED_LANGS:
        score -= 18.0
    if lang_conf < MIN_LANG_CONFIDENCE:
        score -= (MIN_LANG_CONFIDENCE - lang_conf) * 60.0
    score -= noise_ratio * 20.0
    score -= mojibake_ratio * 120.0

    return max(0, min(100, int(round(score))))


def html_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        soup = BeautifulSoup(handle, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    main = soup.find(["main", "article"]) or soup.body or soup
    parts = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    if not parts:
        parts = [main.get_text(" ", strip=True)]
    return "\n".join(parts)


def pdf_text(path: Path) -> str:
    items: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                items.append(txt)
    return "\n".join(items)


def docx_text(path: Path) -> str:
    document = docx.Document(path)
    return "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())


def plain_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        return html_text(path)
    if ext == ".pdf":
        return pdf_text(path)
    if ext == ".docx":
        return docx_text(path)
    if ext in {".txt", ".md"}:
        return plain_text(path)
    return ""


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    acc = [0] * 64
    for token in set(tokens):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for bit in range(64):
            acc[bit] += 1 if ((h >> bit) & 1) else -1
    result = 0
    for bit, val in enumerate(acc):
        if val > 0:
            result |= (1 << bit)
    return result


def hamming_distance_64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_doc_flags(metrics: Dict[str, float], lang: str, lang_conf: float, noise_ratio: float, mojibake_ratio: float, topic_relevant: bool, quality_score: int) -> List[str]:
    flags: List[str] = []
    if lang not in ALLOWED_LANGS:
        flags.append("lang_not_allowed")
    if lang_conf < MIN_LANG_CONFIDENCE:
        flags.append("lang_conf_low")
    if metrics["words"] < MIN_DOC_WORDS:
        flags.append("too_short_words")
    if metrics["chars"] < MIN_DOC_CHARS:
        flags.append("too_short_chars")
    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        flags.append("alpha_ratio_low")
    if metrics["symbol_ratio"] > MAX_SYMBOL_RATIO:
        flags.append("symbol_ratio_high")
    if metrics["digit_ratio"] > MAX_DIGIT_RATIO:
        flags.append("digit_ratio_high")
    if metrics["lexical_diversity"] < MIN_LEXICAL_DIVERSITY:
        flags.append("lexical_diversity_low")
    if metrics["url_count"] > MAX_URLS_PER_CHUNK:
        flags.append("too_many_urls")
    if noise_ratio > 0.38:
        flags.append("structural_noise")
    if mojibake_ratio > 0.004:
        flags.append("encoding_broken")
    if not topic_relevant:
        flags.append("off_topic")
    if quality_score < MIN_CHUNK_QUALITY_SCORE:
        flags.append("quality_score_low")
    return flags


def iter_raw_files() -> Iterable[Path]:
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def audit_documents() -> Tuple[List[DocumentAudit], Dict]:
    docs: List[DocumentAudit] = []
    extension_counter = Counter()
    language_counter = Counter()
    domain_counter = Counter()
    extraction_errors = []
    total_size = 0

    for file_path in iter_raw_files():
        ext = file_path.suffix.lower()
        extension_counter[ext] += 1
        total_size += file_path.stat().st_size
        domain = extract_domain_from_filename(file_path)
        domain_counter[domain] += 1

        try:
            extracted = normalize_text(extract_text(file_path))
        except Exception as exc:
            extraction_errors.append({"path": str(file_path), "error": str(exc)})
            extracted = ""

        metrics = text_metrics(extracted)
        lang, lang_conf = detect_language(extracted)
        language_counter[lang] += 1
        noise_ratio = compute_noise_line_ratio(extracted)
        mojibake_hits = len(MOJIBAKE_PATTERN.findall(extracted))
        mojibake_ratio = safe_ratio(mojibake_hits, max(1, len(extracted)))
        topic_relevant = bool(TOPIC_PATTERN.search(f"{file_path.name} {extracted[:5000]}"))
        q_score = compute_quality_score(metrics, lang, lang_conf, noise_ratio, mojibake_ratio)
        normalized = re.sub(r"\s+", " ", extracted.lower()).strip()
        flags = build_doc_flags(metrics, lang, lang_conf, noise_ratio, mojibake_ratio, topic_relevant, q_score)

        doc = DocumentAudit(
            path=str(file_path),
            extension=ext,
            size_bytes=file_path.stat().st_size,
            domain=domain,
            extracted_chars=int(metrics["chars"]),
            words=int(metrics["words"]),
            alpha_ratio=round(metrics["alpha_ratio"], 4),
            symbol_ratio=round(metrics["symbol_ratio"], 4),
            digit_ratio=round(metrics["digit_ratio"], 4),
            lexical_diversity=round(metrics["lexical_diversity"], 4),
            url_count=int(metrics["url_count"]),
            lang=lang,
            lang_confidence=round(lang_conf, 4),
            quality_score=q_score,
            text_hash=text_hash(extracted),
            normalized_hash=text_hash(normalized),
            simhash64=compute_simhash64(tokenize_words(normalized[:25000])),
            noise_line_ratio=round(noise_ratio, 4),
            mojibake_ratio=round(mojibake_ratio, 5),
            is_topic_relevant=topic_relevant,
            flags=flags,
        )
        docs.append(doc)

    summary = {
        "files_total": len(docs),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "extensions": dict(extension_counter),
        "languages": dict(language_counter),
        "top_domains": domain_counter.most_common(20),
        "extraction_errors": extraction_errors,
    }
    return docs, summary


def resolve_duplicates(docs: List[DocumentAudit]) -> Tuple[Dict[str, List[str]], List[Dict]]:
    eligible = [d for d in docs if not d.flags]
    exact_groups: Dict[str, List[DocumentAudit]] = defaultdict(list)
    for d in eligible:
        exact_groups[d.normalized_hash].append(d)

    duplicate_reasons: Dict[str, List[str]] = defaultdict(list)
    survivors: Set[str] = {d.path for d in eligible}

    for _, group in exact_groups.items():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda x: (x.quality_score, x.words, -x.url_count), reverse=True)
        keep = ordered[0]
        for loser in ordered[1:]:
            duplicate_reasons[loser.path].append(f"exact_duplicate_of:{keep.path}")
            survivors.discard(loser.path)

    candidate_by_domain: Dict[str, List[DocumentAudit]] = defaultdict(list)
    for d in eligible:
        if d.path in survivors:
            candidate_by_domain[d.domain].append(d)

    near_pairs: List[Dict] = []
    for _, group in candidate_by_domain.items():
        group_sorted = sorted(group, key=lambda x: x.extracted_chars)
        for i in range(len(group_sorted)):
            a = group_sorted[i]
            if a.path not in survivors:
                continue
            for j in range(i + 1, len(group_sorted)):
                b = group_sorted[j]
                if b.path not in survivors:
                    continue
                if b.extracted_chars > int(a.extracted_chars / 0.90):
                    pass
                len_ratio = safe_ratio(min(a.extracted_chars, b.extracted_chars), max(a.extracted_chars, b.extracted_chars))
                if len_ratio < 0.92:
                    if b.extracted_chars > a.extracted_chars:
                        break
                    continue
                if hamming_distance_64(a.simhash64, b.simhash64) > 6:
                    continue

                text_a = normalize_text(extract_text(Path(a.path)))[:22000].lower()
                text_b = normalize_text(extract_text(Path(b.path)))[:22000].lower()
                sim = SequenceMatcher(None, text_a, text_b).ratio()
                if sim >= NEAR_DUP_THRESHOLD:
                    better = a if (a.quality_score, a.words) >= (b.quality_score, b.words) else b
                    loser = b if better is a else a
                    survivors.discard(loser.path)
                    duplicate_reasons[loser.path].append(f"near_duplicate_of:{better.path}|similarity:{sim:.4f}")
                    near_pairs.append({"a": a.path, "b": b.path, "similarity": round(sim, 4), "kept": better.path})

    return duplicate_reasons, near_pairs


def build_audit_report(docs: List[DocumentAudit], summary: Dict, duplicate_reasons: Dict[str, List[str]], near_pairs: List[Dict], timestamp: str) -> Dict:
    flag_counter = Counter()
    low_conf = 0
    unauthorized_lang = 0
    for d in docs:
        flag_counter.update(d.flags)
        if d.lang_confidence < MIN_LANG_CONFIDENCE:
            low_conf += 1
        if d.lang not in ALLOWED_LANGS:
            unauthorized_lang += 1

    exact_dup_count = sum(1 for reasons in duplicate_reasons.values() if any(r.startswith("exact_duplicate_of") for r in reasons))
    near_dup_count = sum(1 for reasons in duplicate_reasons.values() if any(r.startswith("near_duplicate_of") for r in reasons))

    report = {
        "generated_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "thresholds": {
            "allowed_languages": sorted(ALLOWED_LANGS),
            "min_lang_confidence": MIN_LANG_CONFIDENCE,
            "min_doc_words": MIN_DOC_WORDS,
            "min_doc_chars": MIN_DOC_CHARS,
            "min_alpha_ratio": MIN_ALPHA_RATIO,
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "max_digit_ratio": MAX_DIGIT_RATIO,
            "min_lexical_diversity": MIN_LEXICAL_DIVERSITY,
            "max_urls_per_chunk": MAX_URLS_PER_CHUNK,
            "min_chunk_quality_score": MIN_CHUNK_QUALITY_SCORE,
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        },
        "summary": summary,
        "quality": {
            "documents_with_any_flag": sum(1 for d in docs if d.flags),
            "documents_without_flags": sum(1 for d in docs if not d.flags),
            "flag_counts": dict(flag_counter),
            "low_lang_confidence_docs": low_conf,
            "unauthorized_language_docs": unauthorized_lang,
        },
        "duplication": {
            "exact_duplicate_files": exact_dup_count,
            "near_duplicate_files": near_dup_count,
            "near_duplicate_pairs": len(near_pairs),
            "near_duplicate_examples": near_pairs[:50],
        },
        "signals": {
            "encoding_broken_docs": flag_counter.get("encoding_broken", 0),
            "structural_noise_docs": flag_counter.get("structural_noise", 0),
            "off_topic_docs": flag_counter.get("off_topic", 0),
        },
        "documents": [
            {
                "path": d.path,
                "extension": d.extension,
                "size_bytes": d.size_bytes,
                "domain": d.domain,
                "language": d.lang,
                "language_confidence": d.lang_confidence,
                "words": d.words,
                "chars": d.extracted_chars,
                "alpha_ratio": d.alpha_ratio,
                "symbol_ratio": d.symbol_ratio,
                "digit_ratio": d.digit_ratio,
                "lexical_diversity": d.lexical_diversity,
                "url_count": d.url_count,
                "noise_line_ratio": d.noise_line_ratio,
                "mojibake_ratio": d.mojibake_ratio,
                "quality_score": d.quality_score,
                "topic_relevant": d.is_topic_relevant,
                "flags": d.flags + duplicate_reasons.get(d.path, []),
            }
            for d in docs
        ],
    }
    return report


def write_audit_files(report: Dict, timestamp: str) -> Dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"raw_quality_audit_{timestamp}.json"
    txt_path = REPORTS_DIR / f"raw_quality_audit_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "RAW QUALITY AUDIT",
        f"Generated at: {report['generated_at']}",
        "",
        f"Files total: {report['summary']['files_total']}",
        f"Total size (MB): {report['summary']['total_size_mb']}",
        f"Extensions: {report['summary']['extensions']}",
        f"Languages: {report['summary']['languages']}",
        f"Top domains: {report['summary']['top_domains'][:10]}",
        "",
        "[Quality]",
        f"Docs without flags: {report['quality']['documents_without_flags']}",
        f"Docs with flags: {report['quality']['documents_with_any_flag']}",
        f"Flag counts: {report['quality']['flag_counts']}",
        "",
        "[Duplication]",
        f"Exact duplicate files: {report['duplication']['exact_duplicate_files']}",
        f"Near duplicate files: {report['duplication']['near_duplicate_files']}",
        f"Near duplicate pairs: {report['duplication']['near_duplicate_pairs']}",
        "",
        "[Signals]",
        f"Encoding broken docs: {report['signals']['encoding_broken_docs']}",
        f"Structural noise docs: {report['signals']['structural_noise_docs']}",
        f"Off-topic docs: {report['signals']['off_topic_docs']}",
    ]
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return {"json": json_path, "txt": txt_path}


def create_raw_backup(timestamp: str) -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUPS_DIR / f"raw_{timestamp}"
    if target.exists():
        raise RuntimeError(f"Backup target already exists: {target}")
    shutil.copytree(RAW_DIR, target)
    return target


def pick_primary_reason(flags: List[str]) -> str:
    priority = [
        "lang_not_allowed",
        "lang_conf_low",
        "encoding_broken",
        "structural_noise",
        "off_topic",
        "too_short_words",
        "too_short_chars",
        "alpha_ratio_low",
        "symbol_ratio_high",
        "digit_ratio_high",
        "lexical_diversity_low",
        "too_many_urls",
        "quality_score_low",
    ]
    for key in priority:
        if key in flags:
            return key
    for flag in flags:
        if flag.startswith("exact_duplicate_of"):
            return "exact_duplicate"
    for flag in flags:
        if flag.startswith("near_duplicate_of"):
            return "near_duplicate"
    return "other"


def unique_destination(base_dir: Path, rel_path: Path) -> Path:
    destination = base_dir / rel_path
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    index = 1
    while True:
        candidate = parent / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def curate_dataset(report: Dict, timestamp: str) -> Dict:
    backup_path = create_raw_backup(timestamp)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    actions = []
    counts = Counter()
    for row in report["documents"]:
        path = Path(row["path"])
        flags = row.get("flags", [])
        if not flags:
            continue
        reason = pick_primary_reason(flags)
        rel = path.relative_to(RAW_DIR) if path.is_relative_to(RAW_DIR) else Path(path.name)
        target_dir = QUARANTINE_DIR / reason / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(target_dir, Path(rel.name))

        if path.exists():
            shutil.move(str(path), str(destination))
            counts[reason] += 1
            actions.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "move_to_quarantine",
                    "reason": reason,
                    "all_reasons": flags,
                    "source": str(path),
                    "destination": str(destination),
                    "metrics": {
                        "language": row.get("language"),
                        "language_confidence": row.get("language_confidence"),
                        "words": row.get("words"),
                        "chars": row.get("chars"),
                        "alpha_ratio": row.get("alpha_ratio"),
                        "symbol_ratio": row.get("symbol_ratio"),
                        "digit_ratio": row.get("digit_ratio"),
                        "lexical_diversity": row.get("lexical_diversity"),
                        "quality_score": row.get("quality_score"),
                        "topic_relevant": row.get("topic_relevant"),
                    },
                }
            )

    kept_files = sum(1 for row in report["documents"] if not row.get("flags"))
    curation = {
        "generated_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "backup_path": str(backup_path),
        "raw_dir": str(RAW_DIR),
        "quarantine_dir": str(QUARANTINE_DIR),
        "summary": {
            "files_before": len(report["documents"]),
            "files_kept": kept_files,
            "files_quarantined": len(actions),
            "quarantined_by_reason": dict(counts),
        },
        "actions": actions,
    }
    return curation


def write_curation_files(curation: Dict, timestamp: str) -> Dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"raw_curation_actions_{timestamp}.json"
    txt_path = REPORTS_DIR / f"raw_curation_actions_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(curation, handle, indent=2, ensure_ascii=False)

    lines = [
        "RAW CURATION ACTIONS",
        f"Generated at: {curation['generated_at']}",
        f"Backup path: {curation['backup_path']}",
        f"Files before: {curation['summary']['files_before']}",
        f"Files kept: {curation['summary']['files_kept']}",
        f"Files quarantined: {curation['summary']['files_quarantined']}",
        f"Quarantined by reason: {curation['summary']['quarantined_by_reason']}",
        "",
        "Sample actions (first 30):",
    ]
    for action in curation["actions"][:30]:
        lines.append(
            f"- {action['reason']} | {action['source']} -> {action['destination']} | "
            f"quality={action['metrics']['quality_score']} lang={action['metrics']['language']} "
            f"conf={action['metrics']['language_confidence']}"
        )

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return {"json": json_path, "txt": txt_path}


def run(audit_only: bool) -> Dict[str, Dict[str, Path]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    docs, summary = audit_documents()
    duplicate_reasons, near_pairs = resolve_duplicates(docs)
    report = build_audit_report(docs, summary, duplicate_reasons, near_pairs, timestamp)
    audit_files = write_audit_files(report, timestamp)

    outputs: Dict[str, Dict[str, Path]] = {"audit": audit_files}
    if not audit_only:
        curation = curate_dataset(report, timestamp)
        outputs["curation"] = write_curation_files(curation, timestamp)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit + curation qualite des donnees RAW pour pipeline RAG.")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Genere uniquement le rapport d'audit sans deplacer de fichiers.",
    )
    args = parser.parse_args()

    outputs = run(audit_only=args.audit_only)
    print(f"Audit JSON: {outputs['audit']['json']}")
    print(f"Audit TXT : {outputs['audit']['txt']}")
    if "curation" in outputs:
        print(f"Curation JSON: {outputs['curation']['json']}")
        print(f"Curation TXT : {outputs['curation']['txt']}")


if __name__ == "__main__":
    main()
