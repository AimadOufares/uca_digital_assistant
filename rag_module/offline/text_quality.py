import hashlib
import re
import string
import unicodedata
from html import unescape
from typing import Dict, List


MIN_WORDS = 8
MIN_DOC_WORDS = 120
MIN_DOC_CHARS = 700
MIN_QUALITY_SCORE = 55
MIN_ALPHA_RATIO = 0.60
MAX_DIGIT_RATIO = 0.30
MAX_SYMBOL_RATIO = 0.22
MIN_UNIQUE_TOKEN_RATIO = 0.30
MAX_URLS_PER_CHUNK = 1
MAX_REPEAT_CHAR_RUN = 6

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

MOJIBAKE_TOKENS = ("Ã", "Â", "Ă", "â", "ð", "�")


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def mojibake_score(text: str) -> int:
    return sum(text.count(token) for token in MOJIBAKE_TOKENS)


def repair_mojibake(text: str) -> str:
    if not text or mojibake_score(text) == 0:
        return text

    best_text = text
    best_score = mojibake_score(text)
    for encoding in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(encoding, errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            continue
        if not candidate:
            continue
        candidate_score = mojibake_score(candidate)
        if candidate_score < best_score and len(candidate.strip()) >= max(10, int(len(text.strip()) * 0.6)):
            best_text = candidate
            best_score = candidate_score
    return best_text


def corpus_policy(corpus: str) -> Dict[str, float]:
    return CORPUS_POLICIES.get(corpus, CORPUS_POLICIES["main"])


def looks_like_url_or_path(line: str, corpus: str = "main") -> bool:
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


def is_noise_line(line: str, corpus: str = "main") -> bool:
    candidate = line.strip()
    if not candidate:
        return True

    policy = corpus_policy(corpus)
    preserve_short_lines = bool(policy.get("preserve_short_lines"))

    if len(candidate) <= 2 and not preserve_short_lines:
        return True

    for pattern in NOISE_LINE_REGEXES:
        if re.match(pattern, candidate, flags=re.IGNORECASE):
            return True

    lower = candidate.lower()
    if any(phrase in lower for phrase in NOISE_PHRASES):
        return True
    if looks_like_url_or_path(candidate, corpus=corpus):
        return True

    non_space_len = sum(1 for ch in candidate if not ch.isspace())
    if non_space_len == 0:
        return True

    alpha_count = sum(1 for ch in candidate if ch.isalpha())
    symbol_count = sum(1 for ch in candidate if not ch.isalnum() and not ch.isspace())
    if not preserve_short_lines and safe_ratio(alpha_count, non_space_len) < 0.25:
        return True
    if safe_ratio(symbol_count, non_space_len) > 0.45:
        return True
    return False


def text_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize_words(text)
    words = len(tokens)
    unique_ratio = safe_ratio(len(set(tokens)), words)

    non_space_len = sum(1 for ch in text if not ch.isspace())
    alpha_count = sum(1 for ch in text if ch.isalpha())
    digit_count = sum(1 for ch in text if ch.isdigit())
    symbol_count = sum(
        1
        for ch in text
        if ch in string.punctuation or (not ch.isalnum() and not ch.isspace())
    )
    sentence_count = len(
        [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(tokenize_words(s)) >= 3]
    )
    url_count = len(re.findall(r"https?://|www\.", text.lower()))
    repeated_run = bool(re.search(rf"(.)\1{{{MAX_REPEAT_CHAR_RUN},}}", text))

    return {
        "words": float(words),
        "unique_ratio": unique_ratio,
        "alpha_ratio": safe_ratio(alpha_count, non_space_len),
        "digit_ratio": safe_ratio(digit_count, non_space_len),
        "symbol_ratio": safe_ratio(symbol_count, non_space_len),
        "sentence_count": float(sentence_count),
        "url_count": float(url_count),
        "repeated_run": 1.0 if repeated_run else 0.0,
    }


def clean_text(text: str, corpus: str = "main") -> str:
    if not text:
        return ""

    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = unescape(text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]+", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    previous_line = ""
    seen_line_hashes = set()

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip(" -|\t")
        if is_noise_line(line, corpus=corpus):
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

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()


def quality_score(text: str) -> int:
    if not text:
        return 0

    metrics = text_metrics(text)
    score = 0.0
    score += min(metrics["words"], 180.0) * 0.22
    score += min(metrics["sentence_count"], 12.0) * 2.8
    score += min(metrics["unique_ratio"], 1.0) * 24.0
    score += min(metrics["alpha_ratio"], 1.0) * 20.0
    score -= metrics["digit_ratio"] * 30.0
    score -= metrics["symbol_ratio"] * 35.0
    score -= max(0.0, metrics["url_count"] - 1.0) * 8.0
    if metrics["repeated_run"] > 0:
        score -= 15.0
    return int(max(0.0, min(100.0, round(score))))


def is_high_quality_chunk(text: str, corpus: str = "main") -> bool:
    policy = corpus_policy(corpus)
    metrics = text_metrics(text)
    if metrics["words"] < policy["min_words"]:
        return False
    if metrics["alpha_ratio"] < policy["min_alpha_ratio"]:
        return False
    if metrics["digit_ratio"] > policy["max_digit_ratio"]:
        return False
    if metrics["symbol_ratio"] > policy["max_symbol_ratio"]:
        return False
    if metrics["unique_ratio"] < policy["min_unique_ratio"]:
        return False
    if metrics["url_count"] > policy["max_urls_per_chunk"]:
        return False
    return quality_score(text) >= policy["min_quality_score"]


def is_high_quality_document(text: str, corpus: str = "main") -> bool:
    policy = corpus_policy(corpus)
    if len(text) < policy["min_doc_chars"]:
        return False
    if len(tokenize_words(text)) < policy["min_doc_words"]:
        return False

    metrics = text_metrics(text[: min(len(text), 6000)])
    if metrics["alpha_ratio"] < policy["min_alpha_ratio"]:
        return False
    if metrics["digit_ratio"] > policy["max_digit_ratio"]:
        return False
    if metrics["symbol_ratio"] > policy["max_symbol_ratio"]:
        return False
    if metrics["unique_ratio"] < policy["min_unique_ratio"]:
        return False
    return quality_score(text[: min(len(text), 6000)]) >= policy["min_quality_score"]


def deduplicate_chunk_texts(chunks: List[str]) -> List[str]:
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


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip() and not is_noise_line(sentence)]
