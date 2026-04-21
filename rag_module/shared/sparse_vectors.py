import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .metadata_policy import normalize_text


WORD_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "a", "au", "aux",
    "dans", "par", "pour", "sur", "avec", "en", "the", "an", "and", "in", "on", "at",
    "to", "for", "with", "is", "are", "of", "qui", "que", "quoi", "dont", "ce", "cet",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "pas", "ne", "ni", "se",
    "il", "elle", "ils", "elles", "nous", "vous", "je", "tu", "me", "te"
}

def is_noise_token(token: str) -> bool:
    if len(token) < 2:
        return True
    if token in STOPWORDS:
        return True
    if re.fullmatch(r"\d+([hmds]?[a-zA-Z]*)?", token):
        return True
    return False

def tokenize_sparse_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [token for token in WORD_PATTERN.findall(normalized) if token and not is_noise_token(token)]


def build_sparse_encoder(
    texts: Iterable[str],
    min_df: int = 2,
    max_features: int = 50000,
) -> Dict:
    document_frequencies: Counter = Counter()
    doc_count = 0

    for text in texts:
        tokens = set(tokenize_sparse_text(text))
        if not tokens:
            continue
        doc_count += 1
        document_frequencies.update(tokens)

    kept_tokens = [
        token
        for token, df in document_frequencies.most_common(max_features)
        if int(df) >= int(min_df)
    ]
    vocabulary = {token: index for index, token in enumerate(sorted(kept_tokens))}
    idf = {
        token: round(math.log((doc_count + 1.0) / (document_frequencies[token] + 1.0)) + 1.0, 6)
        for token in vocabulary
    }
    return {
        "doc_count": int(doc_count),
        "min_df": int(min_df),
        "max_features": int(max_features),
        "vocabulary": vocabulary,
        "idf": idf,
    }


def encode_sparse_text(text: str, encoder: Dict) -> Tuple[List[int], List[float]]:
    vocabulary = encoder.get("vocabulary", {}) or {}
    idf = encoder.get("idf", {}) or {}
    term_counts = Counter(tokenize_sparse_text(text))
    if not term_counts:
        return [], []

    max_tf = max(term_counts.values()) or 1
    features: List[Tuple[int, float]] = []
    for token, tf in term_counts.items():
        index = vocabulary.get(token)
        if index is None:
            continue
        weight = (0.5 + (0.5 * float(tf) / float(max_tf))) * float(idf.get(token, 1.0))
        features.append((int(index), round(weight, 6)))

    features.sort(key=lambda item: item[0])
    return [idx for idx, _ in features], [value for _, value in features]


def save_sparse_encoder(path: str, encoder: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f"{target.name}.tmp")
    with temp_target.open("w", encoding="utf-8") as handle:
        json.dump(encoder, handle, ensure_ascii=False, indent=2)
    os.replace(temp_target, target)


def load_sparse_encoder(path: str) -> Dict:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}
