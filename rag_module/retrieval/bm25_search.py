import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    from ..shared.metadata_policy import normalize_text
except ImportError:  # pragma: no cover
    from rag_module.shared.metadata_policy import normalize_text


WORD_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75


def normalize_lexical_text(text: str) -> str:
    return normalize_text(text)


def tokenize_for_bm25(text: str) -> List[str]:
    normalized = normalize_lexical_text(text)
    return [token for token in WORD_PATTERN.findall(normalized) if token]


def load_bm25_corpus(path: str) -> List[Dict]:
    import json
    from pathlib import Path

    corpus_path = Path(path)
    if not corpus_path.exists():
        return []
    with corpus_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def build_bm25_index(corpus: List[Dict]) -> Dict:
    documents: List[Dict] = []
    doc_freqs = defaultdict(int)
    total_length = 0

    for row in corpus:
        text = str(row.get("text", "") or "")
        tokens = tokenize_for_bm25(text)
        if not tokens:
            continue
        token_counts = Counter(tokens)
        total_length += len(tokens)

        for token in token_counts:
            doc_freqs[token] += 1

        documents.append(
            {
                "id": row.get("id"),
                "text": text,
                "metadata": row.get("metadata", {}) or {},
                "tokens": tokens,
                "token_counts": token_counts,
                "length": len(tokens),
            }
        )

    doc_count = len(documents)
    avg_doc_len = (total_length / doc_count) if doc_count else 0.0

    idf = {}
    for token, df in doc_freqs.items():
        idf[token] = math.log(1.0 + ((doc_count - df + 0.5) / (df + 0.5)))

    return {
        "documents": documents,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def normalize_bm25_score(raw_score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, raw_score / max_score))


def _score_document(tokens: List[str], token_counts: Counter, length: int, idf: Dict, avg_doc_len: float) -> float:
    if not tokens or not token_counts or length <= 0:
        return 0.0

    score = 0.0
    for token in tokens:
        if token not in token_counts:
            continue
        tf = token_counts[token]
        token_idf = float(idf.get(token, 0.0))
        denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (length / max(avg_doc_len, 1.0)))
        score += token_idf * ((tf * (BM25_K1 + 1.0)) / max(denominator, 1e-9))
    return score


def search_bm25(query: str, bm25_index: Dict, top_k: int) -> List[Dict]:
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    documents = bm25_index.get("documents", [])
    idf = bm25_index.get("idf", {})
    avg_doc_len = float(bm25_index.get("avg_doc_len") or 0.0)

    scored: List[Tuple[float, Dict]] = []
    for doc in documents:
        raw_score = _score_document(
            tokens=query_tokens,
            token_counts=doc.get("token_counts", Counter()),
            length=int(doc.get("length", 0) or 0),
            idf=idf,
            avg_doc_len=avg_doc_len,
        )
        if raw_score > 0:
            scored.append((raw_score, doc))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    top_results = scored[: max(1, int(top_k))]
    max_score = top_results[0][0]

    results: List[Dict] = []
    for raw_score, doc in top_results:
        results.append(
            {
                "id": doc.get("id"),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}) or {},
                "bm25_raw_score": float(raw_score),
                "score": normalize_bm25_score(float(raw_score), float(max_score)),
                "score_type": "bm25",
            }
        )
    return results
