# rag_module/rag_search.py
import re
import json
import logging
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import unidecode
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# ========================= CONFIG =========================
INDEX_PATH = "data_storage/index/index.faiss"
CHUNKS_PATH = "data_storage/index/chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
MAX_CONTEXT_CHARS = 2500

USE_RERANK = True
USE_SPELLCHECK = False
USE_MULTI_QUERY = True
USE_ASCII_NORMALIZATION = False

# ========================= LOGGING =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ========================= LAZY LOADING =========================
_embedding_model = None
_reranker = None
_index = None
_chunks = None
_index_mtime = None
_chunks_mtime = None


def invalidate_search_cache(clear_models: bool = False) -> None:
    """Force reload of FAISS index and chunks from disk on next query."""
    global _index, _chunks, _index_mtime, _chunks_mtime, _embedding_model, _reranker
    _index = None
    _chunks = None
    _index_mtime = None
    _chunks_mtime = None
    if clear_models:
        _embedding_model = None
        _reranker = None
        embed_text.cache_clear()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Chargement du modele embedding: %s", EMBEDDING_MODEL)
        # Offline-first to avoid runtime failures in restricted environments.
        try:
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device="cpu",
                local_files_only=True,
            )
        except Exception:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        _embedding_model.max_seq_length = 512
    return _embedding_model


def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANK:
        logger.info("Chargement du reranker: %s", RERANK_MODEL)
        try:
            _reranker = CrossEncoder(
                RERANK_MODEL,
                device="cpu",
                local_files_only=True,
            )
        except TypeError:
            # Compatibility fallback for older sentence-transformers releases.
            try:
                _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            except Exception as exc:
                logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
                _reranker = None
        except Exception as exc:
            logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
            _reranker = None
    return _reranker


def get_faiss_index_and_chunks():
    global _index, _chunks, _index_mtime, _chunks_mtime

    index_file = Path(INDEX_PATH)
    chunks_file = Path(CHUNKS_PATH)
    if not index_file.exists() or not chunks_file.exists():
        logger.error("Index FAISS ou chunks.json introuvable. Execute d'abord indexing.py")
        raise FileNotFoundError("Index FAISS non trouve")

    current_index_mtime = index_file.stat().st_mtime
    current_chunks_mtime = chunks_file.stat().st_mtime
    needs_reload = (
        _index is None
        or _chunks is None
        or _index_mtime != current_index_mtime
        or _chunks_mtime != current_chunks_mtime
    )
    if needs_reload:
        logger.info("Chargement de l'index FAISS...")
        _index = faiss.read_index(str(index_file))
        logger.info("Chargement des chunks...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        _index_mtime = current_index_mtime
        _chunks_mtime = current_chunks_mtime

    return _index, _chunks


# ========================= QUERY PREPROCESSING =========================
def preprocess_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    if USE_ASCII_NORMALIZATION:
        query = unidecode.unidecode(query)
    return query


def correct_query(query: str) -> str:
    if not USE_SPELLCHECK:
        return query
    try:
        from spellchecker import SpellChecker

        spell = SpellChecker(language="fr")
        words = query.split()
        corrected = [spell.correction(w) or w for w in words]
        return " ".join(corrected)
    except Exception:
        return query


def enhance_query(query: str) -> str:
    return correct_query(preprocess_query(query))


# ========================= MULTI QUERY =========================
def generate_multi_queries(query: str) -> List[str]:
    if not USE_MULTI_QUERY:
        return [query]

    base = enhance_query(query)
    variations = [
        base,
        base + " explication detaillee",
        base + " informations importantes",
        "comment " + base if not base.startswith(("comment", "comment faire")) else base,
    ]
    return list(dict.fromkeys(variations))


# ========================= EMBEDDING =========================
@lru_cache(maxsize=500)
def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True)


def embed_queries(queries: List[str]) -> np.ndarray:
    return np.array([embed_text(q) for q in queries], dtype="float32")


# ========================= SEARCH =========================
def search_faiss(query_vectors: np.ndarray, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
    index, chunks = get_faiss_index_and_chunks()
    if index.ntotal == 0:
        return []

    top_k = max(1, int(top_k))
    D, I = index.search(query_vectors, top_k)
    metric_type = getattr(index, "metric_type", faiss.METRIC_L2)

    results = []
    for i, idx_list in enumerate(I):
        for rank, idx in enumerate(idx_list):
            if 0 <= idx < len(chunks):
                chunk = chunks[idx].copy()
                raw_score = float(D[i][rank])
                chunk["vector_raw_score"] = raw_score
                chunk["score"] = _normalize_vector_score(raw_score, metric_type)
                chunk["score_type"] = "vector"
                chunk["query_source"] = "multi" if len(query_vectors) > 1 else "single"
                results.append(chunk)
    return results


def deduplicate_chunks(chunks_list: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for c in chunks_list:
        metadata = c.get("metadata", {}) or {}
        text = (c.get("text", "") or "").strip().lower()
        text_fallback = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else None
        chunk_id = c.get("id") or metadata.get("chunk_hash") or metadata.get("hash") or text_fallback
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(c)
    return unique


# ========================= RERANKING =========================
def rerank_chunks(query: str, chunks_list: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not USE_RERANK or not chunks_list:
        return chunks_list[:top_k]

    reranker = get_reranker()
    if reranker is None:
        return chunks_list[:top_k]

    pairs = [(query, c.get("text", "")) for c in chunks_list]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        logger.warning("Reranking indisponible, fallback sans rerank: %s", exc)
        return chunks_list[:top_k]

    ranked = sorted(zip(chunks_list, scores), key=lambda x: x[1], reverse=True)

    selected = []
    for chunk, rerank_score in ranked[:top_k]:
        enriched = chunk.copy()
        enriched["rerank_score"] = float(rerank_score)
        enriched["score_type"] = "rerank"
        selected.append(enriched)
    return selected


# ========================= CONTEXT BUILDER =========================
def truncate_chunks(chunks_list: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> List[Dict]:
    total = 0
    selected = []
    for c in chunks_list:
        text = c.get("text", "")
        if total + len(text) > max_chars and selected:
            break
        selected.append(c)
        total += len(text)
    return selected


# ========================= MAIN SEARCH =========================
def get_relevant_chunks(raw_query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not raw_query or not raw_query.strip():
        return []

    top_k = max(1, int(top_k))
    query = enhance_query(raw_query)
    logger.info("Recherche pour: '%s' -> '%s'", raw_query, query)

    queries = generate_multi_queries(query)
    query_vectors = embed_queries(queries)

    retrieve_k = max(TOP_K_RETRIEVE, top_k * 4)
    retrieved = search_faiss(query_vectors, top_k=retrieve_k)

    retrieved = deduplicate_chunks(retrieved)
    retrieved = rerank_chunks(query, retrieved, top_k=top_k)
    retrieved = truncate_chunks(retrieved, MAX_CONTEXT_CHARS)

    logger.info("%s chunks pertinents retournes apres reranking", len(retrieved))
    return retrieved


def _normalize_vector_score(raw_score: float, metric_type: int) -> float:
    """
    Convertit un score FAISS en score [0, 1] ou plus interpretable.
    - L2 (embeddings normalises): approx cosine = 1 - d/2
    - Inner Product: remap simple de [-1, 1] vers [0, 1]
    """
    if metric_type == faiss.METRIC_L2:
        return float(max(0.0, min(1.0, 1.0 - (raw_score / 2.0))))
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)))
    return raw_score


if __name__ == "__main__":
    test_queries = [
        "Comment s'inscrire a Semlalia ?",
        "Quelles sont les conditions d'admission a la faculte Semlalia ?",
        "Procedure inscription universite Cadi Ayyad",
    ]

    for q in test_queries:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {q}")
        results = get_relevant_chunks(q, top_k=5)
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "unknown")
            print(f"\n[{i}] Score: {score:.4f} | Source: {Path(source).name}")
            print(f"    {r['text'][:280]}..." if len(r["text"]) > 280 else r["text"])
