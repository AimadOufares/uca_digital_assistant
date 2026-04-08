import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import unidecode
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

try:
    from ..offline.indexing import DEFAULT_EMBEDDING_MODEL
    from ..retrieval.bm25_search import build_bm25_index, load_bm25_corpus, search_bm25
    from ..shared.index_manifest import load_manifest, validate_manifest
    from ..shared.relevance_policy import boost_results_with_metadata
except ImportError:  # pragma: no cover
    from rag_module.offline.indexing import DEFAULT_EMBEDDING_MODEL
    from rag_module.retrieval.bm25_search import build_bm25_index, load_bm25_corpus, search_bm25
    from rag_module.shared.index_manifest import load_manifest, validate_manifest
    from rag_module.shared.relevance_policy import boost_results_with_metadata


INDEX_PATH = "data_storage/index/index.faiss"
CHUNKS_PATH = "data_storage/index/chunks.json"
MANIFEST_PATH = "data_storage/index/index_manifest.json"
BM25_CORPUS_PATH = "data_storage/index/bm25_corpus.json"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
MAX_CONTEXT_CHARS = 2500
DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35

USE_RERANK = True
USE_SPELLCHECK = False
USE_MULTI_QUERY = True
USE_ASCII_NORMALIZATION = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_embedding_model = None
_embedding_model_name = None
_reranker = None
_index = None
_chunks = None
_manifest = None
_bm25_corpus = None
_bm25_index = None
_index_mtime = None
_chunks_mtime = None
_manifest_mtime = None
_bm25_mtime = None


def invalidate_search_cache(clear_models: bool = False) -> None:
    global _index, _chunks, _manifest, _bm25_corpus, _bm25_index
    global _index_mtime, _chunks_mtime, _manifest_mtime, _bm25_mtime
    global _embedding_model, _embedding_model_name, _reranker

    _index = None
    _chunks = None
    _manifest = None
    _bm25_corpus = None
    _bm25_index = None
    _index_mtime = None
    _chunks_mtime = None
    _manifest_mtime = None
    _bm25_mtime = None
    embed_text.cache_clear()
    if clear_models:
        _embedding_model = None
        _embedding_model_name = None
        _reranker = None


def _configured_embedding_model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", "").strip()


def get_runtime_embedding_model_name() -> str:
    configured = _configured_embedding_model_name()
    manifest = load_manifest_or_raise()
    manifest_model = str(manifest.get("model_name") or "").strip()
    if configured and manifest_model and configured != manifest_model:
        raise ValueError(
            f"Le modele runtime '{configured}' ne correspond pas au modele de l'index '{manifest_model}'."
        )
    return manifest_model or configured or DEFAULT_EMBEDDING_MODEL


def is_e5_model(model_name: str) -> bool:
    return "e5" in (model_name or "").lower()


def get_embedding_model():
    global _embedding_model, _embedding_model_name
    model_name = get_runtime_embedding_model_name()
    if _embedding_model is None or _embedding_model_name != model_name:
        logger.info("Chargement du modele embedding: %s", model_name)
        try:
            _embedding_model = SentenceTransformer(
                model_name,
                device="cpu",
                local_files_only=True,
            )
        except Exception:
            _embedding_model = SentenceTransformer(model_name, device="cpu")
        _embedding_model.max_seq_length = 512
        _embedding_model_name = model_name
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
            try:
                _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            except Exception as exc:
                logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
                _reranker = None
        except Exception as exc:
            logger.warning("Reranker indisponible, fallback sans rerank: %s", exc)
            _reranker = None
    return _reranker


def load_manifest_or_raise() -> Dict:
    global _manifest, _manifest_mtime

    manifest_file = Path(MANIFEST_PATH)
    if not manifest_file.exists():
        raise FileNotFoundError("Manifest d'index introuvable.")

    current_mtime = manifest_file.stat().st_mtime
    if _manifest is None or _manifest_mtime != current_mtime:
        _manifest = load_manifest(MANIFEST_PATH)
        expected = _configured_embedding_model_name() or str(_manifest.get("model_name") or "").strip()
        validate_manifest(_manifest, expected_model=expected)
        _manifest_mtime = current_mtime
    return _manifest


def get_faiss_index_and_chunks() -> Tuple[faiss.Index, List[Dict]]:
    global _index, _chunks, _index_mtime, _chunks_mtime

    index_file = Path(INDEX_PATH)
    chunks_file = Path(CHUNKS_PATH)
    if not index_file.exists() or not chunks_file.exists():
        logger.error("Index FAISS ou chunks.json introuvable. Execute d'abord indexing.py")
        raise FileNotFoundError("Index FAISS non trouve")

    load_manifest_or_raise()

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
        with open(chunks_file, "r", encoding="utf-8") as handle:
            _chunks = json.load(handle)
        manifest = load_manifest_or_raise()
        if int(manifest.get("chunk_count", len(_chunks)) or len(_chunks)) != len(_chunks):
            logger.warning("Le manifest et chunks.json ne sont pas parfaitement alignes.")
        _index_mtime = current_index_mtime
        _chunks_mtime = current_chunks_mtime

    return _index, _chunks


def get_bm25_resources() -> Tuple[List[Dict], Dict]:
    global _bm25_corpus, _bm25_index, _bm25_mtime

    corpus_file = Path(BM25_CORPUS_PATH)
    if not corpus_file.exists():
        raise FileNotFoundError("Corpus BM25 introuvable.")

    current_mtime = corpus_file.stat().st_mtime
    if _bm25_corpus is None or _bm25_index is None or _bm25_mtime != current_mtime:
        logger.info("Chargement du corpus BM25...")
        _bm25_corpus = load_bm25_corpus(BM25_CORPUS_PATH)
        _bm25_index = build_bm25_index(_bm25_corpus)
        _bm25_mtime = current_mtime

    return _bm25_corpus, _bm25_index


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
        corrected = [spell.correction(word) or word for word in words]
        return " ".join(corrected)
    except Exception:
        return query


def enhance_query(query: str) -> str:
    return correct_query(preprocess_query(query))


def prepare_query_text(query: str) -> str:
    normalized = enhance_query(query)
    model_name = get_runtime_embedding_model_name()
    if is_e5_model(model_name):
        return f"query: {normalized}"
    return normalized


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


@lru_cache(maxsize=500)
def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    prepared = prepare_query_text(text)
    return model.encode(prepared, normalize_embeddings=True)


def embed_queries(queries: List[str]) -> np.ndarray:
    return np.array([embed_text(query) for query in queries], dtype="float32")


def _normalize_vector_score(raw_score: float, metric_type: int) -> float:
    if metric_type == faiss.METRIC_L2:
        return float(max(0.0, min(1.0, 1.0 - (raw_score / 2.0))))
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)))
    return raw_score


def search_faiss(query_vectors: np.ndarray, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
    index, chunks = get_faiss_index_and_chunks()
    if index.ntotal == 0:
        return []

    top_k = max(1, int(top_k))
    distances, indices = index.search(query_vectors, top_k)
    metric_type = getattr(index, "metric_type", faiss.METRIC_L2)

    results: List[Dict] = []
    for i, idx_list in enumerate(indices):
        for rank, idx in enumerate(idx_list):
            if 0 <= idx < len(chunks):
                chunk = dict(chunks[idx])
                raw_score = float(distances[i][rank])
                chunk["vector_raw_score"] = raw_score
                chunk["score"] = _normalize_vector_score(raw_score, metric_type)
                chunk["score_type"] = "dense"
                chunk["query_source"] = "multi" if len(query_vectors) > 1 else "single"
                results.append(chunk)
    return results


def merge_dense_and_bm25(dense_results: List[Dict], bm25_results: List[Dict], top_k: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}

    for result in dense_results:
        metadata = result.get("metadata", {}) or {}
        chunk_id = result.get("id") or metadata.get("chunk_hash") or metadata.get("hash")
        if not chunk_id:
            continue

        entry = merged.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "text": result.get("text", ""),
                "metadata": metadata,
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "score": 0.0,
                "score_type": "hybrid",
                "retrieval_sources": [],
            },
        )
        entry["dense_score"] = max(float(entry["dense_score"]), float(result.get("score", 0.0) or 0.0))
        if "dense" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("dense")

    for result in bm25_results:
        metadata = result.get("metadata", {}) or {}
        chunk_id = result.get("id") or metadata.get("chunk_hash") or metadata.get("hash")
        if not chunk_id:
            continue

        entry = merged.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "text": result.get("text", ""),
                "metadata": metadata,
                "dense_score": 0.0,
                "bm25_score": 0.0,
                "score": 0.0,
                "score_type": "hybrid",
                "retrieval_sources": [],
            },
        )
        entry["bm25_score"] = max(float(entry["bm25_score"]), float(result.get("score", 0.0) or 0.0))
        if "bm25" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("bm25")

    merged_results: List[Dict] = []
    for entry in merged.values():
        dense_score = float(entry.get("dense_score", 0.0))
        bm25_score = float(entry.get("bm25_score", 0.0))
        entry["score"] = max(0.0, min(1.0, (dense_score * DENSE_WEIGHT) + (bm25_score * BM25_WEIGHT)))
        merged_results.append(entry)

    merged_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return merged_results[: max(top_k, TOP_K_RETRIEVE)]


def deduplicate_chunks(chunks_list: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for chunk in chunks_list:
        metadata = chunk.get("metadata", {}) or {}
        text = (chunk.get("text", "") or "").strip().lower()
        text_fallback = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else None
        chunk_id = chunk.get("id") or metadata.get("chunk_hash") or metadata.get("hash") or text_fallback
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk)
    return unique


def apply_metadata_boost(results: List[Dict], query: str) -> List[Dict]:
    return boost_results_with_metadata(results, query)


def rerank_chunks(query: str, chunks_list: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not USE_RERANK or not chunks_list:
        return chunks_list[:top_k]

    reranker = get_reranker()
    if reranker is None:
        return chunks_list[:top_k]

    pairs = [(query, chunk.get("text", "")) for chunk in chunks_list]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        logger.warning("Reranking indisponible, fallback sans rerank: %s", exc)
        return chunks_list[:top_k]

    ranked = sorted(zip(chunks_list, scores), key=lambda item: item[1], reverse=True)
    selected = []
    for chunk, rerank_score in ranked[:top_k]:
        enriched = dict(chunk)
        enriched["rerank_score"] = float(rerank_score)
        enriched["score_type"] = "rerank"
        selected.append(enriched)
    return selected


def truncate_chunks(chunks_list: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> List[Dict]:
    total = 0
    selected = []
    for chunk in chunks_list:
        text = chunk.get("text", "")
        if total + len(text) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(text)
    return selected


def run_hybrid_search_debug(raw_query: str, top_k: int = TOP_K_FINAL) -> Dict[str, object]:
    if not raw_query or not raw_query.strip():
        return {
            "query": "",
            "dense_results": [],
            "bm25_results": [],
            "merged_results": [],
            "boosted_results": [],
            "final_results": [],
        }

    top_k = max(1, int(top_k))
    query = enhance_query(raw_query)
    logger.info("Recherche hybride pour: '%s' -> '%s'", raw_query, query)

    queries = generate_multi_queries(query)
    query_vectors = embed_queries(queries)
    retrieve_k = max(TOP_K_RETRIEVE, top_k * 4)

    dense_results = search_faiss(query_vectors, top_k=retrieve_k)
    _, bm25_index = get_bm25_resources()
    bm25_results = search_bm25(query, bm25_index, top_k=retrieve_k)

    retrieved = merge_dense_and_bm25(dense_results, bm25_results, top_k=retrieve_k)
    retrieved = deduplicate_chunks(retrieved)
    boosted = apply_metadata_boost(retrieved, query)
    reranked = rerank_chunks(query, boosted, top_k=top_k)
    final_results = truncate_chunks(reranked, MAX_CONTEXT_CHARS)

    logger.info("%s chunks pertinents retournes apres fusion et reranking", len(final_results))
    return {
        "query": query,
        "dense_results": dense_results,
        "bm25_results": bm25_results,
        "merged_results": retrieved,
        "boosted_results": boosted,
        "final_results": final_results,
    }


def get_relevant_chunks(raw_query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
    debug_payload = run_hybrid_search_debug(raw_query, top_k=top_k)
    return list(debug_payload.get("final_results", []))


if __name__ == "__main__":
    test_queries = [
        "Comment s'inscrire a Semlalia ?",
        "Quelles sont les conditions d'admission a la faculte Semlalia ?",
        "Procedure inscription universite Cadi Ayyad",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        results = get_relevant_chunks(query, top_k=5)
        for i, result in enumerate(results, 1):
            score = float(result.get("score", 0.0) or 0.0)
            source = result.get("metadata", {}).get("source", "unknown")
            print(f"\n[{i}] Score: {score:.4f} | Source: {Path(source).name}")
            text = result.get("text", "")
            print(f"    {text[:280]}..." if len(text) > 280 else text)
