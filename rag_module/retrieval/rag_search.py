import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import unidecode
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

try:
    from ..adapters.storage import DocumentStorage
    from ..offline.indexing import DEFAULT_EMBEDDING_MODEL
    from ..retrieval.query_intelligence import (
        QUERY_CANONICAL_REPLACEMENTS,
        apply_post_rerank_guardrails as qi_apply_post_rerank_guardrails,
        apply_retrieval_guardrails as qi_apply_retrieval_guardrails,
        build_query_profile as qi_build_query_profile,
        decide_retrieval_abstention as qi_decide_retrieval_abstention,
        score_chunk_thematic_match as qi_score_chunk_thematic_match,
    )
    from ..retrieval.bm25_search import build_bm25_index, load_bm25_corpus, search_bm25
    from ..shared.env_loader import load_env_file
    from ..shared.index_manifest import load_manifest, validate_manifest
    from ..shared.metadata_policy import normalize_text
    from ..shared.relevance_policy import boost_results_with_metadata
    from ..shared.runtime import get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.adapters.storage import DocumentStorage
    from rag_module.offline.indexing import DEFAULT_EMBEDDING_MODEL
    from rag_module.retrieval.query_intelligence import (
        QUERY_CANONICAL_REPLACEMENTS,
        apply_post_rerank_guardrails as qi_apply_post_rerank_guardrails,
        apply_retrieval_guardrails as qi_apply_retrieval_guardrails,
        build_query_profile as qi_build_query_profile,
        decide_retrieval_abstention as qi_decide_retrieval_abstention,
        score_chunk_thematic_match as qi_score_chunk_thematic_match,
    )
    from rag_module.retrieval.bm25_search import build_bm25_index, load_bm25_corpus, search_bm25
    from rag_module.shared.env_loader import load_env_file
    from rag_module.shared.index_manifest import load_manifest, validate_manifest
    from rag_module.shared.metadata_policy import normalize_text
    from rag_module.shared.relevance_policy import boost_results_with_metadata
    from rag_module.shared.runtime import get_runtime_settings

load_env_file()
RUNTIME = get_runtime_settings()
STORAGE = DocumentStorage(RUNTIME)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


INDEX_PATH = "data_storage/index/index.faiss"
CHUNKS_PATH = "data_storage/index/chunks.json"
MANIFEST_PATH = "data_storage/index/index_manifest.json"
BM25_CORPUS_PATH = "data_storage/index/bm25_corpus.json"

RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "RAG_RERANK_FALLBACK_MODELS",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ).split(",")
    if model.strip()
]

TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
MAX_CONTEXT_CHARS = 2500
DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35

USE_RERANK = os.getenv("RAG_USE_RERANK", "true").strip().lower() in {"1", "true", "yes", "on"}
USE_SPELLCHECK = False
USE_MULTI_QUERY = False
USE_ASCII_NORMALIZATION = False

MIN_GUARDRAIL_SCORE = _env_float("RAG_MIN_GUARDRAIL_SCORE", 0.24)
MIN_THEMATIC_SCORE = _env_float("RAG_MIN_THEMATIC_SCORE", 0.18)
MIN_SUPPORT_SCORE = _env_float("RAG_MIN_SUPPORT_SCORE", 0.28)
MIN_FINAL_SUPPORT_SCORE = _env_float("RAG_MIN_FINAL_SUPPORT_SCORE", 0.42)
MIN_TOP_RERANK_NORMALIZED = _env_float("RAG_MIN_TOP_RERANK_NORMALIZED", 0.44)
TOPICAL_MISMATCH_DROP_THRESHOLD = _env_float("RAG_TOPICAL_MISMATCH_DROP_THRESHOLD", 0.45)

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


def _vector_backend() -> str:
    return get_runtime_settings().rag_vector_backend


def _active_faiss_paths() -> Dict[str, Path]:
    return STORAGE.resolve_active_faiss_paths()


def _active_manifest_path() -> Path:
    pointer = STORAGE.load_active_index_pointer()
    if pointer.get("backend") == "qdrant":
        manifest_path = Path(pointer.get("manifest_path", ""))
        if manifest_path.exists():
            return manifest_path
    return _active_faiss_paths()["manifest_file"]


def _qdrant_client() -> QdrantClient:
    runtime = get_runtime_settings()
    kwargs = {"url": runtime.rag_qdrant_url}
    if runtime.rag_qdrant_api_key:
        kwargs["api_key"] = runtime.rag_qdrant_api_key
    return QdrantClient(**kwargs)


def _qdrant_collection_name() -> str:
    pointer = STORAGE.load_active_index_pointer()
    collection_name = str(pointer.get("collection_name") or "").strip()
    if collection_name:
        return collection_name
    return get_runtime_settings().rag_active_index_name


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
        logger.warning(
            "Modele runtime '%s' different du modele de l'index '%s'. Utilisation du modele de l'index.",
            configured,
            manifest_model,
        )
    return manifest_model or configured or DEFAULT_EMBEDDING_MODEL


def get_candidate_reranker_names() -> List[str]:
    candidates = [RERANK_MODEL, *RERANK_FALLBACK_MODELS]
    unique: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        value = (candidate or "").strip()
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


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
        errors: List[str] = []
        for model_name in get_candidate_reranker_names():
            logger.info("Chargement du reranker: %s", model_name)
            try:
                _reranker = CrossEncoder(
                    model_name,
                    device="cpu",
                    local_files_only=True,
                )
                break
            except TypeError:
                try:
                    _reranker = CrossEncoder(model_name, device="cpu")
                    break
                except Exception as exc:
                    errors.append(f"{model_name}: {exc}")
                    _reranker = None
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                _reranker = None
        if _reranker is None and errors:
            logger.warning("Reranker indisponible, fallback sans rerank: %s", " | ".join(errors[:3]))
    return _reranker


def load_manifest_or_raise() -> Dict:
    global _manifest, _manifest_mtime

    manifest_file = _active_manifest_path()
    if not manifest_file.exists():
        raise FileNotFoundError("Manifest d'index introuvable.")

    current_mtime = manifest_file.stat().st_mtime
    if _manifest is None or _manifest_mtime != current_mtime:
        _manifest = load_manifest(str(manifest_file))
        expected = str(_manifest.get("model_name") or "").strip()
        validate_manifest(_manifest, expected_model=expected)
        _manifest_mtime = current_mtime
    return _manifest


def get_faiss_index_and_chunks() -> Tuple[faiss.Index, List[Dict]]:
    global _index, _chunks, _index_mtime, _chunks_mtime

    paths = _active_faiss_paths()
    index_file = paths["index_file"]
    chunks_file = paths["chunks_file"]
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

    corpus_file = _active_faiss_paths()["bm25_file"]
    if not corpus_file.exists():
        raise FileNotFoundError("Corpus BM25 introuvable.")

    current_mtime = corpus_file.stat().st_mtime
    if _bm25_corpus is None or _bm25_index is None or _bm25_mtime != current_mtime:
        logger.info("Chargement du corpus BM25...")
        _bm25_corpus = load_bm25_corpus(str(corpus_file))
        _bm25_index = build_bm25_index(_bm25_corpus)
        _bm25_mtime = current_mtime

    return _bm25_corpus, _bm25_index


def preprocess_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    if USE_ASCII_NORMALIZATION:
        query = unidecode.unidecode(query)

    normalized_query = normalize_text(query)
    for pattern, replacement in QUERY_CANONICAL_REPLACEMENTS:
        normalized_query = re.sub(pattern, replacement, normalized_query)
    normalized_query = re.sub(r"\s+", " ", normalized_query).strip()

    # "faculte" est souvent employe generiquement par les utilisateurs
    # quand ils demandent une inscription universitaire.
    if "faculte" in normalized_query and "inscription" in normalized_query:
        normalized_query = normalized_query.replace("faculte", "universite")

    if normalized_query.startswith("la ") and len(normalized_query.split()) <= 3:
        normalized_query = normalized_query[3:]

    return normalized_query


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
    if "inscription" in base and "universite" not in base:
        variations.append(base + " universite")
    if "bourse" in base and "universitaire" not in base:
        variations.append(base + " universitaire")
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


def search_qdrant(query_vectors: np.ndarray, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
    runtime = get_runtime_settings()
    if not runtime.rag_qdrant_url:
        raise FileNotFoundError("Qdrant n'est pas configure.")

    client = _qdrant_client()
    collection_name = _qdrant_collection_name()
    results: List[Dict] = []

    for vector in query_vectors:
        response = client.query_points(
            collection_name=collection_name,
            query=vector.tolist(),
            limit=max(1, int(top_k)),
            with_payload=True,
            with_vectors=False,
        )
        for point in getattr(response, "points", []) or []:
            payload = getattr(point, "payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            results.append(
                {
                    "id": payload.get("id") or str(getattr(point, "id", "")),
                    "text": payload.get("text", "") or "",
                    "metadata": metadata,
                    "vector_raw_score": float(getattr(point, "score", 0.0) or 0.0),
                    "score": max(0.0, min(1.0, float(getattr(point, "score", 0.0) or 0.0))),
                    "score_type": "dense",
                    "query_source": "multi" if len(query_vectors) > 1 else "single",
                }
            )
    return results


def merge_dense_and_bm25(dense_results: List[Dict], bm25_results: List[Dict], top_k: int) -> List[Dict]:
    # Algorithme de fusion par rangs: Reciprocal Rank Fusion (RRF)
    RRF_K = 60
    
    dense_sorted = sorted(dense_results, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    bm25_sorted = sorted(bm25_results, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    merged: Dict[str, Dict] = {}

    for rank, result in enumerate(dense_sorted):
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
                "rrf_score": 0.0,
                "score_type": "hybrid_rrf",
                "retrieval_sources": [],
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        if "dense" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("dense")

    for rank, result in enumerate(bm25_sorted):
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
                "rrf_score": 0.0,
                "score_type": "hybrid_rrf",
                "retrieval_sources": [],
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        if "bm25" not in entry["retrieval_sources"]:
            entry["retrieval_sources"].append("bm25")

    merged_results: List[Dict] = []
    
    if merged:
        max_rrf = max(entry["rrf_score"] for entry in merged.values())
        for entry in merged.values():
            entry["score"] = float(entry["rrf_score"]) / max_rrf if max_rrf > 0 else 0.0
            merged_results.append(entry)

    merged_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
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


def _query_clip_window(text: str, query: str, max_len: int) -> str:
    if len(text) <= max_len or not query:
        return text[:max_len]
    normalized_text = unidecode.unidecode(text).lower()
    query_tokens = [
        token
        for token in re.findall(r"\b[\w@']+\b", normalize_text(query))
        if len(token) >= 4 and token not in {"comment", "obtenir", "avec", "dans", "pour"}
    ]
    hit_positions = [normalized_text.find(token) for token in query_tokens if normalized_text.find(token) >= 0]
    if not hit_positions:
        return text[:max_len]
    center = min(hit_positions)
    start = max(0, center - max_len // 3)
    end = min(len(text), start + max_len)
    if end - start < max_len:
        start = max(0, end - max_len)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def truncate_chunks(chunks_list: List[Dict], max_chars: int = MAX_CONTEXT_CHARS, query: str = "") -> List[Dict]:
    total = 0
    selected = []
    max_per_chunk = max(450, max_chars // 3)
    for chunk in chunks_list:
        text = chunk.get("text", "") or ""
        remaining = max_chars - total
        if remaining <= 0:
            break
        clipped_text = _query_clip_window(text, query, min(len(text), remaining, max_per_chunk))
        if len(clipped_text) < min(120, len(text)) and selected:
            break
        enriched = dict(chunk)
        enriched["text"] = clipped_text
        selected.append(enriched)
        total += len(clipped_text)
    return selected


def run_hybrid_search_debug(raw_query: str, top_k: int = TOP_K_FINAL) -> Dict[str, object]:
    if not raw_query or not raw_query.strip():
        return {
            "query": "",
            "dense_results": [],
            "bm25_results": [],
            "merged_results": [],
            "boosted_results": [],
            "guarded_results": [],
            "final_results": [],
            "abstain": True,
            "abstain_reason": "empty_query",
            "query_profile": {},
        }

    top_k = max(1, int(top_k))
    query = enhance_query(raw_query)
    logger.info("Recherche hybride pour: '%s' -> '%s'", raw_query, query)

    queries = generate_multi_queries(query)
    query_vectors = embed_queries(queries)
    retrieve_k = max(TOP_K_RETRIEVE, top_k * 4)

    if _vector_backend() == "qdrant":
        dense_results = search_qdrant(query_vectors, top_k=retrieve_k)
        bm25_results = []
    else:
        dense_results = search_faiss(query_vectors, top_k=retrieve_k)
        _, bm25_index = get_bm25_resources()
        bm25_results = search_bm25(query, bm25_index, top_k=retrieve_k)

    retrieved = merge_dense_and_bm25(dense_results, bm25_results, top_k=retrieve_k)
    retrieved = deduplicate_chunks(retrieved)
    boosted = apply_metadata_boost(retrieved, query)
    guarded, guardrail_diagnostics = apply_retrieval_guardrails(query, boosted, top_k=top_k)
    reranked = rerank_chunks(query, guarded, top_k=max(top_k * 2, top_k))
    final_ranked = apply_post_rerank_guardrails(
        reranked,
        query_profile=guardrail_diagnostics.get("query_profile", {}),
        top_k=top_k,
    )
    abstention = decide_retrieval_abstention(final_ranked, guardrail_diagnostics.get("query_profile", {}))
    final_results = [] if abstention["abstain"] else truncate_chunks(final_ranked, MAX_CONTEXT_CHARS, query=query)

    logger.info(
        "Retrieval debug | dense=%s bm25=%s merged=%s guarded=%s final=%s abstain=%s reason=%s",
        len(dense_results),
        len(bm25_results),
        len(retrieved),
        len(guarded),
        len(final_results),
        abstention["abstain"],
        abstention["reason"] or "none",
    )
    return {
        "query": query,
        "dense_results": dense_results,
        "bm25_results": bm25_results,
        "merged_results": retrieved,
        "boosted_results": boosted,
        "guarded_results": guarded,
        "final_results": final_results,
        "abstain": abstention["abstain"],
        "abstain_reason": abstention["reason"],
        "query_profile": guardrail_diagnostics.get("query_profile", {}),
        "guardrail_diagnostics": guardrail_diagnostics,
    }


def get_relevant_chunks_debug(raw_query: str, top_k: int = TOP_K_FINAL) -> Dict[str, object]:
    return run_hybrid_search_debug(raw_query, top_k=top_k)


def get_relevant_chunks(raw_query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
    debug_payload = run_hybrid_search_debug(raw_query, top_k=top_k)
    return list(debug_payload.get("final_results", []))


# Delegation layer extracted to `query_intelligence.py` to keep this module
# focused on vector access, fusion, caching, and runtime orchestration.
def build_query_profile(query: str) -> Dict[str, Any]:
    return qi_build_query_profile(query)


def score_chunk_thematic_match(chunk: Dict, query_profile: Dict[str, Any]) -> Dict[str, Any]:
    return qi_score_chunk_thematic_match(chunk, query_profile)


def apply_retrieval_guardrails(query: str, results: List[Dict], top_k: int) -> Tuple[List[Dict], Dict[str, Any]]:
    return qi_apply_retrieval_guardrails(
        query,
        results,
        top_k,
        TOP_K_RETRIEVE,
        MIN_THEMATIC_SCORE,
        MIN_SUPPORT_SCORE,
        MIN_FINAL_SUPPORT_SCORE,
        MIN_TOP_RERANK_NORMALIZED,
        TOPICAL_MISMATCH_DROP_THRESHOLD,
    )


def apply_post_rerank_guardrails(results: List[Dict], query_profile: Dict[str, Any], top_k: int) -> List[Dict]:
    return qi_apply_post_rerank_guardrails(
        results,
        query_profile=query_profile,
        top_k=top_k,
        min_thematic_score=MIN_THEMATIC_SCORE,
        min_final_support_score=MIN_FINAL_SUPPORT_SCORE,
    )


def decide_retrieval_abstention(results: List[Dict], query_profile: Dict[str, Any]) -> Dict[str, Any]:
    return qi_decide_retrieval_abstention(
        results,
        query_profile=query_profile,
        min_thematic_score=MIN_THEMATIC_SCORE,
        min_final_support_score=MIN_FINAL_SUPPORT_SCORE,
        min_top_rerank_normalized=MIN_TOP_RERANK_NORMALIZED,
    )


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
