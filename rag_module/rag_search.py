# rag_module/rag_search.py
import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from functools import lru_cache

import faiss
import unidecode
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# ========================= CONFIG =========================
INDEX_PATH = "data_storage/index/index.faiss"
CHUNKS_PATH = "data_storage/index/chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20      # On récupère plus pour reranker
TOP_K_FINAL = 5
MAX_CONTEXT_CHARS = 2500

USE_RERANK = True
USE_SPELLCHECK = False   # Désactivé par défaut (trop imprécis en FR)
USE_MULTI_QUERY = True

# ========================= LOGGING =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= LAZY LOADING DES MODÈLES =========================
_embedding_model = None
_reranker = None
_index = None
_chunks = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Chargement du modèle embedding: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        _embedding_model.max_seq_length = 512
    return _embedding_model


def get_reranker():
    global _reranker
    if _reranker is None and USE_RERANK:
        logger.info(f"Chargement du reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    return _reranker


def get_faiss_index_and_chunks():
    global _index, _chunks
    if _index is None or _chunks is None:
        index_file = Path(INDEX_PATH)
        chunks_file = Path(CHUNKS_PATH)

        if not index_file.exists() or not chunks_file.exists():
            logger.error("Index FAISS ou chunks.json introuvable ! Exécute d'abord indexing.py")
            raise FileNotFoundError("Index FAISS non trouvé")

        logger.info("Chargement de l'index FAISS...")
        _index = faiss.read_index(str(index_file))

        logger.info("Chargement des chunks...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            _chunks = json.load(f)

    return _index, _chunks


# ========================= QUERY PREPROCESSING =========================
def preprocess_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = unidecode.unidecode(query)
    return query


def correct_query(query: str) -> str:
    """Spell checking léger - désactivé par défaut car imprécis en français"""
    if not USE_SPELLCHECK:
        return query
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language="fr")
        words = query.split()
        corrected = [spell.correction(w) or w for w in words]
        return " ".join(corrected)
    except:
        return query


def enhance_query(query: str) -> str:
    query = preprocess_query(query)
    query = correct_query(query)
    return query


# ========================= MULTI-QUERY (meilleure version) =========================
def generate_multi_queries(query: str) -> List[str]:
    if not USE_MULTI_QUERY:
        return [query]

    base = enhance_query(query)
    variations = [
        base,
        base + " explication détaillée",
        base + " informations importantes",
        "comment " + base if not base.startswith(("comment", "comment faire")) else base,
    ]
    return list(dict.fromkeys(variations))  # garde l'ordre + supprime doublons


# ========================= EMBEDDING =========================
@lru_cache(maxsize=500)
def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True)


def embed_queries(queries: List[str]) -> np.ndarray:
    return np.array([embed_text(q) for q in queries])


# ========================= SEARCH =========================
def search_faiss(query_vectors: np.ndarray, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
    index, chunks = get_faiss_index_and_chunks()
    D, I = index.search(query_vectors, top_k)

    results = []
    for i, idx_list in enumerate(I):
        for rank, idx in enumerate(idx_list):
            if idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk["score"] = float(D[i][rank])   # similarité
                chunk["query_source"] = "multi" if len(query_vectors) > 1 else "single"
                results.append(chunk)

    return results


def deduplicate_chunks(chunks_list: List[Dict]) -> List[Dict]:
    """Dédoublonnage basé sur chunk_hash (beaucoup plus fiable)"""
    seen = set()
    unique = []
    for c in chunks_list:
        chunk_id = c.get("id") or c.get("metadata", {}).get("chunk_hash")
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(c)
    return unique


# ========================= RERANKING =========================
def rerank_chunks(query: str, chunks_list: List[Dict]) -> List[Dict]:
    if not USE_RERANK or not chunks_list:
        return chunks_list

    reranker = get_reranker()
    if reranker is None:
        return chunks_list

    pairs = [(query, c.get("text", "")) for c in chunks_list]
    scores = reranker.predict(pairs)

    # Tri par score de reranking (plus élevé = mieux)
    ranked = sorted(zip(chunks_list, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:TOP_K_FINAL]]


# ========================= CONTEXT BUILDER =========================
def truncate_chunks(chunks_list: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> List[Dict]:
    """Limite la taille totale du contexte envoyé au LLM"""
    total = 0
    selected = []
    for c in chunks_list:
        text = c.get("text", "")
        if total + len(text) > max_chars:
            break
        selected.append(c)
        total += len(text)
    return selected


# ========================= MAIN SEARCH FUNCTION =========================
def get_relevant_chunks(raw_query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not raw_query or not raw_query.strip():
        return []

    query = enhance_query(raw_query)
    logger.info(f"🔎 Recherche pour : '{raw_query}' → '{query}'")

    # Multi-query
    queries = generate_multi_queries(query)
    query_vectors = embed_queries(queries)

    # Recherche vectorielle
    retrieved = search_faiss(query_vectors, top_k=TOP_K_RETRIEVE)

    # Post-traitement
    retrieved = deduplicate_chunks(retrieved)
    retrieved = rerank_chunks(query, retrieved)
    retrieved = truncate_chunks(retrieved, MAX_CONTEXT_CHARS)

    logger.info(f"✅ {len(retrieved)} chunks pertinents retournés après reranking")
    return retrieved


# ========================= TEST =========================
if __name__ == "__main__":
    test_queries = [
        "Comment s'inscrire à Semlalia ?",
        "Quelles sont les conditions d'admission à la faculté Semlalia ?",
        "Procédure inscription université Cadi Ayyad"
    ]

    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {q}")
        results = get_relevant_chunks(q, top_k=5)

        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "unknown")
            print(f"\n[{i}] Score: {score:.4f} | Source: {Path(source).name}")
            print(f"    {r['text'][:280]}..." if len(r['text']) > 280 else r['text'])