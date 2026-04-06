import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================
PROCESSED_PATH = "data_storage/processed"
INDEX_PATH = "data_storage/index"
CACHE_PATH = "data_storage/cache/embeddings_cache.json"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

os.makedirs(INDEX_PATH, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# UTILS
# ==============================
def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


# ==============================
# LOAD CACHE
# ==============================
def load_cache() -> Dict[str, List[float]]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("Cache corrompu -> reset")
            return {}
    return {}


def save_cache(cache: Dict[str, List[float]]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ==============================
# LOAD CHUNKS
# ==============================
def load_chunks() -> List[Dict]:
    files = sorted(Path(PROCESSED_PATH).glob("*.json"))
    chunks: List[Dict] = []
    seen_ids = set()
    seen_text_hashes = set()

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = normalize(data.get("text", ""))
            if len(text) < 30:
                continue

            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            chunk_id = (
                data.get("id")
                or metadata.get("chunk_hash")
                or metadata.get("hash")
                or get_hash(text)
            )
            if chunk_id in seen_ids:
                continue
            text_hash = get_hash(text.lower())
            if text_hash in seen_text_hashes:
                continue
            seen_ids.add(chunk_id)
            seen_text_hashes.add(text_hash)

            source_path = metadata.get("source", str(file))
            source_name = metadata.get("file_name") or Path(source_path).name

            merged_metadata = dict(metadata)
            merged_metadata.update(
                {
                    "source": source_path,
                    "file_name": source_name,
                    "hash": chunk_id,
                    "date_indexed": datetime.now().isoformat(),
                }
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": merged_metadata,
                }
            )
        except Exception as exc:
            logger.warning("Erreur fichier %s: %s", file, exc)

    logger.info("%s chunks charges", len(chunks))
    return chunks


# ==============================
# EMBEDDING
# ==============================
def embed(texts: List[str], cache: Dict[str, List[float]]) -> List[List[float]]:
    model = SentenceTransformer(MODEL_NAME)

    embeddings: List[List[float]] = []
    new_cache = False

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        batch_emb: List[List[float]] = []
        to_compute: List[str] = []
        idx_map: List[int] = []

        for j, text in enumerate(batch):
            h = get_hash(text)
            cached = cache.get(h)
            if cached is not None:
                batch_emb.append(cached)
            else:
                batch_emb.append([])
                to_compute.append(text)
                idx_map.append(j)

        if to_compute:
            computed = model.encode(to_compute, normalize_embeddings=True)
            for k, emb in enumerate(computed):
                j = idx_map[k]
                h = get_hash(batch[j])
                emb_list = emb.tolist()
                batch_emb[j] = emb_list
                cache[h] = emb_list
                new_cache = True

        embeddings.extend(batch_emb)
        logger.info("Progress: %s/%s", i + len(batch), len(texts))

    if new_cache:
        save_cache(cache)

    return embeddings


# ==============================
# BUILD INDEX
# ==============================
def build_index(chunks: List[Dict]) -> None:
    texts = [c["text"] for c in chunks]
    if not texts:
        raise RuntimeError("Aucun texte disponible pour l'indexation.")

    cache = load_cache()
    embeddings = embed(texts, cache)

    vectors = np.array(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise RuntimeError("Aucun vecteur genere pour construire l'index.")

    dim = vectors.shape[1]

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(vectors)

    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))
    with open(os.path.join(INDEX_PATH, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("INDEX CREE AVEC SUCCES")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    chunks_data = load_chunks()
    if not chunks_data:
        logger.error("Aucun chunk trouve !")
    else:
        build_index(chunks_data)
