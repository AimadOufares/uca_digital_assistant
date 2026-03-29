import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime

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
BATCH_SIZE = 32   # stable

os.makedirs(INDEX_PATH, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# UTILS
# ==============================
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def normalize(text):
    return " ".join(text.strip().split())

# ==============================
# LOAD CACHE
# ==============================
def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except:
            logger.warning("Cache corrompu → reset")
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

# ==============================
# LOAD CHUNKS
# ==============================
def load_chunks():
    files = list(Path(PROCESSED_PATH).glob("*.json"))
    chunks = []
    seen = set()

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = normalize(data.get("text", ""))

            if len(text) < 30:
                continue

            h = get_hash(text)
            if h in seen:
                continue
            seen.add(h)

            chunks.append({
                "text": text,
                "metadata": {
                    "source": file.name,
                    "hash": h,
                    "date": datetime.now().isoformat()
                }
            })
        except Exception as e:
            logger.warning(f"Erreur fichier {file}: {e}")

    logger.info(f"{len(chunks)} chunks chargés")
    return chunks

# ==============================
# EMBEDDING (STABLE)
# ==============================
def embed(texts, cache):
    model = SentenceTransformer(MODEL_NAME)

    embeddings = []
    new_cache = False

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        batch_emb = []
        to_compute = []
        idx_map = []

        for j, text in enumerate(batch):
            h = get_hash(text)
            if h in cache:
                batch_emb.append(cache[h])
            else:
                batch_emb.append(None)
                to_compute.append(text)
                idx_map.append(j)

        if to_compute:
            computed = model.encode(
                to_compute,
                normalize_embeddings=True
            )

            for k, emb in enumerate(computed):
                j = idx_map[k]
                h = get_hash(batch[j])
                emb_list = emb.tolist()

                batch_emb[j] = emb_list
                cache[h] = emb_list
                new_cache = True

        embeddings.extend(batch_emb)

        logger.info(f"Progress: {i+len(batch)}/{len(texts)}")

    if new_cache:
        save_cache(cache)

    return embeddings

# ==============================
# BUILD INDEX
# ==============================
def build_index(chunks):
    texts = [c["text"] for c in chunks]

    cache = load_cache()
    embeddings = embed(texts, cache)

    vectors = np.array(embeddings).astype("float32")
    dim = vectors.shape[1]

    # ⚡ INDEX SIMPLE & STABLE
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200

    index.add(vectors)

    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))

    with open(os.path.join(INDEX_PATH, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    logger.info("✅ INDEX CRÉÉ AVEC SUCCÈS")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    chunks = load_chunks()

    if not chunks:
        logger.error("Aucun chunk trouvé !")
    else:
        build_index(chunks)