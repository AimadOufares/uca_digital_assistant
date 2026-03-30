import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# CONFIG
# ==============================
INDEX_PATH = "data_storage/index/index.faiss"
CHUNKS_PATH = "data_storage/index/chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10  # Nombre de chunks à récupérer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT libre
GPT_MODEL = "tiiuae/falcon-7b-instruct"  # ou "mosaicml/mpt-7b-chat"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# UTILITIES
# ==============================
def normalize(text: str):
    return " ".join(text.strip().split())

# ==============================
# RAG ENGINE
# ==============================
class RAGEngine:
    def __init__(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
        # Embedding
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # FAISS
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
        self.index = faiss.read_index(index_path)

        # Chunks
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks introuvables : {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # GPT libre
        self.tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            GPT_MODEL,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if DEVICE=="cuda" else -1
        )

        logger.info(f"[INFO] {len(self.chunks)} chunks chargés, index et GPT prêts.")

    # ==============================
    # SEARCH & RERANK
    # ==============================
    def search(self, query: str, top_k=TOP_K):
        query_vec = self.embedder.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, top_k*2)  # plus de candidats pour reranking
        candidates = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                candidates.append(self.chunks[idx])

        # Reranking avec cosine similarity
        embeddings = np.array([self.embedder.encode(c["text"]) for c in candidates])
        sims = cosine_similarity(query_vec, embeddings)[0]
        ranked = sorted(
            zip(candidates, sims),
            key=lambda x: x[1],
            reverse=True
        )
        top_candidates = ranked[:top_k]

        results = []
        for c, score in top_candidates:
            results.append({
                "text": c["text"],
                "score": float(score),
                "source": c["metadata"].get("source", "unknown")
            })
        return results

    # ==============================
    # BUILD PROMPT
    # ==============================
    def build_prompt(self, query: str, contexts):
        context_text = "\n\n".join(
            [f"[Source: {c['source']}]\n{c['text']}" for c in contexts]
        )
        prompt = f"""
Tu es un assistant universitaire expert.

Règles :
- Réponds uniquement à partir du contexte fourni
- Si l'information n'existe pas, dis "Je ne sais pas"
- Sois clair, structuré et concis

CONTEXTE :
{context_text}

QUESTION :
{query}

RÉPONSE :
"""
        return prompt

    # ==============================
    # GENERATE
    # ==============================
    def generate(self, query: str):
        contexts = self.search(query)
        prompt = self.build_prompt(query, contexts)

        outputs = self.generator(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.2
        )

        answer = outputs[0]["generated_text"].strip()
        return {
            "answer": answer,
            "sources": [c["source"] for c in contexts]
        }


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    rag = RAGEngine()
    query = "Comment s'inscrire à l'université ?"
    result = rag.generate(query)

    print("\n=== RÉPONSE ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    for s in result["sources"]:
        print("-", s)