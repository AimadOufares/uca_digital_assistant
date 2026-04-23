import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ..contracts import IndexBuildResult
from ..shared.index_manifest import build_manifest, save_manifest
from ..shared.runtime import RuntimeSettings, get_runtime_settings
from .storage import DocumentStorage


def _utc_build_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _chunk_processing_policy(chunks: List[Dict]) -> str:
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        candidate = metadata.get("processing_policy_version")
        if candidate:
            return str(candidate)
    return "unknown"


def _stable_point_id(chunk: Dict) -> str:
    chunk_id = str(chunk.get("id") or "")
    if chunk_id:
        return hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:32]
    payload = json.dumps(chunk, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:32]


class VectorStoreAdapter(ABC):
    def __init__(self, settings: RuntimeSettings | None = None, storage: DocumentStorage | None = None):
        self.settings = settings or get_runtime_settings()
        self.storage = storage or DocumentStorage(self.settings)

    @abstractmethod
    def build_index(self, chunks: List[Dict], build_id: str | None = None, publish: bool = False) -> IndexBuildResult:
        raise NotImplementedError

    @abstractmethod
    def publish(self, build_id: str) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> Dict:
        raise NotImplementedError


class FaissVectorStoreAdapter(VectorStoreAdapter):
    def build_index(self, chunks: List[Dict], build_id: str | None = None, publish: bool = False) -> IndexBuildResult:
        if not chunks:
            raise RuntimeError("Aucun chunk disponible pour l'indexation.")

        from ..offline.indexing import (
            HNSW_EF_CONSTRUCTION,
            HNSW_EF_SEARCH,
            HNSW_M,
            build_bm25_corpus,
            embed,
            get_active_model_name,
            get_model_name,
            load_cache,
        )
        build_id = build_id or _utc_build_id()
        paths = self.storage.faiss_build_paths(build_id)

        cache = load_cache()
        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = embed(texts, cache)
        vectors = np.array(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise RuntimeError("Aucun vecteur genere pour construire l'index.")

        dim = int(vectors.shape[1])
        index = faiss.IndexHNSWFlat(dim, HNSW_M)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(vectors)

        faiss.write_index(index, str(paths["index_file"]))
        with paths["chunks_file"].open("w", encoding="utf-8") as handle:
            json.dump(chunks, handle, ensure_ascii=False, indent=2)

        bm25_corpus = build_bm25_corpus(chunks)
        with paths["bm25_file"].open("w", encoding="utf-8") as handle:
            json.dump(bm25_corpus, handle, ensure_ascii=False, indent=2)

        manifest = build_manifest(
            model_name=get_active_model_name(),
            dim=dim,
            chunk_count=len(chunks),
            policy_version=_chunk_processing_policy(chunks),
            index_type="faiss_hnsw_dense_plus_bm25",
        )
        manifest["requested_model_name"] = get_model_name()
        manifest["build_id"] = build_id
        manifest["backend"] = "faiss"
        save_manifest(str(paths["manifest_file"]), manifest)

        if publish:
            self.publish(build_id)

        return IndexBuildResult(
            build_id=build_id,
            backend="faiss",
            chunk_count=len(chunks),
            manifest_path=str(paths["manifest_file"]),
            published=publish,
        )

    def publish(self, build_id: str) -> Dict:
        return self.storage.publish_faiss_build(build_id)

    def health(self) -> Dict:
        paths = self.storage.resolve_active_faiss_paths()
        exists = paths["index_file"].exists() and paths["chunks_file"].exists()
        return {
            "backend": "faiss",
            "ok": exists,
            "active_index_present": exists,
            "paths": {key: str(value) for key, value in paths.items()},
        }


class QdrantVectorStoreAdapter(VectorStoreAdapter):
    def _client(self) -> QdrantClient:
        kwargs = {"url": self.settings.rag_qdrant_url}
        if self.settings.rag_qdrant_api_key:
            kwargs["api_key"] = self.settings.rag_qdrant_api_key
        return QdrantClient(**kwargs)

    def _collection_name(self, build_id: str) -> str:
        return f"{self.settings.rag_qdrant_collection_prefix}_{build_id}"

    def build_index(self, chunks: List[Dict], build_id: str | None = None, publish: bool = False) -> IndexBuildResult:
        if not self.settings.rag_qdrant_url:
            raise RuntimeError("RAG_QDRANT_URL est requis pour l'indexation Qdrant.")
        if not chunks:
            raise RuntimeError("Aucun chunk disponible pour l'indexation.")

        from ..offline.indexing import embed, get_active_model_name, get_model_name, load_cache

        build_id = build_id or _utc_build_id()
        collection_name = self._collection_name(build_id)
        build_dir = self.storage.build_dir(build_id) / "qdrant"
        build_dir.mkdir(parents=True, exist_ok=True)

        cache = load_cache()
        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = embed(texts, cache)
        if not embeddings:
            raise RuntimeError("Aucun embedding genere pour Qdrant.")

        dim = len(embeddings[0])
        client = self._client()
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE),
        )
        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=[
                {
                    "id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}) or {},
                }
                for chunk in chunks
            ],
            ids=[_stable_point_id(chunk) for chunk in chunks],
            wait=True,
        )

        manifest = build_manifest(
            model_name=get_active_model_name(),
            dim=dim,
            chunk_count=len(chunks),
            policy_version=_chunk_processing_policy(chunks),
            index_type="qdrant_dense_cosine",
        )
        manifest["requested_model_name"] = get_model_name()
        manifest["build_id"] = build_id
        manifest["backend"] = "qdrant"
        manifest["collection_name"] = collection_name
        manifest_path = build_dir / "index_manifest.json"
        save_manifest(str(manifest_path), manifest)
        with (build_dir / "chunks.json").open("w", encoding="utf-8") as handle:
            json.dump(chunks, handle, ensure_ascii=False, indent=2)

        if publish:
            self.publish(build_id)

        return IndexBuildResult(
            build_id=build_id,
            backend="qdrant",
            chunk_count=len(chunks),
            manifest_path=str(manifest_path),
            published=publish,
        )

    def publish(self, build_id: str) -> Dict:
        collection_name = self._collection_name(build_id)
        client = self._client()
        active_alias = self.settings.rag_active_index_name
        operations = [
            qdrant_models.DeleteAliasOperation(
                delete_alias=qdrant_models.DeleteAlias(alias_name=active_alias)
            ),
            qdrant_models.CreateAliasOperation(
                create_alias=qdrant_models.CreateAlias(
                    collection_name=collection_name,
                    alias_name=active_alias,
                )
            ),
        ]
        try:
            client.update_collection_aliases(operations)
        except Exception:
            client.update_collection_aliases(operations[1:])
        return self.storage.publish_qdrant_build(build_id, collection_name)

    def health(self) -> Dict:
        if not self.settings.rag_qdrant_url:
            return {
                "backend": "qdrant",
                "ok": False,
                "active_index_present": False,
                "reason": "missing_qdrant_url",
            }
        try:
            client = self._client()
            aliases = client.get_aliases()
            alias_names = {item.alias_name for item in getattr(aliases, "aliases", [])}
            alias_present = self.settings.rag_active_index_name in alias_names
            return {
                "backend": "qdrant",
                "ok": True,
                "active_index_present": alias_present,
                "alias": self.settings.rag_active_index_name,
                "url": self.settings.rag_qdrant_url,
            }
        except Exception as exc:
            return {
                "backend": "qdrant",
                "ok": False,
                "active_index_present": False,
                "reason": str(exc),
                "url": self.settings.rag_qdrant_url,
            }


def get_vector_store_adapter(
    settings: RuntimeSettings | None = None,
    storage: DocumentStorage | None = None,
) -> VectorStoreAdapter:
    resolved_settings = settings or get_runtime_settings()
    if resolved_settings.rag_vector_backend == "qdrant":
        return QdrantVectorStoreAdapter(resolved_settings, storage)
    return FaissVectorStoreAdapter(resolved_settings, storage)
