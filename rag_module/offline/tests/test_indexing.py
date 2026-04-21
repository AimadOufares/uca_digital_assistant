import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_module.offline import indexing, preparation, qdrant_indexing
from rag_module.retrieval import qdrant_search
from rag_module.shared.index_manifest import validate_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@contextmanager
def _workspace_tempdir():
    target = Path.cwd() / f".tmp_indexing_tests_{uuid.uuid4().hex}"
    target.mkdir(parents=True, exist_ok=False)
    try:
        yield str(target)
    finally:
        shutil.rmtree(target, ignore_errors=True)


class BuildIndexableTextTests(unittest.TestCase):
    def test_build_indexable_text_includes_metadata_and_normalized_text(self):
        result = indexing.build_indexable_text(
            "  Conditions   d inscription en master  ",
            {
                "section_title": " Admission ",
                "section_path": [" UCA ", " Master "],
                "document_type": " formation ",
                "etablissement": " FSSM ",
                "year": 2026,
            },
        )

        self.assertEqual(
            result,
            "\n".join(
                [
                    "Admission",
                    "UCA Master",
                    "formation",
                    "FSSM",
                    "2026",
                    "Conditions d inscription en master",
                ]
            ),
        )


class LoadChunksTests(unittest.TestCase):
    def test_load_chunks_deduplicates_and_builds_indexed_text_without_date_indexed(self):
        with _workspace_tempdir() as temp_dir:
            base = Path(temp_dir)
            _write_json(
                base / "a.json",
                {
                    "id": "chunk-a",
                    "text": "Conditions d'inscription en master avec dossier complet et calendrier detaille.",
                    "metadata": {
                        "section_title": "Admission",
                        "section_path": ["UCA", "Master"],
                        "document_type": "formation",
                        "etablissement": "FSSM",
                        "year": 2026,
                    },
                },
            )
            _write_json(
                base / "b.json",
                {
                    "id": "chunk-a",
                    "text": "Autre texte assez long pour etre retenu mais ignore a cause du meme id.",
                    "metadata": {"document_type": "formation"},
                },
            )
            _write_json(
                base / "c.json",
                {
                    "id": "chunk-c",
                    "text": "Conditions d'inscription en master avec dossier complet et calendrier detaille.",
                    "metadata": {"document_type": "formation"},
                },
            )
            _write_json(
                base / "d.json",
                {
                    "id": "chunk-d",
                    "text": "Calendrier des inscriptions administratives avec pieces a fournir et delais.",
                    "metadata": {
                        "section_title": "Calendrier",
                        "document_type": "inscription",
                        "faculty": "ENSA",
                    },
                },
            )

            with patch.object(indexing, "PROCESSED_PATH", temp_dir):
                chunks = indexing.load_chunks()

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk["id"] for chunk in chunks], ["chunk-a", "chunk-d"])
        self.assertTrue(all("indexed_text" in chunk for chunk in chunks))
        self.assertTrue(all("date_indexed" not in chunk["metadata"] for chunk in chunks))
        self.assertIn("Admission", chunks[0]["indexed_text"])
        self.assertIn("UCA Master", chunks[0]["indexed_text"])
        self.assertIn("formation", chunks[0]["indexed_text"])
        self.assertIn("FSSM", chunks[0]["indexed_text"])
        self.assertIn("2026", chunks[0]["indexed_text"])


class PreparationTests(unittest.TestCase):
    def test_verify_qdrant_indexing_prerequisites_ignores_non_indexing_dependencies(self):
        def fake_check_import(module_name: str) -> str:
            statuses = {
                "qdrant_client": "available",
                "sentence_transformers": "available",
                "trafilatura": "missing",
                "docling": "missing",
                "fasttext": "missing",
            }
            return statuses.get(module_name, "available")

        with patch.object(preparation, "_check_import", side_effect=fake_check_import):
            preparation.verify_qdrant_indexing_prerequisites()

    def test_verify_qdrant_indexing_prerequisites_fails_when_qdrant_client_missing(self):
        def fake_check_import(module_name: str) -> str:
            return "missing" if module_name == "qdrant_client" else "available"

        with patch.object(preparation, "_check_import", side_effect=fake_check_import):
            with self.assertRaisesRegex(RuntimeError, "qdrant_client"):
                preparation.verify_qdrant_indexing_prerequisites()


class BuildIndexTests(unittest.TestCase):
    def test_build_index_fails_when_embedding_model_is_unavailable(self):
        chunks = [
            {
                "id": "chunk-a",
                "text": "Texte long d'indexation pour les tests.",
                "indexed_text": "Titre\nTexte long d'indexation pour les tests.",
                "metadata": {"document_type": "formation"},
            }
        ]

        with patch.object(indexing, "verify_qdrant_indexing_prerequisites", return_value=None):
            with patch.object(indexing, "get_embedding_model", side_effect=RuntimeError("embedding indisponible")):
                with self.assertRaisesRegex(RuntimeError, "embedding indisponible"):
                    indexing.build_index(chunks)


class ManifestValidationTests(unittest.TestCase):
    def test_validate_manifest_requires_qdrant_specific_fields(self):
        manifest = {
            "model_name": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunk_count": 10,
            "index_type": "qdrant_hybrid_dense_sparse",
            "processing_policy_version": "v1",
            "vector_store": "qdrant",
        }

        with self.assertRaisesRegex(ValueError, "collection_name"):
            validate_manifest(manifest, expected_model="BAAI/bge-m3", expected_vector_store="qdrant")


class FakeQdrantClient:
    def __init__(self, existing_collection: str = ""):
        self.existing_collection = existing_collection
        self.deleted_collections = []
        self.created_collection = None
        self.payload_indexes = []
        self.upsert_batches = []
        self.aliases = {}
        self.alias_operations = []

    def get_collections(self):
        collections = []
        if self.existing_collection:
            collections.append(SimpleNamespace(name=self.existing_collection))
        collections.extend(
            SimpleNamespace(name=name)
            for name in self.aliases.values()
            if name and name != self.existing_collection
        )
        return SimpleNamespace(collections=collections)

    def collection_exists(self, collection_name: str):
        return collection_name == self.existing_collection or collection_name in self.aliases.values()

    def delete_collection(self, collection_name: str):
        self.deleted_collections.append(collection_name)

    def create_collection(self, collection_name: str, **kwargs):
        self.created_collection = {"collection_name": collection_name, **kwargs}

    def create_payload_index(self, collection_name: str, field_name: str, field_schema=None):
        self.payload_indexes.append((collection_name, field_name, field_schema))

    def upsert(self, collection_name: str, points, wait: bool = True):
        self.upsert_batches.append((collection_name, list(points), wait))

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name=alias_name, collection_name=collection_name)
                for alias_name, collection_name in self.aliases.items()
            ]
        )

    def update_collection_aliases(self, change_aliases_operations):
        self.alias_operations = list(change_aliases_operations)
        for operation in change_aliases_operations:
            if hasattr(operation, "delete_alias") and operation.delete_alias:
                self.aliases.pop(operation.delete_alias.alias_name, None)
            if hasattr(operation, "create_alias") and operation.create_alias:
                self.aliases[operation.create_alias.alias_name] = operation.create_alias.collection_name
        return True


class QdrantIndexingTests(unittest.TestCase):
    def test_build_qdrant_index_writes_expected_payload_and_manifest(self):
        fake_client = FakeQdrantClient(existing_collection="uca_chunks_2026_04_21_120000")
        chunks = [
            {
                "id": "chunk-a",
                "text": "Texte source principal.",
                "indexed_text": "Admission\nformation\nFSSM\nTexte source principal.",
                "metadata": {"processing_policy_version": "v7"},
            }
        ]

        with _workspace_tempdir() as temp_dir:
            base = Path(temp_dir)
            with patch.object(qdrant_indexing, "get_qdrant_client", return_value=fake_client):
                with patch.object(qdrant_indexing, "qdrant_collection_name", return_value="uca_chunks"):
                    manifest = qdrant_indexing.build_qdrant_index(
                        chunks=chunks,
                        model_name="active-model",
                        embedding_dim=3,
                        dense_vectors=[[0.1, 0.2, 0.3]],
                        requested_model_name="requested-model",
                        fallback_model_name="active-model",
                        target_collection_name="uca_chunks_2026_04_21_120000",
                        manifest_path=base / "candidate.manifest.json",
                        sparse_encoder_path=base / "candidate.encoder.json",
                        chunks_snapshot_path=base / "candidate.chunks.json",
                    )

        self.assertEqual(fake_client.deleted_collections, ["uca_chunks_2026_04_21_120000"])
        self.assertEqual(fake_client.created_collection["collection_name"], "uca_chunks_2026_04_21_120000")
        self.assertIn(("uca_chunks_2026_04_21_120000", "metadata.section_title", unittest.mock.ANY), fake_client.payload_indexes)
        self.assertIn(("uca_chunks_2026_04_21_120000", "metadata.language", unittest.mock.ANY), fake_client.payload_indexes)
        self.assertTrue(fake_client.upsert_batches)
        payload = fake_client.upsert_batches[0][1][0].payload
        self.assertEqual(payload["text"], "Texte source principal.")
        self.assertEqual(payload["indexed_text"], "Admission\nformation\nFSSM\nTexte source principal.")
        self.assertEqual(manifest["requested_model_name"], "requested-model")
        self.assertEqual(manifest["fallback_model_name"], "active-model")
        self.assertEqual(manifest["vector_store"], "qdrant")
        self.assertEqual(manifest["dense_vector_name"], qdrant_indexing.DENSE_VECTOR_NAME)
        self.assertEqual(manifest["sparse_vector_name"], qdrant_indexing.SPARSE_VECTOR_NAME)
        self.assertTrue(manifest["candidate"])

    def test_build_qdrant_index_omits_fallback_model_when_not_used(self):
        fake_client = FakeQdrantClient()
        chunks = [
            {
                "id": "chunk-a",
                "text": "Texte source principal.",
                "indexed_text": "Texte source principal.",
                "metadata": {"processing_policy_version": "v7"},
            }
        ]

        with _workspace_tempdir() as temp_dir:
            base = Path(temp_dir)
            with patch.object(qdrant_indexing, "get_qdrant_client", return_value=fake_client):
                with patch.object(qdrant_indexing, "qdrant_collection_name", return_value="uca_chunks"):
                    manifest = qdrant_indexing.build_qdrant_index(
                        chunks=chunks,
                        model_name="active-model",
                        embedding_dim=3,
                        dense_vectors=[[0.1, 0.2, 0.3]],
                        requested_model_name="active-model",
                        fallback_model_name="active-model",
                        target_collection_name="uca_chunks_2026_04_21_120001",
                        manifest_path=base / "candidate.manifest.json",
                        sparse_encoder_path=base / "candidate.encoder.json",
                        chunks_snapshot_path=base / "candidate.chunks.json",
                    )

        self.assertEqual(manifest["requested_model_name"], "active-model")
        self.assertNotIn("fallback_model_name", manifest)

    def test_publish_qdrant_index_switches_alias_without_touching_candidate_collection(self):
        fake_client = FakeQdrantClient(existing_collection="uca_chunks_2026_04_21_120000")
        fake_client.aliases = {
            "uca_chunks_current": "uca_chunks_2026_04_20_235959",
        }
        manifest = {
            "model_name": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunk_count": 120,
            "index_type": "qdrant_hybrid_dense_sparse",
            "processing_policy_version": "v7",
            "vector_store": "qdrant",
            "collection_name": "uca_chunks_2026_04_21_120000",
            "dense_vector_name": "dense",
            "sparse_vector_name": "lexical",
            "sparse_encoder_path": "",
        }

        with _workspace_tempdir() as temp_dir:
            base = Path(temp_dir)
            encoder = base / "candidate.encoder.json"
            chunks_path = base / "candidate.chunks.json"
            manifest_path = base / "candidate.manifest.json"
            encoder.write_text("{}", encoding="utf-8")
            chunks_path.write_text("[]", encoding="utf-8")
            manifest["sparse_encoder_path"] = str(encoder)
            manifest["chunks_snapshot_path"] = str(chunks_path)
            manifest["manifest_path"] = str(manifest_path)

            with patch.object(qdrant_indexing, "get_qdrant_client", return_value=fake_client):
                with patch.object(qdrant_indexing, "QDRANT_SPARSE_ENCODER_PATH", base / "active.encoder.json"):
                    with patch.object(qdrant_indexing, "QDRANT_CHUNKS_PATH", base / "active.chunks.json"):
                        with patch.object(qdrant_indexing, "QDRANT_MANIFEST_PATH", base / "active.manifest.json"):
                            with patch.object(qdrant_indexing, "qdrant_active_alias_name", return_value="uca_chunks_current"):
                                with patch.object(qdrant_indexing, "qdrant_previous_alias_name", return_value="uca_chunks_previous"):
                                    published = qdrant_indexing.publish_qdrant_index(candidate_manifest=manifest)

        self.assertEqual(fake_client.aliases["uca_chunks_current"], "uca_chunks_2026_04_21_120000")
        self.assertEqual(fake_client.aliases["uca_chunks_previous"], "uca_chunks_2026_04_20_235959")
        self.assertEqual(published["published_collection_name"], "uca_chunks_2026_04_21_120000")
        self.assertEqual(published["collection_name"], "uca_chunks_current")
        self.assertEqual(published["previous_collection_name"], "uca_chunks_2026_04_20_235959")


class QdrantReadinessTests(unittest.TestCase):
    def test_qdrant_index_ready_returns_false_for_stale_manifest(self):
        with patch.object(qdrant_search, "load_manifest", return_value={"model_name": "BAAI/bge-m3", "embedding_dim": 1024, "chunk_count": 3}):
            with patch.object(qdrant_search, "get_active_model_name", return_value="BAAI/bge-m3"):
                self.assertFalse(qdrant_search.qdrant_index_ready())

    def test_qdrant_index_ready_returns_false_when_sparse_encoder_is_missing(self):
        manifest = {
            "model_name": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunk_count": 3,
            "index_type": "qdrant_hybrid_dense_sparse",
            "processing_policy_version": "v7",
            "vector_store": "qdrant",
            "collection_name": "uca_chunks",
            "dense_vector_name": "dense",
            "sparse_vector_name": "lexical",
            "sparse_encoder_path": "C:/tmp/missing-encoder.json",
        }
        fake_client = FakeQdrantClient(existing_collection="uca_chunks")
        fake_client.aliases = {"uca_chunks_current": "uca_chunks"}

        with patch.object(qdrant_search, "load_manifest", return_value=manifest):
            with patch.object(qdrant_search, "get_active_model_name", return_value="BAAI/bge-m3"):
                with patch.object(qdrant_search, "get_qdrant_client", return_value=fake_client):
                    self.assertFalse(qdrant_search.qdrant_index_ready())


if __name__ == "__main__":
    unittest.main()
