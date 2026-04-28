import json
import shutil
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from rag_module.adapters.vector_store import FaissVectorStoreAdapter
from rag_module.adapters.storage import DocumentStorage
from rag_module.contracts import AnswerResult
from rag_module.offline.indexing import load_chunks
from rag_module.offline.ingestion_utils import (
    _download_document,
    compute_download_quality,
    decide_document,
    default_seeds,
)
from rag_module.offline.processing import clean_text
from rag_module.retrieval.rag_search import get_candidate_reranker_names, get_reranker
from rag_module.shared.metadata_policy import prepare_chunk_metadata
from rag_module.services.offline import run_indexing


class ChatApiTests(APITestCase):
    @patch("api_app.views.answer_question")
    def test_chat_endpoint_returns_answer_payload(self, mocked_answer_question):
        mocked_answer_question.return_value = AnswerResult(
            answer="Reponse test",
            sources=[],
            confidence="moyen",
            backend="faiss",
            retrieval_meta={},
        )

        response = self.client.post(reverse("api-chat"), {"message": "Bonjour"}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"answer": "Reponse test"})
        mocked_answer_question.assert_called_once()

    def test_chat_endpoint_rejects_empty_message(self):
        response = self.client.post(reverse("api-chat"), {"message": ""}, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("detail", response.json())


class HealthApiTests(APITestCase):
    def test_live_health_endpoint_returns_ok(self):
        response = self.client.get(reverse("api-health-live"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.json()["ok"])

    @patch("api_app.views.build_ready_health")
    def test_ready_health_endpoint_returns_503_when_not_ready(self, mocked_health):
        mocked_health.return_value = {
            "ready": False,
            "ok": False,
            "database": {"ok": True},
            "vector_store": {"ok": False, "active_index_present": False},
            "llm": {"state": "degraded"},
        }

        response = self.client.get(reverse("api-health-ready"))

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertFalse(response.json()["ready"])


class IngestionPolicyTests(APITestCase):
    def test_fast_default_seeds_are_subset_of_extended(self):
        fast_seeds = set(default_seeds("fast"))
        extended_seeds = set(default_seeds("extended"))

        self.assertTrue(fast_seeds)
        self.assertTrue(fast_seeds.issubset(extended_seeds))

    def test_high_value_student_page_routes_to_main(self):
        decision = decide_document(
            url="https://www.uca.ma/fr/inscription-administrative",
            quality_score=82,
            keyword_hits=["inscription", "scolarite"],
            depth=0,
            extension=".html",
            mode="fast",
        )

        self.assertEqual(decision.corpus_target, "main")
        self.assertEqual(decision.source_priority, "A")

    def test_secondary_research_page_routes_to_archive(self):
        decision = decide_document(
            url="https://www.uca.ma/fr/recherche/laboratoire-innovation",
            quality_score=70,
            keyword_hits=[],
            depth=1,
            extension=".html",
            mode="extended",
        )

        self.assertEqual(decision.corpus_target, "archive")
        self.assertIn(decision.source_priority, {"B", "C"})

    def test_low_quality_page_is_rejected(self):
        decision = decide_document(
            url="https://www.uca.ma/fr/navigation",
            quality_score=20,
            keyword_hits=[],
            depth=0,
            extension=".html",
            mode="fast",
        )

        self.assertEqual(decision.corpus_target, "reject")


class ProcessingAndIndexingTests(APITestCase):
    def test_clean_text_preserves_drive_urls(self):
        text = "Plateforme officielle\nhttps://pucastaff.uca.ma/\nDescription"

        cleaned = clean_text(text, corpus="drive")

        self.assertIn("https://pucastaff.uca.ma/", cleaned)

    def test_clean_text_repairs_common_mojibake_sequences(self):
        text = "UniversitÃ© Cadi Ayyad et prÃ©inscription en ligne"

        cleaned = clean_text(text, corpus="main")

        self.assertIn("Université", cleaned)
        self.assertIn("préinscription", cleaned)

    def test_default_seeds_include_student_digital_services(self):
        seeds = set(default_seeds("fast"))

        self.assertIn("https://ucaplat.uca.ma/", seeds)
        self.assertIn("https://cip.uca.ma/", seeds)
        self.assertIn("https://diplomes.uca.ma/", seeds)

    def test_static_html_quality_stays_in_static_mode(self):
        html = (
            b"<html><body><main><h1>Attestation UC@Student</h1><p>Guide pour demander une attestation."
            b"</p><p>Connectez-vous puis choisissez le service.</p></main></body></html>"
        )

        quality = compute_download_quality("https://ucastudent.uca.ma/attestation", 0, html, "text/html", ".html")

        self.assertFalse(quality["js_dependent"])
        self.assertEqual(quality["page_kind"], "guide")
        self.assertIn("attestation", quality["intent"])

    def test_sparse_spa_html_is_detected_as_js_dependent(self):
        html = (
            b'<html><body><div id="root"></div><script src="/app.js"></script>'
            b"<script>window.__NEXT_DATA__={};</script><script src=\"/chunk.js\"></script>"
            b"<script src=\"/vendor.js\"></script><script src=\"/runtime.js\"></script>"
            b"<script src=\"/boot.js\"></script></body></html>"
        )

        quality = compute_download_quality("https://ucaplat.uca.ma/dashboard", 0, html, "text/html", ".html")

        self.assertTrue(quality["js_dependent"])

    @patch("rag_module.offline.ingestion_utils.render_url")
    def test_download_document_uses_playwright_fallback_when_needed(self, mocked_render):
        mocked_response = MagicMock()
        mocked_response.status_code = 200
        mocked_response.content = (
            b'<html><body><div id="root"></div><script src="/app.js"></script>'
            b"<script>window.__NEXT_DATA__={};</script><script src=\"/chunk.js\"></script>"
            b"<script src=\"/vendor.js\"></script><script src=\"/runtime.js\"></script>"
            b"<script src=\"/boot.js\"></script></body></html>"
        )
        mocked_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mocked_render.return_value = MagicMock(
            ok=True,
            html="<html><body><main><h1>UCAPLAT</h1><p>Cours en ligne et devoir.</p></main></body></html>",
            error="",
            final_url="https://ucaplat.uca.ma/dashboard",
            selector_used="main",
        )

        with patch("rag_module.offline.ingestion_utils.requests.get", return_value=mocked_response), patch(
            "rag_module.offline.ingestion_utils.RUNTIME"
        ) as mocked_runtime:
            mocked_runtime.rag_js_fallback_enabled = True
            mocked_runtime.rag_js_max_pages_per_run = 5
            mocked_runtime.rag_js_render_timeout_ms = 12000
            mocked_runtime.rag_js_allowed_domains = ["ucaplat.uca.ma"]
            download = _download_document(
                "https://ucaplat.uca.ma/dashboard",
                0,
                js_usage={"count": 0},
                js_lock=Lock(),
            )

        self.assertIsNotNone(download)
        self.assertEqual(download["render_mode"], "playwright")
        self.assertTrue(download["render_success"])
        self.assertIn("cours", download["quality"]["intent"])

    def test_prepare_chunk_metadata_adds_page_kind_and_intent(self):
        chunk = {
            "text": "Guide UCAPLAT pour suivre les cours et deposer les devoirs en ligne.",
            "metadata": {
                "language": "fr",
                "file_type": "html",
                "url": "https://ucaplat.uca.ma/guide",
                "last_modified": "Tue, 01 Apr 2025 10:00:00 GMT",
            },
        }

        enriched = prepare_chunk_metadata(chunk, "https://ucaplat.uca.ma/guide")

        self.assertIsNotNone(enriched)
        self.assertEqual(enriched["metadata"]["page_kind"], "guide")
        self.assertIn("cours", enriched["metadata"]["intent"])
        self.assertEqual(enriched["metadata"]["service_type"], "pedagogie_numerique")

    def test_load_chunks_published_merges_main_and_drive(self):
        root = Path.cwd() / ".tmp_test_chunks_case"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        try:
            main_dir = root / "main"
            drive_dir = root / "drive"
            archive_dir = root / "archive"
            main_dir.mkdir()
            drive_dir.mkdir()
            archive_dir.mkdir()

            (main_dir / "main_chunk.json").write_text(
                json.dumps(
                    {
                        "text": (
                            "Informations d inscription administrative pour les etudiants de l Universite Cadi Ayyad "
                            "avec calendrier, procedure et pieces a fournir."
                        ),
                        "metadata": {
                            "chunk_hash": "main-1",
                            "source": "main-source",
                            "corpus": "main",
                            "page_kind": "procedure",
                            "intent": ["reinscription"],
                            "chunk_relevance_score": 2,
                            "service_type": "scolarite",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (drive_dir / "drive_chunk.json").write_text(
                json.dumps(
                    {
                        "text": (
                            "PUCAStaff plateforme officielle pour la gestion administrative avec guide de connexion "
                            "et demarches numeriques pour les utilisateurs concernes."
                        ),
                        "metadata": {
                            "chunk_hash": "drive-1",
                            "source": "drive-source",
                            "corpus": "drive",
                            "page_kind": "guide",
                            "intent": ["connexion"],
                            "chunk_relevance_score": 2,
                            "service_type": "digital_service",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("rag_module.offline.indexing.RUNTIME") as mocked_runtime:
                mocked_runtime.rag_processed_main_dir = main_dir
                mocked_runtime.rag_processed_drive_dir = drive_dir
                mocked_runtime.rag_processed_archive_dir = archive_dir
                mocked_runtime.rag_index_published_corpora = ["main", "drive"]

                chunks = load_chunks(corpus="published")
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(len(chunks), 2)
        self.assertEqual({chunk["metadata"]["corpus"] for chunk in chunks}, {"main", "drive"})

    def test_load_chunks_filters_generic_landing_and_enriches_retrieval_metadata(self):
        root = Path.cwd() / ".tmp_test_indexing_filter_case"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        try:
            main_dir = root / "main"
            drive_dir = root / "drive"
            archive_dir = root / "archive"
            main_dir.mkdir()
            drive_dir.mkdir()
            archive_dir.mkdir()

            (main_dir / "landing.json").write_text(
                json.dumps(
                    {
                        "text": "Bienvenue sur la plateforme UCA. Decouvrez nos actualites et notre univers.",
                        "metadata": {
                            "chunk_hash": "landing-1",
                            "source": "landing-source",
                            "page_kind": "landing",
                            "intent": [],
                            "chunk_relevance_score": 0,
                            "service_name": "uca",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (main_dir / "student_service.json").write_text(
                json.dumps(
                    {
                        "text": (
                            "Guide UC@Student pour demander une attestation de scolarite et consulter les notes. "
                            "Connectez-vous, ouvrez le service numerique puis telechargez votre document."
                        ),
                        "metadata": {
                            "chunk_hash": "student-1",
                            "source": "student-source",
                            "page_kind": "guide",
                            "intent": ["attestation", "notes"],
                            "service_name": "ucastudent",
                            "service_type": "digital_service",
                            "document_type": "digital_service",
                            "document_category": "digital_service",
                            "source_priority": "A",
                            "chunk_relevance_score": 3,
                            "freshness_score": 0.8,
                            "main_actions": ["demander attestation", "consulter notes"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("rag_module.offline.indexing.RUNTIME") as mocked_runtime:
                mocked_runtime.rag_processed_main_dir = main_dir
                mocked_runtime.rag_processed_drive_dir = drive_dir
                mocked_runtime.rag_processed_archive_dir = archive_dir
                mocked_runtime.rag_index_published_corpora = ["main"]

                chunks = load_chunks(corpus="main")
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(len(chunks), 1)
        metadata = chunks[0]["metadata"]
        self.assertTrue(metadata["is_actionable"])
        self.assertGreater(metadata["student_relevance_score"], 0.5)
        self.assertIn("attestation", metadata["retrieval_keywords"])
        self.assertIn("ucastudent", metadata["retrieval_haystack"].lower())

    def test_faiss_manifest_exposes_student_service_distributions(self):
        root = Path.cwd() / ".tmp_test_faiss_manifest_case"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        try:
            build_root = root / "build"
            faiss_root = build_root / "faiss"
            faiss_root.mkdir(parents=True)
            paths = {
                "root": faiss_root,
                "index_file": faiss_root / "index.faiss",
                "chunks_file": faiss_root / "chunks.json",
                "manifest_file": faiss_root / "index_manifest.json",
                "bm25_file": faiss_root / "bm25_corpus.json",
            }
            storage = MagicMock()
            storage.faiss_build_paths.return_value = paths

            chunks = [
                {
                    "id": "chunk-1",
                    "text": "UC@Student attestation et notes",
                    "metadata": {
                        "service_name": "ucastudent",
                        "service_type": "digital_service",
                        "document_type": "digital_service",
                        "document_category": "digital_service",
                        "page_kind": "guide",
                        "intent": ["attestation", "notes"],
                        "render_mode": "static",
                        "is_actionable": True,
                        "student_relevance_score": 0.91,
                        "freshness_score": 0.8,
                        "source_priority": "A",
                        "processing_policy_version": "v-test",
                        "corpus": "main",
                    },
                }
            ]

            adapter = FaissVectorStoreAdapter(storage=storage)
            with patch("rag_module.offline.indexing.load_cache", return_value={"version": 2, "models": {}}), patch(
                "rag_module.offline.indexing.embed", return_value=[[0.1, 0.2]]
            ), patch("rag_module.offline.indexing.get_active_model_name", return_value="test-model"), patch(
                "rag_module.offline.indexing.get_model_name", return_value="test-model"
            ):
                result = adapter.build_index(chunks, corpus="published", build_id="build-test", publish=False)

            manifest = json.loads(paths["manifest_file"].read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(manifest["service_name_distribution"]["ucastudent"], 1)
        self.assertEqual(manifest["page_kind_distribution"]["guide"], 1)
        self.assertEqual(manifest["intent_distribution"]["attestation"], 1)
        self.assertEqual(manifest["render_mode_distribution"]["static"], 1)
        self.assertEqual(manifest["actionable_chunk_count"], 1)
        self.assertGreater(manifest["average_student_relevance_score"], 0.8)

    def test_publish_faiss_build_copies_active_files_to_legacy_root(self):
        root = Path.cwd() / ".tmp_test_publish_faiss_case"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        try:
            settings = MagicMock()
            settings.rag_index_dir = root / "index"
            settings.rag_reports_dir = root / "reports"
            settings.ensure_directories = MagicMock()
            storage = DocumentStorage(settings=settings)

            build_id = "build-demo"
            build_paths = storage.faiss_build_paths(build_id)
            build_paths["index_file"].write_bytes(b"faiss-index")
            build_paths["chunks_file"].write_text(json.dumps([{"id": "1"}]), encoding="utf-8")
            build_paths["manifest_file"].write_text(json.dumps({"chunk_count": 1}), encoding="utf-8")
            build_paths["bm25_file"].write_text(json.dumps([{"id": "1"}]), encoding="utf-8")

            payload = storage.publish_faiss_build(build_id)
            legacy_paths = storage.legacy_faiss_paths()
            self.assertEqual(payload["build_id"], build_id)
            self.assertIn("legacy_paths", payload)
            self.assertTrue(legacy_paths["index_file"].exists())
            self.assertTrue(legacy_paths["chunks_file"].exists())
            self.assertTrue(legacy_paths["manifest_file"].exists())
            self.assertTrue(legacy_paths["bm25_file"].exists())
            self.assertEqual(legacy_paths["index_file"].read_bytes(), b"faiss-index")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @patch("rag_module.retrieval.rag_search.RERANK_FALLBACK_MODELS", ["cross-encoder/ms-marco-MiniLM-L-6-v2"])
    @patch("rag_module.retrieval.rag_search.RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    def test_get_candidate_reranker_names_prioritizes_primary_model(self):
        candidates = get_candidate_reranker_names()

        self.assertEqual(candidates[0], "BAAI/bge-reranker-v2-m3")
        self.assertIn("cross-encoder/ms-marco-MiniLM-L-6-v2", candidates)

    @patch("rag_module.retrieval.rag_search.USE_RERANK", True)
    @patch("rag_module.retrieval.rag_search._reranker", None)
    @patch("rag_module.retrieval.rag_search.RERANK_FALLBACK_MODELS", ["cross-encoder/ms-marco-MiniLM-L-6-v2"])
    @patch("rag_module.retrieval.rag_search.RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    @patch("rag_module.retrieval.rag_search.CrossEncoder")
    def test_get_reranker_falls_back_to_ms_marco_when_primary_fails(self, mocked_cross_encoder):
        mocked_cross_encoder.side_effect = [
            RuntimeError("primary unavailable"),
            MagicMock(name="fallback-reranker"),
        ]

        reranker = get_reranker()

        self.assertIsNotNone(reranker)
        self.assertEqual(mocked_cross_encoder.call_args_list[0].args[0], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(
            mocked_cross_encoder.call_args_list[1].args[0],
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    @patch("rag_module.services.offline.invalidate_search_cache")
    @patch("rag_module.services.offline.get_vector_store_adapter")
    @patch("rag_module.services.offline.load_chunks")
    def test_run_indexing_published_passes_scope_to_adapter(
        self,
        mocked_load_chunks,
        mocked_get_adapter,
        mocked_invalidate,
    ):
        mocked_load_chunks.return_value = [{"id": "1", "text": "texte", "metadata": {"corpus": "main"}}]
        mocked_adapter = MagicMock()
        mocked_adapter.build_index.return_value = MagicMock(
            build_id="build",
            backend="faiss",
            chunk_count=1,
            manifest_path="manifest.json",
            published=True,
        )
        mocked_get_adapter.return_value = mocked_adapter

        run_indexing(corpus="published", publish=True, build_id="build")

        mocked_load_chunks.assert_called_once_with(corpus="published")
        mocked_adapter.build_index.assert_called_once()
        _, kwargs = mocked_adapter.build_index.call_args
        self.assertEqual(kwargs["corpus"], "published")
        self.assertTrue(kwargs["publish"])
        mocked_invalidate.assert_called_once()
