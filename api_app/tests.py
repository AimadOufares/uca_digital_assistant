import json
import shutil
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from rag_module.contracts import AnswerResult
from rag_module.offline.indexing import load_chunks
from rag_module.offline.ingestion_utils import (
    _download_document,
    compute_download_quality,
    decide_document,
    default_seeds,
)
from rag_module.offline.processing import clean_text
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
                        "text": "Informations inscription universite Cadi Ayyad",
                        "metadata": {"chunk_hash": "main-1", "source": "main-source", "corpus": "main"},
                    }
                ),
                encoding="utf-8",
            )
            (drive_dir / "drive_chunk.json").write_text(
                json.dumps(
                    {
                        "text": "PUCAStaff plateforme officielle gestion administrative du personnel",
                        "metadata": {"chunk_hash": "drive-1", "source": "drive-source", "corpus": "drive"},
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
