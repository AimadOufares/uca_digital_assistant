import json
import shutil
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APITestCase

from api_app.models import Conversation
from rag_module.adapters.vector_store import FaissVectorStoreAdapter
from rag_module.adapters.storage import DocumentStorage
from rag_module.adapters.llm_provider import LLMProviderAdapter
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
from rag_module.services.health import build_ready_health
from rag_module.services.offline import run_indexing


class ChatApiTests(APITestCase):
    def setUp(self):
        super().setUp()
        self.student_user = get_user_model().objects.create_user(
            username="etudiant-test",
            email="etudiant@uca.ac.ma",
            password="Secret12345!",
            first_name="Etudiant",
            last_name="UCA",
        )

    def test_chat_endpoint_requires_authentication(self):
        response = self.client.post(reverse("api-chat"), {"message": "Bonjour"}, format="json")

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch("api_app.views.answer_question")
    def test_chat_endpoint_returns_answer_payload(self, mocked_answer_question):
        mocked_answer_question.return_value = AnswerResult(
            answer="Reponse test",
            sources=[],
            confidence="moyen",
            backend="faiss",
            retrieval_meta={},
        )
        self.client.force_login(self.student_user)

        response = self.client.post(reverse("api-chat"), {"message": "Bonjour"}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        payload = response.json()
        self.assertEqual(payload["answer"], "Reponse test")
        self.assertEqual(payload["sources"], [])
        self.assertEqual(payload["confidence"], "moyen")
        self.assertEqual(payload["backend"], "faiss")
        self.assertEqual(payload["retrieval_meta"], {})
        self.assertIn("conversation_id", payload)
        conversation = Conversation.objects.get(pk=payload["conversation_id"])
        self.assertEqual(conversation.user, self.student_user)
        self.assertEqual(conversation.messages.count(), 2)
        mocked_answer_question.assert_called_once()

    def test_chat_endpoint_rejects_empty_message(self):
        self.client.force_login(self.student_user)
        response = self.client.post(reverse("api-chat"), {"message": ""}, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("detail", response.json())

    @patch("api_app.views.answer_question")
    def test_chat_history_endpoint_returns_saved_messages(self, mocked_answer_question):
        mocked_answer_question.return_value = AnswerResult(
            answer="Historique test",
            sources=[{"name": "guide.pdf", "score": 0.88}],
            confidence="eleve",
            backend="faiss",
            retrieval_meta={"provider": "lmstudio"},
        )
        self.client.force_login(self.student_user)
        self.client.post(reverse("api-chat"), {"message": "Comment obtenir mon attestation ?"}, format="json")

        response = self.client.get(reverse("api-chat"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        payload = response.json()
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][1]["role"], "assistant")
        self.assertEqual(payload["messages"][1]["confidence"], "eleve")
        self.assertEqual(len(payload["conversations"]), 1)

    def test_chat_get_returns_selected_conversation_only(self):
        conversation_a = Conversation.objects.create(user=self.student_user, title="Conversation A")
        conversation_b = Conversation.objects.create(user=self.student_user, title="Conversation B")
        conversation_a.messages.create(role="user", content="Bonjour A")
        conversation_b.messages.create(role="user", content="Bonjour B")
        self.client.force_login(self.student_user)

        response = self.client.get(reverse("api-chat"), {"conversation_id": conversation_b.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        payload = response.json()
        self.assertEqual(payload["conversation_id"], conversation_b.id)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["content"], "Bonjour B")
        self.assertEqual(len(payload["conversations"]), 2)

    def test_chat_can_create_new_conversation(self):
        self.client.force_login(self.student_user)

        response = self.client.post(reverse("api-chat-conversations"), {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        payload = response.json()
        self.assertIn("conversation_id", payload)
        self.assertEqual(payload["messages"], [])
        self.assertEqual(Conversation.objects.filter(user=self.student_user).count(), 1)

    @patch("api_app.views.answer_question")
    def test_chat_post_can_target_specific_conversation(self, mocked_answer_question):
        mocked_answer_question.return_value = AnswerResult(
            answer="Reponse ciblee",
            sources=[],
            confidence="moyen",
            backend="faiss",
            retrieval_meta={},
        )
        selected = Conversation.objects.create(user=self.student_user, title="Cible")
        self.client.force_login(self.student_user)

        response = self.client.post(
            reverse("api-chat"),
            {"message": "Question ciblee", "conversation_id": selected.id},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        selected.refresh_from_db()
        self.assertEqual(selected.messages.count(), 2)

    def test_chat_conversation_can_be_renamed(self):
        conversation = Conversation.objects.create(user=self.student_user, title="Ancien titre")
        self.client.force_login(self.student_user)

        response = self.client.patch(
            reverse("api-chat-conversation-detail", kwargs={"conversation_id": conversation.id}),
            {"title": "Nouveau titre"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        conversation.refresh_from_db()
        self.assertEqual(conversation.title, "Nouveau titre")

    def test_chat_conversation_can_be_archived(self):
        conversation = Conversation.objects.create(user=self.student_user, title="A archiver")
        self.client.force_login(self.student_user)

        response = self.client.delete(reverse("api-chat-conversation-detail", kwargs={"conversation_id": conversation.id}))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        conversation.refresh_from_db()
        self.assertTrue(conversation.is_archived)


class StudentAuthTests(APITestCase):
    def test_signup_page_accepts_allowed_uca_email(self):
        response = self.client.post(
            reverse("student-signup"),
            {
                "first_name": "Sara",
                "last_name": "UCA",
                "email": "SARA@UCA.AC.MA",
                "password1": "Secret12345!",
                "password2": "Secret12345!",
            },
        )

        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertEqual(response.headers["Location"], reverse("chat-page"))
        created_user = get_user_model().objects.get(email="sara@uca.ac.ma")
        self.assertTrue(created_user.check_password("Secret12345!"))

    def test_signup_rejects_non_uca_email(self):
        response = self.client.post(
            reverse("student-signup"),
            {
                "first_name": "Sara",
                "last_name": "UCA",
                "email": "sara@gmail.com",
                "password1": "Secret12345!",
                "password2": "Secret12345!",
            },
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertContains(response, "Inscription reservee aux emails UCA autorises")

    def test_login_page_accepts_email_authentication(self):
        get_user_model().objects.create_user(
            username="student-email-login",
            email="student.login@uca.ac.ma",
            password="Secret12345!",
        )

        response = self.client.post(
            reverse("student-login"),
            {"username": "student.login@uca.ac.ma", "password": "Secret12345!"},
        )

        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertEqual(response.headers["Location"], reverse("chat-page"))

    def test_chat_page_requires_login(self):
        response = self.client.get(reverse("chat-page"))

        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertIn("/login/", response.headers["Location"])


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
            "llm": {"state": "down", "usable": False},
        }

        response = self.client.get(reverse("api-health-ready"))

        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertFalse(response.json()["ready"])


class HealthLogicTests(APITestCase):
    def test_llm_provider_health_is_down_when_single_configured_provider_fails(self):
        settings = MagicMock()
        settings.rag_llm_provider = "lmstudio"
        settings.lm_studio_base_url = ""
        settings.lm_studio_api_key = "lm-studio"
        settings.openai_api_key = ""

        payload = LLMProviderAdapter(settings).health()

        self.assertEqual(payload["state"], "down")
        self.assertFalse(payload["usable"])
        self.assertEqual(payload["provider_order"], ["lmstudio"])

    @patch("rag_module.adapters.llm_provider.OpenAI")
    def test_llm_provider_health_is_degraded_when_auto_fallback_is_partially_available(self, mocked_openai):
        models_response = MagicMock()
        models_response.data = [MagicMock(id="gpt-4o-mini")]
        openai_client = MagicMock()
        openai_client.models.list.return_value = models_response
        mocked_openai.side_effect = [RuntimeError("lmstudio offline"), openai_client]

        settings = MagicMock()
        settings.rag_llm_provider = "auto"
        settings.lm_studio_base_url = "http://127.0.0.1:1234/v1"
        settings.lm_studio_api_key = "lm-studio"
        settings.openai_api_key = "sk-test"

        payload = LLMProviderAdapter(settings).health()

        self.assertEqual(payload["state"], "degraded")
        self.assertTrue(payload["usable"])
        self.assertEqual(payload["provider_order"], ["lmstudio", "openai"])
        self.assertFalse(payload["providers"]["lmstudio"]["ok"])
        self.assertTrue(payload["providers"]["openai"]["ok"])

    @patch("rag_module.services.health.get_runtime_settings")
    @patch("rag_module.services.health.DocumentStorage")
    @patch("rag_module.services.health.get_vector_store_adapter")
    @patch("rag_module.services.health.LLMProviderAdapter")
    @patch("rag_module.services.health._database_health")
    def test_build_ready_health_requires_usable_llm(
        self,
        mocked_database_health,
        mocked_llm_adapter,
        mocked_get_vector_store_adapter,
        mocked_storage_cls,
        mocked_get_runtime_settings,
    ):
        mocked_get_runtime_settings.return_value = MagicMock(app_env="test")
        mocked_database_health.return_value = {"ok": True}
        mocked_storage = MagicMock()
        mocked_storage.load_active_index_pointer.return_value = {"backend": "faiss", "build_id": "build-1"}
        mocked_storage_cls.return_value = mocked_storage
        mocked_vector = MagicMock()
        mocked_vector.health.return_value = {"ok": True, "active_index_present": True}
        mocked_get_vector_store_adapter.return_value = mocked_vector
        mocked_llm_adapter.return_value.health.return_value = {
            "state": "down",
            "usable": False,
            "providers": {"lmstudio": {"ok": False, "reason": "offline"}},
        }

        payload = build_ready_health()

        self.assertFalse(payload["ready"])
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["checks"]["llm_ready"])

    @patch("rag_module.services.health.get_runtime_settings")
    @patch("rag_module.services.health.DocumentStorage")
    @patch("rag_module.services.health.get_vector_store_adapter")
    @patch("rag_module.services.health.LLMProviderAdapter")
    @patch("rag_module.services.health._database_health")
    def test_build_ready_health_accepts_degraded_but_usable_llm(
        self,
        mocked_database_health,
        mocked_llm_adapter,
        mocked_get_vector_store_adapter,
        mocked_storage_cls,
        mocked_get_runtime_settings,
    ):
        mocked_get_runtime_settings.return_value = MagicMock(app_env="test")
        mocked_database_health.return_value = {"ok": True}
        mocked_storage = MagicMock()
        mocked_storage.load_active_index_pointer.return_value = {"backend": "faiss", "build_id": "build-1"}
        mocked_storage_cls.return_value = mocked_storage
        mocked_vector = MagicMock()
        mocked_vector.health.return_value = {"ok": True, "active_index_present": True}
        mocked_get_vector_store_adapter.return_value = mocked_vector
        mocked_llm_adapter.return_value.health.return_value = {
            "state": "degraded",
            "usable": True,
            "providers": {"lmstudio": {"ok": False}, "openai": {"ok": True}},
        }

        payload = build_ready_health()

        self.assertTrue(payload["ready"])
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["checks"]["llm_ready"])


class DriveDocumentsApiTests(APITestCase):
    def setUp(self):
        super().setUp()
        self.admin_user = get_user_model().objects.create_user(
            username="admin_dashboard",
            password="secret123",
            is_staff=True,
            is_superuser=True,
        )

    def test_admin_dashboard_page_requires_login(self):
        response = self.client.get(reverse("admin-dashboard"))

        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertIn("/admin/login/", response.headers["Location"])

    def test_dashboard_metrics_requires_admin(self):
        response = self.client.get(reverse("api-dashboard-metrics"))

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch("api_app.views.build_dashboard_payload")
    def test_dashboard_metrics_returns_enriched_payload_for_admin(self, mocked_payload):
        mocked_payload.return_value = {
            "system_status": {"ready": True},
            "active_index": {"build_id": "build-1"},
            "drive_sync_status": {"status": "up_to_date"},
            "rag_eval": {"benchmark": "drive", "summary": {"service_top1_accuracy": 1.0}},
        }
        self.client.force_login(self.admin_user)

        response = self.client.get(reverse("api-dashboard-metrics"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["active_index"]["build_id"], "build-1")
        mocked_payload.assert_called_once()

    def test_drive_documents_can_be_uploaded_listed_and_deleted(self):
        root = Path.cwd() / ".tmp_test_drive_docs_case"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        try:
            drive_dir = root / "drive"
            drive_dir.mkdir()
            uploaded = SimpleUploadedFile(
                "guide-test.txt",
                b"Guide de test pour le corpus drive",
                content_type="text/plain",
            )
            with patch("api_app.views.get_runtime_settings") as mocked_runtime:
                mocked_runtime.return_value.rag_raw_drive_dir = drive_dir
                self.client.force_login(self.admin_user)

                upload_response = self.client.post(
                    reverse("api-drive-documents"),
                    {"file": uploaded},
                    format="multipart",
                )
                self.assertEqual(upload_response.status_code, status.HTTP_201_CREATED)
                self.assertTrue((drive_dir / "guide-test.txt").exists())

                list_response = self.client.get(reverse("api-drive-documents"))
                self.assertEqual(list_response.status_code, status.HTTP_200_OK)
                self.assertEqual(list_response.json()["count"], 1)

                delete_response = self.client.delete(
                    reverse("api-drive-document-detail", kwargs={"filename": "guide-test.txt"})
                )
                self.assertEqual(delete_response.status_code, status.HTTP_200_OK)
                self.assertFalse((drive_dir / "guide-test.txt").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_drive_documents_requires_admin(self):
        response = self.client.get(reverse("api-drive-documents"))

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch("api_app.views.run_indexing")
    @patch("api_app.views.run_processing")
    def test_drive_rebuild_endpoint_runs_processing_and_indexing(self, mocked_processing, mocked_indexing):
        mocked_processing.return_value = {"status": "ok", "step": "processing", "corpus": "drive"}
        mocked_indexing.return_value = MagicMock(
            backend="faiss",
            build_id="build-test",
            chunk_count=33,
            published=True,
        )
        self.client.force_login(self.admin_user)

        response = self.client.post(reverse("api-drive-rebuild"), {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["index"]["build_id"], "build-test")
        mocked_processing.assert_called_once_with(corpus="drive")
        mocked_indexing.assert_called_once_with(corpus="published", publish=True)

    @patch("api_app.views.latest_report_payload")
    @patch("api_app.views.run_evaluation")
    def test_drive_evaluate_endpoint_runs_drive_benchmark(self, mocked_run_evaluation, mocked_latest_report):
        mocked_run_evaluation.return_value = {"json": "report.json", "txt": "report.txt"}
        mocked_latest_report.return_value = {"available": True, "report": {"benchmark": "drive"}}
        self.client.force_login(self.admin_user)

        response = self.client.post(reverse("api-drive-evaluate"), {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["report"]["report"]["benchmark"], "drive")
        mocked_run_evaluation.assert_called_once_with(top_k=5, skip_generation=True, benchmark="drive")

    @patch("api_app.views.latest_report_payload")
    def test_latest_report_endpoint_returns_payload_for_admin(self, mocked_latest_report):
        mocked_latest_report.return_value = {"available": True, "report": {"summary": {}}}
        self.client.force_login(self.admin_user)

        response = self.client.get(reverse("api-report-latest", kwargs={"kind": "rag_eval"}))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mocked_latest_report.assert_called_once_with("rag_eval")


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
