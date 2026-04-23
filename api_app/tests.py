from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from rag_module.contracts import AnswerResult
from rag_module.offline.ingestion_utils import decide_document, default_seeds


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
