from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from rag_module.contracts import AnswerResult


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
