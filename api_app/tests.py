import json
from unittest.mock import patch

from django.test import Client, SimpleTestCase
from django.urls import reverse

from rag_module.generation.rag_engine import RAGEngine, answer_question
from rag_module.retrieval.rag_search import (
    _filter_results_by_allowed_establishments,
    select_support_fallback_results,
)
from rag_module.shared.context_resolution import resolve_context


class ContextResolutionTests(SimpleTestCase):
    def test_known_user_and_neutral_question_falls_back_to_user_establishment(self):
        resolution = resolve_context("Comment se passe l'inscription ?", user_establishment="FSSM")
        self.assertEqual(resolution["mode"], "single_target")
        self.assertEqual(resolution["target_establishments"], ["FSSM"])

    def test_explicit_target_overrides_user_establishment(self):
        resolution = resolve_context("Comment s'inscrire a l'ENSA ?", user_establishment="FSSM")
        self.assertEqual(resolution["mode"], "single_target")
        self.assertEqual(resolution["target_establishments"], ["ENSA"])

    def test_comparison_query_resolves_to_multi_target(self):
        resolution = resolve_context("Quelle difference entre FSSM et ENSA pour l'inscription ?")
        self.assertEqual(resolution["mode"], "multi_target")
        self.assertEqual(resolution["target_establishments"], ["FSSM", "ENSA"])

    def test_global_uca_query_resolves_to_global_mode(self):
        resolution = resolve_context("Quelles sont les filieres disponibles a l'UCA ?")
        self.assertEqual(resolution["mode"], "global_uca")

    def test_unknown_user_and_ambiguous_question_requires_clarification(self):
        resolution = resolve_context("Comment se passe l'inscription ?")
        self.assertEqual(resolution["mode"], "clarification")

    def test_outside_uca_query_is_rejected(self):
        resolution = resolve_context("Comment s'inscrire a l'Universite Mohammed V ?")
        self.assertEqual(resolution["mode"], "out_of_scope")


class RetrievalFilteringTests(SimpleTestCase):
    def test_filter_results_keeps_only_allowed_establishments_and_global_docs(self):
        rows = [
            {"id": "1", "metadata": {"etablissement": "FSSM"}, "text": "FSSM doc"},
            {"id": "2", "metadata": {"etablissement": "ENSA"}, "text": "ENSA doc"},
            {"id": "3", "metadata": {"etablissement": "UCA_GLOBAL"}, "text": "Global doc"},
            {"id": "4", "metadata": {"faculty": "FSSM"}, "text": "Legacy FSSM doc"},
        ]

        filtered = _filter_results_by_allowed_establishments(rows, ["FSSM", "UCA_GLOBAL"])

        self.assertEqual([row["id"] for row in filtered], ["1", "3", "4"])

    def test_support_fallback_keeps_high_support_chunks_when_reranker_is_weak(self):
        query_profile = {"has_strong_topic": True}
        rows = [
            {
                "id": "1",
                "score": 0.8,
                "support_score": 0.83,
                "thematic_score": 0.52,
                "conflicting_topics": [],
                "metadata": {"etablissement": "FSSM"},
            },
            {
                "id": "2",
                "score": 0.6,
                "support_score": 0.59,
                "thematic_score": 0.44,
                "conflicting_topics": [],
                "metadata": {"etablissement": "FSSM"},
            },
        ]

        selected = select_support_fallback_results(rows, query_profile=query_profile, top_k=5)

        self.assertEqual([row["id"] for row in selected], ["1"])


class EngineResolutionTests(SimpleTestCase):
    def test_answer_question_returns_clarification_without_hitting_retrieval(self):
        with patch.object(RAGEngine, "retrieve", side_effect=AssertionError("retrieve should not be called")):
            result = answer_question("Comment se passe l'inscription ?")

        self.assertTrue(result["needs_clarification"])
        self.assertIn("preciser", result["answer"].lower())

    def test_answer_question_returns_out_of_scope_without_hitting_retrieval(self):
        with patch.object(RAGEngine, "retrieve", side_effect=AssertionError("retrieve should not be called")):
            result = answer_question("Comment s'inscrire a l'Universite Mohammed V ?")

        self.assertIn("universite cadi ayyad", result["answer"].lower())

    def test_engine_passes_resolved_context_to_retrieval(self):
        engine = RAGEngine(retrieval_k=1)
        fake_chunks = [{"id": "1", "text": "test", "metadata": {"etablissement": "FSSM"}}]

        with patch.object(RAGEngine, "retrieve", return_value=fake_chunks) as mocked_retrieve:
            with patch.object(RAGEngine, "generate", return_value="ok"):
                engine.answer("Comment se passe l'inscription ?", user_establishment="FSSM")

        resolution_context = mocked_retrieve.call_args.kwargs["resolution_context"]
        self.assertEqual(resolution_context["target_establishments"], ["FSSM"])
        self.assertIn("FSSM", resolution_context["allowed_establishments"])


class ChatApiTests(SimpleTestCase):
    def test_chat_page_sets_csrf_cookie(self):
        response = self.client.get(reverse("chat-page"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("csrftoken", response.cookies)

    def test_chat_api_rejects_invalid_user_establishment(self):
        response = self.client.post(
            reverse("api-chat"),
            data=json.dumps({"message": "Bonjour", "user_establishment": "INVALID"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("user_establishment", response.json()["errors"])

    @patch("api_app.views.answer_question", return_value={"answer": "Veuillez preciser.", "needs_clarification": True})
    def test_chat_api_returns_clarification_flag(self, mocked_answer_question):
        response = self.client.post(
            reverse("api-chat"),
            data=json.dumps({"message": "Comment s'inscrire ?", "user_establishment": "FSSM"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["needs_clarification"])
        mocked_answer_question.assert_called_once_with("Comment s'inscrire ?", user_establishment="FSSM")

    @patch("api_app.views.answer_question", return_value={"answer": "ok"})
    def test_chat_api_accepts_csrf_protected_post_from_chat_page(self, mocked_answer_question):
        client = Client(enforce_csrf_checks=True)
        page_response = client.get(reverse("chat-page"))
        csrf_token = page_response.cookies["csrftoken"].value

        response = client.post(
            reverse("api-chat"),
            data=json.dumps({"message": "Comment s'inscrire a la faculte ?", "user_establishment": "FSSM"}),
            content_type="application/json",
            HTTP_X_CSRFTOKEN=csrf_token,
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        mocked_answer_question.assert_called_once_with(
            "Comment s'inscrire a la faculte ?",
            user_establishment="FSSM",
        )
