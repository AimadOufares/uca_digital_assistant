import unittest
from unittest.mock import patch

from rag_module.generation import rag_engine


def _sample_chunks():
    return [
        {
            "id": "chunk-1",
            "text": "La procedure d'inscription administrative en master a la FSSM comprend le depot du dossier, la verification des pieces et le respect du calendrier officiel.",
            "score": 0.88,
            "score_type": "hybrid",
            "metadata": {
                "source": "data_storage/processed/fssm_master_2026.json",
                "file_name": "fssm_master_2026.json",
                "document_type": "inscription",
                "etablissement": "FSSM",
                "year": 2026,
            },
        }
    ]


class GenerationFormattingTests(unittest.TestCase):
    def test_generate_with_debug_normalizes_backend_answer_into_required_sections(self):
        engine = rag_engine.RAGEngine(retrieval_k=2, prompt_style="standard")
        chunks = _sample_chunks()

        with patch.object(rag_engine, "_generation_order", return_value=["openai"]):
            with patch.object(
                rag_engine,
                "_generate_with_openai",
                return_value={
                    "backend": "openai",
                    "success": True,
                    "model": "gpt-4o-mini",
                    "latency_ms": 42.0,
                    "answer": "Les etudiants doivent deposer leur dossier selon le calendrier officiel de la FSSM.",
                    "answer_preview": "Les etudiants doivent deposer leur dossier selon le calendrier officiel de la FSSM.",
                    "answer_chars": 87,
                    "error": "",
                },
            ):
                payload = engine.generate_with_debug(
                    "Comment faire l'inscription administrative ?",
                    chunks,
                    resolution_context={"allowed_establishments": ["FSSM"]},
                )

        self.assertFalse(payload["used_fallback"])
        self.assertEqual(payload["backend"], "openai")
        self.assertIn("Reponse", payload["answer"])
        self.assertIn("Sources utiles", payload["answer"])
        self.assertIn("Niveau de confiance:", payload["answer"])
        self.assertEqual(payload["prompt"]["chunk_count"], 1)

    def test_generate_with_debug_falls_back_to_extractive_when_backend_fails(self):
        engine = rag_engine.RAGEngine(retrieval_k=2, prompt_style="standard")
        chunks = _sample_chunks()

        with patch.object(rag_engine, "_generation_order", return_value=["lmstudio", "openai"]):
            with patch.object(
                rag_engine,
                "_generate_with_lm_studio",
                return_value={
                    "backend": "lmstudio",
                    "success": False,
                    "model": "",
                    "latency_ms": 10.0,
                    "answer": "",
                    "answer_preview": "",
                    "answer_chars": 0,
                    "error": "lmstudio_down",
                },
            ):
                with patch.object(
                    rag_engine,
                    "_generate_with_openai",
                    return_value={
                        "backend": "openai",
                        "success": False,
                        "model": "gpt-4o-mini",
                        "latency_ms": 12.0,
                        "answer": "",
                        "answer_preview": "",
                        "answer_chars": 0,
                        "error": "openai_down",
                    },
                ):
                    payload = engine.generate_with_debug("Comment faire l'inscription ?", chunks)

        self.assertTrue(payload["used_fallback"])
        self.assertEqual(payload["fallback_type"], "extractive")
        self.assertEqual(len(payload["backend_attempts"]), 2)
        self.assertIn("Sources utiles", payload["answer"])


class AnswerPipelineTests(unittest.TestCase):
    def test_answer_includes_retrieval_and_generation_debug_payloads(self):
        engine = rag_engine.RAGEngine(retrieval_k=2, prompt_style="standard")
        chunks = _sample_chunks()
        retrieval_debug = {
            "final_results": chunks,
            "trace": {"decision_summary": {"abstain": False}},
        }
        generation_debug = {
            "answer": "Reponse\nTest\n\nSources utiles\n- fssm_master_2026.json\n\nNiveau de confiance: moyen",
            "backend": "openai",
            "used_fallback": False,
            "fallback_type": "",
            "backend_attempts": [],
            "prompt": {"prompt_style": "standard", "prompt_chars": 100, "chunk_count": 1},
        }

        with patch.object(rag_engine, "resolve_context", return_value={"mode": "answer"}):
            with patch.object(rag_engine, "allowed_establishments_for_resolution", return_value=["FSSM"]):
                with patch.object(engine, "retrieve_debug", return_value=retrieval_debug):
                    with patch.object(engine, "generate_with_debug", return_value=generation_debug):
                        payload = engine.answer("Comment faire l'inscription ?", user_establishment="FSSM")

        self.assertIn("retrieval_debug", payload)
        self.assertIn("generation_debug", payload)
        self.assertEqual(payload["generation_debug"]["backend"], "openai")
        self.assertEqual(payload["sources"][0]["name"], "fssm_master_2026.json")


if __name__ == "__main__":
    unittest.main()
