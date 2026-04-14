import unittest

from rag_module.shared.html_extraction import extract_main_text
from rag_module.shared.language_detection import detect_language
from rag_module.shared.sparse_vectors import build_sparse_encoder, encode_sparse_text


class UpgradeHelpersTests(unittest.TestCase):
    def test_sparse_encoder_keeps_informative_terms(self):
        encoder = build_sparse_encoder(
            [
                "inscription master calendrier pedagogique",
                "inscription licence documents dossier",
                "bourse universitaire inscription etudiant",
            ],
            min_df=1,
            max_features=100,
        )
        indices, values = encode_sparse_text("inscription master", encoder)
        self.assertTrue(indices)
        self.assertEqual(len(indices), len(values))

    def test_html_extraction_has_fallback(self):
        html = "<html><body><main><h1>Admission</h1><p>Conditions d'inscription</p></main></body></html>"
        payload = extract_main_text(html)
        self.assertIn("Admission", payload.get("text", ""))
        self.assertTrue(payload.get("method"))

    def test_language_detection_handles_short_text(self):
        lang, confidence, detector = detect_language("Bonjour")
        self.assertEqual(lang, "unknown")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertTrue(detector)


if __name__ == "__main__":
    unittest.main()
