import logging
from functools import lru_cache
from typing import Tuple

from .runtime_config import fasttext_model_path, language_detector_name

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_fasttext_model():
    try:
        import fasttext

        model_path = fasttext_model_path()
        if not model_path.exists():
            return None
        return fasttext.load_model(str(model_path))
    except Exception as exc:
        logger.warning("Modele fastText indisponible, fallback actif: %s", exc)
        return None


def detect_language(text: str, min_confidence: float = 0.85) -> Tuple[str, float, str]:
    sample = (text or "").strip()
    if len(sample.split()) < 20:
        return "unknown", 0.0, "none"

    if language_detector_name() == "fasttext":
        model = _load_fasttext_model()
        if model is not None:
            try:
                labels, scores = model.predict(sample.replace("\n", " "), k=1)
                label = str(labels[0]).replace("__label__", "") if labels else "unknown"
                score = float(scores[0]) if scores else 0.0
                if score >= min_confidence:
                    return label, score, "fasttext"
            except Exception as exc:
                logger.warning("fastText language detection en erreur, fallback actif: %s", exc)

    try:
        from langdetect import LangDetectException, detect_langs

        candidates = detect_langs(sample[:1500])
        if not candidates:
            return "unknown", 0.0, "langdetect"
        top = candidates[0]
        label = getattr(top, "lang", "unknown") or "unknown"
        score = float(getattr(top, "prob", 0.0) or 0.0)
        if score >= min_confidence:
            return label, score, "langdetect"
        return "unknown", score, "langdetect"
    except Exception:
        return "unknown", 0.0, "none"
