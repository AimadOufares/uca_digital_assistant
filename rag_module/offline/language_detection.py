from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

try:
    import fasttext  # type: ignore
except ImportError:  # pragma: no cover
    fasttext = None

try:
    from langdetect import LangDetectException, detect_langs
except ImportError:  # pragma: no cover
    LangDetectException = Exception  # type: ignore[assignment]
    detect_langs = None

try:
    from ..shared.runtime import RuntimeSettings, get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.runtime import RuntimeSettings, get_runtime_settings


MIN_LANG_WORDS = 20
SUPPORTED_LANGUAGES = {"fr", "ar", "en"}


def _normalize_lang(label: str) -> str:
    value = (label or "").strip().lower()
    if value.startswith("__label__"):
        value = value.replace("__label__", "", 1)
    if value in {"ara", "ar"}:
        return "ar"
    if value in {"fra", "fr"}:
        return "fr"
    if value in {"eng", "en"}:
        return "en"
    return value


@lru_cache(maxsize=1)
def _fasttext_model(model_path: str):
    if fasttext is None:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return fasttext.load_model(str(path))
    except Exception:
        return None


def detect_language(text: str, settings: Optional[RuntimeSettings] = None) -> Tuple[str, float]:
    candidate = (text or "").strip()
    if len(candidate.split()) < MIN_LANG_WORDS:
        return "unknown", 0.0

    runtime = settings or get_runtime_settings()
    detector = runtime.rag_language_detector

    if detector == "fasttext":
        model = _fasttext_model(str(runtime.rag_fasttext_model_path))
        if model is not None:
            try:
                labels, scores = model.predict(candidate.replace("\n", " "), k=1)
                if labels and scores:
                    lang = _normalize_lang(labels[0])
                    score = float(scores[0])
                    if lang in SUPPORTED_LANGUAGES:
                        return lang, score
                    return "unknown", score
            except Exception:
                pass

    if detect_langs is None:
        return "unknown", 0.0

    try:
        candidates = detect_langs(candidate[:1500])
        if not candidates:
            return "unknown", 0.0
        top = candidates[0]
        lang = _normalize_lang(getattr(top, "lang", "unknown") or "unknown")
        score = float(getattr(top, "prob", 0.0) or 0.0)
        if lang in SUPPORTED_LANGUAGES:
            return lang, score
        return "unknown", score
    except LangDetectException:
        return "unknown", 0.0
