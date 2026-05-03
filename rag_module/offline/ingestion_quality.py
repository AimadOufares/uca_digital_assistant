import hashlib
import re
import string
import unicodedata
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from .structured_parser import extract_main_text


WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
URL_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
JS_APP_MARKERS = ("__next", "data-reactroot", "react", "vue", "ng-app", 'id="app"', 'id="root"')
PAGE_KIND_PATTERNS = {
    "calendrier": ("calendrier", "planning", "emploi du temps", "schedule", "date limite", "deadline"),
    "resultats": ("resultat", "resultats", "notes", "deliberation", "rattrapage"),
    "faq": ("faq", "foire aux questions", "questions frequentes"),
    "formulaire": ("formulaire", "deposer", "soumettre", "postuler", "candidature en ligne"),
    "procedure": ("procedure", "etape", "demarche", "pieces a fournir", "comment", "instructions"),
    "guide": ("guide", "manuel", "mode d'emploi", "accompagnement"),
}
INTENT_KEYWORDS = {
    "connexion": ("connexion", "se connecter", "login", "authentification", "acceder"),
    "mot_de_passe": ("mot de passe", "password", "reinitialiser", "oubli"),
    "attestation": ("attestation", "certificat", "e-diplome", "diplome"),
    "notes": ("notes", "resultats", "releve", "deliberation"),
    "reinscription": ("reinscription", "reinscrire", "inscription administrative"),
    "candidature": ("candidature", "postuler", "depot dossier", "preinscription"),
    "cours": ("cours", "module", "examen en ligne", "devoir", "classe virtuelle"),
    "depot_document": ("depot", "televerser", "soumettre", "upload", "piece jointe"),
}
MOJIBAKE_PATTERN = re.compile(r"(ÃƒÆ’.|Ãƒâ€š.|ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦|ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢|ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ|ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“|ÃƒÂ¯Ã‚Â¿Ã‚Â½)")
SKIPPED_CONTENT_TYPE_PREFIXES = ("audio/", "image/", "video/")


def normalize_quality_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_text_hash(text: str) -> str:
    normalized = normalize_quality_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_text_preview(content: bytes, ext: str) -> str:
    if not content:
        return ""
    if ext in {".html", ".htm"}:
        try:
            html = content[:2_000_000].decode("utf-8", errors="replace")
            extracted = extract_main_text(html)
            return str(extracted.get("text") or "")
        except Exception:
            return ""
    if ext in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    return ""


def extract_structured_preview(content: bytes, ext: str) -> Dict[str, object]:
    if not content:
        return {"text": "", "title": "", "headings": [], "links": [], "lists": 0, "tables": 0}
    if ext in {".html", ".htm"}:
        try:
            html = content[:2_000_000].decode("utf-8", errors="replace")
            return extract_main_text(html)
        except Exception:
            return {"text": "", "title": "", "headings": [], "links": [], "lists": 0, "tables": 0}
    preview = extract_text_preview(content, ext)
    return {"text": preview, "title": "", "headings": [], "links": [], "lists": 0, "tables": 0}


def score_text_quality(text: str) -> Dict:
    cleaned = unescape(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = WORD_PATTERN.findall(cleaned.lower())
    word_count = len(words)
    char_count = len(cleaned)
    non_space = sum(1 for ch in cleaned if not ch.isspace())
    alpha_count = sum(1 for ch in cleaned if ch.isalpha())
    symbol_count = sum(
        1 for ch in cleaned if ch in string.punctuation or (not ch.isalnum() and not ch.isspace())
    )
    url_count = len(URL_PATTERN.findall(cleaned))
    alpha_ratio = (alpha_count / non_space) if non_space else 0.0
    symbol_ratio = (symbol_count / non_space) if non_space else 0.0
    url_density = url_count / max(word_count, 1)
    mojibake_ratio = len(MOJIBAKE_PATTERN.findall(cleaned)) / max(char_count, 1)
    return {
        "text": cleaned,
        "words": word_count,
        "chars": char_count,
        "alpha_ratio": alpha_ratio,
        "symbol_ratio": symbol_ratio,
        "url_density": url_density,
        "mojibake_ratio": mojibake_ratio,
    }


def detect_page_kind(url: str, preview_text: str, title: str = "") -> str:
    haystack = normalize_quality_text(f"{url} {title} {preview_text[:4000]}")
    for page_kind, keywords in PAGE_KIND_PATTERNS.items():
        if any(normalize_quality_text(keyword) in haystack for keyword in keywords):
            return page_kind
    return "landing"


def detect_intents(url: str, preview_text: str, title: str = "") -> List[str]:
    haystack = normalize_quality_text(f"{url} {title} {preview_text[:6000]}")
    return [
        intent
        for intent, keywords in INTENT_KEYWORDS.items()
        if any(normalize_quality_text(keyword) in haystack for keyword in keywords)
    ]


def compute_freshness_score(last_modified: str, preview_text: str) -> float:
    dates = re.findall(r"\b(?:19|20)\d{2}\b", preview_text[:5000])
    score = 0.45
    if last_modified:
        score += 0.2
    if dates:
        latest_year = max(int(year) for year in dates)
        if latest_year >= datetime.now().year:
            score += 0.3
        elif latest_year >= datetime.now().year - 1:
            score += 0.2
        elif latest_year >= 2024:
            score += 0.1
    return max(0.0, min(1.0, round(score, 4)))


def detect_quality_issue(metrics: Dict, page_kind: str, render_mode: str, render_success: bool) -> str:
    if render_mode == "playwright" and not render_success:
        return "empty_after_playwright"
    if metrics.get("chars", 0) < 120:
        return "empty_after_static"
    if metrics.get("mojibake_ratio", 0.0) > 0.01:
        return "encoding_suspect"
    lowered = str(metrics.get("text", "") or "").lower()
    if any(token in lowered for token in ("se connecter", "login", "mot de passe", "authentification")) and metrics.get("words", 0) < 70:
        return "login_wall"
    if page_kind == "landing" and metrics.get("words", 0) < 140:
        return "too_generic"
    return ""


def should_use_js_fallback(html_text: str, structured: Dict[str, object], metrics: Dict[str, object]) -> bool:
    lowered_html = (html_text or "").lower()
    text = str(structured.get("text") or "")
    heading_count = len(structured.get("headings", []) or [])
    list_count = int(structured.get("lists", 0) or 0)
    if int(metrics.get("words", 0) or 0) < 45 and any(marker in lowered_html for marker in JS_APP_MARKERS):
        return True
    if int(metrics.get("chars", 0) or 0) < 250 and lowered_html.count("<script") >= 6:
        return True
    if heading_count == 0 and list_count == 0 and int(metrics.get("words", 0) or 0) < 35:
        return True
    if "<iframe" in lowered_html and int(metrics.get("words", 0) or 0) < 80:
        return True
    if text.strip().lower() in {"", "connexion", "login"}:
        return True
    return False


def compute_download_quality(url: str, depth: int, content: bytes, content_type: str, ext: str) -> Dict:
    content_size = len(content or b"")
    lowered_type = (content_type or "").lower().strip()
    normalized_url = normalize_quality_text(url)
    keyword_hits = sorted(
        keyword
        for keyword in (
            "inscription",
            "preinscription",
            "reinscription",
            "candidature",
            "admission",
            "resultat",
            "resultats",
            "rattrapage",
            "calendrier",
            "scolarite",
            "attestation",
            "releve",
            "notes",
            "bourse",
            "contact",
            "formation",
            "master",
            "licence",
            "reglement",
            "ucastudent",
            "e-candidature",
        )
        if keyword in normalized_url
    )

    if content_size <= 0:
        return {"keep": False, "score": 0, "reason": "empty_content", "keyword_hits": keyword_hits, "metrics": {}, "preview_text": "", "text_content_hash": ""}
    if content_size > 25_000_000:
        return {"keep": False, "score": 0, "reason": "file_too_large", "keyword_hits": keyword_hits, "metrics": {"bytes": content_size}, "preview_text": "", "text_content_hash": ""}
    if any(lowered_type.startswith(prefix) for prefix in SKIPPED_CONTENT_TYPE_PREFIXES):
        return {"keep": False, "score": 0, "reason": "unsupported_media_content_type", "keyword_hits": keyword_hits, "metrics": {"content_type": lowered_type}, "preview_text": "", "text_content_hash": ""}
    if ext == ".doc":
        return {"keep": False, "score": 0, "reason": "unsupported_legacy_doc", "keyword_hits": keyword_hits, "metrics": {"bytes": content_size}, "preview_text": "", "text_content_hash": ""}

    if ext in {".pdf", ".docx"}:
        score = 55
        if content_size >= 8_000:
            score += 20
        else:
            score -= 25
        if keyword_hits:
            score += min(20, len(keyword_hits) * 5)
        elif depth >= 2:
            score -= 10
        score = max(0, min(100, int(round(score))))
        return {
            "keep": score >= 52,
            "score": score,
            "reason": "binary_quality_gate",
            "keyword_hits": keyword_hits,
            "metrics": {"bytes": content_size},
            "preview_text": "",
            "text_content_hash": "",
            "page_kind": "landing",
            "intent": [],
            "freshness_score": 0.45,
            "quality_issue": "",
            "title": "",
            "headings": [],
            "links": [],
            "js_dependent": False,
        }

    structured = extract_structured_preview(content, ext)
    preview = str(structured.get("text") or "")
    title = str(structured.get("title") or "")
    metrics = score_text_quality(preview)
    page_kind = detect_page_kind(url, preview, title=title)
    intents = detect_intents(url, preview, title=title)
    freshness_score = compute_freshness_score("", preview)
    score = 100.0
    if metrics["words"] < 90:
        score -= min(35.0, (90 - metrics["words"]) * 0.5)
    if metrics["chars"] < 600:
        score -= min(30.0, (600 - metrics["chars"]) / 20.0)
    if metrics["alpha_ratio"] < 0.55:
        score -= (0.55 - metrics["alpha_ratio"]) * 90.0
    if metrics["symbol_ratio"] > 0.30:
        score -= (metrics["symbol_ratio"] - 0.30) * 130.0
    if metrics["url_density"] > 0.08:
        score -= min(25.0, (metrics["url_density"] - 0.08) * 250.0)
    if metrics["mojibake_ratio"] > 0.008:
        score -= min(25.0, (metrics["mojibake_ratio"] - 0.008) * 1200.0)
    if keyword_hits:
        score += min(16.0, len(keyword_hits) * 4.0)
    elif depth >= 2:
        score -= 12.0
    if intents:
        score += min(14.0, len(intents) * 3.0)
    if page_kind in {"procedure", "guide", "faq", "formulaire"}:
        score += 8.0
    elif page_kind == "landing":
        score -= 10.0
    if "404" in metrics["text"] and metrics["words"] < 140:
        score -= 20.0
    score = max(0, min(100, int(round(score))))
    quality_issue = detect_quality_issue(metrics, page_kind, render_mode="static", render_success=True)
    return {
        "keep": score >= 52,
        "score": score,
        "reason": "text_quality_gate",
        "keyword_hits": keyword_hits,
        "metrics": {
            "bytes": content_size,
            "words": metrics["words"],
            "chars": metrics["chars"],
            "alpha_ratio": round(metrics["alpha_ratio"], 4),
            "symbol_ratio": round(metrics["symbol_ratio"], 4),
            "url_density": round(metrics["url_density"], 4),
            "mojibake_ratio": round(metrics["mojibake_ratio"], 5),
        },
        "preview_text": metrics["text"][:12000],
        "text_content_hash": compute_text_hash(metrics["text"][:12000]),
        "page_kind": page_kind,
        "intent": intents,
        "freshness_score": freshness_score,
        "quality_issue": quality_issue,
        "title": title,
        "headings": structured.get("headings", []),
        "links": structured.get("links", []),
        "js_dependent": should_use_js_fallback(
            content[:200000].decode("utf-8", errors="replace") if ext in {".html", ".htm"} else "",
            structured,
            metrics,
        ),
    }
