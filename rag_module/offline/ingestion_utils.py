import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from ..adapters.dynamic_renderer import render_url
from ..contracts import IngestedDocumentDecision, IngestionJobConfig
from ..shared.runtime import get_runtime_settings
from .ingestion_quality import compute_download_quality
from .source_map import default_seeds_for_mode, get_profile, match_source_rule

logger = logging.getLogger(__name__)

RUNTIME = get_runtime_settings()
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20
RETRIES = 3

ALLOWED_DOMAIN_SUFFIXES = {
    "uca.ma",
    "fstg-marrakech.ac.ma",
}
ALLOWED_DOMAINS = {
    "onousc.ma",
    "www.onousc.ma",
    "enssup.gov.ma",
    "www.enssup.gov.ma",
}

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "sessionid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
BLOCKED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".gif",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".png",
    ".svg",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
SKIPPED_CONTENT_TYPE_PREFIXES = ("audio/", "image/", "video/")
EXCLUDE_PATHS = [
    "/login",
    "/admin",
    "/wp-admin",
    "/logout",
    "javascript:",
    "mailto:",
    "tel:",
    "#",
    "sessionid",
    "utm_",
    "ref=",
]


MAIN_CATEGORIES = {
    "digital_service",
    "inscription",
    "reinscription",
    "admission",
    "candidature",
    "resultats",
    "calendrier",
    "bourse",
    "scolarite",
    "emploi_du_temps",
    "attestation",
    "contact",
    "reglement",
    "formation",
    "vie_etudiante",
}

STATE_PATH = RUNTIME.rag_ingestion_state_dir / "state.json"
RAW_MAIN_METADATA_PATH = RUNTIME.rag_raw_main_dir / ".metadata.json"
RAW_ARCHIVE_METADATA_PATH = RUNTIME.rag_raw_archive_dir / ".metadata.json"
LEGACY_METADATA_PATH = RUNTIME.rag_index_dir / "metadata.json"

_state_lock = threading.Lock()
_storage_lock = threading.Lock()


def default_seeds(mode: str = "fast", premium_only: bool = False) -> List[str]:
    return default_seeds_for_mode(mode, premium_only=premium_only)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def clean_url(url: str) -> str:
    parsed = urlparse(url.strip())
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
    ]
    cleaned = parsed._replace(query=urlencode(filtered_query, doseq=True), fragment="")
    return urlunparse(cleaned)


def compute_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _load_state() -> Dict:
    state = _load_json(STATE_PATH)
    if not isinstance(state, dict):
        return {"items": {}}
    if not isinstance(state.get("items"), dict):
        state["items"] = {}
    return state


def _save_state(state: Dict) -> None:
    _save_json_atomic(STATE_PATH, state)


def _corpus_metadata_path(corpus: str) -> Path:
    return RAW_MAIN_METADATA_PATH if corpus == "main" else RAW_ARCHIVE_METADATA_PATH


def _load_corpus_metadata(corpus: str) -> Dict[str, Dict]:
    payload = _load_json(_corpus_metadata_path(corpus))
    return payload if isinstance(payload, dict) else {}


def _save_corpus_metadata(corpus: str, payload: Dict[str, Dict]) -> None:
    _save_json_atomic(_corpus_metadata_path(corpus), payload)


def _is_allowed_domain(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if host in ALLOWED_DOMAINS:
        return True
    return any(host == domain or host.endswith(f".{domain}") for domain in ALLOWED_DOMAIN_SUFFIXES)


def should_accept_url(url: str, base: Optional[str] = None) -> Optional[str]:
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    url = clean_url(url)
    parsed = urlparse(url)
    path_ext = Path(parsed.path).suffix.lower()

    if parsed.scheme not in {"http", "https"}:
        return None
    if not _is_allowed_domain(parsed.netloc):
        return None
    if path_ext in BLOCKED_EXTENSIONS:
        return None
    if any(fragment in url.lower() for fragment in EXCLUDE_PATHS):
        return None
    if len(url) > 300:
        return None
    return url


def extract_links(content: bytes, base: str) -> List[str]:
    soup = BeautifulSoup(content, "lxml")
    links = set()
    for tag in soup.find_all(["a", "iframe", "embed"]):
        href = tag.get("href") or tag.get("src")
        full = should_accept_url(href, base)
        if full:
            links.add(full)
    return list(links)


def infer_extension(url: str, content_type: str, content: bytes = b"", content_disposition: str = "") -> str:
    lowered_type = (content_type or "").lower()
    lowered_disposition = (content_disposition or "").lower()
    path_ext = Path(urlparse(url).path).suffix.lower()
    allowed_exts = {".html", ".htm", ".pdf", ".doc", ".docx", ".txt", ".md"}

    if "filename=" in lowered_disposition:
        disposition_name = lowered_disposition.split("filename=", 1)[1].strip(" \"'")
        disposition_ext = Path(disposition_name).suffix.lower()
        if disposition_ext in allowed_exts:
            return disposition_ext
    if "text/html" in lowered_type:
        return ".html"
    if "application/pdf" in lowered_type:
        return ".pdf"
    if "application/msword" in lowered_type:
        return ".doc"
    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in lowered_type:
        return ".docx"
    if "text/plain" in lowered_type:
        return ".txt"
    if content.startswith(b"%PDF"):
        return ".pdf"
    if path_ext in allowed_exts:
        return path_ext
    return ".html"


def _generate_filename(url: str, ext: str) -> str:
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = re.sub(r"[^\w\-_.]", "_", name)
    base = name[:130].rstrip("._") or "document"
    suffix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{suffix}{ext}"


def _merge_limits(config: IngestionJobConfig) -> Dict:
    profile = get_profile(config.mode)
    limits = {
        "max_depth": profile["max_depth"],
        "max_total_urls": profile["max_total_urls"],
        "max_urls_per_domain": profile["max_urls_per_domain"],
        "max_urls_per_subdomain": profile["max_urls_per_subdomain"],
    }
    limits.update(config.limits or {})
    limits["max_total_urls"] = min(limits["max_total_urls"], _env_int("RAG_MAX_TOTAL_URLS", limits["max_total_urls"]))
    return limits


def _refresh_days(refresh_mode: str) -> int:
    value = (refresh_mode or "").lower().strip()
    if value == "daily":
        return 1
    if value == "monthly":
        return 30
    if value == "manual":
        return 3650
    return _env_int("RAG_INCREMENTAL_REFRESH_DAYS", 7)


def _priority_points(priority: str) -> int:
    return {"A": 85, "B": 65, "C": 35}.get(priority, 25)


def decide_document(
    url: str,
    quality_score: int,
    keyword_hits: List[str],
    depth: int,
    extension: str,
    mode: str = "fast",
) -> IngestedDocumentDecision:
    rule_info = match_source_rule(url)
    priority = str(rule_info["source_priority"])
    category = str(rule_info["document_category"])
    is_premium = bool(rule_info["is_premium"])
    business_score = _priority_points(priority)
    business_score += min(18, len(keyword_hits) * 4)
    if extension == ".html":
        business_score += 6
    elif extension == ".pdf":
        business_score += 2
    elif extension == ".docx":
        business_score -= 4
    business_score -= min(depth * 4, 12)
    if is_premium:
        business_score += 10
    business_score = max(0, min(100, business_score))

    if quality_score < 52:
        return IngestedDocumentDecision("reject", priority, category, quality_score, business_score, "quality_below_minimum", is_premium)

    if is_premium and quality_score >= 55 and business_score >= 60:
        target = "main" if category in MAIN_CATEGORIES else "archive"
        return IngestedDocumentDecision(target, priority, category, quality_score, business_score, "premium_source", True)

    if priority == "A" and quality_score >= 58 and business_score >= 68:
        return IngestedDocumentDecision("main", priority, category, quality_score, business_score, "high_priority_student_source", is_premium)

    if priority == "B" and category in MAIN_CATEGORIES and quality_score >= 66 and business_score >= 76:
        return IngestedDocumentDecision("main", priority, category, quality_score, business_score, "secondary_source_promoted_to_main", is_premium)

    if priority in {"A", "B"} and quality_score >= 55 and business_score >= 45:
        return IngestedDocumentDecision("archive", priority, category, quality_score, business_score, "secondary_or_exploratory_archive", is_premium)

    if mode == "extended" and priority == "C" and quality_score >= 58 and business_score >= 35:
        return IngestedDocumentDecision("archive", priority, category, quality_score, business_score, "extended_archive_capture", is_premium)

    return IngestedDocumentDecision("reject", priority, category, quality_score, business_score, "insufficient_business_relevance", is_premium)


def _should_keep_for_target(target_corpus: str, decision: IngestedDocumentDecision) -> bool:
    if decision.corpus_target == "reject":
        return False
    if target_corpus == "all":
        return True
    return decision.corpus_target == target_corpus


def _save_document(content: bytes, url: str, corpus: str, ext: str) -> Path:
    target_dir = RUNTIME.rag_raw_main_dir if corpus == "main" else RUNTIME.rag_raw_archive_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = _generate_filename(url, ext)
    path = target_dir / filename
    with path.open("wb") as handle:
        handle.write(content)
    return path


def _save_document_metadata(corpus: str, file_path: Path, payload: Dict) -> None:
    with _storage_lock:
        metadata = _load_corpus_metadata(corpus)
        metadata[str(file_path)] = payload
        _save_corpus_metadata(corpus, metadata)


def _remove_document_metadata(corpus: str, file_path: Path) -> None:
    with _storage_lock:
        metadata = _load_corpus_metadata(corpus)
        metadata.pop(str(file_path), None)
        _save_corpus_metadata(corpus, metadata)


def _save_legacy_metadata(rows: List[Dict]) -> None:
    _save_json_atomic(LEGACY_METADATA_PATH, rows)


def _is_due_for_refresh(entry: Dict, refresh_days: int) -> bool:
    if not entry:
        return True
    raw = str(entry.get("last_checked_at") or "").strip()
    if not raw:
        return True
    try:
        last_checked = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(timezone.utc) - last_checked >= timedelta(days=refresh_days)


def _download_document(
    url: str,
    depth: int,
    js_usage: Optional[Dict[str, int]] = None,
    js_lock: Optional[threading.Lock] = None,
) -> Optional[Dict]:
    for _ in range(RETRIES):
        try:
            timeout = TIMEOUT if depth < 2 else 10
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            content = response.content
            content_type = response.headers.get("Content-Type", "")
            ext = infer_extension(
                url,
                content_type,
                content=content,
                content_disposition=response.headers.get("Content-Disposition", ""),
            )
            quality = compute_download_quality(url, depth, content, content_type, ext)
            render_mode = "static"
            render_success = False
            render_error = ""
            final_url = url
            selector_used = ""

            if (
                ext in {".html", ".htm"}
                and bool(quality.get("js_dependent"))
                and RUNTIME.rag_js_fallback_enabled
            ):
                allowed = False
                if js_usage is not None and js_lock is not None:
                    with js_lock:
                        if js_usage.get("count", 0) < RUNTIME.rag_js_max_pages_per_run:
                            js_usage["count"] = js_usage.get("count", 0) + 1
                            allowed = True
                if allowed:
                    render_result = render_url(url, settings=RUNTIME)
                    render_mode = "playwright"
                    render_success = render_result.ok
                    render_error = render_result.error
                    selector_used = render_result.selector_used
                    final_url = render_result.final_url or url
                    if render_result.ok and render_result.html.strip():
                        content = render_result.html.encode("utf-8", errors="ignore")
                        content_type = "text/html; charset=utf-8"
                        ext = ".html"
                        quality = compute_download_quality(final_url, depth, content, content_type, ext)
                    elif not quality.get("quality_issue"):
                        quality["quality_issue"] = "empty_after_playwright"
                elif not quality.get("quality_issue"):
                    quality["quality_issue"] = quality.get("quality_issue") or "empty_after_static"

            return {
                "url": final_url,
                "depth": depth,
                "content": content,
                "content_type": content_type,
                "extension": ext,
                "headers": dict(response.headers),
                "quality": quality,
                "content_hash": compute_hash(content),
                "render_mode": render_mode,
                "render_success": render_success,
                "render_error": render_error,
                "selector_used": selector_used,
            }
        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            time.sleep(1)
    return None


def _cleanup_previous_artifact(previous: Dict) -> None:
    previous_file = str(previous.get("file_path") or "").strip()
    previous_corpus = str(previous.get("corpus_target") or "").strip()
    if not previous_file or previous_corpus not in {"main", "archive"}:
        return
    previous_path = Path(previous_file)
    try:
        if previous_path.exists():
            previous_path.unlink()
    except Exception:
        logger.warning("Unable to remove stale raw file: %s", previous_path)
    _remove_document_metadata(previous_corpus, previous_path)


def crawl(config: IngestionJobConfig) -> Dict:
    limits = _merge_limits(config)
    refresh_days = _refresh_days(config.refresh_mode or "weekly")
    seeds = list(config.seeds or default_seeds(config.mode, premium_only=config.premium_only))
    state = _load_state()
    state_items = state.setdefault("items", {})
    report_rows: List[Dict] = []
    metrics = {
        "downloaded": 0,
        "main": 0,
        "archive": 0,
        "reject": 0,
        "unchanged": 0,
        "skipped": 0,
        "categories": Counter(),
        "priorities": Counter(),
        "services": Counter(),
        "intents": Counter(),
        "page_kinds": Counter(),
        "quality_issues": Counter(),
        "render_modes": Counter(),
        "documents": [],
    }

    queue_items: "queue.PriorityQueue[tuple[int,int,str]]" = queue.PriorityQueue()
    visited = set()
    domain_counts = defaultdict(int)
    subdomain_counts = defaultdict(int)
    condition = threading.Condition()
    js_lock = threading.Lock()
    js_usage = {"count": 0}
    active_workers = 0
    done = False

    def _domain_bucket(hostname: str) -> str:
        host = (hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _subdomain_bucket(hostname: str) -> str:
        host = (hostname or "").lower()
        return host[4:] if host.startswith("www.") else host

    def _priority_for_url(url: str, depth: int) -> int:
        info = match_source_rule(url)
        base = {"A": 10, "B": 30, "C": 60}.get(str(info["source_priority"]), 80)
        if info["is_premium"]:
            base -= 15
        path = url.lower()
        if any(
            token in path
            for token in (
                "inscription",
                "reinscription",
                "candidature",
                "resultat",
                "calendrier",
                "bourse",
                "scolarite",
                "contact",
                "attestation",
                "note",
                "connexion",
                "password",
                "login",
                "cours",
                "module",
            )
        ):
            base -= 10
        if path.endswith(".pdf"):
            base += 8
        return max(0, base + (depth * 4))

    def _enqueue(url: str, depth: int) -> None:
        nonlocal done
        accepted = should_accept_url(url)
        if not accepted or depth > int(limits["max_depth"]):
            return
        info = match_source_rule(accepted)
        allowed = get_profile(config.mode)["allowed_priorities"]
        if str(info["source_priority"]) not in allowed:
            return
        host = urlparse(accepted).netloc
        domain_bucket = _domain_bucket(host)
        subdomain_bucket = _subdomain_bucket(host)
        if domain_counts[domain_bucket] >= int(limits["max_urls_per_domain"]):
            return
        if subdomain_counts[subdomain_bucket] >= int(limits["max_urls_per_subdomain"]):
            return
        if accepted in visited:
            return
        visited.add(accepted)
        domain_counts[domain_bucket] += 1
        subdomain_counts[subdomain_bucket] += 1
        queue_items.put((_priority_for_url(accepted, depth), depth, accepted))
        if len(visited) >= int(limits["max_total_urls"]):
            done = True

    for seed in seeds:
        _enqueue(seed, 0)

    for corpus in ("main", "archive"):
        (RUNTIME.rag_raw_main_dir if corpus == "main" else RUNTIME.rag_raw_archive_dir).mkdir(parents=True, exist_ok=True)

    def worker() -> None:
        nonlocal active_workers, done
        while True:
            with condition:
                while queue_items.empty() and active_workers > 0 and not done:
                    condition.wait()
                if queue_items.empty() and (active_workers == 0 or done):
                    return
                _, depth, current_url = queue_items.get()
                active_workers += 1

            try:
                previous = state_items.get(current_url, {})
                if not _is_due_for_refresh(previous, refresh_days):
                    metrics["unchanged"] += 1
                    continue

                download = _download_document(current_url, depth, js_usage=js_usage, js_lock=js_lock)
                if not download:
                    metrics["reject"] += 1
                    report_rows.append(
                        {
                            "url": current_url,
                            "status": "rejected",
                            "decision_reason": "download_failed",
                            "quality_issue": "empty_after_static",
                            "saved_at": now_iso(),
                        }
                    )
                    state_items[current_url] = {
                        **previous,
                        "url": current_url,
                        "status": "rejected",
                        "decision_reason": "download_failed",
                        "last_checked_at": now_iso(),
                    }
                    continue

                quality = download["quality"]
                decision = decide_document(
                    url=current_url,
                    quality_score=int(quality["score"]),
                    keyword_hits=list(quality.get("keyword_hits", [])),
                    depth=depth,
                    extension=str(download["extension"]),
                    mode=config.mode,
                )
                source_rule = match_source_rule(current_url)
                text_hash = str(quality.get("text_content_hash") or "")
                previous_hash = str(previous.get("content_hash") or "")
                status = "new" if not previous else "updated"
                if previous_hash and previous_hash == download["content_hash"] and previous.get("text_hash") == text_hash:
                    metrics["unchanged"] += 1
                    status = "unchanged"

                file_path = None
                if status != "unchanged":
                    _cleanup_previous_artifact(previous)

                if status != "unchanged" and _should_keep_for_target(config.target_corpus, decision):
                    file_path = _save_document(download["content"], current_url, decision.corpus_target, str(download["extension"]))
                    metadata_payload = {
                        "url": current_url,
                        "file": str(file_path),
                        "depth": depth,
                        "hash": download["content_hash"],
                        "text_hash": text_hash,
                        "content_type": download["content_type"],
                        "is_html": str(download["extension"]) in {".html", ".htm"},
                        "download_quality_score": quality.get("score", 0),
                        "download_quality_reason": quality.get("reason", ""),
                        "download_keyword_hits": quality.get("keyword_hits", []),
                        "download_quality_metrics": quality.get("metrics", {}),
                        "corpus_target": decision.corpus_target,
                        "source_priority": decision.source_priority,
                        "document_category": decision.document_category,
                        "quality_score_initial": decision.quality_score_initial,
                        "business_relevance_score": decision.business_relevance_score,
                        "decision_reason": decision.decision_reason,
                        "is_premium": decision.is_premium,
                        "etag": download["headers"].get("ETag", ""),
                        "last_modified": download["headers"].get("Last-Modified", ""),
                        "ingestion_mode": config.mode,
                        "saved_at": now_iso(),
                        "render_mode": download.get("render_mode", "static"),
                        "js_dependent": bool(quality.get("js_dependent", False)),
                        "render_success": bool(download.get("render_success", False)),
                        "render_error": str(download.get("render_error", "") or ""),
                        "render_selector": str(download.get("selector_used", "") or ""),
                        "page_kind": str(quality.get("page_kind", "landing") or "landing"),
                        "intent": list(quality.get("intent", [])),
                        "freshness_score": float(quality.get("freshness_score", 0.0) or 0.0),
                        "quality_issue": str(quality.get("quality_issue", "") or ""),
                        "source_rule_name": str(source_rule.get("rule_name", "")),
                        "title": str(quality.get("title", "") or ""),
                        "headings": list(quality.get("headings", [])),
                        "official_links": list(quality.get("links", [])),
                    }
                    _save_document_metadata(decision.corpus_target, file_path, metadata_payload)
                    metrics["downloaded"] += 1
                    metrics[decision.corpus_target] += 1
                    metrics["categories"][decision.document_category] += 1
                    metrics["priorities"][decision.source_priority] += 1
                    metrics["services"][metadata_payload["source_rule_name"] or "unknown"] += 1
                    metrics["page_kinds"][metadata_payload["page_kind"]] += 1
                    metrics["render_modes"][metadata_payload["render_mode"]] += 1
                    for intent in metadata_payload["intent"]:
                        metrics["intents"][intent] += 1
                    if metadata_payload["quality_issue"]:
                        metrics["quality_issues"][metadata_payload["quality_issue"]] += 1
                    metrics["documents"].append(metadata_payload)
                    report_rows.append(metadata_payload)
                else:
                    report_rows.append(
                        {
                            "url": current_url,
                            "status": "rejected" if decision.corpus_target == "reject" else "skipped",
                            "decision_reason": decision.decision_reason,
                            "quality_issue": str(quality.get("quality_issue", "") or ""),
                            "page_kind": str(quality.get("page_kind", "landing") or "landing"),
                            "intent": list(quality.get("intent", [])),
                            "render_mode": download.get("render_mode", "static"),
                            "render_success": bool(download.get("render_success", False)),
                            "render_error": str(download.get("render_error", "") or ""),
                            "saved_at": now_iso(),
                        }
                    )
                    if decision.corpus_target == "reject":
                        metrics["reject"] += 1
                        if quality.get("quality_issue"):
                            metrics["quality_issues"][str(quality.get("quality_issue"))] += 1
                    else:
                        metrics["skipped"] += 1

                state_items[current_url] = {
                    "url": current_url,
                    "content_hash": download["content_hash"],
                    "text_hash": text_hash,
                    "etag": download["headers"].get("ETag", ""),
                    "last_modified": download["headers"].get("Last-Modified", ""),
                    "corpus_target": decision.corpus_target,
                    "decision_reason": decision.decision_reason,
                    "status": status if decision.corpus_target != "reject" else "rejected",
                    "last_checked_at": now_iso(),
                    "file_path": str(file_path or previous.get("file_path", "")),
                    "mode": config.mode,
                }

                if str(download["extension"]) in {".html", ".htm"} and depth < int(limits["max_depth"]):
                    for discovered in extract_links(download["content"], current_url):
                        _enqueue(discovered, depth + 1)
            finally:
                with condition:
                    active_workers -= 1
                    queue_items.task_done()
                    condition.notify_all()

    threads = []
    worker_count = min(_env_int("RAG_INGESTION_THREADS", 8), 8)
    for _ in range(worker_count):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    state["last_run"] = {
        "mode": config.mode,
        "target_corpus": config.target_corpus,
        "started_at": now_iso(),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"documents", "categories", "priorities", "services", "intents", "page_kinds", "quality_issues", "render_modes"}
        },
    }
    _save_state(state)
    _save_legacy_metadata(report_rows)

    report = {
        "generated_at": now_iso(),
        "mode": config.mode,
        "target_corpus": config.target_corpus,
        "seed_count": len(seeds),
        "main_count": metrics["main"],
        "archive_count": metrics["archive"],
        "reject_count": metrics["reject"],
        "unchanged_count": metrics["unchanged"],
        "skipped_count": metrics["skipped"],
        "downloaded_count": metrics["downloaded"],
        "category_distribution": dict(metrics["categories"]),
        "priority_distribution": dict(metrics["priorities"]),
        "service_coverage": dict(metrics["services"]),
        "intent_coverage": dict(metrics["intents"]),
        "page_kind_distribution": dict(metrics["page_kinds"]),
        "quality_issue_distribution": dict(metrics["quality_issues"]),
        "render_mode_distribution": dict(metrics["render_modes"]),
        "js_render_count": js_usage.get("count", 0),
        "documents": metrics["documents"],
        "audit_documents": report_rows,
    }
    report_path = RUNTIME.rag_reports_dir / f"ingestion_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _save_json_atomic(report_path, report)

    return {
        "status": "ok",
        "step": "ingestion",
        "mode": config.mode,
        "target_corpus": config.target_corpus,
        "seeds": seeds,
        "documents_collected": metrics["downloaded"],
        "main_count": metrics["main"],
        "archive_count": metrics["archive"],
        "reject_count": metrics["reject"],
        "unchanged_count": metrics["unchanged"],
        "skipped_count": metrics["skipped"],
        "js_render_count": js_usage.get("count", 0),
        "documents": metrics["documents"],
        "report_path": str(report_path),
    }
