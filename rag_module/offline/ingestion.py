# rag_module/ingestion.py
import os
import re
import json
import hashlib
import logging
import queue
import string
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

try:
    from .structured_parser import extract_main_text
    from ..audit.offline_pipeline_report import update_offline_pipeline_report
except ImportError:  # pragma: no cover
    from rag_module.offline.structured_parser import extract_main_text
    from rag_module.audit.offline_pipeline_report import update_offline_pipeline_report

from .ingestion_utils import *
def load_ingestion_state() -> Dict:
    raw = _load_json(INGESTION_STATE_PATH)
    if isinstance(raw, dict):
        urls = raw.get("urls", {})
        if isinstance(urls, dict):
            normalized = {}
            for key, value in urls.items():
                if isinstance(key, str) and isinstance(value, dict):
                    normalized[key] = value
            return {"version": 2, "urls": normalized}
    return {"version": 2, "urls": {}}


def save_ingestion_state(state: Dict) -> None:
    _save_json_atomic(INGESTION_STATE_PATH, state)


def _update_state_entry(
    state: Dict,
    canonical_url: str,
    *,
    url: Optional[str] = None,
    domain: Optional[str] = None,
    last_status: Optional[str] = None,
    last_http_code: Optional[int] = None,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    content_hash: Optional[str] = None,
    text_content_hash: Optional[str] = None,
    file_path: Optional[str] = None,
    last_fetched_at: Optional[str] = None,
    quality_score: Optional[int] = None,
    quality_reason: Optional[str] = None,
    skip_reason: Optional[str] = None,
    crawl_depth: Optional[int] = None,
) -> None:
    entry = dict(state.get("urls", {}).get(canonical_url, {}))
    now_value = now_iso()
    
    # Initialize all required keys if missing
    for key in [
        "url", "canonical_url", "domain", "last_status", "last_http_code",
        "etag", "last_modified", "content_hash", "text_content_hash", "file_path",
        "first_seen_at", "last_seen_at", "last_fetched_at",
        "quality_score", "quality_reason", "skip_reason", "crawl_depth"
    ]:
        if key not in entry:
            entry[key] = None if key in ["last_http_code", "quality_score", "crawl_depth"] else ""

    if not entry.get("first_seen_at"):
        entry["first_seen_at"] = now_value
    entry["last_seen_at"] = now_value
    entry["canonical_url"] = canonical_url

    if url is not None:
        entry["url"] = url
    if domain is not None:
        entry["domain"] = domain
    if last_status is not None:
        entry["last_status"] = last_status
    if last_http_code is not None:
        entry["last_http_code"] = int(last_http_code)
    if etag is not None:
        entry["etag"] = etag
    if last_modified is not None:
        entry["last_modified"] = last_modified
    if content_hash is not None:
        entry["content_hash"] = content_hash
    if text_content_hash is not None:
        entry["text_content_hash"] = text_content_hash
    if file_path is not None:
        entry["file_path"] = file_path
    if last_fetched_at is not None:
        entry["last_fetched_at"] = last_fetched_at
    if quality_score is not None:
        entry["quality_score"] = int(quality_score)
    if quality_reason is not None:
        entry["quality_reason"] = quality_reason
    if skip_reason is not None:
        entry["skip_reason"] = skip_reason
    if crawl_depth is not None:
        entry["crawl_depth"] = int(crawl_depth)

    state.setdefault("urls", {})[canonical_url] = entry


def _build_ingestion_metadata_row(
    *,
    url: str,
    canonical_url: str,
    domain: str,
    subdomain: str = "",
    depth: int,
    status: str,
    skip_reason: str = "",
    file_path: str = "",
    content_type: str = "",
    content_hash: str = "",
    text_content_hash: str = "",
    etag: str = "",
    last_modified: str = "",
    fetched_at: str = "",
    quality_score: int = 0,
    quality_reason: str = "",
    quality_metrics: Optional[Dict] = None,
    keyword_hits: Optional[List[str]] = None,
) -> Dict:
    return {
        "url": url,
        "canonical_url": canonical_url,
        "domain": domain,
        "subdomain": subdomain,
        "depth": int(depth),
        "file": file_path,
        "content_type": content_type,
        "content_hash": content_hash,
        "text_content_hash": text_content_hash,
        "etag": etag,
        "last_modified": last_modified,
        "fetched_at": fetched_at or now_iso(),
        "download_quality_score": int(quality_score or 0),
        "download_quality_reason": str(quality_reason or ""),
        "download_quality_metrics": quality_metrics or {},
        "download_keyword_hits": list(keyword_hits or []),
        "skip_reason": skip_reason,
        "status": status,
    }


def _metadata_skip_row(url: str, canonical_url: str, depth: int, reason: str, domain: str,
                       quality_score: int = 0, quality_reason: str = "",
                       content_hash: str = "", text_content_hash: str = "",
                       etag: str = "", last_modified: str = "", content_type: str = "") -> Dict:
    return {
        **_build_ingestion_metadata_row(
            url=url,
            canonical_url=canonical_url,
            domain=domain,
            depth=depth,
            status="skipped",
            skip_reason=reason,
            content_type=content_type,
            content_hash=content_hash,
            text_content_hash=text_content_hash,
            etag=etag,
            last_modified=last_modified,
            quality_score=quality_score,
            quality_reason=quality_reason,
        ),
    }


def _should_refresh_weekly(previous_entry: Dict) -> bool:
    if REFRESH_MODE != "weekly":
        return True
    fetched_at = str(previous_entry.get("last_fetched_at") or "").strip()
    if not fetched_at:
        return True
    try:
        dt = datetime.fromisoformat(fetched_at)
    except Exception:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt <= (datetime.now(timezone.utc) - timedelta(days=REFRESH_DAYS))


def _conditional_headers(previous_entry: Dict) -> Dict[str, str]:
    headers = dict(HEADERS)
    if not INCREMENTAL_FETCH:
        return headers
    etag = str(previous_entry.get("etag") or "").strip()
    last_modified = str(previous_entry.get("last_modified") or "").strip()
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _extract_sitemap_locs(xml_bytes: bytes) -> List[str]:
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []

    locs: List[str] = []
    for elem in root.iter():
        if elem.tag.lower().endswith("loc") and elem.text:
            locs.append(elem.text.strip())
    return locs


def _discover_seed_urls(seed_url: str) -> List[str]:
    if not ENABLE_SITEMAP_DISCOVERY:
        return []

    parsed = urlparse(seed_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_candidates = {urljoin(root, "/sitemap.xml")}

    try:
        robots_response = requests.get(urljoin(root, "/robots.txt"), headers=HEADERS, timeout=12)
        if robots_response.status_code == 200:
            for line in robots_response.text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("sitemap:"):
                    candidate = stripped.split(":", 1)[1].strip()
                    if candidate:
                        sitemap_candidates.add(candidate)
    except Exception:
        pass

    discovered: List[str] = []
    sitemap_queue = list(sitemap_candidates)
    seen_sitemaps: Set[str] = set()
    scanned_sitemaps = 0

    while sitemap_queue and scanned_sitemaps < MAX_SITEMAPS_PER_SEED and len(discovered) < MAX_DISCOVERY_PER_SEED:
        sitemap_url = clean_url(sitemap_queue.pop(0))
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)
        scanned_sitemaps += 1

        try:
            response = requests.get(sitemap_url, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                continue
            locs = _extract_sitemap_locs(response.content[:MAX_SITEMAP_URLS * 120])
        except Exception:
            continue

        for loc in locs:
            if len(discovered) >= MAX_DISCOVERY_PER_SEED:
                break
            candidate = should_accept_url(loc)
            if not candidate:
                continue
            if candidate.endswith(".xml"):
                if candidate not in seen_sitemaps:
                    sitemap_queue.append(candidate)
                continue
            discovered.append(candidate)

    return list(dict.fromkeys(discovered))


def _load_previous_metadata_rows() -> List[Dict]:
    payload = _load_json(META_PATH)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _increment_counter(stats: Dict, key: str, amount: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + amount


# ==============================
# LOCKS MULTITHREADING
# ==============================
visited_lock = threading.Lock()
results_lock = threading.Lock()

def save_metadata_safely(results_list):
    try:
        _save_json_atomic(META_PATH, {"version": 2, "rows": results_list})
    except Exception as exc:
        logger.error("Erreur lors de la sauvegarde metadata: %s", exc)

# ==============================
# DOWNLOAD
# ==============================
def download(
    url: str,
    depth: int,
    seen_hashes: Set[str],
    near_dup_signatures: List[int],
    state: Dict,
    state_lock: threading.Lock,
    dedup_lock: threading.Lock,
    stats: Dict,
    stats_lock: threading.Lock,
) -> Optional[Dict]:
    canonical_url = clean_url(url)
    host = normalize_host(urlparse(canonical_url).netloc)
    domain = domain_bucket(host)
    subdomain = subdomain_bucket(host)

    with state_lock:
        previous_entry = dict(state.get("urls", {}).get(canonical_url, {}))
        _update_state_entry(
            state,
            canonical_url,
            url=url,
            domain=domain,
            crawl_depth=depth,
            last_status="seen",
            skip_reason="",
        )

    if not _should_refresh_weekly(previous_entry):
        with stats_lock:
            _increment_counter(stats, "refresh_skipped")
        with state_lock:
            _update_state_entry(
                state,
                canonical_url,
                url=url,
                domain=domain,
                last_status="refresh_skipped",
                last_http_code=int(previous_entry.get("last_http_code") or 0),
                skip_reason="refresh_window_not_elapsed",
                crawl_depth=depth,
            )
        if RECORD_SKIPPED_IN_METADATA:
            return _metadata_skip_row(
                url, canonical_url, depth, "refresh_window_not_elapsed", domain,
                etag=str(previous_entry.get("etag") or ""),
                last_modified=str(previous_entry.get("last_modified") or "")
            )
        return None

    for _ in range(RETRIES):
        try:
            timeout = TIMEOUT if depth < 2 else 10
            response = requests.get(url, headers=_conditional_headers(previous_entry), timeout=timeout)

            if response.status_code == 304:
                with stats_lock:
                    _increment_counter(stats, "not_modified")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="not_modified",
                        last_http_code=304,
                        skip_reason="not_modified",
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(
                        url, canonical_url, depth, "not_modified", domain,
                        etag=str(previous_entry.get("etag") or ""),
                        last_modified=str(previous_entry.get("last_modified") or "")
                    )
                return None

            if response.status_code != 200:
                with stats_lock:
                    _increment_counter(stats, "http_errors")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="http_error",
                        last_http_code=response.status_code,
                        skip_reason=f"http_{response.status_code}",
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(url, canonical_url, depth, f"http_{response.status_code}", domain)
                return None

            content = response.content
            content_type = response.headers.get("Content-Type", "")
            lowered_content_type = content_type.lower()

            is_binary_document = any(
                token in lowered_content_type
                for token in (
                    "application/pdf",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/octet-stream",
                )
            )
            if not is_binary_document and (len(content) < 1200 or b"<script" in content):
                pw_content = fetch_with_playwright(url)
                if pw_content:
                    content = pw_content
                    content_type = "text/html; charset=utf-8"

            ext = infer_extension(
                url,
                content_type,
                content=content,
                content_disposition=response.headers.get("Content-Disposition", ""),
            )
            quality = compute_download_quality(
                url=url,
                depth=depth,
                content=content,
                content_type=content_type,
                ext=ext,
            )
            if not quality["keep"]:
                with stats_lock:
                    _increment_counter(stats, "skipped_quality")
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="skipped_quality",
                        last_http_code=response.status_code,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or ""),
                        skip_reason=str(quality.get("reason") or "quality_rejected"),
                        crawl_depth=depth,
                    )
                if RECORD_SKIPPED_IN_METADATA:
                    return _metadata_skip_row(
                        url,
                        canonical_url,
                        depth,
                        str(quality.get("reason") or "quality_rejected"),
                        domain,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or "")
                    )
                return None

            content_hash = compute_hash(content)
            text_content_hash = str(quality.get("text_content_hash") or "").strip()
            dedup_hash = text_content_hash or content_hash
            with dedup_lock:
                if dedup_hash in seen_hashes:
                    with stats_lock:
                        _increment_counter(stats, "skipped_dup_exact")
                    with state_lock:
                        _update_state_entry(
                            state,
                            canonical_url,
                            url=url,
                            domain=domain,
                            last_status="skipped_dup_exact",
                            last_http_code=response.status_code,
                            content_hash=content_hash,
                            text_content_hash=text_content_hash,
                            quality_score=int(quality.get("score") or 0),
                            quality_reason=str(quality.get("reason") or ""),
                            skip_reason="duplicate_exact",
                            crawl_depth=depth,
                        )
                    if RECORD_SKIPPED_IN_METADATA:
                        return _metadata_skip_row(
                            url, canonical_url, depth, "duplicate_exact", domain,
                            quality_score=int(quality.get("score") or 0),
                            quality_reason=str(quality.get("reason") or ""),
                            content_hash=content_hash,
                            text_content_hash=text_content_hash,
                            content_type=content_type,
                        )
                    return None

                if ENABLE_NEAR_DUP_INGESTION and ext in {".html", ".htm", ".txt", ".md"}:
                    preview_text = str(quality.get("preview_text") or "")
                    signature = _simhash64(_tokenize_for_simhash(preview_text[:8000]))
                    if signature:
                        for existing in near_dup_signatures:
                            if _hamming_distance(signature, existing) <= NEAR_DUP_SIMHASH_DISTANCE:
                                with stats_lock:
                                    _increment_counter(stats, "skipped_dup_near")
                                with state_lock:
                                    _update_state_entry(
                                        state,
                                        canonical_url,
                                        url=url,
                                        domain=domain,
                                        last_status="skipped_dup_near",
                                        last_http_code=response.status_code,
                                        content_hash=content_hash,
                                        text_content_hash=text_content_hash,
                                        quality_score=int(quality.get("score") or 0),
                                        quality_reason=str(quality.get("reason") or ""),
                                        skip_reason="duplicate_near",
                                        crawl_depth=depth,
                                    )
                                if RECORD_SKIPPED_IN_METADATA:
                                    return _metadata_skip_row(
                                        url, canonical_url, depth, "duplicate_near", domain,
                                        quality_score=int(quality.get("score") or 0),
                                        quality_reason=str(quality.get("reason") or ""),
                                        content_hash=content_hash,
                                        text_content_hash=text_content_hash,
                                        content_type=content_type,
                                    )
                                return None
                        near_dup_signatures.append(signature)

                seen_hashes.add(dedup_hash)

            filename = generate_filename(url, ext)
            path = save_file(content, filename)
            fetched_at = now_iso()
            etag = str(response.headers.get("ETag") or "").strip()
            last_modified = str(response.headers.get("Last-Modified") or "").strip()

            with stats_lock:
                _increment_counter(stats, "downloaded")
                stats["per_domain"][domain] = int(stats["per_domain"].get(domain, 0)) + 1
                stats["per_subdomain"][subdomain] = int(stats["per_subdomain"].get(subdomain, 0)) + 1

            with state_lock:
                _update_state_entry(
                    state,
                    canonical_url,
                    url=url,
                    domain=domain,
                    last_status="downloaded",
                    last_http_code=response.status_code,
                    etag=etag,
                    last_modified=last_modified,
                    content_hash=content_hash,
                    text_content_hash=text_content_hash,
                    file_path=path,
                    last_fetched_at=fetched_at,
                    quality_score=int(quality.get("score") or 0),
                    quality_reason=str(quality.get("reason") or ""),
                    skip_reason="",
                    crawl_depth=depth,
                )

            time.sleep(0.25)
            row = _build_ingestion_metadata_row(
                url=url,
                canonical_url=canonical_url,
                domain=domain,
                subdomain=subdomain,
                depth=depth,
                status="downloaded",
                skip_reason="",
                file_path=path,
                content_type=content_type,
                content_hash=content_hash,
                text_content_hash=text_content_hash,
                etag=etag,
                last_modified=last_modified,
                fetched_at=fetched_at,
                quality_score=int(quality.get("score", 0)),
                quality_reason=str(quality.get("reason", "")),
                quality_metrics=quality.get("metrics", {}),
                keyword_hits=quality.get("keyword_hits", []),
            )
            row["hash"] = dedup_hash
            row["is_html"] = ext in {".html", ".htm"}
            return row

        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            with stats_lock:
                _increment_counter(stats, "download_errors")
            time.sleep(1)

    with state_lock:
        _update_state_entry(
            state,
            canonical_url,
            url=url,
            domain=domain,
            last_status="download_error",
            skip_reason="download_error",
            crawl_depth=depth,
        )
    if RECORD_SKIPPED_IN_METADATA:
        return _metadata_skip_row(url, canonical_url, depth, "download_error", domain)
    return None


# ==============================
# CRAWLER
# ==============================
def _crawl_with_legacy_fetch(seeds):
    q = queue.PriorityQueue()
    visited: Set[str] = set()
    metadata_rows: List[Dict] = []
    downloaded_rows: List[Dict] = []

    state = load_ingestion_state()
    state_lock = threading.Lock()
    dedup_lock = threading.Lock()
    queue_lock = threading.Lock()
    stats_lock = threading.Lock()

    seen_hashes: Set[str] = set()
    near_dup_signatures: List[int] = []

    for entry in state.get("urls", {}).values():
        if not isinstance(entry, dict):
            continue
        content_hash = str(entry.get("content_hash") or "").strip()
        text_content_hash = str(entry.get("text_content_hash") or "").strip()
        if content_hash:
            seen_hashes.add(content_hash)
        if text_content_hash:
            seen_hashes.add(text_content_hash)

    stats: Dict = {
        "started_at": now_iso(),
        "crawled": 0,
        "downloaded": 0,
        "not_modified": 0,
        "refresh_skipped": 0,
        "skipped_quality": 0,
        "skipped_dup_exact": 0,
        "skipped_dup_near": 0,
        "download_errors": 0,
        "http_errors": 0,
        "quota_skipped_domain": 0,
        "quota_skipped_subdomain": 0,
        "per_domain": {},
        "per_subdomain": {},
        "seeds_count": len(seeds or []),
    }
    queued_per_domain: Dict[str, int] = defaultdict(int)

    def put_url(url_value: str, depth_value: int) -> None:
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        with queue_lock:
            pressure = int(stats["per_domain"].get(domain, 0)) + int(queued_per_domain.get(domain, 0))
            priority = score_url(url_value, depth_value) + (pressure * DOMAIN_PRIORITY_STEP)
            q.put((priority, depth_value, url_value))
            queued_per_domain[domain] += 1

    def pop_url() -> Tuple[int, int, str]:
        priority, depth_value, url_value = q.get()
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        with queue_lock:
            queued_per_domain[domain] = max(0, int(queued_per_domain.get(domain, 0)) - 1)
        return priority, depth_value, url_value

    def can_fetch(url_value: str) -> Tuple[bool, str]:
        host = normalize_host(urlparse(url_value).netloc)
        domain = domain_bucket(host)
        subdomain = subdomain_bucket(host)
        with stats_lock:
            if int(stats["per_domain"].get(domain, 0)) >= MAX_URLS_PER_DOMAIN:
                return False, "domain_quota_exceeded"
            if int(stats["per_subdomain"].get(subdomain, 0)) >= MAX_URLS_PER_SUBDOMAIN:
                return False, "subdomain_quota_exceeded"
        return True, ""

    accepted_seeds: List[str] = []
    for seed in seeds:
        candidate = should_accept_url(seed)
        if candidate:
            accepted_seeds.append(candidate)

    discovered_urls: List[str] = []
    if ENABLE_SITEMAP_DISCOVERY:
        for seed in accepted_seeds:
            discovered_urls.extend(_discover_seed_urls(seed))
        discovered_urls = discovered_urls[:MAX_SITEMAP_URLS]

    for url in list(dict.fromkeys([*accepted_seeds, *discovered_urls])):
        with visited_lock:
            if url in visited:
                continue
            visited.add(url)
        put_url(url, 0)

    active_workers = 0
    condition = threading.Condition()

    def worker():
        nonlocal active_workers
        while True:
            with condition:
                while q.empty() and active_workers > 0 and int(stats.get("downloaded", 0)) < MAX_TOTAL_URLS:
                    condition.wait()
                
                if int(stats.get("downloaded", 0)) >= MAX_TOTAL_URLS:
                    return
                if q.empty() and active_workers == 0:
                    return
                
                _, depth, url = pop_url()
                active_workers += 1

            with stats_lock:
                _increment_counter(stats, "crawled")

            can_continue, quota_reason = can_fetch(url)
            if not can_continue:
                canonical_url = clean_url(url)
                domain = domain_bucket(urlparse(canonical_url).netloc)
                with state_lock:
                    _update_state_entry(
                        state,
                        canonical_url,
                        url=url,
                        domain=domain,
                        last_status="quota_skipped",
                        skip_reason=quota_reason,
                        crawl_depth=depth,
                    )
                with stats_lock:
                    if quota_reason == "domain_quota_exceeded":
                        _increment_counter(stats, "quota_skipped_domain")
                    else:
                        _increment_counter(stats, "quota_skipped_subdomain")
                if RECORD_SKIPPED_IN_METADATA:
                    with results_lock:
                        metadata_rows.append(_metadata_skip_row(url, canonical_url, depth, quota_reason, domain))

                with condition:
                    active_workers -= 1
                    q.task_done()
                    condition.notify_all()
                continue

            # Téléchargement hors du verrou principal
            res = download(
                url=url,
                depth=depth,
                seen_hashes=seen_hashes,
                near_dup_signatures=near_dup_signatures,
                state=state,
                state_lock=state_lock,
                dedup_lock=dedup_lock,
                stats=stats,
                stats_lock=stats_lock,
            )
            
            if res:
                is_downloaded = str(res.get("status", "")) == "downloaded"
                with results_lock:
                    metadata_rows.append(res)
                    if is_downloaded:
                        downloaded_rows.append(res)
                    # Sauvegarde périodique (Checkpointing)
                    if len(downloaded_rows) % 100 == 0 and downloaded_rows:
                        save_metadata_safely(metadata_rows)
                        save_ingestion_state(state)
                        logger.info("Checkpoint: %s documents téléchargés.", len(downloaded_rows))

                if is_downloaded and depth < MAX_DEPTH and res.get("is_html"):
                    links = []
                    try:
                        with open(res["file"], "rb") as fp:
                            links = extract_links(fp.read(), url)
                    except Exception as exc:
                        logger.warning("Link extraction failed for %s: %s", res["file"], exc)

                    for link in links:
                        with visited_lock:
                            if link not in visited:
                                visited.add(link)
                                added = True
                            else:
                                added = False
                        
                        if added:
                            put_url(link, depth + 1)

            with condition:
                active_workers -= 1
                q.task_done()
                condition.notify_all()
                if int(stats.get("downloaded", 0)) % 100 == 0 and int(stats.get("downloaded", 0)) > 0:
                    logger.info("Progress: %s/%s", stats.get("downloaded", 0), MAX_TOTAL_URLS)

    # Démarrage des workers
    threads = []
    for _ in range(INGESTION_THREADS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    try:
        # Attente de la fin de tous les threads
        for t in threads:
            t.join()

        # Sauvegarde finale
        save_metadata_safely(metadata_rows)
        save_ingestion_state(state)

        report_payload = {
            "started_at": stats.get("started_at"),
            "finished_at": now_iso(),
            "settings": {
                "max_total_urls": MAX_TOTAL_URLS,
                "max_depth": MAX_DEPTH,
                "ingestion_threads": INGESTION_THREADS,
                "max_urls_per_domain": MAX_URLS_PER_DOMAIN,
                "max_urls_per_subdomain": MAX_URLS_PER_SUBDOMAIN,
                "incremental_fetch": INCREMENTAL_FETCH,
                "refresh_mode": REFRESH_MODE,
                "near_dup_enabled": ENABLE_NEAR_DUP_INGESTION,
                "near_dup_simhash_distance": NEAR_DUP_SIMHASH_DISTANCE,
                "min_download_quality_score": MIN_DOWNLOAD_QUALITY_SCORE,
                "sitemap_discovery": ENABLE_SITEMAP_DISCOVERY,
            },
            "metrics": {
                "crawled": int(stats.get("crawled", 0)),
                "downloaded": int(stats.get("downloaded", 0)),
                "skipped_quality": int(stats.get("skipped_quality", 0)),
                "skipped_dup_exact": int(stats.get("skipped_dup_exact", 0)),
                "skipped_dup_near": int(stats.get("skipped_dup_near", 0)),
                "not_modified": int(stats.get("not_modified", 0)),
                "refresh_skipped": int(stats.get("refresh_skipped", 0)),
                "download_errors": int(stats.get("download_errors", 0)),
                "http_errors": int(stats.get("http_errors", 0)),
                "quota_skipped_domain": int(stats.get("quota_skipped_domain", 0)),
                "quota_skipped_subdomain": int(stats.get("quota_skipped_subdomain", 0)),
            },
            "per_domain": dict(
                sorted(stats.get("per_domain", {}).items(), key=lambda item: item[1], reverse=True)
            ),
            "per_subdomain_top20": dict(
                sorted(stats.get("per_subdomain", {}).items(), key=lambda item: item[1], reverse=True)[:20]
            ),
            "metadata_rows_count": len(metadata_rows),
            "downloaded_rows_count": len(downloaded_rows),
        }
        report_path = str(update_offline_pipeline_report("ingestion", report_payload))
        logger.info("Ingestion report updated: %s", report_path)
        return downloaded_rows
    finally:
        cleanup_playwright_resources()


def crawl(seeds):
    backend = (INGESTION_BACKEND or "scrapy").lower()
    if backend == "scrapy":
        try:
            from .scrapy_ingestion import crawl_with_scrapy

            return crawl_with_scrapy(seeds or DEFAULT_SEEDS)
        except Exception as exc:
            logger.warning("Backend Scrapy indisponible, fallback legacy requests actif: %s", exc)
    return _crawl_with_legacy_fetch(seeds or DEFAULT_SEEDS)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("Crawler RAG demarre")
    crawl(DEFAULT_SEEDS)