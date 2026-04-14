import logging
from collections import defaultdict
from typing import Dict, List, Set
from urllib.parse import urlparse

from . import ingestion as legacy

logger = logging.getLogger(__name__)


class _ScrapyCollector:
    def __init__(self) -> None:
        self.state = legacy.load_ingestion_state()
        self.metadata_rows: List[Dict] = []
        self.downloaded_rows: List[Dict] = []
        self.seen_hashes: Set[str] = set()
        self.near_dup_signatures: List[int] = []
        self.visited: Set[str] = set()
        self.stats: Dict = {
            "started_at": legacy.now_iso(),
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
        }
        for entry in self.state.get("urls", {}).values():
            if not isinstance(entry, dict):
                continue
            content_hash = str(entry.get("content_hash") or "").strip()
            if content_hash:
                self.seen_hashes.add(content_hash)

    def can_fetch(self, url: str) -> tuple[bool, str]:
        host = legacy.normalize_host(urlparse(url).netloc)
        domain = legacy.domain_bucket(host)
        subdomain = legacy.subdomain_bucket(host)
        if int(self.stats["per_domain"].get(domain, 0)) >= legacy.MAX_URLS_PER_DOMAIN:
            return False, "domain_quota_exceeded"
        if int(self.stats["per_subdomain"].get(subdomain, 0)) >= legacy.MAX_URLS_PER_SUBDOMAIN:
            return False, "subdomain_quota_exceeded"
        return True, ""


def crawl_with_scrapy(seeds: List[str]) -> List[Dict]:
    try:
        import scrapy
        from scrapy.crawler import CrawlerProcess
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scrapy n'est pas installe dans l'environnement courant.") from exc

    collector = _ScrapyCollector()
    accepted_seeds = [candidate for seed in (seeds or legacy.DEFAULT_SEEDS) if (candidate := legacy.should_accept_url(seed))]
    discovered_urls: List[str] = []
    if legacy.ENABLE_SITEMAP_DISCOVERY:
        for seed in accepted_seeds:
            discovered_urls.extend(legacy._discover_seed_urls(seed))
        discovered_urls = discovered_urls[: legacy.MAX_SITEMAP_URLS]

    queued_per_domain: Dict[str, int] = defaultdict(int)

    class UCASpider(scrapy.Spider):
        name = "uca_hybrid"
        custom_settings = {
            "DOWNLOAD_TIMEOUT": legacy.TIMEOUT,
            "CONCURRENT_REQUESTS": legacy.INGESTION_THREADS,
            "RETRY_TIMES": max(0, legacy.RETRIES - 1),
            "TELNETCONSOLE_ENABLED": False,
            "LOG_LEVEL": "ERROR",
            "ROBOTSTXT_OBEY": False,
        }

        def start_requests(self):
            for url in list(dict.fromkeys([*accepted_seeds, *discovered_urls])):
                if url in collector.visited:
                    continue
                collector.visited.add(url)
                host = legacy.normalize_host(urlparse(url).netloc)
                domain = legacy.domain_bucket(host)
                queued_per_domain[domain] += 1
                priority = -(legacy.score_url(url, 0) + (queued_per_domain[domain] * legacy.DOMAIN_PRIORITY_STEP))
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_page,
                    errback=self.handle_error,
                    dont_filter=True,
                    priority=priority,
                    meta={"depth_value": 0},
                    headers=legacy.HEADERS,
                )

        def handle_error(self, failure):
            request = failure.request
            url = str(getattr(request, "url", "") or "")
            canonical_url = legacy.clean_url(url)
            domain = legacy.domain_bucket(urlparse(canonical_url).netloc)
            legacy._update_state_entry(
                collector.state,
                canonical_url,
                url=url,
                domain=domain,
                last_status="download_error",
                skip_reason="download_error",
                crawl_depth=int(request.meta.get("depth_value", 0) or 0),
            )
            collector.stats["download_errors"] += 1
            collector.metadata_rows.append(legacy._metadata_skip_row(url, canonical_url, int(request.meta.get("depth_value", 0) or 0), "download_error", domain))

        def parse_page(self, response):
            depth = int(response.meta.get("depth_value", 0) or 0)
            url = str(response.url)
            canonical_url = legacy.clean_url(url)
            domain = legacy.domain_bucket(urlparse(canonical_url).netloc)
            subdomain = legacy.subdomain_bucket(urlparse(canonical_url).netloc)
            collector.stats["crawled"] += 1

            allowed, reason = collector.can_fetch(url)
            if not allowed:
                legacy._update_state_entry(
                    collector.state,
                    canonical_url,
                    url=url,
                    domain=domain,
                    last_status="quota_skipped",
                    skip_reason=reason,
                    crawl_depth=depth,
                )
                if reason == "domain_quota_exceeded":
                    collector.stats["quota_skipped_domain"] += 1
                else:
                    collector.stats["quota_skipped_subdomain"] += 1
                collector.metadata_rows.append(legacy._metadata_skip_row(url, canonical_url, depth, reason, domain))
                return

            content = bytes(response.body or b"")
            content_type = response.headers.get("Content-Type", b"").decode("latin-1", errors="replace")
            if b"<script" in content[:5000] and len(content) < 2000:
                pw_content = legacy.fetch_with_playwright(url)
                if pw_content:
                    content = pw_content
                    content_type = "text/html; charset=utf-8"

            ext = legacy.infer_extension(
                url,
                content_type,
                content=content,
                content_disposition=response.headers.get("Content-Disposition", b"").decode("latin-1", errors="replace"),
            )
            quality = legacy.compute_download_quality(url, depth, content, content_type, ext)
            if not quality["keep"]:
                collector.stats["skipped_quality"] += 1
                legacy._update_state_entry(
                    collector.state,
                    canonical_url,
                    url=url,
                    domain=domain,
                    last_status="skipped_quality",
                    last_http_code=int(response.status),
                    quality_score=int(quality.get("score") or 0),
                    quality_reason=str(quality.get("reason") or ""),
                    skip_reason=str(quality.get("reason") or "quality_rejected"),
                    crawl_depth=depth,
                )
                collector.metadata_rows.append(
                    legacy._metadata_skip_row(
                        url,
                        canonical_url,
                        depth,
                        str(quality.get("reason") or "quality_rejected"),
                        domain,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or ""),
                    )
                )
                return

            content_hash = legacy.compute_hash(content)
            if content_hash in collector.seen_hashes:
                collector.stats["skipped_dup_exact"] += 1
                collector.metadata_rows.append(
                    legacy._metadata_skip_row(
                        url,
                        canonical_url,
                        depth,
                        "duplicate_exact",
                        domain,
                        quality_score=int(quality.get("score") or 0),
                        quality_reason=str(quality.get("reason") or ""),
                        content_hash=content_hash,
                    )
                )
                return

            if legacy.ENABLE_NEAR_DUP_INGESTION and ext in {".html", ".htm", ".txt", ".md"}:
                preview_text = str(quality.get("preview_text") or "")
                signature = legacy._simhash64(legacy._tokenize_for_simhash(preview_text[:8000]))
                if signature:
                    if any(legacy._hamming_distance(signature, existing) <= legacy.NEAR_DUP_SIMHASH_DISTANCE for existing in collector.near_dup_signatures):
                        collector.stats["skipped_dup_near"] += 1
                        collector.metadata_rows.append(
                            legacy._metadata_skip_row(
                                url,
                                canonical_url,
                                depth,
                                "duplicate_near",
                                domain,
                                quality_score=int(quality.get("score") or 0),
                                quality_reason=str(quality.get("reason") or ""),
                                content_hash=content_hash,
                            )
                        )
                        return
                    collector.near_dup_signatures.append(signature)

            collector.seen_hashes.add(content_hash)
            filename = legacy.generate_filename(url, ext)
            path = legacy.save_file(content, filename)
            fetched_at = legacy.now_iso()
            etag = response.headers.get("ETag", b"").decode("latin-1", errors="replace")
            last_modified = response.headers.get("Last-Modified", b"").decode("latin-1", errors="replace")

            collector.stats["downloaded"] += 1
            collector.stats["per_domain"][domain] = int(collector.stats["per_domain"].get(domain, 0)) + 1
            collector.stats["per_subdomain"][subdomain] = int(collector.stats["per_subdomain"].get(subdomain, 0)) + 1
            legacy._update_state_entry(
                collector.state,
                canonical_url,
                url=url,
                domain=domain,
                last_status="downloaded",
                last_http_code=int(response.status),
                etag=etag,
                last_modified=last_modified,
                content_hash=content_hash,
                file_path=path,
                last_fetched_at=fetched_at,
                quality_score=int(quality.get("score") or 0),
                quality_reason=str(quality.get("reason") or ""),
                skip_reason="",
                crawl_depth=depth,
            )
            row = {
                "url": url,
                "canonical_url": canonical_url,
                "domain": domain,
                "subdomain": subdomain,
                "file": path,
                "depth": depth,
                "hash": content_hash,
                "content_hash": content_hash,
                "content_type": content_type,
                "is_html": ext in {".html", ".htm"},
                "download_quality_score": int(quality.get("score", 0)),
                "download_quality_reason": str(quality.get("reason", "")),
                "download_keyword_hits": quality.get("keyword_hits", []),
                "download_quality_metrics": quality.get("metrics", {}),
                "etag": etag,
                "last_modified": last_modified,
                "fetched_at": fetched_at,
                "skip_reason": "",
                "status": "downloaded",
            }
            collector.metadata_rows.append(row)
            collector.downloaded_rows.append(row)

            if depth >= legacy.MAX_DEPTH or ext not in {".html", ".htm"}:
                return
            for link in legacy.extract_links(content, url):
                if link in collector.visited:
                    continue
                collector.visited.add(link)
                next_depth = depth + 1
                host = legacy.normalize_host(urlparse(link).netloc)
                next_domain = legacy.domain_bucket(host)
                queued_per_domain[next_domain] += 1
                priority = -(legacy.score_url(link, next_depth) + (queued_per_domain[next_domain] * legacy.DOMAIN_PRIORITY_STEP))
                yield scrapy.Request(
                    url=link,
                    callback=self.parse_page,
                    errback=self.handle_error,
                    priority=priority,
                    meta={"depth_value": next_depth},
                    headers=legacy.HEADERS,
                )

    process = CrawlerProcess()
    try:
        process.crawl(UCASpider)
        process.start()
        legacy.save_metadata_safely(collector.metadata_rows)
        legacy.save_ingestion_state(collector.state)
        report_payload = {
            "started_at": collector.stats.get("started_at"),
            "finished_at": legacy.now_iso(),
            "settings": {
                "backend": "scrapy",
                "max_total_urls": legacy.MAX_TOTAL_URLS,
                "max_depth": legacy.MAX_DEPTH,
                "ingestion_threads": legacy.INGESTION_THREADS,
                "max_urls_per_domain": legacy.MAX_URLS_PER_DOMAIN,
                "max_urls_per_subdomain": legacy.MAX_URLS_PER_SUBDOMAIN,
                "incremental_fetch": legacy.INCREMENTAL_FETCH,
                "refresh_mode": legacy.REFRESH_MODE,
                "near_dup_enabled": legacy.ENABLE_NEAR_DUP_INGESTION,
                "near_dup_simhash_distance": legacy.NEAR_DUP_SIMHASH_DISTANCE,
                "min_download_quality_score": legacy.MIN_DOWNLOAD_QUALITY_SCORE,
                "sitemap_discovery": legacy.ENABLE_SITEMAP_DISCOVERY,
            },
            "metrics": {
                "crawled": int(collector.stats.get("crawled", 0)),
                "downloaded": int(collector.stats.get("downloaded", 0)),
                "skipped_quality": int(collector.stats.get("skipped_quality", 0)),
                "skipped_dup_exact": int(collector.stats.get("skipped_dup_exact", 0)),
                "skipped_dup_near": int(collector.stats.get("skipped_dup_near", 0)),
                "not_modified": int(collector.stats.get("not_modified", 0)),
                "refresh_skipped": int(collector.stats.get("refresh_skipped", 0)),
                "download_errors": int(collector.stats.get("download_errors", 0)),
                "http_errors": int(collector.stats.get("http_errors", 0)),
                "quota_skipped_domain": int(collector.stats.get("quota_skipped_domain", 0)),
                "quota_skipped_subdomain": int(collector.stats.get("quota_skipped_subdomain", 0)),
            },
            "per_domain": dict(sorted(collector.stats.get("per_domain", {}).items(), key=lambda item: item[1], reverse=True)),
            "per_subdomain_top20": dict(sorted(collector.stats.get("per_subdomain", {}).items(), key=lambda item: item[1], reverse=True)[:20]),
            "metadata_rows_count": len(collector.metadata_rows),
            "downloaded_rows_count": len(collector.downloaded_rows),
        }
        legacy.update_offline_pipeline_report("ingestion", report_payload)
        return collector.downloaded_rows
    finally:
        legacy.cleanup_playwright_resources()
