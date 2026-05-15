from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from ..shared.runtime import RuntimeSettings, get_runtime_settings

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    PlaywrightError = Exception
    sync_playwright = None


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    html: str = ""
    error: str = ""
    final_url: str = ""
    selector_used: str = "body"


def _is_allowed_domain(url: str, settings: RuntimeSettings) -> bool:
    host = (urlparse(url).netloc or "").strip().lower()
    if not host:
        return False
    allowed = {item.strip().lower() for item in settings.rag_js_allowed_domains if item.strip()}
    return any(host == candidate or host.endswith(f".{candidate}") for candidate in allowed)


def render_url(url: str, settings: Optional[RuntimeSettings] = None) -> RenderResult:
    runtime = settings or get_runtime_settings()
    if not runtime.rag_js_fallback_enabled:
        return RenderResult(ok=False, error="js_fallback_disabled")
    if sync_playwright is None:
        return RenderResult(ok=False, error="playwright_not_installed")
    if not _is_allowed_domain(url, runtime):
        return RenderResult(ok=False, error="domain_not_allowed_for_js")

    selectors = ("main", "article", "body")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=runtime.rag_js_render_timeout_ms)

                selector_used = "body"
                for selector in selectors:
                    try:
                        page.wait_for_selector(
                            selector,
                            timeout=max(1000, runtime.rag_js_render_timeout_ms // 3),
                        )
                        selector_used = selector
                        break
                    except PlaywrightError:
                        continue

                html = page.content() or ""
                final_url = page.url or url
                if not html.strip():
                    return RenderResult(
                        ok=False,
                        error="empty_rendered_html",
                        final_url=final_url,
                        selector_used=selector_used,
                    )
                return RenderResult(
                    ok=True,
                    html=html,
                    final_url=final_url,
                    selector_used=selector_used,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        logger.warning("Playwright render failed for %s: %s", url, exc)
        return RenderResult(ok=False, error=f"playwright_error:{str(exc)[:120]}")
    except Exception as exc:  # pragma: no cover
        logger.warning("Unexpected render failure for %s: %s", url, exc)
        return RenderResult(ok=False, error=f"render_exception:{str(exc)[:120]}")
