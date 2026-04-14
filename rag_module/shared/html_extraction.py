import logging
import re
from html import unescape
from typing import Dict

from bs4 import BeautifulSoup

from .runtime_config import html_extractor_name

logger = logging.getLogger(__name__)


def _fallback_extract(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main = soup.find(["main", "article"]) or soup.body or soup
    parts = [
        node.get_text(" ", strip=True)
        for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"])
        if node.get_text(strip=True)
    ]
    text = "\n\n".join(parts) if parts else main.get_text("\n", strip=True)
    text = unescape(text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_main_text(html: str, url: str = "") -> Dict[str, str]:
    preferred = html_extractor_name()
    if preferred == "trafilatura":
        try:
            import trafilatura

            text = trafilatura.extract(
                html or "",
                url=url or None,
                include_links=False,
                include_images=False,
                include_tables=True,
                favor_precision=True,
                deduplicate=True,
            )
            cleaned = (text or "").strip()
            if cleaned:
                return {"text": cleaned, "method": "trafilatura"}
        except Exception as exc:
            logger.warning("Trafilatura indisponible, fallback HTML parser actif: %s", exc)

    return {"text": _fallback_extract(html), "method": "beautifulsoup"}
