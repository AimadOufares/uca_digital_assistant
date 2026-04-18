import logging
import re
from pathlib import Path
from typing import Dict, List
from html import unescape

import docx
import pdfplumber
from bs4 import BeautifulSoup

from ..shared.runtime_config import document_parser_name, html_extractor_name

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


def _normalize_block(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _append_section(sections: List[Dict], title: str, path: List[str], lines: List[str], is_table: bool = False) -> None:
    text = _normalize_block("\n".join(lines))
    if not text:
        return
    section_title = (title or "").strip() or "Section"
    sections.append(
        {
            "title": section_title,
            "path": [part for part in path if part],
            "text": text,
            "is_table": bool(is_table),
        }
    )


def _markdown_to_sections(markdown_text: str) -> List[Dict]:
    sections: List[Dict] = []
    current_title = "Document"
    current_path: List[str] = []
    buffer: List[str] = []
    table_buffer: List[str] = []
    in_table = False

    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            if table_buffer:
                _append_section(sections, current_title, current_path, table_buffer, is_table=True)
                table_buffer = []
                in_table = False
            if buffer:
                _append_section(sections, current_title, current_path, buffer)
                buffer = []

            level = max(1, len(line) - len(line.lstrip("#")))
            title = line[level:].strip() or current_title
            current_path = current_path[: level - 1]
            current_path.append(title)
            current_title = title
            continue

        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            table_buffer.append(line)
            in_table = True
            continue

        if in_table and not line.strip():
            _append_section(sections, current_title, current_path, table_buffer, is_table=True)
            table_buffer = []
            in_table = False
            continue

        if in_table:
            table_buffer.append(line)
            continue

        buffer.append(line)

    if table_buffer:
        _append_section(sections, current_title, current_path, table_buffer, is_table=True)
    if buffer:
        _append_section(sections, current_title, current_path, buffer)
    return sections


def _looks_like_heading(line: str) -> bool:
    candidate = (line or "").strip()
    if len(candidate) < 4 or len(candidate.split()) > 14:
        return False
    if candidate.endswith(":"):
        return True
    if candidate.isupper():
        return True
    alpha_words = [word for word in candidate.split() if word.isalpha()]
    if alpha_words and sum(1 for word in alpha_words if word[:1].isupper()) >= max(1, len(alpha_words) - 1):
        return True
    return False


def _text_to_sections(text: str, default_title: str = "Document") -> List[Dict]:
    sections: List[Dict] = []
    current_title = default_title
    current_path = [default_title]
    buffer: List[str] = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if buffer:
                buffer.append("")
            continue
        if _looks_like_heading(line):
            if buffer:
                _append_section(sections, current_title, current_path, buffer)
                buffer = []
            current_title = line.rstrip(":").strip()
            current_path = [default_title, current_title]
            continue
        buffer.append(line)

    if buffer:
        _append_section(sections, current_title, current_path, buffer)

    return sections or [{"title": default_title, "path": [default_title], "text": _normalize_block(text), "is_table": False}]


def _extract_text_pdf(path: str) -> str:
    text_parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    text_parts.append(extracted)
                tables = page.extract_tables() or []
                for table in tables:
                    if table and any(any(cell for cell in row) for row in table):
                        markdown_rows = [
                            " | ".join(str(cell) if cell is not None else "" for cell in row)
                            for row in table
                        ]
                        text_parts.append(f"# Tableau page {page_num}\n" + "\n".join(markdown_rows))
    except Exception as exc:
        logger.warning("PDF extraction fallback error %s: %s", path, exc)
    return "\n\n".join(part for part in text_parts if part.strip())


def _extract_text_docx(path: str) -> str:
    try:
        document = docx.Document(path)
        blocks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n\n".join(blocks)
    except Exception as exc:
        logger.warning("DOCX extraction fallback error %s: %s", path, exc)
        return ""


def _extract_text_plain(path: str) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding, errors="replace")
        except Exception:
            continue
    return ""


def _extract_with_docling(path: str) -> Dict:
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(path)
        document = getattr(result, "document", None)
        if document is None:
            return {}

        if hasattr(document, "export_to_markdown"):
            markdown_text = document.export_to_markdown() or ""
        else:
            markdown_text = str(document)

        sections = _markdown_to_sections(markdown_text)
        full_text = "\n\n".join(section.get("text", "") for section in sections if section.get("text"))
        return {"text": full_text, "sections": sections, "parser": "docling"}
    except Exception as exc:
        logger.warning("Docling indisponible pour %s, fallback parser actif: %s", path, exc)
        return {}


def extract_document_structure(path: str) -> Dict:
    suffix = Path(path).suffix.lower()
    preferred = document_parser_name()

    if preferred == "docling" and suffix in {".pdf", ".docx", ".html", ".htm"}:
        payload = _extract_with_docling(path)
        if payload.get("text"):
            return payload

    if suffix in {".html", ".htm"}:
        raw_html = _extract_text_plain(path)
        extracted = extract_main_text(raw_html, url="")
        text = extracted.get("text", "")
        return {
            "text": text,
            "sections": _text_to_sections(text, default_title="Page web"),
            "parser": extracted.get("method", "html"),
        }
    if suffix == ".pdf":
        text = _extract_text_pdf(path)
        return {"text": text, "sections": _text_to_sections(text, default_title="PDF"), "parser": "pdfplumber"}
    if suffix == ".docx":
        text = _extract_text_docx(path)
        return {"text": text, "sections": _text_to_sections(text, default_title="DOCX"), "parser": "python-docx"}

    text = _extract_text_plain(path)
    return {"text": text, "sections": _text_to_sections(text, default_title="Document"), "parser": "plain"}
