from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List

import docx
import pdfplumber
from bs4 import BeautifulSoup
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None

try:
    from ..shared.runtime import RuntimeSettings, get_runtime_settings
except ImportError:  # pragma: no cover
    from rag_module.shared.runtime import RuntimeSettings, get_runtime_settings


logger = logging.getLogger(__name__)


def extract_text_html(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            soup = BeautifulSoup(handle, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        main = soup.find(["main", "article"]) or soup.body or soup

        for tag in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(tag.name[1])
            tag.insert_before(f"\n\n{'#' * level} ")
            tag.insert_after("\n\n")

        for tag in main.find_all("li"):
            tag.insert_before("\n- ")
            tag.insert_after("\n")

        for tag in main.find_all("p"):
            tag.insert_before("\n\n")
            tag.insert_after("\n\n")

        text = main.get_text(" ", strip=True)
        return re.sub(r"\n[ \t]*\n+", "\n\n", text).strip()
    except Exception as exc:
        logger.warning("HTML extraction error %s: %s", path, exc)
        return ""


def _extract_text_pdf_pymupdf(path: str) -> str:
    if fitz is None:
        return ""

    text_parts: List[str] = []
    try:
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, 1):
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    text_parts.append(page_text.strip())

                try:
                    tables = page.find_tables()
                except Exception:
                    tables = None

                if not tables:
                    continue

                for table in getattr(tables, "tables", []) or []:
                    extracted = table.extract() or []
                    if not extracted:
                        continue
                    rows = []
                    for row in extracted:
                        if not row or not any(cell for cell in row):
                            continue
                        rows.append(" | ".join(str(cell or "").strip() for cell in row))
                    if rows:
                        text_parts.append(
                            f"\n[TABLE_PAGE_{page_num}]\n" + "\n".join(rows) + "\n[/TABLE]\n"
                        )
    except Exception as exc:
        logger.warning("PyMuPDF extraction error %s: %s", path, exc)
        return ""

    return "\n\n".join(text_parts)


def _extract_text_pdf_pdfplumber(path: str) -> str:
    text_parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(text)

                tables = page.extract_tables()
                for table in tables:
                    if table and any(any(cell for cell in row) for row in table):
                        rows = [
                            " | ".join(str(cell) if cell is not None else "" for cell in row)
                            for row in table
                        ]
                        text_parts.append(f"\n[TABLE_PAGE_{page_num}]\n" + "\n".join(rows) + "\n[/TABLE]\n")
    except Exception as exc:
        logger.warning("PDF extraction error %s: %s", path, exc)
        return ""

    return "\n\n".join(text_parts)


def extract_text_pdf(path: str, settings: RuntimeSettings | None = None) -> str:
    runtime = settings or get_runtime_settings()
    extractor = runtime.rag_pdf_extractor

    if extractor == "pymupdf":
        text = _extract_text_pdf_pymupdf(path)
        if text.strip():
            return text
        return _extract_text_pdf_pdfplumber(path)

    return _extract_text_pdf_pdfplumber(path)


def extract_text_docx(path: str) -> str:
    def _iter_block_items(parent):
        if isinstance(parent, DocxDocument):
            parent_element = parent.element.body
        else:
            parent_element = parent._element

        for child in parent_element.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def _paragraph_to_text(paragraph: Paragraph) -> str:
        text = paragraph.text.strip()
        if not text:
            return ""

        style_name = (getattr(paragraph.style, "name", "") or "").lower()
        if "heading" in style_name:
            return f"\n## {text}\n"
        if "list" in style_name or text.startswith(("•", "-", "*")):
            normalized = text.lstrip("•*- ").strip()
            return f"- {normalized}" if normalized else ""
        return text

    def _table_to_text(table: Table) -> str:
        rows: List[str] = []
        seen_rows = set()

        for row in table.rows:
            cells = []
            for cell in row.cells:
                cell_text = " ".join(
                    part.strip()
                    for part in cell.text.splitlines()
                    if part.strip()
                ).strip()
                cells.append(cell_text)

            if not any(cells):
                continue

            normalized_row = tuple(cells)
            if normalized_row in seen_rows:
                continue

            seen_rows.add(normalized_row)
            rows.append(" | ".join(cells))

        if not rows:
            return ""

        return "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"

    try:
        doc = docx.Document(path)
        blocks: List[str] = []

        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                paragraph_text = _paragraph_to_text(block)
                if paragraph_text:
                    blocks.append(paragraph_text)
            elif isinstance(block, Table):
                table_text = _table_to_text(block)
                if table_text:
                    blocks.append(table_text)

        for section in doc.sections:
            for region in (section.header, section.footer):
                for paragraph in region.paragraphs:
                    paragraph_text = _paragraph_to_text(paragraph)
                    if paragraph_text:
                        blocks.append(paragraph_text)

        return "\n\n".join(blocks)
    except Exception as exc:
        logger.warning("DOCX extraction error %s: %s", path, exc)
        return ""


def extract_text_plain(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Plain text extraction error %s: %s", path, exc)
        return ""


def build_extractors(settings: RuntimeSettings | None = None) -> Dict[str, Callable[[str], str]]:
    runtime = settings or get_runtime_settings()
    return {
        ".html": extract_text_html,
        ".pdf": lambda path: extract_text_pdf(path, settings=runtime),
        ".docx": extract_text_docx,
        ".txt": extract_text_plain,
        ".md": extract_text_plain,
    }
