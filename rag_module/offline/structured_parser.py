from __future__ import annotations

import re
from typing import Dict, List

from bs4 import BeautifulSoup


def extract_main_text(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
        tag.decompose()

    main = soup.find(["main", "article"]) or soup.body or soup
    if not main:
        return {"text": "", "title": "", "headings": [], "links": [], "lists": 0, "tables": 0}

    parts: List[str] = []
    headings: List[str] = []
    links: List[str] = []
    list_count = 0
    table_count = 0

    for tag in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "a"]):
        name = tag.name.lower()
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if name.startswith("h"):
            headings.append(text)
            parts.append(f"\n\n{'#' * int(name[1])} {text}\n")
            continue
        if name == "p":
            parts.append(f"\n\n{text}\n")
            continue
        if name == "li":
            list_count += 1
            parts.append(f"\n- {text}\n")
            continue
        if name == "table":
            table_count += 1
            rows = []
            for row in tag.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                parts.append("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")
            continue
        if name == "a":
            href = (tag.get("href") or "").strip()
            if href and href not in {"#", "/"} and len(links) < 20:
                links.append(href)

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else (headings[0] if headings else "")
    text = "".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if not text:
        text = main.get_text(" ", strip=True)
    if links:
        text = (text + "\n\n[LIENS_OFFICIELS]\n" + "\n".join(f"- {link}" for link in links[:10])).strip()

    return {
        "text": text,
        "title": title,
        "headings": headings[:12],
        "links": links,
        "lists": list_count,
        "tables": table_count,
    }
