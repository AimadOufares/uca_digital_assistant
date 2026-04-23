from bs4 import BeautifulSoup


def extract_main_text(html: str) -> dict:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    main = soup.find(["main", "article"]) or soup.body or soup
    text = main.get_text(" ", strip=True) if main else ""
    return {"text": text}
