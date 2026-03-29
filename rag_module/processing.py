import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

from bs4 import BeautifulSoup
import pdfplumber
import docx
from tiktoken import encoding_for_model
from html import unescape

# ===================== CONFIGURATION =====================
RAW_PATH = "data_storage/raw"
PROCESSED_PATH = "data_storage/processed"

# Paramètres de chunking (recommandés en 2026)
CHUNK_TOKENS = 512          # Plus flexible que 400 (meilleur équilibre précision/contexte)
OVERLAP_TOKENS = 80         # ~15% d'overlap → bon compromis selon les benchmarks récents
LLM_MODEL = "gpt-4o-mini"   # ou "gpt-3.5-turbo" selon ton usage

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================== UTILITAIRES =====================
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Décodage des entités HTML
    text = unescape(text)
    # Normalisation des espaces (remplace tous les whitespace par un seul espace)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def hash_text(text: str) -> str:
    """Hash du contenu du chunk"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_file(file_path: str) -> str:
    """Hash complet du fichier source (pour détecter les modifications)"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ===================== PARSING SPÉCIALISÉ =====================
def extract_text_html(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # Suppression des éléments non pertinents
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            tag.decompose()
        
        # Priorité à <main> ou <article>, sinon tout le body
        main = soup.find(['main', 'article']) or soup.body or soup
        return main.get_text(separator=' ')
    except Exception as e:
        logger.error(f"Erreur extraction HTML {file_path}: {e}")
        return ""


def extract_text_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
    except Exception as e:
        logger.error(f"Erreur extraction PDF {file_path}: {e}")
    return text


def extract_text_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Optionnel : extraire aussi les tableaux si besoin plus tard
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Erreur extraction DOCX {file_path}: {e}")
        return ""


def extract_text_plain(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Erreur extraction texte brut {file_path}: {e}")
        return ""


# ===================== CHUNKING AVANCÉ =====================
def split_into_chunks(text: str, 
                      chunk_tokens: int = CHUNK_TOKENS, 
                      overlap: int = OVERLAP_TOKENS, 
                      model: str = LLM_MODEL) -> List[str]:
    """Chunking par tokens avec overlap correct"""
    if not text.strip():
        return []
    
    enc = encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    step = max(1, chunk_tokens - overlap)   # évite step=0

    for i in range(0, len(tokens), step):
        chunk_slice = tokens[i:i + chunk_tokens]
        if not chunk_slice:
            break
        chunk_text = enc.decode(chunk_slice).strip()
        if chunk_text and len(chunk_text) > 20:   # filtre les chunks trop petits
            chunks.append(chunk_text)

    logger.debug(f"Créé {len(chunks)} chunks à partir d'un texte de {len(tokens)} tokens")
    return chunks


# ===================== PRÉTRAITEMENT D’UN FICHIER =====================
def preprocess_file(file_path: str) -> List[Dict]:
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()
    
    # Mapping des extracteurs
    extractors = {
        ".html": extract_text_html,
        ".htm": extract_text_html,
        ".pdf": extract_text_pdf,
        ".docx": extract_text_docx,
        ".txt": extract_text_plain,
        ".md": extract_text_plain,
    }
    
    extractor = extractors.get(ext)
    if not extractor:
        logger.warning(f"Format non supporté : {ext} → {file_path}")
        return []

    text = extractor(file_path)
    text = clean_text(text)
    
    if len(text) < 50:  # fichier presque vide
        logger.info(f"Fichier ignoré (trop court) : {file_path}")
        return []

    chunks = split_into_chunks(text)
    result = []
    file_hash = hash_file(file_path)

    for idx, chunk in enumerate(chunks):
        chunk_hash = hash_text(chunk)
        
        metadata = {
            "source": file_path,
            "source_hash": file_hash,           # pour détecter les fichiers modifiés
            "type": ext[1:],
            "chunk_hash": chunk_hash,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "date_ingestion": datetime.now(timezone.utc).isoformat(),
            "chunk_tokens": len(encoding_for_model(LLM_MODEL).encode(chunk)),
        }
        
        result.append({
            "text": chunk,
            "metadata": metadata
        })

    logger.info(f"{len(chunks)} chunks générés depuis {file_path}")
    return result


# ===================== TRAITEMENT GLOBAL =====================
def preprocess_all(raw_path: str = RAW_PATH, 
                   processed_path: str = PROCESSED_PATH,
                   skip_existing: bool = True) -> None:
    
    os.makedirs(processed_path, exist_ok=True)
    seen_chunk_hashes = set()
    processed_count = 0

    logger.info(f"Démarrage du prétraitement RAG - Raw: {raw_path}")

    for root, _, files in os.walk(raw_path):
        for file in files:
            # Ignorer fichiers temporaires / cachés
            if file.startswith('.') or file.startswith('~') or file.endswith('.tmp'):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                chunks = preprocess_file(file_path)
                
                for chunk in chunks:
                    chash = chunk['metadata']['chunk_hash']
                    
                    if chash in seen_chunk_hashes:
                        continue  # doublon exact
                    
                    seen_chunk_hashes.add(chash)
                    
                    # Sauvegarde
                    filename = f"{chash}.json"
                    out_path = os.path.join(processed_path, filename)
                    
                    # Option : skip si déjà existant
                    if skip_existing and os.path.exists(out_path):
                        continue
                    
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Erreur critique sur {file_path}: {e}", exc_info=True)

    logger.info(f"Prétraitement terminé ! {processed_count} nouveaux chunks sauvegardés dans {processed_path}")


# ===================== EXÉCUTION =====================
if __name__ == "__main__":
    preprocess_all()