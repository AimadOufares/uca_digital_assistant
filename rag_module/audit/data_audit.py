import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from rag_module.adapters.storage import DocumentStorage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE = DocumentStorage()
RAW_DIR = STORAGE.settings.rag_raw_dir
PROCESSED_DIR = STORAGE.settings.rag_processed_dir
REPORT_DIR = STORAGE.report_dir


def _active_index_paths() -> Dict[str, Path]:
    return STORAGE.resolve_active_faiss_paths()


def _iter_json_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.json"))


def _raw_stats() -> Dict:
    if not RAW_DIR.exists():
        return {"exists": False, "files": 0, "size_mb": 0.0, "extensions": {}}

    files = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    ext_counter = Counter((path.suffix.lower() or "none") for path in files)
    total_size = sum(path.stat().st_size for path in files)
    return {
        "exists": True,
        "files": len(files),
        "size_mb": round(total_size / (1024 * 1024), 2),
        "extensions": dict(ext_counter),
    }


def _load_index_metadata() -> Dict:
    paths = _active_index_paths()
    manifest_path = paths["manifest_file"]
    if not manifest_path.exists():
        return {"exists": False, "entries": 0, "domains_top10": [], "file_ext": {}, "build_id": "", "model_name": ""}

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    return {
        "exists": True,
        "entries": int(manifest.get("chunk_count", 0) or 0),
        "domains_top10": list((manifest.get("domain_distribution") or {}).items())[:10],
        "file_ext": dict(manifest.get("file_type_distribution") or {}),
        "build_id": str(manifest.get("build_id") or ""),
        "model_name": str(manifest.get("model_name") or ""),
        "backend": str(manifest.get("backend") or ""),
        "service_name_distribution": dict(manifest.get("service_name_distribution") or {}),
    }


def _load_chunks() -> List[Dict]:
    chunks_path = _active_index_paths()["chunks_file"]
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    chunks: List[Dict] = []
    for json_file in _iter_json_files(PROCESSED_DIR):
        try:
            with json_file.open("r", encoding="utf-8") as handle:
                chunks.append(json.load(handle))
        except Exception:
            continue
    return chunks


def _chunk_stats(chunks: List[Dict]) -> Dict:
    languages = Counter()
    file_types = Counter()
    faculties = Counter()
    doc_types = Counter()
    sources = Counter()
    duplicate_hash_count = 0
    duplicate_text_count = 0
    seen_hashes = set()
    seen_text_hashes = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("file_name") or "unknown"
        language = (metadata.get("language") or "unknown").lower()
        file_type = (metadata.get("file_type") or "unknown").lower()
        faculty = metadata.get("faculty") or "unknown"
        doc_type = metadata.get("document_type") or "unknown"

        chunk_hash = metadata.get("chunk_hash") or chunk.get("id") or ""
        text_value = (chunk.get("text") or "").strip().lower()
        text_hash = hashlib.sha1(text_value.encode("utf-8")).hexdigest() if text_value else ""

        languages[language] += 1
        file_types[file_type] += 1
        faculties[faculty] += 1
        doc_types[doc_type] += 1
        sources[source] += 1

        if chunk_hash:
            if chunk_hash in seen_hashes:
                duplicate_hash_count += 1
            else:
                seen_hashes.add(chunk_hash)
        if text_hash:
            if text_hash in seen_text_hashes:
                duplicate_text_count += 1
            else:
                seen_text_hashes.add(text_hash)

    top_sources = sources.most_common(10)
    top5_share = 0.0
    if chunks:
        top5_share = round(sum(value for _, value in sources.most_common(5)) * 100.0 / len(chunks), 2)

    return {
        "chunks": len(chunks),
        "languages_top10": languages.most_common(10),
        "file_types": dict(file_types),
        "faculties_top10": faculties.most_common(10),
        "document_types_top10": doc_types.most_common(10),
        "sources_unique": len(sources),
        "sources_top10": top_sources,
        "sources_top5_share_pct": top5_share,
        "duplicates": {
            "duplicate_chunk_hashes": duplicate_hash_count,
            "duplicate_text_chunks": duplicate_text_count,
        },
    }


def build_report() -> Dict:
    raw = _raw_stats()
    metadata = _load_index_metadata()
    chunks = _load_chunks()
    chunk_stats = _chunk_stats(chunks)
    active_paths = _active_index_paths()
    return {
        "generated_at": datetime.now().isoformat(),
        "paths": {
            "raw_dir": str(RAW_DIR),
            "processed_dir": str(PROCESSED_DIR),
            "index_chunks": str(active_paths["chunks_file"]),
            "index_metadata": str(active_paths["manifest_file"]),
        },
        "raw": raw,
        "index_metadata": metadata,
        "chunks": chunk_stats,
    }


def write_report(report: Dict) -> Dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"data_audit_{timestamp}.json"
    txt_path = REPORT_DIR / f"data_audit_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "RAG DATA AUDIT",
        f"Generated at: {report['generated_at']}",
        "",
        "[Raw]",
        f"Files: {report['raw']['files']}",
        f"Size MB: {report['raw']['size_mb']}",
        f"Extensions: {report['raw']['extensions']}",
        "",
        "[Index Metadata]",
        f"Entries: {report['index_metadata']['entries']}",
        f"Top domains: {report['index_metadata']['domains_top10']}",
        "",
        "[Chunks]",
        f"Total chunks: {report['chunks']['chunks']}",
        f"Languages top10: {report['chunks']['languages_top10']}",
        f"File types: {report['chunks']['file_types']}",
        f"Faculties top10: {report['chunks']['faculties_top10']}",
        f"Document types top10: {report['chunks']['document_types_top10']}",
        f"Unique sources: {report['chunks']['sources_unique']}",
        f"Top5 source share (%): {report['chunks']['sources_top5_share_pct']}",
        f"Duplicate hash chunks: {report['chunks']['duplicates']['duplicate_chunk_hashes']}",
        f"Duplicate text chunks: {report['chunks']['duplicates']['duplicate_text_chunks']}",
    ]
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return {"json": json_path, "txt": txt_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit des donnees RAG (raw/processed/index).")
    parser.parse_args()

    report = build_report()
    output_paths = write_report(report)
    print(f"Audit termine. JSON: {output_paths['json']}")
    print(f"Audit termine. TXT : {output_paths['txt']}")


if __name__ == "__main__":
    main()
