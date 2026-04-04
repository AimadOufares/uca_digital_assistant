import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT_DIR = Path("data_storage/processed")
DEFAULT_OUTPUT_DIR = Path("data_storage/exports/processed_txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporte les JSON de data_storage/processed en fichiers texte organises "
            "sans resumer le contenu."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Dossier des chunks JSON (defaut: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie des .txt (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "all", "source"),
        default="both",
        help="both=all_chunks + fichiers par source, all=un seul fichier global, source=seulement par source",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limiter le nombre de JSON traites (0 = tous).",
    )
    return parser.parse_args()


def sanitize_filename(value: str, fallback: str = "source") -> str:
    cleaned = value.strip().replace("\\", "_").replace("/", "_")
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip(" ._")
    if not cleaned:
        return fallback
    return cleaned[:160]


def resolve_source_identity(metadata: Dict, default_stem: str) -> Tuple[str, str]:
    source_hash = str(metadata.get("source_hash", "") or "").strip()
    source_path = str(metadata.get("source", "") or "").strip()
    file_name = str(metadata.get("file_name", "") or "").strip()

    source_key = source_hash or source_path or file_name or default_stem
    source_label = file_name or Path(source_path).name or default_stem
    return source_key, source_label


def as_int_or_fallback(value, fallback: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def load_processed_entries(input_dir: Path, max_files: int) -> Tuple[List[Dict], List[Dict]]:
    json_files = sorted(input_dir.glob("*.json"))
    if max_files > 0:
        json_files = json_files[:max_files]

    entries: List[Dict] = []
    errors: List[Dict] = []

    for file_path in json_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Le contenu JSON n'est pas un objet.")

            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            source_key, source_label = resolve_source_identity(metadata, file_path.stem)
            entries.append(
                {
                    "processed_file": str(file_path.as_posix()),
                    "processed_name": file_path.name,
                    "source_key": source_key,
                    "source_label": source_label,
                    "chunk_index": metadata.get("index"),
                    "chunk_hash": metadata.get("chunk_hash"),
                    "metadata": metadata,
                    "json_data": data,
                }
            )
        except Exception as exc:
            errors.append({"file": str(file_path.as_posix()), "error": str(exc)})

    return entries, errors


def write_global_file(entries: List[Dict], output_dir: Path) -> Path:
    out_file = output_dir / "all_chunks.txt"
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"TOTAL_CHUNKS={len(entries)}\n\n")
        for i, entry in enumerate(entries, start=1):
            f.write(f"===== CHUNK {i}/{len(entries)} =====\n")
            f.write(f"processed_file: {entry['processed_file']}\n")
            f.write(f"source_key: {entry['source_key']}\n")
            f.write(f"source_label: {entry['source_label']}\n")
            f.write("json:\n")
            f.write(json.dumps(entry["json_data"], ensure_ascii=False, indent=2))
            f.write("\n\n")
    return out_file


def write_source_files(entries: List[Dict], output_dir: Path) -> Tuple[Path, List[Path]]:
    source_dir = output_dir / "by_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        grouped[entry["source_key"]].append(entry)

    generated_files: List[Path] = []
    for source_key, group_entries in grouped.items():
        group_entries.sort(
            key=lambda x: (
                as_int_or_fallback(x.get("chunk_index")),
                x.get("processed_name", ""),
            )
        )

        first = group_entries[0]
        base_name = sanitize_filename(first.get("source_label") or "source")
        suffix = sanitize_filename(source_key)[-12:] if source_key else "nohash"
        out_file = source_dir / f"{base_name}__{suffix}.txt"

        with out_file.open("w", encoding="utf-8") as f:
            f.write(f"source_key: {source_key}\n")
            f.write(f"source_label: {first.get('source_label', '')}\n")
            f.write(f"total_chunks: {len(group_entries)}\n\n")
            for i, entry in enumerate(group_entries, start=1):
                f.write(f"----- CHUNK {i}/{len(group_entries)} -----\n")
                f.write(f"processed_file: {entry['processed_file']}\n")
                f.write("json:\n")
                f.write(json.dumps(entry["json_data"], ensure_ascii=False, indent=2))
                f.write("\n\n")

        generated_files.append(out_file)

    return source_dir, generated_files


def write_errors_file(errors: List[Dict], output_dir: Path) -> Path:
    out_file = output_dir / "read_errors.txt"
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"TOTAL_ERRORS={len(errors)}\n\n")
        for i, item in enumerate(errors, start=1):
            f.write(f"{i}. file={item['file']}\n")
            f.write(f"   error={item['error']}\n")
    return out_file


def write_manifest(
    output_dir: Path,
    total_entries: int,
    total_errors: int,
    mode: str,
    global_file: Path = None,
    source_dir: Path = None,
    source_files: List[Path] = None,
    errors_file: Path = None,
) -> Path:
    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"mode={mode}\n")
        f.write(f"total_entries={total_entries}\n")
        f.write(f"total_errors={total_errors}\n")
        if global_file is not None:
            f.write(f"global_file={global_file.as_posix()}\n")
        if source_dir is not None:
            f.write(f"source_dir={source_dir.as_posix()}\n")
            f.write(f"source_files_count={len(source_files or [])}\n")
        if errors_file is not None:
            f.write(f"errors_file={errors_file.as_posix()}\n")
    return manifest


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    mode: str = args.mode
    max_files: int = args.max_files

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    entries, errors = load_processed_entries(input_dir=input_dir, max_files=max_files)

    if not entries:
        raise RuntimeError("Aucune entree JSON valide trouvee dans le dossier processed.")

    global_file = None
    source_dir = None
    source_files: List[Path] = []

    if mode in ("both", "all"):
        global_file = write_global_file(entries=entries, output_dir=output_dir)

    if mode in ("both", "source"):
        source_dir, source_files = write_source_files(entries=entries, output_dir=output_dir)

    errors_file = None
    if errors:
        errors_file = write_errors_file(errors=errors, output_dir=output_dir)

    manifest = write_manifest(
        output_dir=output_dir,
        total_entries=len(entries),
        total_errors=len(errors),
        mode=mode,
        global_file=global_file,
        source_dir=source_dir,
        source_files=source_files,
        errors_file=errors_file,
    )

    print(f"Export termine. Entrees: {len(entries)} | Erreurs: {len(errors)}")
    print(f"Manifest: {manifest.as_posix()}")
    if global_file is not None:
        print(f"Fichier global: {global_file.as_posix()}")
    if source_dir is not None:
        print(f"Dossier par source: {source_dir.as_posix()} ({len(source_files)} fichiers)")
    if errors_file is not None:
        print(f"Fichier erreurs: {errors_file.as_posix()}")


if __name__ == "__main__":
    main()
