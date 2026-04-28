import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from ..shared.runtime import RuntimeSettings, get_runtime_settings


class DocumentStorage:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or get_runtime_settings()
        self.settings.ensure_directories()

    @property
    def report_dir(self) -> Path:
        return self.settings.rag_reports_dir

    @property
    def index_dir(self) -> Path:
        return self.settings.rag_index_dir

    def builds_dir(self) -> Path:
        path = self.index_dir / "builds"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_dir(self, build_id: str) -> Path:
        path = self.builds_dir() / build_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def faiss_build_paths(self, build_id: str) -> Dict[str, Path]:
        root = self.build_dir(build_id) / "faiss"
        root.mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "index_file": root / "index.faiss",
            "chunks_file": root / "chunks.json",
            "manifest_file": root / "index_manifest.json",
            "bm25_file": root / "bm25_corpus.json",
        }

    def legacy_faiss_paths(self) -> Dict[str, Path]:
        root = self.index_dir
        return {
            "root": root,
            "index_file": root / "index.faiss",
            "chunks_file": root / "chunks.json",
            "manifest_file": root / "index_manifest.json",
            "bm25_file": root / "bm25_corpus.json",
        }

    def active_index_pointer_path(self) -> Path:
        return self.index_dir / "active_index.json"

    def save_json_atomic(self, path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path.replace(path)

    def load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def save_active_index_pointer(self, payload: Dict) -> None:
        self.save_json_atomic(self.active_index_pointer_path(), payload)

    def load_active_index_pointer(self) -> Dict:
        pointer = self.load_json(self.active_index_pointer_path())
        if pointer:
            return pointer

        legacy = self.legacy_faiss_paths()
        if legacy["index_file"].exists() and legacy["chunks_file"].exists():
            return {
                "backend": "faiss",
                "build_id": "legacy",
                "published": True,
                "paths": {key: str(value) for key, value in legacy.items()},
            }
        return {}

    def resolve_active_faiss_paths(self) -> Dict[str, Path]:
        pointer = self.load_active_index_pointer()
        if pointer.get("backend") == "faiss":
            raw_paths = pointer.get("paths", {}) or {}
            index_file = Path(raw_paths.get("index_file", ""))
            chunks_file = Path(raw_paths.get("chunks_file", ""))
            manifest_file = Path(raw_paths.get("manifest_file", ""))
            bm25_file = Path(raw_paths.get("bm25_file", ""))
            if index_file.exists() and chunks_file.exists():
                return {
                    "root": Path(raw_paths.get("root", index_file.parent)),
                    "index_file": index_file,
                    "chunks_file": chunks_file,
                    "manifest_file": manifest_file,
                    "bm25_file": bm25_file,
                }
        return self.legacy_faiss_paths()

    def _mirror_faiss_build_to_legacy_root(self, build_paths: Dict[str, Path]) -> Dict[str, Path]:
        legacy_paths = self.legacy_faiss_paths()
        for key in ("index_file", "chunks_file", "manifest_file", "bm25_file"):
            source = build_paths.get(key)
            destination = legacy_paths.get(key)
            if not source or not destination or not source.exists():
                raise FileNotFoundError(f"Artefact FAISS manquant pour publication: {key}")
            self.copy_file(source, destination)
        return legacy_paths

    def publish_faiss_build(self, build_id: str) -> Dict:
        build_paths = self.faiss_build_paths(build_id)
        legacy_paths = self._mirror_faiss_build_to_legacy_root(build_paths)
        payload = {
            "backend": "faiss",
            "build_id": build_id,
            "published": True,
            "paths": {key: str(value) for key, value in build_paths.items()},
            "legacy_paths": {key: str(value) for key, value in legacy_paths.items()},
        }
        self.save_active_index_pointer(payload)
        return payload

    def publish_qdrant_build(self, build_id: str, collection_name: str) -> Dict:
        manifest_path = self.build_dir(build_id) / "qdrant" / "index_manifest.json"
        payload = {
            "backend": "qdrant",
            "build_id": build_id,
            "published": True,
            "collection_name": collection_name,
            "manifest_path": str(manifest_path),
        }
        self.save_active_index_pointer(payload)
        return payload

    def copy_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def latest_report(self, prefix: str) -> Optional[Dict]:
        if not self.report_dir.exists():
            return None
        files = sorted(
            self.report_dir.glob(f"{prefix}_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        try:
            return self.load_json(files[0])
        except Exception:
            return None
