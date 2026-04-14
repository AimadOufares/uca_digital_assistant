import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def build_manifest(
    model_name: str,
    dim: int,
    chunk_count: int,
    policy_version: str,
    index_type: str,
    **extra,
) -> Dict:
    manifest = {
        "model_name": model_name,
        "embedding_dim": int(dim),
        "chunk_count": int(chunk_count),
        "index_type": index_type,
        "processing_policy_version": policy_version or "unknown",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(extra)
    return manifest


def load_manifest(path: str) -> Dict:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_manifest(path: str, manifest: Dict) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def validate_manifest(manifest: Dict, expected_model: str, expected_vector_store: str = "") -> None:
    if not manifest:
        raise ValueError("Manifest d'index introuvable ou vide.")

    model_name = str(manifest.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("Manifest d'index invalide: model_name manquant.")

    if expected_model and model_name != expected_model:
        raise ValueError(
            f"Incoherence index/modele: index construit avec '{model_name}', "
            f"mais runtime configure pour '{expected_model}'."
        )

    vector_store = str(manifest.get("vector_store") or "").strip().lower()
    if expected_vector_store and vector_store and vector_store != expected_vector_store.strip().lower():
        raise ValueError(
            f"Incoherence index/backend: index construit pour '{vector_store}', "
            f"mais runtime configure pour '{expected_vector_store}'."
        )

    dim = manifest.get("embedding_dim")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Manifest d'index invalide: embedding_dim incorrect.")
