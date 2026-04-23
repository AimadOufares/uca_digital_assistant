from typing import Dict

from ..adapters.storage import DocumentStorage


REPORT_PREFIXES = {
    "data_audit": "data_audit",
    "raw_quality_audit": "raw_quality_audit",
    "rag_eval": "rag_eval",
}


def load_latest_reports() -> Dict[str, Dict]:
    storage = DocumentStorage()
    payload: Dict[str, Dict] = {}
    for key, prefix in REPORT_PREFIXES.items():
        payload[key] = storage.latest_report(prefix) or {}
    return payload
