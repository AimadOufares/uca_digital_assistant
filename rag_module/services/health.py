from typing import Dict

from django.db import connections

from ..adapters.llm_provider import LLMProviderAdapter
from ..adapters.storage import DocumentStorage
from ..adapters.vector_store import get_vector_store_adapter
from ..shared.runtime import get_runtime_settings


def _database_health() -> Dict:
    try:
        with connections["default"].cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def build_live_health() -> Dict:
    settings = get_runtime_settings()
    return {
        "ok": True,
        "app_env": settings.app_env,
        "debug": settings.django_debug,
    }


def build_ready_health() -> Dict:
    settings = get_runtime_settings()
    storage = DocumentStorage(settings)
    db_health = _database_health()
    vector_health = get_vector_store_adapter(settings=settings, storage=storage).health()
    llm_health = LLMProviderAdapter(settings).health()
    pointer = storage.load_active_index_pointer()

    llm_ready = bool(llm_health.get("usable"))
    ready = bool(
        db_health.get("ok")
        and vector_health.get("ok")
        and vector_health.get("active_index_present")
        and llm_ready
    )

    return {
        "ok": ready,
        "ready": ready,
        "app_env": settings.app_env,
        "database": db_health,
        "vector_store": vector_health,
        "llm": llm_health,
        "checks": {
            "database_ready": bool(db_health.get("ok")),
            "vector_store_ready": bool(vector_health.get("ok") and vector_health.get("active_index_present")),
            "llm_ready": llm_ready,
        },
        "active_index": pointer,
    }
