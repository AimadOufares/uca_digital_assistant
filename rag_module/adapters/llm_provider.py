from typing import Dict

from openai import OpenAI

from ..shared.runtime import RuntimeSettings, get_runtime_settings


class LLMProviderAdapter:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or get_runtime_settings()

    def provider_order(self) -> list[str]:
        provider = self.settings.rag_llm_provider
        if provider in {"lmstudio", "local"}:
            return ["lmstudio"]
        if provider == "openai":
            return ["openai"]
        if provider == "auto":
            return ["lmstudio", "openai"]
        return ["lmstudio", "openai"]

    def health(self) -> Dict:
        provider = self.settings.rag_llm_provider
        statuses: Dict[str, Dict] = {}

        if provider in {"lmstudio", "local", "auto"}:
            try:
                if not self.settings.lm_studio_base_url:
                    raise RuntimeError("LM_STUDIO_BASE_URL manquant")
                client = OpenAI(
                    base_url=self.settings.lm_studio_base_url,
                    api_key=self.settings.lm_studio_api_key or "lm-studio",
                    timeout=5.0,
                )
                models = client.models.list()
                model_ids = [getattr(item, "id", "") for item in getattr(models, "data", []) or []]
                statuses["lmstudio"] = {
                    "ok": True,
                    "base_url": self.settings.lm_studio_base_url,
                    "models": [model_id for model_id in model_ids if model_id][:5],
                }
            except Exception as exc:
                statuses["lmstudio"] = {
                    "ok": False,
                    "base_url": self.settings.lm_studio_base_url,
                    "reason": str(exc),
                }

        if provider in {"openai", "auto"}:
            if not self.settings.openai_api_key:
                statuses["openai"] = {"ok": False, "reason": "OPENAI_API_KEY manquant"}
            else:
                try:
                    client = OpenAI(api_key=self.settings.openai_api_key, timeout=5.0)
                    models = client.models.list()
                    model_ids = [getattr(item, "id", "") for item in getattr(models, "data", []) or []]
                    statuses["openai"] = {"ok": True, "models": [model_id for model_id in model_ids if model_id][:5]}
                except Exception as exc:
                    statuses["openai"] = {"ok": False, "reason": str(exc)}

        fallback_allowed = provider in {"lmstudio", "local", "openai", "auto"}
        all_failed = bool(statuses) and not any(item.get("ok") for item in statuses.values())
        overall_state = "ok"
        if all_failed and fallback_allowed:
            overall_state = "degraded"
        elif all_failed:
            overall_state = "down"

        return {
            "provider": provider,
            "state": overall_state,
            "fallback_allowed": fallback_allowed,
            "providers": statuses,
        }
