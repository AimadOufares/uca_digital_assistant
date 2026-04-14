import logging
import json
import subprocess
import sys
import time
from pathlib import Path

import portalocker
from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_module.generation.rag_engine import (
    RAGGenerationError,
    RAGIndexNotReadyError,
    answer_question,
)
from rag_module.shared.context_resolution import CANONICAL_ESTABLISHMENTS, normalize_establishment_label

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    return Path(settings.BASE_DIR)


def _get_reports_dir() -> Path:
    return _get_project_root() / "data_storage" / "reports"


def _get_audit_lock_path(task: str) -> Path:
    locks_dir = _get_project_root() / "data_storage" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir / f"{task}.lock"


def _load_latest_report(reports_dir: Path, prefix: str) -> dict:
    if not reports_dir.exists():
        return {}

    files = list(reports_dir.glob(f"{prefix}_*.json"))
    if not files:
        return {}

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    try:
        with latest_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data.setdefault("_report_file", latest_file.name)
            data.setdefault("_report_mtime", latest_file.stat().st_mtime)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("Erreur lecture rapport %s", latest_file)
        return {}


class StaffRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    login_url = "/admin/login/"

    def test_func(self):
        user = getattr(self.request, "user", None)
        return bool(user and user.is_authenticated and user.is_staff)


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        max_length=2000,
    )
    user_establishment = serializers.CharField(
        required=False,
        allow_blank=True,
        trim_whitespace=True,
        max_length=64,
    )

    def validate_user_establishment(self, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            return ""
        normalized = normalize_establishment_label(cleaned)
        if normalized is None or normalized not in CANONICAL_ESTABLISHMENTS:
            raise serializers.ValidationError(
                "Etablissement invalide. Valeurs acceptees: " + ", ".join(CANONICAL_ESTABLISHMENTS)
            )
        return normalized


class TestView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"message": "API fonctionne !"})


class ChatAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "detail": "Entrée invalide. Le champ 'message' est obligatoire.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        message = serializer.validated_data["message"]
        user_establishment = serializer.validated_data.get("user_establishment") or None
        try:
            result = answer_question(message, user_establishment=user_establishment)
            payload = {
                "answer": result.get("answer", "").strip(),
            }
            if result.get("needs_clarification"):
                payload["needs_clarification"] = True
            return Response(payload, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RAGIndexNotReadyError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except RAGGenerationError as exc:
            logger.exception("Erreur de génération RAG")
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception:
            logger.exception("Erreur inattendue sur /api/chat/")
            return Response(
                {"detail": "Erreur interne du serveur."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatPageView(TemplateView):
    template_name = "api_app/chat.html"


@method_decorator(ensure_csrf_cookie, name="dispatch")
class AdminDashboardPageView(StaffRequiredMixin, TemplateView):
    template_name = "api_app/admin_dashboard.html"


class AdminDashboardAPIView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        reports_dir = _get_reports_dir()

        sections = {
            "data_audit": "data_audit",
            "raw_quality_audit": "raw_quality_audit",
            "rag_eval": "rag_eval",
        }

        payload = {}
        for key, prefix in sections.items():
            try:
                payload[key] = _load_latest_report(reports_dir, prefix)
            except Exception:
                logger.exception("Erreur chargement section dashboard : %s", key)
                payload[key] = {}

        return Response(payload, status=status.HTTP_200_OK)


# ── Audit runner ─────────────────────────────────────────────────────────────

# Map de chaque tâche vers son module Python et ses arguments CLI
_AUDIT_TASKS = {
    "data_audit": {
        "module": "rag_module.audit.data_audit",
        "args": [],
        "label": "Data Audit (raw/chunks/index)",
        "timeout": 120,
    },
    "raw_quality": {
        "module": "rag_module.audit.raw_quality_pipeline",
        "args": [],
        "label": "Raw Quality Audit",
        "timeout": 300,
    },
    "rag_eval": {
        "module": "rag_module.evaluation.evaluate_rag",
        "args": ["--skip-generation"],
        "label": "RAG Evaluation (retrieval)",
        "timeout": 180,
    },
    "rag_eval_full": {
        "module": "rag_module.evaluation.evaluate_rag",
        "args": [],
        "label": "RAG Evaluation (retrieval + génération)",
        "timeout": 600,
    },
}


class RunAuditView(APIView):
    """POST /api/run-audit/<task>/ — Lance un script d'audit en sous-processus."""
    permission_classes = [IsAdminUser]

    def post(self, request, task):
        task_cfg = _AUDIT_TASKS.get(task)
        if task_cfg is None:
            return Response(
                {"detail": f"Tâche inconnue : '{task}'. Choix : {list(_AUDIT_TASKS)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        cmd = [sys.executable, "-m", task_cfg["module"]] + task_cfg["args"]
        timeout = task_cfg["timeout"]
        started_at = time.time()
        logger.info("[RunAudit] Lancement : %s", " ".join(cmd))
        lock_path = _get_audit_lock_path(task)

        try:
            with portalocker.Lock(
                str(lock_path),
                mode="a",
                timeout=0,
                fail_when_locked=True,
            ):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(_get_project_root()),
                )
            elapsed = round(time.time() - started_at, 1)
            success = (result.returncode == 0)
            logger.info(
                "[RunAudit] %s terminé en %ss — code %s",
                task, elapsed, result.returncode,
            )
            return Response(
                {
                    "task": task,
                    "label": task_cfg["label"],
                    "success": success,
                    "returncode": result.returncode,
                    "elapsed_s": elapsed,
                    "stdout": result.stdout[-3000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                },
                status=status.HTTP_200_OK if success else status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except portalocker.exceptions.BaseLockException:
            logger.warning("[RunAudit] Tâche déjà en cours: %s", task)
            return Response(
                {
                    "task": task,
                    "label": task_cfg["label"],
                    "success": False,
                    "error": "Cette tâche est déjà en cours d'exécution.",
                },
                status=status.HTTP_409_CONFLICT,
            )
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - started_at, 1)
            logger.error("[RunAudit] Timeout (%ss) pour la tâche %s", timeout, task)
            return Response(
                {
                    "task": task,
                    "label": task_cfg["label"],
                    "success": False,
                    "error": f"Timeout après {timeout}s",
                    "elapsed_s": elapsed,
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            logger.exception("[RunAudit] Erreur inattendue pour la tâche %s", task)
            return Response(
                {"task": task, "success": False, "error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
