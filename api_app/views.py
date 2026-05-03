import logging
from pathlib import Path

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_module.contracts import QuestionRequest
from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError
from rag_module.services.offline import run_evaluation, run_indexing, run_processing
from rag_module.services.health import build_live_health, build_ready_health
from rag_module.services.online import answer_question
from rag_module.services.reports import build_dashboard_payload, build_drive_sync_status, latest_report_payload
from rag_module.shared.runtime import get_runtime_settings

logger = logging.getLogger(__name__)
ALLOWED_DRIVE_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md"}


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        max_length=2000,
    )


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
        try:
            result = answer_question(QuestionRequest(question=message))
            return Response(
                {
                    "answer": result.answer.strip(),
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "backend": result.backend,
                    "retrieval_meta": result.retrieval_meta,
                },
                status=status.HTTP_200_OK,
            )
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


@method_decorator(ensure_csrf_cookie, name="dispatch")
class ChatPageView(TemplateView):
    template_name = "api_app/chat.html"


@method_decorator(ensure_csrf_cookie, name="dispatch")
class AdminDashboardPageView(LoginRequiredMixin, UserPassesTestMixin, TemplateView):
    template_name = "api_app/admin_dashboard.html"
    login_url = "/admin/login/"

    def test_func(self):
        return bool(self.request.user and self.request.user.is_staff)


class AdminOnlyAPIView(APIView):
    permission_classes = [IsAdminUser]


class AdminDashboardAPIView(AdminOnlyAPIView):

    def get(self, request):
        return Response(build_dashboard_payload(), status=status.HTTP_200_OK)


def _drive_dir() -> Path:
    runtime = get_runtime_settings()
    drive_dir = Path(runtime.rag_raw_drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    return drive_dir


def _serialize_drive_document(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024.0, 2),
        "updated_at": stat.st_mtime,
    }


def _safe_drive_path(filename: str) -> Path | None:
    cleaned = Path(filename).name
    if not cleaned or cleaned != filename:
        return None
    return _drive_dir() / cleaned


class DriveDocumentsAPIView(AdminOnlyAPIView):

    def get(self, request):
        drive_dir = _drive_dir()
        documents = [
            _serialize_drive_document(path)
            for path in sorted(drive_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file()
        ]
        return Response(
            {
                "documents": documents,
                "count": len(documents),
                "drive_sync_status": build_drive_sync_status(),
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if uploaded_file is None:
            return Response({"detail": "Aucun fichier n'a ete fourni."}, status=status.HTTP_400_BAD_REQUEST)

        filename = Path(uploaded_file.name).name
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_DRIVE_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_DRIVE_EXTENSIONS))
            return Response(
                {"detail": f"Extension non autorisee. Formats acceptes: {allowed}."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        destination = _safe_drive_path(filename)
        if destination is None:
            return Response({"detail": "Nom de fichier invalide."}, status=status.HTTP_400_BAD_REQUEST)

        with destination.open("wb+") as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)

        return Response(
            {
                "detail": "Document ajoute au corpus drive.",
                "document": _serialize_drive_document(destination),
            },
            status=status.HTTP_201_CREATED,
        )


class DriveDocumentDetailAPIView(AdminOnlyAPIView):

    def delete(self, request, filename: str):
        target = _safe_drive_path(filename)
        if target is None:
            return Response({"detail": "Nom de fichier invalide."}, status=status.HTTP_400_BAD_REQUEST)
        if not target.exists() or not target.is_file():
            return Response({"detail": "Document introuvable."}, status=status.HTTP_404_NOT_FOUND)

        target.unlink()
        return Response({"detail": "Document supprime du corpus drive."}, status=status.HTTP_200_OK)


class DriveRebuildAPIView(AdminOnlyAPIView):

    def post(self, request):
        try:
            processing_result = run_processing(corpus="drive")
            indexing_result = run_indexing(corpus="published", publish=True)
        except Exception as exc:
            logger.exception("Erreur pendant le rebuild drive")
            return Response(
                {"detail": f"Echec du rebuild drive: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "detail": "Corpus drive retraite et index publie avec succes.",
                "processing": processing_result,
                "index": {
                    "backend": indexing_result.backend,
                    "build_id": indexing_result.build_id,
                    "chunk_count": indexing_result.chunk_count,
                    "published": indexing_result.published,
                },
            },
            status=status.HTTP_200_OK,
        )


class DriveEvaluateAPIView(AdminOnlyAPIView):
    def post(self, request):
        try:
            report_paths = run_evaluation(top_k=5, skip_generation=True, benchmark="drive")
            payload = latest_report_payload("rag_eval")
        except Exception as exc:
            logger.exception("Erreur pendant l'evaluation drive")
            return Response(
                {"detail": f"Echec de l'evaluation drive: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "detail": "Benchmark drive relance avec succes.",
                "report_paths": report_paths,
                "report": payload,
            },
            status=status.HTTP_200_OK,
        )


class LatestAdminReportAPIView(AdminOnlyAPIView):
    def get(self, request, kind: str):
        try:
            payload = latest_report_payload(kind)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if not payload.get("available"):
            return Response(payload, status=status.HTTP_404_NOT_FOUND)
        return Response(payload, status=status.HTTP_200_OK)


class LiveHealthAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response(build_live_health(), status=status.HTTP_200_OK)


class ReadyHealthAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        payload = build_ready_health()
        http_status = status.HTTP_200_OK if payload.get("ready") else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(payload, status=http_status)
