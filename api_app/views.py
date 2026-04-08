import logging
import json
from pathlib import Path

from django.views.generic import TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_module.generation.rag_engine import (
    RAGGenerationError,
    RAGIndexNotReadyError,
    answer_question,
)

logger = logging.getLogger(__name__)


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
            result = answer_question(message)
            return Response(
                {
                    "answer": result.get("answer", "").strip(),
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


class ChatPageView(TemplateView):
    template_name = "api_app/chat.html"


class AdminDashboardPageView(TemplateView):
    template_name = "api_app/admin_dashboard.html"


class AdminDashboardAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        project_root = Path(__file__).resolve().parents[2]
        reports_dir = project_root / "data_storage" / "reports"
        
        def get_latest_json(prefix):
            if not reports_dir.exists():
                return {}
            files = list(reports_dir.glob(f"{prefix}_*.json"))
            if not files:
                return {}
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            try:
                with latest_file.open("r", encoding="utf-8") as h:
                    return json.load(h)
            except Exception:
                return {}

        return Response({
            "data_audit": get_latest_json("data_audit"),
            "raw_quality_audit": get_latest_json("raw_quality_audit"),
            "rag_eval": get_latest_json("rag_eval")
        }, status=status.HTTP_200_OK)
