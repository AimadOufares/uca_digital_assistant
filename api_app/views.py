import logging

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_module.contracts import QuestionRequest
from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError
from rag_module.services.health import build_live_health, build_ready_health
from rag_module.services.online import answer_question
from rag_module.services.reports import load_latest_reports

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
            result = answer_question(QuestionRequest(question=message))
            return Response(
                {
                    "answer": result.answer.strip(),
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


class AdminDashboardPageView(TemplateView):
    template_name = "api_app/admin_dashboard.html"


class AdminDashboardAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response(load_latest_reports(), status=status.HTTP_200_OK)


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
