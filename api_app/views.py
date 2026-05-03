import logging
from pathlib import Path

from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Count, Max
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.urls import reverse, reverse_lazy
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import FormView, TemplateView
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api_app.forms import StudentLoginForm, StudentSignupForm
from api_app.models import Conversation, Message
from api_app.services.identity import allowed_uca_email_domains
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
    conversation_id = serializers.IntegerField(required=False, min_value=1)


class ConversationUpdateSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=False, trim_whitespace=True, max_length=255)
    archive = serializers.BooleanField(required=False)


class TestView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"message": "API fonctionne !"})


class ChatAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        conversation_id = request.query_params.get("conversation_id")
        conversation = _resolve_requested_conversation(request.user, conversation_id)
        return Response(
            {
                "conversation_id": conversation.id,
                "conversation_title": conversation.title or "Nouvelle conversation",
                "messages": [_serialize_message(message) for message in conversation.messages.all()],
                "conversations": _serialize_conversation_list(request.user, selected_conversation_id=conversation.id),
            },
            status=status.HTTP_200_OK,
        )

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
        conversation = _resolve_requested_conversation(
            request.user,
            serializer.validated_data.get("conversation_id"),
        )
        _save_user_message(conversation, message)
        try:
            result = answer_question(QuestionRequest(question=message))
            _save_assistant_message(conversation, result)
            return Response(
                {
                    "conversation_id": conversation.id,
                    "conversation_title": conversation.title or "Nouvelle conversation",
                    "answer": result.answer.strip(),
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "backend": result.backend,
                    "retrieval_meta": result.retrieval_meta,
                    "conversations": _serialize_conversation_list(request.user, selected_conversation_id=conversation.id),
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


class ChatConversationsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        conversation = Conversation.objects.create(user=request.user, title="")
        return Response(
            {
                "conversation_id": conversation.id,
                "conversation_title": "Nouvelle conversation",
                "conversations": _serialize_conversation_list(request.user, selected_conversation_id=conversation.id),
                "messages": [],
            },
            status=status.HTTP_201_CREATED,
        )


class ChatConversationDetailAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, conversation_id: int):
        serializer = ConversationUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        conversation = get_object_or_404(
            Conversation.objects.filter(user=request.user, is_archived=False),
            pk=conversation_id,
        )

        updated_fields: list[str] = []
        title = serializer.validated_data.get("title")
        if title is not None:
            conversation.title = title.strip()
            updated_fields.append("title")

        archive = serializer.validated_data.get("archive")
        if archive is True:
            conversation.is_archived = True
            updated_fields.append("is_archived")

        if updated_fields:
            updated_fields.append("updated_at")
            conversation.save(update_fields=updated_fields)

        selected_conversation = None if conversation.is_archived else conversation.id
        return Response(
            {
                "conversation": _serialize_single_conversation(conversation, selected=not conversation.is_archived),
                "conversations": _serialize_conversation_list(
                    request.user,
                    selected_conversation_id=selected_conversation,
                ),
            },
            status=status.HTTP_200_OK,
        )

    def delete(self, request, conversation_id: int):
        conversation = get_object_or_404(
            Conversation.objects.filter(user=request.user, is_archived=False),
            pk=conversation_id,
        )
        conversation.is_archived = True
        conversation.save(update_fields=["is_archived", "updated_at"])
        return Response(
            {
                "detail": "Conversation archivee.",
                "conversations": _serialize_conversation_list(request.user),
            },
            status=status.HTTP_200_OK,
        )


@method_decorator(ensure_csrf_cookie, name="dispatch")
class ChatPageView(LoginRequiredMixin, TemplateView):
    template_name = "api_app/chat.html"
    login_url = "/login/"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["student_logout_url"] = reverse("student-logout")
        context["student_display_name"] = (
            self.request.user.get_full_name().strip()
            or self.request.user.email
            or self.request.user.username
        )
        return context


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


class StudentSignupPageView(FormView):
    template_name = "api_app/signup.html"
    form_class = StudentSignupForm
    success_url = reverse_lazy("chat-page")

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect("chat-page")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["allowed_domains"] = allowed_uca_email_domains()
        return context

    def form_valid(self, form):
        user = form.save()
        login(self.request, user, backend="api_app.auth_backends.EmailOrUsernameModelBackend")
        return HttpResponseRedirect(self.get_success_url())

    def get_success_url(self):
        next_url = self.request.POST.get("next", "").strip() or self.request.GET.get("next", "").strip()
        return next_url or str(self.success_url)


class StudentLoginPageView(LoginView):
    template_name = "api_app/login.html"
    authentication_form = StudentLoginForm
    redirect_authenticated_user = True

    def get_success_url(self):
        next_url = self.get_redirect_url()
        return next_url or str(reverse_lazy("chat-page"))


class StudentLogoutView(LogoutView):
    next_page = reverse_lazy("student-login")


def _get_or_create_active_conversation(user) -> Conversation:
    conversation = (
        Conversation.objects.filter(user=user, is_archived=False)
        .prefetch_related("messages")
        .order_by("-updated_at", "-created_at")
        .first()
    )
    if conversation is not None:
        return conversation
    return Conversation.objects.create(user=user)


def _resolve_requested_conversation(user, conversation_id) -> Conversation:
    if conversation_id in (None, "", 0, "0"):
        return _get_or_create_active_conversation(user)
    return get_object_or_404(
        Conversation.objects.filter(user=user, is_archived=False).prefetch_related("messages"),
        pk=int(conversation_id),
    )


def _serialize_message(message: Message) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "sources": list(message.sources or []),
        "confidence": message.confidence,
        "retrieval_meta": dict(message.retrieval_meta or {}),
        "created_at": message.created_at.isoformat(),
    }


def _serialize_conversation_list(user, selected_conversation_id: int | None = None) -> list[dict]:
    conversations = (
        Conversation.objects.filter(user=user, is_archived=False)
        .annotate(message_count=Count("messages"), last_message_at=Max("messages__created_at"))
        .order_by("-updated_at", "-created_at")[:12]
    )
    payload: list[dict] = []
    for conversation in conversations:
        preview = conversation.messages.order_by("created_at").values_list("content", flat=True).first() if conversation.message_count else ""
        payload.append(_serialize_single_conversation(
            conversation,
            selected=conversation.id == selected_conversation_id,
            preview_override=str(preview or "")[:120],
            message_count_override=int(conversation.message_count or 0),
        ))
    return payload


def _serialize_single_conversation(
    conversation: Conversation,
    selected: bool = False,
    preview_override: str = "",
    message_count_override: int | None = None,
) -> dict:
    return {
        "id": conversation.id,
        "title": (conversation.title or "Nouvelle conversation").strip(),
        "message_count": int(message_count_override if message_count_override is not None else conversation.messages.count()),
        "preview": preview_override,
        "updated_at": conversation.updated_at.isoformat(),
        "selected": selected,
        "is_archived": bool(conversation.is_archived),
    }


def _save_user_message(conversation: Conversation, text: str) -> Message:
    title = conversation.title.strip()
    if not title:
        conversation.title = text[:120].strip()
        conversation.save(update_fields=["title", "updated_at"])
    return Message.objects.create(
        conversation=conversation,
        role=Message.ROLE_USER,
        content=text,
    )


def _save_assistant_message(conversation: Conversation, result) -> Message:
    return Message.objects.create(
        conversation=conversation,
        role=Message.ROLE_ASSISTANT,
        content=result.answer.strip(),
        sources=list(result.sources or []),
        confidence=str(result.confidence or ""),
        retrieval_meta=dict(result.retrieval_meta or {}),
    )
