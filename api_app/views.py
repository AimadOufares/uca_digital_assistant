import logging
from pathlib import Path
import re
from threading import Lock
from time import monotonic
from urllib.parse import urlparse

from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.db.models import Count, Max, Q
from django.http import FileResponse, HttpResponseRedirect
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
from api_app.models import Conversation, Message, MessageFeedback
from api_app.services.conversation_context import build_conversation_context, update_conversation_context
from api_app.services.identity import allowed_uca_email_domains
from rag_module.audit import data_audit, raw_quality_pipeline
from rag_module.contracts import QuestionRequest
from rag_module.generation.rag_engine import RAGGenerationError, RAGIndexNotReadyError
from rag_module.services.offline import run_evaluation, run_indexing, run_processing
from rag_module.services.health import build_live_health, build_ready_health
from rag_module.services.online import answer_question
from rag_module.services.reports import build_dashboard_payload, latest_report_payload
from rag_module.shared.runtime import get_runtime_settings
from rag_module.evaluation.evaluate_rag import load_benchmark_set, save_benchmark_set, _retrieval_metrics

logger = logging.getLogger(__name__)
ALLOWED_DRIVE_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md"}
AUDIT_TASK_LOCKS = {
    "data_audit": Lock(),
    "raw_quality": Lock(),
    "rag_eval": Lock(),
    "rag_eval_full": Lock(),
    "context_eval": Lock(),
}
SERVICE_LABELS = {
    "ucastudent": "UC@Student",
    "ucaplat": "UCAPLAT",
    "pedoc": "PEDOC",
    "cip": "CIP",
    "e-candidature": "E-Candidature",
    "espace diplomes": "Espace Diplomes",
    "soutien-recherche": "Soutien-Recherche",
}
ANSWER_SECTION_BREAK_RE = re.compile(
    r"^\s*(sources utiles|niveau de confiance|points a verifier|si necessaire\s*:\s*points a verifier)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ANSWER_LEAD_RE = re.compile(r"^\s*reponse\s*$", re.IGNORECASE | re.MULTILINE)
HASHED_FILENAME_RE = re.compile(r"_([0-9a-f]{8,})(?=\.[a-z0-9]+$)", re.IGNORECASE)
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


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
        context_payload = build_conversation_context(conversation, message)
        rag_question = str(context_payload.get("rewritten_question") or message).strip()
        _save_user_message(conversation, message)
        try:
            result = answer_question(QuestionRequest(question=rag_question))
            result.retrieval_meta.update(
                {
                    "original_question": message,
                    "rewritten_question": rag_question,
                    "conversation_context_used": bool(context_payload.get("context_used")),
                    "context_service": (
                        context_payload.get("detected_service")
                        or (context_payload.get("context_meta") or {}).get("service", "")
                    ),
                    "context_intent": (
                        context_payload.get("detected_intent")
                        or (context_payload.get("context_meta") or {}).get("intent", "")
                    ),
                }
            )
            clean_answer = _sanitize_answer_text(result.answer, question=message)
            clean_sources = _sanitize_sources(result.sources)
            msg_obj = _save_assistant_message(conversation, result, answer=clean_answer, sources=clean_sources)
            update_conversation_context(conversation, message, clean_answer, result, context_payload=context_payload)
            return Response(
                {
                    "conversation_id": conversation.id,
                    "conversation_title": conversation.title or "Nouvelle conversation",
                    "answer": clean_answer,
                    "sources": clean_sources,
                    "confidence": result.confidence,
                    "backend": result.backend,
                    "retrieval_meta": result.retrieval_meta,
                    "message_id": msg_obj.id,
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
                "detail": "Conversation supprimee de l'historique actif.",
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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        # Provide simple boolean flags for template to avoid complex filter() calls
        context["can_view_global"] = bool(user and user.is_authenticated and user.is_staff)
        context["can_view_corpus"] = bool(user and user.is_authenticated and (user.is_staff or user.groups.filter(name="corpus_manager").exists()))
        context["can_view_quality"] = bool(user and user.is_authenticated and (user.is_staff or user.groups.filter(name="auditor").exists()))
        context["can_view_benchmark"] = bool(user and user.is_authenticated and (user.is_staff or user.groups.filter(name="evaluator").exists()))
        context["can_view_maintenance"] = bool(user and user.is_authenticated and user.is_superuser)
        return context


class AdminOnlyAPIView(APIView):
    permission_classes = [IsAdminUser]


class AdminDashboardAPIView(AdminOnlyAPIView):

    def get(self, request):
        return Response(build_dashboard_payload(), status=status.HTTP_200_OK)


class RunAuditAPIView(AdminOnlyAPIView):

    def post(self, request, task: str):
        normalized = str(task or "").strip().lower()
        audit_lock = AUDIT_TASK_LOCKS.get(normalized)
        if audit_lock is None:
            return Response(
                {
                    "detail": "Tache d'audit inconnue.",
                    "choices": sorted(AUDIT_TASK_LOCKS.keys()),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not audit_lock.acquire(blocking=False):
            return Response(
                {"detail": "Cette tache est deja en cours d'execution.", "task": normalized},
                status=status.HTTP_409_CONFLICT,
            )

        started = monotonic()
        try:
            payload = _run_admin_audit_task(normalized)
            return Response(
                {
                    "task": normalized,
                    "success": True,
                    "elapsed_s": round(monotonic() - started, 2),
                    **payload,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as exc:
            logger.exception("Erreur pendant l'audit admin %s", normalized)
            return Response(
                {
                    "task": normalized,
                    "success": False,
                    "detail": str(exc),
                    "elapsed_s": round(monotonic() - started, 2),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            audit_lock.release()


class AdminConversationsAPIView(AdminOnlyAPIView):

    def get(self, request):
        # --- Query params ---
        status_filter = request.query_params.get("status", "active")  # active | archived | all
        search = request.query_params.get("search", "").strip()
        try:
            page = max(1, int(request.query_params.get("page", 1)))
        except (ValueError, TypeError):
            page = 1
        try:
            per_page = min(50, max(5, int(request.query_params.get("per_page", 20))))
        except (ValueError, TypeError):
            per_page = 20

        # --- Base queryset with annotations ---
        qs = Conversation.objects.select_related("user").annotate(
            message_count=Count("messages"),
            assistant_count=Count("messages", filter=Q(messages__role=Message.ROLE_ASSISTANT)),
            source_answer_count=Count(
                "messages",
                filter=Q(messages__role=Message.ROLE_ASSISTANT) & ~Q(messages__sources=[]),
            ),
            last_message_at=Max("messages__created_at"),
            feedback_count=Count("messages__feedback"),
        )

        # --- Status filter ---
        if status_filter == "active":
            qs = qs.filter(is_archived=False)
        elif status_filter == "archived":
            qs = qs.filter(is_archived=True)
        # else: all

        # --- Has feedback filter ---
        has_feedback = request.query_params.get("has_feedback", "")
        if has_feedback == "1":
            qs = qs.filter(messages__feedback__isnull=False).distinct()

        # --- Search ---
        if search:
            qs = qs.filter(
                Q(title__icontains=search)
                | Q(user__first_name__icontains=search)
                | Q(user__last_name__icontains=search)
                | Q(user__email__icontains=search)
                | Q(user__username__icontains=search)
            )

        # --- Ordering ---
        qs = qs.order_by("-last_message_at", "-updated_at")

        # --- Global summary (unfiltered) ---
        all_qs = Conversation.objects.select_related("user")
        assistant_messages = Message.objects.filter(role=Message.ROLE_ASSISTANT)
        answers_with_sources = assistant_messages.exclude(sources=[])
        assistant_count_total = assistant_messages.count()
        source_count_total = answers_with_sources.count()

        summary = {
            "active_conversations": all_qs.filter(is_archived=False).count(),
            "archived_conversations": all_qs.filter(is_archived=True).count(),
            "total_messages": Message.objects.count(),
            "assistant_answers": assistant_count_total,
            "answers_with_sources": source_count_total,
            "source_coverage": round(
                (source_count_total / assistant_count_total) if assistant_count_total else 0, 4
            ),
            "conversations_with_feedback": Conversation.objects.filter(
                messages__feedback__isnull=False
            ).distinct().count(),
        }

        # --- Pagination ---
        total_count = qs.count()
        total_pages = max(1, (total_count + per_page - 1) // per_page)
        page = min(page, total_pages)
        offset = (page - 1) * per_page
        page_qs = qs[offset: offset + per_page]

        return Response(
            {
                "summary": summary,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total_count,
                    "total_pages": total_pages,
                },
                "recent": [_serialize_admin_conversation(item) for item in page_qs],
            },
            status=status.HTTP_200_OK,
        )


class ConversationManageAPIView(AdminOnlyAPIView):
    """Archive, restore, or delete a conversation by id."""

    def get(self, request, conversation_id: int):
        try:
            conv = Conversation.objects.select_related("user").get(pk=conversation_id)
        except Conversation.DoesNotExist:
            return Response({"detail": "Conversation introuvable."}, status=status.HTTP_404_NOT_FOUND)

        return Response(
            {
                "id": conv.id,
                "user": conv.user.get_full_name().strip() or conv.user.email or conv.user.username,
                "title": (conv.title or "Nouvelle conversation").strip(),
                "is_archived": conv.is_archived,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "messages": [_serialize_message(message) for message in conv.messages.all().select_related("feedback").order_by("created_at", "id")],
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request, conversation_id: int):
        action = request.data.get("action", "")  # archive | restore | delete
        try:
            conv = Conversation.objects.select_related("user").get(pk=conversation_id)
        except Conversation.DoesNotExist:
            return Response({"detail": "Conversation introuvable."}, status=status.HTTP_404_NOT_FOUND)

        if action == "archive":
            conv.is_archived = True
            conv.save(update_fields=["is_archived", "updated_at"])
            return Response({"detail": "Conversation archivée.", "id": conversation_id}, status=status.HTTP_200_OK)
        elif action == "restore":
            conv.is_archived = False
            conv.save(update_fields=["is_archived", "updated_at"])
            return Response({"detail": "Conversation restaurée.", "id": conversation_id}, status=status.HTTP_200_OK)
        elif action == "delete":
            conv.delete()
            return Response({"detail": "Conversation supprimée.", "id": conversation_id}, status=status.HTTP_200_OK)
        else:
            return Response(
                {"detail": "Action invalide. Utilisez : archive, restore, delete."},
                status=status.HTTP_400_BAD_REQUEST,
            )


def _drive_dir() -> Path:
    runtime = get_runtime_settings()
    drive_dir = Path(runtime.rag_raw_drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    return drive_dir


def _serialize_drive_document(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
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


class DriveDocumentDownloadAPIView(AdminOnlyAPIView):

    def get(self, request, filename: str):
        target = _safe_drive_path(filename)
        if target is None:
            return Response({"detail": "Nom de fichier invalide."}, status=status.HTTP_400_BAD_REQUEST)
        if not target.exists() or not target.is_file():
            return Response({"detail": "Document introuvable."}, status=status.HTTP_404_NOT_FOUND)

        try:
            return FileResponse(
                open(target, "rb"),
                as_attachment=True,
                filename=target.name,
            )
        except Exception as exc:
            return Response(
                {"detail": f"Impossible d'ouvrir le fichier : {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DriveRebuildAPIView(AdminOnlyAPIView):

    def post(self, request):
        clear_cache = bool(request.data.get("clear_cache", False) if isinstance(request.data, dict) else False)
        if clear_cache:
            try:
                settings = get_runtime_settings()
                processed_dir = Path(settings.rag_processed_drive_dir)
                cache_file = Path(settings.rag_cache_dir / "file_cache_drive.json")
                if processed_dir.exists():
                    for item in processed_dir.iterdir():
                        if item.is_file():
                            try:
                                item.unlink()
                            except Exception:
                                pass
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Erreur lors de la réinitialisation du cache drive: %s", exc)

        try:
            processing_result = run_processing(corpus="drive")
            indexing_result = run_indexing(corpus="published", publish=True)
        except Exception as exc:
            logger.exception("Erreur pendant le rebuild drive")
            return Response(
                {"detail": f"Echec du rebuild drive: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        detail_msg = "Corpus drive réinitialisé et index reconstruit avec succès." if clear_cache else "Corpus drive retraité et index publié avec succès."
        return Response(
            {
                "detail": detail_msg,
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


class BenchmarkDeleteQuestionAPIView(AdminOnlyAPIView):
    def post(self, request):
        benchmark = request.data.get("benchmark", "drive")
        question = request.data.get("question")
        if not question:
            return Response({"detail": "La question est requise."}, status=status.HTTP_400_BAD_REQUEST)
        
        dataset = load_benchmark_set(benchmark)
        original_len = len(dataset)
        dataset = [item for item in dataset if item.get("question") != question]
        if len(dataset) == original_len:
            return Response({"detail": "Question introuvable."}, status=status.HTTP_404_NOT_FOUND)
        
        save_benchmark_set(benchmark, dataset)
        return Response({"detail": "Question supprimée avec succès."}, status=status.HTTP_200_OK)


class BenchmarkAddQuestionAPIView(AdminOnlyAPIView):
    def post(self, request):
        benchmark = request.data.get("benchmark", "drive")
        question = request.data.get("question", "").strip()
        expected_service = request.data.get("expected_service", "").strip()
        keywords_str = request.data.get("keywords", "").strip()
        
        if not question or not expected_service:
            return Response({"detail": "La question et le service attendu sont requis."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Split keywords by commas and clean them
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        dataset = load_benchmark_set(benchmark)
        # Check if question already exists
        if any(item.get("question") == question for item in dataset):
            return Response({"detail": "Cette question existe déjà dans ce benchmark."}, status=status.HTTP_400_BAD_REQUEST)
            
        dataset.append({
            "question": question,
            "expected_service": expected_service,
            "keywords": keywords,
            "expected_doc_types": [],  # Optional, default empty
        })
        save_benchmark_set(benchmark, dataset)
        return Response({"detail": "Question ajoutée avec succès au benchmark."}, status=status.HTTP_200_OK)


class BenchmarkTestQuestionAPIView(AdminOnlyAPIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        expected_service = request.data.get("expected_service", "").strip()
        keywords_str = request.data.get("keywords", "").strip()
        
        if not question:
            return Response({"detail": "La question est requise."}, status=status.HTTP_400_BAD_REQUEST)
            
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        try:
            metrics = _retrieval_metrics(question, keywords, [], top_k=5, expected_service=expected_service)
            row = {
                "question": question,
                "expected_service": expected_service,
                "top1_service": metrics.get("top1_service", "-"),
                "service_top1_match": metrics.get("service_top1_match", 0),
                "precision_at_k": metrics.get("precision_at_k", 0.0),
                "latency_ms": metrics.get("latency_ms", 0.0),
                "top1_source": metrics.get("top1_source", "-"),
            }
            return Response(row, status=status.HTTP_200_OK)
        except Exception as exc:
            logger.exception("Erreur lors du test de la question")
            return Response({"detail": f"Erreur pendant l'évaluation de la question: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


def _run_admin_audit_task(task: str) -> dict:
    if task == "data_audit":
        report = data_audit.build_report()
        paths = data_audit.write_report(report)
        return {
            "label": "Data audit",
            "report_paths": {key: str(value) for key, value in paths.items()},
            "report": latest_report_payload("data_audit"),
        }
    if task == "raw_quality":
        outputs = raw_quality_pipeline.run(audit_only=True)
        return {
            "label": "Raw quality audit",
            "report_paths": {
                group: {key: str(value) for key, value in paths.items()}
                for group, paths in outputs.items()
            },
            "report": latest_report_payload("raw_quality_audit"),
        }
    if task == "rag_eval":
        paths = run_evaluation(top_k=5, skip_generation=True, benchmark="drive")
        return {
            "label": "Benchmark drive",
            "report_paths": paths,
            "report": latest_report_payload("rag_eval"),
        }
    if task == "rag_eval_full":
        paths = run_evaluation(top_k=5, skip_generation=False, benchmark="drive")
        return {
            "label": "Benchmark drive complet",
            "report_paths": paths,
            "report": latest_report_payload("rag_eval"),
        }
    if task == "context_eval":
        paths = run_evaluation(top_k=5, skip_generation=True, benchmark="context")
        return {
            "label": "Benchmark contextuel",
            "report_paths": paths,
            "report": latest_report_payload("rag_eval"),
        }
    raise ValueError("Tache d'audit inconnue.")


def _serialize_admin_conversation(conversation: Conversation) -> dict:
    last_message = conversation.messages.order_by("-created_at", "-id").first()
    return {
        "id": conversation.id,
        "user": (
            conversation.user.get_full_name().strip()
            or conversation.user.email
            or conversation.user.username
        ),
        "title": (conversation.title or "Nouvelle conversation").strip(),
        "message_count": int(getattr(conversation, "message_count", 0) or 0),
        "assistant_count": int(getattr(conversation, "assistant_count", 0) or 0),
        "source_answer_count": int(getattr(conversation, "source_answer_count", 0) or 0),
        "feedback_count": int(getattr(conversation, "feedback_count", 0) or 0),
        "is_archived": conversation.is_archived,
        "last_message_at": (
            getattr(conversation, "last_message_at", None).isoformat()
            if getattr(conversation, "last_message_at", None)
            else conversation.updated_at.isoformat()
        ),
        "preview": str(last_message.content if last_message else "")[:160],
    }


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
    feedback_data = None
    try:
        if message.feedback:
            feedback_data = {
                "rating": message.feedback.rating,
                "comment": message.feedback.comment,
            }
    except MessageFeedback.DoesNotExist:
        pass

    return {
        "id": message.id,
        "role": message.role,
        "content": _sanitize_answer_text(message.content) if message.role == Message.ROLE_ASSISTANT else message.content,
        "sources": _sanitize_sources(message.sources or []),
        "confidence": message.confidence,
        "retrieval_meta": dict(message.retrieval_meta or {}),
        "created_at": message.created_at.isoformat(),
        "feedback": feedback_data,
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


def _save_assistant_message(
    conversation: Conversation,
    result,
    *,
    answer: str | None = None,
    sources: list[dict] | None = None,
) -> Message:
    return Message.objects.create(
        conversation=conversation,
        role=Message.ROLE_ASSISTANT,
        content=(answer or result.answer or "").strip(),
        sources=list(sources if sources is not None else result.sources or []),
        confidence=str(result.confidence or ""),
        retrieval_meta=dict(result.retrieval_meta or {}),
    )


def _sanitize_answer_text(raw_answer: str, question: str = "") -> str:
    text = str(raw_answer or "").strip()
    if not text:
        return ""

    lead_match = ANSWER_LEAD_RE.search(text)
    if lead_match:
        text = text[lead_match.end():].lstrip(" \n:-")

    section_match = ANSWER_SECTION_BREAK_RE.search(text)
    if section_match:
        text = text[:section_match.start()].rstrip()

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*#{1,6}\s*", "", line).rstrip()
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text or "Information non disponible dans mes sources actuelles."
    if question:
        text = _organize_answer_text(question, text)
    return text


def _organize_answer_text(question: str, answer: str) -> str:
    normalized_question = question.lower()
    if "information non disponible" in answer.lower():
        return answer

    if _is_ucastudent_attestation_question(normalized_question):
        specialized = _format_ucastudent_attestation_answer(answer)
        if specialized:
            return specialized

    if _is_procedure_question(normalized_question):
        lines = _extract_informative_lines(answer, normalized_question)
        if lines:
            topic = _question_topic_label(question)
            ordered = "\n".join(f"{index}. {line}" for index, line in enumerate(lines[:4], start=1))
            return f"Pour {topic} :\n{ordered}"

    return answer


def _is_procedure_question(normalized_question: str) -> bool:
    return any(token in normalized_question for token in ("comment", "obtenir", "demande", "demander", "procedure"))


def _is_ucastudent_attestation_question(normalized_question: str) -> bool:
    return (
        "attestation" in normalized_question
        and any(token in normalized_question for token in ("uc@student", "ucastudent", "student"))
    )


def _question_topic_label(question: str) -> str:
    clean = question.strip().rstrip(" ?")
    lower = clean.lower()
    if lower.startswith("comment "):
        return clean[len("comment "):]
    return clean[:1].lower() + clean[1:] if clean else "effectuer cette demarche"


def _extract_informative_lines(answer: str, normalized_question: str) -> list[str]:
    query_tokens = {
        token for token in re.findall(r"\b[\w@']+\b", normalized_question)
        if len(token) >= 4 and token not in {"comment", "obtenir", "votre", "cette", "demande", "faire", "avec"}
    }
    action_tokens = {"connect", "demande", "demandes", "certificat", "attestation", "releve", "ligne", "telecharg", "suivi"}
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in answer.splitlines():
        line = LIST_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        line = re.sub(r"\s+", " ", line)
        lower = line.lower()
        if "confiance:" in lower or "sources utiles" in lower:
            continue
        if "espace d’administration" in lower or "espace d'administration" in lower:
            continue
        if "les etablissements" in lower or "responsables administratifs" in lower:
            continue
        if "avantages cles" in lower or "objectifs de la plateforme" in lower:
            continue
        if query_tokens and not any(token in lower for token in query_tokens) and not any(token in lower for token in action_tokens):
            continue
        normalized = lower.strip(" .")
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line.rstrip(".") + ".")
    return lines


def _format_ucastudent_attestation_answer(answer: str) -> str:
    lines = _extract_informative_lines(answer, "attestation ucastudent")
    admin_line = next(
        (
            line for line in lines
            if "demandes administratives" in line.lower()
            or ("certificat" in line.lower() and "ligne" in line.lower())
        ),
        "",
    )
    if not admin_line:
        return ""

    admin_line = re.sub(r"^demandes administratives\s*", "", admin_line, flags=re.IGNORECASE).strip()
    admin_line = admin_line.lstrip(":;- ").strip()
    admin_line = admin_line[:1].lower() + admin_line[1:] if admin_line else admin_line
    if admin_line.lower().startswith("effectuez "):
        admin_line = "d'effectuer " + admin_line[len("effectuez "):].lstrip()
    intro = (
        "D'apres les sources disponibles, l'attestation se demande en ligne depuis la rubrique "
        "\"Demandes administratives\" de UC@Student."
    )
    detail = admin_line.rstrip(".")
    detail_sentence = f"Cette rubrique permet {detail}." if detail else ""
    steps = [
        "Connectez-vous a UC@Student.",
        "Ouvrez la rubrique \"Demandes administratives\".",
        "Selectionnez la demande d'attestation ou de certificat si elle est proposee a votre niveau.",
    ]
    note = (
        "Remarque: les extraits recuperes ne detaillent pas davantage le chemin complet, les delais ou les pieces a fournir."
    )
    body = [intro]
    if detail_sentence:
        body.append(detail_sentence)
    body.append("")
    body.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    body.append("")
    body.append(note)
    return "\n".join(body).strip()


def _sanitize_sources(raw_sources) -> list[dict]:
    if not isinstance(raw_sources, list):
        return []

    cleaned: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw_source in raw_sources[:3]:
        if not isinstance(raw_source, dict):
            continue
        label = _source_label(raw_source)
        if not label:
            continue
        url = str(raw_source.get("url") or raw_source.get("official_url") or "").strip()
        path = str(raw_source.get("path") or "").strip()
        key = (label.lower(), url or path)
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "name": label,
            "path": path,
        }
        if url:
            entry["url"] = url
        score = raw_source.get("score")
        if score is not None:
            entry["score"] = score
        cleaned.append(entry)
    return cleaned


def _source_label(source: dict) -> str:
    service_name = str(source.get("service_name") or "").strip().lower()
    if service_name in SERVICE_LABELS:
        return SERVICE_LABELS[service_name]

    for key in ("label", "title", "page_title"):
        value = str(source.get(key) or "").strip()
        if value:
            return value

    for key in ("url", "official_url"):
        value = str(source.get(key) or "").strip()
        if value:
            host = (urlparse(value).netloc or "").strip()
            if host:
                return host

    raw_name = str(source.get("name") or source.get("path") or "").strip()
    if not raw_name:
        return ""

    filename = Path(raw_name).name
    filename = HASHED_FILENAME_RE.sub("", filename)
    filename = re.sub(r"\.(html?|pdf|docx?|md|txt)$", "", filename, flags=re.IGNORECASE)
    filename = filename.replace("_", " ").replace("-", " ").strip()
    filename = re.sub(r"\s{2,}", " ", filename).strip()
    return filename or raw_name


class MessageFeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, message_id: int):
        rating = request.data.get("rating")
        comment = request.data.get("comment", "").strip()

        if rating not in [MessageFeedback.RATING_UP, MessageFeedback.RATING_DOWN]:
            return Response(
                {"detail": "La note doit être 'up' ou 'down'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        message = get_object_or_404(
            Message.objects.filter(conversation__user=request.user, role=Message.ROLE_ASSISTANT),
            pk=message_id,
        )

        feedback, created = MessageFeedback.objects.update_or_create(
            message=message,
            defaults={"rating": rating, "comment": comment},
        )

        return Response(
            {
                "detail": "Feedback enregistré avec succès.",
                "rating": feedback.rating,
                "comment": feedback.comment,
            },
            status=status.HTTP_200_OK,
        )
