from django.contrib import admin

from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    fields = ("role", "content", "confidence", "created_at")
    readonly_fields = ("role", "content", "confidence", "created_at")
    can_delete = False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "is_archived", "updated_at", "created_at")
    list_filter = ("is_archived", "created_at", "updated_at")
    search_fields = ("user__username", "user__email", "title")
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "confidence", "created_at")
    list_filter = ("role", "confidence", "created_at")
    search_fields = ("conversation__user__email", "content")
