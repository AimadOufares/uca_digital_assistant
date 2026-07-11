from django.conf import settings
from django.db import models


class Conversation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations")
    title = models.CharField(max_length=255, blank=True)
    context_summary = models.TextField(blank=True)
    context_meta = models.JSONField(default=dict, blank=True)
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-created_at"]

    def __str__(self) -> str:
        label = self.title.strip() if self.title else f"Conversation {self.pk}"
        return f"{self.user} - {label}"


class Message(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = (
        (ROLE_USER, "Utilisateur"),
        (ROLE_ASSISTANT, "Assistant"),
    )

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    confidence = models.CharField(max_length=32, blank=True)
    retrieval_meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.conversation_id} - {self.role}"


class MessageFeedback(models.Model):
    RATING_UP = "up"
    RATING_DOWN = "down"
    RATING_CHOICES = (
        (RATING_UP, "Positif 👍"),
        (RATING_DOWN, "Négatif 👎"),
    )

    message = models.OneToOneField(
        Message,
        on_delete=models.CASCADE,
        related_name="feedback",
    )
    rating = models.CharField(max_length=8, choices=RATING_CHOICES)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.message_id} - {self.rating}"
