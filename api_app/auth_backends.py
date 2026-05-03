from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

from .services.identity import normalize_student_email


class EmailOrUsernameModelBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        user_model = get_user_model()
        username = username or kwargs.get(user_model.USERNAME_FIELD) or kwargs.get("email")
        if not username or not password:
            return None

        lookup_value = str(username).strip()
        user = None
        if "@" in lookup_value:
            normalized_email = normalize_student_email(lookup_value)
            try:
                user = user_model.objects.get(email__iexact=normalized_email)
            except user_model.DoesNotExist:
                return None
        else:
            try:
                user = user_model.objects.get(username__iexact=lookup_value)
            except user_model.DoesNotExist:
                return None

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
