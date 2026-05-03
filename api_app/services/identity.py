import re
from typing import Iterable

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

from rag_module.shared.runtime import get_runtime_settings


def normalize_student_email(email: str) -> str:
    return (email or "").strip().lower()


def allowed_uca_email_domains() -> list[str]:
    runtime = get_runtime_settings()
    domains = [str(domain).strip().lower() for domain in runtime.uca_allowed_email_domains if str(domain).strip()]
    return list(dict.fromkeys(domains))


def email_domain(email: str) -> str:
    normalized = normalize_student_email(email)
    if "@" not in normalized:
        return ""
    return normalized.rsplit("@", 1)[1]


def is_allowed_uca_email(email: str, allowed_domains: Iterable[str] | None = None) -> bool:
    domains = [str(item).strip().lower() for item in (allowed_domains or allowed_uca_email_domains()) if str(item).strip()]
    if not domains:
        return False
    return email_domain(email) in set(domains)


def validate_uca_email(email: str) -> str:
    normalized = normalize_student_email(email)
    if not normalized:
        raise ValidationError("L'adresse email UCA est obligatoire.")
    if "@" not in normalized:
        raise ValidationError("Veuillez saisir une adresse email valide.")
    if not is_allowed_uca_email(normalized):
        allowed = ", ".join(allowed_uca_email_domains()) or "aucun domaine configure"
        raise ValidationError(f"Inscription reservee aux emails UCA autorises : {allowed}.")
    return normalized


def email_already_exists(email: str) -> bool:
    normalized = normalize_student_email(email)
    if not normalized:
        return False
    user_model = get_user_model()
    return user_model.objects.filter(email__iexact=normalized).exists()


def build_unique_username_from_email(email: str) -> str:
    normalized = normalize_student_email(email)
    local_part = normalized.split("@", 1)[0] if "@" in normalized else normalized
    slug = re.sub(r"[^a-z0-9._-]+", "-", local_part).strip(".-_") or "etudiant"
    user_model = get_user_model()
    candidate = slug
    suffix = 1
    while user_model.objects.filter(username__iexact=candidate).exists():
        suffix += 1
        candidate = f"{slug}-{suffix}"
    return candidate
