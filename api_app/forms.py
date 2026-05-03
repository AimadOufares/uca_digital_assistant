from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from .services.identity import (
    build_unique_username_from_email,
    email_already_exists,
    normalize_student_email,
    validate_uca_email,
)


class StudentSignupForm(UserCreationForm):
    first_name = forms.CharField(max_length=150, label="Prenom")
    last_name = forms.CharField(max_length=150, label="Nom")
    email = forms.EmailField(label="Email UCA")

    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = ("first_name", "last_name", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, placeholder in {
            "first_name": "Prenom",
            "last_name": "Nom",
            "email": "prenom.nom@uca.ac.ma",
            "password1": "Mot de passe",
            "password2": "Confirmer le mot de passe",
        }.items():
            self.fields[field_name].widget.attrs.update({"placeholder": placeholder})

    def clean_email(self):
        email = validate_uca_email(self.cleaned_data.get("email", ""))
        if email_already_exists(email):
            raise forms.ValidationError("Un compte existe deja avec cet email.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        email = normalize_student_email(self.cleaned_data["email"])
        user.email = email
        user.username = build_unique_username_from_email(email)
        user.first_name = self.cleaned_data["first_name"].strip()
        user.last_name = self.cleaned_data["last_name"].strip()
        if commit:
            user.save()
        return user


class StudentLoginForm(AuthenticationForm):
    username = forms.CharField(label="Email UCA")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({"placeholder": "prenom.nom@uca.ac.ma", "autofocus": True})
        self.fields["password"].widget.attrs.update({"placeholder": "Mot de passe"})
