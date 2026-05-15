import re
import unicodedata
from typing import Any


SERVICE_RULES = {
    "UC@Student": {"uc@student", "ucastudent", "student.uca", "uc student"},
    "UCAPLAT": {"ucaplat", "uca plat", "plateforme ucaplat", "plateforme pedagogique", "cours en ligne", "devoirs en ligne"},
    "PEDOC": {"pedoc"},
    "CIP": {"cip", "centre d innovation pedagogique", "centre innovation pedagogique"},
    "PUCAStaff": {"pucastaff", "puca staff"},
    "Espace Diplomes": {"diplome", "diplomes", "e-diplome", "e diplome"},
    "Mobilite internationale": {"mobilite internationale", "mobilite", "mobilite-internationale", "bourse", "bourse mobilite"},
    "HPC UCA": {"hpc", "calcul haute performance"},
    "Soutien-Recherche": {"soutien-recherche", "soutien recherche", "soutien a la recherche", "projet de recherche"},
    "Clubs des etudiants": {
        "clubs des etudiants",
        "clubs etudiants",
        "club etudiant",
        "association etudiante",
        "associations etudiantes",
        "vie associative",
    },
    "Centre de conferences": {"centre conferences", "centre de conferences", "centre-de-conferences"},
    "Appels a Projets": {"appels a projets", "appel a projets", "call uca", "call for projects"},
    "Club UCA": {"club uca", "club de l universite"},
}

INTENT_RULES = {
    "attestation": {"attestation", "certificat", "scolarite"},
    "notes": {"note", "notes", "releve", "releves"},
    "candidature": {"candidature", "candidater", "postuler", "admission"},
    "cours": {"cours", "module", "plateforme pedagogique"},
    "devoirs": {"devoir", "devoirs", "deposer", "depot"},
    "diplome": {"diplome", "diplomes", "e-diplome", "suivi diplome"},
    "conge": {"conge", "absence", "autorisation"},
    "procedure": {"procedure", "demande", "demander", "obtenir", "comment"},
    "delai": {"delai", "delais", "temps", "duree", "combien de temps"},
    "rubrique": {"rubrique", "page", "menu", "ou trouver", "où trouver"},
}

FOLLOW_UP_PATTERNS = (
    r"\bet pour\b",
    r"\bet les\b",
    r"\bet la\b",
    r"\bet le\b",
    r"\bce service\b",
    r"\bcette rubrique\b",
    r"\bcela\b",
    r"\bca\b",
    r"\bça\b",
    r"\bla\b",
    r"\ble\b",
    r"\bles\b",
    r"\bil\b",
    r"\belle\b",
)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("@", "@")
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\b[\w@']+\b", normalize_text(value)))


def detect_service(text: str) -> str:
    normalized = normalize_text(text)
    for service, aliases in SERVICE_RULES.items():
        if any(alias in normalized for alias in aliases):
            return service
    return ""


def detect_intent(text: str) -> str:
    normalized = normalize_text(text)
    for intent, aliases in INTENT_RULES.items():
        if any(alias in normalized for alias in aliases):
            return intent
    return ""


def extract_entities(text: str) -> list[str]:
    normalized = normalize_text(text)
    entities: list[str] = []
    for entity in (
        "attestation",
        "certificat",
        "notes",
        "releve",
        "candidature",
        "cours",
        "devoirs",
        "diplome",
        "bourse",
        "conge",
        "demandes administratives",
    ):
        if entity in normalized and entity not in entities:
            entities.append(entity)
    return entities[:8]


def is_follow_up_question(question: str) -> bool:
    normalized = normalize_text(question)
    token_count = len(_tokens(normalized))
    if token_count <= 5:
        return True
    return any(re.search(pattern, normalized) for pattern in FOLLOW_UP_PATTERNS)


def _recent_messages(conversation, limit: int = 5) -> list[dict[str, str]]:
    messages = list(conversation.messages.order_by("-created_at", "-id")[:limit])
    messages.reverse()
    return [{"role": message.role, "content": message.content} for message in messages]


def build_conversation_context(conversation, new_message: str) -> dict[str, Any]:
    previous_meta = dict(conversation.context_meta or {})
    detected_service = detect_service(new_message)
    detected_intent = detect_intent(new_message)
    detected_entities = extract_entities(new_message)

    previous_service = str(previous_meta.get("service") or "")
    previous_intent = str(previous_meta.get("intent") or previous_meta.get("main_topic") or "")
    has_previous_context = bool(previous_service or previous_intent or conversation.context_summary)
    service_changed = bool(detected_service and previous_service and detected_service != previous_service)
    follow_up = is_follow_up_question(new_message)
    context_used = bool(has_previous_context and follow_up and not service_changed)

    context_payload = {
        "original_question": new_message,
        "context_summary": conversation.context_summary or "",
        "context_meta": previous_meta,
        "recent_messages": _recent_messages(conversation),
        "detected_service": detected_service,
        "detected_intent": detected_intent,
        "detected_entities": detected_entities,
        "service_changed": service_changed,
        "is_follow_up": follow_up,
        "context_used": context_used,
    }
    context_payload["rewritten_question"] = rewrite_question_with_context(new_message, context_payload)
    return context_payload


def _topic_label(intent: str, entities: list[str]) -> str:
    if intent and intent not in {"procedure", "delai", "rubrique"}:
        return intent
    for entity in entities:
        if entity:
            return entity
    return intent or "la demande"


def rewrite_question_with_context(new_message: str, context_payload: dict[str, Any]) -> str:
    question = (new_message or "").strip()
    if not context_payload.get("context_used"):
        return question

    meta = dict(context_payload.get("context_meta") or {})
    service = str(meta.get("service") or context_payload.get("detected_service") or "").strip()
    intent = str(context_payload.get("detected_intent") or meta.get("intent") or meta.get("main_topic") or "").strip()
    entities = list(meta.get("entities") or []) + list(context_payload.get("detected_entities") or [])
    topic = _topic_label(intent, entities)
    normalized = normalize_text(question)

    if "delai" in normalized or "combien de temps" in normalized or "temps" in normalized:
        return f"Quel est le delai pour {topic} sur {service} ?".strip()
    if "rubrique" in normalized or "ou trouver" in normalized:
        return f"Ou trouver la rubrique liee a {topic} sur {service} ?".strip()
    if "telecharg" in normalized:
        return f"Comment telecharger {topic} sur {service} ?".strip()
    if "document" in normalized or "piece" in normalized:
        return f"Quels documents sont necessaires pour {topic} sur {service} ?".strip()

    if service and topic:
        return f"{question} Concernant {topic} sur {service}."
    if service:
        return f"{question} Concernant {service}."
    return question


def _summary_for(service: str, intent: str, entities: list[str]) -> str:
    parts = []
    if service:
        parts.append(f"service {service}")
    if intent:
        parts.append(f"intention {intent}")
    if entities:
        parts.append("elements " + ", ".join(entities[:4]))
    if not parts:
        return ""
    return "Contexte courant: " + "; ".join(parts) + "."


def update_conversation_context(
    conversation,
    user_message: str,
    assistant_answer: str,
    result,
    context_payload: dict[str, Any] | None = None,
) -> None:
    payload = context_payload or build_conversation_context(conversation, user_message)
    previous_meta = dict(conversation.context_meta or {})

    service = payload.get("detected_service") or ("" if payload.get("service_changed") else previous_meta.get("service", ""))
    intent = payload.get("detected_intent") or ("" if payload.get("service_changed") else previous_meta.get("intent", ""))
    entities = list(payload.get("detected_entities") or [])
    if not entities and not payload.get("service_changed"):
        entities = list(previous_meta.get("entities") or [])

    if not service and result is not None:
        for source in getattr(result, "sources", []) or []:
            service = detect_service(" ".join(str(source.get(key) or "") for key in ("service_name", "name", "path", "url")))
            if service:
                break

    if not intent:
        intent = detect_intent(" ".join([user_message, assistant_answer or ""]))

    next_meta = {
        "service": service or "",
        "intent": intent or "",
        "main_topic": intent or (entities[0] if entities else ""),
        "entities": entities[:8],
        "last_original_question": user_message,
        "last_rewritten_query": payload.get("rewritten_question") or user_message,
        "last_context_used": bool(payload.get("context_used")),
    }
    conversation.context_meta = next_meta
    conversation.context_summary = _summary_for(next_meta["service"], next_meta["intent"], next_meta["entities"])
    conversation.save(update_fields=["context_summary", "context_meta", "updated_at"])
