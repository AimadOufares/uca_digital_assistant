import re
import unicodedata
from typing import Dict, List, Optional


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


UCA_GLOBAL = "UCA_GLOBAL"

CANONICAL_ESTABLISHMENTS: List[str] = [
    "FSSM",
    "ENSA",
    "FSTG",
    "FSJES",
    "FLSH",
    "FMPM",
    "ENCG",
    "ENS",
]

ESTABLISHMENT_ALIASES: Dict[str, List[str]] = {
    "FSSM": [
        "fssm",
        "semlalia",
        "faculte des sciences semlalia",
        "faculte sciences semlalia",
        "faculte semlalia",
    ],
    "ENSA": [
        "ensa",
        "ensa marrakech",
        "ensa safi",
        "ecole nationale des sciences appliquees",
    ],
    "FSTG": [
        "fstg",
        "fst",
        "fst gueliz",
        "faculte des sciences et techniques",
        "faculte des sciences et techniques gueliz",
    ],
    "FSJES": [
        "fsjes",
        "faculte des sciences juridiques economiques et sociales",
        "faculte de droit",
    ],
    "FLSH": [
        "flsh",
        "faculte des lettres",
        "faculte des lettres et des sciences humaines",
    ],
    "FMPM": [
        "fmpm",
        "faculte de medecine",
        "faculte de medecine et de pharmacie",
    ],
    "ENCG": [
        "encg",
        "ecole nationale de commerce et de gestion",
    ],
    "ENS": [
        "ecole normale superieure",
        "ens marrakech",
    ],
    UCA_GLOBAL: [
        "uca",
        "universite cadi ayyad",
        "universite cadi ayad",
        "cadi ayyad",
    ],
}

OUTSIDE_SCOPE_ALIASES = [
    "universite mohammed v",
    "mohammed v",
    "universite hassan ii",
    "hassan ii",
    "universite ibn zohr",
    "ibn zohr",
    "universite abdelmalek essaadi",
    "abdelmalek essaadi",
    "universite sultan moulay slimane",
    "sultan moulay slimane",
]

COMPARISON_KEYWORDS = [
    "difference",
    "differences",
    "comparer",
    "comparaison",
    "compare",
    "versus",
    "vs",
    "entre",
    "par rapport",
]

OTHER_ESTABLISHMENT_KEYWORDS = [
    "autre etablissement",
    "autres etablissements",
    "autre faculte",
    "autres facultes",
    "autre ecole",
    "autres ecoles",
]


def _word_bounded_match(text: str, alias: str) -> bool:
    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
    return re.search(pattern, text) is not None


def normalize_establishment_label(value: Optional[str]) -> Optional[str]:
    normalized = normalize_text(str(value or ""))
    if not normalized:
        return None

    for establishment in CANONICAL_ESTABLISHMENTS + [UCA_GLOBAL]:
        if establishment.lower() == normalized:
            return establishment

    for establishment, aliases in ESTABLISHMENT_ALIASES.items():
        for alias in aliases:
            if _word_bounded_match(normalized, normalize_text(alias)):
                return establishment
    return None


def get_metadata_establishment(metadata: Dict) -> str:
    if not metadata:
        return "unknown"
    raw_value = metadata.get("etablissement") or metadata.get("faculty") or metadata.get("establishment")
    normalized = normalize_establishment_label(str(raw_value or ""))
    return normalized or "unknown"


def detect_establishments_in_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    matches: List[str] = []
    ordered_establishments = [name for name in CANONICAL_ESTABLISHMENTS] + [UCA_GLOBAL]
    for establishment in ordered_establishments:
        aliases = ESTABLISHMENT_ALIASES.get(establishment, [])
        if any(_word_bounded_match(normalized, normalize_text(alias)) for alias in aliases):
            matches.append(establishment)
    return matches


def detect_primary_establishment(source_path: str, text: str) -> str:
    source_matches = detect_establishments_in_text(source_path)
    specific_source_matches = [match for match in source_matches if match != UCA_GLOBAL]
    if specific_source_matches:
        return specific_source_matches[0]

    text_matches = detect_establishments_in_text(text)
    specific_text_matches = [match for match in text_matches if match != UCA_GLOBAL]
    if specific_text_matches:
        return specific_text_matches[0]

    if UCA_GLOBAL in source_matches or UCA_GLOBAL in text_matches:
        return UCA_GLOBAL
    return "unknown"


def is_comparison_query(text: str) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in COMPARISON_KEYWORDS)


def references_other_establishment(text: str) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in OTHER_ESTABLISHMENT_KEYWORDS)


def is_outside_uca_query(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if detect_establishments_in_text(normalized):
        return False
    return any(alias in normalized for alias in OUTSIDE_SCOPE_ALIASES)


def allowed_establishments_for_resolution(resolution: Dict) -> List[str]:
    mode = str(resolution.get("mode") or "")
    targets = [target for target in resolution.get("target_establishments", []) if target]

    if mode == "single_target":
        return list(dict.fromkeys(targets + [UCA_GLOBAL]))
    if mode == "multi_target":
        return list(dict.fromkeys(targets + [UCA_GLOBAL]))
    if mode == "global_uca":
        return list(dict.fromkeys(CANONICAL_ESTABLISHMENTS + [UCA_GLOBAL]))
    return []


def build_clarification_message(resolution: Dict) -> str:
    user_establishment = resolution.get("user_establishment")
    targets = [target for target in resolution.get("target_establishments", []) if target]
    if user_establishment and targets:
        return (
            f"Parlez-vous de votre etablissement ({user_establishment}) "
            f"ou de {', '.join(targets)} ?"
        )
    if targets:
        return f"Pouvez-vous preciser l'etablissement cible entre {', '.join(targets)} ?"
    return "Veuillez preciser l'etablissement concerne par votre question."


def build_out_of_scope_message() -> str:
    return (
        "Je suis actuellement configure pour repondre aux questions portant sur "
        "les etablissements de l'Universite Cadi Ayyad."
    )


def resolve_context(message: str, user_establishment: Optional[str] = None) -> Dict:
    normalized_message = normalize_text(message)
    normalized_user = normalize_establishment_label(user_establishment)

    explicit_mentions = detect_establishments_in_text(normalized_message)
    explicit_specific = [item for item in explicit_mentions if item != UCA_GLOBAL]
    comparison_query = is_comparison_query(normalized_message)
    other_establishment_query = references_other_establishment(normalized_message)

    if is_outside_uca_query(normalized_message):
        return {
            "mode": "out_of_scope",
            "user_establishment": normalized_user,
            "target_establishments": [],
            "confidence": "high",
            "reason": "outside_uca_detected",
        }

    if len(explicit_specific) >= 2:
        mode = "multi_target" if comparison_query else "clarification"
        confidence = "high" if comparison_query else "low"
        return {
            "mode": mode,
            "user_establishment": normalized_user,
            "target_establishments": explicit_specific,
            "confidence": confidence,
            "reason": "multiple_explicit_targets",
        }

    if len(explicit_specific) == 1:
        return {
            "mode": "single_target",
            "user_establishment": normalized_user,
            "target_establishments": explicit_specific,
            "confidence": "high",
            "reason": "explicit_target_detected",
        }

    if UCA_GLOBAL in explicit_mentions:
        return {
            "mode": "global_uca",
            "user_establishment": normalized_user,
            "target_establishments": [UCA_GLOBAL],
            "confidence": "high",
            "reason": "uca_global_detected",
        }

    if other_establishment_query:
        return {
            "mode": "clarification",
            "user_establishment": normalized_user,
            "target_establishments": [],
            "confidence": "low",
            "reason": "other_establishment_unspecified",
        }

    if normalized_user:
        return {
            "mode": "single_target",
            "user_establishment": normalized_user,
            "target_establishments": [normalized_user],
            "confidence": "medium",
            "reason": "fallback_to_user_establishment",
        }

    return {
        "mode": "clarification",
        "user_establishment": None,
        "target_establishments": [],
        "confidence": "low",
        "reason": "missing_target_context",
    }
