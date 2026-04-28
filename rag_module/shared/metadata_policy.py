import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LANG_ALLOWLIST = {
    lang.strip().lower()
    for lang in os.getenv("RAG_LANG_ALLOWLIST", "fr,ar,en").split(",")
    if lang.strip()
}

AUDIENCE_RULES = {
    "etudiants": ["etudiant", "étudiant", "student", "bachelier", "laureat"],
    "enseignants": ["enseignant", "professeur", "prof", "chercheur", "doctorant"],
    "personnel": ["personnel", "fonctionnaire", "staff", "administratif"],
    "chercheurs": ["chercheur", "doctorant", "laboratoire", "hpc"],
}

SERVICE_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("scolarite", ["scolarite", "inscription", "diplome", "pedoc", "attestation", "note", "releve", "reinscription", "e-candidature"]),
    ("pedagogie_numerique", ["ucaplat", "cours en ligne", "devoir", "examen en ligne", "moodle", "module pedagogique"]),
    ("recherche", ["recherche", "projet", "hpc", "calcul", "laboratoire"]),
    ("rh", ["rh", "personnel", "pucastaff", "gestion", "rh"]),
    ("infrastructure", ["infrastructure", "salle", "conference", "equipement", "centre"]),
    ("vie_etudiante", ["club", "mobilite", "bourse", "oeuvre", "sociale", "sport"]),
    ("digital_service", ["cip", "compte numerique", "mot de passe", "authentification", "plateforme digitale"]),
]

SERVICE_NAME_RULES = {
    "UC@Student": ["uc@student", "ucastudent"],
    "Espace Diplômes": ["espace diplome", "espace diplôme"],
    "Clubs des étudiants": ["club étudiant", "clubs étudiants", "club uca"],
    "Mobilité internationale": ["mobilite internationale", "mobilité internationale"],
    "PEDOC": ["pedoc"],
    "PUCAStaff": ["pucastaff"],
    "Club UCA": ["club uca", "oeuvre sociale"],
    "Centre de conférences": ["centre de conference", "centre de conférences"],
    "HPC UCA": ["hpc", "calcul haute performance"],
    "UCAPLAT": ["ucaplat"],
    "Appels à Projets": ["appels à projets", "appel à projet"],
    "Soutien-Recherche": ["soutien-recherche", "soutien recherche"],
    "CIP": ["cip", "centre d'innovation pedagogique"],
}

SERVICE_OFFICIAL_URLS = {
    "UC@Student": "https://ucastudent.uca.ma",
    "Espace Diplômes": "https://diplomes.uca.ma",
    "Clubs des étudiants": "https://clubs.uca.ma",
    "Mobilité internationale": "https://mobilite.uca.ma",
    "PEDOC": "https://pedoc.uca.ma",
    "PUCAStaff": "https://pucastaff.uca.ma",
    "Club UCA": "https://club.uca.ma",
    "Centre de conférences": "https://conferences.uca.ma",
    "HPC UCA": "https://hpc.uca.ma",
    "UCAPLAT": "https://ucaplat.uca.ma",
    "Appels à Projets": "https://projets.uca.ma",
    "Soutien-Recherche": "https://recherche.uca.ma",
    "CIP": "https://cip.uca.ma",
}

SERVICE_MAIN_ACTIONS = {
    "UC@Student": ["demander une attestation", "consulter les notes", "suivre le cursus", "réinscription"],
    "Espace Diplômes": ["authentifier un diplôme", "télécharger e-diplôme", "demander un duplicata"],
    "Clubs des étudiants": ["créer un club", "rejoindre un club", "demander une subvention"],
    "Mobilité internationale": ["postuler à une bourse", "consulter les offres", "soumettre un dossier"],
    "PEDOC": ["déposer une thèse", "suivre l'état d'avancement", "valider les documents"],
    "PUCAStaff": ["demander un congé", "télécharger bulletin de paie", "suivre carrière"],
    "Club UCA": ["réserver un espace", "consulter les événements", "s'inscrire aux activités"],
    "Centre de conférences": ["réserver une salle", "consulter le planning", "demander un équipement"],
    "HPC UCA": ["demander un accès", "lancer des calculs", "consulter les ressources"],
    "UCAPLAT": ["suivre des cours", "passer des examens", "déposer des devoirs"],
    "Appels à Projets": ["soumettre un projet", "consulter les résultats", "demander un financement"],
    "Soutien-Recherche": ["demander un soutien", "justifier des dépenses", "suivre l'aide"],
    "CIP": ["réserver du matériel pédagogique", "demander un accompagnement", "consulter les guides"],
}

SERVICE_WORKFLOW_STEPS = {
    "PEDOC": ["1. Connexion", "2. Dépôt de la version initiale", "3. Validation par le directeur", "4. Dépôt final"],
    "Mobilité internationale": ["1. Choix du programme", "2. Dépôt du dossier", "3. Entretien", "4. Validation"],
    "UC@Student": ["1. Connexion via compte académique", "2. Choix de la scolarité", "3. Soumission de la demande"],
    "UCAPLAT": ["1. Connexion", "2. Accès au cours", "3. Dépôt ou évaluation"],
    "CIP": ["1. Connexion", "2. Choix du service", "3. Validation de la demande"],
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NORMALIZED_SERVICE_TYPE_RULES: List[Tuple[str, List[str]]] = [
    (doc_type, [normalize_text(keyword) for keyword in keywords])
    for doc_type, keywords in SERVICE_TYPE_RULES
]


def canonical_file_type(raw_file_type: str, source_path: str) -> str:
    value = (raw_file_type or "").strip().lower().lstrip(".")
    if value:
        return value
    suffix = Path(source_path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def detect_audience(source_path: str, text: str) -> str:
    haystack = normalize_text(f"{source_path} {text}")
    for target, keywords in AUDIENCE_RULES.items():
        if any(normalize_text(keyword) in haystack for keyword in keywords):
            return target
    return "general"


def detect_service_type(source_path: str, text: str) -> str:
    haystack = normalize_text(f"{source_path} {text}")
    for service_type, keywords in NORMALIZED_SERVICE_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return service_type
    return "general"

def detect_service_name(source_path: str, text: str) -> str:
    haystack = normalize_text(f"{source_path} {text}")
    for name, keywords in SERVICE_NAME_RULES.items():
        if any(normalize_text(keyword) in haystack for keyword in keywords):
            return name
    return "unknown"


def detect_year(source_path: str, text: str) -> Optional[int]:
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{source_path} {text}")
    if not years:
        return None
    year_values = sorted({int(year) for year in years})
    return year_values[-1]


def prepare_chunk_metadata(chunk: Dict, source_path: str) -> Optional[Dict]:
    text = (chunk.get("text", "") or "").strip()
    if not text:
        return None

    metadata = dict(chunk.get("metadata", {}) or {})
    language = (metadata.get("language", "unknown") or "unknown").lower()
    if language not in LANG_ALLOWLIST:
        return None

    file_type = canonical_file_type(str(metadata.get("file_type", "")), source_path)
    audience = detect_audience(source_path, text)
    service_type = detect_service_type(source_path, text)
    service_name = detect_service_name(source_path, text)
    year = detect_year(source_path, text)

    metadata["file_type"] = file_type
    metadata["target_audience"] = audience
    metadata["service_type"] = service_type
    metadata["service_name"] = service_name
    metadata["official_url"] = SERVICE_OFFICIAL_URLS.get(service_name, "")
    metadata["main_actions"] = SERVICE_MAIN_ACTIONS.get(service_name, [])
    metadata["workflow_steps"] = SERVICE_WORKFLOW_STEPS.get(service_name, [])
    # Map document_type and faculty to defaults to avoid breaking older pipeline logic
    metadata["document_type"] = service_type
    metadata["faculty"] = "UCA"
    if year is not None:
        metadata["year"] = year
    else:
        metadata.pop("year", None)

    updated_chunk = dict(chunk)
    updated_chunk["metadata"] = metadata
    return updated_chunk
