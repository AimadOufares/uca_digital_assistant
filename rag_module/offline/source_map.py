from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceRule:
    name: str
    hosts: tuple[str, ...]
    path_keywords: tuple[str, ...]
    priority: str
    category: str
    default_corpus: str
    premium: bool = False


PRIORITY_LEVELS = {"A": 100, "B": 70, "C": 40}

RULES: tuple[SourceRule, ...] = (
    SourceRule("ucastudent", ("ucastudent.uca.ma",), ("",), "A", "scolarite", "main", True),
    SourceRule("reins", ("reins.uca.ma",), ("",), "A", "reinscription", "main", True),
    SourceRule("e-candidature", ("e-candidature.uca.ma",), ("",), "A", "candidature", "main", True),
    SourceRule("onousc-student", ("onousc.ma", "www.onousc.ma"), ("bourse", "restaurant", "logement", "etudiant", "centres-medicaux"), "A", "bourse", "main"),
    SourceRule("uca-admin", ("uca.ma", "www.uca.ma", "fsjes.uca.ma", "flsh.uca.ma", "ensa-marrakech.uca.ma", "fmpm.uca.ma", "fps.uca.ma", "ensas.uca.ma", "ests.uca.ma", "estk.uca.ma", "fstg-marrakech.ac.ma"), ("scolarite", "admission", "inscription", "candidature", "contact", "attestation", "note", "releve", "emploi-du-temps", "calendrier", "reglement"), "A", "scolarite", "main"),
    SourceRule("uca-formation", ("uca.ma", "www.uca.ma", "fsjes.uca.ma", "flsh.uca.ma", "ensa-marrakech.uca.ma", "fmpm.uca.ma", "fps.uca.ma", "ensas.uca.ma", "ests.uca.ma", "estk.uca.ma", "fstg-marrakech.ac.ma"), ("master", "licence", "doctorat", "filiere", "programme", "orientation"), "B", "formation", "archive"),
    SourceRule("enssup-student", ("enssup.gov.ma", "www.enssup.gov.ma"), ("etudiant", "student"), "B", "formation", "archive"),
    SourceRule("institutionnel", ("uca.ma", "www.uca.ma", "fsjes.uca.ma", "flsh.uca.ma", "ensa-marrakech.uca.ma", "fmpm.uca.ma", "fps.uca.ma", "ensas.uca.ma", "ests.uca.ma", "estk.uca.ma", "fstg-marrakech.ac.ma"), ("recherche", "laboratoire", "laboratoires", "colloque", "conference", "partenariat", "gouvernance", "presidence"), "C", "recherche", "archive"),
)

PREMIUM_URLS: tuple[str, ...] = (
    "https://ucastudent.uca.ma/",
    "https://reins.uca.ma/",
    "https://e-candidature.uca.ma/",
    "https://www.onousc.ma/Bourses",
    "https://www.onousc.ma/Acces-aux-restaurants-universitaires",
    "https://www.onousc.ma/Centres-medicaux",
)

FAST_PROFILE = {
    "mode": "fast",
    "max_depth": 2,
    "max_total_urls": 2500,
    "max_urls_per_domain": 700,
    "max_urls_per_subdomain": 250,
    "allowed_priorities": {"A", "B"},
}

EXTENDED_PROFILE = {
    "mode": "extended",
    "max_depth": 4,
    "max_total_urls": 6000,
    "max_urls_per_domain": 1800,
    "max_urls_per_subdomain": 700,
    "allowed_priorities": {"A", "B", "C"},
}


def get_profile(mode: str) -> dict:
    return dict(FAST_PROFILE if (mode or "fast").lower() == "fast" else EXTENDED_PROFILE)


def default_seeds_for_mode(mode: str, premium_only: bool = False) -> list[str]:
    if premium_only:
        return list(PREMIUM_URLS)

    allowed = get_profile(mode)["allowed_priorities"]
    seeds = list(PREMIUM_URLS)
    for rule in RULES:
        if rule.priority not in allowed:
            continue
        for host in rule.hosts:
            root = f"https://{host}/"
            if root not in seeds:
                seeds.append(root)
    return seeds


def match_source_rule(url: str) -> dict:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    haystack = f"{host}{parsed.path}".lower()

    best_rule = None
    best_score = -1
    for rule in RULES:
        if not any(host == candidate or host.endswith(f".{candidate}") for candidate in rule.hosts):
            continue
        matched_keyword = rule.path_keywords == ("",) or any(keyword in haystack for keyword in rule.path_keywords)
        if not matched_keyword:
            continue
        score = PRIORITY_LEVELS.get(rule.priority, 0)
        if rule.premium:
            score += 50
        if rule.path_keywords == ("",):
            score += 5
        else:
            score += 25
        if score > best_score:
            best_rule = rule
            best_score = score

    if best_rule is None:
        return {
            "rule_name": "fallback",
            "source_priority": "C",
            "document_category": "autre",
            "default_corpus": "archive",
            "is_premium": url in PREMIUM_URLS,
        }

    return {
        "rule_name": best_rule.name,
        "source_priority": best_rule.priority,
        "document_category": best_rule.category,
        "default_corpus": best_rule.default_corpus,
        "is_premium": best_rule.premium or url in PREMIUM_URLS,
    }
