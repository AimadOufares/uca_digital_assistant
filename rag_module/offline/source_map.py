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
    SourceRule(
        "uca-core-student",
        ("www.uca.ma", "uca.ma"),
        ("inscription", "reinscription", "candidature", "attestation", "bourse", "resultat", "calendrier"),
        "A",
        "scolarite",
        "main",
        True,
    ),
    SourceRule("pole-digitalisation", ("pole-digitalisation.uca.ma",), ("",), "A", "digital_service", "main", True),
    SourceRule("ucastudent", ("ucastudent.uca.ma",), ("",), "A", "scolarite", "main", True),
    SourceRule("pedoc", ("pedoc.uca.ma",), ("",), "A", "scolarite", "main", True),
    SourceRule("ucaplat", ("ucaplat.uca.ma",), ("",), "A", "digital_service", "main", True),
    SourceRule("cip", ("cip.uca.ma",), ("",), "A", "digital_service", "main", True),
    SourceRule("diplomes", ("diplomes.uca.ma",), ("",), "A", "attestation", "main", True),
    SourceRule("pucastaff", ("pucastaff.uca.ma",), ("",), "A", "rh", "main", True),
    SourceRule("reins", ("reins.uca.ma",), ("",), "B", "reinscription", "main", True),
    SourceRule("e-candidature", ("e-candidature.uca.ma",), ("",), "B", "candidature", "main", True),
    SourceRule("mobilite", ("mobilite.uca.ma",), ("",), "B", "bourse", "main", True),
    SourceRule("clubs", ("clubs.uca.ma",), ("",), "B", "vie_etudiante", "main", True),
)

PREMIUM_URLS: tuple[str, ...] = (
    "https://pole-digitalisation.uca.ma/",
    "https://ucastudent.uca.ma/",
    "https://pedoc.uca.ma/",
    "https://ucaplat.uca.ma/",
    "https://cip.uca.ma/",
    "https://diplomes.uca.ma/",
    "https://mobilite.uca.ma/",
    "https://pucastaff.uca.ma/",
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
