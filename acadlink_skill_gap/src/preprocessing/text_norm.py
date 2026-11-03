"""Text normalization utilities for the AcadLink Skill Gap project.

Used across all stages (skill normalization, occupation mapping, evidence
span search). Deterministic and dependency-light on purpose.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Abbreviation / synonym expansion (case-insensitive, applied on normalized text)
# Keep entries conservative: single-meaning tech terms. Ordered longest-first
# when expanded via regex to avoid prefix collisions.
# ---------------------------------------------------------------------------
ABBREVIATIONS: dict[str, str] = {
    "k8s": "kubernetes",
    "restful api": "rest api",
    "restful web service": "rest api",
    "restful services": "rest api",
    "machine learning": "ml",  # handled below via expansion back
    "natural language processing": "nlp",
    "ui": "user interface",
    "ux": "user experience",
    "api": "application programming interface",
    "sql": "structured query language",
    "oop": "object oriented programming",
    "db": "database",
    "dbms": "database management system",
    "dwh": "data warehouse",
    "etl": "extract transform load",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "cicd": "continuous integration and delivery",
    "mlops": "machine learning operations",
    "oops": "object oriented programming",
    "node": "node js",
    "js": "javascript",
    "ts": "typescript",
    "html": "html",
    "css": "cascading style sheets",
    "nosql": "nosql",
    "frontend": "front end",
    "front end": "front end",
    "backend": "back end",
    "back end": "back end",
    "fullstack": "full stack",
    "full stack": "full stack",
    "jquery": "jquery",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "azure": "microsoft azure",
    "gpu": "graphics processing unit",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "mvn": "maven",
    "crm": "customer relationship management",
    "erp": "enterprise resource planning",
    "soa": "service oriented architecture",
    "soap": "soap",
    "rest": "rest",
    "http": "http",
    "git": "git",
    "ide": "integrated development environment",
    "sdk": "software development kit",
    "jdk": "java development kit",
    "jvm": "java virtual machine",
    "id": "identifier",
    "devops": "devops",
    "vm": "virtual machine",
    "uat": "user acceptance testing",
    "qa": "quality assurance",
    "sre": "site reliability engineering",
    "dba": "database administrator",
    "bi": "business intelligence",
    "olap": "online analytical processing",
    "oltp": "online transaction processing",
    "seo": "search engine optimization",
    "uiux": "user interface and user experience",
    "pdf": "pdf",
    "docx": "microsoft word document",
    "keras": "keras",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "sklearn": "scikit learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "spark": "apache spark",
    "hadoop": "apache hadoop",
    "hive": "apache hive",
    "kafka": "apache kafka",
    "flink": "apache flink",
    "airflow": "apache airflow",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "terraform": "terraform",
    "ansible": "ansible",
    "jenkins": "jenkins",
    "graphql": "graphql",
    "redis": "redis",
    "mongodb": "mongodb",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "oracle": "oracle database",
    "snowflake": "snowflake",
    "databricks": "databricks",
    "tableau": "tableau",
    "powerbi": "microsoft power bi",
    "power bi": "microsoft power bi",
    "excel": "microsoft excel",
    "spreadsheet": "spreadsheet",
}

# Expand the common tech abbreviation tokens that appear as standalone tokens
_STANDALONE = re.compile(r"(?<!\w)([a-z]+|\d+)(?!\w)")


def normalize_text(text: str | None, lowercase: bool = True, expand_abbrev: bool = True) -> str:
    """Normalize text for matching:
    lowercase -> NFKC unicode -> [abbreviation expansion] -> punctuation to space -> collapse.
    `expand_abbrev=False` keeps abbreviations verbatim (used for containment/fuzzy so that
    e.g. "rest api development" does NOT become a superset of "application development").
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if lowercase:
        text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # abbreviation expansion on whole phrase (multi-word) then single tokens
    if expand_abbrev:
        text = " {t} ".format(t=text)
        for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
            if abbr in text:
                text = text.replace(f" {abbr} ", f" {full} ")
        text = text.strip()
    return text


def tokens(text: str, expand_abbrev: bool = False) -> set[str]:
    return set(normalize_text(text, expand_abbrev=expand_abbrev).split())


def token_jaccard(a: str, b: str, expand_abbrev: bool = False) -> float:
    ta, tb = tokens(a, expand_abbrev), tokens(b, expand_abbrev)
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def token_containment(sub: str, full: str, expand_abbrev: bool = False) -> float:
    """Fraction of `sub` tokens contained in `full` tokens."""
    s = tokens(sub, expand_abbrev)
    f = tokens(full, expand_abbrev)
    if not s:
        return 0.0
    return len(s & f) / len(s)


def naive_normalize_amap(text: str) -> str:
    """Lightweight norm for exact-hash duplicate detection (no abbrev expansion)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()