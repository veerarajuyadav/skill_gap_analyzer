"""Exploratory data analysis and data audit across all five data sources.

Produces:
- ``reports/eda_audit.json``  (machine-readable audit)
- ``reports/eda_audit.md``    (human-readable summary)

This script ADAPTS to whatever columns exist in each source; it never assumes
a uniform schema. All text length / null / duplicate stats are computed from
the actual loaded frames.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data.load_datasets import (
    load_hf_source,
    load_onenet_file,
    load_esco_file,
    ONET_FILES,
    ESCO_FILES,
)
from src.paths import REPORTS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _text_len_stats(s: pd.Series) -> dict[str, Any]:
    s = s.dropna().astype(str)
    if s.empty:
        return {"n": 0, "mean_chars": 0, "median": 0, "min": 0, "max": 0}
    lens = s.str.len()
    return {
        "n": int(len(lens)),
        "mean_chars": round(float(lens.mean()), 1),
        "median": int(lens.median()),
        "min": int(lens.min()),
        "max": int(lens.max()),
    }


def _null_report(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for c in df.columns:
        out[c] = {
            "nulls": int(df[c].isna().sum()),
            "null_pct": round(float(df[c].isna().mean()) * 100, 2),
            "dtype": str(df[c].dtype),
            "nunique": int(df[c].nunique(dropna=True)),
        }
    return out


def _dup_report(df: pd.DataFrame, subset) -> dict[str, Any]:
    n_full = int(df.duplicated().sum())
    n_sub = int(df.duplicated(subset=subset).sum()) if subset else 0
    return {"exact_foreground_duplicates": n_full, "subset_duplicates": n_sub, "subset_by": subset}


# ---------------------------------------------------------------------------
# per-source audits
# ---------------------------------------------------------------------------
def audit_resume_jd_fit() -> dict[str, Any]:
    df = load_hf_source("resume_jd_fit")
    a = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": _null_report(df),
        "label_counts": df["label"].value_counts().to_dict() if "label" in df else {},
        "split_counts": df.get("split", pd.Series(dtype=object)).value_counts().to_dict(),
        "resume_len": _text_len_stats(df["resume_text"]),
        "jd_len": _text_len_stats(df["job_description_text"]),
        "duplicates": _dup_report(df, ["resume_text"]),
        "language": _language_probe(df[["resume_text", "job_description_text"]]),
    }
    return a


def audit_exai_resumes() -> dict[str, Any]:
    df = load_hf_source("exai_resumes")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": _null_report(df),
        "category_counts": df.get("Category", pd.Series(dtype=object)).value_counts().to_dict(),
        "id_duplicates": int(df["ID"].duplicated().sum()) if "ID" in df else None,
        "feature_len": _text_len_stats(df["Feature"]),
        "duplicates": _dup_report(df, None),
        "synthetic_flag": "high (placeholders 'company name city state', lowercased text)",
        "language": _language_probe(df[["Feature"]]),
    }


def audit_exai_job_postings() -> dict[str, Any]:
    df = load_hf_source("exai_job_postings")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": _null_report(df),
        "role_counts": df["Role"].value_counts().to_dict() if "Role" in df else {},
        "role_nunique": int(df["Role"].nunique()) if "Role" in df else None,
        "duplicates": _dup_report(df, ["Role"]),
        "feature_len": _text_len_stats(df["Features"]),
        "synthetic_flag": "high (aggregated/generated role-profile strings)",
    }


def audit_job_skill_set() -> dict[str, Any]:
    df = load_hf_source("job_skill_set")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": _null_report(df),
        "category_counts": df.get("category", pd.Series(dtype=object)).value_counts().to_dict(),
        "job_title_nunique": int(df.get("job_title", pd.Series(dtype=object)).nunique()),
        "jd_len": _text_len_stats(df["job_description"]) if "job_description" in df else None,
        "skill_set_len": _text_len_stats(df["job_skill_set"]) if "job_skill_set" in df else None,
        "duplicates": _dup_report(df, None),
        "language": _language_probe(df[["job_description", "job_title"]]),
    }


def audit_ontology() -> dict[str, Any]:
    out = {}

    # O*NET
    onet_report = {}
    onet = load_onenet_file("occupation_data")
    onet_report["occupation_data"] = {
        "rows": int(len(onet)),
        "cols": list(onet.columns),
        "n_soc_codes": int(onet["O*NET-SOC Code"].nunique()) if "O*NET-SOC Code" in onet else None,
        "nulls": _null_report(onet),
    }
    for key, f in [
        ("software_skills", "software_skills"),
        ("knowledge", "knowledge"),
        ("essential_skills", "essential_skills"),
        ("abilities", "abilities"),
    ]:
        d = load_onenet_file(key)
        uniq_elem = int(d["Element Name"].nunique()) if "Element Name" in d else None
        uniq_occ = int(d["O*NET-SOC Code"].nunique()) if "O*NET-SOC Code" in d else None
        onet_report[key] = {
            "rows": int(len(d)),
            "cols": list(d.columns),
            "unique_element_names": uniq_elem,
            "unique_occupations": uniq_occ,
            "nulls": _null_report(d),
        }
    jt = load_onenet_file("job_titles")
    onet_report["job_titles"] = {
        "rows": int(len(jt)),
        "cols": list(jt.columns),
        "unique_titles": int(jt["Job Title"].nunique()) if "Job Title" in jt else None,
        "unique_occupations": int(jt["O*NET-SOC Code"].nunique()) if "O*NET-SOC Code" in jt else None,
        "nulls": _null_report(jt),
    }
    out["onet"] = onet_report

    # ESCO
    esco_report = {}
    sk = load_esco_file("skills")
    esco_report["skills"] = {
        "rows": int(len(sk)),
        "cols": list(sk.columns),
        "unique_skills": int(sk["conceptUri"].nunique()) if "conceptUri" in sk else int(len(sk)),
        "skillType": sk["skillType"].value_counts().to_dict() if "skillType" in sk else {},
        "status": sk["status"].value_counts().to_dict() if "status" in sk else {},
        "nulls": _null_report(sk),
    }
    oc = load_esco_file("occupations")
    esco_report["occupations"] = {
        "rows": int(len(oc)),
        "cols": list(oc.columns),
        "unique_occupations": int(oc["conceptUri"].nunique()) if "conceptUri" in oc else int(len(oc)),
        "status": oc["status"].value_counts().to_dict() if "status" in oc else {},
        "nulls": _null_report(oc),
    }
    rel = load_esco_file("occupation_skill_relations")
    esco_report["occupation_skill_relations"] = {
        "rows": int(len(rel)),
        "cols": list(rel.columns),
        "relationType": rel["relationType"].value_counts().to_dict() if "relationType" in rel else {},
        "skillType": rel["skillType"].value_counts().to_dict() if "skillType" in rel else {},
        "nulls": _null_report(rel),
    }
    hi = load_esco_file("skills_hierarchy")
    esco_report["skills_hierarchy"] = {
        "rows": int(len(hi)),
        "cols": list(hi.columns),
        "unique_l0": int(hi.get("Level 0 preferred term", pd.Series(dtype=object)).nunique()),
        "nulls": _null_report(hi),
    }
    out["esco"] = esco_report
    return out


def _language_probe(frames) -> dict[str, Any]:
    """Crude language probe on a sample of text columns."""
    import re

    try:
        from langdetect import detect
        HAVE = True
    except Exception:
        HAVE = False

    rows = []
    for col in frames.columns:
        s = frames[col].dropna().astype(str).head(200)
        for t in s:
            if len(t) > 40:
                rows.append(t)
                if len(rows) >= 100:
                    break
        if len(rows) >= 100:
            break
    if not rows or not HAVE:
        return {"note": "langdetect unavailable or sample empty", "sampled": len(rows), "tool": "langdetect" if HAVE else "n/a"}
    counts: dict[str, int] = {}
    for t in rows:
        try:
            l = detect(t)
            counts[l] = counts.get(l, 0) + 1
        except Exception:
            pass
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]
    return {"top_languages": top, "sampled": len(rows), "tool": "langdetect"}


# ---------------------------------------------------------------------------
# master audit
# ---------------------------------------------------------------------------
def run_full_audit(verbose: bool = True) -> dict[str, Any]:
    audit: dict[str, Any] = {"sources": {}}
    audit["sources"]["resume_jd_fit"] = audit_resume_jd_fit()
    audit["sources"]["exai_resumes"] = audit_exai_resumes()
    audit["sources"]["exai_job_postings"] = audit_exai_job_postings()
    audit["sources"]["job_skill_set"] = audit_job_skill_set()
    audit["ontology"] = audit_ontology()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "eda_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    if verbose:
        print(json.dumps(audit, indent=2, default=str)[:8000])
    return audit


if __name__ == "__main__":
    run_full_audit()
