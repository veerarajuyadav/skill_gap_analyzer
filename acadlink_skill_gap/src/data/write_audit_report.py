"""Render the EDA audit into human-readable markdown (UTF-8 to disk)."""
from __future__ import annotations

import json
from pathlib import Path

from src.paths import REPORTS_DIR


def _fmt_counts(counts) -> str:
    if counts is None:
        return "n/a"
    if not isinstance(counts, dict):
        return str(counts)
    items = sorted(counts.items(), key=lambda x: -x[1])[:15]
    return ", ".join(f"{k}={v}" for k, v in items)


def render() -> str:
    audit = json.loads((REPORTS_DIR / "eda_audit.json").read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# AcadLink Skill Gap — Data Audit & EDA Report")
    lines.append("")
    lines.append("This report summarises an exploratory audit of all five data sources "
                 "(three Hugging Face datasets + two ontologies). It is the precursor to "
                 "dataset construction; no model has been trained yet.")
    lines.append("")

    sources = audit["sources"]
    for name, a in sources.items():
        lines.append(f"## Source: {name}")
        lines.append("")
        lines.append(f"- **Records:** {a.get('rows')}")
        lines.append(f"- **Columns:** {', '.join(a.get('columns', []))}")
        lines.append("")
        nu = a.get("nulls", {})
        if nu:
            lines.append("### Null / dtype / cardinality")
            lines.append("")
            lines.append("| column | dtype | nulls | null% | nunique |")
            lines.append("|---|---|---|---|---|")
            for c, info in nu.items():
                lines.append(f"| {c} | {info.get('dtype')} | {info.get('nulls')} | "
                             f"{info.get('null_pct')} | {info.get('nunique')} |")
            lines.append("")
        for key in ["label_counts", "split_counts", "category_counts", "role_counts", "id_duplicates"]:
            if key in a and a[key] is not None:
                lines.append(f"- **{key}:** {_fmt_counts(a[key])}")
                lines.append("")
        for key in ["resume_len", "jd_len", "feature_len", "skill_set_len"]:
            if a.get(key):
                v = a[key]
                lines.append(f"- **{key} (chars):** mean={v.get('mean_chars')} median={v.get('median')} "
                             f"min={v.get('min')} max={v.get('max')} n={v.get('n')}")
        if a.get("duplicates"):
            lines.append(f"- **Duplicates:** {a['duplicates']}")
        if a.get("synthetic_flag"):
            lines.append(f"- **Synthetic flag:** {a['synthetic_flag']}")
        if a.get("language"):
            lines.append(f"- **Language probe:** {a['language']}")
        lines.append("")

    ont = audit.get("ontology", {})
    lines.append("## Ontology — O*NET 30.3")
    for key, info in ont.get("onet", {}).items():
        lines.append(f"- **{key}:** rows={info.get('rows')}, cols={list(info.get('cols', []))} | "
                     f"unique occupations={info.get('unique_occupations')} unique elements={info.get('unique_element_names')}")
    lines.append("")
    lines.append("## Ontology — ESCO v1.2.1")
    for key, info in ont.get("esco", {}).items():
        lines.append(f"- **{key}:** rows={info.get('rows')}, cols={list(info.get('cols', []))} | "
                     f"{ {k: v for k, v in info.items() if k not in ('cols', 'rows', 'nulls')} }")
    lines.append("")

    out = "\n".join(lines)
    (REPORTS_DIR / "eda_audit.md").write_text(out, encoding="utf-8")
    return out


if __name__ == "__main__":
    render()
    print("wrote", REPORTS_DIR / "eda_audit.md")
