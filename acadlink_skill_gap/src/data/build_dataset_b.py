"""Dataset B — Requirement corpus (JD -> required / preferred skills).

Layers built here (see reports/dataset_construction_proposal.md §5, Layer 2):

 1. occupation_skill_requirements.parquet — knowledge base per occupation:
      * ESCO  v1.2.1  occupation-skill relations (essential / optional)
      * O*NET 30.3    knowledge / essential skills / abilities importance
                       (IM scale 1-5, every occupation) and software-skill
                       Hot-Technology / In-Demand flags
 2. jd_skill_mentions.parquet          — deterministic mention evidence:
      * SkillSpanFinder over  1167 job_skill_set JDs + 376 EXAI role digests
      * explicit job-skill-set lists (weak positives, canonicalized)
      * required/preferred derived from keyword context around each span
 3. dataset_B.parquet                  — final Dataset B per (jd_entity, skill):
        union of mention evidence + occupation-level requirements, with
        requirement = required-if-any, importance = max, confidence = max.
        Every row carries label_sources for auditability.

Deterministic: no untracked randomness; thresholds/costs documented in
reports/skill_importance_methodology.md.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd

from src.paths import PROCESSED_DIR, REPORTS_DIR, ESCO_DIR, ONET_DIR
from src.preprocessing.skill_spans import SkillSpanFinder
from src.ontology.occupations import build_occupation_lexicon
from src.ontology.skills import SkillMatcher

JSS_KEY = "job_skill_set"
EXAI_KEY = "exai_job_postings"

# --- requirement context keywords (raw, case-insensitive) -------------------
REQUIRED_RX = re.compile(
    r"\b(must\s?have|requirement|required|require|proficien\w+\s+in|"
    r"expertise\s+in|skilled\s+in|well[- ]versed|deep\s+understanding\s+of|"
    r"hands[- ]on\s+with|experience(d)?\s+with|experience\s+of|"
    r"need(s)?\s+to\s+know|should\s+have|expected\s+to\s+have|respos?ible\s+for)\b",
    re.IGNORECASE)
PREFERRED_RX = re.compile(
    r"\b(nice\s?to\s?have|prefer\w*|a\s+plus|plus\b|bonus|desir\w+|"
    r"good\s+to\s?have|beneficial|asset\b|advantage|familiarity\s+with|"
    r"basic\s+knowledge\s+of|working\s+knowledge\s+of)\b",
    re.IGNORECASE)
CONTEXT_RADIUS = 150

# --- importance policy (documented in skill_importance_methodology.md) -------
IMP = {
    "esco_essential":     (1.0, True,  "required"),
    "esco_optional":      (0.6, False, "preferred"),
    "onet_im_high":       (0.9, True,  "required"),   # O*NET IM >= 4.0
    "onet_im_medium":     (0.6, False, "preferred"),  # O*NET IM 3.0-3.9
    "onet_hot_tech":      (0.9, True,  "required"),
    "onet_software":      (0.5, False, "preferred"),
    "jss_skill_list":     (0.7, True,  "required"),
    "jd_mention_required":(0.8, True,  "required"),
    "jd_mention_preferred":(0.5, False, "preferred"),
}
ONET_IM_REQUIRED = 4.0

_JOIN_CACHE: dict[str, pd.DataFrame] = {}


def _get(name: str, path: Path) -> pd.DataFrame:
    if name not in _JOIN_CACHE:
        _JOIN_CACHE[name] = pd.read_parquet(path)
    return _JOIN_CACHE[name]


# ---------------------------------------------------------------------------
# 1. Occupation-level requirements (ESCO + O*NET)
# ---------------------------------------------------------------------------
def build_occupation_requirements(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "occupation_skill_requirements.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    recs: list[dict] = []

    # --- ESCO occupation_skill_relations -----------------------------------
    osr = pd.read_csv(ESCO_DIR / "occupationSkillRelations_en.csv", dtype=str)
    osr["occ_id"] = "ESCO-" + osr["occupationUri"].str.extract(r"/([^/]+)$", expand=False)
    osr["skill_id"] = osr["skillUri"].str.extract(r"/([^/]+)$", expand=False)
    for r in osr.itertuples(index=False):
        essential = r.relationType == "essential"
        key = "esco_essential" if essential else "esco_optional"
        imp_val, _, req = IMP[key]
        recs.append({
            "occupation_id": r.occ_id,
            "occupation_name": r.occupationLabel,
            "source_system": "ESCO",
            "requirement": req,
            "importance": imp_val,
            "label_source": key,
            "confidence": 0.95,
            "relation_type": r.relationType,
            "skill_id": r.skill_id,
        })

    # --- O*NET importance files ---------------------------------------------
    # skill_id is prefixed to match canonical_skills (ONET-KNOW-/ONET-ESS-/ONET-AB-)
    onet_prefix = {"knowledge": "ONET-KNOW-", "essential_skills": "ONET-ESS-",
                   "abilities": "ONET-AB-"}
    for fname in ("knowledge", "essential_skills", "abilities"):
        d = pd.read_csv(ONET_DIR / f"{fname}.csv", dtype=str)
        im = d[d["Scale ID"] == "IM"].copy()
        im["Data Value"] = pd.to_numeric(im["Data Value"], errors="coerce")
        im = im.dropna(subset=["Data Value"])
        for _, r in im.iterrows():
            v = float(r["Data Value"])
            if v >= ONET_IM_REQUIRED:
                key, imp_val, req = "onet_im_high", IMP["onet_im_high"][0], IMP["onet_im_high"][2]
            else:
                key, imp_val, req = "onet_im_medium", IMP["onet_im_medium"][0], IMP["onet_im_medium"][2]
            recs.append({
                "occupation_id": "ONET-" + r["O*NET-SOC Code"],
                "occupation_name": r["Title"],
                "source_system": "ONET",
                "requirement": req,
                "importance": round(imp_val, 3),
                "label_source": key,
                "confidence": 0.9,
                "relation_type": f"importance={v:.1f}",
                "skill_id": onet_prefix[fname] + r["Element ID"],
            })

    # --- O*NET software / technology skills ---------------------------------
    sw = pd.read_csv(ONET_DIR / "software_skills.csv", dtype=str)
    for _, r in sw.drop_duplicates(["O*NET-SOC Code", "Element ID"]).iterrows():
        hot = str(r.get("Hot Technology", "")).strip().lower() in ("yes", "y")
        imp_val, req = (IMP["onet_hot_tech"][0], IMP["onet_hot_tech"][2]) if hot else (
            IMP["onet_software"][0], IMP["onet_software"][2])
        recs.append({
            "occupation_id": "ONET-" + r["O*NET-SOC Code"],
            "occupation_name": r["Title"],
            "source_system": "ONET",
            "requirement": req,
            "importance": imp_val,
            "label_source": "onet_hot_tech" if hot else "onet_software",
            "confidence": 0.85,
            "relation_type": "hot_technology" if hot else "software",
            "skill_id": "ONET-SOFT-" + r["Element ID"],
        })

    df = pd.DataFrame(recs)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. JD mention evidence
# ---------------------------------------------------------------------------
def _classify(text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_RADIUS)
    hi = min(len(text), end + CONTEXT_RADIUS)
    ctx = text[lo:hi]
    preferred = PREFERRED_RX.search(ctx)
    required = REQUIRED_RX.search(ctx)
    if preferred:
        return "preferred"
    if required:
        return "required"
    return "required"  # skills stated in a JD are expectations by default


def _sanitize(src: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", src).strip("-")[:80]


def build_jd_mentions(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "jd_skill_mentions.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    finder = SkillSpanFinder()
    matcher = SkillMatcher(use_embedding=False)
    mapping = _get("mapping", PROCESSED_DIR / "occupation_mapping.parquet")
    from src.data.load_datasets import load_hf_source
    jss = load_hf_source(JSS_KEY)
    exai = load_hf_source(EXAI_KEY)
    lex = build_occupation_lexicon()

    rows: list[dict] = []

    def occ_of(title: str) -> dict | None:
        r = mapping[mapping["raw_title"].eq(title)]
        if len(r):
            r = r.iloc[0]
            occ_id = r["esco_occupation_id"] or r["onet_occupation_id"]
            if occ_id:
                name = ""
                fm = r["occupation_family"]
                if occ_id in lex["occ_id"].values:
                    name = lex[lex["occ_id"].eq(occ_id)].iloc[0]["name"]
                return {"occupation_id": occ_id, "occupation_name": name,
                        "occupation_family": fm}
        return None

    # job_skill_set JDs
    for r in jss.itertuples(index=False):
        title = str(r.job_title).strip()
        desc = "" if pd.isna(r.job_description) else str(r.job_description)
        o = occ_of(title)
        for m in finder.find(desc):
            req = _classify(desc, m["span_start"], m["span_end"])
            rows.append({
                "jd_id": f"JSS-{r.job_id}",
                "jd_entity": title,
                "jd_text": desc,
                "occupation_id": o["occupation_id"] if o else "",
                "occupation_name": o["occupation_name"] if o else "",
                "occupation_family": o["occupation_family"] if o else "",
                "skill_id": m["skill_id"],
                "skill_name": m["label"],
                "skill_type": "",
                "evidence_span": f"{m['span_start']}:{m['span_end']}",
                "mention_text": m["span_text"],
                "requirement": req,
                "importance": IMP["jd_mention_required" if req == "required"
                                  else "jd_mention_preferred"][0],
                "label_source": "jd_mention",
                "confidence": m["confidence"],
            })
        # explicit skill list -> canonicalized weak positives
        if not pd.isna(r.job_skill_set):
            try:
                items = ast.literal_eval(r.job_skill_set) or []
            except Exception:
                items = []
            for item in items:
                if not isinstance(item, str) or not item.strip():
                    continue
                m_ = matcher.match(item.strip())
                if m_ is None:
                    continue
                rows.append({
                    "jd_id": f"JSS-{r.job_id}",
                    "jd_entity": title,
                    "jd_text": desc,
                    "occupation_id": o["occupation_id"] if o else "",
                    "occupation_name": o["occupation_name"] if o else "",
                    "occupation_family": o["occupation_family"] if o else "",
                    "skill_id": m_["skill_id"],
                    "skill_name": m_["canonical_name"],
                    "skill_type": m_["skill_type"],
                    "evidence_span": "",
                    "mention_text": "",
                    "requirement": IMP["jss_skill_list"][2],
                    "importance": IMP["jss_skill_list"][0],
                    "label_source": "jss_skill_list",
                    "confidence": IMP["jss_skill_list"][0] * m_["confidence"],
                })

    # EXAI role digests (one jd entity per Role; Features are templated per role)
    for role in exai["Role"].drop_duplicates():
        feat = exai[exai["Role"].eq(role)]["Features"].iloc[0]
        feat = "" if pd.isna(feat) else str(feat)
        o = occ_of(str(role).strip())
        jd_id = "EXAI-" + _sanitize(str(role))
        for m in finder.find(feat):
            req = _classify(feat, m["span_start"], m["span_end"])
            rows.append({
                "jd_id": jd_id,
                "jd_entity": str(role).strip(),
                "jd_text": feat,
                "occupation_id": o["occupation_id"] if o else "",
                "occupation_name": o["occupation_name"] if o else "",
                "occupation_family": o["occupation_family"] if o else "",
                "skill_id": m["skill_id"],
                "skill_name": m["label"],
                "skill_type": "",
                "evidence_span": f"{m['span_start']}:{m['span_end']}",
                "mention_text": m["span_text"],
                "requirement": req,
                "importance": IMP["jd_mention_required" if req == "required"
                                  else "jd_mention_preferred"][0],
                "label_source": "exai_digest",
                "confidence": m["confidence"],
            })

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Final Dataset B
# ---------------------------------------------------------------------------
def build_dataset_b(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "dataset_B.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    mentions = build_jd_mentions(force=force)
    occ_reqs = build_occupation_requirements(force=force)

    groups = mentions.groupby(["jd_id", "jd_entity", "occupation_id",
                               "occupation_name", "occupation_family", "skill_id"],
                              dropna=False)
    rows = []
    for (jd_id, jd_entity, occ_id, occ_name, occ_fam, skill_id), g in groups:
        reqs = g["requirement"].tolist()
        imps = g["importance"].tolist()
        confs = g["confidence"].tolist()
        sources = g["label_source"].tolist()
        spans = [(s, t) for s, t in zip(g["evidence_span"], g["mention_text"]) if s]
        rows.append({
            "jd_id": jd_id,
            "jd_entity": jd_entity,
            "occupation_id": occ_id,
            "occupation_name": occ_name,
            "occupation_family": occ_fam,
            "skill_id": skill_id,
            "skill_name": g["skill_name"].iloc[0],
            "requirement": "required" if "required" in reqs else "preferred",
            "importance": round(max(imps), 3),
            "label_sources": "|".join(sorted(set(sources))),
            "confidence": round(max(confs), 3),
            "evidence_spans": ";".join(f"{s}:{t}" for s, t in spans),
            "n_source_types": len(set(sources)),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_summary(b_df: pd.DataFrame, mention_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Dataset B — JD → Required Skills (summary)")
    lines.append("")
    lines.append(f"- **JD entities covered:** {b_df['jd_id'].nunique()}")
    lines.append(f"- **Rows (jd_entity × skill):** {len(b_df)}")
    lines.append(f"- **Unique canonical skills:** {b_df['skill_id'].nunique()}")
    lines.append(f"- **Rows with mapped occupation:** {b_df['occupation_id'].astype(str).str.startswith(('ESCO','ONET')).sum()}")
    lines.append("")
    lines.append("## Requirement mix")
    lines.append(b_df["requirement"].value_counts().to_string())
    lines.append("")
    lines.append("## Label sources")
    lines.append(b_df["label_sources"].str.split("|").explode().value_counts().to_string())
    lines.append("")
    lines.append("## Mentio­n-level recall (jd_skill_mentions)")
    lines.append(f"- mentions rows: {len(mention_df)}")
    if len(mention_df):
        lines.append(mention_df["label_source"].value_counts().to_string())
    lines.append("")
    lines.append(f"- JSs entities: {mention_df['jd_id'].nunique()}")
    lines.append(f"- average mentions per jd: {len(mention_df)/max(1, mention_df['jd_id'].nunique()):.1f}")
    lines.append("")
    lines.append("## Coverage caveats")
    lines.append("- jd_mention / exai_digest rows rely on deterministic contiguous-token")
    lines.append("  matches only (high precision, low recall).")
    lines.append("- Occupation-anchored rows come from ESCO essential/optional and")
    lines.append("  O*NET importance; unmapped titles carry empty occupation_id (kept")
    lines.append("  for auditability, excluded from target-role pooling).")
    (REPORTS_DIR / "dataset_B_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORTS_DIR / "dataset_B_summary.md")


if __name__ == "__main__":
    force = "--force" in sys.argv
    ments = build_jd_mentions(force=force)
    print("jd_skill_mentions:", ments.shape)
    occdf = build_occupation_requirements(force=force)
    print("occupation_skill_requirements:", occdf.shape,
          occdf["source_system"].value_counts().to_dict())
    b = build_dataset_b(force=force)
    print("dataset_B:", b.shape)
    print(b["requirement"].value_counts().to_dict())
    write_summary(b, ments)