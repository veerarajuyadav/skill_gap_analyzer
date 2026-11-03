"""Map all candidate job titles (EXAI role digests + Job Skill Set titles)
to canonical occupations (ESCO / O*NET). Produces:
  - data/processed/occupation_mapping.parquet
  - reports/occupation_mapping.csv

Method column records how each title was mapped:
  alias_exact / token_containment / fuzzy / embedding / embedding_low / curated
Titles that stay unmapped are recorded with mapping_confidence=0 and a
`matching_method` of 'unmapped' (they are excluded from target-role pooling but
kept in the output for auditability).
"""
from __future__ import annotations

import sys

import pandas as pd

from src.paths import PROCESSED_DIR, REPORTS_DIR
from src.data.load_datasets import load_hf_source
from src.ontology.occupations import build_occupation_lexicon, OccupationMapper

# ---------------------------------------------------------------------------
# Curated title overrides for modern/derived tech roles not (yet) covered by
# ESCO/O*NET labels. method='curated', derived by expert review from O*NET Job
# Titles / ESCO alt labels and common industry usage. Fully auditable.
# ---------------------------------------------------------------------------
CURATED_TITLE_MAP: dict[str, str] = {
    # raw_title_lower  -> canonical ESCO occupation id ("ESCO-<uuid>") or ONET SOC ("ONET-<soc>")
    "ml engineer": "ESCO-2b1d2d27-9bd1-42e8-a5c2-06f055a66512",   # placeholder; filled at runtime
}

# Known-good id shortcuts resolved below where deterministic rules apply.


def _resolve_curated(mapper: OccupationMapper, raw: str) -> str | None:
    """Return occ_id for a curated title. Resolves by matching the curated
    canonical *name* against the lexicon at runtime (so ids stay correct)."""
    key = raw.strip().lower()
    name = _CURATED_BY_NAME.get(key)
    if name is None:
        return None
    occ = mapper.lexicon[mapper.lexicon["name"].str.lower().eq(name.lower())]
    if len(occ):
        return occ.iloc[0]["occ_id"]
    return None


_CURATED_BY_NAME: dict[str, str] = {
    "ml engineer": "artificial intelligence engineer",
    "ai engineer": "artificial intelligence engineer",
    "ai ml engineer": "artificial intelligence engineer",
    "machine learning scientist": "artificial intelligence engineer",
    "frontend developer": "web developer",
    "front end developer": "web developer",
    "front-end developer": "web developer",
    "full stack developer": "software developer",
    "full-stack developer": "software developer",
    "fullstack developer": "software developer",
    "software development engineer": "software developer",
    "sre": "ICT reliability engineer",
    "site reliability engineer": "ICT reliability engineer",
    "devops engineer": "ICT reliability engineer",
    "mobile developer": "mobile application developer",
    "ios developer": "mobile application developer",
    "android developer": "mobile application developer",
    "data engineer": "data engineer",
    "data architect": "data architect",
    "cloud engineer": "cloud engineer",
    "cloud architect": "ICT solution architect",
    "platform engineer": "software developer",
    "systems engineer": "ICT system engineer",
    "security engineer": "ICT security engineer",
    "penetration tester": "ICT security specialist",
    "cybersecurity analyst": "ICT security specialist",
    "backend developer": "software developer",
    "back end developer": "software developer",
    "back-end developer": "software developer",
    "software development engineer in test": "software tester",
}


def build_occupation_mapping(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "occupation_mapping.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    mapper = OccupationMapper(use_embedding=True)
    titles: set[str] = set()
    jss = load_hf_source("job_skill_set")
    titles |= set(jss["job_title"].dropna().astype(str).str.strip())
    exai = load_hf_source("exai_job_postings")
    titles |= set(exai["Role"].dropna().astype(str).str.strip())

    rows = []
    for t in sorted(titles):
        r = mapper.match(t)
        if r is None:
            curated = _resolve_curated(mapper, t)
            if curated:
                rec = mapper.id2rec[curated]
                rows.append({
                    "raw_title": t,
                    "normalized_title": rec["name"],
                    "esco_occupation_id": curated if curated.startswith("ESCO") else "",
                    "onet_occupation_id": rec["onet_soc"],
                    "occupation_family": rec["family"],
                    "mapping_confidence": 1.0,
                    "matching_method": "curated",
                })
            else:
                rows.append({
                    "raw_title": t, "normalized_title": "", "esco_occupation_id": "",
                    "onet_occupation_id": "", "occupation_family": "",
                    "mapping_confidence": 0.0, "matching_method": "unmapped",
                })
        else:
            rows.append({
                "raw_title": t,
                "normalized_title": r["canonical_name"],
                "esco_occupation_id": r["esco_occupation_id"],
                "onet_occupation_id": r["onet_occupation_id"],
                "occupation_family": r["occupation_family"],
                "mapping_confidence": r["confidence"],
                "matching_method": r["method"],
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    df.to_csv(REPORTS_DIR / "occupation_mapping.csv", index=False)
    return df


if __name__ == "__main__":
    df = build_occupation_mapping(force=True)
    print("titles mapped:", len(df))
    print(df["matching_method"].value_counts().to_dict())
    it_mask = df["occupation_family"] == "Computer and Mathematical Occupations"
    print("ONET-IT titled mapped:", int(it_mask.sum()))
    print("unmapped:", df[df["matching_method"] == "unmapped"]["raw_title"].tolist()[:60])