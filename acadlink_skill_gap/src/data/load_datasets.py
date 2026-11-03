"""Load, stage, and cache all data sources for the AcadLink Skill Gap project.

Each source is cached to a parquet file under ``data/interim`` so downstream
stages do not need to re-download large remote data.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.paths import INTERIM_DIR, ONET_DIR, ESCO_DIR

# ---------------------------------------------------------------------------
# Source catalogue
# ---------------------------------------------------------------------------
SOURCE_MANIFEST = {
    "resume_jd_fit": {
        "hf_id": "cnamuangtoun/resume-job-description-fit",
        "description": "Resume-Job Description Fit classification",
        "parquet": INTERIM_DIR / "resume_jd_fit.parquet",
        "hf_splits": ["train", "test"],
        "final_cols": ["resume_text", "job_description_text", "label", "split"],
    },
    "exai_resumes": {
        "hf_id": "mithinsagar/exai-resumeintel-data",
        "hf_config": "resumes",
        "description": "EXAI ResumeIntel resume data",
        "parquet": INTERIM_DIR / "exai_resumes.parquet",
    },
    "exai_job_postings": {
        "hf_id": "mithinsagar/exai-resumeintel-data",
        "hf_config": "job_postings",
        "description": "EXAI ResumeIntel aggregated job-posting/role data (synthetic)",
        "parquet": INTERIM_DIR / "exai_job_postings.parquet",
    },
    "job_skill_set": {
        "hf_id": "batuhanmtl/job-skill-set",
        "description": "Job Skill Set (job title, description, skill set)",
        "parquet": INTERIM_DIR / "job_skill_set.parquet",
    },
}

ONET_FILES = {
    "occupation_data": ONET_DIR / "occupation_data.csv",
    "job_titles": ONET_DIR / "job_titles.csv",
    "software_skills": ONET_DIR / "software_skills.csv",
    "knowledge": ONET_DIR / "knowledge.csv",
    "essential_skills": ONET_DIR / "essential_skills.csv",
    "abilities": ONET_DIR / "abilities.csv",
}

ESCO_FILES = {
    "occupations": ESCO_DIR / "occupations_en.csv",
    "skills": ESCO_DIR / "skills_en.csv",
    "occupation_skill_relations": ESCO_DIR / "occupationSkillRelations_en.csv",
    "skills_hierarchy": ESCO_DIR / "skillsHierarchy_en.csv",
    "broad_relations": ESCO_DIR / "broaderRelationsSkillPillar_en.csv",
}


def _ensure_interim() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def _to_parquet(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _load_hf_hub(src_cfg: dict) -> pd.DataFrame:
    from datasets import load_dataset

    kwargs = {}
    if src_cfg.get("hf_config"):
        kwargs["name"] = src_cfg["hf_config"]

    splits = src_cfg.get("hf_splits")
    if splits:
        frames = []
        for split in splits:
            ds = load_dataset(src_cfg["hf_id"], split=split, **kwargs)
            f = ds.to_pandas()
            f["split"] = split
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        cols = src_cfg.get("final_cols")
        if cols:
            df = df[cols]
        return df

    ds = load_dataset(src_cfg["hf_id"], split="train", **kwargs)
    return ds.to_pandas()


def load_hf_source(key: str, force: bool = False) -> pd.DataFrame:
    """Load an HF-backed source, using the parquet cache when present."""
    if key not in SOURCE_MANIFEST:
        raise KeyError(f"Unknown source key: {key}")
    _ensure_interim()
    cfg = SOURCE_MANIFEST[key]
    cache = cfg["parquet"]
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    df = _load_hf_hub(cfg)
    return _to_parquet(df, cache)


def load_onenet_file(key: str) -> pd.DataFrame:
    """Load an O*NET tabular file (comma-delimited)."""
    if key not in ONET_FILES:
        raise KeyError(f"Unknown O*NET key: {key}")
    path = ONET_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"O*NET file missing: {path}")
    return pd.read_csv(path, dtype=str)


def load_esco_file(key: str) -> pd.DataFrame:
    """Load an ESCO CSV file (comma-delimited)."""
    if key not in ESCO_FILES:
        raise KeyError(f"Unknown ESCO key: {key}")
    path = ESCO_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"ESCO file missing: {path}")
    return pd.read_csv(path, dtype=str)


def stage_all_hf(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download + cache every HF source and return a dict of DataFrames."""
    _ensure_interim()
    out = {}
    for key in SOURCE_MANIFEST:
        print(f"[load_datasets] staging: {key}")
        out[key] = load_hf_source(key, force=force)
    return out


if __name__ == "__main__":
    frames = stage_all_hf()
    for k, df in frames.items():
        print(f"{k}: {df.shape}")
