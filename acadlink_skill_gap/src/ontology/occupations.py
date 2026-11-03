"""Occupation lexicon + title mapping (ESCO v1.2.1 + O*NET 30.3).

Produces `data/processed/occupation_lexicon.parquet` and mapping functions.
Mapping methodology:
  1. exact (ESCO preferredLabel/altLabels, O*NET Job Titles/Short Titles) conf 1.0
  2. token containment conf 0.95-0.97 (with the same phrase-size guards as skills)
  3. fuzzy token_set_ratio  >= 0.90 -> conf 0.90, 0.86-0.90 -> 0.86 (low/flag)
  4. embedding (MiniLM)      >= 0.90 -> conf 0.90, 0.82-0.90 -> 0.82 (low/flag)

"Software Engineer" vs "Software Developer" vs "Backend Engineer" are NOT
auto-considered identical: they map to occupation concepts / families, and only
records sharing the SAME canonical occupation id are treated as identical.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import rapidfuzz

from src.paths import PROCESSED_DIR, ESCO_DIR, ONET_DIR
from src.preprocessing.text_norm import normalize_text, token_containment
from src.ontology.skills import EmbeddingCache

FUZZY_HI = 0.90
FUZZY_LO = 0.86
EMB_HI = 0.90
EMB_LO = 0.82
OCC_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

ISCO_MAJOR = {  # ISCO-08 major groups (for occupation_family)
    "0": "Armed forces occupations", "1": "Managers", "2": "Professionals",
    "3": "Technicians and associate professionals", "4": "Clerical support workers",
    "5": "Service and sales workers", "6": "Skilled agricultural workers",
    "7": "Craft and related trades workers", "8": "Plant and machine operators",
    "9": "Elementary occupations",
}


def _isco_major_group(isco: str) -> str:
    if not isco or not str(isco).strip():
        return ""
    prefix = str(isco).strip()[:1]
    return ISCO_MAJOR.get(prefix, "")


# ---------------------------------------------------------------------------
# Lexicon construction
# ---------------------------------------------------------------------------
def _esco_occupations() -> pd.DataFrame:
    oc = pd.read_csv(ESCO_DIR / "occupations_en.csv", dtype=str)
    rows = []
    for r in oc.itertuples(index=False):
        aliases = [a.strip() for a in (r.altLabels or "").split("\n") if a.strip()] if r.altLabels else []
        isco = r.iscoGroup or ""
        family = _isco_major_group(isco) or ""
        rows.append({
            "occ_id": "ESCO-" + r.conceptUri.rsplit("/", 1)[-1],
            "name": r.preferredLabel,
            "aliases": aliases,
            "esco_uri": r.conceptUri,
            "isco_group": isco,
            "family": family,
            "onet_soc": pd.NA,
            "source": "ESCO",
            "description": r.description if isinstance(r.description, str) else "",
        })
    return pd.DataFrame(rows)


def _onet_occupations() -> pd.DataFrame:
    od = pd.read_csv(ONET_DIR / "occupation_data.csv", dtype=str)
    jt = pd.read_csv(ONET_DIR / "job_titles.csv", dtype=str)
    rows = []
    titles_by_soc = jt.groupby("O*NET-SOC Code")["Job Title"].apply(list).to_dict()
    for _, r in od.iterrows():
        soc = str(r["O*NET-SOC Code"])
        scode = re.sub(r"\.00$", "", soc)
        major = ""
        if scode.startswith("11-"): major = "Management Occupations"
        elif scode.startswith("13-"): major = "Business and Financial Operations Occupations"
        elif scode.startswith("15-"): major = "Computer and Mathematical Occupations"
        elif scode.startswith("17-"): major = "Architecture and Engineering Occupations"
        elif scode.startswith("19-"): major = "Life, Physical, and Social Science Occupations"
        elif scode.startswith("21-"): major = "Community and Social Service Occupations"
        elif scode.startswith("23-"): major = "Legal Occupations"
        elif scode.startswith("25-"): major = "Educational Instruction and Library Occupations"
        elif scode.startswith("27-"): major = "Arts, Design, Entertainment, Sports, and Media Occupations"
        elif scode.startswith("29-"): major = "Healthcare Practitioners and Technical Occupations"
        elif scode.startswith("31-"): major = "Healthcare Support Occupations"
        elif scode.startswith("33-"): major = "Protective Service Occupations"
        elif scode.startswith("35-"): major = "Food Preparation and Serving Related Occupations"
        elif scode.startswith("37-"): major = "Building and Grounds Cleaning and Maintenance Occupations"
        elif scode.startswith("39-"): major = "Personal Care and Service Occupations"
        elif scode.startswith("41-"): major = "Sales and Related Occupations"
        elif scode.startswith("43-"): major = "Office and Administrative Support Occupations"
        elif scode.startswith("45-"): major = "Farming, Fishing, and Forestry Occupations"
        elif scode.startswith("47-"): major = "Construction and Extraction Occupations"
        elif scode.startswith("49-"): major = "Installation, Maintenance, and Repair Occupations"
        elif scode.startswith("51-"): major = "Production Occupations"
        elif scode.startswith("53-"): major = "Transportation and Material Moving Occupations"
        elif scode.startswith("55-"): major = "Military Specific Occupations"
        alt = titles_by_soc.get(soc, []) + ([str(r["Title"])] if "Title" in od.columns else [])
        rows.append({
            "occ_id": "ONET-" + soc,
            "name": str(r["Title"]),
            "aliases": sorted(set(a for a in alt if isinstance(a, str) and a.strip())),
            "esco_uri": pd.NA,
            "isco_group": pd.NA,
            "family": major,
            "onet_soc": soc,
            "source": "ONET",
            "description": str(r["Description"]) if isinstance(r["Description"], str) else "",
        })
    return pd.DataFrame(rows)


def build_occupation_lexicon(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "occupation_lexicon.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    df = pd.concat([_esco_occupations(), _onet_occupations()], ignore_index=True)
    df = df[df["name"].notna()]
    df = df.drop_duplicates(subset=["occ_id"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------
class OccupationMapper:
    def __init__(self, lexicon: pd.DataFrame | None = None, use_embedding: bool = True):
        self.lexicon = lexicon if lexicon is not None else build_occupation_lexicon()
        self.embed_cache = EmbeddingCache(model=OCC_EMBED_MODEL) if use_embedding else None
        self._canon_emb = None
        self._build_index()

    def _build_index(self):
        self.id2rec: dict[str, dict] = {}
        self.alias2id: dict[str, str] = {}
        self.token2ids: dict[str, set[str]] = {}
        alias_owners: dict[str, set[str]] = {}
        recs = []
        for rec in self.lexicon.itertuples(index=False):
            name = str(rec.name)
            aliases = getattr(rec, "aliases", None)
            if aliases is None or (hasattr(aliases, "__len__") and len(aliases) == 0):
                aliases = []
            elif isinstance(aliases, np.ndarray):
                aliases = aliases.tolist()
            elif isinstance(aliases, str):
                aliases = [a for a in aliases.split("\n") if a.strip()]
            norm_aliases = [name] + [a for a in aliases if isinstance(a, str)]
            alias_norms = set()
            for a in norm_aliases:
                for key in (normalize_text(a, expand_abbrev=False), normalize_text(a, expand_abbrev=True)):
                    if key and len(key) >= 2:
                        alias_norms.add(key)
                        alias_owners.setdefault(key, set()).add(rec.occ_id)
            recs.append((rec.occ_id, name, alias_norms, rec))
        for key, owners in alias_owners.items():
            if len(owners) == 1:
                self.alias2id[key] = next(iter(owners))
        for occ_id, name, alias_norms, rec in recs:
            def _s(v):
                return v if isinstance(v, str) else ""
            self.id2rec[occ_id] = {
                "name": name,
                "norm_name": normalize_text(name, expand_abbrev=False),
                "family": _s(rec.family),
                "isco_group": _s(rec.isco_group),
                "onet_soc": _s(rec.onet_soc),
                "source": rec.source,
            }
            for a in alias_norms:
                for t in a.split():
                    self.token2ids.setdefault(t, set()).add(occ_id)

    def _canon_embeddings(self) -> tuple[list[str], dict[str, np.ndarray]]:
        if self._canon_emb is None:
            names = [self.id2rec[sid]["norm_name"] for sid in self.id2rec]
            em = self.embed_cache.embed_batch(names)
            self._canon_emb = (names, em)
        return self._canon_emb

    def _result(self, occ_id: str, conf: float, method: str) -> dict:
        rec = self.id2rec[occ_id]
        return {
            "occ_id": occ_id,
            "canonical_name": rec["name"],
            "occupation_family": rec["family"],
            "esco_occupation_id": occ_id if occ_id.startswith("ESCO") else "",
            "onet_occupation_id": rec["onet_soc"],
            "isco_group": rec["isco_group"],
            "confidence": round(float(conf), 3),
            "method": method,
        }

    def match(self, raw: str, use_embedding: bool = True) -> dict | None:
        n = normalize_text(raw, expand_abbrev=False)
        if not n or len(n) < 2:
            return None
        if n in self.alias2id:
            return self._result(self.alias2id[n], 1.0, "alias_exact")
        n_exp = normalize_text(raw, expand_abbrev=True)
        if n_exp != n and n_exp in self.alias2id:
            return self._result(self.alias2id[n_exp], 1.0, "alias_exact")
        raw_toks = set(n.split())
        cands: set[str] = set()
        for t in raw_toks:
            cands |= self.token2ids.get(t, set())
        best_id, best_score = None, -1.0
        for sid in cands:
            na = self.id2rec[sid]["norm_name"]
            target_toks = set(na.split())
            c = token_containment(n, na)  # raw -> target
            if c < 1.0:
                continue
            extra = len(target_toks - raw_toks)
            if extra > 0 and len(raw_toks) < 2:
                continue
            if extra > len(raw_toks):
                continue
            s = max(c, token_containment(na, n))
            if s > best_score:
                best_score, best_id = s, sid
        if best_id is not None:
            both_single = len(raw_toks) == 1 and len(self.id2rec[best_id]["norm_name"].split()) == 1
            return self._result(best_id, 0.95 if both_single else 0.97, "token_containment")
        fuzzy_best_id, fuzzy_score = None, 0.0
        for sid in list(cands)[:400]:
            nm = self.id2rec[sid]["norm_name"]
            s = rapidfuzz.fuzz.token_set_ratio(n, nm) / 100.0
            if s > fuzzy_score:
                fuzzy_score, fuzzy_best_id = s, sid
        if fuzzy_best_id and fuzzy_score >= FUZZY_HI:
            return self._result(fuzzy_best_id, FUZZY_HI, "fuzzy")
        if use_embedding and self.embed_cache is not None:
            names, em = self._canon_embeddings()
            q = self.embed_cache.embed_batch([n])[n]
            best_sid, best_sim = None, -1.0
            for sid, nm in zip(self.id2rec, names):
                sim = float(q @ em[nm])
                if sim > best_sim:
                    best_sim, best_sid = sim, sid
            if best_sid and best_sim >= EMB_HI:
                return self._result(best_sid, EMB_HI, "embedding")
            if best_sid and best_sim >= EMB_LO:
                return self._result(best_sid, EMB_LO, "embedding_low")
        return None

    def is_it(self, occ_id: str | None) -> bool:
        if not occ_id or occ_id not in self.id2rec:
            return False
        rec = self.id2rec[occ_id]
        isco = rec["isco_group"]
        soc = rec["onet_soc"]
        if isco and isco.startswith("25"):  # ISCO-08 ICT professionals
            return True
        if soc and soc.startswith("15-"):   # O*NET Computer and Mathematical
            return True
        if rec["family"] == "Computer and Mathematical Occupations":
            return True
        nm = rec["norm_name"]
        if rec["source"] == "ESCO" and rec["family"] in ("Professionals", "Technicians and associate professionals"):
            if any(k in nm for k in ("software", "developer", "programmer", "it ", "information",
                                     "cybersec", "digital", "computer", "data", "network",
                                     "database", "web", "cloud", "artificial", "machine learning")):
                return True
        return False


if __name__ == "__main__":
    lex = build_occupation_lexicon()
    print("occupation lexicon:", len(lex))
    m = OccupationMapper(lex)
    for t in ["Software Engineer", "Backend Developer", "Data Scientist", "ML Engineer",
              "Frontend Developer", "Accountant", "Sales Manager", "Full-Stack Developer"]:
        r = m.match(t)
        print(f"{t!r:24} ->", {k: r[k] for k in ("canonical_name", "occ_id", "occupation_family", "confidence")} if r else None,
              "| IT =", m.is_it(r["occ_id"]) if r else None)