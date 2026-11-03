"""Canonical skill ontology + multi-stage skill matcher for AcadLink.

Builds a unified skill lexicon from ESCO v1.2.1 and O*NET 30.3 (plus mined
high-frequency vendor tech names) and matches raw skill strings to canonical
skill concepts using staged exact / containment / fuzzy / embedding matching.

Stages (in order, first hit wins):
  1. alias exact   confidence 1.00
  2. token containment       confidence ~0.95-0.97
  3. fuzzy (rapidfuzz)       >= 0.90 -> 0.90, 0.86-0.90 -> 0.86 (flagged low)
  4. embedding (MiniLM)      >= 0.90 -> 0.90, >= 0.82 -> 0.82 (flagged low)

Low-confidence / unresolved matches go to the manual-review bucket and are
NEVER auto-assigned to a canonical skill.
"""
from __future__ import annotations

import ast
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rapidfuzz

import sys
import os
# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import ast
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rapidfuzz

from paths import PROCESSED_DIR, ESCO_DIR, ONET_DIR
from acadlink_skill_gap.src.preprocessing.text_norm import normalize_text, token_containment

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FUZZY_HI = 0.90
FUZZY_LO = 0.86
EMB_HI = 0.90
EMB_LO = 0.82
MINED_MIN_COUNT = 3


# ---------------------------------------------------------------------------
# Embedding cache (persisted so we never recompute full passes)
# ---------------------------------------------------------------------------
class EmbeddingCache:
    def __init__(self, path: Path = PROCESSED_DIR / "_text_embeddings.parquet",
                 model: str = EMBED_MODEL):
        self.path = path
        self.model = model
        self.data: dict[tuple[str, str], np.ndarray] = {}  # (model, text) -> vec
        self._model = None
        if path.exists():
            df = pd.read_parquet(path)
            if "model" not in df.columns:
                df["model"] = EMBED_MODEL  # migrate legacy (MiniLM) cache
            for r in df.itertuples(index=False):
                self.data[(r.model, r.text)] = np.asarray(r.embedding, dtype=np.float32)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model, local_files_only=True)
            self._model.eval()
        return self._model

    def embed_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        tset = set(texts)
        miss = [t for t in tset if (self.model, t) not in self.data]
        out = {t: self.data[(self.model, t)] for t in tset if (self.model, t) in self.data}
        if miss:
            model = self._get_model()
            vecs = model.encode(list(miss), batch_size=256, show_progress_bar=False,
                                normalize_embeddings=True)
            rows = []
            for t, v in zip(miss, vecs):
                v = np.asarray(v, dtype=np.float32)
                self.data[(self.model, t)] = v
                rows.append({"model": self.model, "text": t, "embedding": v})
            out.update({t: self.data[(self.model, t)] for t in miss})
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                combined = pd.DataFrame(rows)
                if self.path.exists():
                    combined = pd.concat([pd.read_parquet(self.path), combined], ignore_index=True)
                # atomic write: temp file + rename so a killed process never corrupts the cache
                fd, tmp = tempfile.mkstemp(prefix=".emb_", suffix=".parquet",
                                           dir=self.path.parent)
                os.close(fd)
                combined.to_parquet(tmp, index=False)
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        return out


def _clean(v, default: str = ""):
    return v if isinstance(v, str) and v.strip() else default


# ---------------------------------------------------------------------------
# Canonical skill lexicon construction
# ---------------------------------------------------------------------------
def _esco_skill_records() -> pd.DataFrame:
    sk = pd.read_csv(ESCO_DIR / "skills_en.csv", dtype=str)
    br = pd.read_csv(ESCO_DIR / "broaderRelationsSkillPillar_en.csv", dtype=str)
    parent = br.drop_duplicates("conceptUri").set_index("conceptUri")["broaderLabel"].to_dict()
    rows = []
    for r in sk.itertuples(index=False):
        concept_uri = r.conceptUri
        raw_alt = _clean(r.altLabels)
        aliases = [a.strip() for a in raw_alt.split("\n") if a.strip()] if raw_alt else []
        rows.append({
            "skill_id": concept_uri.rsplit("/", 1)[-1],
            "canonical_name": r.preferredLabel,
            "aliases": aliases,
            "description": _clean(r.description) or _clean(r.scopeNote),
            "skill_type": "knowledge" if r.skillType == "knowledge" else "skill_competence",
            "parent_skill": parent.get(concept_uri, ""),
            "esco_id": concept_uri,
            "onet_id": pd.NA,
            "source": "ESCO",
        })
    return pd.DataFrame(rows)


def _onet_skill_records() -> pd.DataFrame:
    rows = []
    sw = pd.read_csv(ONET_DIR / "software_skills.csv", dtype=str)
    for eid, elem in sw.drop_duplicates("Element ID")[["Element ID", "Element Name"]].itertuples(index=False):
        examples = sw[sw["Element ID"] == eid]["Workplace Example"].dropna().str.strip().unique().tolist()
        rows.append({
            "skill_id": "ONET-SOFT-" + eid,
            "canonical_name": elem,
            "aliases": examples,
            "description": "",
            "skill_type": "technology",
            "parent_skill": "technology and information technology",
            "esco_id": pd.NA,
            "onet_id": eid,
            "source": "ONET",
        })
    for fname, prefix, s_type, parent in [
        ("knowledge", "ONET-KNOW", "knowledge", "knowledge"),
        ("essential_skills", "ONET-ESS", "skill_competence", ""),
        ("abilities", "ONET-AB", "ability", ""),
    ]:
        d = pd.read_csv(ONET_DIR / f"{fname}.csv", dtype=str)
        for eid, elem in d.drop_duplicates("Element ID")[["Element ID", "Element Name"]].itertuples(index=False):
            rows.append({
                "skill_id": f"{prefix}-{eid}",
                "canonical_name": elem,
                "aliases": [],
                "description": "",
                "skill_type": s_type,
                "parent_skill": parent,
                "esco_id": pd.NA,
                "onet_id": eid,
                "source": "ONET",
            })
    return pd.DataFrame(rows)


def _mined_vendor_skills() -> pd.DataFrame:
    """High-frequency vendor tech terms from O*NET workplace examples + job_skill_set."""
    counts: dict[str, int] = {}
    ex = pd.read_csv(ONET_DIR / "software_skills.csv", dtype=str)
    for v in ex["Workplace Example"].dropna().str.strip():
        counts[v] = counts.get(v, 0) + 1
    from src.data.load_datasets import load_hf_source
    jss = load_hf_source("job_skill_set")
    for s in jss["job_skill_set"]:
        try:
            lst = ast.literal_eval(s) if isinstance(s, str) else s
        except Exception:
            lst = []
        for item in (lst or []):
            if isinstance(item, str) and item.strip():
                v = item.strip()
                counts[v] = counts.get(v, 0) + 1
    rows = []
    for name, c in counts.items():
        if c >= MINED_MIN_COUNT:
            rows.append({
                "skill_id": "MINE-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "canonical_name": name,
                "aliases": [],
                "description": "",
                "skill_type": "technology",
                "parent_skill": "technology and information technology",
                "esco_id": pd.NA,
                "onet_id": pd.NA,
                "source": "MINED",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "skill_id", "canonical_name", "aliases", "description", "skill_type",
        "parent_skill", "esco_id", "onet_id", "source"])


def build_canonical_skills(force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / "canonical_skills.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    frames = [_esco_skill_records(), _onet_skill_records(), _mined_vendor_skills()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["canonical_name"].notna()]
    df = df[df["canonical_name"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["canonical_name"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Matcher (index over canonical records)
# ---------------------------------------------------------------------------
class SkillMatcher:
    PROVIDER_RANK = {"ESCO": 0, "ONET": 1, "MINED": 2}

    def __init__(self, canonical: pd.DataFrame | None = None, use_embedding: bool = True):
        self.canonical = canonical if canonical is not None else build_canonical_skills()
        self.embed_cache = EmbeddingCache() if use_embedding else None
        self._canon_emb = None
        self._build_index()

    def _build_index(self):
        self.id2rec: dict[str, dict] = {}
        self.alias2id: dict[str, str] = {}
        self.token2ids: dict[str, set[str]] = {}

        # pass 1: canonical metadata + alias occurrence map (to drop ambiguous aliases)
        alias_owners: dict[str, set[str]] = {}
        recs = []
        for rec in self.canonical.itertuples(index=False):
            name = str(rec.canonical_name)
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
                        alias_owners.setdefault(key, set()).add(rec.skill_id)
            recs.append((rec.skill_id, name, alias_norms, rec))
        # exact alias allowed only when a given alias string maps to ONE canonical
        for key, owners in alias_owners.items():
            if len(owners) == 1:
                self.alias2id[key] = next(iter(owners))
        # pass 2: index
        for sid, name, alias_norms, rec in recs:
            self.id2rec[sid] = {
                "name": name,
                "norm_name": normalize_text(name, expand_abbrev=False),
                "alias_norms": alias_norms,
                "skill_type": rec.skill_type,
                "source": rec.source,
                "esco_id": rec.esco_id,
                "onet_id": rec.onet_id,
                "parent_skill": str(rec.parent_skill or ""),
            }
            for a in alias_norms:
                for t in a.split():
                    self.token2ids.setdefault(t, set()).add(sid)

    def _pkg_key(self, rec: dict) -> tuple:
        # deterministic tie-break: provider rank, then shorter canonical label
        return (self.PROVIDER_RANK.get(rec["source"], 9), len(rec["norm_name"].split()), rec["norm_name"])

    def _canon_embeddings(self) -> tuple[list[str], dict[str, np.ndarray]]:
        if self._canon_emb is None:
            names = [self.id2rec[sid]["norm_name"] for sid in self.id2rec]
            em = self.embed_cache.embed_batch(names)
            self._canon_emb = (names, em)
        return self._canon_emb

    # -- public API --------------------------------------------------------
    def match(self, raw: str, required: bool = False, deep: bool = True) -> dict | None:
        """Staged match. `required=True` enables the (slow) embedding stage;
        `deep=False` runs only alias-exact + token-containment (fast, used for
        weak-positive lists where fuzzy/embedding add little)."""
        n = normalize_text(raw, expand_abbrev=False)
        if not n or len(n) < 2:
            return None

        # 1. exact alias (checks expanded form too for abbreviations)
        if n in self.alias2id:
            return self._result(self.alias2id[n], 1.0, "alias_exact")

        # 2. token containment (both directions), tie-broken deterministically.
        #    Guards: raw must be fully contained in a candidate, and the candidate
        #    must not be a much larger phrase (prevents "react" -> "react to mining emergencies").
        raw_toks = set(n.split())
        cands: set[str] = set()
        for t in raw_toks:
            cands |= self.token2ids.get(t, set())
        best_id, best_score = None, -1.0
        best_key = None
        for sid in cands:
            na = self.id2rec[sid]["norm_name"]
            target_toks = set(na.split())
            c_ab = token_containment(n, na)   # raw -> target
            if c_ab < 1.0:
                continue
            extra = len(target_toks - raw_toks)
            if extra > 0 and len(raw_toks) < 2:
                continue                       # single-token raw must not match a bigger phrase
            if extra > len(raw_toks):
                continue                       # target phrase far bigger than raw
            s = max(c_ab, token_containment(na, n))
            if s > best_score:
                best_score, best_id, best_key = s, sid, self._pkg_key(self.id2rec[sid])
            elif s == best_score and self._pkg_key(self.id2rec[sid]) < best_key:
                best_id, best_key = sid, self._pkg_key(self.id2rec[sid])
        if best_id is not None:
            both_single = (len(raw_toks) == len(set(n.split())) and
                           len(raw_toks) == 1 and len(self.id2rec[best_id]["norm_name"].split()) == 1)
            return self._result(best_id, 0.95 if both_single else 0.97, "token_containment")

        # 3. fuzzy on token-overlap candidates
        fuzzy_best_id, fuzzy_score = None, 0.0
        for sid in list(cands)[:400]:
            rec = self.id2rec[sid]
            s1 = rapidfuzz.fuzz.ratio(n, rec["norm_name"]) / 100.0
            s2 = rapidfuzz.fuzz.token_set_ratio(n, rec["norm_name"]) / 100.0
            s = max(s1, s2)
            if s > fuzzy_score:
                fuzzy_score, fuzzy_best_id = s, sid
        if fuzzy_best_id and fuzzy_score >= FUZZY_HI:
            return self._result(fuzzy_best_id, FUZZY_HI, "fuzzy")
        if fuzzy_best_id and fuzzy_score >= FUZZY_LO:
            return self._result(fuzzy_best_id, FUZZY_LO, "fuzzy_low")

        # 4. embedding (slow; canonical labels only)
        if self.embed_cache is not None and required:
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

    def _result(self, sid: str, conf: float, method: str) -> dict:
        rec = self.id2rec[sid]
        return {
            "skill_id": sid,
            "canonical_name": rec["name"],
            "skill_type": rec["skill_type"],
            "source": rec["source"],
            "confidence": round(float(conf), 3),
            "method": method,
            "parent_skill": rec["parent_skill"],
        }

    def similarity(self, a: str, b: str) -> float:
        if self.embed_cache is None:
            return 0.0
        na, nb = normalize_text(a), normalize_text(b)
        em = self.embed_cache.embed_batch([na, nb])
        if na not in em or nb not in em:
            return 0.0
        return float(em[na] @ em[nb])


if __name__ == "__main__":
    df = build_canonical_skills()
    print("canonical skills:", len(df))
    print(df.groupby("source").size().to_dict())
    m = SkillMatcher(df)
    for probe in ["Python programming", "Python", "RESTful APIs", "REST API development",
                  "Kubernetes", "Adobe Photoshop", "Distributed Systems",
                  "Object-oriented Programming", "SQL", "communication"]:
        print(f"{probe!r:35} ->", m.match(probe, required=True))