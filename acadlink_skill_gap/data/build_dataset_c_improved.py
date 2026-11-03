"""Build separate, auditable Dataset C outputs from the existing Dataset A/B.

Defaults: 200 resumes x 20 actual job IDs, seed 42; --max-resumes 0 and
--max-jobs 0 select all IDs. Pairing is constructed; resume text is untouched.
Unresolved cases have NULL labels and go only to the review output.
Confidence is a policy score, not a calibrated probability. No balancing.

SkillMatcher.match returns one dict or None, not a list of extracted skills.
Its containment/fuzzy matching and punctuation-stripping normalization cannot
prove presence. This builder uses complete canonical/alias phrases with source
offsets, preserving C++, C#, .NET and word boundaries. It reads cached ontology
data directly, avoiding rebuilds or embedding-cache writes.

Run --self-test for embedded policy and validation tests (no output files).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import unittest

import numpy as np
import pandas as pd

ACADLINK_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ACADLINK_ROOT / "data/processed"
RELATIONS_PATH = ACADLINK_ROOT / "data/raw/ontology/esco/broaderRelationsSkillPillar_en.csv"
POLICY_VERSION = "dataset-c-evidence-v1"
LABEL_MISSING, LABEL_PARTIAL, LABEL_PRESENT = 0, 1, 2

# Narrow equivalences only; never recursively expand the whole resume.
VARIANTS = {
    "kubernetes": ["k8s"], "javascript": ["js"], "typescript": ["ts"],
    "structured query language": ["sql"], "natural language processing": ["nlp"],
    "google cloud platform": ["gcp"], "amazon web services": ["aws"],
    "object oriented programming": ["oop"], "scikit learn": ["sklearn"],
}
AMBIGUOUS = {"c", "r", "j", "go", "ada", "ai", "access", "word", "excel",
             "office", "windows", "react", "spark", "spring", "net"}
NEGATION = re.compile(
    r"\b(?:no|without|lack(?:ing)?|not|never)\s+(?:(?:\w+)\s+){0,5}$", re.I)
ASPIRATION = re.compile(
    r"\b(?:want(?:ing)? to learn|plan(?:ning)? to learn|interested in learning|"
    r"seeking (?:a |an )?|willing to learn|will learn)\s*$", re.I)
LIMITED = re.compile(
    r"\b(?:(?:basic|beginner|introductory)\s+"
    r"(?:(?:knowledge|experience|understanding|proficiency|skills|exposure)\s+)?|"
    r"limited\s+(?:knowledge|experience|understanding|proficiency|skills|exposure)\s+)"
    r"(?:(?:of|in|with|to)\s+)?$", re.I)
BLOCKED_CLAUSES = {"management", "development", "skills", "experience", "knowledge",
                   "ability", "ability to", "use", "work", "and", "or", "training"}


def _clean(value):
    return value.strip() if isinstance(value, str) else ""


def _normalized_with_offsets(text):
    """Normalize case/Unicode/separators, mapping each character to the source."""
    chars, offsets = [], []
    for i, char in enumerate(text):
        for c in unicodedata.normalize("NFKC", char).casefold():
            c = c if c.isalnum() or c in "+#." else " "
            if c == " " and (not chars or chars[-1] == " "):
                continue
            chars.append(c)
            offsets.append(i)
    return "".join(chars), offsets


def _normalize_skill_name(skill):
    return _normalized_with_offsets(_clean(skill))[0].strip()


def _unambiguous_local_context(term, prefix, suffix):
    """A nearby technology word alone cannot turn ordinary 'go' into Go."""
    before, after = _normalize_skill_name(prefix), _normalize_skill_name(suffix)
    if term in {"c", "r", "j", "go", "ada"}:
        return bool(re.match(r"^(?:programming|language|software)\b", after) or
                    re.search(r"\b(?:programming|coding|written|developed)\s+(?:in|with)$", before))
    if term in {"access", "word", "excel", "office", "windows"}:
        return bool(re.search(r"\b(?:microsoft|ms)$", before) or
                    (term == "excel" and re.match(r"^spreadsheets?\b", after)) or
                    (term == "windows" and re.match(r"^(?:server|operating system)\b", after)))
    if term == "react":
        return bool(re.match(r"^(?:js|javascript|framework)\b", after))
    if term == "spark":
        return bool(re.search(r"\bapache$", before) or re.match(r"^sql\b", after))
    if term == "spring":
        return bool(re.match(r"^(?:boot|framework)\b", after))
    if term == "ai":
        return bool(re.match(r"^(?:models?|algorithms?|systems?)\b", after))
    return False


@lru_cache(maxsize=50000)
def _phrase_pattern(phrase):
    # java/javascript, C/C++/C#, SQL/NoSQL and node/Node.js must not collide.
    return re.compile(r"(?<![\w+#.])" + re.escape(phrase).replace(r"\ ", r"\s+")
                      + r"(?![\w+#]|\.[\w])")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _aliases(value):
    if isinstance(value, str):
        return [v for v in value.splitlines() if v.strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


class EvidenceIndex:
    """Complete phrases and direct ESCO concept edges; no generic sibling links."""

    def __init__(self, canonical, relations):
        self.records = defaultdict(list)
        self.names = defaultdict(set)
        self.owners = defaultdict(set)
        self.relations = defaultdict(set)
        for row in canonical.to_dict("records"):
            sid = _clean(row["skill_id"])
            name = _normalize_skill_name(row["canonical_name"])
            row = dict(row, aliases=_aliases(row.get("aliases")))
            self.records[sid].append(row)
            self.names[name].add(sid)
            for term in [row["canonical_name"]] + row["aliases"]:
                self.owners[_normalize_skill_name(term)].add(name)
        uri_ids = defaultdict(set)
        for sid, rows in self.records.items():
            for row in rows:
                if _clean(row.get("esco_id")):
                    uri_ids[row["esco_id"]].add(sid)
        for row in relations.to_dict("records"):
            if (row.get("conceptType") != "KnowledgeSkillCompetence" or
                    row.get("broaderType") != "KnowledgeSkillCompetence"):
                continue  # Shared SkillGroup is not sufficient evidence.
            for child in uri_ids.get(row["conceptUri"], ()):
                for parent in uri_ids.get(row["broaderUri"], ()):
                    if child != parent:
                        self.relations[child].add((parent, "target_narrower_than_evidence"))
                        self.relations[parent].add((child, "target_broader_than_evidence"))

    @lru_cache(maxsize=10000)
    def terms(self, sid, name):
        norm = _normalize_skill_name(name)
        terms = {norm: "exact_phrase"} if norm else {}
        # Repeated IDs do not make unrelated canonical names equivalent.
        rows = [r for r in self.records.get(sid, ())
                if _normalize_skill_name(r["canonical_name"]) == norm]
        for row in rows:
            for alias in row["aliases"]:
                term = _normalize_skill_name(alias)
                if term and self.owners[term] == {norm}:
                    terms.setdefault(term, "canonical_alias")
        for full, short_forms in VARIANTS.items():
            if norm in [full] + short_forms:
                for term in [full] + short_forms:
                    if not self.owners.get(term) or self.owners[term] == {norm}:
                        terms.setdefault(term, "normalized_variant")
        return tuple(sorted(terms.items(), key=lambda x: (-len(x[0]), x[0])))

    def related_terms(self, sid):
        for related_id, relation in sorted(self.relations.get(sid, ())):
            for row in self.records[related_id]:
                for term, method in self.terms(related_id, row["canonical_name"]):
                    yield term, related_id, relation, method

    def ambiguous_aliases(self, sid, name):
        norm = _normalize_skill_name(name)
        for row in self.records.get(sid, ()):
            if _normalize_skill_name(row["canonical_name"]) == norm:
                for alias in row["aliases"]:
                    term = _normalize_skill_name(alias)
                    if term and self.owners[term] != {norm}:
                        yield term


class ResumeEvidence:
    def __init__(self, text):
        self.text = text
        self.normalized, self.offsets = _normalized_with_offsets(text)
        self.tokens = set(re.findall(r"\w+|[+#]", self.normalized))
        self.cache = {}

    def find(self, term):
        if term in self.cache:
            return self.cache[term]
        # Prefilter only; a complete boundary-aware phrase still must match.
        if not set(re.findall(r"\w+|[+#]", term)).issubset(self.tokens):
            self.cache[term] = (None, None)
            return None, None
        good, uncertain = None, None
        for match in _phrase_pattern(term).finditer(self.normalized):
            start, end = self.offsets[match.start()], self.offsets[match.end() - 1] + 1
            prefix = re.split(r"[.;\n!?]", self.text[max(0, start - 100):start])[-1]
            context = self.text[max(0, start - 65):min(len(self.text), end + 65)]
            reason = ""
            if NEGATION.search(prefix):
                reason = "negated_or_uncertain_scope"
            elif ASPIRATION.search(prefix):
                reason = "aspirational_mention"
            elif term in AMBIGUOUS and not _unambiguous_local_context(
                    term, self.text[max(0, start - 65):start], self.text[end:end + 65]):
                reason = "ambiguous_term_without_technical_context"
            hit = {"resume_evidence": context, "matched_text": self.text[start:end],
                   "matched_term": term, "evidence_start": start, "evidence_end": end,
                   "incomplete": bool(LIMITED.search(prefix))}
            if reason:
                uncertain = uncertain or (hit, reason)
            elif good is None or (good["incomplete"] and not hit["incomplete"]):
                good = hit
        self.cache[term] = good, uncertain
        return good, uncertain


def _decision(label, source, confidence, hit=None, related="", relation="", reason=""):
    return {"skill_status": label, "label_source": source, "confidence": confidence,
            "resume_evidence": "", "matched_text": "", "matched_term": "",
            "evidence_start": -1, "evidence_end": -1,
            **({k: v for k, v in hit.items() if k != "incomplete"} if hit else {}),
            "related_skill_id": related, "evidence_relation": relation,
            "needs_review": label is None, "review_reason": reason}


def _classify_record(resume, skill, sid, index):
    if len(resume.normalized.strip()) < 40 or len(resume.tokens) < 8:
        return _decision(None, "review_invalid_resume", 0., reason="insufficient_resume_text")
    norm = _normalize_skill_name(skill)
    if not norm or sid not in index.records:
        return _decision(None, "review_invalid_requirement", 0., reason="invalid_skill_or_unknown_id")
    if not any(_normalize_skill_name(r["canonical_name"]) == norm for r in index.records[sid]):
        return _decision(None, "review_canonical_mismatch", 0., reason="skill_id_name_mismatch")
    uncertain, partial = None, None
    for term, source in index.terms(sid, skill):
        hit, suspect = resume.find(term)
        uncertain = uncertain or suspect
        if hit:
            if hit["incomplete"]:
                partial = partial or _decision(1, "explicit_limited_evidence", .75, hit)
            else:
                return _decision(2, source, .95 if source == "exact_phrase" else .9, hit)
    if uncertain:
        return _decision(None, "review_ambiguous_evidence", 0., uncertain[0], reason=uncertain[1])
    if partial:
        return partial
    for term in index.ambiguous_aliases(sid, skill):
        hit, suspect = resume.find(term)
        if hit or suspect:
            return _decision(None, "review_ambiguous_alias", 0., hit or suspect[0],
                             reason="alias_maps_to_multiple_concepts")
    broad_related = None
    for term, related_id, relation, _ in index.related_terms(sid):
        hit, suspect = resume.find(term)
        uncertain = uncertain or suspect
        if hit:
            if relation == "target_narrower_than_evidence":
                # A broad prerequisite such as thinking quickly cannot establish
                # partial competence in a specialist skill such as childbirth.
                broad_related = broad_related or (hit, related_id, relation)
                continue
            return _decision(1, "esco_direct_related_skill", .7, hit, related_id, relation)
    # One recognized concept in a required conjunction is incomplete evidence.
    # Generic fragments such as "management" never qualify by themselves.
    clauses = norm.split(" and ")
    if len(clauses) > 1:
        hits, generic_hit = [], None
        for clause in clauses:
            hit, suspect = resume.find(clause)
            uncertain = uncertain or suspect
            if hit and clause in BLOCKED_CLAUSES:
                generic_hit = generic_hit or hit
            elif hit and clause in index.names:
                hits.append((hit, sorted(index.names[clause])[0]))
        if len(hits) == len(clauses):
            return _decision(None, "review_composite_evidence", 0., hits[0][0],
                             reason="all_conjuncts_mentioned_separately_requires_review")
        if hits:
            return _decision(1, "incomplete_conjunct_evidence", .65, hits[0][0],
                             hits[0][1], "one_required_conjunct")
        if generic_hit:
            return _decision(None, "review_generic_conjunct", 0., generic_hit,
                             reason="generic_conjunct_is_insufficient_evidence")
    # Upstream text cleaning can erase meaningful technology punctuation.
    lossy_base = re.sub(r"[+#.]", "", norm).strip()
    if lossy_base != norm and lossy_base:
        hit, suspect = resume.find(lossy_base)
        if hit or suspect:
            return _decision(None, "review_lossy_technology_name", 0., hit or suspect[0],
                             reason="source_may_have_lost_skill_punctuation")
    if uncertain:
        return _decision(None, "review_ambiguous_related_evidence", 0., uncertain[0],
                         reason=uncertain[1])
    if broad_related:
        hit, related_id, relation = broad_related
        return _decision(None, "review_broad_related_evidence", 0., hit, related_id, relation,
                         reason="broad_parent_does_not_establish_specialist_competence")
    return _decision(0, "no_meaningful_evidence", .7)


def _require_columns(df, columns, name):
    absent = sorted(set(columns) - set(df.columns))
    if absent:
        raise ValueError(f"{name}: required columns absent: {absent}")


def _select_ids(values, maximum, seed):
    ids = sorted(set(values))
    if maximum and len(ids) > maximum:
        ids = sorted(np.random.default_rng(seed).choice(ids, maximum, replace=False).tolist())
    return ids


def _split_resumes(resume_texts, seed):
    """Group identical normalized resume texts as well as IDs to avoid leakage."""
    groups = defaultdict(list)
    for rid, text in sorted(resume_texts.items()):
        groups[_digest(_normalize_skill_name(text))].append(rid)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three distinct resume texts are needed for isolated splits")
    np.random.default_rng(seed).shuffle(keys)
    n_test, n_val = max(1, round(len(keys) * .15)), max(1, round(len(keys) * .15))
    n_train = len(keys) - n_val - n_test
    assignment = {}
    for name, part in [("train", keys[:n_train]), ("val", keys[n_train:n_train + n_val]),
                       ("test", keys[n_train + n_val:])]:
        for key in part:
            for rid in groups[key]:
                assignment[rid] = name
    return assignment, len(groups)


def _distribution(df):
    counts = df.skill_status.value_counts()
    return {name: int(counts.get(label, 0))
            for label, name in [(0, "missing"), (1, "partial"), (2, "present")]}


def validate_outputs(all_records, splits):
    """Fail before saving on invalid keys, missing values, evidence or leakage."""
    required = ["record_id", "resume_id", "job_id", "resume_text", "skill_id", "skill_name",
                "required_skills_raw", "skill_importance", "label_source", "confidence",
                "resume_evidence", "source_requirement_id", "source_requirement_json",
                "source_resume_rows", "needs_review", "review_reason", "split"]
    _require_columns(all_records, required + ["skill_status"], "improved records")
    if all_records.empty or all_records.record_id.duplicated().any():
        raise ValueError("Empty dataset or duplicate record identifiers")
    if all_records[required].isna().any().any():
        raise ValueError("Unexpected missing required values")
    for col in ["record_id", "resume_id", "job_id", "skill_id", "skill_name",
                "required_skills_raw", "label_source", "source_requirement_id"]:
        if all_records[col].str.strip().eq("").any():
            raise ValueError(f"Empty required value in {col}")
    if not all_records.confidence.between(0, 1).all():
        raise ValueError("Confidence outside [0, 1]")
    if not all_records.skill_importance.between(0, 1).all():
        raise ValueError("Skill importance outside [0, 1]")
    if not all_records.skill_status.isna().equals(all_records.needs_review):
        raise ValueError("Review cases must have null labels; labeled cases cannot need review")
    allowed = {0: {"no_meaningful_evidence"},
               1: {"explicit_limited_evidence", "esco_direct_related_skill", "incomplete_conjunct_evidence"},
               2: {"exact_phrase", "canonical_alias", "normalized_variant"}}
    for row in all_records.to_dict("records"):
        if row["needs_review"]:
            if not row["review_reason"] or not row["label_source"].startswith("review_"):
                raise ValueError("Review case lacks a reason/source")
        else:
            label = int(row["skill_status"])
            if row["label_source"] not in allowed.get(label, set()) or row["review_reason"]:
                raise ValueError("Label/source inconsistency")
            if not row["resume_text"].strip():
                raise ValueError("Labeled case lacks resume text")
            if label == 0 and (row["resume_evidence"] or row["matched_text"]):
                raise ValueError("Missing label has affirmative evidence")
            if label in (1, 2) and not row["matched_text"]:
                raise ValueError("Positive/partial label lacks a source span")
            if row["label_source"] in {"esco_direct_related_skill", "incomplete_conjunct_evidence"}:
                if not row["related_skill_id"] or not row["evidence_relation"]:
                    raise ValueError("Related evidence lacks relationship provenance")
        if row["matched_text"]:
            start, end = row["evidence_start"], row["evidence_end"]
            if not 0 <= start < end <= len(row["resume_text"]):
                raise ValueError("Invalid evidence offsets")
            if row["resume_text"][start:end] != row["matched_text"]:
                raise ValueError("Evidence is not an exact slice of source resume")
            if row["resume_evidence"] not in row["resume_text"]:
                raise ValueError("Evidence context not present in source resume")
            if _normalize_skill_name(row["matched_text"]) != row["matched_term"]:
                raise ValueError("Evidence does not normalize to the claimed complete phrase")
    labeled = all_records[~all_records.needs_review]
    joined = pd.concat(splits.values(), ignore_index=True)
    if joined.record_id.duplicated().any() or set(joined.record_id) != set(labeled.record_id):
        raise ValueError("Splits must partition all and only labeled records")
    isolation = {}
    names = list(splits)
    for i, left in enumerate(names):
        if splits[left].empty:
            raise ValueError(f"Empty {left} split")
        if not splits[left].split.eq(left).all():
            raise ValueError("Row assigned to the wrong split")
        for right in names[i + 1:]:
            overlap = set(splits[left].resume_id) & set(splits[right].resume_id)
            hashes_left = {_digest(_normalize_skill_name(t)) for t in splits[left].resume_text.unique()}
            hashes_right = {_digest(_normalize_skill_name(t)) for t in splits[right].resume_text.unique()}
            if overlap or hashes_left & hashes_right:
                raise ValueError("Resume ID or identical resume text leaked across splits")
            isolation[f"{left}/{right}"] = {"resume_id_overlap": 0, "resume_text_overlap": 0}
    return isolation


def build_dataset_c(processed_dir=PROCESSED_DIR, relations_path=RELATIONS_PATH,
                    max_resumes=200, max_jobs=20, seed=42, overwrite=False):
    processed_dir, relations_path = Path(processed_dir), Path(relations_path)
    output_names = ["dataset_C_skill_gap_matching_improved.parquet",
                    "dataset_C_review_improved.parquet", "dataset_C_validation_improved.json"]
    output_names += [f"dataset_C_split_{s}_improved.parquet" for s in ("train", "val", "test")]
    if not overwrite and any((processed_dir / n).exists() for n in output_names):
        raise FileExistsError("Improved outputs exist; --overwrite-improved replaces only those files")
    protected = {p: _file_hash(p) for p in processed_dir.iterdir()
                 if p.is_file() and "_improved" not in p.name}
    protected[relations_path] = _file_hash(relations_path)
    a_path = processed_dir / "dataset_A_resume_skill_extraction.parquet"
    b_path = processed_dir / "dataset_B.parquet"
    c_path = processed_dir / "canonical_skills.parquet"
    a, b, canonical = (pd.read_parquet(p) for p in (a_path, b_path, c_path))
    relations = pd.read_csv(relations_path).fillna("")
    _require_columns(a, ["resume_id", "resume_text", "skill", "skill_id"], "Dataset A")
    _require_columns(b, ["jd_id", "jd_entity", "skill_id", "skill_name", "requirement",
                         "importance", "confidence", "label_sources"], "Dataset B")
    _require_columns(canonical, ["skill_id", "canonical_name", "aliases", "esco_id"], "canonical")
    _require_columns(relations, ["conceptType", "conceptUri", "broaderType", "broaderUri"], "ESCO edges")
    if a.resume_id.isna().any() or b.jd_id.isna().any():
        raise ValueError("Missing source IDs require upstream review; cannot group safely")
    a = a.copy()
    a["resume_id"] = a.resume_id.astype(str)
    source_counts = {"dataset_a_rows": len(a), "dataset_b_rows": len(b),
                     "available_resumes": a.resume_id.nunique(), "available_jobs": b.jd_id.nunique()}
    resume_ids = _select_ids(a.resume_id, max_resumes, seed)
    job_ids = _select_ids(b.jd_id, max_jobs, seed + 1)
    selected_a, selected_b = a[a.resume_id.isin(resume_ids)], b[b.jd_id.isin(job_ids)]
    resume_data, conflicting_resumes = {}, []
    for rid, rows in selected_a.groupby("resume_id", sort=True):
        raw_texts = sorted({v for v in rows.resume_text if isinstance(v, str) and v.strip()})
        conflict = len(raw_texts) != 1 or rows.resume_text.isna().any()
        text = raw_texts[0] if raw_texts else ""
        if conflict:
            conflicting_resumes.append(rid)
        resume_data[rid] = (text, rows.index.tolist(), sorted(set(rows.skill.dropna())), conflict)
    assignment, text_groups = _split_resumes({rid: v[0] for rid, v in resume_data.items()}, seed)
    # Hash the entire B row, including provenance. Distinct importance, requirement
    # type, job ID or source metadata must retain a distinct skill-level record.
    requirements = {}
    for source_row, row in selected_b.iterrows():
        payload = row.to_dict()
        key = _digest(payload)
        if key not in requirements:
            requirements[key] = (payload, [])
        requirements[key][1].append(int(source_row))
    index = EvidenceIndex(canonical, relations)
    records = []
    for number, (rid, (text, source_rows, extracted, conflict)) in enumerate(resume_data.items(), 1):
        resume = ResumeEvidence(text)
        for requirement_id, (row, b_rows) in sorted(requirements.items()):
            name, sid = _clean(row["skill_name"]), _clean(row["skill_id"])
            if conflict:
                decision = _decision(None, "review_conflicting_resume", 0.,
                                     reason="missing_or_conflicting_source_resume_text")
            else:
                decision = _classify_record(resume, name, sid, index)
            importance = pd.to_numeric(row["importance"], errors="coerce")
            if pd.isna(importance) or not 0 <= importance <= 1:
                raise ValueError(f"Invalid source importance in B rows {b_rows}")
            records.append({
                "record_id": "CI-" + _digest([rid, requirement_id]),
                "resume_id": rid, "job_id": _clean(row["jd_id"]),
                "job_title": _clean(row["jd_entity"]), "resume_text": text,
                "resume_skills_raw": ", ".join(extracted),
                "required_skills_raw": name, "required_skills_normalized": _normalize_skill_name(name),
                "skill_id": sid, "skill_name": name, "skill_importance": float(importance),
                "requirement_type": _clean(row["requirement"]), "preferred_skills": _clean(row["requirement"]),
                "occupation_id": _clean(row.get("occupation_id")),
                "occupation_name": _clean(row.get("occupation_name")),
                "occupation_family": _clean(row.get("occupation_family")),
                "source_requirement_id": requirement_id, "source_requirement_rows": json.dumps(b_rows),
                "source_requirement_json": json.dumps(row, sort_keys=True, ensure_ascii=False, default=str),
                "source_resume_rows": json.dumps(source_rows),
                "source_resume_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_requirement_label_sources": _clean(row["label_sources"]),
                "source_requirement_confidence": float(row["confidence"]),
                "pairing_source": "seeded_resume_job_cross_product",
                "policy_version": POLICY_VERSION, "split": assignment[rid], **decision})
        if number % 50 == 0 or number == len(resume_data):
            print(f"Labeled {number}/{len(resume_data)} resumes; {len(records)} records", flush=True)
    frame = pd.DataFrame(records)
    frame["skill_status"] = frame.skill_status.astype("Int64")
    labeled, review = frame[~frame.needs_review].copy(), frame[frame.needs_review].copy()
    splits = {s: labeled[labeled.split == s].copy() for s in ("train", "val", "test")}
    isolation = validate_outputs(frame, splits)
    report = {
        "policy_version": POLICY_VERSION,
        "build_command": [sys.executable, "-B", str(Path(__file__).resolve()),
                          "--processed-dir", str(processed_dir), "--relations-path", str(relations_path),
                          "--max-resumes", str(max_resumes), "--max-jobs", str(max_jobs), "--seed", str(seed)]
                         + (["--overwrite-improved"] if overwrite else []),
        "parameters": {"max_resumes": max_resumes, "max_jobs": max_jobs, "seed": seed,
                       "split_ratios": [.7, .15, .15]},
        "source_counts": source_counts,
        "selected": {"resumes": len(resume_data), "jobs": len(job_ids),
                     "source_b_rows": len(selected_b), "distinct_source_requirements": len(requirements),
                     "resume_text_groups": text_groups, "conflicting_resume_ids": conflicting_resumes},
        "duplicates": {"exact_source_b_copies_collapsed": len(selected_b) - len(requirements),
                       "repeated_b_job_skill_keys_preserved": int(selected_b.duplicated(["jd_id", "skill_id"]).sum()),
                       "canonical_repeated_ids": int(canonical.skill_id.duplicated().sum()),
                       "output_record_id_duplicates": int(frame.record_id.duplicated().sum())},
        "generated_rows": len(frame), "labeled_rows": len(labeled), "review_rows": len(review),
        "class_distribution": _distribution(labeled),
        "splits": {s: {"rows": len(d), "resumes": d.resume_id.nunique(),
                        "classes": _distribution(d)} for s, d in splits.items()},
        "split_isolation": isolation, "review_reasons": review.review_reason.value_counts().to_dict(),
        "label_sources": labeled.label_source.value_counts().to_dict(),
        "required_null_counts": frame[["record_id", "resume_id", "job_id", "skill_name",
                                        "resume_text", "label_source", "confidence"]].isna().sum().to_dict(),
        "input_sha256": {str(p): protected[p] for p in (a_path, b_path, c_path, relations_path)},
        "limitations": [
            "Weak labels need human audit; confidence scores are not calibrated probabilities.",
            "Resume text is preprocessed upstream; original casing/punctuation/context may be lost.",
            "Source A extractions are provenance only, never accepted as presence proof.",
            "Only explicit narrower ESCO skills and recognized conjunctions support related labels; broad-parent evidence is reviewed and semantic paraphrases can be missed.",
            "Mined names/aliases and source job requirements can still be semantically ambiguous or wrong.",
            "Context rules are conservative heuristics, not full linguistic scope resolution.",
            "Resume-job pairs are constructed, not observed applications; jobs are shared across splits.",
            "Resume IDs and identical normalized texts are isolated; near duplicates are not exhaustively detected.",
        ],
    }
    original_c_path = processed_dir / "dataset_C_skill_gap_matching.parquet"
    if original_c_path.exists():
        original_c = pd.read_parquet(original_c_path)
        collided = original_c[original_c.record_id.duplicated(keep=False)]
        report["original_duplicate_investigation"] = {
            "duplicate_record_id_excess": int(original_c.record_id.duplicated().sum()),
            "colliding_ids": int(collided.record_id.nunique()),
            "explanation": "Old IDs used role title plus skill ID, omitting source job ID and requirement attributes.",
            "resolution": "Use resume ID plus full source requirement hash; retain actual jd_id and source-row provenance.",
            "examples": collided[["record_id", "skill_name", "preferred_skills", "skill_importance"]].head(8).to_dict("records"),
        }
    for s, d in splits.items():
        absent = [name for name, count in _distribution(d).items() if count == 0]
        if absent:
            report["limitations"].append(f"{s} has no {', '.join(absent)} examples; do not invent labels to fill it.")
    if any(_file_hash(p) != digest for p, digest in protected.items()):
        raise RuntimeError("A protected original input changed during the build")
    report["protected_original_files_verified"] = len(protected)
    outputs = {"dataset_C_skill_gap_matching_improved.parquet": labeled,
               "dataset_C_review_improved.parquet": review,
               **{f"dataset_C_split_{s}_improved.parquet": d for s, d in splits.items()}}
    for name, data in outputs.items():
        data.to_parquet(processed_dir / name, index=False)
        pd.testing.assert_frame_equal(data.reset_index(drop=True), pd.read_parquet(processed_dir / name))
    if any(_file_hash(p) != digest for p, digest in protected.items()):
        raise RuntimeError("A protected original file changed while saving outputs")
    report["saved_output_roundtrip_verified"] = True
    report["outputs"] = list(outputs) + ["dataset_C_validation_improved.json"]
    report_path = processed_dir / "dataset_C_validation_improved.json"
    report_path.write_text(json.dumps(report, indent=2, default=int) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["generated_rows", "labeled_rows", "review_rows",
                                            "class_distribution", "splits", "duplicates",
                                            "split_isolation", "review_reasons", "label_sources"]},
                     indent=2, default=int))
    print(f"SUCCESS: separate improved outputs verified; report: {report_path}")
    return labeled, review, report


class BuilderPolicyTests(unittest.TestCase):
    """Embedded to keep this task's code changes within the authorized file."""

    @classmethod
    def setUpClass(cls):
        rows = [
            ("python", "Python", ["Python programming"]), ("java", "Java", []),
            ("k8s", "Kubernetes", ["K8s"]), ("c", "C", []), ("cpp", "C++", []),
            ("csharp", "C#", []), ("sql", "SQL", []), ("go", "Go", []),
            ("pm", "project management", []), ("sales", "sales", []),
            ("neg", "negotiation", []), ("both", "sales and negotiation", []),
            ("child", "Python testing", []), ("other", "unrelated skill", ["shared alias"]),
            ("amb", "another skill", ["shared alias"]),
            ("training", "training", []), ("td", "training and development", []),
            ("software", "software development", []), ("js", "JavaScript", []),
        ]
        canonical = pd.DataFrame([{"skill_id": sid, "canonical_name": name,
                                   "aliases": aliases, "esco_id": "uri/" + sid}
                                  for sid, name, aliases in rows])
        relations = pd.DataFrame([
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": "uri/child",
             "broaderType": "KnowledgeSkillCompetence", "broaderUri": "uri/python"},
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": "uri/java",
             "broaderType": "KnowledgeSkillCompetence", "broaderUri": "uri/software"}])
        cls.index = EvidenceIndex(canonical, relations)

    def classify(self, text, name, sid):
        return _classify_record(ResumeEvidence(text + " Experienced professional with a detailed work history."),
                                name, sid, self.index)

    def test_exact_alias_normalization_and_source_offsets(self):
        for text, name, sid in [("PYTHON", "Python", "python"), ("K8s", "Kubernetes", "k8s"),
                                ("project-management", "project management", "pm"),
                                ("Ｐｙｔｈｏｎ", "Python", "python"),
                                ("Python\n programming", "Python", "python"), ("Python.", "Python", "python")]:
            d = self.classify(text, name, sid)
            self.assertEqual(d["skill_status"], 2)
            full = text + " Experienced professional with a detailed work history."
            self.assertEqual(full[d["evidence_start"]:d["evidence_end"]], d["matched_text"])

    def test_no_substring_or_arbitrary_overlap_positives(self):
        for text, name, sid in [("JavaScript", "Java", "java"), ("NoSQL", "SQL", "sql"),
                                ("Project work and people management", "project management", "pm"),
                                ("Business sales experience", "Python", "python")]:
            self.assertEqual(self.classify(text, name, sid)["skill_status"], 0)

    def test_symbol_names_remain_distinct(self):
        self.assertEqual(self.classify("C++ programming", "C", "c")["skill_status"], 0)
        self.assertEqual(self.classify("C# programming", "C++", "cpp")["skill_status"], 0)
        self.assertEqual(self.classify("C++ programming", "C++", "cpp")["skill_status"], 2)
        self.assertIsNone(self.classify("C programming", "C++", "cpp")["skill_status"])

    def test_ambiguous_negated_and_aspirational_are_unlabeled(self):
        for text, name, sid in [("We go to meetings", "Go", "go"),
                                ("Software staff go to meetings", "Go", "go"),
                                ("No experience with Python", "Python", "python"),
                                ("Want to learn Python", "Python", "python")]:
            d = self.classify(text, name, sid)
            self.assertIsNone(d["skill_status"])
            self.assertTrue(d["needs_review"])

    def test_partial_requires_structured_evidence(self):
        for text, name, sid, source in [
            ("Basic knowledge of Java", "Java", "java", "explicit_limited_evidence"),
            ("Python", "Python testing", "child", "review_broad_related_evidence"),
            ("Java programming", "software development", "software", "esco_direct_related_skill"),
            ("Sales", "sales and negotiation", "both", "incomplete_conjunct_evidence")]:
            d = self.classify(text, name, sid)
            self.assertEqual(d["skill_status"], None if source.startswith("review_") else 1)
            self.assertEqual(d["label_source"], source)

    def test_ambiguous_alias_not_accepted(self):
        self.assertIsNone(self.classify("Shared alias", "unrelated skill", "other")["skill_status"])

    def test_generic_and_complete_conjuncts_need_review(self):
        self.assertIsNone(self.classify("Education training", "training and development", "td")["skill_status"])
        self.assertIsNone(self.classify("Sales experience. Also negotiation.", "sales and negotiation", "both")["skill_status"])

    def test_company_limited_is_not_a_proficiency_level(self):
        self.assertEqual(self.classify("Company Limited Java", "Java", "java")["skill_status"], 2)

    def test_variant_is_explicit_and_not_fuzzy(self):
        self.assertEqual(self.classify("JS programming", "JavaScript", "js")["label_source"], "normalized_variant")
        self.assertEqual(self.classify("Javva", "Java", "java")["skill_status"], 0)
        self.assertEqual(self.classify("Go programming", "Go", "go")["skill_status"], 2)

    def validation_fixture(self):
        rows = []
        for i, split in enumerate(["train", "val", "test"]):
            text = f"Python programming professional number {i} with detailed experience in systems and software."
            rows.append({"record_id": str(i), "resume_id": str(i), "job_id": "job",
                         "resume_text": text, "skill_id": "python", "skill_name": "Python",
                         "required_skills_raw": "Python", "skill_importance": .8,
                         "source_requirement_id": "req", "source_requirement_json": "{}",
                         "source_resume_rows": "[0]", "split": split,
                         **_classify_record(ResumeEvidence(text), "Python", "python", self.index)})
        frame = pd.DataFrame(rows)
        frame["skill_status"] = frame.skill_status.astype("Int64")
        return frame

    def test_validator_accepts_valid_fixture(self):
        frame = self.validation_fixture()
        splits = {s: frame[frame.split == s].copy() for s in ["train", "val", "test"]}
        self.assertEqual(len(validate_outputs(frame, splits)), 3)

    def test_validator_rejects_duplicates_leakage_and_false_evidence(self):
        for corruption in ["id", "resume", "evidence", "source", "null"]:
            with self.subTest(corruption=corruption):
                frame = self.validation_fixture()
                if corruption == "id":
                    frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
                elif corruption == "resume":
                    frame.loc[1, "resume_id"] = frame.loc[0, "resume_id"]
                elif corruption == "evidence":
                    frame.loc[0, "matched_text"] = "fabricated"
                elif corruption == "source":
                    frame.loc[0, "label_source"] = "no_meaningful_evidence"
                else:
                    frame.loc[0, "skill_status"] = pd.NA
                splits = {s: frame[frame.split == s].copy() for s in ["train", "val", "test"]}
                with self.assertRaises(ValueError):
                    validate_outputs(frame, splits)

    def test_invalid_input_not_a_missing_label(self):
        self.assertIsNone(_classify_record(ResumeEvidence(""), "Python", "python", self.index)["skill_status"])
        self.assertIsNone(self.classify("Python", "Java", "python")["skill_status"])

    def test_split_determinism_and_duplicate_text_grouping(self):
        texts = {str(i): f"Resume number {i} with unique content" for i in range(20)}
        texts["copy"] = texts["0"]
        assignment, _ = _split_resumes(texts, 42)
        self.assertEqual(assignment, _split_resumes(texts, 42)[0])
        self.assertEqual(assignment["copy"], assignment["0"])
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_source_requirement_keys_keep_legitimate_variants(self):
        row = {"jd_id": "one", "skill_id": "python", "importance": .7}
        self.assertEqual(_digest(row), _digest(dict(row)))
        self.assertNotEqual(_digest(row), _digest(dict(row, importance=.8)))
        self.assertNotEqual(_digest(row), _digest(dict(row, jd_id="two")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--relations-path", type=Path, default=RELATIONS_PATH)
    parser.add_argument("--max-resumes", type=int, default=200)
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-improved", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.max_resumes < 0 or args.max_jobs < 0:
        parser.error("Limits must be nonnegative; 0 means all")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BuilderPolicyTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    build_dataset_c(args.processed_dir, args.relations_path, args.max_resumes,
                    args.max_jobs, args.seed, args.overwrite_improved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
