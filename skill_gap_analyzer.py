"""Skill Gap Analyzer for AcadLink AI.

Given a resume and a target role/company, this analyzer:
1. Extracts skills from the resume using the canonical skill ontology
2. Retrieves required skills for the target role from dataset B / occupation mapping
3. Computes the skill gap (missing vs. present)
4. Ranks missing skills by importance/confidence
5. Produces structured JSON output for LLM consumption
"""
from pathlib import Path
import sys
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from rapidfuzz import fuzz

# Add the acadlink_skill_gap source to path
project_root = Path(__file__).resolve().parent
acadlink_src = project_root / "acadlink_skill_gap" / "src"
sys.path.insert(0, str(acadlink_src))

from acadlink_skill_gap.src.paths import PROCESSED_DIR, DATA_DIR
from acadlink_skill_gap.src.ontology.skills import build_canonical_skills, SkillMatcher
from acadlink_skill_gap.src.preprocessing.text_norm import normalize_text


def detect_resume_format(resume_path: str) -> str:
    """Detect resume format from file extension.

    Returns 'pdf', 'docx', or 'text'.
    """
    if not resume_path:
        return 'text'
    ext = resume_path.lower().strip()
    if ext.endswith('.pdf'):
        return 'pdf'
    elif ext.endswith('.docx'):
        return 'docx'
    else:
        return 'text'


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfminer.six.

    Uses only free, open-source processing. No paid APIs.
    Returns empty string on failure.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        return text if text else ""
    except Exception:
        return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx.

    Uses only free, open-source processing. No paid APIs.
    Returns empty string on failure.
    """
    try:
        from docx import Document
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text else ""
    except Exception:
        return ""


class SkillGapAnalyzer:
    """Analyzes skill gaps between a resume and a target role."""

    def __init__(self, use_embedding: bool = True):
        self.canonical = build_canonical_skills()
        self.matcher = SkillMatcher(self.canonical, use_embedding=use_embedding)
        self._skill_aliases_by_id = {}
        for record in self.canonical.itertuples(index=False):
            aliases = self._alias_values(getattr(record, "aliases", None))
            aliases.insert(0, str(record.canonical_name))
            unique_aliases = {}
            for alias in aliases:
                normalized = self._normalize_skill_name(alias)
                if normalized:
                    unique_aliases.setdefault(normalized, alias.strip())
            self._skill_aliases_by_id[str(record.skill_id)] = list(unique_aliases.values())

    @staticmethod
    def _normalize_skill_name(skill: str) -> str:
        """Normalize display names for comparison without changing output labels."""
        return " ".join(skill.split()).casefold() if isinstance(skill, str) else ""

    @staticmethod
    def _alias_values(value) -> list[str]:
        if isinstance(value, str):
            aliases = value.splitlines()
        elif isinstance(value, np.ndarray):
            aliases = value.tolist()
        elif isinstance(value, (list, tuple, set)):
            aliases = value
        else:
            aliases = []
        return [alias.strip() for alias in aliases
                if isinstance(alias, str) and alias.strip()]

    @staticmethod
    def _text_contains_phrase(text: str, phrase: str) -> bool:
        if not isinstance(text, str) or not isinstance(phrase, str) or not phrase.strip():
            return False
        words = phrase.strip().split()
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def extract_skills_from_resume(self, resume_text: str) -> dict:
        """Extract canonical skills from resume text.

        Returns dict with:
        - extracted_skills: list of {skill, skill_id, skill_type, source, confidence}
        - all_canonical_skills: set of canonical skill names found
        """
        if not resume_text or len(resume_text.strip()) < 10:
            return {"extracted_skills": [], "all_canonical_skills": set()}

        normalized = normalize_text(resume_text, expand_abbrev=False)
        extracted = {}

        # Deduplicate repeated sliding token windows before calling the matcher
        # to avoid redundant matching while preserving all evidence
        tokens = normalized.split()

        # Collect all candidate phrases from sliding windows, deduplicated
        # Section A: 2-3 token phrases (lines 121-127 original)
        phrases_to_match: set[tuple[str, int]] = set()  # (phrase, length)
        for i in range(len(tokens)):
            for length in [2, 3]:
                if i + length <= len(tokens):
                    phrase = " ".join(tokens[i:i + length])
                    if len(phrase) >= 4:
                        phrases_to_match.add((phrase, length))

        # Section B: 1-4 token phrases (lines 159-167 original)
        for i in range(len(tokens)):
            for length in [1, 2, 3, 4]:
                if i + length <= len(tokens):
                    phrase = " ".join(tokens[i:i + length])
                    if len(phrase) < 3:
                        continue
                    phrases_to_match.add((phrase, length))

        # Deduplicate: group by phrase text, keep best length priority
        # Prefer longer phrases for the same text (more specific)
        phrase_best_length: dict[str, int] = {}
        for phrase, length in phrases_to_match:
            if phrase not in phrase_best_length or length > phrase_best_length[phrase]:
                phrase_best_length[phrase] = length

        # Deduplicate: keep only the best (longest) version of each unique phrase
        unique_phrases = [phrase for phrase, _ in
                          sorted([(p, phrase_best_length[p]) for p in phrase_best_length],
                                 key=lambda x: -x[1])]

        # Match each unique phrase against the ontology using deep=False
        # (alias-exact + token-containment only, fast path)
        matched_phrases: set[str] = set()
        for phrase in unique_phrases:
            match = self.matcher.match(phrase, required=False, deep=False)
            if match and match["confidence"] >= 0.80:
                # Normalize the matched canonical name
                canonical_name = match["canonical_name"].casefold()
                if canonical_name not in matched_phrases:
                    matched_phrases.add(canonical_name)
                    if canonical_name not in extracted:
                        extracted[canonical_name] = {
                            "skill": match["canonical_name"],
                            "skill_id": match["skill_id"],
                            "skill_type": match["skill_type"],
                            "source": match["source"],
                            "confidence": match["confidence"],
                        }

        # Build all_canonical_skills set
        all_canonical_skills = set(extracted.keys())

        return {
            "extracted_skills": list(extracted.values()),
            "all_canonical_skills": all_canonical_skills,
        }

        # Check common tech skills
        common_tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'sql',
            'r', 'matlab', 'tensorflow', 'pytorch', 'scikit-learn', 'opencv',
            'tableau', 'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'linux', 'unix', 'agile', 'scrum', 'react', 'angular', 'vue',
            'node.js', 'html', 'css', 'webpack', 'git', 'jira', 'junit',
            'machine learning', 'deep learning', 'data analysis', 'data science',
            'project management', 'communication', 'problem solving',
        ]

        extracted_skills_list = []
        for skill in common_tech_skills:
            pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
            if re.search(pattern, normalized, re.IGNORECASE):
                # Match to canonical
                match = self.matcher.match(skill, required=False, deep=False)
                if match:
                    extracted_skills_list.append({
                        "skill": match["canonical_name"],
                        "skill_id": match["skill_id"],
                        "skill_type": match["skill_type"],
                        "source": match["source"],
                        "confidence": match["confidence"],
                        "method": match["method"],
                        "matched_text": skill,
                    })
                    normalized_name = self._normalize_skill_name(match["canonical_name"])
                    extracted.setdefault(normalized_name, match["canonical_name"])
                else:
                    extracted.setdefault(self._normalize_skill_name(skill), skill)

        # Also try matching longer phrases from the resume using the matcher
        # Tokenize and try multi-word phrases
        tokens = normalize_text(resume_text, expand_abbrev=True).split()
        # Check for 2-3 token phrases that might be skills
        for i in range(len(tokens)):
            for length in [2, 3]:
                if i + length <= len(tokens):
                    phrase = " ".join(tokens[i:i+length])
                    if len(phrase) >= 4:
                        match = self.matcher.match(phrase, required=False, deep=False)
                        if match and match["confidence"] >= 0.80:
                            # Avoid false positives: only accept high-confidence
                            # alias/containment matches, and require the matched
                            # skill to be fully contained in the phrase (no
                            # partial skill-name fragments).
                            if match["method"] not in {"alias_exact", "token_containment"}:
                                continue
                            if match["method"] == "token_containment":
                                raw_tokens = phrase.split()
                                canon_tokens = self._normalize_skill_name(match["canonical_name"]).split()
                                if len(raw_tokens) < len(canon_tokens) and len(raw_tokens) < 3:
                                    continue
                            normalized_name = self._normalize_skill_name(match["canonical_name"])
                            if normalized_name not in extracted:
                                extracted_skills_list.append({
                                    "skill": match["canonical_name"],
                                    "skill_id": match["skill_id"],
                                    "skill_type": match["skill_type"],
                                    "source": match["source"],
                                    "confidence": match["confidence"],
                                    "method": match["method"],
                                    "matched_text": phrase,
                                })
                                extracted[normalized_name] = match["canonical_name"]

        # Phase 3: Ontology-based matching for broader skill coverage.
        # Try matching each token and short phrase against the full canonical
        # ontology (lengths 1-4).  Confidence >= 0.80 matches existing
        # threshold; only alias/containment matches are accepted to avoid
        # fuzzy false positives, and containment matches require the matched
        # skill to be fully represented in the phrase.
        tokens = normalize_text(resume_text, expand_abbrev=True).split()
        for i in range(len(tokens)):
            for length in [1, 2, 3, 4]:
                if i + length <= len(tokens):
                    phrase = " ".join(tokens[i:i + length])
                    if len(phrase) < 3:
                        continue
                    match = self.matcher.match(phrase, required=False, deep=False)
                    if match and match["confidence"] >= 0.80:
                        if match["method"] not in {"alias_exact", "token_containment"}:
                            continue
                        if match["method"] == "token_containment":
                            raw_tokens = phrase.split()
                            canon_tokens = self._normalize_skill_name(match["canonical_name"]).split()
                            if len(raw_tokens) < len(canon_tokens) and len(raw_tokens) < 3:
                                continue
                        normalized_name = self._normalize_skill_name(match["canonical_name"])
                        if normalized_name not in extracted:
                            extracted_skills_list.append({
                                "skill": match["canonical_name"],
                                "skill_id": match["skill_id"],
                                "skill_type": match["skill_type"],
                                "source": match["source"],
                                "confidence": match["confidence"],
                                "method": match["method"],
                                "matched_text": phrase,
                            })
                            extracted.setdefault(normalized_name, match["canonical_name"])

        # Deduplicate by normalized display name and canonical identifier.
        seen_ids = set()
        seen_names = set()
        deduped = []
        for s in extracted_skills_list:
            normalized_name = self._normalize_skill_name(s["skill"])
            if s["skill_id"] not in seen_ids and normalized_name not in seen_names:
                seen_ids.add(s["skill_id"])
                seen_names.add(normalized_name)
                deduped.append(s)

        # Remove single-token skills that are merely components of another
        # extracted multi-token skill (e.g. "Science" inside "data science").
        filtered = []
        for s in deduped:
            s_tokens = set(self._normalize_skill_name(s["skill"]).split())
            if len(s_tokens) == 1 and any(
                s_tokens < set(self._normalize_skill_name(o["skill"]).split())
                for o in deduped if o is not s
            ):
                continue
            filtered.append(s)
        deduped = filtered

        return {
            "extracted_skills": deduped,
            "all_canonical_skills": {s["skill"] for s in deduped},
        }

    
    
    def get_required_skills_for_role(self, role_name: str, company: str = "") -> dict:
        """Retrieve role requirements using exact titles and explicit role aliases."""

        dataset_b_path = PROCESSED_DIR / "dataset_B.parquet"

        result = {
            "required_skills": [],
            "skill_details": [],
            "role": role_name,
            "matched_role": None,
            "match_confidence": 0.0,
            "matching_method": "no_match",
            "company": company,
        }

        if not dataset_b_path.exists():
            return result

        db = pd.read_parquet(dataset_b_path)

        if db.empty or "jd_entity" not in db.columns:
            return result

        def normalize_title(value):
            return " ".join(str(value).casefold().strip().split())

        requested = normalize_title(role_name)

        # Explicit aliases: only use occupations supported by the dataset.
        titles = (
            db["jd_entity"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        title_lookup = {
            normalize_title(title): title
            for title in titles
        }

        # Strict matching: accept only an exact normalized job title.
        if requested in title_lookup:
            matched_role = title_lookup[requested]
            matching_method = "exact"
            confidence = 100.0
        else:
            return result

        role_rows = db[
            db["jd_entity"].astype(str).str.casefold()
            == matched_role.casefold()
        ]

        unique_details = {}

        for _, row in role_rows.iterrows():
            skill = row.get("skill_name")

            if not isinstance(skill, str) or not skill.strip():
                continue

            importance = row.get("importance", 0.8)
            importance = 0.8 if pd.isna(importance) else float(importance)

            detail = {
                "skill": skill.strip(),
                "skill_id": row.get("skill_id", ""),
                "requirement_type": row.get("requirement", "required"),
                "importance": importance,
                "source": row.get("label_sources", ""),
            }

            key = self._normalize_skill_name(skill)

            current = unique_details.get(key)

            if current is None or importance > current["importance"]:
                unique_details[key] = detail

        skill_details = sorted(
            unique_details.values(),
            key=lambda item: item["importance"],
            reverse=True,
        )[:50]

        result.update({
            "required_skills": [item["skill"] for item in skill_details],
            "skill_details": skill_details,
            "matched_role": matched_role,
            "match_confidence": confidence,
            "matching_method": matching_method,
        })

        return result
    def analyze_gap(self, resume_text: str, target_role: str,
                    company: str = "") -> dict:
        """Analyze resume skills against the requirements for a target role."""
        gap_analysis, _ = self._build_gap_analysis(
            resume_text, target_role, company
        )
        return gap_analysis
    
    
    def _build_gap_analysis(self, resume_text: str, target_role: str,
                            company: str = "") -> tuple[dict, dict]:
        # 1. Extract skills from resume
        resume_analysis = self.extract_skills_from_resume(resume_text)
        extracted_by_key = {}
        for item in resume_analysis["extracted_skills"]:
            skill_name = item.get("skill", "")
            normalized = self._normalize_skill_name(skill_name)
            if normalized:
                extracted_by_key.setdefault(normalized, skill_name)

        # 2. Get required skills for target role
        role_analysis = self.get_required_skills_for_role(target_role, company)
        required_by_key = {}
        for skill_name in role_analysis["required_skills"]:
            normalized = self._normalize_skill_name(skill_name)
            if normalized:
                required_by_key.setdefault(normalized, skill_name.strip())

        detail_by_key = {}
        for detail in role_analysis["skill_details"]:
            normalized = self._normalize_skill_name(detail.get("skill", ""))
            if normalized and normalized not in detail_by_key:
                detail_by_key[normalized] = detail

        # One classification pass is shared by analyze_gap() and predict().
        classifications = {}
        present_skills = []
        missing_with_detail = []
        for normalized, display_name in required_by_key.items():
            status, confidence, explanation = self._classify_skill_two_step(
                display_name,
                resume_text,
                display_name,
                self.matcher,
                resume_skills=resume_analysis["extracted_skills"],
            )
            classifications[normalized] = {
                "status": status,
                "confidence": confidence,
                "explanation": explanation,
            }
            if status == "present":
                present_skills.append(display_name)
                continue

            detail = detail_by_key.get(normalized)
            imp = detail["importance"] if detail else 0.5
            missing_with_detail.append({
                "skill": display_name,
                "importance": imp,
                "requirement_type": detail.get("requirement_type", "required") if detail else "required",
                "source": detail.get("source", "") if detail else "",
            })

        # Sort by importance descending, then alphabetically
        missing_with_detail.sort(
            key=lambda x: (-x["importance"], self._normalize_skill_name(x["skill"]))
        )

        # 5. Compute summary
        total_required = len(required_by_key)
        total_extracted = len(present_skills)
        gap_size = len(missing_with_detail)
        coverage = (total_extracted / total_required * 100) if total_required > 0 else 0

        gap_analysis = {
            "extracted_skills": sorted(extracted_by_key.values(), key=str.casefold),
            "required_skills": sorted(required_by_key.values(), key=str.casefold),
            "missing_skills": [m["skill"] for m in missing_with_detail],
            "present_skills": sorted(present_skills, key=str.casefold),
            "missing_with_detail": missing_with_detail,
            "gap_summary": {
                "total_required": total_required,
                "present_in_resume": total_extracted,
                "missing": gap_size,
                "coverage_percentage": round(coverage, 2),
                "gap_ratio": round(gap_size / total_required, 3) if total_required > 0 else 0,
            },
            "role": role_analysis["role"],
            "company": role_analysis["company"],
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
        }
        return gap_analysis, classifications

    def _classify_skill_two_step(self, skill: str, resume_text: str,
                                  required_skill: str, matcher: SkillMatcher,
                                  threshold_present: float = 0.70,
                                  threshold_partial: float = 0.45,
                                  resume_skills: list[dict] | None = None) -> tuple[str, float, str]:
        """Classify only from resume evidence; ontology membership is not evidence."""
        target_name = required_skill.strip() or skill.strip()
        match = matcher.match(target_name, required=False)
        method = match.get("method", "") if match else ""
        confidence = float(match.get("confidence", 0.0)) if match else 0.0
        skill_id = str(match.get("skill_id", "")) if match else ""
        aliases = self._skill_aliases_by_id.get(skill_id, [])
        target_evidence = any(
            self._text_contains_phrase(resume_text, alias) for alias in aliases
        )
        recognized_resume_evidence = any(
            resume_skill.get("skill_id") == skill_id
            and resume_skill.get("method") == "alias_exact"
            for resume_skill in resume_skills or []
        )
        direct_evidence = self._text_contains_phrase(resume_text, target_name)
        is_approximate = method in {"fuzzy", "fuzzy_low"}

        if (direct_evidence and not is_approximate) or (
            method == "alias_exact"
            and (target_evidence or recognized_resume_evidence)
            and confidence >= threshold_present
        ):
            return "present", 1.0, "Present (exact resume evidence)"

        if is_approximate and confidence >= max(0.86, threshold_partial):
            if direct_evidence or target_evidence:
                return "partial", confidence, "Partial (approximate skill match)"

        target_record = matcher.id2rec.get(skill_id, {}) if skill_id else {}
        target_parent = target_record.get("parent_skill", "").strip().casefold()
        for resume_skill in resume_skills or []:
            resume_method = resume_skill.get("method", "")
            resume_id = str(resume_skill.get("skill_id", ""))
            matched_text = resume_skill.get("matched_text", "")

            if resume_id == skill_id and resume_method in {"fuzzy", "fuzzy_low"}:
                resume_confidence = float(resume_skill.get("confidence", 0.0))
                if (resume_confidence >= max(0.86, threshold_partial)
                        and self._text_contains_phrase(resume_text, matched_text)):
                    return "partial", resume_confidence, "Partial (approximate resume evidence)"

            if not target_parent or resume_id == skill_id:
                continue
            resume_record = matcher.id2rec.get(resume_id, {})
            resume_parent = resume_record.get("parent_skill", "").strip().casefold()
            resume_aliases = self._skill_aliases_by_id.get(resume_id, [])
            resume_has_evidence = any(
                self._text_contains_phrase(resume_text, alias) for alias in resume_aliases
            )
            target_normalized = self._normalize_skill_name(target_name)
            resume_normalized = self._normalize_skill_name(resume_record.get("name", ""))
            similarity = max(
                fuzz.ratio(target_normalized, resume_normalized),
                fuzz.token_set_ratio(target_normalized, resume_normalized),
            ) / 100
            target_tokens = set(target_normalized.split())
            resume_tokens = set(resume_normalized.split())
            token_overlap = (
                len(target_tokens & resume_tokens) / min(len(target_tokens), len(resume_tokens))
                if target_tokens and resume_tokens else 0.0
            )
            if (resume_parent and resume_parent == target_parent and resume_has_evidence
                    and (similarity >= 0.60 or token_overlap >= 0.50)):
                return "partial", max(similarity, token_overlap), "Partial (related skill evidence)"

        if method == "token_containment" and target_evidence:
            return "partial", confidence, "Partial (related ontology match)"
        return "missing", 0.0, "Missing (no meaningful resume evidence)"

    def predict(self, resume_text: str, target_role: str, company: str = "") -> dict:
        """Run skill gap analysis and return structured JSON output.

        Returns dict with keys:
        - target_role: str
        - overall_match_score: float [0,1] - proportion of required skills present
        - skills: list of {skill, status, confidence, importance}
        - priority_gaps: list of skill names (missing, ranked by importance)
        """
        gap_analysis, classifications = self._build_gap_analysis(
            resume_text, target_role, company
        )

        all_skills = {}
        for skill in gap_analysis["required_skills"] + gap_analysis["extracted_skills"]:
            normalized = self._normalize_skill_name(skill)
            if normalized:
                all_skills.setdefault(normalized, skill)

        skills = []
        missing_importance = {
            self._normalize_skill_name(item["skill"]): item["importance"]
            for item in gap_analysis["missing_with_detail"]
        }
        for normalized in sorted(all_skills):
            skill = all_skills[normalized]
            classification = classifications.get(normalized)
            if classification is None:
                status, confidence, explanation = self._classify_skill_two_step(
                    skill, resume_text, skill, self.matcher
                )
            else:
                status = classification["status"]
                confidence = classification["confidence"]
                explanation = classification["explanation"]

            imp = 0.5
            if normalized in missing_importance:
                imp = missing_importance[normalized]
            elif normalized in classifications:
                imp = 0.9

            skills.append({
                "skill": skill,
                "status": status,
                "confidence": round(confidence, 2),
                "importance": round(imp, 2),
                "explanation": explanation,
            })

        # Priority gaps - missing skills sorted by importance
        priority_gaps = [
            item["skill"] for item in gap_analysis["missing_with_detail"][:5]
        ]
        total_required = gap_analysis["gap_summary"]["total_required"]
        present_in_resume = gap_analysis["gap_summary"]["present_in_resume"]
        match_score = round(present_in_resume / total_required, 3) if total_required else 0.0

        return {
            "target_role": target_role,
            "overall_match_score": match_score,
            "skills": skills,
            "priority_gaps": priority_gaps,
        }

    def format_report(self, gap_analysis: dict) -> str:
        """Format the gap analysis as a human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("SKILL GAP ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Target Role: {gap_analysis['role']}")
        lines.append(f"Company: {gap_analysis['company']}")
        lines.append("-" * 70)

        # Summary
        summary = gap_analysis["gap_summary"]
        lines.append("SUMMARY")
        lines.append(f"  Required Skills:    {summary['total_required']}")
        lines.append(f"  Present in Resume:  {summary['present_in_resume']}")
        lines.append(f"  Missing Skills:     {summary['missing']}")
        lines.append(f"  Coverage:           {summary['coverage_percentage']}%")
        lines.append(f"  Gap Ratio:          1:{round(1/summary['gap_ratio'], 2) if summary['gap_ratio'] > 0 else 'N/A'}")
        lines.append("")

        # Present skills (good news)
        lines.append("- PRESENT SKILLS (Found in Your Resume)")
        lines.append("-" * 70)
        for skill in gap_analysis["present_skills"]:
            lines.append(f"  - {skill}")
        lines.append("")

        # Missing skills (the gap)
        lines.append("- MISSING SKILLS (Gap to Fill)")
        lines.append("-" * 70)
        for ms in gap_analysis["missing_with_detail"]:
            if ms["importance"] >= 0.8:
                marker = "HIGH"
            elif ms["importance"] >= 0.6:
                marker = "MEDIUM"
            else:
                marker = "LOW"
            lines.append(f"  [{marker}] {ms['skill']}")
            lines.append(f"      Importance: {ms['importance']:.2f} | Type: {ms['requirement_type']}")
            if ms["source"]:
                lines.append(f"      Source: {ms['source']}")
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        high_priority = [m for m in gap_analysis["missing_with_detail"] if m["importance"] >= 0.8]
        medium_priority = [m for m in gap_analysis["missing_with_detail"] if 0.6 <= m["importance"] < 0.8]
        low_priority = [m for m in gap_analysis["missing_with_detail"] if m["importance"] < 0.6]

        if high_priority:
            lines.append("  HIGH PRIORITY (importance >= 0.8):")
        if medium_priority:
            lines.append("  MEDIUM PRIORITY (importance 0.6-0.8):")
            for ms in medium_priority:
                lines.append(f"    - {ms['skill']} ({(ms['importance']):.2f})")
        if low_priority:
            lines.append("  LOWER PRIORITY (importance < 0.6):")
            for ms in low_priority:
                lines.append(f"    - {ms['skill']} ({(ms['importance']):.2f})")

        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """CLI entry point for skill gap analysis."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="AcadLink AI: Skill Gap Analyzer")
    parser.add_argument("--resume", required=False, default="", help="Resume text to analyze")
    parser.add_argument("--resume-file", required=False, default="", help="Path to resume PDF or DOCX file")
    parser.add_argument("--role", required=True, help="Target job role (e.g., Software Engineer)")
    parser.add_argument("--company", default="", help="Target company (optional)")
    args = parser.parse_args()

    # Make --resume and --resume-file mutually exclusive
    if args.resume and args.resume_file:
        print("Error: --resume and --resume-file are mutually exclusive. Please provide one or the other.")
        return

    # Determine resume text: from file or raw argument
    resume_text = args.resume
    if not resume_text and args.resume_file:
        # Validate file exists
        if not os.path.isfile(args.resume_file):
            print(f"Error: File not found: '{args.resume_file}'")
            return

        # Detect format and extract text
        resume_format = detect_resume_format(args.resume_file)
        if resume_format == "pdf":
            resume_text = extract_text_from_pdf(args.resume_file)
        elif resume_format == "docx":
            resume_text = extract_text_from_docx(args.resume_file)
        else:
            # Treat as plain text file
            try:
                with open(args.resume_file, "r", encoding="utf-8") as f:
                    resume_text = f.read()
            except Exception:
                print(f"Error: Could not read file: '{args.resume_file}'")
                return

    # Handle empty resume (whether from text or extracted from file)
    if not resume_text or len(resume_text.strip()) < 10:
        print("Error: Resume text is too short or empty. Please provide a valid resume with at least 10 characters of content.")
        return

    # Initialize analyzer
    print("Initializing Skill Gap Analyzer...")
    analyzer = SkillGapAnalyzer(use_embedding=False)

    # Run analysis
    print(f"Analyzing resume against target role: {args.role}")
    gap_analysis = analyzer.analyze_gap(args.resume, args.role, args.company)

    # Display report
    print("\n" + analyzer.format_report(gap_analysis))

    # Also save detailed results as structured JSON
    result = analyzer.predict(args.resume, args.role, args.company)
    output_path = Path("skill_gap_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nStructured JSON results saved to: {output_path}")
    print(f"  Overall match score: {result['overall_match_score']}")
    print(f"  Priority gaps: {', '.join(result['priority_gaps'])}")


if __name__ == "__main__":
    main()