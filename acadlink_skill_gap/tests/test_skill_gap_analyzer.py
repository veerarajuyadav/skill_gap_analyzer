import unittest
from unittest.mock import patch

import pandas as pd

from skill_gap_analyzer import SkillGapAnalyzer


class SkillGapAnalyzerClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = pd.DataFrame([
            {
                "skill_id": "python",
                "canonical_name": "Python",
                "aliases": ["Python programming"],
                "skill_type": "technology",
                "source": "ESCO",
                "esco_id": "python",
                "onet_id": None,
                "parent_skill": "Programming",
            },
            {
                "skill_id": "kubernetes",
                "canonical_name": "Kubernetes",
                "aliases": ["K8s"],
                "skill_type": "technology",
                "source": "ESCO",
                "esco_id": "kubernetes",
                "onet_id": None,
                "parent_skill": "Cloud",
            },
            {
                "skill_id": "data-analysis",
                "canonical_name": "Data analysis",
                "aliases": ["Data analytics"],
                "skill_type": "skill_competence",
                "source": "ESCO",
                "esco_id": "data-analysis",
                "onet_id": None,
                "parent_skill": "Analytics",
            },
            {
                "skill_id": "data-science",
                "canonical_name": "Data science",
                "aliases": [],
                "skill_type": "skill_competence",
                "source": "ESCO",
                "esco_id": "data-science",
                "onet_id": None,
                "parent_skill": "Analytics",
            },
            {
                "skill_id": "java",
                "canonical_name": "Java",
                "aliases": [],
                "skill_type": "technology",
                "source": "ESCO",
                "esco_id": "java",
                "onet_id": None,
                "parent_skill": "Programming",
            },
        ])

    def make_analyzer(self, required_skills):
        with patch("skill_gap_analyzer.build_canonical_skills", return_value=self.canonical):
            analyzer = SkillGapAnalyzer(use_embedding=False)
        analyzer.get_required_skills_for_role = lambda role, company: {
            "required_skills": required_skills,
            "skill_details": [
                {"skill": skill, "importance": 0.8, "requirement_type": "required", "source": "test"}
                for skill in required_skills
            ],
            "role": role,
            "company": company,
        }
        return analyzer

    def statuses(self, result):
        return {item["skill"].casefold(): item["status"] for item in result["skills"]}

    def test_exact_skill_with_identical_and_different_casing(self):
        analyzer = self.make_analyzer(["Python"])
        gap = analyzer.analyze_gap("Experienced in PYTHON development.", "Engineer")
        self.assertEqual(gap["present_skills"], ["Python"])
        self.assertEqual(gap["missing_skills"], [])

        analyzer = self.make_analyzer(["pYtHoN"])
        gap = analyzer.analyze_gap("Experienced in Python development.", "Engineer")
        self.assertEqual(gap["present_skills"], ["pYtHoN"])

    def test_normalizes_whitespace_and_deduplicates_required_and_extracted_skills(self):
        analyzer = self.make_analyzer([" PYTHON ", "python", "  Python   "])
        analyzer.extract_skills_from_resume = lambda text: {
            "extracted_skills": [
                {"skill": "Python", "skill_id": "python"},
                {"skill": "  python ", "skill_id": "python"},
            ],
            "all_canonical_skills": {"Python"},
        }
        gap = analyzer.analyze_gap("Python experience listed in the resume.", "Engineer")
        self.assertEqual(gap["required_skills"], ["PYTHON"])
        self.assertEqual(gap["extracted_skills"], ["Python"])
        self.assertEqual(gap["present_skills"], ["PYTHON"])
        self.assertEqual(gap["gap_summary"]["total_required"], 1)

    def test_exact_recognized_alias_and_exact_phrase_are_present(self):
        analyzer = self.make_analyzer(["K8s", "Python programming"])
        gap = analyzer.analyze_gap(
            "Hands-on experience with Kubernetes and Python programming.", "Engineer"
        )
        self.assertEqual(gap["present_skills"], ["K8s", "Python programming"])
        self.assertEqual(gap["missing_skills"], [])

    def test_high_confidence_fuzzy_match_is_partial_not_present(self):
        analyzer = self.make_analyzer(["Data analysi"])
        gap = analyzer.analyze_gap("Experience with Data analysis.", "Analyst")
        result = analyzer.predict("Experience with Data analysis.", "Analyst")
        self.assertEqual(gap["present_skills"], [])
        self.assertEqual(gap["missing_skills"], ["Data analysi"])
        self.assertEqual(self.statuses(result)["data analysi"], "partial")

    def test_low_confidence_fuzzy_match_is_missing(self):
        analyzer = self.make_analyzer(["Data zxq"])
        result = analyzer.predict("Experience with Data analysis.", "Analyst")
        self.assertEqual(self.statuses(result)["data zxq"], "missing")

    def test_related_skill_is_partial_and_unmentioned_skill_is_missing(self):
        analyzer = self.make_analyzer(["Data analysis", "Java"])
        result = analyzer.predict("Experience with Data science.", "Analyst")
        self.assertEqual(self.statuses(result)["data analysis"], "partial")
        self.assertEqual(self.statuses(result)["java"], "missing")

    def test_embeddings_disabled_and_results_are_consistent(self):
        analyzer = self.make_analyzer(["Python", "Java"])
        resume = "Experienced in Python development."
        gap = analyzer.analyze_gap(resume, "Engineer")
        result = analyzer.predict(resume, "Engineer")

        self.assertIsNone(analyzer.matcher.embed_cache)
        statuses = self.statuses(result)
        self.assertEqual(statuses["python"], "present")
        self.assertEqual(statuses["java"], "missing")
        self.assertEqual(set(gap["present_skills"]), {
            skill for skill in gap["required_skills"] if statuses[skill.casefold()] == "present"
        })
        self.assertFalse(set(gap["present_skills"]) & set(gap["missing_skills"]))

    def test_coverage_and_overall_match_score_use_exact_present_count(self):
        analyzer = self.make_analyzer(["Python", "Kubernetes", "Java"])
        resume = "Experienced in Python and Kubernetes."
        gap = analyzer.analyze_gap(resume, "Engineer")
        result = analyzer.predict(resume, "Engineer")

        self.assertEqual(gap["gap_summary"]["total_required"], 3)
        self.assertEqual(gap["gap_summary"]["present_in_resume"], 2)
        self.assertEqual(gap["gap_summary"]["coverage_percentage"], 66.67)
        self.assertEqual(result["overall_match_score"], 0.667)



class SkillGapAnalyzerExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = SkillGapAnalyzer(use_embedding=False)

    def test_extracts_new_ontology_skills(self):
        result = self.analyzer.extract_skills_from_resume(
            "Experienced with Apache Spark and Scala programming."
        )
        names = {item["skill"].casefold() for item in result["extracted_skills"]}
        self.assertIn("apache spark", names)
        self.assertIn("scala", names)

    def test_false_positive_terms_are_not_extracted_as_unrelated_skills(self):
        result = self.analyzer.extract_skills_from_resume(
            "I love cooking Italian food and playing guitar."
        )
        names = {item["skill"].casefold() for item in result["extracted_skills"]}
        self.assertNotIn("guitar", names)
        self.assertNotIn("cooking", names)

    def test_duplicate_skill_forms_are_deduplicated(self):
        result = self.analyzer.extract_skills_from_resume(
            "Python python PYTHON and machine learning machine learning."
        )
        names = [item["skill"].casefold() for item in result["extracted_skills"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("python", names)
        self.assertIn("machine learning", names)

    def test_duplicate_skill_forms_are_deduplicated(self):
        result = self.analyzer.extract_skills_from_resume(
            "Python python PYTHON and machine learning machine learning."
        )
        names = [item["skill"].casefold() for item in result["extracted_skills"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("python", names)
        self.assertIn("machine learning", names)

    def test_sliding_token_window_deduplication(self):
        # Test that repeated sliding token windows are deduplicated before matching
        # Resume with repeated phrases that would generate overlapping token windows
        result = self.analyzer.extract_skills_from_resume(
            "Python programming Python development"
        )
        names = {item["skill"].casefold() for item in result["extracted_skills"]}
        # "Python" should be extracted (possibly once despite repeated occurrences)
        self.assertIn("python", names)
        # Result should be deterministic - no duplicate extractions from sliding windows
        self.assertEqual(len(names), len([item for item in result["extracted_skills"] if True]))

    def test_alias_skill_is_extracted(self):
        canonical = pd.DataFrame([
            {
                "skill_id": "kubernetes",
                "canonical_name": "Kubernetes",
                "aliases": ["K8s"],
                "skill_type": "technology",
                "source": "ESCO",
                "esco_id": "kubernetes",
                "onet_id": None,
                "parent_skill": "Cloud",
            },
        ])
        with patch("skill_gap_analyzer.build_canonical_skills", return_value=canonical):
            analyzer = SkillGapAnalyzer(use_embedding=False)
        result = analyzer.extract_skills_from_resume("Hands-on with K8s.")
        names = {item["skill"].casefold() for item in result["extracted_skills"]}
        self.assertIn("kubernetes", names)


if __name__ == "__main__":
    unittest.main()
