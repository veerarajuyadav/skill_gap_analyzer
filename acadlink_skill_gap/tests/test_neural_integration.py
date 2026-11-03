"""Analyzer integration tests; no dataset writes or training.

Run: python -B -m unittest discover -s acadlink_skill_gap/tests -p 'test*.py' -v
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import skill_gap_analyzer as application
from skill_gap_analyzer import SkillGapAnalyzer, DEFAULT_NEURAL_CHECKPOINT
from acadlink_skill_gap.src.matching.model import HybridSkillMatcher


NAMES = ["Python", "Rust", "coaching and mentoring", "coaching", "mentoring", "Kubernetes", "SQL"]


def canonical_fixture():
    return pd.DataFrame([dict(skill_id=name, canonical_name=name, aliases=[],
                              skill_type="technology", source="MINED", esco_id=None,
                              onet_id=None, parent_skill="") for name in NAMES])


def make_analyzer(use_neural=True, extract=False, **kwargs):
    with patch("skill_gap_analyzer.build_canonical_skills", return_value=canonical_fixture()):
        analyzer = SkillGapAnalyzer(use_embedding=False, use_neural=use_neural, **kwargs)
    if not extract:
        analyzer.extract_skills_from_resume = lambda text: {"extracted_skills": [], "all_canonical_skills": set()}
    analyzer.get_required_skills_for_role = lambda role, company: {
        "required_skills": NAMES[:3],
        "skill_details": [{"skill": s, "importance": .8} for s in NAMES[:3]],
        "role": role, "company": company}
    return analyzer


def write_pdf_fixture(path):
    """Small valid PDF fixture without adding a PDF writer dependency."""
    content = b"BT /F1 12 Tf 72 720 Td (Experienced Python developer.) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(data)
    data.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)


class FakeModel(torch.nn.Module):
    """Deterministic output stub for failure/caching contract tests, not data."""
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.encode_calls = []
        self.forward_calls = 0

    def encode_texts(self, texts, kind):
        self.encode_calls.append((kind, list(texts)))
        labels = {"Python": 2, "Rust": 0, "coaching and mentoring": 1}
        return torch.tensor([[float(labels.get(t, 0))] for t in texts])

    def forward(self, resume_embeddings, skill_embeddings, features):
        self.forward_calls += 1
        classes = skill_embeddings[:, 0].long()
        return {"probabilities": torch.nn.functional.one_hot(classes, 3).float()}


class NeuralIntegrationContractTests(unittest.TestCase):
    def test_c_language_boundaries_and_explicit_evidence(self):
        analyzer = make_analyzer(use_neural=False)
        for target in ["C", "C++", "C#"]:
            for mentioned in ["C", "C++", "C#"]:
                with self.subTest(target=target, mentioned=mentioned):
                    status, _, _ = analyzer._classify_skill_two_step(
                        target, f"Experienced in {mentioned} programming.", target, analyzer.matcher)
                    self.assertEqual(status, "present" if target == mentioned else "missing")
        for phrase in ["C++", "C#", "Objective-C++"]:
            self.assertFalse(analyzer._text_contains_phrase(phrase, "C"))
        self.assertTrue(analyzer._text_contains_phrase("C, Python and SQL", "C"))

    def test_kubernetes_alias_works_both_directions_without_cached_alias(self):
        analyzer = make_analyzer(use_neural=False, extract=True)
        # Fixture Kubernetes has no aliases, just like the cached project ontology.
        for target, mentioned in [("K8s", "Kubernetes"), ("Kubernetes", "K8s"), ("k8S", "KUBERNETES")]:
            with self.subTest(target=target, mentioned=mentioned):
                status, _, _ = analyzer._classify_skill_two_step(
                    target, f"Experienced operating {mentioned} clusters.", target, analyzer.matcher)
                self.assertEqual(status, "present")
        names = {item["skill"].casefold() for item in
                 analyzer.extract_skills_from_resume("Experienced operating K8s clusters.")["extracted_skills"]}
        self.assertIn("kubernetes", names)
        jd = analyzer.get_required_skills_from_job_description("Operate K8s clusters.", "Engineer")
        self.assertIn("k8s", [s.casefold() for s in jd["required_skills"]])

    def test_alias_is_bounded_and_requires_ontology_support(self):
        analyzer = make_analyzer(use_neural=False)
        self.assertEqual(analyzer._expand_supported_aliases("K8s-based clusters"), "Kubernetes-based clusters")
        untouched = "K8sOperator myk8s k8s123 Python C C++ C# JavaScript"
        self.assertEqual(analyzer._expand_supported_aliases(untouched), untouched)
        canonical = canonical_fixture()
        with patch("skill_gap_analyzer.build_canonical_skills", return_value=canonical[canonical.canonical_name != "Kubernetes"]):
            unsupported = SkillGapAnalyzer(use_embedding=False)
        self.assertEqual(unsupported._expand_supported_aliases("K8s"), "K8s")

    def test_registered_kubernetes_alias_precedes_ambiguous_fuzzy_lookup(self):
        analyzer = make_analyzer(use_neural=False)
        fuzzy = {"skill_id": "broader-software", "method": "fuzzy", "confidence": .86}
        with patch.object(analyzer.matcher, "match", return_value=fuzzy) as lookup:
            for target, text in [("Kubernetes", "Experience with K8s clusters."),
                                 ("K8s", "Experience with Kubernetes clusters.")]:
                self.assertEqual(analyzer._classify_skill_two_step(
                    target, text, target, analyzer.matcher)[0], "present")
            lookup.assert_not_called()

    def test_neural_alias_uses_canonical_inputs_and_original_display_name(self):
        analyzer, model = make_analyzer(), FakeModel()
        with patch.object(HybridSkillMatcher, "load", return_value=model):
            result = analyzer._neural_predictions("Operating K8s clusters with Python.", ["K8s", "Python"], [.8, .8])
        self.assertEqual(model.encode_calls[0], ("resume", ["Operating Kubernetes clusters with Python."]))
        self.assertEqual(model.encode_calls[1], ("skill", ["Kubernetes", "Python"]))
        self.assertEqual(set(result), {"k8s", "python"})

    def test_rule_default_keeps_original_output_and_does_not_load_model(self):
        analyzer = make_analyzer(use_neural=False)
        with patch.object(HybridSkillMatcher, "load") as loader:
            result = analyzer.predict("Experienced Python developer.", "Engineer")
        loader.assert_not_called()
        self.assertEqual(set(result), {"target_role", "overall_match_score", "skills", "priority_gaps"})
        self.assertEqual(next(s for s in result["skills"] if s["skill"] == "Python")["status"], "present")

    def test_batched_predictions_loaded_once_and_cached_between_calls(self):
        analyzer, model = make_analyzer(), FakeModel()
        with patch.object(HybridSkillMatcher, "load", return_value=model) as loader:
            result = analyzer.predict("Experienced Python developer.", "Engineer")
            gap = analyzer.analyze_gap("Experienced Python developer.", "Engineer")
        loader.assert_called_once()
        self.assertEqual(len(model.encode_calls), 2)
        self.assertEqual(model.forward_calls, 1)
        statuses = {s["skill"]: s["status"] for s in result["skills"]}
        self.assertEqual(statuses, {"Python": "present", "Rust": "missing", "coaching and mentoring": "partial"})
        self.assertEqual(gap["partial_skills"], ["coaching and mentoring"])
        self.assertEqual(gap["present_skills"], ["Python"])
        self.assertEqual(result["overall_match_score"], .333)
        self.assertEqual(result["matching"]["active"], "neural")
        self.assertEqual(result["matching"]["disagreements"][0]["neural_status"], "partial")

    def test_failed_load_falls_back_once_without_retries(self):
        analyzer = make_analyzer(checkpoint_path="missing-checkpoint.pt")
        with patch.object(HybridSkillMatcher, "load", side_effect=FileNotFoundError("missing")) as loader:
            with self.assertLogs("skill_gap_analyzer", level="WARNING") as logs:
                first = analyzer.predict("Experienced Python developer.", "Engineer")
                second = analyzer.predict("Another Python developer.", "Engineer")
        loader.assert_called_once()
        self.assertEqual(len(logs.output), 1)
        self.assertTrue(first["matching"]["fallback"])
        self.assertEqual(second["matching"]["active"], "rule")
        self.assertFalse(any("probabilities" in s for s in first["skills"]))
        self.assertEqual(next(s for s in first["skills"] if s["skill"] == "Python")["status"], "present")

    def test_runtime_failure_discards_neural_batch(self):
        analyzer, model = make_analyzer(), FakeModel()
        with patch.object(HybridSkillMatcher, "load", return_value=model):
            with patch.object(model, "forward", side_effect=RuntimeError("inference failed")):
                with self.assertLogs("skill_gap_analyzer", level="WARNING"):
                    result = analyzer.predict("Experienced Python developer.", "Engineer")
        self.assertEqual(result["matching"]["active"], "rule")
        self.assertTrue(result["matching"]["fallback"])
        self.assertFalse(any("probabilities" in s for s in result["skills"]))

    def test_invalid_probability_output_falls_back(self):
        analyzer, model = make_analyzer(), FakeModel()
        with patch.object(HybridSkillMatcher, "load", return_value=model):
            with patch.object(model, "forward", return_value={"probabilities": torch.full((3, 3), float("nan"))}):
                with self.assertLogs("skill_gap_analyzer", level="WARNING"):
                    result = analyzer.predict("Experienced Python developer.", "Engineer")
        self.assertTrue(result["matching"]["fallback"])

    def test_cpu_auto_selection_when_cuda_unavailable(self):
        analyzer = make_analyzer()
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(HybridSkillMatcher, "load", return_value=FakeModel()) as loader:
                analyzer.predict("Experienced Python developer.", "Engineer")
        self.assertEqual(loader.call_args.kwargs["device"], "cpu")

    def test_explicit_unavailable_cuda_falls_back(self):
        analyzer = make_analyzer(device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertLogs("skill_gap_analyzer", level="WARNING"):
                result = analyzer.predict("Experienced Python developer.", "Engineer")
        self.assertTrue(result["matching"]["fallback"])

    def test_job_description_bypasses_role_lookup(self):
        analyzer = make_analyzer(use_neural=False, extract=True)
        with patch.object(analyzer, "get_required_skills_for_role", side_effect=AssertionError("role lookup used")):
            gap = analyzer.analyze_gap("Experienced Python developer.", "Custom role",
                                       job_description="Python and SQL experience required.")
        self.assertIn("Python", gap["required_skills"])
        self.assertIn("SQL", gap["required_skills"])
        self.assertIn("Python", gap["present_skills"])

    def test_explicit_job_description_preserves_unknown_skill_names(self):
        analyzer = make_analyzer(use_neural=False, extract=True)
        result = analyzer.get_required_skills_from_job_description(
            "Required skills: Python, Rust, coaching and mentoring.", "Custom role")
        self.assertEqual(result["required_skills"], ["Python", "Rust", "coaching and mentoring"])
        self.assertTrue(all(d["source"] == "job_description_explicit_list" for d in result["skill_details"]))


class NeuralCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_NEURAL_CHECKPOINT.is_file():
            raise unittest.SkipTest("Experimental checkpoint is not installed")
        cls.analyzer = make_analyzer(device="cpu")

    def test_actual_cpu_checkpoint_c_and_kubernetes_regressions(self):
        analyzer = self.analyzer
        for text, names, expected in [
            ("Experienced developer using C++ programming.", ["C", "C++"], {"C": "missing", "C++": "present"}),
            ("Experienced developer using C# programming.", ["C", "C#"], {"C": "missing", "C#": "present"}),
            ("Experienced engineer operating Kubernetes clusters.", ["K8s", "Kubernetes"], {"K8s": "present", "Kubernetes": "present"}),
            ("Experienced engineer operating K8s clusters.", ["K8s", "Kubernetes"], {"K8s": "present", "Kubernetes": "present"}),
            ("Experienced engineer operating Docker containers.", ["K8s", "Kubernetes"], {"K8s": "missing", "Kubernetes": "missing"}),
        ]:
            with self.subTest(text=text):
                result = analyzer.predict(text, "Engineer", job_description="Required skills: " + ", ".join(names))
                selected = {s["skill"]: s for s in result["skills"] if s["skill"] in names}
                self.assertEqual({name: s["status"] for name, s in selected.items()}, expected)
                self.assertEqual({name: s["rule_status"] for name, s in selected.items()}, expected)
                self.assertEqual(result["matching"]["disagreements"], [])
                self.assertEqual(result["matching"]["device"], "cpu")

    def test_alias_fixes_survive_real_checkpoint_failure(self):
        analyzer = make_analyzer(checkpoint_path="missing-regression-model.pt", device="cpu")
        with self.assertLogs("skill_gap_analyzer", level="WARNING"):
            result = analyzer.predict("Experienced engineer operating K8s clusters and C++ applications.",
                                      "Engineer", job_description="Required skills: Kubernetes, C, C++")
        statuses = {s["skill"]: s["status"] for s in result["skills"]}
        self.assertEqual(statuses, {"Kubernetes": "present", "C": "missing", "C++": "present"})
        self.assertTrue(result["matching"]["fallback"])

    def test_actual_checkpoint_all_three_statuses_and_reuse(self):
        analyzer = self.analyzer
        result = analyzer.predict("Experienced Python developer building SQL data processing applications.", "Engineer")
        loaded = analyzer._neural_model
        self.assertIsNotNone(loaded)
        self.assertEqual(result["matching"]["device"], "cpu")
        by_name = {s["skill"]: s for s in result["skills"]}
        self.assertEqual(by_name["Python"]["status"], "present")
        self.assertEqual(by_name["Rust"]["status"], "missing")
        partial = analyzer.predict("Experienced professional coaching colleagues in workplace leadership.", "Engineer")
        self.assertIs(analyzer._neural_model, loaded)
        item = next(s for s in partial["skills"] if s["skill"] == "coaching and mentoring")
        self.assertEqual(item["status"], "partial")
        self.assertEqual(item["rule_status"], "missing")
        self.assertAlmostEqual(sum(item["probabilities"].values()), 1., places=5)
        self.assertFalse(partial["matching"]["fallback"])

    def test_real_missing_checkpoint_preserves_rule_output(self):
        analyzer = make_analyzer(checkpoint_path="does-not-exist-neural.pt", device="cpu")
        with self.assertLogs("skill_gap_analyzer", level="WARNING"):
            result = analyzer.predict("Experienced Python developer.", "Engineer")
        self.assertTrue(result["matching"]["fallback"])
        self.assertIn("FileNotFoundError", result["matching"]["error"])

    def test_real_missing_encoder_and_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"format_version": 2, "encoder_directory": "missing-encoder"}, checkpoint)
            analyzer = make_analyzer(checkpoint_path=checkpoint)
            with self.assertLogs("skill_gap_analyzer", level="WARNING"):
                result = analyzer.predict("Experienced Python developer.", "Engineer")
            self.assertIn("Frozen encoder files missing", result["matching"]["error"])
            checkpoint.write_bytes(b"not a torch checkpoint")
            analyzer = make_analyzer(checkpoint_path=checkpoint)
            with self.assertLogs("skill_gap_analyzer", level="WARNING"):
                result = analyzer.predict("Experienced Python developer.", "Engineer")
            self.assertTrue(result["matching"]["fallback"])


class AnalyzerCLITests(unittest.TestCase):
    def run_cli(self, args, expected_resume):
        analyzer = make_analyzer(use_neural=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["skill_gap_analyzer.py", "--role", "Engineer", "--output", str(output)] + args
            with patch("sys.argv", argv), patch.object(application, "SkillGapAnalyzer", return_value=analyzer):
                with patch.object(analyzer, "analyze_gap", wraps=analyzer.analyze_gap) as analyze:
                    with redirect_stdout(io.StringIO()):
                        application.main()
            self.assertEqual(analyze.call_args.args[0], expected_resume)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["overall_match_score"], .333)

    def test_text_argument(self):
        self.run_cli(["--resume", "Experienced Python developer."], "Experienced Python developer.")

    def test_plain_text_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.txt"
            path.write_text("Experienced Python developer.", encoding="utf-8")
            self.run_cli(["--resume-file", str(path)], "Experienced Python developer.")

    def test_docx_resume_file(self):
        from docx import Document
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.docx"
            doc = Document()
            doc.add_paragraph("Experienced Python developer.")
            doc.save(path)
            self.run_cli(["--resume-file", str(path)], "Experienced Python developer.")

    def test_pdf_resume_parser_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pdf"
            path.write_bytes(b"placeholder; parser mocked in this routing test")
            with patch.object(application, "extract_text_from_pdf", return_value="Experienced Python developer.") as extract:
                self.run_cli(["--resume-file", str(path)], "Experienced Python developer.")
            extract.assert_called_once_with(str(path))

    def test_actual_pdf_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pdf"
            write_pdf_fixture(path)
            parsed = application.extract_text_from_pdf(str(path))
            self.assertIn("Experienced Python developer.", parsed)
            self.run_cli(["--resume-file", str(path)], parsed)


if __name__ == "__main__":
    unittest.main()
