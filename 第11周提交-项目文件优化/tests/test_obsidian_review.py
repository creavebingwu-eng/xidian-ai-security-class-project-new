import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from review_obsidian import build_packet, validate_review, recover_quotes, save_review


class ReviewTests(unittest.TestCase):
    def example_review(self, quote="existing claim"):
        return {"summary": "bounded review", "strengths": [], "revision_order": ["R1-M1"],
                "not_assessable": [], "limitations": [], "major_concerns": [{
                    "id": "R1-M1", "blocking": False, "claim_pointer": "C1",
                    "claim_quote": quote, "evidence_pointer": "无法核验", "concern": "scope",
                    "why_it_matters": "scope", "resolution_test": "check source"}], "minor_comments": []}

    def test_unique_markdown_match_preserves_original_response(self):
        original = self.example_review("a claim [S1]")
        result, corrections, unresolved = recover_quotes(original, "a **claim** [S1]")
        self.assertEqual(result["major_concerns"][0]["claim_quote"], "a **claim** [S1]")
        self.assertEqual(original["major_concerns"][0]["claim_quote"], "a claim [S1]")
        self.assertEqual(len(corrections), 1)
        self.assertEqual(unresolved, [])

    def test_missing_composite_and_ambiguous_quotes_are_quarantined(self):
        for quote, draft in (("first last", "first middle last"),
                             ("**claim**", "`claim` and `claim`"),
                             ("a*b", "ab"),
                             ("claim!", "claim?"), ("a b", "a\nb")):
            with self.subTest(quote=quote):
                result, corrections, unresolved = recover_quotes(self.example_review(quote), draft)
                self.assertEqual(result["major_concerns"], [])
                self.assertEqual(corrections, [])
                self.assertEqual(len(unresolved), 1)

    def test_schema_errors_still_fail(self):
        review = self.example_review()
        review["major_concerns"][0]["blocking"] = "false"
        with self.assertRaises(ValueError):
            recover_quotes(review, "existing claim")

    def test_partial_report_and_raw_response_survive_offline_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = root / "draft.md"
            draft.write_text("a **claim** [S1]\nactual sentence", encoding="utf-8")
            _, packet, metadata = build_packet(draft)
            review = self.example_review("a claim [S1]")
            issue = copy.deepcopy(review["major_concerns"][0])
            issue.update(id="R1-M2", claim_quote="invented sentence")
            review["major_concerns"].append(issue)
            metadata.update(model="unused", created_at="test", packet_sha256=hashlib.sha256(packet.encode()).hexdigest())
            raw = json.dumps([{"role": "assistant", "content": json.dumps(review)}])
            (root / "model_response.json").write_text(raw, encoding="utf-8")
            save_review(root, review, metadata, draft.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "needs_attention")
            self.assertIn("R1-M2", (root / "review.md").read_text(encoding="utf-8"))
            fixes = root / "fixes.json"
            fixes.write_text(json.dumps([{"id":"R1-M2", "old":"invented sentence", "new":"actual sentence"}]), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "review_obsidian.py"),
                       "--input", str(draft), "--output-dir", str(root), "--resume-dir", str(root),
                       "--quote-corrections", str(fixes)]
            completed = subprocess.run(command, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            final = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "completed")
            self.assertTrue(final["recovered_without_model_call"])
            self.assertEqual(len(final["quote_corrections"]), 2)
            self.assertEqual(final["unresolved_issues"], [])
            self.assertEqual((root / "model_response.json").read_text(encoding="utf-8"), raw)
            # A changed input cannot inherit the previous review.
            final["status"] = "needs_attention"
            (root / "manifest.json").write_text(json.dumps(final), encoding="utf-8")
            draft.write_text("changed manuscript", encoding="utf-8")
            refused = subprocess.run(command, capture_output=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn(b"Inputs changed", refused.stderr)

    def test_log_extracts_only_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / "draft.md"
            log = Path(directory) / "run.log"
            draft.write_text("已有文献结果", encoding="utf-8")
            log.write_text("unrelated log\n<OBSIDIAN_EVIDENCE_BUNDLE>source excerpt</OBSIDIAN_EVIDENCE_BUNDLE>\nother logs", encoding="utf-8")
            _, packet, meta = build_packet(draft, log)
            self.assertIn("source excerpt", packet)
            self.assertNotIn("unrelated log", packet)
            self.assertNotIn("other logs", packet)
            self.assertEqual(len(meta["manuscript"]["sha256"]), 64)

    def test_invalid_log_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text("partial evidence", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_packet(path, path)

    def test_review_cannot_fabricate_quote_or_block_on_minor(self):
        review = {"summary": "bounded review", "strengths": [], "revision_order": [],
                  "not_assessable": [], "limitations": [], "major_concerns": [],
                  "minor_comments": [{"id": "R1-m1", "blocking": False,
                    "claim_pointer": "C1", "claim_quote": "existing claim",
                    "evidence_pointer": "无法核验", "concern": "qualify",
                    "why_it_matters": "scope", "resolution_test": "narrow claim"}]}
        validate_review(review, "An existing claim in draft")
        bad = copy.deepcopy(review)
        bad["minor_comments"][0]["claim_quote"] = "invented quotation"
        with self.assertRaises(ValueError):
            validate_review(bad, "An existing claim in draft")
        review["minor_comments"][0]["blocking"] = True
        with self.assertRaises(ValueError):
            validate_review(review, "An existing claim in draft")

    def test_scientist_review_supports_bounded_response_function(self):
        from ai_scientist.perform_llm_review import perform_review
        calls = []
        def respond(prompt, **kwargs):
            calls.append(prompt)
            return '```json\n{"summary": "bounded"}\n```', []
        result = perform_review("draft", "unused", None, num_fs_examples=0,
                                response_fn=respond)
        self.assertEqual(result, {"summary": "bounded"})
        self.assertEqual(len(calls), 1)

    def test_nested_raw_json_does_not_return_inner_issue(self):
        from ai_scientist.llm import extract_json_between_markers
        result = extract_json_between_markers('{"summary":"review","issues":[{"id":"R1"}]}')
        self.assertEqual(result["summary"], "review")
        self.assertIsNone(extract_json_between_markers('{"summary":"review","issues":[{"id":"R1"}'))


if __name__ == "__main__":
    unittest.main()
