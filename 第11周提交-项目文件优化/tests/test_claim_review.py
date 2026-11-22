import unittest
from review_obsidian import validate_claim_checks, resolve_evidence_ids, FORM

class ClaimTests(unittest.TestCase):
    def test_prompt_example_uses_ids_not_retyped_quotes(self):
        self.assertIn('"evidence_id":"E6","evidence_source":"","evidence_quote":""',FORM)

    def test_evidence_id_expansion_preserves_verdict_and_raw(self):
        raw={'claim_checks':[{'claim_id':'C1','verdict':'partial','reason':'gap','evidence_id':'E1','evidence_quote':'','evidence_source':''}]}
        source='Source note: notes/a.md\nSource note SHA-256: hash\n[Evidence E1 | source S1 | kind note_analysis]\nreal evidence'
        result, changes=resolve_evidence_ids(raw,source)
        self.assertEqual(result['claim_checks'][0]['evidence_quote'],'real evidence')
        self.assertEqual(result['claim_checks'][0]['verdict'],'partial')
        self.assertEqual(raw['claim_checks'][0]['evidence_quote'],'')
        self.assertEqual(len(changes),1)
        validate_claim_checks(result,self.draft,source)

    draft='| C1 | claim | S1 | E1 | type | condition | section | gap |'
    source='Source note: notes/a.md\nSource note SHA-256: hash\nreal evidence'
    check={'claim_id':'C1','verdict':'supported','evidence_source':'notes/a.md','evidence_quote':'real evidence','reason':'bounded support'}

    def test_exact_coverage_and_quote(self):
        validate_claim_checks({'claim_checks':[self.check]},self.draft,self.source)

    def test_missing_duplicate_unknown_and_wrong_source(self):
        cases=[[],[self.check,self.check],[{**self.check,'claim_id':'C9'}],[{**self.check,'evidence_source':'other.md'}],[{**self.check,'evidence_quote':'invented'}],[{**self.check,'evidence_quote':''}]]
        for checks in cases:
            with self.subTest(checks=checks),self.assertRaises(ValueError):
                validate_claim_checks({'claim_checks':checks},self.draft,self.source)

    def test_unknown_is_not_faked_support(self):
        check={**self.check,'verdict':'not_assessable','evidence_source':'','evidence_quote':'','reason':'no evidence'}
        validate_claim_checks({'claim_checks':[check]},self.draft,self.source)
