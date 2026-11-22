import unittest
from local_review import select_context
from review_obsidian import validate_claim_checks

class LocalTests(unittest.TestCase):
    def test_selects_neighbours_and_source_order_without_other_claim_rows(self):
        draft='范围\n\ncontext before\n\n正文C2论述\n\ncontext after\n\n| C1 | one | S1 | E1 | note | scope | section | gap |\n| C2 | two | S2 | E2 | note | scope | section | gap |'
        source='Source note: a.md\nSource note SHA-256: x\n[Evidence E1 | source S1 | kind note_analysis]\none\nSource note: b.md\nSource note SHA-256: y\n[Evidence E2 | source S2 | kind note_analysis]\ntwo'
        selected,evidence,ids=select_context(draft,source,['C2'])
        self.assertIn('context before',selected);self.assertIn('context after',selected)
        self.assertNotIn('| C1 |',selected);self.assertIn('| C2 |',selected)
        self.assertNotIn('[Evidence E1',evidence);self.assertIn('Source note: a.md',evidence)
        validate_claim_checks({'claim_checks':[{'claim_id':'C2','verdict':'supported','reason':'bounded','evidence_quote':'two','evidence_source':'b.md'}]},selected,evidence)
    def test_unknown_claim_rejected(self):
        with self.assertRaises(ValueError):select_context('draft','source',['C99'])
