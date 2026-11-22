import unittest
from tests.test_llm_entry_scope import make_blocks, candidate, evidence, relation, validate


class CompositionRoleTests(unittest.TestCase):
    def test_same_herb_preparation_and_discussion_same_block(self):
        title, body = make_blocks("桂枝汤", "人参二钱水捣碎温服。用人参助正以驱邪也")
        base = body.text.index("人参")
        discuss = body.text.index("人参", base + 1)
        def ev(start):
            return [{"block_ref": body.block_ref, "quote": "人参", "start": start}]
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h1", "人参", "药材", ev(base)), candidate("h2", "人参", "药材", ev(discuss))]
        rels = [relation("f", "h1", "f", "组成", evidence(title, "桂枝汤"), ev(base)),
                relation("f", "h2", "f", "组成", evidence(title, "桂枝汤"), ev(discuss))]
        result = validate([title, body], entities, rels)
        self.assertEqual(len(result.triples), 1)
        self.assertEqual(result.triples[0].relation_evidence[0]["start"], base)
        self.assertEqual(result.unresolved[0]["reason"], "composition_after_preparation_uncertain")
        self.assertTrue(result.unresolved[0]["scope_evidence"])

    def test_uncertain_ocr_preserves_raw_source(self):
        title, body = make_blocks("桂枝汤", "柴朝一钱")
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "柴朝", "药材", evidence(body, "柴朝"))]
        rel = relation("f", "h", "f", "组成", evidence(title, "桂枝汤"), evidence(body, "柴朝"), "OCR作柴朝，疑为柴胡")
        result = validate([title, body], entities, [rel])
        self.assertFalse(result.triples)
        self.assertEqual(result.unresolved[0]["reason"], "uncertain_ocr_herb")
        self.assertEqual(result.unresolved[0]["object_evidence"][0]["quote"], "柴朝")

    def test_unrelated_composition_quote_cannot_prove_herb(self):
        title, body = make_blocks("桂枝汤", "人参二钱甘草三钱")
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "人参", "药材", evidence(body, "人参"))]
        rel = relation("f", "h", "f", "组成", evidence(title, "桂枝汤"), evidence(body, "甘草三钱"))
        result = validate([title, body], entities, [rel])
        self.assertFalse(result.triples)
        self.assertEqual(result.unresolved[0]["reason"], "composition_herb_evidence_mismatch")

    def test_later_formula_cannot_claim_preparation_discussion(self):
        first, body, prep, discussion, later = make_blocks("桂枝汤", "桂枝三钱", "水捣碎温服", "汤中用川芎生地是也", "羌活汤")
        entities = [candidate("f", "羌活汤", "方剂", evidence(later, "羌活汤")),
                    candidate("h", "川芎", "药材", evidence(discussion, "川芎"))]
        rel = relation("f", "h", "f", "组成", evidence(later, "羌活汤"), evidence(discussion, "川芎生地是也"))
        result = validate([first, body, prep, discussion, later], entities, [rel])
        self.assertFalse(result.triples)
        self.assertEqual(result.unresolved[0]["reason"], "composition_after_preparation_uncertain")

    def test_conditional_addition_after_preparation_remains_valid(self):
        title, body, prep, addition = make_blocks("桂枝汤", "桂枝三钱甘草二钱", "水捣碎温服", "咳嗽加杏仁二钱")
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "杏仁", "药材", evidence(addition, "杏仁"))]
        rel = relation("f", "h", "f", "组成", evidence(title, "桂枝汤"), evidence(addition, "杏仁"), "咳嗽加味")
        self.assertEqual(len(validate([title, body, prep, addition], entities, [rel]).triples), 1)


if __name__ == "__main__":
    unittest.main()
