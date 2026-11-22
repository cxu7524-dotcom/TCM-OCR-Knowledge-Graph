import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import store
from app.schemas import Block, OCRResult
from extract.formula_segments import extract_entry_scopes
from extract.llm_validate import validate_candidates


def make_blocks(*texts):
    return [Block(page_id="p", line_id=f"L{i}", block_ref=f"p:L{i}", reading_order=i, text=text,
                  bbox=[i, 0, i + 1, 20], source_hash=f"hash-{i}") for i, text in enumerate(texts, 1)]


def evidence(block, text):
    return [{"block_ref": block.block_ref, "quote": text}]


def candidate(ident, name, kind, source):
    return {"id": ident, "name": name, "type": kind, "evidence": source}


def relation(subject, obj, anchor, kind, subject_ev, relation_ev, qualifier=""):
    return {"subject_id": subject, "object_id": obj, "entry_anchor_id": anchor, "type": kind,
            "subject_evidence": subject_ev, "relation_evidence": relation_ev, "qualifier": qualifier}


def validate(blocks, entities, relations):
    return validate_candidates(blocks, [b.block_ref for b in blocks],
                               {"error": "", "output": {"entities": entities, "relations": relations, "unresolved": []}})


class EntryScopeTests(unittest.TestCase):
    def test_same_block_later_formula_does_not_create_false_continuation(self):
        cases = [
            ("三化汤", "羌活三钱芒硝二钱。六君子汤人参二钱白术三钱。", "六君子汤", "羌活", "三钱", "人参", "二钱", "title"),
            ("四物汤", "当归一钱川芎二钱。桂枝汤桂枝三钱甘草二钱", "桂枝汤", "当归", "一钱", "桂枝", "三钱", "paragraph"),
        ]
        for first_name, body_text, second_name, first_herb, first_dose, second_herb, second_dose, heading_kind in cases:
            with self.subTest(first=first_name):
                title, body, finish = make_blocks(first_name, body_text, "水煎服")
                title.kind = heading_kind
                scopes = extract_entry_scopes([title, body, finish])
                self.assertTrue({first_name, second_name}.issubset({entry["text"] for entry in scopes["entries"]}))
                first_start = body.text.index(first_herb)
                self.assertFalse(any(src["block_ref"] == body.block_ref and
                                     src["block_raw_range"][0] <= first_start < src["block_raw_range"][1]
                                     for scope in scopes["continuation"] for src in scope["sources"]))
                def at(text, start):
                    return [{"block_ref": body.block_ref, "quote": text, "start": start}]
                second_start = body.text.index(second_name)
                first_dose_start = body.text.index(first_dose, first_start)
                second_herb_start = body.text.index(second_herb, second_start + len(second_name))
                second_dose_start = body.text.index(second_dose, second_herb_start)
                entities = [candidate("a", first_name, "方剂", evidence(title, first_name)),
                            candidate("b", second_name, "方剂", at(second_name, second_start)),
                            candidate("h1", first_herb, "药材", at(first_herb, first_start)),
                            candidate("d1", first_dose, "剂量", at(first_dose, first_dose_start)),
                            candidate("h2", second_herb, "药材", at(second_herb, second_herb_start)),
                            candidate("d2", second_dose, "剂量", at(second_dose, second_dose_start))]
                rels = [relation("h1", "d1", "a", "剂量", at(first_herb, first_start), at(first_dose, first_dose_start)),
                        relation("h2", "d2", "b", "剂量", at(second_herb, second_herb_start), at(second_dose, second_dose_start))]
                result = validate([title, body, finish], entities, rels)
                self.assertEqual({t.head for t in result.triples}, {first_herb, second_herb})
                self.assertFalse(result.unresolved)

    def test_continuation_rejected_and_audit_roundtrip(self):
        blocks = make_blocks("桂枝甘草各一钱", "四物汤", "当归三钱川芎二钱")
        first, title, _ = blocks
        entities = [candidate("a", "四物汤", "方剂", evidence(title, "四物汤")),
                    candidate("h", "桂枝", "药材", evidence(first, "桂枝")),
                    candidate("d", "各一钱", "剂量", evidence(first, "各一钱"))]
        rel = relation("h", "d", "a", "剂量", evidence(first, "桂枝"), evidence(first, "各一钱"))
        result = validate(blocks, entities, [rel])
        self.assertFalse(result.triples)
        audit = result.unresolved[0]
        self.assertEqual(audit["reason"], "continuation_entry_conflict")
        self.assertEqual(audit["candidate_anchor"]["name"], "四物汤")
        self.assertEqual(audit["subject_evidence"][0]["quote"], "桂枝")
        self.assertEqual(audit["object_evidence"][0]["quote"], "各一钱")
        self.assertTrue(audit["scope_evidence"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            paths = SimpleNamespace(store_dir=path, db_path=path / "test.sqlite", ensure_dirs=lambda: None)
            with patch.object(store, "PATHS", paths):
                store.save_page(OCRResult(page_id="p", blocks=blocks), [], [], unresolved=result.unresolved)
                saved = store.get_unresolved("p")[0]
                self.assertEqual(saved["candidate_anchor"], audit["candidate_anchor"])
                self.assertEqual(saved["subject_evidence"], audit["subject_evidence"])
                self.assertEqual(saved["scope_evidence"], audit["scope_evidence"])

    def test_conflict_requires_anchor_to_follow_continuation_evidence(self):
        title, body = make_blocks("三化汤", "羌活三钱芒硝二钱")
        entities = [candidate("f", "三化汤", "方剂", evidence(title, "三化汤")),
                    candidate("h", "羌活", "药材", evidence(body, "羌活")),
                    candidate("d", "三钱", "剂量", evidence(body, "三钱"))]
        rel = relation("h", "d", "f", "剂量", evidence(body, "羌活"), evidence(body, "三钱"))
        scopes = {"entries": [{"text": "三化汤", "sources": [{"block_ref": title.block_ref, "block_raw_range": [0, 3]}]}],
                  "continuation": [{"sources": [{"block_ref": body.block_ref, "block_raw_range": [0, len(body.text)]}]}]}
        with patch("extract.llm_validate.extract_entry_scopes", return_value=scopes):
            result = validate([title, body], entities, [rel])
        self.assertEqual(len(result.triples), 1)
        self.assertFalse(result.unresolved)

    def test_same_block_boundary_does_not_reject_later_entry(self):
        blocks = make_blocks("桂枝甘草各一钱。四物汤当归三钱川芎二钱")
        b = blocks[0]
        scopes = extract_entry_scopes(blocks)
        self.assertEqual(scopes["continuation"][0]["sources"][0]["block_raw_range"], [0, 8])
        entities = [candidate("a", "四物汤", "方剂", evidence(b, "四物汤")),
                    candidate("old", "桂枝", "药材", evidence(b, "桂枝")),
                    candidate("old_d", "各一钱", "剂量", evidence(b, "各一钱")),
                    candidate("new", "当归", "药材", evidence(b, "当归")),
                    candidate("new_d", "三钱", "剂量", evidence(b, "三钱"))]
        rels = [relation("old", "old_d", "a", "剂量", evidence(b, "桂枝"), evidence(b, "各一钱")),
                relation("new", "new_d", "a", "剂量", evidence(b, "当归"), evidence(b, "三钱"))]
        result = validate(blocks, entities, rels)
        self.assertEqual([(x.head, x.tail) for x in result.triples], [("当归", "三钱")])
        self.assertEqual([x["reason"] for x in result.unresolved], ["continuation_entry_conflict"])

    def test_cross_column_shared_dose(self):
        blocks = make_blocks("桂枝汤", "桂枝、", "芍药各三钱")
        title, a, b = blocks
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("a", "桂枝", "药材", evidence(a, "桂枝")),
                    candidate("b", "芍药", "药材", evidence(b, "芍药")),
                    candidate("d", "各三钱", "剂量", evidence(b, "各三钱"))]
        rels = [relation("a", "d", "f", "剂量", evidence(a, "桂枝"), evidence(b, "各三钱"), "各"),
                relation("b", "d", "f", "剂量", evidence(b, "芍药"), evidence(b, "各三钱"), "各")]
        result = validate(blocks, entities, rels)
        self.assertEqual({t.head for t in result.triples}, {"桂枝", "芍药"})
        self.assertFalse(result.unresolved)

    def test_explicit_later_formula_is_allowed(self):
        blocks = make_blocks("治风寒，用桂枝汤")
        b = blocks[0]
        entities = [candidate("f", "桂枝汤", "方剂", evidence(b, "桂枝汤")),
                    candidate("d", "风寒", "主治", evidence(b, "风寒"))]
        rel = relation("f", "d", "f", "主治", evidence(b, "桂枝汤"), evidence(b, "治风寒"))
        self.assertEqual(len(validate(blocks, entities, [rel]).triples), 1)

    def test_conditional_group_requires_original_condition(self):
        blocks = make_blocks("桂枝汤", "咳嗽加杏仁桔梗各二钱")
        title, body = blocks
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "杏仁", "药材", evidence(body, "杏仁")),
                    candidate("h2", "桔梗", "药材", evidence(body, "桔梗")),
                    candidate("d", "各二钱", "剂量", evidence(body, "各二钱"))]
        rels = [relation("f", "h", "f", "组成", evidence(title, "桂枝汤"), evidence(body, "杏仁"), "条件加减：咳嗽"),
                relation("h", "d", "f", "剂量", evidence(body, "杏仁"), evidence(body, "各二钱"), "咳嗽加味"),
                relation("h2", "d", "f", "剂量", evidence(body, "桔梗"), evidence(body, "各二钱"), "咳嗽加味")]
        result = validate(blocks, entities, rels)
        self.assertEqual(len(result.triples), 3)
        self.assertTrue(all(t.condition_evidence[0]["quote"] == body.text for t in result.triples))
        for rel in rels:
            wrong = dict(rel, qualifier="基础配伍")
            rejected = validate(blocks, entities, [wrong])
            self.assertFalse(rejected.triples)
            self.assertEqual(rejected.unresolved[0]["reason"], "condition_missing")
        contradictory = dict(rels[0], qualifier="基础配伍，咳嗽加味")
        self.assertFalse(validate(blocks, entities, [contradictory]).triples)

    def test_cross_block_condition_and_nonconditional_add(self):
        title, condition, drug = make_blocks("桂枝汤", "若咳嗽，加", "杏仁二钱")
        blocks = [title, condition, drug]
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "杏仁", "药材", evidence(drug, "杏仁")),
                    candidate("d", "二钱", "剂量", evidence(drug, "二钱"))]
        rel = relation("h", "d", "f", "剂量", evidence(drug, "杏仁"), evidence(drug, "二钱"), "若咳嗽")
        result = validate(blocks, entities, [rel])
        self.assertEqual(len(result.triples), 1)
        self.assertEqual([e["line_id"] for e in result.triples[0].condition_evidence], ["L2", "L3"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            paths = SimpleNamespace(store_dir=path, db_path=path / "test.sqlite", ensure_dirs=lambda: None)
            with patch.object(store, "PATHS", paths):
                store.save_page(OCRResult(page_id="p", blocks=blocks), result.entities, result.triples)
                self.assertEqual(store.get_triples("p")[0].condition_evidence, result.triples[0].condition_evidence)
        title, body = make_blocks("桂枝汤", "引加生姜二钱")
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "生姜", "药材", evidence(body, "生姜")),
                    candidate("d", "二钱", "剂量", evidence(body, "二钱"))]
        rel = relation("h", "d", "f", "剂量", evidence(body, "生姜"), evidence(body, "二钱"), "原文剂量")
        result = validate([title, body], entities, [rel])
        self.assertFalse(result.triples)
        self.assertEqual(result.unresolved[0]["reason"], "addition_role_uncertain")
        rel["qualifier"] = "引药"
        self.assertEqual(len(validate([title, body], entities, [rel]).triples), 1)

    def test_condition_scope_spans_multiple_blocks_but_stops_at_next_title(self):
        title, condition, first, second, next_title, next_drug = make_blocks(
            "桂枝汤", "咳嗽加", "杏仁二钱", "桔梗一钱", "四物汤", "当归三钱")
        blocks = [title, condition, first, second, next_title, next_drug]
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "桔梗", "药材", evidence(second, "桔梗")),
                    candidate("d", "一钱", "剂量", evidence(second, "一钱")),
                    candidate("f2", "四物汤", "方剂", evidence(next_title, "四物汤")),
                    candidate("h2", "当归", "药材", evidence(next_drug, "当归")),
                    candidate("d2", "三钱", "剂量", evidence(next_drug, "三钱"))]
        conditional = relation("h", "d", "f", "剂量", evidence(second, "桔梗"), evidence(second, "一钱"), "咳嗽加味")
        normal = relation("h2", "d2", "f2", "剂量", evidence(next_drug, "当归"), evidence(next_drug, "三钱"), "原文剂量")
        result = validate(blocks, entities, [conditional, normal])
        self.assertEqual({t.head for t in result.triples}, {"桔梗", "当归"})
        group = next(t for t in result.triples if t.head == "桔梗")
        self.assertEqual([e["line_id"] for e in group.condition_evidence], ["L2", "L3", "L4"])
        wrong = validate(blocks, entities, [dict(conditional, qualifier="基础配伍")])
        self.assertEqual(wrong.unresolved[0]["reason"], "condition_missing")

    def test_dose_range_add_to_is_not_symptom_addition(self):
        title, body = make_blocks("六味地黄汤", "熟地自三钱可以加至五钱")
        entities = [candidate("f", "六味地黄汤", "方剂", evidence(title, "六味地黄汤")),
                    candidate("h", "熟地", "药材", evidence(body, "熟地")),
                    candidate("d", "自三钱可以加至五钱", "剂量", evidence(body, "自三钱可以加至五钱"))]
        rel = relation("h", "d", "f", "剂量", evidence(body, "熟地"), evidence(body, "自三钱可以加至五钱"), "原文剂量")
        result = validate([title, body], entities, [rel])
        self.assertEqual(len(result.triples), 1)
        self.assertFalse(result.triples[0].condition_evidence)

    def test_negated_addition_is_not_positive_composition(self):
        title, body = make_blocks("桂枝汤", "不加杏仁二钱")
        entities = [candidate("f", "桂枝汤", "方剂", evidence(title, "桂枝汤")),
                    candidate("h", "杏仁", "药材", evidence(body, "杏仁"))]
        rel = relation("f", "h", "f", "组成", evidence(title, "桂枝汤"), evidence(body, "杏仁"), "引药")
        result = validate([title, body], entities, [rel])
        self.assertFalse(result.triples)
        self.assertEqual(result.unresolved[0]["reason"], "negated_addition")


if __name__ == "__main__":
    unittest.main()
