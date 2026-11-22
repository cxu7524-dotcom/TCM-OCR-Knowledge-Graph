import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import store
from app.schemas import Block, Entity, OCRResult, Triple
from extract.llm_validate import validate_candidates
from graph.build import build_graph, dedup_triples


def block(line, text):
    return Block(page_id="p", block_ref="p:" + line, line_id=line, text=text,
                 reading_order=int(line[1:]), bbox=[1, 2, 3, 4], source_hash="hash-" + line)


def ev(line, quote, **kwargs):
    return {"block_ref": "p:" + line, "quote": quote, **kwargs}


def entity(ident, name, kind, evidence):
    return {"id": ident, "name": name, "type": kind, "evidence": evidence}


def relation(subject, obj, anchor, kind, subject_evidence, relation_evidence, qualifier=""):
    return {"subject_id": subject, "object_id": obj, "entry_anchor_id": anchor, "type": kind,
            "subject_evidence": subject_evidence, "relation_evidence": relation_evidence, "qualifier": qualifier}


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [block("L1", "桂枝"), block("L2", "汤"), block("L3", "甘草三钱，主治风寒。"),
                       block("L4", "桂枝汤"), block("L5", "甘草三钱。"), block("L6", "甘草一钱。")]
        self.targets = [b.block_ref for b in self.blocks[2:]]
        self.first_title = [ev("L1", "桂枝"), ev("L2", "汤")]
        self.entities = [entity("a", "桂枝汤", "方剂", self.first_title),
                         entity("b", "甘草", "药材", [ev("L3", "甘草")]),
                         entity("c", "三钱", "剂量", [ev("L3", "三钱")]),
                         entity("d", "桂枝汤", "方剂", [ev("L4", "桂枝汤")]),
                         entity("e", "甘草", "药材", [ev("L5", "甘草")]),
                         entity("f", "三钱", "剂量", [ev("L5", "三钱")]),
                         entity("g", "一钱", "剂量", [ev("L6", "一钱")])]
        self.rel = [relation("b", "c", "a", "剂量", [ev("L3", "甘草")], [ev("L3", "甘草三钱")], "原文剂量"),
                    relation("e", "f", "d", "剂量", [ev("L5", "甘草")], [ev("L5", "甘草三钱")], "原文剂量"),
                    relation("e", "g", "d", "剂量", [ev("L5", "甘草")], [ev("L6", "甘草一钱")], "原文剂量"),
                    relation("a", "b", "a", "组成", self.first_title, [ev("L3", "甘草三钱")], "基础配伍")]

    def run_validation(self, relations=None, entities=None, unresolved=None):
        output = {"entities": copy.deepcopy(self.entities if entities is None else entities),
                  "relations": copy.deepcopy(self.rel if relations is None else relations), "unresolved": unresolved or []}
        return validate_candidates(self.blocks, self.targets, {"error": "", "output": output})

    def test_valid_cross_line_subject_and_dedup(self):
        result = self.run_validation(self.rel + [self.rel[0]])
        self.assertEqual(result.status, "success_with_results")
        self.assertEqual(len(result.triples), 4)
        composed = next(t for t in result.triples if t.relation == "组成")
        self.assertEqual([e["line_id"] for e in composed.subject_evidence], ["L1", "L2"])
        self.assertEqual((composed.line_id, composed.quote, composed.source_hash), ("L3", "甘草三钱", "hash-L3"))
        self.assertEqual(composed.bbox, [1, 2, 3, 4])
        self.assertFalse(composed.validate())
        self.assertTrue(all(not e.validate() for e in result.entities))

    def test_invalid_evidence_and_directions(self):
        for mutate, reason in [
            (lambda r: r["relation_evidence"].__setitem__(0, ev("L3", "虚构")), "quote_not_found"),
            (lambda r: r["relation_evidence"].__setitem__(0, ev("L99", "三钱")), "unknown_block"),
            (lambda r: r.__setitem__("object_id", "missing"), "unknown_endpoint_or_anchor"),
            (lambda r: r.__setitem__("type", "组成"), "invalid_relation_direction"),
            (lambda r: r["relation_evidence"].__setitem__(0, ev("L1", "桂枝")), "outside_target"),
            (lambda r: r.__setitem__("qualifier", "条件加减"), "condition_missing"),
        ]:
            bad = copy.deepcopy(self.rel[0])
            mutate(bad)
            with self.subTest(reason=reason):
                result = self.run_validation([bad], self.entities[:3])
                self.assertFalse(result.triples)
                self.assertIn(reason, [item["reason"] for item in result.unresolved])

    def test_repeated_quote_requires_offset(self):
        repeated = block("L7", "甘草甘草三钱")
        self.blocks.append(repeated)
        self.targets.append(repeated.block_ref)
        items = [self.entities[0], entity("b", "甘草", "药材", [ev("L7", "甘草")]),
                 entity("c", "三钱", "剂量", [ev("L7", "三钱")])]
        rel = relation("b", "c", "a", "剂量", [ev("L7", "甘草")], [ev("L7", "甘草三钱")])
        result = self.run_validation([rel], items)
        self.assertIn("ambiguous_quote", [a["reason"] for a in result.unresolved])
        items[1]["evidence"][0]["start"] = 2
        rel["subject_evidence"][0]["start"] = 2
        result = self.run_validation([rel], items)
        self.assertEqual(result.triples[0].subject_evidence[0]["start"], 2)

    def test_status_and_unresolved(self):
        self.assertEqual(validate_candidates(self.blocks, self.targets, {"error": "timeout", "output": None}).status, "failure")
        self.assertEqual(validate_candidates(self.blocks, self.targets, {"error": "invalid_protocol", "output": None}).status, "failure")
        self.assertEqual(self.run_validation([], []).status, "success_empty")
        rejected = self.run_validation([dict(self.rel[0], object_id="unknown")], self.entities[:3])
        self.assertEqual(rejected.status, "success_empty")
        item = {"evidence": [ev("L3", "主治风寒")], "reason": "主体不明", "candidate_subject_ids": ["a"]}
        result = self.run_validation([], self.entities[:1], [item])
        self.assertEqual(result.unresolved[0]["source_hash"], "hash-L3")
        self.assertTrue(result.unresolved[0]["candidate_entry_ids"][0].startswith("entry-"))

    def test_conditional_composition_preserves_condition(self):
        self.blocks.append(block("L7", "若咳嗽，加杏仁二钱。"))
        self.targets.append("p:L7")
        items = self.entities[:1] + [entity("x", "杏仁", "药材", [ev("L7", "杏仁")])]
        conditional = relation("a", "x", "a", "组成", self.first_title,
                               [ev("L7", "若咳嗽，加杏仁二钱")], "条件加减：若咳嗽，加")
        result = self.run_validation([conditional], items)
        self.assertEqual(result.triples[0].qualifier, "条件加减：若咳嗽，加")
        alternate = dict(conditional, qualifier="条件加减：若咳嗽，加，另法")
        result = self.run_validation([conditional, alternate], items)
        self.assertEqual(len(dedup_triples(result.triples)), 2)
        unconditional = dict(conditional, qualifier="基础配伍")
        result = self.run_validation([unconditional], items)
        self.assertEqual(result.status, "success_empty")
        self.assertEqual(result.unresolved[0]["reason"], "condition_missing")

    def test_anchor_identity_uses_name_position_not_quote_extent(self):
        self.blocks[3].text = "第二方桂枝汤。"
        first = self.run_validation()
        items = copy.deepcopy(self.entities)
        items[3]["evidence"] = [ev("L4", "第二方桂枝汤")]
        second = self.run_validation(entities=items)
        self.assertEqual({t.entry_id for t in first.triples}, {t.entry_id for t in second.triples})

    def test_roundtrip_scoped_records_and_legacy(self):
        result = self.run_validation(self.rel + [self.rel[0]])
        self.assertEqual(len(dedup_triples(result.triples)), 4)
        dose_entries = [t.entry_id for t in result.triples if t.relation == "剂量" and t.tail == "三钱"]
        self.assertEqual(len(set(dose_entries)), 2)
        graph = build_graph(result.triples, result.entities)
        self.assertEqual(len([n for n in graph.nodes if n.label == "甘草"]), 2)
        self.assertEqual(len({e.source for e in graph.edges if e.relation == "剂量"}), 2)
        self.assertTrue(all(e.entry_id and e.qualifier for e in graph.edges))
        self.assertEqual(Entity.from_dict({"name": "甘草", "type": "药材"}).entry_id, "")
        self.assertEqual(Triple.from_dict({"head": "甘草", "tail": "三钱", "relation": "剂量"}).qualifier, "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            paths = SimpleNamespace(store_dir=path, db_path=path / "test.sqlite", ensure_dirs=lambda: None)
            with patch.object(store, "PATHS", paths):
                ocr = OCRResult(page_id="p", blocks=self.blocks)
                note = {"evidence": [ev("L3", "主治风寒")], "reason": "主体不明", "candidate_subject_ids": ["a"]}
                unresolved = self.run_validation([], self.entities[:1], [note]).unresolved
                store.save_page(ocr, result.entities, result.triples, unresolved=unresolved)
                saved = store.get_triples("p")
                self.assertEqual(len(saved), 4)
                self.assertEqual({(t.entry_id, t.head, t.tail, t.qualifier) for t in saved},
                                 {(t.entry_id, t.head, t.tail, t.qualifier) for t in result.triples})
                self.assertEqual(len(store.get_entities("p")), len(result.entities))
                self.assertEqual(store.get_unresolved("p")[0]["candidate_entry_ids"], unresolved[0]["candidate_entry_ids"])
                self.assertEqual(store.get_unresolved("p")[0]["block_ref"], "p:L3")


if __name__ == "__main__":
    unittest.main()
