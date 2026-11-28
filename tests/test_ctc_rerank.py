# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ocr.ctc_rerank import ClassicalTCMLM, _GROUPS, _canon, ctc_rerank_context, rerank_sequence_from_probs


class CtcRerankTests(unittest.TestCase):
    def model(self, terms=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        lex = root / "lex.json"
        lex.write_text(json.dumps({"术语": list(terms)}, ensure_ascii=False), encoding="utf-8")
        return ClassicalTCMLM(root / "missing.jsonl", lex)

    def test_corpus_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"book": "温疫方論", "ocr": "天地玄"},
                       {"name": "温疫附录", "book": "其他", "ocr": "宇宙洪"},
                       {"book": "伤寒杂病", "ocr": "寒熱氣"}]
            corpus = root / "corpus.jsonl"
            corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
            lm = ClassicalTCMLM(corpus, root / "missing.json", exclude_books=("温疫",))
            self.assertEqual(lm.unigram, {"寒": 1, "热": 1, "气": 1})
            self.assertEqual(lm.bigram["寒", "热"], 1)
            self.assertEqual(lm.trigram["寒", "热", "气"], 1)

    def matrix(self, initial, replacement, confidence=0.60):
        chars = ["blank"] + list(dict.fromkeys(initial + replacement))
        probs = np.zeros((len(initial), len(chars)))
        for t, ch in enumerate(initial):
            probs[t, chars.index(ch)] = 0.95 if ch == replacement[t] else confidence
            if ch != replacement[t]:
                probs[t, chars.index(replacement[t])] = 0.25
        return probs, chars

    def test_supported_changes(self):
        for before, after, term in (("寒熟", "寒熱", "寒热"), ("道痰", "導痰", "导痰"), ("頃刺", "頃刻", "頃刻")):
            with self.subTest(before=before):
                lm = self.model([term])
                probs, chars = self.matrix(before, after)
                text, trace = rerank_sequence_from_probs(probs, chars, lm)
                self.assertEqual(text, after)
                self.assertEqual(len(trace), 1)
                self.assertEqual(trace[0]["before"], next(a for a, b in zip(before, after) if a != b))
                self.assertGreaterEqual(trace[0]["bg_after"], 2)

    def test_safety_guards(self):
        lm = self.model(["寒热"])
        probs, chars = self.matrix("寒熟", "寒熱", confidence=0.82)
        self.assertEqual(rerank_sequence_from_probs(probs, chars, lm), ("寒熟", []))
        probs, chars = self.matrix("寒熱", "寒热")
        self.assertEqual(_canon("熱"), _canon("热"))
        self.assertEqual(rerank_sequence_from_probs(probs, chars, lm), ("寒熱", []))
        probs, chars = self.matrix("寒熟", "寒熱")
        self.assertEqual(rerank_sequence_from_probs(probs, chars, self.model()), ("寒熟", []))

    def test_traditional_variant_is_not_replaced(self):
        class CandidateFavoringLM:
            def context_score(self, left, cand, right):
                return (10.0, 100) if cand == "润" else (0.0, 0)

        probs, chars = self.matrix("滋潤", "滋润")
        self.assertEqual(_canon("潤"), _canon("润"))
        self.assertEqual(rerank_sequence_from_probs(probs, chars, CandidateFavoringLM()), ("滋潤", []))

    def test_canonical_groups_are_disjoint_and_nontrivial(self):
        seen = set()
        for group in _GROUPS:
            with self.subTest(group=group):
                self.assertGreaterEqual(len(group), 2)
                self.assertEqual(len(group), len(set(group)))
                self.assertTrue(seen.isdisjoint(group))
                seen.update(group)

    def test_ctc_collapse_and_blank(self):
        chars = ["blank", "寒", "熟", "熱"]
        probs = np.array([[0, .95, 0, 0], [0, .95, 0, 0], [1, 0, 0, 0],
                          [0, 0, .60, .25]])
        text, trace = rerank_sequence_from_probs(probs, chars, self.model(["寒热"]))
        self.assertEqual(text, "寒熱")
        self.assertEqual(trace[0]["pos"], 1)

    def test_context_preserves_fields_and_restores(self):
        probs, chars = self.matrix("寒熟", "寒熱")
        class Postprocess:
            character = chars
            def __call__(self, preds, *args, **kwargs):
                return [("寒熟", 0.73, {"extra": [1, 2]})]
        orig = Postprocess()
        rec = SimpleNamespace(postprocess_op=orig)
        engine = SimpleNamespace(text_rec=rec)
        lm = self.model(["寒热"])
        with ctc_rerank_context(engine, lm):
            result = rec.postprocess_op(np.array([probs]), True, extra=True)
            self.assertEqual(result, [("寒熱", 0.73, {"extra": [1, 2]})])
        self.assertIs(rec.postprocess_op, orig)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with ctc_rerank_context(engine, lm):
                raise RuntimeError("stop")
        self.assertIs(rec.postprocess_op, orig)


if __name__ == "__main__":
    unittest.main()
