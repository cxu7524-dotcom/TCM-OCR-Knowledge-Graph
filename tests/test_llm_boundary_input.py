import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.schemas import Block
from extract.formula_segments import extract_entry_scopes
from extract.llm_extraction import PROMPT_VERSION, build_input, build_messages
from extract.llm_validate import validate_candidates


def blocks(*texts):
    return [Block(page_id="p", line_id=f"L{i}", block_ref=f"p:L{i}", reading_order=i, text=text,
                  bbox=[i, 0, i + 1, 20], source_hash=f"h{i}") for i, text in enumerate(texts, 1)]


class BoundaryInputTests(unittest.TestCase):
    def test_same_block_shares_exact_ranges_with_validator(self):
        items = blocks("桂枝甘草各一钱。四物汤当归三钱川芎二钱")
        expected = extract_entry_scopes(items)
        payload = build_input(items)
        self.assertEqual(payload["entry_scopes"]["continuation"], expected["continuation"])
        self.assertEqual(payload["entry_scopes"]["entries"], expected["entries"])
        self.assertEqual(expected["continuation"][0]["sources"][0]["block_raw_range"], [0, 8])
        self.assertEqual(expected["entries"][0]["sources"][0]["block_raw_range"], [8, 11])
        self.assertTrue(any(x["block_raw_range"] == [11, len(items[0].text)] for x in payload["entry_scopes"]["unclassified"]))
        self.assertEqual(json.loads(build_messages(payload)[1]["content"]), payload)
        with patch("extract.llm_validate.extract_entry_scopes", wraps=extract_entry_scopes) as scope:
            validate_candidates(items, [items[0].block_ref], {"error": "", "output": {"entities": [], "relations": [], "unresolved": []}})
            self.assertEqual(scope.call_count, 1)

    def test_two_formulas_are_not_continuation_and_context_is_explicit(self):
        items = blocks("三化汤", "羌活三钱芒硝二钱。六君子汤人参二钱白术三钱。")
        items[0].kind = "title"
        payload = build_input(items, [items[1].block_ref])
        scopes = payload["entry_scopes"]
        self.assertEqual(scopes["continuation"], [])
        self.assertTrue({"三化汤", "六君子汤"}.issubset({x["text"] for x in scopes["entries"]}))
        self.assertEqual([x["scope"] for x in payload["blocks"]], ["context", "target"])

    def test_prompt_and_unclassified_protect_normal_cases(self):
        payload = build_input(blocks("桂枝汤", "桂枝、", "芍药各三钱", "治风寒，用桂枝汤", "熟地自三钱可以加至五钱"))
        prompt = build_messages(payload)[0]["content"]
        for phrase in ("页首续文", "药味自身", "治某证，用某方", "制法", "剂量范围", "共用"):
            self.assertIn(phrase, prompt)
        self.assertEqual(PROMPT_VERSION, "tcm-extract-prompt-2.0")
        self.assertTrue(any(x["block_ref"] == "p:L3" for passage in payload["entry_scopes"]["entry_passages"] for x in passage["passage"]["sources"]))

    def test_fixed_pages_candidate_passages_do_not_overlap_unknown(self):
        path = Path(__file__).resolve().parents[1] / "output/llm_eval_step5_nothink/fixed_dataset.json"
        dataset = json.loads(path.read_text(encoding="utf8"))
        for page in (p for p in dataset["pages"] if p["number"] in (35, 37)):
            scopes = build_input([Block.from_dict(b) for b in page["blocks"]])["entry_scopes"]
            for passage in scopes["entry_passages"]:
                for src in passage["passage"]["sources"]:
                    if src["mapping_precision"] not in {"exact", "aligned"}:
                        continue
                    for unknown in scopes["unclassified"]:
                        if src["block_ref"] == unknown["block_ref"]:
                            a, b = src["block_raw_range"]
                            c, d = unknown["block_raw_range"]
                            self.assertFalse(a < d and c < b, (page["number"], src, unknown))
            if page["number"] == 37:
                self.assertTrue(any(x["block_ref"].endswith(":R012") for x in scopes["unclassified"]))


if __name__ == "__main__":
    unittest.main()
