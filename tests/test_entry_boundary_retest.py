import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.schemas import Block
from extract.llm_validate import validate_candidates
from tests.test_llm_entry_scope import candidate, evidence, relation
from tools import entry_boundary_retest as retest
from tools import llm_eval_step5


class RetestTests(unittest.TestCase):
    def test_historical_count_is_verified_from_saved_results(self):
        self.assertEqual(retest.historical_wrong_count(), 4)

    def test_model_guess_and_validator_rejection_are_separate(self):
        def block(line, order, text):
            return Block(page_id="p", line_id=line, block_ref="p:" + line, reading_order=order,
                         text=text, bbox=[order, 0, order + 1, 20], source_hash="h")
        first = block("R018", 1, "遍身瘙痒加薄荷叶防风各一钱")
        title = block("R002", 2, "四物汤")
        body = block("R013", 3, "当归三钱川芎二钱")
        blocks = [first, title, body]
        entities = [candidate("a", "四物汤", "方剂", evidence(title, "四物汤")),
                    candidate("h", "薄荷叶", "药材", evidence(first, "薄荷叶")),
                    candidate("d", "一钱", "剂量", evidence(first, "一钱"))]
        rel = relation("h", "d", "a", "剂量", evidence(first, "薄荷叶"), evidence(first, "各一钱"), "与防风共用各一钱")
        output = {"entities": entities, "relations": [rel], "unresolved": []}
        validated = validate_candidates(blocks, [b.block_ref for b in blocks], {"error": "", "output": output})
        projected = retest.project({"number": 37, "blocks": [b.to_dict() for b in blocks]}, output, validated)
        self.assertEqual(projected["guessed_later_formula_count"], 1)
        self.assertEqual(projected["after_validation_wrong_entry_count"], 0)
        self.assertEqual(projected["validation_unresolved"][0]["reason"], "continuation_entry_conflict")
        self.assertIn("防风", projected["before_validation"][0]["qualifier"])

    def test_historical_report_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = {"prompt_version": "tcm-extract-prompt-1.0", "protocol_version": "tcm-extract-protocol-1.0",
                       "pages": [{"number": n} for n in llm_eval_step5.PAGES]}
            (root / "fixed_dataset.json").write_text(json.dumps(dataset), encoding="utf8")
            old = root / "comparison.json"
            old.write_text("unchanged", encoding="utf8")
            with patch("sys.argv", ["llm_eval_step5.py", "report", "--dir", str(root)]), \
                 patch.object(llm_eval_step5, "summarize", return_value={"summary": {}}), \
                 contextlib.redirect_stdout(io.StringIO()):
                llm_eval_step5.main()
            self.assertEqual(old.read_text(encoding="utf8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
