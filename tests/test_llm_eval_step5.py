import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas import Block
from extract.llm_local import ChatResult
from tools.llm_eval_step5 import BASE_URL, MODEL, PROMPT_VERSION, PROTOCOL_VERSION, main, maximum_matching, review_probe, run_page, score_page, summarize


class EvaluationTests(unittest.TestCase):
    def test_matching_maximizes_hits_instead_of_first_match(self):
        self.assertEqual(maximum_matching([[0, 1], [0]]), {0: 1, 1: 0})

    def setUp(self):
        self.page = {"number": 1, "split": "holdout", "ocr_page_id": "p", "source_image": "test.png", "ocr_source": "test.json",
                     "blocks": [{"block_ref": "p:a", "bbox": [0, 0, 10, 10]},
                                {"block_ref": "p:b", "bbox": [20, 0, 30, 10]}],
                     "gold_rows": [{"id": "A:1", "entry": "桂枝汤", "entry_bbox": [0, 0, 10, 10],
                                    "herb": "甘草", "herb_bbox": [20, 0, 30, 10],
                                    "dose": "各三钱", "dose_bbox": [20, 0, 30, 10]}]}
        self.entities = [{"entry_id": "one", "name": "桂枝汤", "type": "方剂", "block_ref": "p:a"},
                         {"entry_id": "one", "name": "甘草", "type": "药材", "block_ref": "p:b"},
                         {"entry_id": "one", "name": "三钱", "type": "剂量", "block_ref": "p:b"}]
        self.triples = [{"entry_id": "one", "head": "桂枝汤", "relation": "组成", "tail": "甘草", "evidence_refs": ["p:b"], "qualifier": "基础配伍"},
                        {"entry_id": "one", "head": "甘草", "relation": "剂量", "tail": "三钱", "evidence_refs": ["p:b"], "qualifier": "各"}]

    def score(self, triples):
        return score_page(self.page, {"predictions": {"entities": self.entities, "triples": triples}})

    def test_gold_match_and_unreviewed_predictions(self):
        result = self.score(self.triples)
        self.assertEqual((result["composition_hits"], result["dose_hits"], result["entity_hits"]), (1, 1, 3))
        extra = {"entry_id": "one", "head": "桂枝汤", "relation": "主治", "tail": "风寒", "evidence_refs": ["p:b"]}
        result = self.score(self.triples + [extra])
        self.assertEqual(result["relation_unscored"], 1)
        self.assertIsNone(result["precision"])

    def test_wrong_entry_and_conditional_not_basic(self):
        wrong = [dict(self.triples[0], entry_id="other"), dict(self.triples[1], qualifier="若咳嗽，加")]
        result = self.score(wrong)
        self.assertEqual((result["composition_hits"], result["dose_hits"]), (0, 0))
        self.assertEqual(result["relation_unscored"], 2)

    def test_incomplete_cloud_is_not_successful_comparison(self):
        dataset = {"pages": [self.page], "parameters": {}, "scope": "test"}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            result = {"page": 1, "dataset_sha256": "hash", "status": "failure", "error": "timeout", "elapsed_seconds": 1.0}
            (directory / "cloud_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf8")
            report = summarize(dataset, directory, "hash")
            self.assertEqual(report["summary"]["cloud"]["failure_rate"], 1.0)
            self.assertIsNone(report["summary"]["cloud"]["relation_recall"])
            self.assertEqual(report["summary"]["rules"]["status"], "not_run")

    def test_simulated_cloud_response_is_parsed_and_scored(self):
        page = dict(self.page)
        page["blocks"] = [Block(page_id="p", line_id="a", block_ref="p:a", text="桂枝汤", bbox=[0, 0, 10, 10], reading_order=1, source_hash="a").to_dict(),
                          Block(page_id="p", line_id="b", block_ref="p:b", text="甘草三钱", bbox=[20, 0, 30, 10], reading_order=2, source_hash="b").to_dict()]
        def ev(ref, quote):
            return [{"block_ref": ref, "quote": quote}]
        entities = [{"id": "a", "name": "桂枝汤", "type": "方剂", "evidence": ev("p:a", "桂枝汤")},
                    {"id": "b", "name": "甘草", "type": "药材", "evidence": ev("p:b", "甘草")},
                    {"id": "c", "name": "三钱", "type": "剂量", "evidence": ev("p:b", "三钱")}]
        def relation(subject, obj, kind, subject_ev):
            return {"subject_id": subject, "object_id": obj, "type": kind, "entry_anchor_id": "a",
                    "subject_evidence": subject_ev, "relation_evidence": ev("p:b", "甘草三钱"), "qualifier": ""}
        output = {"entities": entities, "relations": [relation("a", "b", "组成", ev("p:a", "桂枝汤")),
                                                        relation("b", "c", "剂量", ev("p:b", "甘草"))], "unresolved": []}
        class Client:
            def chat_result(self, messages, schema):
                self.messages = messages
                return ChatResult(content=json.dumps(output, ensure_ascii=False), status=200, finish_reason="stop",
                                  completion_tokens=50, reasoning_tokens=0)
        client = Client()
        result = run_page(page, "cloud", client)
        self.assertEqual((result["score"]["composition_hits"], result["score"]["dose_hits"]), (1, 1))
        self.assertTrue(result["probe_passed"])
        self.assertEqual(result["diagnostics"]["completion_tokens"], 50)
        self.assertEqual(json.loads(client.messages[1]["content"])["blocks"][1]["text"], "甘草三钱")
        output["entities"][1]["evidence"][0]["quote"] = "虚构"
        self.assertFalse(run_page(page, "cloud", client)["probe_passed"])

    def test_no_thinking_cloud_requires_matching_development_probe(self):
        parameters = {"response_format": "json_object", "temperature": 0.1, "max_tokens": 16384, "timeout": 180, "thinking": "disabled"}
        dataset = {"model": MODEL, "base_url": BASE_URL, "prompt_version": PROMPT_VERSION,
                   "protocol_version": PROTOCOL_VERSION, "parameters": parameters,
                   "pages": [{"number": number, "split": "development" if number == 35 else "holdout"}
                             for number in (35, 37, 39, 40, 41, 50)]}
        config = SimpleNamespace(enabled=True, backend="cloud", base_url=BASE_URL, model=MODEL,
                                 response_format="json_object", temperature=0.1, max_tokens=16384, timeout=180, thinking="disabled")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            original = directory / "source.json"
            original_dataset = {**dataset, "parameters": {k: v for k, v in parameters.items() if k != "thinking"}}
            original.write_text(json.dumps(original_dataset), encoding="utf8")
            dataset["derived_from_dataset_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
            (directory / "fixed_dataset.json").write_text(json.dumps(dataset), encoding="utf8")
            with patch("tools.llm_eval_step5.EXTENDED_DATASET", original), \
                    patch("tools.llm_eval_step5.LocalLLMExtractor", return_value=SimpleNamespace(cfg=config)), \
                    patch("sys.argv", ["eval", "cloud", "--dir", str(directory)]):
                with self.assertRaisesRegex(SystemExit, "开发页探测尚未通过"):
                    main()
            self.assertFalse((directory / "cloud_results.jsonl").exists())

    def test_probe_sends_only_development_then_gate_allows_fixed_batch(self):
        parameters = {"response_format": "json_object", "temperature": 0.1, "max_tokens": 16384, "timeout": 180, "thinking": "disabled"}
        pages = [{"number": number, "split": "development" if number == 35 else "holdout", "ocr_page_id": "p",
                  "blocks": [{"block_ref": "p:a", "page_id": "p", "line_id": "a", "bbox": [0, 0, 1, 1],
                              "source_hash": "hash", "text": "甘草"}]}
                 for number in (35, 37, 39, 40, 41, 50)]
        dataset = {"model": MODEL, "base_url": BASE_URL, "prompt_version": PROMPT_VERSION,
                   "protocol_version": PROTOCOL_VERSION, "parameters": parameters, "pages": pages}
        config = SimpleNamespace(enabled=True, backend="cloud", base_url=BASE_URL, model=MODEL,
                                 response_format="json_object", temperature=0.1, max_tokens=16384, timeout=180, thinking="disabled")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            original = directory / "source.json"
            original_dataset = {**dataset, "parameters": {k: v for k, v in parameters.items() if k != "thinking"}}
            original.write_text(json.dumps(original_dataset), encoding="utf8")
            dataset["derived_from_dataset_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
            (directory / "fixed_dataset.json").write_text(json.dumps(dataset), encoding="utf8")
            loc = {"page_id": "p", "block_ref": "p:a", "line_id": "a", "bbox": [0, 0, 1, 1],
                   "source_hash": "hash", "quote": "甘草", "start": 0, "end": 2}
            triple = {"page_id": "p", "line_id": "a", "bbox": [0, 0, 1, 1], "source_hash": "hash", "quote": "甘草",
                      "head": "甘草", "relation": "主治", "tail": "风寒", "entry_id": "entry",
                      "evidence_refs": ["p:a"], "subject_evidence": [loc], "relation_evidence": [loc]}
            success = {"status": "success_with_results", "error": "", "probe_passed": True, "http_status": 200,
                       "diagnostics": {"finish_reason": "stop"}, "predictions": {"triples": [triple], "unresolved": [{"reason": "condition_missing"}]}}
            with patch("tools.llm_eval_step5.EXTENDED_DATASET", original), \
                    patch("tools.llm_eval_step5.LocalLLMExtractor", return_value=SimpleNamespace(cfg=config)), \
                    patch("tools.llm_eval_step5.run_page", side_effect=lambda page, *_: {**success, "page": page["number"], "split": page["split"]}) as run:
                with patch("sys.argv", ["eval", "probe", "--dir", str(directory)]):
                    self.assertEqual(main(), 0)
                self.assertEqual([call.args[0]["number"] for call in run.call_args_list], [35])
                run.reset_mock()
                with patch("sys.argv", ["eval", "cloud", "--dir", str(directory)]):
                    self.assertEqual(main(), 0)
                self.assertEqual([call.args[0]["number"] for call in run.call_args_list], [35, 37, 39, 40, 41, 50])

    def test_review_allows_unresolved_but_rejects_truncation_and_invalid_evidence(self):
        loc = {"page_id": "p", "block_ref": "p:a", "line_id": "a", "bbox": [0, 0, 1, 1],
               "source_hash": "hash", "quote": "甘草", "start": 0, "end": 2}
        triple = {"page_id": "p", "line_id": "a", "bbox": [0, 0, 1, 1], "source_hash": "hash", "quote": "甘草",
                  "head": "甘草", "relation": "主治", "tail": "风寒", "entry_id": "entry",
                  "evidence_refs": ["p:a"], "subject_evidence": [loc], "relation_evidence": [dict(loc)]}
        page = {"number": 35, "split": "development", "ocr_page_id": "p",
                "blocks": [{"page_id": "p", "block_ref": "p:a", "line_id": "a", "bbox": [0, 0, 1, 1], "source_hash": "hash", "text": "甘草"}]}
        dataset = {"pages": [page], "parameters": {"thinking": "disabled"}}
        record = {"page": 35, "split": "development", "dataset_sha256": "hash", "parameters": dataset["parameters"],
                  "model": MODEL, "backend": "probe", "prompt_version": PROMPT_VERSION, "protocol_version": PROTOCOL_VERSION,
                  "status": "success_with_results", "error": "", "http_status": 200,
                  "diagnostics": {"finish_reason": "stop"}, "predictions": {"triples": [triple], "unresolved": [{"reason": "condition_missing"}]}}
        checked = review_probe(dataset, "hash", record)
        self.assertTrue(checked["technical_probe_passed"])
        self.assertTrue(checked["evidence_validation_passed"])
        self.assertEqual(checked["candidate_review_count"], 1)
        truncated = {**record, "diagnostics": {"finish_reason": "length"}, "error": "truncated"}
        self.assertFalse(review_probe(dataset, "hash", truncated)["probe_passed"])
        invalid = {**record, "error": "invalid_protocol"}
        self.assertFalse(review_probe(dataset, "hash", invalid)["probe_passed"])
        for field, value in (("model", "other-model"), ("dataset_sha256", "other"),
                             ("protocol_version", "other"), ("page", 37),
                             ("parameters", {"thinking": "enabled"})):
            with self.subTest(field=field):
                self.assertFalse(review_probe(dataset, "hash", {**record, field: value})["probe_passed"])
        triple["relation_evidence"][0]["start"] = 1
        self.assertFalse(review_probe(dataset, "hash", record)["probe_passed"])

    def test_extended_configuration_preserves_frozen_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original"
            original.mkdir()
            dataset = {"pages": [self.page], "parameters": {"max_tokens": 4096}}
            (original / "fixed_dataset.json").write_text(json.dumps(dataset), encoding="utf8")
            destination = root / "extended"
            with patch("tools.llm_eval_step5.DEFAULT", original), patch("sys.argv", ["eval", "prepare-extended", "--dir", str(destination)]):
                main()
            revised = json.loads((destination / "fixed_dataset.json").read_text(encoding="utf8"))
            self.assertEqual(revised["pages"], dataset["pages"])
            self.assertEqual((revised["parameters"]["max_tokens"], revised["parameters"]["timeout"]), (16384, 180))
            self.assertIn("derived_from_dataset_sha256", revised)


if __name__ == "__main__":
    unittest.main()
