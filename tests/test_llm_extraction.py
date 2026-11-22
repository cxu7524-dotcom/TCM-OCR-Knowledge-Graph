import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.config import CONFIG
from app.schemas import Block
from extract.llm_extraction import build_input, build_messages, extract_candidates, parse_output
from extract.llm_local import LocalLLMExtractor
from tools.llm_extraction_six_cases import CASES, judge, make_blocks


class MockResponse:
    status = 200
    def __init__(self, content):
        self.content = content
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": self.content}}]}).encode()


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.blocks = make_blocks("page", ["桂枝汤", "桂枝三钱，甘草二钱。", "主治风寒。"])
        self.targets = [self.blocks[-1].block_ref]

    def test_input_order_context_and_protocol(self):
        payload = build_input(reversed(self.blocks), self.targets)
        self.assertEqual([x["reading_order"] for x in payload["blocks"]], [1, 2, 3])
        self.assertEqual([x["scope"] for x in payload["blocks"]], ["context", "context", "target"])
        self.assertEqual(payload["blocks"][0]["text"], "桂枝汤")
        self.assertIn("主治", payload["relation_types"])
        self.assertIn("方剂", payload["entity_types"])
        self.assertEqual(json.loads(build_messages(payload)[1]["content"]), payload)
        with self.assertRaises(ValueError):
            build_input(self.blocks, ["missing"])

    def test_empty_and_invalid_outputs(self):
        empty = {"entities": [], "relations": [], "unresolved": []}
        self.assertEqual(parse_output(json.dumps(empty)), empty)
        for bad in ({"entities": [], "relations": []}, {"entities": {}, "relations": [], "unresolved": []},
                    {"entities": [], "relations": [{"subject_id": "x", "object_id": "y", "type": "主治", "entry_anchor_id": "x", "subject_evidence": [], "relation_evidence": [], "qualifier": ""}], "unresolved": []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_output(json.dumps(bad))

    def test_valid_ids_and_unknown_reference(self):
        evidence = [{"block_ref": "page:L1", "quote": "桂枝汤"}]
        entities = [{"id": "a", "name": "桂枝汤", "type": "方剂", "evidence": evidence},
                    {"id": "b", "name": "风寒", "type": "主治", "evidence": evidence}]
        relation = {"subject_id": "a", "object_id": "b", "type": "主治", "entry_anchor_id": "a",
                    "subject_evidence": evidence, "relation_evidence": evidence, "qualifier": ""}
        output = {"entities": entities, "relations": [relation], "unresolved": []}
        self.assertEqual(parse_output(json.dumps(output)), output)
        relation["object_id"] = "missing"
        with self.assertRaises(ValueError):
            parse_output(json.dumps(output))

    def test_adapter_request_and_parse(self):
        empty = {"entities": [], "relations": [], "unresolved": []}
        client = LocalLLMExtractor()
        client.cfg = replace(CONFIG.llm, backend="cloud", base_url="https://example.invalid/v1", model="example")
        with patch.dict(os.environ, {"TCM_LLM_API_KEY": "simulated"}), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = MockResponse(json.dumps(empty))
            result = extract_candidates(self.blocks, self.targets, client)
            self.assertEqual(result["output"], empty)
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(json.loads(body["messages"][1]["content"])["target_block_refs"], self.targets)
            opener.return_value.open.return_value = MockResponse('{"entities": []}')
            self.assertEqual(extract_candidates(self.blocks, self.targets, client)["error"], "invalid_protocol")
            client.cfg = replace(client.cfg, response_format="json_schema")
            opener.return_value.open.return_value = MockResponse(json.dumps(empty))
            self.assertEqual(extract_candidates(self.blocks, self.targets, client)["output"], empty)
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_protocol_failure_keeps_usage_without_reasoning(self):
        client = LocalLLMExtractor()
        client.cfg = replace(CONFIG.llm, backend="cloud", base_url="https://example.invalid/v1", model="example")
        response = MockResponse("")
        response.read = lambda: json.dumps({"model": "actual", "usage": {"completion_tokens": 42,
                                            "completion_tokens_details": {"reasoning_tokens": 30}},
                                           "choices": [{"finish_reason": "stop", "message": {
                                               "content": '{"entities":[]}', "reasoning_content": "do-not-save"}}]}).encode()
        with patch.dict(os.environ, {"TCM_LLM_API_KEY": "simulated"}), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = response
            result = extract_candidates(self.blocks, self.targets, client)
        self.assertEqual(result["error"], "invalid_protocol")
        self.assertEqual((result["diagnostics"]["completion_tokens"], result["diagnostics"]["reasoning_tokens"]), (42, 30))
        self.assertNotIn("do-not-save", repr(result))

    def test_six_cases_are_available_and_do_not_accept_empty(self):
        self.assertEqual(len(CASES), 6)
        for case_id, lines, expected in CASES:
            self.assertTrue(lines and expected)
            self.assertFalse(judge(case_id, {"entities": [], "relations": [], "unresolved": []}))

    def test_shared_dose_crosses_columns_and_rejects_extra_dose(self):
        _, lines, _ = next(case for case in CASES if case[0] == "shared_dose")
        blocks = make_blocks("shared_dose", lines)
        payload = build_input(blocks, [block.block_ref for block in blocks[1:]])
        self.assertEqual([block["text"] for block in payload["blocks"]], ["桂枝汤", "桂枝、", "芍药各三钱，甘草一钱。"])
        self.assertNotEqual(payload["blocks"][1]["col_id"], payload["blocks"][2]["col_id"])
        evidence = [{"block_ref": blocks[1].block_ref, "quote": "桂枝"}]
        names = ["桂枝汤", "桂枝", "芍药", "甘草", "三钱", "一钱"]
        types = ["方剂", "药材", "药材", "药材", "剂量", "剂量"]
        entities = [{"id": str(i), "name": name, "type": kind, "evidence": evidence}
                    for i, (name, kind) in enumerate(zip(names, types))]
        def dose(subject, obj):
            return {"subject_id": str(subject), "object_id": str(obj), "type": "剂量", "entry_anchor_id": "0",
                    "subject_evidence": evidence, "relation_evidence": evidence, "qualifier": ""}
        output = {"entities": entities, "relations": [dose(1, 4), dose(2, 4), dose(3, 5)], "unresolved": []}
        self.assertTrue(judge("shared_dose", parse_output(json.dumps(output))))
        output["relations"].append(dose(3, 4))
        self.assertFalse(judge("shared_dose", parse_output(json.dumps(output))))

    def test_conditional_rejects_unconditional_composition_even_with_unresolved(self):
        evidence = [{"block_ref": "conditional:L2", "quote": "若咳嗽，加杏仁二钱"}]
        entities = [{"id": "formula", "name": "桂枝汤", "type": "方剂", "evidence": evidence},
                    {"id": "herb", "name": "杏仁", "type": "药材", "evidence": evidence}]
        relation = {"subject_id": "formula", "object_id": "herb", "type": "组成", "entry_anchor_id": "formula",
                    "subject_evidence": evidence, "relation_evidence": evidence, "qualifier": "若咳嗽"}
        unresolved = {"evidence": evidence, "reason": "待核", "candidate_subject_ids": ["formula"]}
        output = {"entities": entities, "relations": [relation], "unresolved": [unresolved]}
        self.assertTrue(judge("conditional", parse_output(json.dumps(output))))
        output["relations"].append({**relation, "qualifier": ""})
        self.assertFalse(judge("conditional", parse_output(json.dumps(output))))
        output["relations"] = []
        self.assertTrue(judge("conditional", parse_output(json.dumps(output))))
        output["relations"] = [{**relation, "qualifier": "基础配伍"}]
        self.assertFalse(judge("conditional", parse_output(json.dumps(output))))


if __name__ == "__main__":
    unittest.main()
