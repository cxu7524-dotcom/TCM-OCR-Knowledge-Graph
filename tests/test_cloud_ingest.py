import asyncio
import copy
import http.client
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image

from app import main, store
from app.schemas import Block, Entity, OCRResult, Triple
from extract import llm_local
from extract.llm_extraction import PROMPT_VERSION
from graph.build import build_graph
from graph.query import run_query


def make_ocr():
    texts = ["桂枝汤", "甘草二钱。"] + [f"续文{i}。" for i in range(3, 10)] + ["主治风寒。"]
    blocks = [Block(page_id="integration", line_id=f"L{i}", block_ref=f"integration:L{i}",
                    reading_order=i, text=text, bbox=[i, 2, i + 2, 4], source_hash=f"block-{i}")
              for i, text in enumerate(texts, 1)]
    return OCRResult(page_id="integration", blocks=list(reversed(blocks)), is_demo=True,
                     source_hash="test-page", engine="test", model_loaded=True)


def output():
    def ev(ref, quote):
        return [{"block_ref": f"integration:{ref}", "quote": quote}]
    return {"entities": [
                {"id": "a", "name": "桂枝汤", "type": "方剂", "evidence": ev("L1", "桂枝汤")},
                {"id": "b", "name": "风寒", "type": "主治", "evidence": ev("L10", "风寒")}],
            "relations": [{"subject_id": "a", "object_id": "b", "type": "主治", "entry_anchor_id": "a",
                           "subject_evidence": ev("L1", "桂枝汤"), "relation_evidence": ev("L10", "主治风寒"), "qualifier": ""}],
            "unresolved": [{"evidence": ev("L2", "甘草二钱"), "reason": "人工核对", "candidate_subject_ids": ["a"]}]}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, self.headers.get("Authorization"), json.loads(body)))
        status, content = self.server.outcome
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if status == 200:
            data = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content, ensure_ascii=False)}}]}
        else:
            data = {"error": "sensitive server response"}
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, *args):
        pass


class LocalOpener:
    def __init__(self, port):
        self.port = port

    def open(self, request, timeout):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", request.data, dict(request.header_items()))
        response = conn.getresponse()
        if response.status >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status, "mock", response.headers, response)
        return response


async def asgi_request(method, path, body=b"", content_type="application/json", query=b""):
    messages = []
    received = False
    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}
    async def send(message):
        messages.append(message)
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": query,
             "headers": [(b"content-type", content_type.encode())], "server": ("test", 80), "client": ("test", 1), "root_path": ""}
    await main.app(scope, receive, send)
    return messages[0]["status"], json.loads(b"".join(m.get("body", b"") for m in messages))


class CloudIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        paths = SimpleNamespace(store_dir=root, db_path=root / "test.sqlite", ensure_dirs=lambda: None)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests = []
        self.server.outcome = (200, output())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        uploads = root / "uploads"
        uploads.mkdir()
        cfg = replace(main.CONFIG, paths=replace(main.CONFIG.paths, uploads_dir=uploads),
                      llm=replace(main.CONFIG.llm, backend="cloud", extract_backend="cloud",
                                  base_url="https://example.invalid/v1", model="mock-model", response_format="json_object"))
        def fake_ocr(path, page_id):
            result = make_ocr()
            result.is_demo = False
            result.image_path = path
            return result
        for item in [patch.object(store, "PATHS", paths), patch.object(main, "CONFIG", cfg), patch.object(llm_local, "CONFIG", cfg),
                     patch.dict(os.environ, {"TCM_LLM_API_KEY": "temporary-test-key"}),
                     patch.object(main, "get_adapter", return_value=(SimpleNamespace(name="test"), [])),
                     patch.object(main, "run_ocr", side_effect=fake_ocr),
                     patch.object(main, "export_graph_json"),
                     patch("urllib.request.build_opener", side_effect=lambda *args: LocalOpener(self.server.server_port)),
                     patch.dict(main._STATE, {"page_id": None, "graph": None}, clear=False)]:
            item.start()
            self.addCleanup(item.stop)

    def request(self, method, path, body=b"", content_type="application/json", query=b""):
        return asyncio.run(asgi_request(method, path, body, content_type, query))

    def upload(self):
        image = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(image, format="PNG")
        boundary = b"integration-boundary"
        body = (b"--" + boundary + b'\r\nContent-Disposition: form-data; name="page_id"\r\n\r\nintegration\r\n'
                + b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\n'
                + b"Content-Type: image/png\r\n\r\n" + image.getvalue() + b"\r\n--" + boundary + b"--\r\n")
        return self.request("POST", "/api/pages", body, "multipart/form-data; boundary=integration-boundary")

    def test_upload_reload_qa_failure_empty_and_retry(self):
        status, result = self.upload()
        self.assertEqual(status, 200)
        self.assertEqual(result["extraction"]["status"], "success_with_relations")
        self.assertFalse(result["is_demo"])
        self.assertFalse(store.get_triples("integration")[0].is_demo)
        path, auth, request = self.server.requests[-1]
        self.assertEqual((path, auth), ("/v1/chat/completions", "Bearer " + os.environ["TCM_LLM_API_KEY"]))
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual([x["reading_order"] for x in payload["blocks"]], list(range(1, 11)))
        self.assertEqual(len(store.get_triples("integration")), 1)
        status, answer = self.request("POST", "/api/qa", json.dumps({"page_id": "integration", "question": "风寒用什么方剂"}).encode())
        self.assertEqual(status, 200)
        self.assertFalse(answer["refused"])
        self.assertEqual(answer["objects"][0]["head"], "桂枝汤")
        self.assertEqual({e["line_id"] for e in answer["evidences"]}, {"L1", "L10"})
        self.server.outcome = (401, {})
        status, failure = self.upload()
        self.assertEqual(status, 503)
        self.assertNotIn(os.environ["TCM_LLM_API_KEY"], json.dumps(failure))
        self.assertNotIn("sensitive server response", json.dumps(failure))
        self.assertEqual(len(store.get_triples("integration")), 1)
        self.server.outcome = (200, "invalid protocol")
        self.assertEqual(self.upload()[0], 503)
        self.assertEqual(len(store.get_triples("integration")), 1)
        main._STATE.update({"page_id": None, "graph": None, "ocr": None})
        status, restored = self.request("GET", "/api/pages/integration/ocr")
        self.assertEqual(restored["extraction"]["model"], "mock-model")
        self.assertEqual(restored["extraction"]["prompt_version"], PROMPT_VERSION)
        self.assertEqual((len(restored["triples"]), restored["unresolved_count"]), (1, 1))
        self.assertEqual(restored["triples"], [t.to_dict() for t in store.get_triples("integration")])
        self.assertEqual(restored["entities"], [e.to_dict() for e in store.get_entities("integration")])
        self.assertNotIn(os.environ["TCM_LLM_API_KEY"], json.dumps(restored, ensure_ascii=False))
        self.assertEqual(len(self.request("GET", "/api/graph", query=b"page_id=integration")[1]["links"]), 1)
        self.assertFalse(self.request("POST", "/api/qa", json.dumps({"page_id": "integration", "question": "风寒用什么方剂"}).encode())[1]["refused"])
        empty = {"entities": output()["entities"][:1], "relations": [], "unresolved": []}
        self.server.outcome = (200, empty)
        status, response = self.upload()
        self.assertEqual(response["extraction"]["status"], "success_no_relations")
        self.assertEqual(store.get_triples("integration"), [])
        main._STATE.update({"page_id": None, "graph": None, "ocr": None})
        self.assertEqual(self.request("GET", "/api/pages/integration/ocr")[1]["triples"], [])
        self.assertEqual(self.request("GET", "/api/graph", query=b"page_id=integration")[1]["links"], [])
        self.assertTrue(self.request("POST", "/api/qa", json.dumps({"page_id": "integration", "question": "风寒用什么方剂"}).encode())[1]["refused"])
        self.server.outcome = (200, output())
        self.assertEqual(self.upload()[0], 200)
        self.assertEqual(self.upload()[0], 200)
        self.assertEqual(len(store.get_triples("integration")), 1)
        self.assertEqual(len(store.get_unresolved("integration")), 1)

    def test_explicit_rules_mode(self):
        cfg = replace(main.CONFIG, llm=replace(main.CONFIG.llm, extract_backend="rules"))
        with patch.object(main, "CONFIG", cfg):
            status, response = self.upload()
        self.assertEqual(status, 200)
        self.assertEqual(response["extraction"]["mode"], "rules")
        self.assertEqual(self.server.requests, [])

    def test_demo_flag_follows_ocr(self):
        with patch.object(main, "run_demo", side_effect=lambda *_: make_ocr()):
            status, response = self.request("POST", "/api/pages", b"demo=1&page_id=integration", "application/x-www-form-urlencoded")
        self.assertEqual(status, 200)
        self.assertTrue(response["is_demo"])
        self.assertTrue(store.get_triples("integration")[0].is_demo)

    def test_disabled_thinking_reaches_mock_http_endpoint(self):
        cfg = replace(main.CONFIG, llm=replace(main.CONFIG.llm, thinking="disabled"))
        with patch.object(main, "CONFIG", cfg), patch.object(llm_local, "CONFIG", cfg):
            status, _ = self.upload()
        self.assertEqual(status, 200)
        self.assertEqual(self.server.requests[-1][2]["thinking"], {"type": "disabled"})

    def test_formula_query_retains_scope_and_qualifier(self):
        entities = [Entity(name="桂枝汤", type="方剂", entry_id="one"),
                    Entity(name="桂枝汤", type="方剂", entry_id="two")]
        triples = [Triple(page_id="p", head="桂枝汤", relation="组成", tail="杏仁", entry_id="one", qualifier="若咳嗽，加"),
                   Triple(page_id="p", head="甘草", relation="剂量", tail="三钱", entry_id="one", qualifier="原文剂量"),
                   Triple(page_id="p", head="甘草", relation="剂量", tail="一钱", entry_id="two", qualifier="原文剂量")]
        result = run_query("formula", "桂枝汤", triples, page_id="p", entities=entities)
        self.assertEqual({(r["entry_id"], r["tail"], r["qualifier"]) for r in result["results"]},
                         {("one", "杏仁", "若咳嗽，加"), ("one", "三钱", "原文剂量"), ("two", "一钱", "原文剂量")})
        links = build_graph(triples, entities).to_vis_payload()["links"]
        self.assertEqual(links[0]["qualifier"], "若咳嗽，加")
        self.assertNotEqual(links[1]["source"], links[2]["source"])


if __name__ == "__main__":
    unittest.main()
