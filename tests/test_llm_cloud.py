import io
import asyncio
import json
import logging
import os
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from unittest.mock import patch

from app.config import CONFIG
from extract.llm_local import LocalLLMExtractor


class Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.body).encode()


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TCM_LLM_API_KEY": "fake-key-for-test"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = LocalLLMExtractor()
        self.client.cfg = replace(CONFIG.llm, backend="cloud", base_url="https://example.invalid/v1/", model="model-id", thinking="")

    def test_request_and_response(self):
        response = Response({"choices": [{"finish_reason": "stop", "message": {"content": '{"entities": []}'}}]})
        with patch("urllib.request.build_opener") as opener:
            send = opener.return_value.open
            send.return_value = response
            result = self.client.chat_result([{"role": "user", "content": "short sample"}])
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + os.environ["TCM_LLM_API_KEY"])
        body = json.loads(request.data)
        self.assertEqual(body["model"], "model-id")
        self.assertNotIn("thinking", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(json.loads(result.content), {"entities": []})
        self.assertEqual(result.status, 200)

    def test_thinking_option_and_usage_diagnostics(self):
        response = Response({"model": "actual-model", "usage": {"completion_tokens": 123,
                            "completion_tokens_details": {"reasoning_tokens": 100}},
                            "choices": [{"finish_reason": "stop", "message": {"content": "{}", "reasoning_content": "private-reasoning"}}]})
        for thinking in ("disabled", "enabled"):
            self.client.cfg = replace(self.client.cfg, thinking=thinking)
            with self.subTest(thinking=thinking), patch("urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = response
                result = self.client.chat_result([])
                body = json.loads(opener.return_value.open.call_args.args[0].data)
                self.assertEqual(body["thinking"], {"type": thinking})
                self.assertEqual((result.finish_reason, result.response_model, result.completion_tokens,
                                  result.reasoning_tokens, result.content_length), ("stop", "actual-model", 123, 100, 2))
                self.assertNotIn("private-reasoning", repr(result))
        self.client.cfg = replace(self.client.cfg, thinking="invalid")
        with patch("urllib.request.build_opener") as opener:
            self.assertEqual(self.client.chat_result([]).error, "configuration")
            opener.assert_not_called()

    def test_truncation_and_invalid_content_keep_diagnostics(self):
        for content, finish, expected in [("{", "length", "truncated"), ("not json", "stop", "invalid_response")]:
            response = Response({"model": "actual-model", "usage": {"completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 4}},
                                 "choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": "secret-reasoning"}}]})
            with self.subTest(expected=expected), patch("urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = response
                result = self.client.chat_result([])
                self.assertEqual((result.error, result.finish_reason, result.response_model, result.completion_tokens,
                                  result.reasoning_tokens, result.content_length), (expected, finish, "actual-model", 5, 4, len(content)))
                self.assertEqual(result.content, "")
                self.assertNotIn("secret-reasoning", repr(result))
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = Response({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
            result = self.client.chat_result([])
            self.assertIsNone(result.response_model)
            self.assertIsNone(result.completion_tokens)
            self.assertIsNone(result.reasoning_tokens)

    def test_provider_bad_request_is_not_retried(self):
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = urllib.error.HTTPError("https://example.invalid", 400, "", {}, io.BytesIO())
            result = self.client.chat_result([])
            self.assertEqual((result.error, result.status, result.attempts), ("http_error", 400, 1))
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_errors_and_retries(self):
        for status, expected, attempts in [(401, "authentication", 1), (403, "authentication", 1), (404, "model_unavailable", 1), (429, "rate_limit", 3), (503, "server_error", 3)]:
            with self.subTest(status=status), patch("time.sleep"), patch("urllib.request.build_opener") as opener:
                send = opener.return_value.open
                send.side_effect = urllib.error.HTTPError("https://example.invalid", status, "", {}, io.BytesIO())
                result = self.client.chat_result([])
                self.assertEqual((result.error, result.attempts, send.call_count), (expected, attempts, attempts))
                self.assertNotIn(os.environ["TCM_LLM_API_KEY"], repr(result))
        with patch("time.sleep"), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError
            result = self.client.chat_result([])
            self.assertEqual((result.error, result.attempts), ("timeout", 3))
        for content, reason, expected in [("bad", "stop", "invalid_response"), ("{", "length", "truncated")]:
            response = Response({"choices": [{"finish_reason": reason, "message": {"content": content}}]})
            with patch("urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = response
                self.assertEqual(self.client.chat_result([]).error, expected)

    def test_config_does_not_expose_key(self):
        cfg = replace(CONFIG, llm=self.client.cfg)
        self.assertNotIn(os.environ["TCM_LLM_API_KEY"], repr(cfg.summary()))
        self.assertNotIn(os.environ["TCM_LLM_API_KEY"], repr(cfg.to_dict()))

    def test_call_output_does_not_expose_key(self):
        secret = os.environ["TCM_LLM_API_KEY"]
        response = Response({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
        cases = [
            ("success", [response], "", 1),
            ("authentication", [urllib.error.HTTPError("https://example.invalid", 401, secret, {}, io.BytesIO(secret.encode()))], "authentication", 1),
            ("retry", [urllib.error.HTTPError("https://example.invalid", 429, secret, {}, io.BytesIO(secret.encode())) for _ in range(3)], "rate_limit", 3),
        ]
        for name, outcomes, expected_error, attempts in cases:
            with self.subTest(name=name):
                captured = io.StringIO()
                handler = logging.StreamHandler(captured)
                root_logger = logging.getLogger()
                previous_level = root_logger.level
                root_logger.setLevel(logging.DEBUG)
                root_logger.addHandler(handler)
                try:
                    with redirect_stdout(captured), redirect_stderr(captured), patch("time.sleep"), patch("urllib.request.build_opener") as opener:
                        opener.return_value.open.side_effect = outcomes
                        result = self.client.chat_result([{"role": "user", "content": "short sample"}])
                        handler.flush()
                    self.assertEqual((result.error, result.attempts), (expected_error, attempts))
                    self.assertEqual(opener.return_value.open.call_count, attempts)
                    self.assertNotIn(secret, captured.getvalue())
                    self.assertNotIn(secret, repr(result))
                finally:
                    root_logger.removeHandler(handler)
                    root_logger.setLevel(previous_level)
                    handler.close()

    def test_schema_and_health(self):
        self.client.cfg = replace(self.client.cfg, response_format="json_schema")
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        response = Response({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = response
            self.client.chat_result([], schema)
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(body["response_format"]["json_schema"]["schema"], schema)
        from app import main
        async def get_response(endpoint):
            events = []
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}
            async def send(event):
                events.append(event)
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "GET", "scheme": "http", "path": endpoint, "raw_path": endpoint.encode(), "query_string": b"", "headers": [], "server": ("test", 80), "client": ("test", 1234), "root_path": ""}
            await main.app(scope, receive, send)
            return events
        with patch.object(main, "CONFIG", replace(CONFIG, llm=self.client.cfg)):
            for endpoint in ("/api/health", "/api/meta", "/"):
                events = asyncio.run(get_response(endpoint))
                self.assertEqual(events[0]["status"], 200)
                body = b"".join(e.get("body", b"") for e in events)
                self.assertTrue(body)
                self.assertNotIn(os.environ["TCM_LLM_API_KEY"].encode(), body)


if __name__ == "__main__":
    unittest.main()
