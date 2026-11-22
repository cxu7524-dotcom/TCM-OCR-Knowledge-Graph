# -*- coding: utf-8 -*-
"""
extract/llm_local.py —— LLM HTTP 适配层（默认关闭）

用途：后期用本机 Ollama / vLLM 补足规则抽取搞不定的部分（跨行主语、隐含关系、
      异文考据）。本阶段**默认后端 none，不加载模型、不发任何请求**。

cloud 模式通过 HTTPS 调用兼容接口。抽取入口仍为桩。

TODO（下一阶段）：
  - 设计抽取 prompt 与 JSON Schema 约束输出（含 page_id / line_id / quote）；
  - 实现 Few-shot 示例与温度 / 超时 / 重试策略；
  - 与 rules.py 结果做冲突消解并写入 trace，供 eval/audit.py 审计。
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List

from app.config import CONFIG
from app.schemas import Block, Entity, Triple


@dataclass
class ExtractionResult:
    """LLM 抽取产出。桩状态下一律为空集合 + 明确标注。"""

    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    backend: str = "none"
    model_loaded: bool = False
    is_demo: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class ChatResult:
    content: str = ""
    error: str = ""
    status: int = 0
    attempts: int = 0
    finish_reason: str | None = None
    response_model: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    content_length: int | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalLLMExtractor:
    """本地 OpenAI 兼容 HTTP 客户端（本阶段仅提供骨架，不默认启用）。"""

    def __init__(self) -> None:
        self.cfg = CONFIG.llm

    def available(self) -> bool:
        """后端启用 + 地址为本机，才认为可用。"""
        return self.cfg.enabled

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """调用本机 /chat/completions；任何异常返回空串，绝不向外抛。

        注意：本阶段 extract() 不会调用它，仅作为下一阶段的接口占位。
        """
        return self.chat_result(messages).content

    def chat_result(self, messages: List[Dict[str, str]], schema: dict | None = None) -> ChatResult:
        """调用兼容接口并返回安全的错误类别。"""
        if not self.available():
            return ChatResult(error="configuration")
        cloud = self.cfg.backend.lower() == "cloud"
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if cloud:
            thinking = self.cfg.thinking.strip().lower()
            if thinking not in ("", "enabled", "disabled"):
                return ChatResult(error="configuration")
            if thinking:
                body["thinking"] = {"type": thinking}
            if self.cfg.response_format == "json_schema" and schema is not None:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "extraction", "strict": True, "schema": schema}}
            elif self.cfg.response_format == "json_object":
                body["response_format"] = {"type": "json_object"}
            else:
                return ChatResult(error="configuration")
            headers["Authorization"] = "Bearer " + os.environ.get("TCM_LLM_API_KEY", "")
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        base = self.cfg.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        req = urllib.request.Request(
            url, data=payload, headers=headers, method="POST")
        for attempt in range(1, 4):
            status = 0
            result = ChatResult(attempts=attempt)
            try:
                send = urllib.request.build_opener(_NoRedirect()).open if cloud else urllib.request.urlopen
                with send(req, timeout=self.cfg.timeout) as resp:
                    status = resp.status
                    data = json.loads(resp.read().decode("utf-8"))
                result.status = status
                if isinstance(data, dict):
                    model = data.get("model")
                    secret = os.environ.get("TCM_LLM_API_KEY", "")
                    if isinstance(model, str) and len(model) <= 256 and not (secret and secret in model):
                        result.response_model = model
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        completion = usage.get("completion_tokens")
                        details = usage.get("completion_tokens_details")
                        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
                        if type(completion) is int and completion >= 0:
                            result.completion_tokens = completion
                        if type(reasoning) is int and reasoning >= 0:
                            result.reasoning_tokens = reasoning
                choice = data["choices"][0]
                reason = choice.get("finish_reason")
                if reason in ("stop", "length", "content_filter", "tool_calls", "insufficient_system_resource"):
                    result.finish_reason = reason
                message = choice.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    result.content_length = len(content)
                if choice.get("finish_reason") == "length":
                    result.error = "truncated"
                    return result
                if not isinstance(content, str):
                    raise ValueError
                if cloud and not isinstance(json.loads(content), dict):
                    raise ValueError
                result.content = content
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status in (401, 403):
                    error = "authentication"
                elif status == 404:
                    error = "model_unavailable"
                elif status == 429:
                    error = "rate_limit"
                elif status >= 500:
                    error = "server_error"
                else:
                    error = "http_error"
                retry = status in (429, 500, 502, 503, 504)
            except (TimeoutError, socket.timeout):
                error, retry = "timeout", True
            except urllib.error.URLError as exc:
                error = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "connection_error"
                retry = True
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, UnicodeError):
                error, retry = "invalid_response", False
            except OSError:
                error, retry = "connection_error", True
            if not retry or attempt == 3:
                result.error = error
                result.status = status
                return result
            time.sleep(0.2 * attempt)

    def extract(self, blocks: List[Block]) -> ExtractionResult:
        """抽取入口（桩）。

        输入：Block 列表；输出：ExtractionResult
        当前实现：返回空结果 + 「本地 LLM 未接入」说明，规则抽取仍由 rules.py 承担。
        """
        return ExtractionResult(
            entities=[], triples=[],
            backend=self.cfg.backend if self.cfg.backend else "none",
            model_loaded=False, is_demo=True,
            notes=[
                CONFIG.demo_badge,
                "本地 LLM 未接入：本阶段实体与关系完全来自 extract/rules.py 的确定性规则",
                "切换方式：设 TCM_LLM_BACKEND=ollama 并启动本机 Ollama 服务",
            ],
        )


def extract_with_llm(blocks: List[Block]) -> ExtractionResult:
    """模块级便捷入口，不抛异常。"""
    try:
        return LocalLLMExtractor().extract(blocks)
    except Exception as exc:
        return ExtractionResult(notes=[f"LLM 抽取降级：{type(exc).__name__}"])
