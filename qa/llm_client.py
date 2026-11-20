# -*- coding: utf-8 -*-
"""
qa/llm_client.py —— 本机大模型调用客户端（Ollama 的 OpenAI 兼容 /api/chat）

只允许指向 127.0.0.1 / localhost（复用 LlmConfig.is_local() 硬拦截），
用标准库 urllib 调用，不引入额外依赖；调用失败一律返回 None，
由 qa/generate.py 回退到抽取式原文摘录，绝不因为模型不可用而伪造输出。

与上一阶段「生成链路未实现」的空桩不同，这里实现的是真·LLM 生成：
  1. 把已验证的证据（quote + 出处）与问题一起喂给模型；
  2. prompt 硬约束「只能基于给定证据作答，禁止编造、禁止给诊疗建议」；
  3. 拿到模型输出后，还要让每条被引用的原文在证据里能命中，命中失败即弃用。

安全边界：
  - 越界问题（个人诊疗/疗效/剂量）在 refuse.should_refuse 已先拦截，不进入这里；
  - 这里只负责「有证据时的生成」，不改变任何拒答判定。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import List, Optional

from app.config import CONFIG
from app.schemas import Evidence

logger = logging.getLogger("tcm.llm")


def _chat_endpoint() -> str:
    """构造 Ollama /api/chat 地址（复用 LlmConfig.base_url）。"""
    base = (CONFIG.llm.base_url or "").rstrip("/")
    # 兼容两种情况：用户可能填了 /v1 或没填
    if base.endswith("/v1"):
        return base[:-3] + "/api/chat"
    return base + "/api/chat"


def _system_prompt() -> str:
    return (
        "你是一名中医古籍文献辅助检索助手。你只能依据用户提供的【原文证据】来回答，"
        "不得编造、不得补充证据之外的信息，不得给出任何诊断、用药、疗效或剂量建议。"
        "回答要简洁、忠于原文，并在关键结论后用【出处】标注对应的证据编号。"
        "如果证据不足以回答，就如实说『依据不足』，不要猜测。"
    )


def _build_user_prompt(query: str, evidences: List[Evidence]) -> str:
    lines = [f"问题：{query}", "", "以下是可用的原文证据："]
    for i, e in enumerate(evidences, start=1):
        loc = f"{e.page_id} / {e.line_id}"
        lines.append(f"[{i}] 原文：{e.quote}（出处：{loc}）")
    lines.append("")
    lines.append("请仅依据以上证据回答上面的问题。")
    return "\n".join(lines)


def chat_messages(messages: List[dict],
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None,
                  timeout: Optional[float] = None) -> Optional[str]:
    """通用对话入口：直接传 OpenAI 风格 messages，供 RAG 提示词使用。

    失败（超时 / 连接失败 / 模型缺失 / 返回空）一律返回 None，由上层降级，
    绝不返回伪造文本。
    """
    if not CONFIG.llm.enabled:
        return None
    if not messages:
        return None
    payload = {
        "model": CONFIG.llm.model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": float(CONFIG.llm.temperature if temperature is None else temperature),
            "num_predict": int(CONFIG.llm.max_tokens if max_tokens is None else max_tokens),
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _chat_endpoint(), data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
                req, timeout=float(CONFIG.llm.timeout if timeout is None else timeout)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("本机 LLM 调用失败：%s", type(exc).__name__)
        return None
    text = ((body.get("message") or {}).get("content") or "").strip()
    return text or None


def chat(query: str, evidences: List[Evidence]) -> Optional[str]:
    """调用本机 Ollama 生成一句话式回答；失败返回 None。

    超时、连接失败、模型缺失等都吞掉并返回 None，调用方据此回退。
    """
    if not CONFIG.llm.enabled:
        return None
    if not (evidences or []):
        return None

    payload = {
        "model": CONFIG.llm.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _build_user_prompt(query, evidences)},
        ],
        "options": {
            "temperature": float(CONFIG.llm.temperature),
            "num_predict": int(CONFIG.llm.max_tokens),
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _chat_endpoint(),
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=float(CONFIG.llm.timeout)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("本机 LLM 调用失败：%s", type(exc).__name__)
        return None

    # Ollama /api/chat 返回 {"message": {"content": "..."}}
    msg = body.get("message") or {}
    text = (msg.get("content") or "").strip()
    return text or None


def llm_available() -> bool:
    """轻量探活：只判断配置是否启用，不实际调用模型（避免每次问答都发探测请求）。"""
    return CONFIG.llm.enabled