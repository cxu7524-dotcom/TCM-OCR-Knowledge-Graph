# -*- coding: utf-8 -*-
"""
extract/llm_local.py —— 本地 LLM 抽取适配层（默认关闭）

用途：后期用本机 Ollama / vLLM 补足规则抽取搞不定的部分（跨行主语、隐含关系、
      异文考据）。本阶段**默认后端 none，不加载模型、不发任何请求**。

安全约束：
  - 只接受 127.0.0.1 / localhost 的 OpenAI 兼容地址；非本机直接拒绝；
  - 不读取任何密钥，不访问公网；调用失败一律降级，不抛异常、不伪造抽取结果；
  - 模型输出必须经 app/schemas.py 校验并保留原文 quote，否则丢弃该条。

TODO（下一阶段）：
  - 设计抽取 prompt 与 JSON Schema 约束输出（含 page_id / line_id / quote）；
  - 实现 Few-shot 示例与温度 / 超时 / 重试策略；
  - 与 rules.py 结果做冲突消解并写入 trace，供 eval/audit.py 审计。
"""

from __future__ import annotations

import json
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


class LocalLLMExtractor:
    """本地 OpenAI 兼容 HTTP 客户端（本阶段仅提供骨架，不默认启用）。"""

    def __init__(self) -> None:
        self.cfg = CONFIG.llm

    def available(self) -> bool:
        """后端启用 + 地址为本机，才认为可用。"""
        return self.cfg.enabled and self.cfg.is_local()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """调用本机 /chat/completions；任何异常返回空串，绝不向外抛。

        注意：本阶段 extract() 不会调用它，仅作为下一阶段的接口占位。
        """
        if not self.available():
            return ""
        if not self.cfg.is_local():
            return ""  # 硬性拦截非本机地址
        payload = json.dumps({
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TimeoutError):
            return ""

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
