# -*- coding: utf-8 -*-
"""
qa/index.py —— 本地向量索引（占位实现，默认关闭）

本阶段**不是语义向量**：用标准库把字符 bigram 哈希成定长向量（hashing trick），
只用于把「稀疏 + 可选向量」的混合检索链路打通，不具备语义泛化能力，
**不得用于任何指标实测**。

TODO（下一阶段）：
  - 接入本地 embedding 模型（本机推理，禁止云端），记录模型名 / 版本 / 维度；
  - 换成持久化向量索引（SQLite 表或本地文件），支持增量写入与幂等重建；
  - 与稀疏通道做统一归一化，避免两路分数量纲不一致。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.config import CONFIG
from app.schemas import Block, norm_for_match


def _bigrams(text: str) -> List[str]:
    s = norm_for_match(text)
    return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


def _hash_vector(text: str, dim: int) -> List[float]:
    vec = [0.0] * max(1, dim)
    for bg in _bigrams(text):
        h = int(hashlib.md5(bg.encode("utf-8")).hexdigest()[:8], 16) % len(vec)
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


@dataclass
class VectorIndex:
    """本地内存索引。build() 幂等，可反复调用；不落盘、不联网。"""

    backend: str = CONFIG.vector.backend
    dim: int = CONFIG.vector.dim
    docs: Dict[str, List[float]] = field(default_factory=dict)
    texts: Dict[str, str] = field(default_factory=dict)
    built: bool = False
    note: str = "占位实现（哈希 bigram，非语义向量）；本阶段未接入 embedding 模型"

    def available(self) -> bool:
        return bool(CONFIG.vector.enabled) and self.built

    def build(self, blocks: List[Block]) -> "VectorIndex":
        """从 Block 列表构建索引（幂等覆盖）。异常时退回空索引，不抛异常。"""
        try:
            docs, texts = {}, {}
            for b in blocks:
                if not isinstance(b, Block):
                    continue
                ref = b.block_ref or b.line_id
                docs[ref] = _hash_vector(b.text_norm or b.text, self.dim)
                texts[ref] = b.text
            self.docs, self.texts, self.built = docs, texts, True
        except Exception:
            self.docs, self.texts, self.built = {}, {}, False
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """返回 [(block_ref, cosine)]，按分数降序；不可用时返回空列表。"""
        if not self.available() or not query:
            return []
        try:
            qv = _hash_vector(query, self.dim)
            scored = [(ref, _cosine(qv, vec)) for ref, vec in self.docs.items()]
            scored.sort(key=lambda x: (-x[1], x[0]))
            return [(ref, round(s, 4)) for ref, s in scored[:max(0, top_k)] if s > 0]
        except Exception:
            return []


def build_index(blocks: List[Block]) -> VectorIndex:
    """模块级便捷入口。"""
    return VectorIndex().build(blocks)
