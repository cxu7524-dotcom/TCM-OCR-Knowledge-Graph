# -*- coding: utf-8 -*-
"""
qa/embed.py —— 语义向量统一入口（多后端，纯本地）

把任意中文文本转成定长稠密向量。支持两种本机后端，按配置切换：

  local  （默认）qa/embed_local.py，ONNX bge-small-zh-v1.5（512 维）
          —— 零外部服务、零云端，onnxruntime 直接推理；
  ollama         本机 Ollama 的 /api/embed 接口（需 ollama pull 向量模型）。

两条硬约束：
  1. 只允许本机：Ollama 地址必须是 127.0.0.1 / localhost，否则直接视为不可用；
  2. 绝不伪造向量：任一环节失败返回 None，由上层降级为字面检索，
     而不是返回随机/零向量导致"看起来有检索结果"。

模型名与维度会写进索引 manifest；换模型后旧索引自动失效重建，
避免用 A 模型的向量空间去比 B 模型产出的查询向量。
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from typing import List, Optional

from app.config import CONFIG

from . import embed_local

logger = logging.getLogger("tcm.embed")

_LAST_DIM: int = 0


# ------------------------------------------------------------------
# Ollama 后端
# ------------------------------------------------------------------

def _api_base() -> str:
    base = CONFIG.embed.resolved_base_url()
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


def _post(path: str, payload: dict) -> Optional[dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _api_base() + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=float(CONFIG.embed.timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("embedding 调用失败 %s：%s", path, type(exc).__name__)
        return None


def _embed_ollama(texts: List[str]) -> Optional[List[List[float]]]:
    global _LAST_DIM
    out: List[List[float]] = []
    bs = max(1, int(CONFIG.embed.batch_size))
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        body = _post("/api/embed", {"model": model_name(), "input": batch})
        vecs = (body or {}).get("embeddings")
        if isinstance(vecs, list) and len(vecs) == len(batch) and vecs:
            out.extend([[float(x) for x in v] for v in vecs])
            _LAST_DIM = len(vecs[0])
            continue
        for t in batch:                       # 回退旧接口
            one = _post("/api/embeddings", {"model": model_name(), "prompt": t})
            v = (one or {}).get("embedding")
            if not isinstance(v, list) or not v:
                return None
            out.append([float(x) for x in v])
            _LAST_DIM = len(v)
    return out if len(out) == len(texts) else None


# ------------------------------------------------------------------
# 统一接口
# ------------------------------------------------------------------

def available() -> bool:
    """是否配置了可用的本机向量后端（不做重探测，避免每次调用都多一次开销）。"""
    return bool(CONFIG.embed.enabled)


def model_name() -> str:
    """写入索引 manifest 的模型标识；换模型即触发索引重建。"""
    return f"{CONFIG.embed.model}@{(CONFIG.embed.backend or '').lower()}"


def last_dim() -> int:
    if CONFIG.embed.is_local_onnx:
        return embed_local.dim()
    return _LAST_DIM


def embed(texts: List[str]) -> Optional[List[List[float]]]:
    """批量向量化。返回与输入等长的向量列表；任一失败返回 None。"""
    items = [str(t or "") for t in (texts or [])]
    if not items:
        return []
    if not available():
        return None
    if CONFIG.embed.is_local_onnx:
        return embed_local.embed(items)
    return _embed_ollama(items)


def embed_one(text: str) -> Optional[List[float]]:
    vecs = embed([text])
    return vecs[0] if vecs else None


def probe() -> dict:
    """轻量健康检查：真跑一次极短文本，确认模型可用并报告维度。"""
    info = {
        "backend": CONFIG.embed.backend,
        "model": CONFIG.embed.model,
        "enabled": available(),
    }
    if CONFIG.embed.is_local_onnx:
        info.update(embed_local.status())
    else:
        info["base_url"] = _api_base()
    if not available():
        info.update({"reachable": False, "dim": 0,
                     "note": "语义向量后端不可用（检查 TCM_EMBED_BACKEND 与模型文件）"})
        return info
    vec = embed_one("桂枝汤")
    if vec:
        info.update({"reachable": True, "dim": len(vec), "note": "语义向量模型已就绪"})
    else:
        info.update({"reachable": False, "dim": 0,
                     "note": f"向量推理失败（{info.get('error') or '见服务日志'}）"})
    return info


def cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度（向量已归一化时等价于点积）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
