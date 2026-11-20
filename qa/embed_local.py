# -*- coding: utf-8 -*-
"""
qa/embed_local.py —— 本地 ONNX 语义向量引擎（bge-small-zh-v1.5）

为什么自己实现而不是装 sentence-transformers：
  项目要求"纯本地、零云端、依赖尽量少"。ONNX Runtime 已经是 OCR 链路的既有依赖，
  tokenizers 只是 3MB 的 Rust 扩展，加起来就能跑 BERT 类向量模型，
  比引入 800MB 的 torch 更符合本项目的工程约束。

推理流程（标准 BERT 句向量）：
    tokenizer(截断 512) → onnxruntime → last_hidden_state → 取 [CLS] → L2 归一化
bge 系列中文模型的官方推荐池化方式就是 CLS，归一化后余弦相似度 = 点积，
所以索引里存的都是单位向量，检索时一次矩阵乘即可。

线程安全：模型只加载一次（模块级单例），推理本身无状态，可多线程并发。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from app.config import CONFIG

logger = logging.getLogger("tcm.embed.local")

_lock = threading.Lock()
_state: dict = {"session": None, "tokenizer": None, "dim": 0, "error": ""}


def model_dir() -> Path:
    d = Path(CONFIG.embed.model_dir)
    return d if d.is_absolute() else (Path(CONFIG.paths.root) / d)


def weights_path() -> Path:
    return model_dir() / "model.onnx"


def tokenizer_path() -> Path:
    return model_dir() / "tokenizer.json"


def available() -> bool:
    return weights_path().exists() and tokenizer_path().exists()


def _load() -> bool:
    """懒加载模型；失败只记一次错误，不反复重试。"""
    if _state["session"] is not None:
        return True
    with _lock:
        if _state["session"] is not None:
            return True
        if not available():
            _state["error"] = f"模型文件不存在：{model_dir()}"
            return False
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, int(CONFIG.embed.threads))
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(weights_path()), so,
                                        providers=["CPUExecutionProvider"])
            tok = Tokenizer.from_file(str(tokenizer_path()))
            tok.enable_truncation(max_length=int(CONFIG.embed.max_length))
            tok.enable_padding(pad_id=0, pad_token="[PAD]")
            _state.update({"session": sess, "tokenizer": tok, "error": ""})
            logger.info("已加载本地向量模型：%s", model_dir().name)
            return True
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("向量模型加载失败：%s", _state["error"])
            return False


def _pool(last_hidden, cls: bool = True) -> List[List[float]]:
    """CLS 池化：取每个序列第一个 token 的隐状态。"""
    out: List[List[float]] = []
    for row in last_hidden:
        out.append([float(v) for v in row[0]])
    return out


def _l2(vec: List[float]) -> List[float]:
    import math
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n else vec


def embed(texts: List[str]) -> Optional[List[List[float]]]:
    """批量编码；模型不可用时返回 None（由上层降级，绝不伪造向量）。"""
    items = [str(t or "") for t in (texts or [])]
    if not items:
        return []
    if not _load():
        return None
    try:
        import numpy as np

        encs = _state["tokenizer"].encode_batch(items)
        input_ids = [e.ids for e in encs]
        mask = [e.attention_mask for e in encs]
        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(mask, dtype=np.int64),
        }
        names = {i.name for i in _state["session"].get_inputs()}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = [e.type_ids for e in encs]
            feeds["token_type_ids"] = np.asarray(feeds["token_type_ids"], dtype=np.int64)
        feeds = {k: v for k, v in feeds.items() if k in names}

        out_name = "last_hidden_state" if "last_hidden_state" in {
            o.name for o in _state["session"].get_outputs()} else None
        outputs = _state["session"].run([out_name] if out_name else None, feeds)
        hidden = outputs[0]
        _state["dim"] = int(hidden.shape[-1])
        return [_l2(v) for v in _pool(hidden)]
    except Exception as exc:
        _state["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("向量推理失败：%s", _state["error"])
        return None


def embed_one(text: str) -> Optional[List[float]]:
    vecs = embed([text])
    return vecs[0] if vecs else None


def dim() -> int:
    return int(_state.get("dim") or 0)


def status() -> dict:
    return {
        "backend": "local-onnx",
        "model": CONFIG.embed.model,
        "dir": str(model_dir()),
        "available": available(),
        "loaded": _state["session"] is not None,
        "dim": dim(),
        "error": _state.get("error", ""),
    }
