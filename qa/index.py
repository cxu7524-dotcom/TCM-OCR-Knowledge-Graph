# -*- coding: utf-8 -*-
"""
qa/index.py —— 向量索引（语义 · 本机 embedding · 持久化）

两条通道并存，按配置自动选择：

  SemanticIndex（默认，VECTOR_BACKEND=ollama / TCM_EMBED_BACKEND=ollama）
    用本机 Ollama 的 embedding 模型（默认 bge-m3，1024 维）把知识库段落编码成
    稠密向量，余弦相似度检索；索引持久化在 data/store/vectors/<model>/ 下，
    语料指纹变化时自动重建（后台线程，不阻塞请求）。

  VectorIndex（兜底，旧行为）
    标准库哈希 bigram，非语义向量，仅用于链路自检；默认不启用。

持久化文件（每个模型一个目录，换模型自动隔离）：
    manifest.json   模型名 / 维度 / 段落数 / 语料指纹 / 构建时间
    vectors.npy     L2 归一化后的 float32 矩阵（n × dim）
    pids.json       与矩阵行一一对应的段落 pid

设计原则：索引只是**加速结构**，任何时候都不参与"事实判定"；检索不到就如实
返回空，由上层决定"常识作答"还是"拒答"，绝不伪造相似度。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import CONFIG
from app.schemas import Block, norm_for_match

from . import kb
from . import embed as embed_mod

logger = logging.getLogger("tcm.index")

try:  # numpy 是本项目 OCR 链路的既有依赖，这里只用它做矩阵乘
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NUMPY = False


# ------------------------------------------------------------------
# 一、旧哈希向量（保底，非语义；保持原有 API 不变）
# ------------------------------------------------------------------

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
    """本地内存哈希索引（非语义，兜底用）。build() 幂等。"""

    backend: str = CONFIG.vector.backend
    dim: int = CONFIG.vector.dim
    docs: Dict[str, List[float]] = field(default_factory=dict)
    texts: Dict[str, str] = field(default_factory=dict)
    built: bool = False
    note: str = "哈希 bigram 兜底索引（非语义向量）"

    def available(self) -> bool:
        return bool(CONFIG.vector.enabled) and self.built

    def build(self, blocks: List[Block]) -> "VectorIndex":
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
    """旧接口：构建哈希索引（仅供自检与兼容）。"""
    return VectorIndex().build(blocks)


# ------------------------------------------------------------------
# 二、语义索引（主通道）
# ------------------------------------------------------------------

def _slug(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", model or "model")


def _l2(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n else vec


@dataclass
class SemanticIndex:
    """语义向量索引：段落级编码 + 余弦检索 + 磁盘持久化。"""

    model: str = ""
    dim: int = 0
    pids: List[str] = field(default_factory=list)
    matrix: object = None                 # numpy (n, dim) 或 list[list[float]]
    signature: str = ""
    built_at: float = 0.0
    health: str = "未构建"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------- 路径 ----------
    def dir(self) -> Path:
        base = Path(CONFIG.paths.store_dir) / "vectors" / _slug(self.model or embed_mod.model_name())
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _f_manifest(self) -> Path:
        return self.dir() / "manifest.json"

    def _f_matrix(self) -> Path:
        return self.dir() / "vectors.npy"

    def _f_pids(self) -> Path:
        return self.dir() / "pids.json"

    # ---------- 状态 ----------
    def available(self) -> bool:
        return bool(CONFIG.rag.enabled and CONFIG.embed.enabled and self.pids and self.matrix is not None)

    def status(self) -> dict:
        stale = self.signature != kb.corpus_signature() if self.signature else True
        return {
            "backend": "embedding",
            "model": self.model or embed_mod.model_name(),
            "dim": self.dim,
            "vectors": len(self.pids),
            "built_at": self.built_at,
            "signature": self.signature,
            "stale": bool(stale) if self.pids else True,
            "available": self.available(),
            "numpy": _HAS_NUMPY,
            "health": self.health,
        }

    # ---------- 读写 ----------
    def load(self) -> "SemanticIndex":
        """从磁盘加载；缺失/损坏/模型不匹配都视为未构建，不抛异常。"""
        try:
            mf = self._f_manifest()
            if not (mf.exists() and self._f_matrix().exists() and self._f_pids().exists()):
                self.health = "索引文件不存在"
                return self
            meta = json.loads(mf.read_text(encoding="utf-8"))
            if meta.get("model") != embed_mod.model_name():
                self.health = "模型已更换，需重建"
                return self
            pids = json.loads(self._f_pids().read_text(encoding="utf-8"))
            if _HAS_NUMPY:
                mat = _np.load(self._f_matrix())
            else:
                mat = json.loads(self._f_matrix().read_text(encoding="utf-8"))
            if len(pids) != len(mat):
                self.health = "索引行数与 pid 数不一致，需重建"
                return self
            self.model = meta.get("model", "")
            self.dim = int(meta.get("dim") or (len(mat[0]) if len(mat) else 0))
            self.pids, self.matrix = pids, mat
            self.signature = str(meta.get("signature") or "")
            self.built_at = float(meta.get("built_at") or 0.0)
            self.health = "已从磁盘加载"
            return self
        except Exception as exc:
            self.health = f"加载失败：{type(exc).__name__}"
            return self

    def save(self) -> None:
        try:
            meta = {"model": self.model, "dim": self.dim, "passages": len(self.pids),
                    "signature": self.signature, "built_at": self.built_at,
                    "note": "本机 embedding 生成，未上传任何外部服务"}
            self._f_manifest().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._f_pids().write_text(json.dumps(self.pids, ensure_ascii=False), encoding="utf-8")
            if _HAS_NUMPY:
                _np.save(self._f_matrix(), self.matrix)
            else:
                self._f_matrix().write_text(json.dumps(self.matrix), encoding="utf-8")
        except Exception as exc:
            self.health = f"写入失败：{type(exc).__name__}"

    # ---------- 构建 ----------
    def build(self, force: bool = False) -> "SemanticIndex":
        """把知识库全部段落编码入库。返回自身，任何失败只记录 health，不抛。"""
        if not CONFIG.embed.enabled:
            self.health = "embedding 未启用，未构建语义索引"
            return self
        with self._lock:
            t0 = time.time()
            try:
                passages = kb.build_passages(force=True)
                if not passages:
                    self.health = "知识库为空，无段落可编码"
                    self.pids, self.matrix = [], None
                    return self
                sig = kb.corpus_signature()
                if (not force) and self.pids and self.signature == sig:
                    self.health = "语料未变化，跳过重建"
                    return self

                texts = [ps.embed_text for ps in passages]
                vecs: List[List[float]] = []
                step = max(1, int(CONFIG.embed.batch_size))
                for i in range(0, len(texts), step):
                    part = embed_mod.embed(texts[i:i + step])
                    if not part:
                        self.health = "embedding 调用失败，索引保持旧状态"
                        return self
                    vecs.extend(_l2(v) for v in part)

                self.model = embed_mod.model_name()
                self.dim = len(vecs[0]) if vecs else 0
                self.pids = [ps.pid for ps in passages]
                self.matrix = _np.asarray(vecs, dtype=_np.float32) if _HAS_NUMPY else vecs
                self.signature = sig
                self.built_at = time.time()
                self.save()
                self.health = f"构建完成：{len(self.pids)} 段 / {self.dim} 维，耗时 {time.time()-t0:.1f}s"
                logger.info("语义索引构建完成：%s", self.health)
            except Exception as exc:
                self.health = f"构建异常：{type(exc).__name__}"
                logger.warning("语义索引构建失败：%s", exc)
            return self

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 6) -> List[Tuple[object, float]]:
        """返回 [(Passage, cosine)]，按相似度降序；不可用时返回空列表。"""
        if not self.available() or not (query or "").strip():
            return []
        try:
            qv = embed_mod.embed_one(query)
            if not qv or len(qv) != self.dim:
                return []
            q = _l2(qv)
            if _HAS_NUMPY:
                scores = self.matrix @ _np.asarray(q, dtype=_np.float32)
                order = _np.argsort(-scores)[:max(1, top_k)]
                pairs = [(int(i), float(scores[int(i)])) for i in order]
            else:
                raw = [(_cosine(q, v), i) for i, v in enumerate(self.matrix)]  # type: ignore[arg-type]
                raw.sort(reverse=True)
                pairs = [(i, s) for s, i in raw[:max(1, top_k)]]
            by_pid = {ps.pid: ps for ps in kb.build_passages()}
            out = []
            for idx, score in pairs:
                if idx < 0 or idx >= len(self.pids):
                    continue
                ps = by_pid.get(self.pids[idx])
                if ps is not None:
                    out.append((ps, round(float(score), 4)))
            return out
        except Exception as exc:
            logger.warning("语义检索失败：%s", type(exc).__name__)
            return []


# ------------------------------------------------------------------
# 三、全局单例：懒加载 + 语料变化后台重建
# ------------------------------------------------------------------

_SINGLETON: Optional[SemanticIndex] = None
_BUILDING = threading.Event()


def get_index() -> SemanticIndex:
    """取全局语义索引；首次调用自动加载磁盘索引，语料变化则后台重建。"""
    global _SINGLETON
    if _SINGLETON is None:
        idx = SemanticIndex(model=embed_mod.model_name()).load()
        _SINGLETON = idx
        if idx.status().get("stale") and not _BUILDING.is_set():
            kick_rebuild()
    return _SINGLETON


def kick_rebuild(force: bool = False) -> None:
    """后台线程重建索引，避免上传后阻塞请求。"""
    if _BUILDING.is_set():
        return
    _BUILDING.set()

    def _run() -> None:
        try:
            SemanticIndex(model=embed_mod.model_name()).build(force=force)
            global _SINGLETON
            _SINGLETON = SemanticIndex(model=embed_mod.model_name()).load()
        finally:
            _BUILDING.clear()

    threading.Thread(target=_run, name="tcm-index-build", daemon=True).start()


def rebuild_now(force: bool = True) -> dict:
    """同步重建（供 /api/rag/rebuild 使用）。"""
    global _SINGLETON
    idx = SemanticIndex(model=embed_mod.model_name())
    idx.build(force=force)
    _SINGLETON = idx
    return idx.status()
