# -*- coding: utf-8 -*-
"""
qa/retrieve.py —— ④ 混合检索（稀疏 + 可选向量）

沿用旧工程 `matcher/retrieve.py` 的字面 + bigram 三层打分思想：
    A 标签命中（"某词："）→ B 完整命中 → C bigram 局部命中
扩展点：
    - 从「按列名筛候选块」扩展为「按自然语言问题检索可溯源证据」；
    - 新增可选向量通道（qa/index.py），两路分数加权融合；
    - 输出统一为 Evidence，携带 page_id / bbox / line_id / quote / source_hash。

关键约束：检索只负责「找原文」，**不生成结论**；找不到就交给 qa/refuse.py 拒答。

TODO（下一阶段）：
  - 术语权重（IDF）与古籍异体字扩展召回；
  - 两路分数的统一归一化与 RRF 融合；
  - 图谱通道（关系约束召回）与多跳证据链。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from app.config import CONFIG
from app.schemas import Block, Evidence, norm_for_match
from extract.schema import EFFECT_WORDS, FORMULAS, HERBS

from .index import VectorIndex

# 疑问 / 虚词：只用于切出问题的核心词，不参与打分
STOP_WORDS = {
    "的", "了", "是", "什么", "吗", "呢", "啊", "请", "请问", "问", "有",
    "哪些", "哪个", "哪", "如何", "怎么", "怎样", "多少", "几", "在", "中",
    "该", "可以", "能否", "能", "会", "对", "我", "我的", "给", "及", "与",
    "和", "由", "把", "被", "这", "那", "其", "之", "者", "也", "本品",
    "一下", "相关", "记载", "本书", "里", "上", "下", "为", "用", "吗？",
}

# 已知术语优先整体切出，避免「桂枝汤」被拆成「桂枝」导致召回偏差
KNOWN_TERMS = sorted(
    set(HERBS) | set(FORMULAS) | set(EFFECT_WORDS) |
    {"性味", "归经", "功效", "主治", "禁忌", "组成", "剂量", "方剂", "药材", "用量"},
    key=lambda s: (-len(s), s),
)

# 提问词 → 原文常见写法。只做保守扩展，取 max 不累加，避免分数膨胀。
# TODO：下一阶段换成人工维护的中医药同义词表并纳入评测。
TERM_EXPANSION = {
    "性味": ["味甘", "味苦", "味辛", "无毒"],
    "禁忌": ["忌"],
    "桂枝汤": ["桂枝"],           # 方名行后的组成行常只写药名，不重复方名
}

_SPLIT_RE = re.compile(r"[\s,，。；;、：:！!？?（）()《》「」]+")


def bigrams(text: str) -> List[str]:
    """切成连续两字片段：'性味归经' → ['性味','味归','归经']（沿用旧实现的切分思想）。"""
    s = norm_for_match(text)
    return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


def score_block(text: str, core: str) -> float:
    """三层字面打分（旧 score_block 的语义保留，分数尺度沿用）。

    A 标签命中 +90 / B 完整命中 +40 / C bigram 命中比例最多 +40。
    """
    if not text or not core:
        return 0.0
    score = 0.0
    # A) 标签命中：任一中长前缀后紧跟冒号
    prefixes = [core[:i] for i in range(2, len(core) + 1)]
    if any((p + "：" in text) or (p + ":" in text) for p in prefixes):
        score += 90
    # B) 完整命中
    if core in text:
        score += 40
    # C) 局部命中
    parts = bigrams(core)
    if parts:
        hit = sum(1 for p in parts if p in text)
        score += 40 * hit / len(parts)
    return score


def core_terms(query: str) -> List[str]:
    """把自然语言问题切成检索词：已知术语整体切出，余下片段去虚词。"""
    if not query:
        return []
    rest = str(query)
    terms: List[str] = []
    for t in KNOWN_TERMS:
        if t in rest:
            terms.append(t)
            rest = rest.replace(t, " ")
    for w in sorted(STOP_WORDS, key=len, reverse=True):
        rest = rest.replace(w, " ")
    for seg in _SPLIT_RE.split(rest):
        seg = seg.strip()
        if len(seg) >= 2:
            terms.append(seg)
    # 去重保序
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out or [str(query)]


def term_score(text: str, term: str) -> float:
    """单词打分：本词与保守扩展词取最高分（不累加）。"""
    best = score_block(text, term)
    for alias in TERM_EXPANSION.get(term, []):
        best = max(best, score_block(text, alias))
    return best


def sparse_scores(query: str, blocks: Iterable[Block]) -> Dict[str, float]:
    """稀疏通道：对每个块累加各检索词的三层打分。"""
    terms = core_terms(query)
    scores: Dict[str, float] = {}
    for b in blocks:
        if not isinstance(b, Block):
            continue
        text = b.text_norm or b.text
        total = sum(term_score(text, t) for t in terms)
        if total > 0:
            scores[b.block_ref or b.line_id] = round(total, 2)
    return scores


def hybrid_retrieve(query: str, blocks: Iterable[Block],
                    vector_index: Optional[VectorIndex] = None,
                    top_k: Optional[int] = None,
                    min_score: Optional[float] = None) -> List[Evidence]:
    """混合检索入口。

    输入：问题、Block 列表、可选向量索引
    输出：Evidence 列表（按分数降序，字段完整），无命中返回空列表，不抛异常。
    """
    items = [b for b in blocks if isinstance(b, Block)]
    k = int(top_k if top_k is not None else CONFIG.qa.top_k)
    threshold = float(min_score if min_score is not None else CONFIG.qa.min_score)
    try:
        sparse = sparse_scores(query, items)
        vec: Dict[str, float] = {}
        use_vec = vector_index is not None and vector_index.available()
        if use_vec:
            for ref, s in vector_index.search(query, top_k=max(k * 3, 15)):
                vec[ref] = s * 100.0 * CONFIG.qa.vector_weight

        fused: Dict[str, float] = dict(sparse)
        for ref, v in vec.items():
            fused[ref] = fused.get(ref, 0.0) + v * CONFIG.qa.sparse_weight

        by_ref = {b.block_ref or b.line_id: b for b in items}
        best = max(fused.values(), default=0.0)
        keep = max(threshold, best * CONFIG.qa.relative_keep) if best > 0 else threshold
        out: List[Evidence] = []
        for ref, score in sorted(fused.items(), key=lambda x: (-x[1], x[0])):
            if score < keep or len(out) >= k:
                continue
            b = by_ref.get(ref)
            if b is None:
                continue
            channel = "hybrid" if (use_vec and ref in vec and ref in sparse) else (
                "vector" if (use_vec and ref in vec) else "sparse")
            out.append(Evidence(
                page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
                quote=b.text, source_hash=b.source_hash, score=round(score, 2),
                block_ref=ref, channel=channel, is_demo=True,
            ))
        return out
    except Exception:
        return []  # 检索失败 → 空证据 → 由 refuse.py 转拒答


def retrieve_with_trace(query: str, blocks: List[Block]) -> Tuple[List[Evidence], List[str]]:
    """带 trace 的检索，供 Answer.trace 记录检索过程。"""
    terms = core_terms(query)
    ev = hybrid_retrieve(query, blocks)
    trace = [f"检索词：{'、'.join(terms) or '（空）'}",
             f"候选块：{len(blocks)}，命中证据：{len(ev)}，阈值：{CONFIG.qa.min_score}",
             f"向量通道：{'启用' if CONFIG.vector.enabled else '关闭（占位）'}"]
    return ev, trace
