# -*- coding: utf-8 -*-
"""
graph/query.py —— 图谱查询（某症状常用药材 / 某药禁忌 / 方剂组成）

原型阶段直接在三元组列表上做确定性查询，不依赖图数据库。
所有命中都带 evidence_refs 与原文 quote，可回原文核对。

TODO（下一阶段）：
  - 支持多跳路径查询（症状 → 方剂 → 药材 → 禁忌）与结果排序；
  - 别名 / 同义词扩展与版本化出处过滤；
  - 查询结果缓存与分页，供前端大图交互。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.schemas import Triple


def _pack(kind: str, term: str, matches: List[Triple]) -> Dict[str, Any]:
    return {
        "kind": kind,
        "term": term,
        "count": len(matches),
        "results": [
            {"head": t.head, "relation": t.relation, "tail": t.tail,
             "page_id": t.page_id, "bbox": t.bbox, "line_id": t.line_id,
             "quote": t.quote, "source_hash": t.source_hash,
             "evidence_refs": t.evidence_refs}
            for t in matches
        ],
        "is_demo": True,
        "note": "示例数据 · 非命题单位样本",
    }


def query_herbs_by_symptom(symptom: str, triples: Iterable[Triple]) -> Dict[str, Any]:
    """某症状 / 主治对应的常用药材：按「主治 / 功效」关系的 tail 反查 head。"""
    hits = [t for t in triples if isinstance(t, Triple)
            and t.relation in ("主治", "功效") and symptom in (t.tail + t.head)]
    return _pack("herbs_by_symptom", symptom, hits)


def query_contraindications(herb: str, triples: Iterable[Triple]) -> Dict[str, Any]:
    """某药禁忌：按「禁忌」关系查 tail。"""
    hits = [t for t in triples if isinstance(t, Triple)
            and t.relation == "禁忌" and herb in t.head]
    return _pack("contraindications", herb, hits)


def query_formula(formula: str, triples: Iterable[Triple]) -> Dict[str, Any]:
    """方剂组成与剂量。"""
    hits = [t for t in triples if isinstance(t, Triple)
            and formula in t.head and t.relation in ("组成", "剂量")]
    return _pack("formula", formula, hits)


def run_query(kind: str, term: str, triples: Iterable[Triple]) -> Dict[str, Any]:
    """统一查询入口，未知 kind 返回空结果 + 说明，不抛异常。"""
    items = list(triples)
    try:
        if kind in ("herbs_by_symptom", "症状"):
            return query_herbs_by_symptom(term, items)
        if kind in ("contraindications", "禁忌"):
            return query_contraindications(term, items)
        if kind in ("formula", "方剂"):
            return query_formula(term, items)
        return _pack(kind or "unknown", term, [])
    except Exception as exc:
        out = _pack(kind, term, [])
        out["note"] = f"查询降级：{type(exc).__name__}"
        return out
