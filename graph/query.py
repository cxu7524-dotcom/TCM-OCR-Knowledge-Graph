# -*- coding: utf-8 -*-
"""
graph/query.py —— 图谱查询（按症状查药材 / 按症状查方剂 / 某药禁忌）

原型阶段直接在三元组列表上做确定性查询，不依赖图数据库。
所有命中都带 evidence_refs 与原文 quote，可回原文核对。

**方向与主体口径（阶段十修订）**：
三元组方向固定为 `head -relation-> tail`（见 extract/schema.RELATIONS）：
    (药材|方剂) -主治-> 病证      症状在 **tail**，主体在 **head**
    (药材|方剂) -禁忌-> 忌口对象  药名在 **head**，禁忌在 **tail**
旧实现在 `symptom in (t.tail + t.head)` 上做子串匹配，既不区分方向也不区分主体类型，
「症状词恰好出现在药名 / 邻近正文」也能命中——这正是本阶段要堵掉的误召回。
现在一律：先判关系，再判方向，最后按实体类型过滤主体。

同时支持按页限定（`page_id`）：问答只引用当前页的记载，跨页一律不引用。

TODO（下一阶段）：
  - 支持多跳路径查询（症状 → 方剂 → 药材 → 禁忌）与结果排序；
  - 别名 / 同义词扩展与版本化出处过滤；
  - 查询结果缓存与分页，供前端大图交互。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.schemas import Triple, align_norm
from extract.schema import FORMULAS, HERBS

# 方名后缀：词典外的方名（古籍里占多数，如「三化湯」）靠后缀判类型
FORMULA_SUFFIXES = ("汤", "湯", "散", "丸", "丹", "膏", "饮", "飲", "煎", "锭", "錠", "露")

SUBJECT_TYPES = ("药材", "方剂")


def _norm(text: Any) -> str:
    """比对用归一化（去空白标点 + 异体字 + 繁简），与全链路同一套口径。"""
    return align_norm("" if text is None else str(text))


# 词典归一化一次即可（术语表在进程内不变），避免逐条三元组重算
_HERB_NORM = {_norm(h) for h in HERBS}
_FORMULA_NORM = {_norm(f) for f in FORMULAS}


def subject_type(name: str, entities: Optional[Iterable[Any]] = None) -> str:
    """推断三元组主语的实体类型：实体表 → 词典 → 方名后缀。

    「六君子湯」「三化湯」这类词典外方名靠后缀判为方剂；
    词典内药材优先于后缀（避免「散」「丸」类短词把药名判成方剂）。
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    for e in entities or []:
        etype = e.get("type") if isinstance(e, dict) else getattr(e, "type", "")
        ename = e.get("name") if isinstance(e, dict) else getattr(e, "name", "")
        if ename == raw and etype in SUBJECT_TYPES:
            return str(etype)
    key = _norm(raw)
    if key in _HERB_NORM:
        return "药材"
    if key in _FORMULA_NORM:
        return "方剂"
    if raw.endswith(FORMULA_SUFFIXES):
        return "方剂"
    return "其他"


def _page_ok(t: Triple, page_id: str) -> bool:
    """按页限定：page_id 为空时不过滤（兼容旧调用）。"""
    return (not page_id) or (t.page_id == page_id)


def _tail_has(tail: str, term: str) -> bool:
    """症状是否被该关系的**客体**（主治病证）覆盖。

    只认 tail：关系方向是「主体 → 主治」，症状出现在主体名里不算支持；
    出现在正文邻近文本里更不算（那属于文本检索，不是图谱关系）。
    """
    nt, ntail = _norm(term), _norm(tail)
    if not nt or not ntail:
        return False
    return nt in ntail


def _head_is(head: str, term: str) -> bool:
    """主体名匹配：归一化（繁简 / 已定义异体字 / 去标点空白）后**完全相等**。

    不做子串匹配的原因（阶段十补验）：`甘草` 与 `生甘草`、`炙甘草`、`甘草節` 是不同药材或
    不同炮制品，子串重合不足以认定同一主体；此前 `nh in nt or nt in nh` 会让
    「炙甘草有什么禁忌」命中「甘草 忌海藻」，把别的药材 / 别的炮制品的禁忌错归过来。
    归一化本身仍允许同名异写（如「甘草」与「甘草」全半角、繁体写法的差异）。
    """
    nh, nt = _norm(head), _norm(term)
    if not nh or not nt:
        return False
    return nh == nt


def _pack(kind: str, term: str, matches: List[Triple],
          entities: Optional[Iterable[Any]] = None,
          page_id: str = "") -> Dict[str, Any]:
    results = []
    for t in matches:
        results.append({
            "head": t.head, "relation": t.relation, "tail": t.tail,
            "subject_type": subject_type(t.head, entities),
            "page_id": t.page_id, "bbox": t.bbox, "line_id": t.line_id,
            "quote": t.quote, "source_hash": t.source_hash,
            "evidence_refs": t.evidence_refs,
        })
    demo = all(bool(t.is_demo) for t in matches) if matches else True
    return {
        "kind": kind,
        "term": term,
        "page_id": page_id,
        "count": len(matches),
        "results": results,
        "is_demo": demo,
        "note": ("未在图谱中命中该关系" if not matches else
                 ("示例数据 · 非命题单位样本" if demo else "本地 OCR 页抽出的关系，需人工复核")),
    }


def query_herbs_by_symptom(symptom: str, triples: Iterable[Triple],
                           page_id: str = "",
                           entities: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """某症状对应的药材：只认 `药材 -主治-> 病证`，症状必须落在 tail。"""
    ents = list(entities or [])
    hits = [t for t in triples if isinstance(t, Triple)
            and _page_ok(t, page_id)
            and t.relation == "主治"
            and subject_type(t.head, ents) == "药材"
            and _tail_has(t.tail, symptom)]
    return _pack("herbs_by_symptom", symptom, hits, ents, page_id)


def query_formulas_by_symptom(symptom: str, triples: Iterable[Triple],
                              page_id: str = "",
                              entities: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """某症状对应的方剂：只认 `方剂 -主治-> 病证`，症状必须落在 tail。"""
    ents = list(entities or [])
    hits = [t for t in triples if isinstance(t, Triple)
            and _page_ok(t, page_id)
            and t.relation == "主治"
            and subject_type(t.head, ents) == "方剂"
            and _tail_has(t.tail, symptom)]
    return _pack("formulas_by_symptom", symptom, hits, ents, page_id)


def query_contraindications(herb: str, triples: Iterable[Triple],
                            page_id: str = "",
                            entities: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """某药禁忌：只认 `(药材|方剂) -禁忌-> 忌口对象`，药名必须落在 head。"""
    ents = list(entities or [])
    hits = [t for t in triples if isinstance(t, Triple)
            and _page_ok(t, page_id)
            and t.relation == "禁忌"
            and subject_type(t.head, ents) in SUBJECT_TYPES
            and _head_is(t.head, herb)]
    return _pack("contraindications", herb, hits, ents, page_id)


def query_formula(formula: str, triples: Iterable[Triple],
                  page_id: str = "",
                  entities: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """方剂组成与剂量。"""
    ents = list(entities or [])
    hits = [t for t in triples if isinstance(t, Triple)
            and _page_ok(t, page_id)
            and _head_is(t.head, formula) and t.relation in ("组成", "剂量")]
    return _pack("formula", formula, hits, ents, page_id)


def run_query(kind: str, term: str, triples: Iterable[Triple],
              page_id: str = "",
              entities: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """统一查询入口，未知 kind 返回空结果 + 说明，不抛异常。"""
    items = list(triples)
    try:
        if kind in ("herbs_by_symptom", "症状", "herbs"):
            return query_herbs_by_symptom(term, items, page_id, entities)
        if kind in ("formulas_by_symptom", "方剂主治"):
            return query_formulas_by_symptom(term, items, page_id, entities)
        if kind in ("contraindications", "禁忌"):
            return query_contraindications(term, items, page_id, entities)
        if kind in ("formula", "方剂"):
            return query_formula(term, items, page_id, entities)
        return _pack(kind or "unknown", term, [], entities, page_id)
    except Exception as exc:
        out = _pack(kind, term, [], entities, page_id)
        out["note"] = f"查询降级：{type(exc).__name__}"
        return out
