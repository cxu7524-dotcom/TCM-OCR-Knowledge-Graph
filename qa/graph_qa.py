# -*- coding: utf-8 -*-
"""
qa/graph_qa.py —— ④ 图谱关系检索问答（症状→药材 / 症状→方剂 / 药材→禁忌）

与 qa/retrieve.py 的文本混检并列，是**题型识别后的图谱通道**：

    问题 → 题型识别（三类）→ 当前页三元组查询（graph/query.py）
         → 逐条出处校验（page_id / line_id / bbox / quote / source_hash）
         → 主体归属校验（主体必须在关系行，或紧邻的主体行里）
         → 复用 qa/generate.py 的证据校验 + 拒答 + 免责声明

硬约束：
  1. 关系方向按 extract/schema.RELATIONS：`(药材|方剂) -主治-> 病证`、`(药材|方剂) -禁忌-> 对象`。
     症状只认 **tail**，禁忌对象只认 **tail**，药名只认 **head**。
  2. 只引用当前页。跨页命中的条数只写进 trace，绝不进入证据列表。
  3. 关系跨行时，同时给出「关系引文」与「主体引文」，否则视为无法确认主体，丢弃。
  4. 一条合格关系都留不下时，走既有拒答流程（qa/refuse.py + qa/generate.py），
     拒答原因写明「本页原文没有能支持该关系的证据」，不做任何补答。

未命中任一题型时返回 None，由调用方回落到原有文本检索（qa/generate.answer）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.schemas import Answer, Block, Evidence, Triple, align_norm
from extract.schema import FORMULAS, HERBS
from graph.query import run_query
from ocr.normalize import normalize_text

from .generate import generate_answer

INTENT_HERBS = "herbs_by_symptom"
INTENT_FORMULAS = "formulas_by_symptom"
INTENT_CONTRA = "contraindications"

INTENT_TITLES = {
    INTENT_HERBS: "按症状查药材",
    INTENT_FORMULAS: "按症状查方剂",
    INTENT_CONTRA: "按药材查禁忌",
}

# 主体跨行的最大回溯距离（按该页 reading_order 相邻块计）。
# 古籍竖排的常见形态是「药名一行 + 若干主治/功效续行 + 禁忌行」，
# 因此允许回看若干行；但只要中途出现**另一个主体**，归属即视为不明，直接丢弃。
MAX_CARRY_GAP = 6

# 单次最多返回的关系条数，避免整页长文把答案撑爆
MAX_OBJECTS = 8

_TERM = r"[\u4e00-\u9fa5]{2,20}"
_TERM_SHORT = r"[\u4e00-\u9fa5]{2,10}"
_FORMULA_WORD = r"(?:方|方剂|方子|汤头)"
_HERB_WORD = r"(?:药|药材|中药|草药)"
_ASK_WORD = r"(?:什么|哪些|哪个|哪一种|哪一首|哪首|哪味|何种|何)"
_ASK = rf"(?:{_ASK_WORD})"

# 题型模式：命中的 question 必须**整句**匹配，避免「桂枝汤由哪些药材组成」被误判成查药材
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # —— 按药材 / 方剂查禁忌 ——
    (INTENT_CONTRA, re.compile(rf"^(?P<term>{_TERM_SHORT}?)(?:有|的|具有)?(?:{_ASK_WORD})?(?:配伍)?"
                               rf"(?:禁忌|忌口)(?:是|有|为)?(?:{_ASK_WORD})?$")),
    (INTENT_CONTRA, re.compile(rf"^(?P<term>{_TERM_SHORT}?)(?:忌|禁)(?:用|服)?(?:{_ASK_WORD})?$")),
    (INTENT_CONTRA, re.compile(rf"^(?P<term>{_TERM_SHORT}?)(?:不能|不可|不宜)(?:与|和){_ASK}"
                               rf"(?:同用|配伍|一起用)$")),
    # —— 按症状查方剂 ——
    (INTENT_FORMULAS, re.compile(rf"^(?P<term>.{{2,20}}?)(?:该|应|可|可以|宜|当)?用{_ASK}{_FORMULA_WORD}$")),
    (INTENT_FORMULAS, re.compile(rf"^(?P<term>.{{2,20}}?)(?:有|用){_ASK}{_FORMULA_WORD}$")),
    (INTENT_FORMULAS, re.compile(rf"^{_ASK}{_FORMULA_WORD}(?:能|可|可以|用来)?(?:治|疗|主)(?P<term>.{{2,20}})$")),
    # —— 按症状查药材 ——
    (INTENT_HERBS, re.compile(rf"^(?P<term>.{{2,20}}?)(?:该|应|可|可以|宜|当)?用{_ASK}{_HERB_WORD}$")),
    (INTENT_HERBS, re.compile(rf"^(?P<term>.{{2,20}}?)(?:有|用){_ASK}{_HERB_WORD}$")),
    (INTENT_HERBS, re.compile(rf"^{_ASK}{_HERB_WORD}(?:能|可|可以|用来)?(?:治|疗|主)(?P<term>.{{2,20}})$")),
]

_LEAD = ("请问", "请", "问", "查询", "本页", "本书", "该页", "这页", "这个", "其中")
_LEAD_VERB = ("治", "疗", "療", "主")
_TAIL_PUNCT = "？?！!。，,；;：: 　"
_BAD_SUBJECT = ("我", "你", "您")          # 带人称的问题按诊疗咨询处理，不进图谱通道
_BAD_TERM = ("我", "你", "您", "什么", "哪些", "哪个")


def _norm(text: Any) -> str:
    return align_norm("" if text is None else str(text))


def _clean_question(question: str) -> str:
    q = str(question or "").strip().strip(_TAIL_PUNCT)
    q = q.replace(" ", "").replace("　", "")
    q = q.replace("什麼", "什么").replace("甚麼", "什么").replace("麼", "么")
    q = normalize_text(q)          # 繁简 / 已定义异体字：让繁体提问与简体同样识别
    for w in _LEAD:
        if q.startswith(w):
            q = q[len(w):]
    return q


def _clean_term(term: str) -> str:
    t = str(term or "").strip(_TAIL_PUNCT)
    for _ in range(2):
        for w in _LEAD_VERB:
            if t.startswith(w) and len(t) > 2:
                t = t[1:]
    t = t.strip("的之")
    return t.strip(_TAIL_PUNCT)


def detect_intent(question: str) -> Tuple[str, str]:
    """题型识别。返回 (intent, term)；非三类题型返回 ("", "")。

    只做**确定性整句匹配**：宁可回落文本检索，也不把普通提问硬塞进图谱通道。
    """
    q = _clean_question(question)
    if not q or any(b in q for b in _BAD_SUBJECT):
        return "", ""
    for intent, pat in _PATTERNS:
        m = pat.match(q)
        if not m:
            continue
        term = _clean_term(m.group("term"))
        if 2 <= len(term) <= 20 and not any(b in term for b in _BAD_TERM):
            return intent, term
    return "", ""


# ------------------------------------------------------------------
# 出处校验：三元组的五个溯源字段必须与该页原文块逐项对上
# ------------------------------------------------------------------

def _same_bbox(a: Any, b: Any, tol: float = 1.0) -> bool:
    try:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return False
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def _provenance_problems(t: Triple, page_id: str, b: Block) -> List[str]:
    """逐项核对 page_id / line_id / bbox / quote / source_hash 与该页原文块。"""
    probs: List[str] = []
    if t.page_id != page_id:
        probs.append(f"page_id={t.page_id} 非当前页")
    if not t.line_id:
        probs.append("缺 line_id")
    if _norm(t.quote) != _norm(b.text):
        probs.append(f"quote 与 {b.line_id} 原文不一致")
    if t.source_hash and b.source_hash and t.source_hash != b.source_hash:
        probs.append("source_hash 不符")
    if not _same_bbox(t.bbox, b.bbox):
        probs.append("bbox 与该行不一致")
    refs = [str(r) for r in (t.evidence_refs or [])]
    if refs and b.block_ref not in refs and b.line_id not in refs:
        probs.append("evidence_refs 未指向该行")
    return probs


def _has_subject(text: str) -> bool:
    """该行是否出现「另一个主体」（药材 / 方名），用于判定跨行归属是否被打断。"""
    nt = _norm(text)
    if not nt:
        return False
    if any(_norm(h) in nt for h in HERBS):
        return True
    if any(_norm(f) in nt for f in FORMULAS):
        return True
    return bool(re.search(r"[\u4e00-\u9fa5]{2,6}(?:汤|湯|散|丸|丹|膏|饮|飲)", nt))


def _subject_support(head: str, blocks: List[Block], idx: int) -> Tuple[str, Optional[Block], int]:
    """确认这条关系的主体出处。

    返回 (support, subject_block, subject_index)：
      same_line —— 主体就在关系行，单条引文即可确认；
      carry     —— 主体在紧邻的上文行（≤MAX_CARRY_GAP 且中间无其他主体）；
      ""        —— 无法确认主体（中间插入了别的药材 / 方名，或超出回溯距离）。
    """
    nb = _norm(head)
    if not nb:
        return "", None, -1
    if nb in _norm(blocks[idx].text):
        return "same_line", blocks[idx], idx
    for k in range(idx - 1, max(-1, idx - 1 - MAX_CARRY_GAP) - 1, -1):
        if nb in _norm(blocks[k].text):
            return "carry", blocks[k], k
        if _has_subject(blocks[k].text):
            return "", None, k
    return "", None, -1


def _ev(b: Block, role: str, score: float, is_demo: bool) -> Evidence:
    return Evidence(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
        quote=b.text, source_hash=b.source_hash, score=round(float(score), 2),
        block_ref=b.block_ref or b.line_id, channel="graph", role=role,
        is_demo=is_demo,
    )


def _answer_text(intent: str, term: str, objects: List[Dict[str, Any]], page_id: str) -> str:
    """抽取式答案：只陈述「本页原文记载」，不写频次、不写建议。"""
    title = INTENT_TITLES.get(intent, "图谱关系")
    noun = {INTENT_HERBS: "药材", INTENT_FORMULAS: "方剂"}.get(intent, "药材或方剂")
    heads = []
    for o in objects:
        if o["head"] not in heads:
            heads.append(o["head"])
    lines = [f"【{title}】本页原文记载（page_id：{page_id}）："
             f"与「{term}」相关的关系 {len(objects)} 条，涉及{noun} {len(heads)} 个。"]
    for i, o in enumerate(objects, start=1):
        lines.append(f"[{i}] {o['head']}（{o['subject_type']}）—{o['relation']}→ {o['tail']}")
        lines.append(f"    关系引文 {o['relation_line']}：「{o['relation_quote']}」")
        if o["support"] == "carry":
            lines.append(f"    主体引文 {o['subject_line']}：「{o['subject_quote']}」"
                         "（主体与关系分列两行，两条引文共同确认）")
    lines.append("以上仅为该页原文摘录，不代表使用频次、疗效或配伍建议；单页记载不足以推断他页或临床用法。")
    return "\n".join(lines)


def _refusal(question: str, blocks: List[Block], trace: List[str], note: str,
             is_demo: bool) -> Answer:
    """走既有拒答流程：越界规则仍优先，证据为空则按 note 说明原因。"""
    return generate_answer(question, [], blocks, trace_extra=trace,
                           is_demo=is_demo, refuse_note=note)


def graph_answer(question: str, blocks: List[Block],
                 triples: Iterable[Triple],
                 entities: Optional[Iterable[Any]] = None,
                 page_id: str = "",
                 is_demo: bool = True,
                 page_exists: bool = True,
                 other_page_hits: int = 0) -> Optional[Answer]:
    """图谱关系问答入口。未命中三类题型返回 None（调用方回落文本检索）。"""
    intent, term = detect_intent(question)
    if not intent:
        return None

    title = INTENT_TITLES[intent]
    pid = page_id or next((b.page_id for b in blocks if b.page_id), "")
    trace = [f"题型识别：{title}；检索对象「{term}」",
             f"图谱通道：关系={('主治' if intent != INTENT_CONTRA else '禁忌')}，"
             f"范围限定 page_id={pid or '（未指定）'}，只引用当前页"]

    if page_id and not page_exists:
        return _refusal(question, blocks, trace + ["页面校验：指定 page_id 在库中不存在"],
                        "指定的页面不存在或已被删除，无法按该页检索关系证据", is_demo)
    if not blocks:
        return _refusal(question, blocks, trace + ["页面校验：该页没有可用的原文文本块"],
                        "该页没有可识别的原文文本块（空页），无法检索关系证据", is_demo)
    if not pid:
        return _refusal(question, blocks, trace + ["页面校验：文本块缺少 page_id"],
                        "当前页文本块缺少 page_id，无法确认证据归属", is_demo)

    qres = run_query(intent, term, list(triples), page_id=pid, entities=entities)
    raw_results = qres.get("results") or []
    pairs = [(r, Triple.from_dict(r)) for r in raw_results]
    trace.append(f"图谱查询：候选关系 {len(pairs)} 条（已按关系方向 + 主体类型 + 当前页过滤）")

    by_line: Dict[str, Tuple[int, Block]] = {}
    for i, b in enumerate(blocks):
        if b.line_id and b.line_id not in by_line:
            by_line[b.line_id] = (i, b)

    objects: List[Dict[str, Any]] = []
    evidences: List[Evidence] = []
    seen_ev = set()
    dropped: List[str] = []
    for raw, t in pairs:
        pair = by_line.get(t.line_id)
        if pair is None:
            dropped.append(f"{t.head}—{t.relation}→{t.tail}：line_id {t.line_id or '（空）'} 不在本页")
            continue
        idx, b = pair
        probs = _provenance_problems(t, pid, b)
        if probs:
            dropped.append(f"{t.head}—{t.relation}→{t.tail}：{'；'.join(probs)}")
            continue
        if _norm(t.tail) not in _norm(b.text):
            dropped.append(f"{t.head}—{t.relation}→{t.tail}：客体不在该行原文中，关系无法成立")
            continue
        support, sb, _ = _subject_support(t.head, blocks, idx)
        if not support:
            dropped.append(f"{t.head}—{t.relation}→{t.tail}：主体不在关系行、也不在紧邻上文中，归属无法确认")
            continue
        if len(objects) >= MAX_OBJECTS:
            continue
        objects.append({
            "kind": intent, "head": t.head, "relation": t.relation, "tail": t.tail,
            "subject_type": str(raw.get("subject_type") or ""),
            "page_id": pid,
            "relation_line": b.line_id, "relation_quote": b.text,
            "relation_bbox": list(b.bbox),
            "subject_line": sb.line_id if sb is not None else "",
            "subject_quote": sb.text if sb is not None else "",
            "support": support,
        })
        key = (b.line_id, "relation")
        if key not in seen_ev:
            seen_ev.add(key)
            evidences.append(_ev(b, "relation", 100.0 if support == "same_line" else 92.0, is_demo))
        if support == "carry" and sb is not None:
            skey = (sb.line_id, "subject")
            if skey not in seen_ev:
                seen_ev.add(skey)
                evidences.append(_ev(sb, "subject", 80.0, is_demo))

    if dropped:
        trace.append("出处/归属校验未通过：" + "；".join(dropped[:4]) +
                     (f" 等 {len(dropped)} 条" if len(dropped) > 4 else ""))
    if other_page_hits:
        trace.append(f"跨页命中：其他页有同名关系 {other_page_hits} 条，按单页口径不引用（未进入证据）")
    trace.append(f"关系结论：{len(objects)} 条通过校验，证据 {len(evidences)} 条（关系引文 / 主体引文）")

    if not objects:
        note = ("本页原文没有能支持该关系的证据：未命中，或主体归属 / 出处无法核验")
        return _refusal(question, blocks, trace, note, is_demo)

    return generate_answer(
        question, evidences, blocks, trace_extra=trace, is_demo=is_demo,
        answer_text=_answer_text(intent, term, objects, pid), objects=objects,
    )
