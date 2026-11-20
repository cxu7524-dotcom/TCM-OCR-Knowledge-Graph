# -*- coding: utf-8 -*-
"""
qa/rag.py —— 检索增强生成（RAG）主流程

链路：
    用户问题
      → 语义向量检索（qa/index.SemanticIndex，本机 bge-m3）
      → 相似度阈值过滤（绝对阈值 + 相对阈值，压掉低相关噪声）
      → 拼专业提示词（qa/prompt，含原文片段与引用编号）
      → 本机大模型生成（qa/llm_client，qwen2.5:7b）
      → 解析答案中的 [编号]，映射回 page_id / line_id / bbox / 原文
      → 输出 Answer（带 sources 出处、retrieval 元信息、免责声明）

三条硬约束：
  1. **有依据就一定给得出出处**：每个 [编号] 都能点回具体的书 / 页 / 行；
  2. **没依据也不装懂**：检索不到时明确写「本古籍库未检索到直接依据」，
     再给通用中医常识，不伪装成古籍原文；
  3. **输出边界不放宽**：个人诊疗、剂量、疗效类问题一律拒答。

与旧的抽取式链路的关系：本模块是**新增并行的通道**，旧链路（qa/generate.py）
保持不变，供评测与回归使用；/api/qa 默认走本模块，可用 TCM_RAG_ENABLED=0 回退。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.config import CONFIG
from app.schemas import Answer, Evidence, norm_for_match

from . import kb
from . import llm_client
from . import prompt as prompt_mod
from .disclaimer import DISCLAIMER, out_of_scope_note, wrap
from .index import get_index
from .refuse import is_out_of_scope

logger = logging.getLogger("tcm.rag")

_CITE_RE = re.compile(r"\[(\d{1,2})\]")

# 字面锚点用的"功能词"二元组：这些不算实词，命中它们不代表内容相关
_FILLER_BIGRAMS = {
    "什么", "怎么", "如何", "可以", "为什", "哪些", "介绍", "一下", "帮我", "请问",
    "的是", "是什", "么样", "怎样", "有什", "么好", "应该", "需要", "用来", "关于",
    "有关", "这个", "那个", "我们", "你们", "他们", "一个", "以及", "或者", "么办",
}


def _anchor_hits(query: str, text: str, extras: Optional[List[str]] = None) -> int:
    """查询与原文共享的实词二元组个数（去功能词），用于过滤纯语义误命中。

    extras 是查询扩展出来的古籍用词（如「失眠→不寐」），命中它们同样算锚点，
    否则扩展检索回来的段落会因为原文没有现代词而被误杀。
    """
    q = norm_for_match(query)
    t = norm_for_match(text)
    if not t:
        return 0
    n = 0
    if len(q) >= 2:
        grams = {q[i:i + 2] for i in range(len(q) - 1)} - _FILLER_BIGRAMS
        n += sum(1 for g in grams if g in t)
    for term in (extras or []):
        tt = norm_for_match(term)
        if len(tt) >= 2 and tt in t:
            n += 1
    return n


# 现代词 → 古籍用词。古籍里很少出现"失眠""腹泻"这类现代说法，
# 不做映射时向量检索会因为用词不同而漏召回。
_SYNONYMS = {
    "失眠": ["不寐", "不得眠", "目不瞑"],
    "睡眠": ["寐", "卧"],
    "腹泻": ["泄泻", "下利", "下痢", "溏泄"],
    "拉肚": ["泄泻", "下利"],
    "便秘": ["大便不通", "燥结", "不便"],
    "咳嗽": ["咳逆", "嗽"],
    "气喘": ["喘", "哮"],
    "发热": ["身热", "壮热", "潮热", "发热"],
    "发烧": ["身热", "壮热", "发热"],
    "头痛": ["头风", "头疼"],
    "头晕": ["眩", "眩晕", "冒"],
    "高血压": ["眩晕", "肝阳", "上亢"],
    "糖尿病": ["消渴", "三消"],
    "心悸": ["惊悸", "怔忡", "心动悸"],
    "呕吐": ["呕", "吐"],
    "水肿": ["浮肿", "水气", "肿满"],
    "黄疸": ["黄疸", "疸"],
    "中风": ["中風", "卒中", "偏枯"],
    "风湿": ["痹", "历节", "痺"],
    "关节痛": ["痹", "历节"],
    "月经": ["经水", "月水", "天癸", "月信"],
    "怀孕": ["妊娠", "胎", "妊"],
    "孕妇": ["妊娠", "胎前", "妊妇"],
    "产后": ["产后", "產後", "蓐"],
    "小儿": ["婴", "幼", "小儿"],
    "胃痛": ["胃脘痛", "心痛", "胃气"],
    "肚子疼": ["腹痛", "腹中痛"],
    "出血": ["血证", "吐血", "衄血", "下血"],
    "疮": ["疮疡", "痈", "疖"],
    "上火": ["火热", "火证", "热证"],
    "痰": ["痰饮", "痰"],
    "抑郁": ["郁证", "郁"],
    "水肿": ["浮肿", "水气"],
}


def _expand_query(query: str) -> Tuple[str, List[str]]:
    """返回（扩展后的检索串, 命中的古籍等价词）。"""
    q = (query or "").strip()
    extras: List[str] = []
    for modern, classics in _SYNONYMS.items():
        if modern in q:
            extras.extend(classics)
    if not extras:
        return q, []
    uniq = list(dict.fromkeys(extras))
    return f"{q} {' '.join(uniq)}", uniq


# ------------------------------------------------------------------
# 检索
# ------------------------------------------------------------------

def retrieve(query: str, top_k: Optional[int] = None,
             page_id: Optional[str] = None,
             scope: str = "library") -> Tuple[List[Tuple[object, float]], Dict[str, object]]:
    """语义检索 + 阈值过滤。返回（命中段落+分数, 检索元信息）。"""
    k = int(top_k or CONFIG.rag.top_k)
    meta: Dict[str, object] = {
        "mode": "semantic-rag",
        "embed_model": CONFIG.embed.model,
        "llm_model": CONFIG.llm.model,
        "top_k": k,
        "min_score": CONFIG.rag.min_score,
        "scope": scope,
        "hits": 0,
        "top_score": 0.0,
        "has_evidence": False,
    }
    idx = get_index()
    st = idx.status()
    meta["index"] = {"vectors": st.get("vectors", 0), "stale": st.get("stale", True),
                     "dim": st.get("dim", 0), "health": st.get("health", "")}
    if not CONFIG.rag.enabled or not idx.available():
        meta["note"] = "语义索引不可用，本次未做向量检索"
        return [], meta

    try:
        search_q, extras = _expand_query(query)
        if extras:
            meta["expanded_terms"] = extras
        raw = idx.search(search_q, top_k=max(k * 3, 12))
    except Exception as exc:
        meta["note"] = f"检索异常：{type(exc).__name__}"
        return [], meta

    if scope == "page" and page_id:
        raw = [(ps, s) for ps, s in raw if getattr(ps, "page_id", "") == page_id]

    if not raw:
        meta["note"] = "知识库中未检索到任何段落"
        return [], meta

    top = float(raw[0][1])
    floor = max(float(CONFIG.rag.min_score), top * float(CONFIG.rag.relative_keep))
    above = [(ps, s) for ps, s in raw if float(s) >= floor]

    # 字面锚点：语义分数过线还不够，必须至少共享一个实词二元组。
    # 目的是压掉"读起来语义相近、实际毫无关系"的误命中（如"写一首春天的诗"命中古籍）。
    kept: List[Tuple[object, float]] = []
    dropped = 0
    for ps, s in above:
        if _anchor_hits(query, getattr(ps, "text", ""), extras) >= 1:
            kept.append((ps, s))
        else:
            dropped += 1
    kept = kept[:k]
    meta.update({
        "hits": len(kept),
        "top_score": round(top, 4),
        "floor": round(floor, 4),
        "has_evidence": bool(kept),
        "candidates": len(raw),
        "anchor_dropped": dropped,
    })
    if not kept:
        if above and dropped:
            meta["note"] = (f"{len(above)} 条过相似度阈值，但均无实词锚点重合，"
                            "判定为知识库无相关原文")
        else:
            meta["note"] = (f"最高相似度 {top:.2f} 低于阈值 {floor:.2f}，"
                            "判定为知识库无相关原文")
    return kept, meta


# ------------------------------------------------------------------
# 证据与出处
# ------------------------------------------------------------------

def _flat(text: str) -> str:
    """展示用：把原文里的换行压成空格，保持内容不变但排版连续。"""
    return re.sub(r"\s*\n\s*", " ", str(text or "")).strip()


def _passage_payload(ps, score: float, index: int, cited: bool) -> Dict[str, object]:
    """把段落整理成前端可直接渲染的出处对象。"""
    blocks = list(getattr(ps, "blocks", []) or [])
    first = blocks[0] if blocks else None
    last = blocks[-1] if blocks else None
    line_span = ""
    if first is not None:
        line_span = first.line_id if (last is None or last.line_id == first.line_id) \
            else f"{first.line_id}–{last.line_id}"
    return {
        "index": index,
        "book": getattr(ps, "book", ""),
        "page_id": getattr(ps, "page_id", ""),
        "line_id": getattr(first, "line_id", ""),
        "line_span": line_span,
        "quote": _flat(getattr(ps, "text", "")),
        "bbox": list(getattr(first, "bbox", []) or []),
        "source_hash": getattr(first, "source_hash", ""),
        "score": round(float(score), 4),
        "cited": bool(cited),
        "blocks": len(blocks),
    }


def _evidence_of(ps, score: float, is_demo: bool) -> Optional[Evidence]:
    """取段落首块作为可溯源证据（quote 必须是原文片段，能回原文命中）。"""
    blocks = [b for b in (getattr(ps, "blocks", []) or []) if getattr(b, "text", "")]
    if not blocks:
        return None
    b = blocks[0]
    return Evidence(
        page_id=getattr(b, "page_id", getattr(ps, "page_id", "")),
        bbox=list(getattr(b, "bbox", []) or []),
        line_id=getattr(b, "line_id", ""),
        quote=getattr(b, "text", ""),
        source_hash=getattr(b, "source_hash", ""),
        score=round(float(score), 4),
        block_ref=getattr(b, "block_ref", "") or "",
        channel="vector", role="relation", is_demo=is_demo,
    )


def _verify(evidence: Evidence, passages: List) -> bool:
    """轻量回原文校验：quote 必须能在所属段落原文中命中。"""
    q = norm_for_match(evidence.quote)
    if not q:
        return False
    for ps in passages:
        if getattr(ps, "page_id", "") == evidence.page_id and \
                q in norm_for_match(getattr(ps, "text", "")):
            return True
    return False


def _cited_indexes(text: str, upper: int) -> List[int]:
    """从答案里解析 [编号]，去重保序，只保留有效范围。"""
    out: List[int] = []
    for m in _CITE_RE.finditer(text or ""):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if 1 <= n <= upper and n not in out:
            out.append(n)
    return out


def _clean_citations(text: str, upper: int) -> str:
    """清理答案里的引用标注，保证"标了 [n] 就真的有第 n 条出处"：

      1. 超出来源范围的 [n]（模型幻觉出的编号）直接删除；
      2. 「（据中医常识）」等常识句里的 [n] 删除——常识不是古籍原文，不许引用。
    """
    if not text:
        return text
    # 按句切分（保留分隔符），常识句整体去引
    parts = re.split(r"(?<=[。！？；\n])", text)
    out: List[str] = []
    for seg in parts:
        if "中医常识" in seg or "据中医常识" in seg:
            seg = _CITE_RE.sub("", seg)
        else:
            seg = _CITE_RE.sub(
                lambda m: m.group(0) if 1 <= int(m.group(1)) <= upper else "", seg)
        out.append(seg)
    return "".join(out)


def _extractive_fallback(hits: List[Tuple[object, float]]) -> str:
    """模型不可用时的降级答案：只列原文，不编造。"""
    lines = ["以下是古籍中与该问题相关的原文："]
    for i, (ps, score) in enumerate(hits[:CONFIG.rag.max_citations], start=1):
        label = ps.cite_label() if hasattr(ps, "cite_label") else getattr(ps, "page_id", "")
        lines.append(f"[{i}] 《{getattr(ps, 'book', '')}》{label}：{_flat(getattr(ps, 'text', ''))}"
                     f"（相似度 {float(score):.2f}）")
    lines.append("（本机生成模型当前不可用，以上仅为原文摘录，未做解释。）")
    return "\n".join(lines)


def _norm_refusal(query: str, is_demo: bool, reason: str, trace: List[str]) -> Answer:
    return Answer(
        text=wrap(reason, "refusal"), evidences=[], refused=True, refuse_reason=reason,
        disclaimer=DISCLAIMER, trace=trace, backend="rag", is_demo=is_demo,
        sources=[], retrieval={"mode": "semantic-rag", "refused": True, "reason": reason},
    )


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def rag_answer(query: str, *, history: Optional[List[dict]] = None,
               page_id: Optional[str] = None, scope: str = "library",
               is_demo: Optional[bool] = None) -> Answer:
    """RAG 问答主入口。任何异常都返回合规的降级 Answer，不抛。"""
    q = (query or "").strip()
    trace: List[str] = []
    if is_demo is None:
        is_demo = bool(CONFIG.demo_mode)

    # ① 输出边界：个人诊疗 / 剂量 / 疗效类，一律拒答（不进入检索，也不交给模型）
    if q and is_out_of_scope(q):
        trace.append("命中越界规则（个人诊疗 / 疗效 / 用药剂量）→ 拒答")
        return _norm_refusal(q, is_demo, out_of_scope_note(), trace)

    try:
        # ② 语义检索
        hits, meta = retrieve(q, page_id=page_id, scope=scope)
        trace.append(
            f"语义检索：知识库 {meta.get('index', {}).get('vectors', 0)} 段向量，"
            f"候选 {meta.get('candidates', 0)}，命中 {meta.get('hits', 0)}，"
            f"最高相似度 {float(meta.get('top_score') or 0):.2f}，"
            f"阈值 {float(meta.get('floor') or CONFIG.rag.min_score):.2f}"
        )
        if meta.get("note"):
            trace.append(str(meta["note"]))

        passages = [ps for ps, _ in hits]
        has_ev = bool(hits) and bool(meta.get("has_evidence"))

        # ③ 无证据：按配置给「常识作答」或拒答
        if not has_ev:
            if not CONFIG.rag.answer_without_evidence:
                return _norm_refusal(q, is_demo, "原文未检索到依据", trace +
                                     ["未开启常识作答，按约定拒答"])
            text = llm_client.chat_messages(
                prompt_mod.build_open_messages(q, history),
                temperature=0.3, max_tokens=700)
            if not text:
                return _norm_refusal(q, is_demo,
                                     "本机大模型不可用，且古籍库未检索到依据", trace +
                                     ["无证据 + 模型不可用 → 拒答"])
            trace.append("无原文依据：已按「通用中医常识」模式作答并显式声明")
            return Answer(
                text=wrap(text, "generated_open"), evidences=[], refused=False, refuse_reason="",
                disclaimer=DISCLAIMER, trace=trace, backend="rag-open", is_demo=is_demo,
                sources=[],
                retrieval={**meta, "answered_without_evidence": True},
            )

        # ④ 有证据：RAG 生成
        text = llm_client.chat_messages(
            prompt_mod.build_messages(q, hits, history),
            temperature=0.2, max_tokens=900)
        if not text:
            text = _extractive_fallback(hits)
            trace.append("本机大模型不可用：降级为原文摘录（不编造解释）")
            backend = "rag-extractive"
        else:
            trace.append(f"本机大模型已生成答案（{CONFIG.llm.model}）")
            backend = "rag"

        text = _clean_citations(text, len(hits))
        cited = _cited_indexes(text, len(hits))
        trace.append(f"答案引用编号：{cited or '未显式标注'}")

        # ⑤ 出处：被引用的排前面，其余作为「备选出处」一并返回
        ordered = [i for i in cited] + [i for i in range(1, len(hits) + 1) if i not in cited]
        sources: List[Dict[str, object]] = []
        evidences: List[Evidence] = []
        for i in ordered[:max(1, CONFIG.rag.max_citations)]:
            ps, score = hits[i - 1]
            sources.append(_passage_payload(ps, score, i, cited and i in cited))
            ev = _evidence_of(ps, score, is_demo)
            if ev is not None and _verify(ev, [ps]):
                evidences.append(ev)

        trace.append(f"出处回映射：{len(sources)} 条（其中 {len(cited)} 条被答案显式引用）")
        trace.append("输出边界：已附免责声明；不给诊断 / 剂量 / 疗效承诺")

        return Answer(
            text=wrap(text, "generated" if backend == "rag" else "extractive"),
            evidences=evidences, refused=False, refuse_reason="",
            disclaimer=DISCLAIMER, trace=trace, backend=backend, is_demo=is_demo,
            sources=sources,
            retrieval={**meta, "cited": cited, "answered_without_evidence": False},
        )
    except Exception as exc:
        logger.warning("RAG 链路异常：%s", exc)
        return _norm_refusal(q, is_demo, "问答链路异常，未能给出可靠答案",
                             trace + [f"异常：{type(exc).__name__}"])


def rag_status() -> dict:
    """知识库与索引状态，供 /api/rag/status 与前端徽标使用。"""
    idx = get_index()
    stats = kb.library_stats()
    return {
        "enabled": bool(CONFIG.rag.enabled),
        "embed": {"backend": CONFIG.embed.backend, "model": CONFIG.embed.model,
                  "enabled": CONFIG.embed.enabled},
        "llm": {"backend": CONFIG.llm.backend, "model": CONFIG.llm.model,
                "enabled": CONFIG.llm.enabled},
        "library": stats,
        "index": idx.status(),
        "params": {"top_k": CONFIG.rag.top_k, "min_score": CONFIG.rag.min_score,
                   "relative_keep": CONFIG.rag.relative_keep,
                   "answer_without_evidence": CONFIG.rag.answer_without_evidence},
    }
