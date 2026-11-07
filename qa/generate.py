# -*- coding: utf-8 -*-
"""
qa/generate.py —— ④ 本地生成 + 强制引用证据 + 证据回原文校验

硬性约束（改造提示词第 7 节）：输出前必须校验每条 Evidence.quote 能在 Block.text
中命中，否则转拒答。本文件是这条规则的执行点。

本阶段生成后端为 **抽取式规则桩**：只把命中的原文按编号列出，不做任何推断、
不改写原文、不给诊疗建议。本地 LLM 即使已配置，也不在本阶段参与生成
（生成链路未实现的后端不得伪造输出，见 TODO）。

TODO（下一阶段）：
  - 接入本机 Ollama / vLLM，用 prompt 约束「只能引用给定证据」，
    并对模型输出再做一次 quote 命中校验后才放行；
  - 证据冲突 / 异文的版本化呈现；
  - 答案 trace 结构化落库，供 eval/audit.py 逐片段审计。
"""

from __future__ import annotations

from typing import List, Optional

from app.config import CONFIG
from app.schemas import Answer, Block, Evidence, quote_hits_blocks

from .disclaimer import DISCLAIMER, answer_disclaimer, out_of_scope_note
from .index import VectorIndex
from .refuse import should_refuse
from .retrieve import hybrid_retrieve, retrieve_with_trace


def verify_evidences(evidences: List[Evidence],
                     blocks: List[Block]) -> tuple[List[Evidence], List[str]]:
    """逐条校验：契约字段完整 + quote 能回原文命中。返回（合格证据, trace）。"""
    ok: List[Evidence] = []
    trace: List[str] = []
    for i, e in enumerate(evidences or [], start=1):
        if not isinstance(e, Evidence):
            trace.append(f"证据[{i}] 类型不符，丢弃")
            continue
        problems = e.validate()
        if problems:
            trace.append(f"证据[{i}] 契约不全（{'；'.join(problems[:2])}），丢弃")
            continue
        if not quote_hits_blocks(e.quote, blocks):
            trace.append(f"证据[{i}] quote 未命中原文，丢弃并触发拒答")
            continue
        ok.append(e)
    trace.append(f"证据回原文核对：{len(ok)}/{len(evidences or [])} 条通过")
    return ok, trace


def _extractive_text(query: str, evidences: List[Evidence]) -> str:
    """抽取式答案模板：只列原文，不做推断。"""
    lines = [f"原文中与「{query}」相关的记载："]
    for i, e in enumerate(evidences[:CONFIG.qa.max_evidence], start=1):
        loc = f"{e.page_id} / {e.line_id}" + (f" / bbox={e.bbox}" if e.bbox else "")
        lines.append(f"[{i}] 「{e.quote}」（{loc}）")
    lines.append(answer_disclaimer())
    return "\n".join(lines)


def _refusal_text(reason: str) -> str:
    """拒答文案：明确说明无依据 / 越界，并带免责声明。"""
    if reason == out_of_scope_note():
        body = f"{reason} {DISCLAIMER}。"
    else:
        body = (f"{reason}（{DISCLAIMER}）。系统未在原文中检索到可引用的依据，"
                "因此不作答、不做任何推断。")
    return f"{body} {answer_disclaimer()}"


def generate_answer(query: str, evidences: List[Evidence], blocks: List[Block],
                    backend: Optional[str] = None,
                    trace_extra: Optional[List[str]] = None,
                    is_demo: bool = True) -> Answer:
    """生成入口（桩）。

    输入：问题、检索证据、原文块列表
    输出：Answer（结构完整，含 evidences / refused / refuse_reason / disclaimer / trace）
    异常：内部兜底，最坏情况返回合规的拒答 Answer，绝不抛异常。
    """
    trace: List[str] = list(trace_extra or [])
    backend_name = backend or CONFIG.llm.backend
    try:
        for e in evidences or []:
            if isinstance(e, Evidence):
                e.is_demo = is_demo
        verified, vtrace = verify_evidences(evidences, blocks)
        trace.extend(vtrace)
        quotes_ok = bool(verified) or not (evidences or [])
        refused, reason, rtrace = should_refuse(query, verified, quotes_verified=quotes_ok)
        trace.extend(rtrace)

        if refused:
            return Answer(
                text=_refusal_text(reason), evidences=[], refused=True,
                refuse_reason=reason, disclaimer=DISCLAIMER,
                trace=trace + [f"生成后端：{backend_name}（拒答，不调用模型）"],
                backend=backend_name, is_demo=is_demo,
            )

        if CONFIG.llm.enabled:
            trace.append("本机 LLM 已配置，但生成链路未实现；本阶段使用抽取式规则桩（不伪造模型输出）")
        else:
            trace.append("生成后端 none：抽取式原文摘录，未调用任何模型")

        return Answer(
            text=_extractive_text(query, verified),
            evidences=verified,
            refused=False, refuse_reason="", disclaimer=DISCLAIMER,
            trace=trace + ["输出边界：仅原文摘录，未做推断；已附免责声明"],
            backend="stub" if not CONFIG.llm.enabled else f"{backend_name}(stub)",
            is_demo=is_demo,
        )
    except Exception as exc:
        reason = "原文未检索到依据"
        return Answer(
            text=_refusal_text(reason), evidences=[], refused=True,
            refuse_reason=reason, disclaimer=DISCLAIMER,
            trace=trace + [f"生成异常，保守拒答：{type(exc).__name__}"],
            backend=backend_name, is_demo=is_demo,
        )


def answer(query: str, blocks: List[Block],
           vector_index: Optional[VectorIndex] = None,
           is_demo: bool = True) -> Answer:
    """端到端问答便捷入口：混合检索 → 证据校验 → 生成 / 拒答。

    main.py 的 POST /api/qa 只需调用本函数。
    """
    try:
        if vector_index is None:
            ev, trace = retrieve_with_trace(query, blocks, is_demo=is_demo)
        else:
            ev = hybrid_retrieve(query, blocks, vector_index=vector_index, is_demo=is_demo)
            trace = [f"检索词通道：混合检索（向量={'启用' if vector_index.available() else '关闭'}）",
                     f"命中证据：{len(ev)} 条"]
        return generate_answer(query, ev, blocks, trace_extra=trace, is_demo=is_demo)
    except Exception as exc:
        return Answer(
            text=_refusal_text("原文未检索到依据"), evidences=[], refused=True,
            refuse_reason="原文未检索到依据", disclaimer=DISCLAIMER,
            trace=[f"问答链路异常，保守拒答：{type(exc).__name__}"],
            backend="stub", is_demo=is_demo,
        )
