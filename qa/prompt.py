# -*- coding: utf-8 -*-
"""
qa/prompt.py —— RAG 提示词工程

目标：把"原文摘录"升级为"能讲明白的回答"，同时保留可溯源的硬约束。

三条设计原则：
  1. **先给结论、再给解释**：不说"根据提供的资料"这类套话，直接回答问题；
  2. **区分依据层级**：有原文就标 [编号]，没有原文而用常识补充就标"（据中医常识）"，
     让读者一眼看出哪句是古籍原文支撑、哪句是模型补充；
  3. **守住输出边界**：不给个人诊断、不给剂量、不承诺疗效——这是医疗类系统不可让渡的底线。

引用格式固定为 `[1]`、`[2]`…，与检索到的片段顺序一一对应，便于上层把引用
映射回 page_id / line_id / bbox，实现"答案可点回原文"。
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

MAX_QUOTE_CHARS = 220

SYSTEM_PROMPT = (
    "你是一位精通中医古籍文献的资深专家，擅长把文言原文准确、通俗地讲清楚。\n"
    "\n"
    "【回答要求】\n"
    "1. 用现代规范中文作答，专业但通俗，让没有中医背景的读者也能看懂。"
    "先一句话给出结论，再展开解释，不要用「根据您提供的资料」这类套话开头。\n"
    "2. 凡是依据【原文片段】得出的结论，在句末用 [编号] 标注，例如 [1]；"
    "一句话有多个依据可标 [1][3]。\n"
    "3. 原文片段不足以回答时，可以结合中医学常识补充，但这部分必须以「（据中医常识）」"
    "开头标明，不得与原文依据混为一谈。\n"
    "4. 严禁给出个人诊断结论、处方剂量、疗效承诺，或建议替代就医；"
    "遇到这类问题，明确说明本系统只做文献检索与知识整理，不提供诊疗建议。\n"
    "5. 不要复述问题，不要罗列检索过程，直接给答案。\n"
    "6. 篇幅控制在 350 字以内；必要时用分点，但不要滥用标题。\n"
    "7. 不要主动给出具体用量数字（如几两、几钱、几克）；原文里的剂量只能作为"
    "原文内容原样引用，绝不可改写成用药建议或「通常用量」。\n"
)

EXPERT_SYSTEM = (
    "你是一位精通中医古籍文献的资深专家，现在用「专家模式」作答：读者是同行，能读懂术语。\n"
    "\n"
    "【回答要求】\n"
    "1. 第一句先给结论，然后分条展开，不超过四条，每条以「·」开头，条内标注 [编号]。\n"
    "2. 若多处原文说法不同，单列一条「异文」说明各书差异，各自标注 [编号]；"
    "只有一处依据时不必写这一条。\n"
    "3. 凡依据原文的结论必须标 [编号]；靠中医学常识补充的部分以「（据中医常识）」开头。\n"
    "4. 不给个人诊断结论、处方剂量、疗效承诺；原文里的剂量只能原样引用。\n"
    "5. 篇幅 500 字以内，不复述问题，不罗列检索过程。\n"
)


EXPERT_CHECK_SYSTEM = (
    "你是审稿人，负责核对一份答案是否被原文支持。\n"
    "只做一件事：找出答案里「原文片段并不支持」的断言。不要重写答案，不要解释理由。\n"
    "\n"
    "【判定尺度】（这条最重要，直接决定标不标）\n"
    "· 句子是对原文的归纳、改写、换用现代说法，只要意思对得上，就算有据，不要挑措辞；\n"
    "· 句子带 [编号]，且那一段确实谈到这件事，就算有据；\n"
    "· 只有「原文完全没提」或「与原文相反」的断言才判无据；\n"
    "· 已经写明「（据中医常识）」的句子不用标——作者自己声明过不是原文。\n"
    "\n"
    "【输出格式】（严格遵守）\n"
    "· 若全部句子都能被原文片段支持，只输出一行：全部有据\n"
    "· 否则每句一行，格式：无据|<这一句，原样照抄，不要改写>\n"
    "· 最多 3 行；宁可不标，不要错标。\n"
)


NO_EVIDENCE_SYSTEM = (
    "你是一位精通中医古籍文献的资深专家。\n"
    "\n"
    "【本次情况】古籍库中没有检索到与该问题直接相关的原文片段。\n"
    "\n"
    "【回答要求】\n"
    "1. 第一句固定写：「本古籍库未检索到直接依据。」写完空一行再继续。\n"
    "2. 先判断问题是否属于中医 / 健康范畴：\n"
    "   · 属于：用现代规范中文、通俗地解释清楚，先给结论再展开，不要用套话；\n"
    "   · 不属于：用一句话说明本系统是中医古籍问答助手、该问题超出范围，"
    "然后从中医或健康角度给出简洁克制的关联说明或追问；"
    "不要勉强作答，更不要把自己写的内容说成「中医常识」。\n"
    "3. 全文不得出现 [编号] 形式的引用标注（因为没有原文可引）。\n"
    "4. 严禁给出个人诊断结论、处方剂量、疗效承诺，或建议替代就医。\n"
    "5. 篇幅控制在 300 字以内。\n"
)


def _clip(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    t = re.sub(r"\s*\n\s*", " ", str(text or "")).strip()
    return t if len(t) <= limit else t[:limit] + "……"


def format_passages(passages: Sequence) -> str:
    """把检索到的段落排成【原文片段】块，编号从 1 开始。"""
    lines: List[str] = []
    for i, item in enumerate(passages, start=1):
        ps = item[0] if isinstance(item, (tuple, list)) else item
        score = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
        head = f"[{i}] 《{getattr(ps, 'book', '')}》 {getattr(ps, 'page_id', '')} " \
               f"{'/'.join(getattr(ps, 'line_ids', [])[:3])}"
        if score is not None:
            head += f"（相似度 {float(score):.2f}）"
        lines.append(head)
        lines.append(_clip(getattr(ps, "text", "")))
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(query: str, passages: Sequence,
                   history: Optional[List[dict]] = None,
                   style: str = "standard") -> List[dict]:
    """有证据时的消息序列（检索增强）。style=expert 时换用专家提示词。"""
    body = format_passages(passages)
    user = (
        f"【原文片段】\n{body}\n\n"
        f"【用户问题】\n{query}\n\n"
        "请严格依据上面的原文片段回答；片段不足时按第 3 条用常识补充并注明。"
    )
    system = EXPERT_SYSTEM if str(style).lower() == "expert" else SYSTEM_PROMPT
    return _compose(system, history, user)


def build_check_messages(query: str, passages: Sequence, answer: str) -> List[dict]:
    """专家模式第二遍：把第一遍答案送去核对，只找「原文不支持」的句子。"""
    body = format_passages(passages)
    user = (
        f"【原文片段】\n{body}\n\n"
        f"【问题】\n{query}\n\n"
        f"【待审答案】\n{answer}\n\n"
        "请按格式逐句核对。"
    )
    return _compose(EXPERT_CHECK_SYSTEM, None, user)


def build_open_messages(query: str, history: Optional[List[dict]] = None) -> List[dict]:
    """无证据时的消息序列（常识作答，明确无依据）。"""
    user = f"【用户问题】\n{query}\n\n请按上述要求作答。"
    return _compose(NO_EVIDENCE_SYSTEM, history, user)


def _compose(system: str, history: Optional[List[dict]], user: str) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": system}]
    for turn in (history or [])[-6:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": content[:800]})
    msgs.append({"role": "user", "content": user})
    return msgs
