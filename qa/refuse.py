# -*- coding: utf-8 -*-
"""
qa/refuse.py —— 证据不足 / 越界问题的拒答判定

硬性要求（改造提示词第 4.5 条）：问答无证据时必须拒答并说明「原文未检索到依据」，
禁止模型自由发挥。本模块只做**判定**，Answer 的构造交给 qa/generate.py。

判定顺序：
  1. 越界（个人诊疗 / 疗效 / 用药剂量建议）→ 拒答，说明输出边界；
  2. 证据条数 < min_evidence → 拒答「原文未检索到依据」；
  3. 最高分证据 < min_score → 拒答「原文未检索到依据」；
  4. 证据 quote 回原文命中失败 → 拒答（由 generate.py 传入校验结果）。

TODO（下一阶段）：
  - 用人工构造的「无依据问题集」实测拒答率与误拒率（见 eval/metrics.py）；
  - 越界规则改成可审计的策略表，避免正则误伤正常的文献提问。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from app.config import CONFIG
from app.schemas import Evidence

from .disclaimer import NO_EVIDENCE_NOTE, OUT_OF_SCOPE_NOTE

NO_EVIDENCE_REASON = NO_EVIDENCE_NOTE

# 个人诊疗 / 疗效 / 用药建议类：一律拒答，不进入生成
OUT_OF_SCOPE_PATTERNS = [
    r"治好|治愈|根治|疗效|有效率|治愈率",
    r"能不能治|能否治|可不可以治|管用吗|有用吗",
    r"我该|我要|给我开|帮我开|推荐.*药|该吃多少|吃几|服用多少|吃多久",
    r"副作用|会不会伤|孕妇能吃|小孩能吃|剂量给|处方建议",
]


def is_out_of_scope(query: str) -> bool:
    """是否超出「原文摘录」的输出边界。"""
    if not query:
        return False
    return any(re.search(p, query) for p in OUT_OF_SCOPE_PATTERNS)


def should_refuse(query: str, evidences: List[Evidence],
                  quotes_verified: bool = True,
                  min_evidence: int | None = None,
                  min_score: float | None = None) -> Tuple[bool, str, List[str]]:
    """拒答判定入口。

    返回 (是否拒答, 原因, trace)。任何异常都保守地判为拒答，不抛异常。
    """
    trace: List[str] = []
    try:
        evs = [e for e in (evidences or []) if isinstance(e, Evidence)]
        need_n = int(min_evidence if min_evidence is not None else CONFIG.qa.min_evidence)
        need_s = float(min_score if min_score is not None else CONFIG.qa.min_score)
        trace.append(f"证据 {len(evs)} 条，要求 ≥{need_n} 条；最高分 "
                     f"{max([e.score for e in evs], default=0):.2f}，要求 ≥{need_s:.2f}")

        if is_out_of_scope(query):
            trace.append("命中越界规则（个人诊疗 / 疗效 / 用药建议）")
            return True, OUT_OF_SCOPE_NOTE, trace
        if not CONFIG.qa.refuse_on_no_evidence:
            trace.append("拒答开关已按配置关闭（仅用于调试，不应在演示中开启）")
            return False, "", trace
        if len(evs) < need_n:
            trace.append("无有效证据 → 拒答")
            return True, NO_EVIDENCE_REASON, trace
        if max(e.score for e in evs) < need_s:
            trace.append("最高分低于阈值 → 拒答")
            return True, NO_EVIDENCE_REASON, trace
        if not quotes_verified:
            trace.append("证据摘抄回原文核对失败 → 拒答")
            return True, "证据摘抄未能回原文核对，按约定拒答", trace
        trace.append("证据充分且可溯源 → 允许作答（仍只输出原文摘录）")
        return False, "", trace
    except Exception as exc:
        return True, f"拒答判定异常，保守拒答：{type(exc).__name__}", trace
