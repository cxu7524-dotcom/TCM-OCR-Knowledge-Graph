# -*- coding: utf-8 -*-
"""
qa/disclaimer.py —— 免责文案与输出边界（唯一收口）

所有页面、接口响应、报告里出现的免责声明都从这里取，避免各处文案不一致。
边界：本系统只做「原文摘录与检索」，不做诊断、不给剂量建议、不推断疗效。

TODO（下一阶段）：
  - 按赛事 / 主管部门要求补充更完整的合规声明与版本号；
  - 输出边界规则改为可配置策略文件，并纳入评测（越界问题拒答率）。
"""

from __future__ import annotations

from app.config import CONFIG

DISCLAIMER = CONFIG.disclaimer                    # 仅供参考、非诊疗建议
BANNER = f"【免责声明】{DISCLAIMER}：本系统为赛事研发原型，不构成诊断、用药或治疗建议。"
OUT_OF_SCOPE_NOTE = (
    "该问题涉及个人诊疗、疗效或用药剂量建议，超出本原型输出边界；"
    "系统只提供原文摘录，不提供任何医疗建议。"
)
NO_EVIDENCE_NOTE = "原文未检索到依据"


def answer_disclaimer() -> str:
    """答案尾部固定附加的免责文案。"""
    return f"{DISCLAIMER}。以上为原文摘录，未做推断。"


def out_of_scope_note() -> str:
    return OUT_OF_SCOPE_NOTE


def wrap(text: str) -> str:
    """保证任何对外文本都带免责声明，不重复追加。"""
    t = (text or "").strip()
    if DISCLAIMER in t:
        return t
    return (t + " " + answer_disclaimer()).strip()


def page_banner() -> dict:
    """给前端常驻横幅用的结构化文案。"""
    return {
        "banner": BANNER,
        "disclaimer": DISCLAIMER,
        "badge": CONFIG.demo_badge,
        "out_of_scope_note": OUT_OF_SCOPE_NOTE,
        "no_evidence_note": NO_EVIDENCE_NOTE,
    }
