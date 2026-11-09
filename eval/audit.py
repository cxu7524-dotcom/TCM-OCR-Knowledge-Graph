# -*- coding: utf-8 -*-
"""
eval/audit.py —— 零模型确定性反向溯源审计

继承自旧工程 `agent/quality.py` 的核心思想并改造口径：
    数据库 / 答案里的摘抄 → 回原文逐片段核对 → 输出可追溯状态。
    全过程只用归一化比对与子串匹配，不调用任何模型，同样输入必然得出同样结论。

口径（与旧「保真度」对应，但对象换成答案证据）：
    quote 按分隔符拆片段
      全部片段命中原文 = traced（完全可溯）
      部分命中         = partial
      全不命中         = miss
      无摘抄           = no_quote
    审计明细比率 = traced / (total - no_quote)

**重要**：这里的比率是「链路自检明细」，不是五项评测指标里的「可溯源率」。
示例数据带 is_demo=True，一律排除在指标汇总之外（见 eval/metrics.py）。

TODO（下一阶段）：
  - 片段切分支持古籍夹注、双行小字与异体字差异；
  - 与 SQLite 里的 blocks 表直连做全库审计；
  - 输出逐条 HTML / CSV 明细供人工复核。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

from app.schemas import Answer, Block, norm_for_match, quote_hits_blocks

SPLIT_RE = re.compile(r"[\s，。、；：,.;:（）()《》「」『』]+")


def split_quote(quote: str) -> List[str]:
    """把摘抄拆成可逐段回原文核对的片段。"""
    return [p for p in SPLIT_RE.split(norm_for_match(quote or "")) if p]


@dataclass
class AuditItem:
    quote: str = ""
    page_id: str = ""
    line_id: str = ""
    status: str = "no_quote"        # traced | partial | miss | no_quote
    hit: int = 0
    total: int = 0
    is_demo: bool = True


@dataclass
class AuditReport:
    total: int = 0
    traced: int = 0
    partial: int = 0
    miss: int = 0
    no_quote: int = 0
    audit_ratio: float | None = None
    items: List[AuditItem] = field(default_factory=list)
    is_demo: bool = True
    note: str = "零模型确定性审计明细；示例数据不计入评测指标（指标一律待测）"


def _fragment_hit(fragment: str, blocks: List[Block]) -> bool:
    """片段是否在任一原文块中命中（归一化后双向子串判定）。"""
    f = norm_for_match(fragment)
    if not f:
        return False
    for b in blocks:
        t = norm_for_match(b.text)
        if t and (f in t or t in f):
            return True
    return False


def audit_quote(quote: str, blocks: List[Block], page_id: str = "",
                line_id: str = "") -> AuditItem:
    """单条摘抄的反向核对。"""
    frags = split_quote(quote)
    if not frags:
        return AuditItem(quote=quote or "", page_id=page_id, line_id=line_id,
                         status="no_quote")
    hit = sum(1 for f in frags if _fragment_hit(f, blocks))
    if hit == len(frags):
        status = "traced"
    elif hit > 0:
        status = "partial"
    else:
        status = "miss"
    return AuditItem(quote=quote, page_id=page_id, line_id=line_id,
                     status=status, hit=hit, total=len(frags))


def audit_answer(answer: Answer, blocks: List[Block]) -> AuditReport:
    """一条答案的全部证据逐片段回原文核对。"""
    items: List[AuditItem] = []
    for e in (answer.evidences or []):
        items.append(audit_quote(e.quote, blocks, e.page_id, e.line_id))
    return _summarize(items, is_demo=getattr(answer, "is_demo", True))


def audit_quotes(quotes: Iterable[str], blocks: List[Block]) -> AuditReport:
    """批量核对摘抄列表（不依赖 Answer 对象）。"""
    items = [audit_quote(q, blocks) for q in quotes]
    return _summarize(items)


def _summarize(items: List[AuditItem], is_demo: bool = True) -> AuditReport:
    traced = sum(1 for i in items if i.status == "traced")
    partial = sum(1 for i in items if i.status == "partial")
    miss = sum(1 for i in items if i.status == "miss")
    no_quote = sum(1 for i in items if i.status == "no_quote")
    denom = len(items) - no_quote
    return AuditReport(
        total=len(items), traced=traced, partial=partial, miss=miss,
        no_quote=no_quote,
        audit_ratio=round(traced / denom, 4) if denom > 0 else None,
        items=items, is_demo=is_demo,
    )


def audit_qa_logs(records: List[Dict], blocks_by_page: Dict[str, List[Block]]) -> Dict:
    """从**真实问答日志**做反向溯源审计，并与 compute_traceability 交叉核对。

    只统计 is_demo=False 的非拒答答案；逐片段回原文核对，输出
    完全 / 部分 / 未溯 三态与「整串 quote 命中数」，供报告交叉验证。
    """
    from app.schemas import quote_hits_blocks

    total = traced = partial = miss = no_quote = whole_hit = 0
    per_case = []
    for rec in records or []:
        if not isinstance(rec, dict) or rec.get("is_demo") is True or rec.get("refused"):
            continue
        page_blocks = blocks_by_page.get(str(rec.get("page_id", "")), [])
        # 证据可能来自任意页，按证据自身的 page_id 找原文
        for ev in rec.get("evidences") or []:
            if not isinstance(ev, dict) or ev.get("is_demo") is True:
                continue
            total += 1
            pool = blocks_by_page.get(str(ev.get("page_id", "")), []) or page_blocks
            item = audit_quote(str(ev.get("quote", "")), pool,
                               str(ev.get("page_id", "")), str(ev.get("line_id", "")))
            traced += int(item.status == "traced")
            partial += int(item.status == "partial")
            miss += int(item.status == "miss")
            no_quote += int(item.status == "no_quote")
            whole_hit += int(quote_hits_blocks(str(ev.get("quote", "")), pool))
        per_case.append({"question": rec.get("question", ""),
                         "evidence_count": len(rec.get("evidences") or [])})
    return {
        "total": total, "traced": traced, "partial": partial,
        "miss": miss, "no_quote": no_quote,
        "whole_quote_hit": whole_hit,
        "cases": len(per_case),
        "audit_ratio": round(traced / total, 4) if total else None,
        "whole_hit_rate": round(whole_hit / total, 4) if total else None,
        "cross_check": ("逐片段完全可溯数与整串 quote 命中数的差异，"
                        "反映片段切分与整串命中两种口径；compute_traceability 采用整串命中口径"),
        "is_demo": False,
        "note": "基于真实问答日志（合成集，非命题单位样本、非官方评测结果）",
    }


def self_check(blocks: List[Block], queries: List[str]) -> Dict:
    """链路自检：跑几条例示问题，输出审计明细（明确标注示例、非指标）。

    延迟引入 qa 模块，保证本文件在没有 FastAPI 的命令行环境也能独立 import。
    """
    from qa.generate import answer as qa_answer

    reports = []
    for q in queries:
        ans = qa_answer(q, blocks)
        rep = audit_answer(ans, blocks)
        reports.append({
            "question": q,
            "refused": ans.refused,
            "refuse_reason": ans.refuse_reason,
            "evidence_count": len(ans.evidences),
            "quote_hit_all": all(quote_hits_blocks(e.quote, blocks) for e in ans.evidences),
            "audit": asdict(rep),
            "is_demo": True,
        })
    return {
        "note": "示例数据 · 非命题单位样本；以下为链路自检明细，不是评测指标",
        "status": "待测",
        "cases": reports,
    }
