# -*- coding: utf-8 -*-
"""
eval/metrics.py —— 五项指标口径定义（**本阶段一律「待测」**）

严格约束：禁止编造 OCR 准确率、F1、可溯源率、拒答率等实测数字；
         禁止把 data/demo/ 的示例数据当评测结果。本文件不读取 demo 目录。

五项指标与口径：
  1. CER 字符错误率        编辑距离(识别, GT) / GT 字数，按页加权
  2. 字段级 P / R / F1     按 (page_id,type,name,value) 对齐，微平均
  3. 表格结构还原准确率    行列结构 + 单元格内容同时正确才算对
  4. 可溯源率              quote 能回原文命中的证据数 / 证据总数
  5. 拒答率                无依据问题中正确拒答数 / 无依据问题总数

TODO（下一阶段）：补齐 GT 后实现各 compute_*，并由 run_all.py 汇总；
     CER 需引入编辑距离实现（标准库即可），表格结构需定义行列对齐规则。
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.config import CONFIG
from app.schemas import Answer, Block, quote_hits_blocks

METRIC_SPECS: List[Dict[str, Any]] = [
    {
        "key": "cer", "name": "CER 字符错误率", "status": "待测", "value": None,
        "formula": "Σ 编辑距离(识别文本, 人工转写) / Σ 人工转写字数（按页加权）",
        "needs": ["data/annotation/ 逐页人工转写 GT", "真实 OCR 输出（非示例数据）"],
    },
    {
        "key": "field_f1", "name": "字段级 P / R / F1", "status": "待测", "value": None,
        "formula": "按 (page_id,type,name,value) 集合对齐，TP/FP/FN 微平均",
        "needs": ["实体与关系人工标注 GT", "真实抽取输出"],
    },
    {
        "key": "table_structure", "name": "表格结构还原准确率", "status": "待测", "value": None,
        "formula": "行列结构与单元格内容同时正确的表数 / GT 表数",
        "needs": ["方剂表 / 剂量表结构 GT", "真实表格还原输出"],
    },
    {
        "key": "traceability", "name": "可溯源率", "status": "待测", "value": None,
        "formula": "quote 能回原文命中的证据数 / 证据总条数",
        "needs": ["真实问答日志（非示例数据）", "对应原文 Block"],
    },
    {
        "key": "refusal", "name": "拒答率", "status": "待测", "value": None,
        "formula": "无依据问题中正确拒答数 / 无依据问题总数",
        "needs": ["人工构造的无依据问题集及期望标签", "真实问答日志"],
    },
]


def has_ground_truth() -> bool:
    """标注目录里是否存在可用的 GT 文件（只认 .jsonl / .json，排除 README）。"""
    d = CONFIG.paths.annotation_dir
    if not d.exists():
        return False
    return any(p.suffix.lower() in (".json", ".jsonl") for p in d.iterdir() if p.is_file())


def missing_requirements() -> List[Dict[str, Any]]:
    """列出每项指标当前缺什么，供 run_all.py 报告输出。"""
    out = []
    for spec in METRIC_SPECS:
        out.append({"key": spec["key"], "name": spec["name"],
                    "missing": list(spec["needs"])})
    return out


def audit_sample_count(answers: List[Answer]) -> Dict[str, int]:
    """统计可用于审计的问答样本数量（演示样本单独计数，不计入指标）。"""
    demo = sum(1 for a in answers if getattr(a, "is_demo", True))
    return {"total": len(answers), "demo": demo, "real": len(answers) - demo}


def compute_traceability_for_audit(answer: Answer, blocks: List[Block]) -> Dict[str, Any]:
    """单条答案的证据命中明细（**审计明细，不是指标结论**）。

    返回命中数 / 总数；调用方不得把它写进「五项指标」报告当作实测值。
    """
    evs = answer.evidences or []
    hit = 0
    for e in evs:
        if quote_hits_blocks(e.quote, blocks):
            hit += 1
    return {"total": len(evs), "hit": hit, "is_demo": True,
            "note": "审计明细，示例数据不计入指标"}


def report_metrics() -> Dict[str, Any]:
    """评测报告中的五项指标。当前阶段全部「待测」，不含任何数字。"""
    return {
        "phase": "框架搭建阶段（模型未接入、标注未采集）",
        "policy": "禁止编造实测数字；示例数据不得当评测结果",
        "ground_truth_available": has_ground_truth(),
        "metrics": [
            {"key": s["key"], "name": s["name"], "status": "待测",
             "value": None, "formula": s["formula"], "needs": s["needs"]}
            for s in METRIC_SPECS
        ],
    }
