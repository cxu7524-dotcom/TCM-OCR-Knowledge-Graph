# -*- coding: utf-8 -*-
"""
extract/evaluate.py —— 字段级 P / R / F1 计分（口径实现，暂不产出数字）

沿用旧工程 `matcher/evaluate.py` 的「抽取与计分分离」思路，重写字段口径：
  - 对齐键： (page_id, type, name, value) 四元组，忽略抽取顺序；
  - 统计：微平均 TP / FP / FN → P / R / F1；
  - 无 ground truth、或 GT 为空、或数据带 is_demo=True → 一律返回「待测」。

严禁把 data/demo/ 的示例数据当评测结果；本文件不会读取 demo 目录。

TODO（下一阶段）：
  - 支持部分匹配（同义别名、剂量换算）与人工复核裁决记录；
  - 分类型 / 分页面输出明细，供 eval/run_all.py 汇总；
  - 与 CER、表格结构还原、可溯源率、拒答率统一报告格式。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from app.schemas import Entity

AlignKey = Tuple[str, str, str, str]


def align_key(e: Entity | Dict[str, Any], ignore_page: bool = False) -> AlignKey:
    """实体对齐键。预测与 GT 用同一口径。"""
    get = e.get if isinstance(e, dict) else lambda k, d=None: getattr(e, k, d)
    page = "" if ignore_page else str(get("page_id", "") or "")
    return (page, str(get("type", "") or ""), str(get("name", "") or ""),
            str(get("value", "") or ""))


def is_demo_record(records: Iterable[Any]) -> bool:
    """只要记录里带 is_demo=True，就拒绝计分（示例数据不是评测结果）。"""
    for r in records:
        flag = r.get("is_demo") if isinstance(r, dict) else getattr(r, "is_demo", False)
        if flag:
            return True
    return False


def field_level_prf(gold: List[Entity], pred: List[Entity],
                    ignore_page: bool = False) -> Dict[str, Any]:
    """字段级 P / R / F1。

    返回：{"status": "实测"|"待测", "P":..., "R":..., "F1":..., "TP":..., "FP":..., "FN":...,
           "reason": "..."}
    无 GT 或示例数据时 status="待测"，所有数值字段为 None（不填 0，避免被误读为实测值）。
    """
    base: Dict[str, Any] = {"status": "待测", "P": None, "R": None, "F1": None,
                            "TP": None, "FP": None, "FN": None, "reason": ""}
    if not gold:
        base["reason"] = "缺少人工标注 ground truth（data/annotation/ 为空）"
        return base
    if is_demo_record(gold) or is_demo_record(pred):
        base["reason"] = "输入含示例数据（is_demo=True），按约定不得计分"
        return base

    g = [align_key(e, ignore_page) for e in gold]
    p = [align_key(e, ignore_page) for e in pred]
    g_set, p_set = set(g), set(p)
    tp = len(g_set & p_set)
    fp = len(p_set - g_set)
    fn = len(g_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"status": "实测", "P": round(precision, 4), "R": round(recall, 4),
            "F1": round(f1, 4), "TP": tp, "FP": fp, "FN": fn,
            "reason": "微平均，按 (page_id,type,name,value) 集合对齐"}


def load_gold(path) -> List[Entity]:
    """读取标注文件；不存在 / 格式错误时返回空列表（触发「待测」）。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    items = raw.get("entities", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: List[Entity] = []
    for it in items:
        if isinstance(it, dict):
            it = dict(it)
            it.setdefault("is_demo", False)     # GT 默认不是示例数据
            out.append(Entity.from_dict(it))
    return out


def evaluate(gold_path, pred: List[Entity]) -> Dict[str, Any]:
    """评测入口：给定 GT 路径与预测实体，返回口径结果。"""
    return field_level_prf(load_gold(gold_path), pred)
