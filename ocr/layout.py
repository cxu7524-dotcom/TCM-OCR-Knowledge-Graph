# -*- coding: utf-8 -*-
"""
ocr/layout.py —— 版面分析：竖排列切分与阅读顺序

中医古籍多为**竖排、从右向左**阅读，与普通横排 OCR 的顺序相反。
本模块负责把识别出的文本块重新排成正确阅读顺序，输出列结构与排序结果。

本阶段为**规则桩**：依据每个 Block 的 bbox 与 col_id 做确定性排序，
不加载任何版面检测模型；结果里显式标注 engine="stub" / is_demo=True。

TODO（下一阶段）：
  - 接入版面检测（列分割线、书口、天头地脚、批注与小字双行）；
  - 处理夹注、眉批、朱批与正文的层级关系，输出 parent_id；
  - 列内断行合并、跨列续句判定；
  - 与真实 OCR 的 bbox 坐标系对齐并补单元测试。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.schemas import Block, LayoutResult


def group_columns(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    """按 col_id 分组，给出每列的 x 范围、y 范围与块数（确定性、零模型）。

    输入：带 bbox 的 Block 列表
    输出：[{col_id, x_range:[x1,x2], y_range:[y1,y2], block_count}]
    """
    groups: Dict[int, List[Block]] = {}
    for b in blocks:
        if not isinstance(b, Block):
            continue
        groups.setdefault(int(b.col_id or 0), []).append(b)
    cols: List[Dict[str, Any]] = []
    for cid in sorted(groups):
        items = groups[cid]
        try:
            x1 = min(float(b.bbox[0]) for b in items)
            y1 = min(float(b.bbox[1]) for b in items)
            x2 = max(float(b.bbox[2]) for b in items)
            y2 = max(float(b.bbox[3]) for b in items)
        except (IndexError, TypeError, ValueError):
            x1 = y1 = x2 = y2 = 0.0
        cols.append({"col_id": cid, "x_range": [x1, x2],
                     "y_range": [y1, y2], "block_count": len(items)})
    return cols


def assign_reading_order(blocks: Iterable[Block], direction: str = "vertical-rtl") -> List[Block]:
    """给 Block 填 reading_order 并返回排序后的列表。

    vertical-rtl：先按 col_id 升序（右侧为第 0 列），列内按 y 升序；
    horizontal-ltr：先按 y 升序，行内按 x 升序。
    """
    items = [b for b in blocks if isinstance(b, Block)]
    try:
        if direction == "horizontal-ltr":
            items.sort(key=lambda b: (float(b.bbox[1]), float(b.bbox[0])))
        else:
            items.sort(key=lambda b: (int(b.col_id or 0), float(b.bbox[1]), float(b.bbox[0])))
    except (IndexError, TypeError, ValueError):
        items.sort(key=lambda b: (int(b.col_id or 0), str(b.line_id)))
    for i, b in enumerate(items, start=1):
        b.reading_order = i
    return items


def analyze_layout(blocks: Iterable[Block], page_id: str = "",
                   direction: str = "vertical-rtl") -> LayoutResult:
    """版面分析入口（桩）。

    输入：OCR 或示例数据给出的 Block 列表
    输出：LayoutResult（cols / block_order / direction / engine / is_demo / note）
    异常：内部兜底，任何情况下返回结构完整的 LayoutResult，不抛异常。
    """
    blocks = [b for b in blocks if isinstance(b, Block)]
    try:
        cols = group_columns(blocks)
        ordered = assign_reading_order(blocks, direction)
        return LayoutResult(
            page_id=page_id,
            cols=cols,
            block_order=[b.block_ref or b.line_id for b in ordered],
            direction=direction,
            engine="stub",
            is_demo=True,
            note="规则桩：仅按 bbox/col_id 排序，版面检测模型未接入（示例数据 / 模型未接入）",
        )
    except Exception as exc:  # 桩不得抛异常
        return LayoutResult(page_id=page_id, engine="stub", is_demo=True,
                            note=f"版面分析降级为空结果：{type(exc).__name__}")


def detect_direction(blocks: Iterable[Block]) -> str:
    """极简方向判别桩：列数多于行数时判为竖排，否则横排。真实场景需模型。"""
    items = [b for b in blocks if isinstance(b, Block)]
    n_col = len({b.col_id for b in items})
    n_line = len({b.line_id for b in items})
    return "vertical-rtl" if n_col >= 2 and n_col >= n_line / 2 else "horizontal-ltr"
