# -*- coding: utf-8 -*-
"""
ocr/layout.py —— 版面分析（阶段二）：竖排列聚类、阅读顺序、方向可解释判别

本模块只使用 **OCR 检测框的真实 bbox**，不读取任何 GT / 标注坐标，因此可用于真实样本。

链路：
  1. project_columns()   x 投影 + 自适应间隙阈值 → 列聚类，从右向左编号 col_id；
  2. assign_order()      列内按 y 升序，列间按 col_id，得到 reading_order；
  3. merge_fragments()   同列被检测切碎的框（列内断行）合并，保留 bbox 并集与原文 quote；
  4. detect_direction()  基于「竖长条框占比 + 列数」的可解释判别，附 evidence。

局限（如实登记）：小字夹注、眉批、朱批与正文的层级关系本阶段只按 y 间隙粗分，
页眉页脚靠「宽高比远大于阈值」剔除；跨列续句只做同列合并，不做跨列拼接。
单元测试见 tests/test_layout.py。
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from app.schemas import Block, LayoutResult, source_hash_of

VERTICAL_ASPECT = 1.5          # 竖长条框判定：h / w >= 1.5
HEADER_ASPECT = 4.0            # 页眉 / 页脚判定：w / h >= 4.0
COL_GAP_RATIO = 0.8           # 列聚类阈值 = 中位框宽 × 该系数
MERGE_GAP_RATIO = 0.8         # 同列合并阈值 = 中位框高 × 该系数


def _box(b: Block) -> Tuple[float, float, float, float]:
    try:
        return tuple(float(v) for v in b.bbox[:4])  # type: ignore[return-value]
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0, 0.0)


def _median(values: Sequence[float], default: float) -> float:
    vals = [v for v in values if v > 0]
    return float(statistics.median(vals)) if vals else default


def project_columns(blocks: Sequence[Block],
                    gap_ratio: float = COL_GAP_RATIO) -> List[Dict[str, Any]]:
    """按 bbox 的 x 投影把文本块聚成列，从右向左编号 col_id=0,1,2...

    输入：OCR 检测得到的 Block 列表（真实 bbox）
    输出：[{col_id, x1, x2, center, members:[block 下标], aspect_median}]
    """
    items = [b for b in blocks if isinstance(b, Block)]
    if not items:
        return []
    boxes = [_box(b) for b in items]
    widths = [max(1.0, x2 - x1) for x1, _, x2, _ in boxes]
    threshold = max(12.0, _median(widths, 40.0) * gap_ratio)

    # 按 x 中心从右向左扫描，相邻中心差 <= threshold 归为同列
    order = sorted(range(len(items)),
                   key=lambda i: -((boxes[i][0] + boxes[i][2]) / 2))
    cols: List[Dict[str, Any]] = []
    for idx in order:
        cx = (boxes[idx][0] + boxes[idx][2]) / 2
        placed = False
        for c in cols:
            if abs(cx - c["center"]) <= threshold:
                c["members"].append(idx)
                # 用成员均值更新列中心（避免长列漂移）
                c["center"] = sum((boxes[m][0] + boxes[m][2]) / 2
                                  for m in c["members"]) / len(c["members"])
                placed = True
                break
        if not placed:
            cols.append({"col_id": len(cols), "center": cx, "members": [idx]})

    cols.sort(key=lambda c: -c["center"])
    for i, c in enumerate(cols):
        c["col_id"] = i
        xs1 = [boxes[m][0] for m in c["members"]]
        xs2 = [boxes[m][2] for m in c["members"]]
        heights = [max(1.0, boxes[m][3] - boxes[m][1]) for m in c["members"]]
        c["x1"], c["x2"] = min(xs1), max(xs2)
        c["aspect_median"] = round(_median(heights, 40.0) / max(1.0, _median(widths, 40.0)), 3)
    return cols


def group_columns(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    """阶段一兼容接口：按 col_id 分组给出每列范围。"""
    groups: Dict[int, List[Block]] = {}
    for b in blocks:
        if isinstance(b, Block):
            groups.setdefault(int(b.col_id or 0), []).append(b)
    out = []
    for cid in sorted(groups):
        boxes = [_box(b) for b in groups[cid]]
        out.append({"col_id": cid,
                    "x_range": [min(x[0] for x in boxes), max(x[2] for x in boxes)],
                    "y_range": [min(x[1] for x in boxes), max(x[3] for x in boxes)],
                    "block_count": len(boxes)})
    return out


def detect_direction_detail(blocks: Sequence[Block]) -> Dict[str, Any]:
    """可解释的方向判别：竖长条框占比 + 列数 + 页眉数。"""
    items = [b for b in blocks if isinstance(b, Block)]
    if not items:
        return {"direction": "horizontal-ltr", "vertical_ratio": 0.0, "columns": 0,
                "headers": 0, "reason": "无文本块，默认横排"}
    vertical = headers = 0
    for b in items:
        x1, y1, x2, y2 = _box(b)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if h / w >= VERTICAL_ASPECT:
            vertical += 1
        if w / h >= HEADER_ASPECT:
            headers += 1
    ratio = vertical / len(items)
    cols = len(project_columns(items))
    direction = "vertical-rtl" if (ratio >= 0.30 and cols >= 3) else "horizontal-ltr"
    return {
        "direction": direction,
        "vertical_ratio": round(ratio, 4),
        "vertical_boxes": vertical,
        "total_boxes": len(items),
        "columns": cols,
        "headers": headers,
        "thresholds": {"vertical_aspect": VERTICAL_ASPECT,
                       "min_vertical_ratio": 0.30, "min_columns": 3,
                       "header_aspect": HEADER_ASPECT},
        "reason": (f"竖长条框 {vertical}/{len(items)}（{ratio:.2%}），检测到 {cols} 列；"
                   f"占比≥0.30 且列数≥3 判为竖排从右向左，否则横排"),
    }


def detect_direction(blocks: Iterable[Block]) -> str:
    """阶段一兼容接口，返回方向字符串；明细见 detect_direction_detail()。"""
    return detect_direction_detail(list(blocks))["direction"]


def assign_reading_order(blocks: Iterable[Block],
                         direction: str = "vertical-rtl") -> List[Block]:
    """填写 reading_order 并返回排序后的 Block 列表。"""
    items = [b for b in blocks if isinstance(b, Block)]
    if direction == "horizontal-ltr":
        items.sort(key=lambda b: (_box(b)[1], _box(b)[0]))
    else:
        items.sort(key=lambda b: (int(b.col_id or 0), _box(b)[1], _box(b)[0]))
    for i, b in enumerate(items, start=1):
        b.reading_order = i
    return items


def merge_fragments(blocks: Sequence[Block],
                    gap_ratio: float = MERGE_GAP_RATIO) -> List[Block]:
    """同列内被检测器切碎的相邻框合并（列内断行 / 跨行续字）。

    合并条件：同 col_id、y 间隙 <= 中位框高 × gap_ratio、x 区间有重叠。
    合并后 quote 为原文按 y 顺序拼接，bbox 取并集，conf 取最小值（保守）。
    """
    items = [b for b in blocks if isinstance(b, Block)]
    if len(items) < 2:
        return list(items)
    boxes = [_box(b) for b in items]
    med_h = _median([y2 - y1 for _, y1, _, y2 in boxes], 40.0)
    threshold = med_h * gap_ratio

    buckets: Dict[int, List[int]] = {}
    for i, b in enumerate(items):
        buckets.setdefault(int(b.col_id or 0), []).append(i)
    out: List[Block] = []
    for cid in sorted(buckets):
        idxs = sorted(buckets[cid], key=lambda i: boxes[i][1])
        group: List[int] = [idxs[0]]
        for i in idxs[1:]:
            prev = boxes[group[-1]]
            cur = boxes[i]
            gap = cur[1] - prev[3]
            overlap = min(cur[2], prev[2]) - max(cur[0], prev[0]) > 0
            if gap <= threshold and overlap:
                group.append(i)
            else:
                out.append(_merge_group(items, boxes, group, cid))
                group = [i]
        out.append(_merge_group(items, boxes, group, cid))
    return out


def _merge_group(items: List[Block], boxes: List[Tuple[float, ...]],
                 idxs: List[int], cid: int) -> Block:
    if len(idxs) == 1:
        b = items[idxs[0]]
        b.col_id = cid
        return b
    ordered = sorted(idxs, key=lambda i: boxes[i][1])
    text = "".join(items[i].text for i in ordered)
    x1 = min(boxes[i][0] for i in ordered)
    y1 = min(boxes[i][1] for i in ordered)
    x2 = max(boxes[i][2] for i in ordered)
    y2 = max(boxes[i][3] for i in ordered)
    first = items[ordered[0]]
    conf = min(float(items[i].conf or 0) for i in ordered)
    return Block(
        page_id=first.page_id, bbox=[int(x1), int(y1), int(x2), int(y2)],
        line_id=first.line_id, quote=text,
        source_hash=source_hash_of("ocr", first.page_id, first.line_id, text),
        text=text, text_norm=text, kind=first.kind, conf=round(conf, 4),
        col_id=cid, reading_order=first.reading_order,
        block_ref=first.block_ref,
    )


def analyze_layout(blocks: Iterable[Block], page_id: str = "",
                   direction: str | None = None, is_demo: bool = True,
                   merge: bool = True) -> Tuple[List[Block], LayoutResult]:
    """版面分析入口（阶段二）。

    输入：OCR 识别出的 Block 列表（真实 bbox）
    输出：(处理后的 Block 列表, LayoutResult)
    异常：内部兜底，返回结构完整的结果，不抛异常。
    """
    items = [b for b in blocks if isinstance(b, Block)]
    try:
        detail = detect_direction_detail(items)
        direct = direction or detail["direction"]

        # 若上游（ocr/vertical.py）已给出列结构，直接消费，不重复聚类、不再合并碎片
        given = {int(b.col_id or 0) for b in items}
        if len(given) >= 3 and max(given) >= 2:
            gc = group_columns(items)
            items = assign_reading_order(items, direct)
            lay = LayoutResult(
                page_id=page_id,
                cols=[{"col_id": g["col_id"], "x_range": g["x_range"],
                       "block_count": g["block_count"]} for g in gc],
                block_order=[b.block_ref or b.line_id for b in items],
                direction=direct, engine="layout-rules", is_demo=is_demo, detail=detail,
                note=f"采用上游列结构（{len(given)} 列），未重复聚类；{detail['reason']}",
            )
            return items, lay

        cols = project_columns(items)
        col_of = {}
        for c in cols:
            for idx in c["members"]:
                col_of[id(items[idx])] = c["col_id"]
        for b in items:
            b.col_id = col_of.get(id(b), 0)
            # 页眉 / 页脚（宽高比很大的横排块）单独置为 -1 列，阅读顺序排在最前
            bx1, by1, bx2, by2 = _box(b)
            if (bx2 - bx1) / max(1.0, by2 - by1) >= HEADER_ASPECT:
                b.col_id = -1
        if merge:
            items = merge_fragments(items)
        items = assign_reading_order(items, direct)
        lay = LayoutResult(
            page_id=page_id,
            cols=[{"col_id": c["col_id"], "x_range": [c["x1"], c["x2"]],
                   "block_count": len(c["members"]), "aspect_median": c["aspect_median"]}
                  for c in cols],
            block_order=[b.block_ref or b.line_id for b in items],
            direction=direct, engine="layout-rules", is_demo=is_demo,
            detail=detail,
            note=f"基于检测框 bbox 的 x 投影列聚类；{detail['reason']}",
        )
        return items, lay
    except Exception as exc:
        lay = LayoutResult(page_id=page_id, engine="layout-rules", is_demo=is_demo,
                           note=f"版面分析降级：{type(exc).__name__}")
        return list(items), lay
