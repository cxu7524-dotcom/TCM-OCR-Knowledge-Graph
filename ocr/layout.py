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

from app.config import CONFIG
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


def _stable_columns(items: Sequence[Block],
                    gap_ratio: float = COL_GAP_RATIO) -> List[Dict[str, Any]]:
    """阶段七 P1 的**对照实现**：稳定的列聚类，返回与 `project_columns()` 相同结构。

    注意：这是 A/B 用过的候选方案，实测在 4 页上把阅读顺序改坏（严格 CER 变差、
    顺序无关 CER 不变），**主链路已回退到 `project_columns()`**；本函数保留供回归对照，
    不在 `analyze_layout()` 中调用。

    与旧实现的差别：旧实现把列中心更新为「成员均值」，密集页上会随扫描逐步向左漂移，
    把相邻两列串成一列。这里用**最右成员的中心作固定锚**，并限制单列最大横向跨度，
    避免链式合并；最后按列中心从右向左编号 col_id。
    """
    if not items:
        return []
    boxes = [_box(b) for b in items]
    widths = [max(1.0, x2 - x1) for x1, _, x2, _ in boxes]
    threshold = max(12.0, _median(widths, 40.0) * gap_ratio)
    max_span = 1.5 * threshold
    order = sorted(range(len(items)),
                   key=lambda i: -((boxes[i][0] + boxes[i][2]) / 2))
    cols: List[Dict[str, Any]] = []
    for idx in order:
        cx = (boxes[idx][0] + boxes[idx][2]) / 2
        if cols and 0 <= (cols[-1]["anchor"] - cx) <= max_span:
            cols[-1]["members"].append(idx)
        else:
            cols.append({"anchor": cx, "center": cx, "members": [idx]})
    cols.sort(key=lambda c: -c["anchor"])
    for i, c in enumerate(cols):
        c["col_id"] = i
        c["center"] = c["anchor"]
        xs1 = [boxes[m][0] for m in c["members"]]
        xs2 = [boxes[m][2] for m in c["members"]]
        hs = [max(1.0, boxes[m][3] - boxes[m][1]) for m in c["members"]]
        c["x1"], c["x2"] = min(xs1), max(xs2)
        c["aspect_median"] = round(_median(hs, 40.0) /
                                   max(1.0, _median([boxes[m][2] - boxes[m][0]
                                                     for m in c["members"]], 40.0)), 3)
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


def _structure_signals(items: Sequence[Block]) -> Dict[str, Any]:
    """阶段七新增：对 det 碎片鲁棒的无监督结构信号（只用 bbox，不读 GT / CER）。

    - aspect_median：框高宽比**中位数**。碎片再多，中位数不会被「宽扁框占比」稀释。
    - col_gap_cv  ：投影列中心间距的变异系数，越小越像等距竖列（古籍界行）。
    - box_count / columns：密集页的规模。
    """
    if not items:
        return {"aspect_median": 0.0, "col_gap_cv": 1.0,
                "box_count": 0, "columns": 0, "text_len_cv": 1.0}
    boxes = [_box(b) for b in items]
    aspects = [max(1.0, y2 - y1) / max(1.0, x2 - x1) for x1, y1, x2, y2 in boxes]
    cols = project_columns(items)
    centers = [c["center"] for c in cols]           # 已从右向左排好
    gaps = [centers[i] - centers[i + 1] for i in range(len(centers) - 1) if centers[i] > centers[i + 1]]
    if gaps and statistics.mean(gaps) > 0:
        gap_cv = statistics.pstdev(gaps) / statistics.mean(gaps)
    else:
        gap_cv = 1.0
    lens = [len(b.text or "") for b in items]
    len_cv = (statistics.pstdev(lens) / statistics.mean(lens)
              if lens and statistics.mean(lens) > 0 else 1.0)
    return {
        "aspect_median": round(float(statistics.median(aspects)), 3),
        "col_gap_cv": round(float(gap_cv), 3),
        "box_count": len(items),
        "columns": len(cols),
        "text_len_cv": round(float(len_cv), 3),
    }


def _image_column_signal(image_path: str) -> Dict[str, Any]:
    """阶段七新增：图像墨迹层的竖列证据（完全绕开 det 框）。

    `ocr/vertical.py:page_layout()` 用二值化 + 垂直投影直接找物理竖列，
    不受「det 把竖列切成 60~90 个水平碎片」影响。无图像路径时返回全 0。
    """
    out = {"img_columns": 0, "img_col_aspect_median": 0.0, "img_vertical": False}
    if not image_path:
        return out
    try:
        from .vertical import page_layout
        cols = page_layout(image_path)["cols"]
        if not cols:
            return out
        aspects = [max(1.0, c["bbox"][3] - c["bbox"][1]) /
                   max(1.0, c["bbox"][2] - c["bbox"][0]) for c in cols]
        med = float(statistics.median(aspects))
        out.update({
            "img_columns": len(cols),
            "img_col_aspect_median": round(med, 3),
            "img_vertical": (len(cols) >= CONFIG.ocr.dir_min_img_cols
                             and med >= CONFIG.ocr.dir_min_img_aspect),
        })
    except Exception:
        pass                                   # 图像不可读时按「无证据」处理，不抛异常
    return out


def detect_direction_detail(blocks: Sequence[Block],
                            image_path: str = "") -> Dict[str, Any]:
    """可解释的方向判别（阶段七 P0）。

    `TCM_OCR_DIR_RULE=old` 时与阶段六**完全一致**：竖长条框占比 ≥0.30 且投影列数 ≥3。
    `=new`（默认）时追加两条对 det 碎片鲁棒的无监督信号，任一命中即判竖排：

      ① 图像墨迹层检出 ≥5 条规则竖列（列高宽比中位 ≥3）——物理证据，最强；
      ② 密集页（≥20 框）投影出 ≥8 列且列间距变异系数 ≤0.5 —— 等距界行结构。

    所有阈值来自 `app/config.py` 的 `TCM_OCR_DIR_*`，只依据检测框 / 图像几何，
    **不读 GT、不读 CER、不读标注坐标**。返回字段保留阶段六全部键，追加新信号明细。
    """
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
    old_ok = (ratio >= 0.30 and cols >= 3)
    rule = (CONFIG.ocr.dir_rule or "old").strip().lower()

    struct = _structure_signals(items)
    img = _image_column_signal(image_path) if rule == "new" else {
        "img_columns": 0, "img_col_aspect_median": 0.0, "img_vertical": False}
    dense_ok = (struct["box_count"] >= CONFIG.ocr.dir_min_boxes
                and struct["columns"] >= CONFIG.ocr.dir_min_cols
                and struct["col_gap_cv"] <= CONFIG.ocr.dir_max_gap_cv)
    if rule == "new":
        direction = "vertical-rtl" if (old_ok or img["img_vertical"] or dense_ok) else "horizontal-ltr"
    else:
        direction = "vertical-rtl" if old_ok else "horizontal-ltr"

    hits = []
    if old_ok:
        hits.append("旧判据(竖长条框占比)")
    if img["img_vertical"]:
        hits.append(f"图像层竖列({img['img_columns']}列/高宽比{img['img_col_aspect_median']})")
    if dense_ok:
        hits.append(f"密集规则列({struct['columns']}列/gap_cv={struct['col_gap_cv']})")
    reason = (f"竖长条框 {vertical}/{len(items)}（{ratio:.2%}），检测到 {cols} 列；"
              f"rule={rule}，命中[{'、'.join(hits) or '无'}]→{direction}")
    return {
        "direction": direction,
        "vertical_ratio": round(ratio, 4),
        "vertical_boxes": vertical,
        "total_boxes": len(items),
        "columns": cols,
        "headers": headers,
        "thresholds": {"vertical_aspect": VERTICAL_ASPECT,
                       "min_vertical_ratio": 0.30, "min_columns": 3,
                       "header_aspect": HEADER_ASPECT,
                       "dir_rule": rule,
                       "min_img_cols": CONFIG.ocr.dir_min_img_cols,
                       "min_img_aspect": CONFIG.ocr.dir_min_img_aspect,
                       "min_boxes": CONFIG.ocr.dir_min_boxes,
                       "min_cols": CONFIG.ocr.dir_min_cols,
                       "max_gap_cv": CONFIG.ocr.dir_max_gap_cv},
        # 阶段七追加字段（旧报告读不到也不影响，新报告可逐页复核）
        "rule": rule,
        "aspect_median": struct["aspect_median"],
        "col_gap_cv": struct["col_gap_cv"],
        "text_len_cv": struct["text_len_cv"],
        "box_count": struct["box_count"],
        "img_columns": img["img_columns"],
        "img_col_aspect_median": img["img_col_aspect_median"],
        "img_vertical": img["img_vertical"],
        "dense_regular": dense_ok,
        "reason": reason,
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
                   merge: bool = True, image_path: str = "") -> Tuple[List[Block], LayoutResult]:
    """版面分析入口（阶段二；阶段七增 `image_path` 与稳定列聚类）。

    输入：OCR 识别出的 Block 列表（真实 bbox），可选图像路径（用于图像层竖列证据）
    输出：(处理后的 Block 列表, LayoutResult)
    异常：内部兜底，返回结构完整的结果，不抛异常。
    """
    items = [b for b in blocks if isinstance(b, Block)]
    try:
        detail = detect_direction_detail(items, image_path=image_path)
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

        # 阶段七 P1 竖列重组：`project_columns()` 按列中心 x 从右向左分配 col_id，
        # `merge_fragments()` 合并同列相邻碎片（quote 按 y 顺序拼接、bbox 取并集、conf 取最小），
        # `assign_reading_order()` 在 vertical-rtl 分支按 (col_id, y, x) 排序。
        # 曾试过「固定锚点不漂移」的 `_stable_columns()`，A/B 显示它在 4 页上把阅读顺序
        # 明显改坏（严格 CER 变差而顺序无关 CER 不变），已回退；函数保留供回归对照。
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
