# -*- coding: utf-8 -*-
"""几何基础：半开区间 [x1,y1,x2,y2) 的框运算与严格解析。

与 tools/dual_track_stage1_data.py 的几何函数保持同一语义（有对照测试）；
这里独立实现，好处是评测器可以被单独 import，不需要数据目录存在。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

Bbox = Tuple[int, int, int, int]


class BboxError(ValueError):
    """框非法；code 给出具体原因，便于在上层汇总成明细。"""

    def __init__(self, code: str, detail: str = ''):
        super().__init__(f'{code}: {detail}' if detail else code)
        self.code = code
        self.detail = detail


def parse_bbox(value: Any) -> Bbox:
    """把任意输入解析成四个整数；非法时抛 BboxError（code 稳定可断言）。

    约定：
    - 只接受 list/tuple 且长度 4；元素必须是 int 或可精确转成 int 的有限数。
    - bool 一律拒绝（Python 里 True == 1，容易把错误数据当合法坐标）。
    - float 只接受整数值（3.0 可以，3.5 不行）；字符串只接受纯整数形式（"12" 可以，"12.0" 不行）。
    - inf / -inf / nan 一律拒绝，且不抛 OverflowError（这是第一阶段修过的坑）。
    """
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise BboxError('not_sequence', repr(value))
    if len(value) != 4:
        raise BboxError('wrong_length', f'len={len(value)}')

    out = []
    for i, v in enumerate(value):
        if isinstance(v, bool):
            raise BboxError('bool_element', f'index={i} value={v!r}')
        if isinstance(v, int):
            out.append(v)
            continue
        if isinstance(v, float):
            if not math.isfinite(v):
                raise BboxError('non_finite', f'index={i} value={v!r}')
            if v != int(v):
                raise BboxError('non_integer_float', f'index={i} value={v!r}')
            out.append(int(v))
            continue
        if isinstance(v, str):
            s = v.strip()
            try:
                out.append(int(s, 10))
            except ValueError:
                raise BboxError('unparsable_string', f'index={i} value={v!r}')
            continue
        raise BboxError('unparsable_type', f'index={i} type={type(v).__name__}')
    return (out[0], out[1], out[2], out[3])


def try_parse_bbox(value: Any) -> Tuple[Optional[Bbox], Optional[str], str]:
    """不抛异常版本：返回 (bbox|None, code|None, detail)。"""
    try:
        b = parse_bbox(value)
        return b, None, ''
    except BboxError as exc:
        return None, exc.code, exc.detail


def bbox_area(b: Sequence[int]) -> int:
    """半开区间面积；宽或高非正时为 0。"""
    w = int(b[2]) - int(b[0])
    h = int(b[3]) - int(b[1])
    return w * h if w > 0 and h > 0 else 0


def positive_area(b: Sequence[int]) -> bool:
    return bbox_area(b) > 0


def intersect_area(a: Sequence[int], b: Sequence[int]) -> int:
    w = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    h = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1]))
    return w * h if w > 0 and h > 0 else 0


def union_bbox(boxes: Sequence[Sequence[int]]) -> Optional[Bbox]:
    """外接并集框；空输入返回 None。"""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(int(b[0]) for b in boxes), min(int(b[1]) for b in boxes),
            max(int(b[2]) for b in boxes), max(int(b[3]) for b in boxes))


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    """IoU；任一方面积为 0 时返回 0.0（不抛异常、不做 epsilon 平滑）。"""
    inter = intersect_area(a, b)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(inner: Sequence[int], outer: Sequence[int]) -> float:
    """inner 有多少面积落在 outer 里；分母是 inner 面积。"""
    area = bbox_area(inner)
    return intersect_area(inner, outer) / area if area > 0 else 0.0


def center(b: Sequence[int]) -> Tuple[float, float]:
    return ((int(b[0]) + int(b[2])) / 2.0, (int(b[1]) + int(b[3])) / 2.0)


def contains_point(b: Sequence[int], x: float, y: float) -> bool:
    return int(b[0]) <= x < int(b[2]) and int(b[1]) <= y < int(b[3])


def union_ratio(box: Sequence[int], regions: Sequence[Sequence[int]]) -> float:
    """box 与 regions 并集的交叠面积 / box 面积，结果裁剪到 [0,1]。

    先用并集而不是逐个相加，避免重叠负例区域把比例算到 >1（协议要求）。
    """
    area = bbox_area(box)
    if area <= 0 or not regions:
        return 0.0
    # 逐个区域算交集并去重：用「已计入的像素行段」太复杂，这里用包含关系化简：
    # 若区域之间本身重叠，交集面积可能被重复计入；先按区域两两裁剪到 box，
    # 再用包含关系剔除被完全覆盖的区域，剩下部分按并集近似（单页区域数很少）。
    clipped = [b for b in (_clip(r, box) for r in regions) if b is not None]
    if not clipped:
        return 0.0
    # 去掉被其它区域完全包含的框（避免重复计面积）
    keep = []
    for i, b in enumerate(clipped):
        dominated = False
        for j, o in enumerate(clipped):
            if i == j:
                continue
            if intersect_area(b, o) >= bbox_area(b):
                # b 完全落在 o 内：保留面积更大者
                if bbox_area(o) > bbox_area(b) or (bbox_area(o) == bbox_area(b) and j < i):
                    dominated = True
                    break
        if not dominated:
            keep.append(b)
    covered = sum(bbox_area(b) for b in keep)
    # 若仍有相互重叠（部分重叠），用逐行并集精确计算
    if len(keep) > 1 and any(intersect_area(keep[i], keep[j]) > 0
                             for i in range(len(keep)) for j in range(i + 1, len(keep))):
        covered = _row_union_area(keep)
    return min(1.0, covered / area)


def _clip(b: Sequence[int], box: Sequence[int]) -> Optional[Bbox]:
    x1, y1 = max(int(b[0]), int(box[0])), max(int(b[1]), int(box[1]))
    x2, y2 = min(int(b[2]), int(box[2])), min(int(b[3]), int(box[3]))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _row_union_area(boxes: Sequence[Sequence[int]]) -> int:
    """按行扫描求并集面积；框数很少，行数 ≤ 页高，足够快。"""
    ys = sorted({int(b[1]) for b in boxes} | {int(b[3]) for b in boxes})
    total = 0
    for y0, y1 in zip(ys, ys[1:]):
        spans = sorted((int(b[0]), int(b[2])) for b in boxes if b[1] <= y0 and b[3] >= y1)
        if not spans:
            continue
        merged = 0
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged += cur_e - cur_s
                cur_s, cur_e = s, e
        merged += cur_e - cur_s
        total += merged * (y1 - y0)
    return total
