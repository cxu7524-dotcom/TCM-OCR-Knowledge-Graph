# -*- coding: utf-8 -*-
"""从原图 + 现有列槽自动触发跨槽联合取样（stage8-v4）。

与 stage8-v3 的 `tools/dual_track_stage8_joint.py` 的区别
--------------------------------------------------------
v3 的跨槽候选是**离线补修**：用冻结标注的 `gutter_x` / `main_bbox` 先定位
"哪些区段要联合"，再调用 joint_propose。这种做法能证明几何修复有效，但不能
证明"正常 OCR 入口自己会检出这一组"。

本模块把"定位"这一段改成纯图像判据，全部输入来自当前页：

  1. 相邻列槽边界 B：由 `detect()` 已算出的 `regions`（列槽）给出；
  2. 标题/断组位置：检测提案自带的 `main_bbox`（也是从原图估的）在 B 处被
     一条窄竖缝切成左右墨块，且该标题带下方第一个横向空白行给出轨顶；
  3. 纵向区段：轨顶到版框（或下一标题）；
  4. 真实中缝与左右轨：仍由 `joint_propose()` 在相邻两槽的联合区重估。

冻结标注**完全不参与**本模块；它只在预测生成之后被评分器用来匹配/核对。

保守性（19 页实测只命中丹方 p12 的 陽起石一处）
------------------------------------------------
同时要求以下七项，缺一不可（每一项都是图像几何，不是阈值凑数）：

  a. 标题带的列墨迹在 B 处有一条 0.3H–1.2H 的窄缝，B 落在缝内；
  b. 缝两侧 1.5H 内各有 >=0.6H 的墨迹跨度；
  c. 标题带下方 <=1.7H 处有横向空白行（断组位置），轨顶就在空白行之后；
  d. 标题带内行游程中位数 >=1.2H（确认是药名大字，不是正文碎片）；
  e. 同一纵段内只有这一条标题带（两侧各自另有标题 => 是两组，不是一组）；
  f. 正文区段在 B±2 的中缝低谷 <= 局部峰值 6% 且 <= 段高 5%；
  g. joint_propose 重估的缝与 B 偏差 <=1px（只认真正压在槽边界上的中缝）。

不满足就返回空列表，绝不退化成"宽范围扫描 + 后处理挑一个"。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .joint import _extent, ink_mask, joint_propose

SEAM_TOL_AUTO = 1        # 自动触发比离线补修（±3）更严：缝必须就在 B 上
TITLE_GAP_MIN_RATIO = 0.3
TITLE_GAP_MAX_RATIO = 1.2
TITLE_SPAN_MIN_RATIO = 0.6
TITLE_GLYPH_MIN_RATIO = 1.2
SECTION_GAP_MAX_RATIO = 1.7
SEAM_DEPTH_LOCAL_RATIO = 0.06
SEAM_DEPTH_SECTION_RATIO = 0.05


def _col_runs(ink: np.ndarray, x: int, y1: int, y2: int, H: float
              ) -> List[Tuple[int, int]]:
    """标题带 y 区间内在 B±3H 的列墨迹游程。"""
    xa, xb = max(0, int(x - 3 * H)), min(ink.shape[1], int(x + 3 * H))
    if y2 <= y1 or xb <= xa:
        return []
    cs = ink[y1:y2, xa:xb].sum(axis=0)
    on = cs >= max(2, 0.15 * (y2 - y1))
    runs: List[Tuple[int, int]] = []
    st: Optional[int] = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((xa + st, xa + i))
            st = None
    if st is not None:
        runs.append((xa + st, xb))
    return runs


def _gap_after(ink: np.ndarray, x1: int, x2: int, y0: int, H: float
               ) -> Optional[int]:
    """y0 之后的第一个横向空白行（2–8px），返回空白结束行（半开区间上界）。"""
    hi = min(ink.shape[0], int(y0 + SECTION_GAP_MAX_RATIO * H))
    if hi <= y0 or x2 <= x1:
        return None
    rows = ink[y0:hi, x1:x2].sum(axis=1)
    low = rows <= max(1, 0.08 * (x2 - x1))
    st: Optional[int] = None
    for i, v in enumerate(low):
        if v and st is None:
            st = i
        elif not v and st is not None:
            if 2 <= i - st <= 8:
                return y0 + i
            st = None
    if st is not None and 2 <= hi - y0 - st <= 8:
        return hi
    return None


def _title_glyph_h(ink: np.ndarray, x1: int, x2: int, y1: int, y2: int,
                   H: float) -> float:
    """标题带内行游程中位数：大字 >=1.2H，正文碎片通常 1.0H 左右。"""
    rows = ink[y1:y2, x1:x2].sum(axis=1)
    on = rows >= 2
    runs: List[int] = []
    st: Optional[int] = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append(i - st)
            st = None
    if st is not None:
        runs.append(y2 - y1 - st)
    runs = [r for r in runs if 4 <= r <= 3 * H]
    return float(np.median(runs)) if runs else 0.0


def _unique_title(groups: Sequence[Dict[str, Any]], x1: int, x2: int,
                  y1: int, y2: int) -> bool:
    """该纵段里只能有一条标题带；两侧各有一条 => 判为两个独立组。"""
    boxes = set()
    for g in groups:
        m = g.get('main_bbox')
        if not m:
            continue
        if m[0] < x2 and m[2] > x1 and max(0, min(m[3], y2) - max(m[1], y1)) > 0.3 * (y2 - y1):
            boxes.add(tuple(int(v) for v in m))
    return len(boxes) <= 1


def auto_joint_candidates(image: np.ndarray, groups: Sequence[Dict[str, Any]],
                          regions: Sequence[Dict[str, Any]], H: float,
                          bottom_limit: int) -> List[Dict[str, Any]]:
    """返回可直接追加到 `detect()['groups']` 的跨槽候选组。"""
    gray = cvt_gray(image)
    ink = ink_mask(gray)
    by_region: Dict[str, List[Dict[str, Any]]] = {}
    for g in groups:
        by_region.setdefault(str(g.get('region_id')), []).append(g)
    out: List[Dict[str, Any]] = []
    n = len(regions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = regions[i], regions[j]
            ab, bb = [int(v) for v in a['bbox']], [int(v) for v in b['bbox']]
            if ab[1] != bb[1] or ab[3] != bb[3] or ab[2] != bb[0]:
                continue
            B = ab[2]
            rid_a, rid_b = str(a['region_id']), str(b['region_id'])
            mains = set()
            for g in list(by_region.get(rid_a, [])) + list(by_region.get(rid_b, [])):
                m = g.get('main_bbox')
                if m and (abs(int(m[2]) - B) <= 3 or abs(int(m[0]) - B) <= 3):
                    mains.add(tuple(int(v) for v in m))
            for m in sorted(mains):
                cand = _one(gray, ink, groups, ab, bb, B, m, H, bottom_limit)
                if cand:
                    out.append(cand)
    # 同一槽边界的重复标题带只留证据最强的一条（缝最深、区段最长）
    dedup: Dict[int, Dict[str, Any]] = {}
    for c in out:
        B = int(c['boundary_x'])
        if B not in dedup or c['section_y'][1] > dedup[B]['section_y'][1]:
            dedup[B] = c
    return sorted(dedup.values(), key=lambda c: (c['section_y'][0], c['boundary_x']))


def _one(gray: np.ndarray, ink: np.ndarray, groups: Sequence[Dict[str, Any]],
         slot_a: Sequence[int], slot_b: Sequence[int], B: int,
         main: Sequence[int], H: float, bottom_limit: int) -> Optional[Dict[str, Any]]:
    m1, m2 = int(main[1]), int(main[3])
    if m2 <= m1 or m2 >= bottom_limit:
        return None
    runs = _col_runs(ink, B, m1, m2, H)
    gaps = [(runs[k][1], runs[k + 1][0]) for k in range(len(runs) - 1)
            if runs[k][1] <= B < runs[k + 1][0]]
    if not gaps:
        return None
    g1, g2 = min(gaps, key=lambda g: g[1] - g[0])
    if not (TITLE_GAP_MIN_RATIO * H <= g2 - g1 <= TITLE_GAP_MAX_RATIO * H):
        return None
    left = [r for r in runs if r[1] <= g1 and r[1] >= B - 1.5 * H]
    right = [r for r in runs if r[0] >= g2 and r[0] <= B + 1.5 * H]
    if not left or not right:
        return None
    x1, x2 = min(r[0] for r in left), max(r[1] for r in right)
    if x1 > B - TITLE_SPAN_MIN_RATIO * H or x2 < B + TITLE_SPAN_MIN_RATIO * H:
        return None
    top = _gap_after(ink, x1, x2, m2, H)
    if top is None or not (0.2 * H < top - m2 < SECTION_GAP_MAX_RATIO * H):
        return None
    gh = _title_glyph_h(ink, x1, x2, m1, m2, H)
    if gh < TITLE_GLYPH_MIN_RATIO * H:
        return None
    if not _unique_title(groups, x1, x2, m1, m2):
        return None
    cs = ink[top:bottom_limit, B - 2:B + 3].sum(axis=0)
    loc = ink[top:bottom_limit, max(0, B - int(1.5 * H)):
              min(ink.shape[1], B + int(1.5 * H))].sum(axis=0)
    if (cs.size == 0 or loc.size == 0
            or cs.min() > SEAM_DEPTH_LOCAL_RATIO * max(1, loc.max())
            or cs.min() > SEAM_DEPTH_SECTION_RATIO * (bottom_limit - top)):
        return None
    half = max(10, int(1.4 * H))
    le = _extent(ink, top, bottom_limit, max(0, B - half), B - 2)
    ri = _extent(ink, top, bottom_limit, B + 2, min(ink.shape[1], B + half))
    if not le or not ri:
        return None
    lw, rw = le[1] - le[0], ri[1] - ri[0]
    if min(lw, rw) < 0.7 * H or max(lw, rw) > 1.7 * min(lw, rw):
        return None
    prop = joint_propose(gray, slot_a, slot_b, [x1, m1, x2, top],
                         B, H, bottom_limit)
    if not prop or abs(int(prop['seam_x']) - B) > SEAM_TOL_AUTO:
        return None
    prop['trigger'] = {
        'slot_left': [int(v) for v in slot_a], 'slot_right': [int(v) for v in slot_b],
        'boundary_x': B, 'title_band': [x1, m1, x2, m2],
        'title_gap_x': [g1, g2], 'title_glyph_h': round(gh, 2),
        'section_y': [top, bottom_limit], 'body_valley_x': int(prop['seam_x']),
        'body_valley_depth': int(cs.min()), 'lane_widths': [lw, rw],
        'main_from_detector': list(main),
    }
    prop['evidence']['method'] = 'joint_sample_auto_from_image'
    prop['evidence']['trigger'] = prop['trigger']
    return prop


def cvt_gray(image: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
