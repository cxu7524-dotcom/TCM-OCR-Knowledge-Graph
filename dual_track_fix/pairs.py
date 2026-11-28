# -*- coding: utf-8 -*-
"""stage8 双轨检测：空白纵向走廊 = 中缝；走廊两侧墨迹外接框 = 左右轨。

与阶段 5 的差别（本轮唯一改动）：确认判据从**逐窗字形几何状态机**换成
**空白走廊结构**：

  1. 把墨迹取反得到空白面，用高度 0.85H 的竖直线段做形态学开运算——
     只有「连续空出一个字高度以上」的竖条才会留下；
  2. 这些竖条就是中缝走廊。走廊宽度 ≤0.22H 且两侧紧邻有墨才算数
     （真缝 1–4px ≈0.1–0.2H；跨条目的栏间空档 6–7px ≈0.3H，会被卡掉）；
  3. 走廊高度就是这一组的纵向跨度——左轨先结束、右轨继续时仍是一段
     （P39-02 左轨「製」31px、右轨「各二錢」83px，同属一段）；
  4. 跨度内取走廊两侧的墨迹外接框：左轨框、右轨框（原图半开区间）。

为什么这样能补上召回：阶段 5 要求「连续 3 个有效窗 + 每侧 ≥3 个独立字形」，
而思濟堂各页的剂量注只有 1–3 个字，结构上永远过不了；本模块只看
「有没有一条缝 + 两侧各有没有一个字」。

坐标：原图半开区间 [x1,y1,x2,y2)。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_CONFIG


# ---------------------------------------------------------------- 基础工具

def page_ink(gray: np.ndarray, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """与阶段 5 同一套 ink_mask（adaptiveThreshold GAUSSIAN, invert）。"""
    import cv2
    c = (cfg or DEFAULT_CONFIG)['ink']
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               int(c['block_size']), int(c['c']))
    return bw > 0


def rule_mask(lines: Iterable[Dict[str, Any]], shape: Tuple[int, int]) -> np.ndarray:
    """栏线/界行掩膜（True = 直线结构）；只用于审计，不参与配对。"""
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=bool)
    for ln in lines or []:
        for seg in (ln.get('segments') or [ln]):
            x0 = max(0, int(seg.get('x0', 0))); x1 = min(w, int(seg.get('x1', 0)))
            y0 = max(0, int(seg.get('y0', 0))); y1 = min(h, int(seg.get('y1', 0)))
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = True
    return out


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _containment(inner: Sequence[int], outer: Sequence[int]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix2 - ix1) * (iy2 - iy1) / area if area > 0 else 0.0


def _ink_bbox(ink: np.ndarray, xa: int, xb: int, y0: int, y1: int) -> Optional[List[int]]:
    win = ink[y0:y1, xa:xb]
    ys, xs = np.nonzero(win)
    if ys.size == 0:
        return None
    return [int(xs.min()) + xa, int(ys.min()) + y0,
            int(xs.max()) + 1 + xa, int(ys.max()) + 1 + y0]


# ---------------------------------------------------------------- 走廊

def seam_corridors(ink: np.ndarray, H: float, cfg: Dict[str, Any]
                   ) -> List[Dict[str, Any]]:
    """空白竖条（高 ≥0.85H、宽 ≤0.22H、两侧紧邻有墨）= 中缝候选。"""
    import cv2
    p = cfg['seam_pair']
    min_run = max(3, int(np.ceil(p['min_run_H'] * H)))
    max_w = max(1, int(round(p['max_corridor_H'] * H)))
    ker = np.ones((min_run, 1), np.uint8)
    op = cv2.morphologyEx((~ink).astype(np.uint8), cv2.MORPH_OPEN, ker)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(op, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > max_w or h < min_run:
            continue
        if w * h > p['corridor_max_fill'] * area and area < p['corridor_min_area']:
            continue
        # 两侧紧邻必须有墨（否则是栏间空档/页边空白）
        lcol = ink[y:y + h, max(0, x - p['flank_px']):x]
        rcol = ink[y:y + h, x + w:x + w + p['flank_px']]
        if lcol.size == 0 or rcol.size == 0:
            continue
        if lcol.mean() < p['flank_ratio'] or rcol.mean() < p['flank_ratio']:
            continue
        out.append({'x0': int(x), 'x1': int(x + w), 'y0': int(y), 'y1': int(y + h),
                    'width': int(w), 'height': int(h), 'area': int(area)})
    out.sort(key=lambda r: (-r['height'], r['x0']))
    return out


def _track_window(prof: np.ndarray, seam: int, side: str, H: float,
                  cfg: Dict[str, Any]) -> Optional[List[int]]:
    """从缝向一侧扩张到墨量降到峰值 35% 以下的位置，得到该轨列窗。"""
    p = cfg['seam_pair']
    w = prof.shape[0]
    reach = int(round(p['reach_H'] * H))
    if side == 'left':
        lo, hi = max(0, seam - reach), seam
        if hi - lo < 2:
            return None
        seg = prof[lo:hi]
        peak = float(seg.max()) if seg.size else 0.0
        if peak <= 0:
            return None
        thr = max(1.0, p['track_edge_ratio'] * peak)
        x = hi - 1
        while x >= lo and prof[x] >= thr:
            x -= 1
        return [x + 1, hi]
    lo, hi = seam, min(w, seam + reach)
    if hi - lo < 2:
        return None
    seg = prof[lo:hi]
    peak = float(seg.max()) if seg.size else 0.0
    if peak <= 0:
        return None
    thr = max(1.0, p['track_edge_ratio'] * peak)
    x = lo
    while x < hi and prof[x] >= thr:
        x += 1
    return [lo, x]


def refine_corridor(ink: np.ndarray, c: Dict[str, Any], H: float,
                    cfg: Dict[str, Any], half: str) -> Dict[str, Any]:
    p = cfg['seam_pair']
    y0, y1 = c['y0'], c['y1']
    xl, xr = c['x0'], c['x1']
    seam = (xl + xr) // 2
    sub = ink[y0:y1]
    prof = sub.sum(axis=0).astype(float)
    lw = _track_window(prof, xl, 'left', H, cfg)
    rw = _track_window(prof, xr, 'right', H, cfg)
    rec = {'seam_x': seam, 'corridor': [xl, y0, xr, y1], 'span': [y0, y1],
           'corridor_width': c['width'], 'half': half,
           'track_window_left': lw, 'track_window_right': rw}
    if lw is None or rw is None:
        rec['reject'] = ['no_track_window']
        return rec
    lb = _ink_bbox(ink, lw[0], lw[1], y0, y1)
    rb = _ink_bbox(ink, rw[0], rw[1], y0, y1)
    rec['left'], rec['right'] = lb, rb
    if lb is None or rb is None:
        rec['reject'] = ['missing_side_ink']
        return rec
    lwid, rwid = lb[2] - lb[0], rb[2] - rb[0]
    lhei, rhei = lb[3] - lb[1], rb[3] - rb[1]
    rec['group_bbox'] = [min(lb[0], rb[0]), min(lb[1], rb[1]),
                         max(lb[2], rb[2]), max(lb[3], rb[3])]
    why = []
    if lwid < p['track_width_min_H'] * H or rwid < p['track_width_min_H'] * H:
        why.append('track_too_narrow')
    if lwid > p['track_width_max_H'] * H or rwid > p['track_width_max_H'] * H:
        why.append('track_too_wide')
    if lhei < p['side_min_height_H'] * H or rhei < p['side_min_height_H'] * H:
        why.append('side_too_short')
    if max(lwid, rwid) / max(1, min(lwid, rwid)) > p['width_ratio_max']:
        why.append('width_ratio')
    if why:
        rec['reject'] = why
    return rec


def detect_page(ink: np.ndarray, H: float, halves: Sequence[Sequence[int]],
                cfg: Optional[Dict[str, Any]] = None,
                bad: Optional[np.ndarray] = None) -> Dict[str, Any]:
    cfg = cfg or DEFAULT_CONFIG
    groups: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    corridors: List[Dict[str, Any]] = []
    for hid, (x0, x1) in zip(('right', 'left'), halves or []):
        if x1 <= x0:
            continue
        for c in seam_corridors(ink[:, x0:x1], H, cfg):
            c['x0'] += x0; c['x1'] += x0
            corridors.append(c)
            rec = refine_corridor(ink, c, H, cfg, hid)
            if rec.get('reject'):
                rejected.append(rec)
            else:
                groups.append(rec)
    kept, dropped = dedupe(groups, cfg)
    kept.sort(key=lambda g: (g['group_bbox'][1], g['group_bbox'][0]))
    return {'groups': kept, 'rejected': rejected + dropped,
            'corridors': len(corridors)}


def dedupe(groups: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = cfg or DEFAULT_CONFIG
    p = cfg['seam_pair']
    order = sorted(groups, key=lambda g: -(g['span'][1] - g['span'][0]))
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for g in order:
        if any(_iou(g['group_bbox'], k['group_bbox']) >= p['dedupe_iou']
               or _containment(g['group_bbox'], k['group_bbox']) >= p['dedupe_containment']
               for k in kept):
            g2 = dict(g); g2['reject'] = ['duplicate_of_kept']
            dropped.append(g2)
        else:
            kept.append(g)
    return kept, dropped
