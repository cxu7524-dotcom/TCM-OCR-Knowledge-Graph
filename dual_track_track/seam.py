# -*- coding: utf-8 -*-
"""中缝证据：在一个窗口里找「左右各一轨、中间一条缝」的局部轨对。

与初版的关键差别（回退窗口可能含多栏）：
  - 不再用固定带宽去截两侧（1.5×pitch 的带在 126px 窗口里会跨到邻栏），
    而是**从候选中缝向两侧做「宽度受限的贪心扩张」**得到左右轨；
  - L/R 峰值取在**轨内**，V 取中缝带内；这样 r 不会因为邻栏峰值而被稀释；
  - 轨宽/合计宽/缝宽/栏线冲突/跨缝宽字五道几何判据决定接受还是拒绝。

整套判据与契约一致：r = V/(min(L,R)+eps) ≤ 0.15、L/R ≥ 0.10、两侧支持 ≥ 30%。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .projection import RegionInk, projection, smooth_window_px


@dataclass
class SeamEval:
    valid: bool
    reason: str = ''
    seam_x: Optional[int] = None
    L: float = 0.0
    R: float = 0.0
    V: float = 0.0
    r: Optional[float] = None
    left_support: float = 0.0
    right_support: float = 0.0
    left_glyphs: int = 0
    right_glyphs: int = 0
    left_bbox: Optional[List[int]] = None
    right_bbox: Optional[List[int]] = None
    left_boxes: List[List[int]] = field(default_factory=list)
    right_boxes: List[List[int]] = field(default_factory=list)
    track_widths: Optional[List[int]] = None
    seam_gap_px: Optional[int] = None
    crossing_glyphs: int = 0
    rule_distance: Optional[float] = None
    band_px: int = 0
    seam_width_px: int = 0
    tried: int = 0
    best_rejected: Optional[Dict[str, Any]] = None
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'valid': self.valid, 'reason': self.reason, 'seam_x': self.seam_x,
                'L': round(self.L, 4), 'R': round(self.R, 4), 'V': round(self.V, 4),
                'r': (round(self.r, 4) if self.r is not None else None),
                'left_support': round(self.left_support, 4),
                'right_support': round(self.right_support, 4),
                'left_glyphs': self.left_glyphs, 'right_glyphs': self.right_glyphs,
                'left_bbox': self.left_bbox, 'right_bbox': self.right_bbox,
                'track_widths': self.track_widths, 'seam_gap_px': self.seam_gap_px,
                'crossing_glyphs': self.crossing_glyphs,
                'rule_distance': (round(self.rule_distance, 2)
                                  if self.rule_distance is not None else None),
                'band_px': self.band_px, 'seam_width_px': self.seam_width_px,
                'tried_positions': self.tried,
                'best_rejected': self.best_rejected,
                'alternatives': self.alternatives}


def _union(comps: Sequence[Dict[str, Any]]) -> Optional[List[int]]:
    if not comps:
        return None
    return [int(min(c['bbox'][0] for c in comps)), int(min(c['bbox'][1] for c in comps)),
            int(max(c['bbox'][2] for c in comps)), int(max(c['bbox'][3] for c in comps))]


def _side_stats(region: RegionInk, y0: int, y1: int, xa: int, xb: int) -> Dict[str, Any]:
    a = max(0, int(y0) - region.bbox[1])
    b = min(region.height, int(y1) - region.bbox[1])
    x0 = max(0, int(xa) - region.bbox[0])
    x1 = min(region.width, int(xb) - region.bbox[0])
    if b <= a or x1 <= x0:
        return {'support': 0.0, 'ink': 0.0}
    sub = region.mask[a:b, x0:x1]
    return {'support': round(float(sub.any(axis=1).mean()), 4),
            'ink': round(float(sub.mean()), 4)}


def _tracks(comps: Sequence[Dict[str, Any]], seam: int, max_w: float, gap_tol: float
            ) -> tuple:
    """从候选中缝向两侧贪心扩张：只收「离已有簇足够近」且总宽不超限的字形块。"""
    left = [c for c in comps if c['bbox'][2] <= seam + 1]
    right = [c for c in comps if c['bbox'][0] >= seam - 1]
    left.sort(key=lambda c: seam - c['bbox'][2])
    right.sort(key=lambda c: c['bbox'][0] - seam)

    def grow(items):
        chosen: List[Dict[str, Any]] = []
        for c in items:
            if not chosen:
                if c['bbox'][2] - c['bbox'][0] > max_w:
                    continue
                chosen.append(c)
                continue
            lo = min(x['bbox'][0] for x in chosen)
            hi = max(x['bbox'][2] for x in chosen)
            if c['bbox'][0] > hi + gap_tol or c['bbox'][2] < lo - gap_tol:
                continue
            w = max(hi, c['bbox'][2]) - min(lo, c['bbox'][0])
            if w > max_w:
                continue
            chosen.append(c)
        return chosen

    return grow(left), grow(right)


def region_seam_candidates(region: RegionInk, H: float, cfg: Dict[str, Any]
                           ) -> List[int]:
    """区域级「持续低谷」x：只在整栏高度上都几乎没有墨的位置找缝。

    逐窗独立找局部极小会被笔画间的小缝骗到（左右偏旁之间的空隙也会形成低谷），
    所以先用整栏投影筛出**贯穿**的低谷，再让每个窗口在这批位置上选。
    """
    if region.mask.size == 0:
        return []
    p = region.mask.mean(axis=0)
    thr = float(cfg['seam']['long_gap_max'])
    idx = np.flatnonzero(p <= thr)
    out: List[int] = []
    if idx.size == 0:
        return out
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        out.append((start + prev) // 2)
        start = prev = v
    out.append((start + prev) // 2)
    return [x + region.bbox[0] for x in out
            if 1 <= (prev - start + 1) <= 0.8 * H]


def _bad_track_geometry(lb, rb, H, cfg) -> bool:
    """双轨的几何签名：两轨都窄、宽度相近、间隙很小。"""
    tk = cfg['track']
    wl, wr = lb[2] - lb[0], rb[2] - rb[0]
    gap = rb[0] - lb[2]
    if wl < tk['min_width_H'] * H or wr < tk['min_width_H'] * H:
        return True
    if wl > tk['max_width_H'] * H or wr > tk['max_width_H'] * H:
        return True
    if (rb[2] - lb[0]) > tk['pair_max_width_H'] * H:
        return True
    if gap < tk['seam_gap_min_px'] or gap > tk['seam_gap_max_H'] * H:
        return True
    ratio = wl / float(wr) if wr else 99.0
    if ratio < float(tk['width_ratio_min']) or ratio > float(tk['width_ratio_max']):
        return True
    if gap > max(2.0, float(tk['gap_to_track_max']) * min(wl, wr)):
        return True
    return False


def evaluate_window(region: RegionInk, y0: int, y1: int,
                    glyphs: Sequence[Dict[str, Any]], pitch: float, H: float,
                    rule_xs: Sequence[float], cfg: Dict[str, Any],
                    prefer_x: Optional[int] = None,
                    seam_candidates: Optional[Sequence[int]] = None,
                    collect_alternatives: bool = False) -> SeamEval:
    """prefer_x：上一窗的中缝位置。多解时优先「接得上」的那条，避免逐窗跳缝。"""
    sc = cfg['seam']
    tk = cfg['track']
    w_seam = max(int(sc['seam_width_min_px']),
                 int(round(float(sc['seam_width_pitch_multiple']) * pitch)))
    band = max(int(sc['band_min_px']),
               int(round(float(sc['band_pitch_multiple']) * pitch)))
    x1, x2 = region.bbox[0], region.bbox[2]
    lo, hi = x1 + w_seam + 2, x2 - w_seam - 2
    ev = SeamEval(valid=False, band_px=band, seam_width_px=w_seam)
    if hi <= lo:
        ev.reason = 'no_search_band'
        return ev

    P = projection(region, y0, y1, smooth_window_px(pitch, sc))
    # V 用**未平滑**投影：平滑窗（0.12×pitch≈3~4px）会把两侧笔画渗进 1~8px 的窄缝
    a_raw = max(0, int(y0) - region.bbox[1])
    b_raw = min(region.height, int(y1) - region.bbox[1])
    P_raw = (region.mask[a_raw:b_raw].mean(axis=0) if b_raw > a_raw
             else np.zeros(region.width))
    if not P.any():
        ev.reason = 'blank_window'
        return ev

    in_win = [g for g in glyphs if g['bbox'][3] > y0 and g['bbox'][1] < y1]
    in_win.sort(key=lambda c: c['cx'])
    cxs = [c['cx'] for c in in_win]
    max_track_w = float(tk['max_width_H']) * H
    gap_tol = 0.4 * H

    best: Optional[SeamEval] = None
    best_r = 1e9
    best_rej: Optional[Dict[str, Any]] = None
    reasons: Dict[str, int] = {}
    step = max(1, int(sc['search_step_px']))
    if seam_candidates:
        positions = sorted({int(x) for x in seam_candidates if lo <= x <= hi})
    else:
        positions = list(range(int(lo), int(hi) + 1, step))
    for s in positions:
        i = s - x1
        vs = P[max(0, i - w_seam):min(P.size, i + w_seam + 1)]
        if vs.size == 0:
            continue
        ev.tried += 1
        # 只看附近 3H 内的字形块（避免 O(全部)）
        a = bisect.bisect_left(cxs, s - 2.5 * H)
        b = bisect.bisect_right(cxs, s + 2.5 * H)
        near = in_win[a:b]
        left, right = _tracks(near, s, max_track_w, gap_tol)
        lb, rb = _union(left), _union(right)
        # 失败窗也要带两侧证据：状态机靠它判断「哪一侧还在」
        ev.left_glyphs, ev.right_glyphs = len(left), len(right)
        ev.left_bbox, ev.right_bbox = lb, rb
        ev.left_boxes = [list(c['bbox']) for c in left]
        ev.right_boxes = [list(c['bbox']) for c in right]
        reason = ''
        L = R = 0.0
        V = float(vs.mean())
        if lb is None or rb is None:
            reason = ('blank_window' if not near else 'no_glyph_support')
        else:
            # 中缝带取**两轨之间实际间隙**：0.4×正文pitch 的名义带在双轨小字上
            # 会把两侧笔画算进 V（实测缝仅 4~8px），名义带仍记在 band 字段里
            gx0, gx1 = max(x1, lb[2]), min(x2, rb[0])
            if gx1 - gx0 < int(tk['seam_gap_min_px']) or abs(s - (gx0 + gx1) / 2.0) > 2:
                reason = 'track_geometry'
            li0, li1 = max(0, lb[0] - x1), min(P.size, lb[2] - x1)
            ri0, ri1 = max(0, rb[0] - x1), min(P.size, rb[2] - x1)
            if not reason and (li1 <= li0 or ri1 <= ri0):
                reason = 'no_glyph_support'
            if not reason:
                L = float(P[li0:li1].max())
                R = float(P[ri0:ri1].max())
                V = float(P_raw[gx0 - x1:gx1 - x1].mean()) if gx1 > gx0 else 1.0
                lst = _side_stats(region, y0, y1, lb[0], lb[2])
                rst = _side_stats(region, y0, y1, rb[0], rb[2])
                cross = [g for g in near
                         if g['bbox'][0] < s - 1 and g['bbox'][2] > s + 1
                         and g['w'] > 0.6 * H]
                rule_d = min([abs(s - float(x)) for x in rule_xs], default=None)
                gap = rb[0] - lb[2]
                if min(L, R) < float(sc['min_peak']):
                    reason = ('blank_window' if max(L, R) < float(sc['min_peak']) / 2
                              and lst['support'] < 0.05 and rst['support'] < 0.05
                              else 'insufficient_peak')
                elif (lst['support'] < float(sc['min_support_ratio'])
                      or rst['support'] < float(sc['min_support_ratio'])):
                    reason = 'insufficient_support'
                elif len(left) < int(cfg['glyph']['min_per_side_window']) \
                        or len(right) < int(cfg['glyph']['min_per_side_window']):
                    reason = 'no_glyph_support'
                elif cross:
                    reason = 'ambig_ink_crossing'
                elif rule_d is not None and rule_d <= max(2.0, w_seam):
                    reason = 'rule_at_seam'
                elif _bad_track_geometry(lb, rb, H, cfg):
                    reason = 'track_geometry'
                else:
                    r = V / (min(L, R) + float(sc['epsilon']))
                    if r > float(sc['max_r']):
                        reason = 'seam_not_clear'
                if not reason:
                    r = V / (min(L, R) + float(sc['epsilon']))
                    score = r + (0.5 * abs(((gx0 + gx1) / 2.0) - prefer_x) / H
                                 if prefer_x is not None else 0.0)
                    sx = int((gx0 + gx1) // 2)
                    if collect_alternatives and len(ev.alternatives) < 6 and all(
                            abs(sx - a['seam_x']) >= 3 for a in ev.alternatives):
                        ev.alternatives.append({'seam_x': sx, 'r': round(r, 4),
                                                'left_bbox': lb, 'right_bbox': rb,
                                                'left_glyphs': len(left),
                                                'right_glyphs': len(right)})
                    if score < best_r - 1e-12:
                        best_r = score
                        best = SeamEval(
                            valid=True, seam_x=int((gx0 + gx1) // 2), L=L, R=R, V=V, r=r,
                            left_support=lst['support'], right_support=rst['support'],
                            left_glyphs=len(left), right_glyphs=len(right),
                            left_bbox=lb, right_bbox=rb,
                            left_boxes=[list(c['bbox']) for c in left],
                            right_boxes=[list(c['bbox']) for c in right],
                            track_widths=[lb[2] - lb[0], rb[2] - rb[0]],
                            seam_gap_px=int(gap), crossing_glyphs=len(cross),
                            rule_distance=rule_d, band_px=band,
                            seam_width_px=w_seam, tried=ev.tried)
                    continue
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason not in ('seam_not_clear',) or best_rej is None:
                best_rej = {'seam_x': int(s), 'reason': reason, 'L': round(L, 3),
                            'R': round(R, 3), 'V': round(V, 3),
                            'r': (round(V / (min(L, R) + float(sc['epsilon'])), 3)
                                  if min(L, R) > 0 else None)}
    if best is not None:
        # 备选缝记录在 ev 上，返回时挂到胜出的 SeamEval，供逐候选审计使用
        best.alternatives = ev.alternatives
        return best
    ev.reason = (max(reasons, key=reasons.get) if reasons else 'seam_not_clear')
    ev.best_rejected = best_rej
    return ev
