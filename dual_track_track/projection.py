# -*- coding: utf-8 -*-
"""区域墨迹、横向投影与字形块。

约定：
  - 掩膜 0/1（不是 0/255），与阶段 4 契约一致；
  - 合格栏线的**独立段**从投影中剔除（按段，断口与贴字段保留）；
  - P(x) = 窗口内该列墨迹行占比，∈[0,1]，再按 0.12×pitch 做移动平均。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class RegionInk:
    bbox: List[int]                       # 原图 [x1,y1,x2,y2)
    mask: np.ndarray                      # 0/1，已剔除独立栏线段
    raw: np.ndarray                       # 0/1，未剔除（审计用）
    rule_bands: List[List[int]] = field(default_factory=list)
    labels: Optional[np.ndarray] = None
    stats: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def components(self) -> None:
        n, lab, st, cen = cv2.connectedComponentsWithStats(self.mask, 8)
        self.labels, self.stats, self.centroids = lab, st, cen


def build_region_ink(ink: np.ndarray, bbox: Sequence[int],
                     rule_segments: Sequence[Sequence[int]] = ()) -> RegionInk:
    """裁出区域掩膜；把落在区域内的独立栏线段置零（只按段，不按整条 bbox）。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    raw = (ink[y1:y2, x1:x2] > 0).astype(np.uint8)
    mask = raw.copy()
    bands: List[List[int]] = []
    for seg in rule_segments or []:
        sx0, sx1, sy0, sy1 = [int(v) for v in seg[:4]]
        a, b = max(0, sx0 - x1), min(x2 - x1, sx1 - x1)
        c, d = max(0, sy0 - y1), min(y2 - y1, sy1 - y1)
        if b > a and d > c:
            mask[c:d, a:b] = 0
            bands.append([sx0, sx1, sy0, sy1])
    return RegionInk(bbox=[x1, y1, x2, y2], mask=mask, raw=raw, rule_bands=bands)


def moving_average(p: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1 or p.size < w:
        return p.astype(np.float64)
    ker = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    padded = np.pad(p.astype(np.float64), (pad, w - 1 - pad), mode='edge')
    return np.convolve(padded, ker, mode='valid')


def smooth_window_px(pitch: Optional[float], cfg: Dict[str, Any]) -> int:
    base = float(pitch) if pitch else 20.0
    return max(int(cfg['smooth_min_px']),
               int(round(float(cfg['smooth_pitch_multiple']) * base)))


def projection(region: RegionInk, y0: int, y1: int, smooth_px: int) -> np.ndarray:
    """窗口内按列纵向取平均，返回长度 = 区域宽的 P(x)。"""
    a = max(0, int(y0) - region.bbox[1])
    b = min(region.height, int(y1) - region.bbox[1])
    if b <= a:
        return np.zeros(region.width, dtype=np.float64)
    p = region.mask[a:b].mean(axis=0)
    return moving_average(p, smooth_px)


def glyph_components(region: RegionInk, cfg: Dict[str, Any], H: float
                     ) -> List[Dict[str, Any]]:
    """区域内满足「字形块」尺寸的连通域（原图坐标）。"""
    if region.labels is None:
        region.components()
    g = cfg['glyph']
    lo_h = max(int(g['min_height_px']), int(round(g['min_height_H'] * H)))
    hi_h = int(round(g['max_height_H'] * H))
    hi_w = max(int(g['min_width_px']), int(round(g['max_width_H'] * H)))
    out: List[Dict[str, Any]] = []
    n = 0 if region.stats is None else region.stats.shape[0]
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in region.stats[i]]
        if area < int(g['min_area_px']):
            continue
        if h < lo_h or h > hi_h or w > hi_w or w < int(g['min_width_px']):
            continue
        out.append({'label': i,
                    'bbox': [x + region.bbox[0], y + region.bbox[1],
                             x + region.bbox[0] + w, y + region.bbox[1] + h],
                    'w': w, 'h': h, 'area': area,
                    'cx': float(region.centroids[i][0] + region.bbox[0]),
                    'cy': float(region.centroids[i][1] + region.bbox[1])})
    out.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
    return out


def side_stats(region: RegionInk, y0: int, y1: int, xa: int, xb: int
               ) -> Dict[str, Any]:
    """一侧搜索带的墨迹支持：有墨行占比 + 带内墨量。"""
    a = max(0, int(y0) - region.bbox[1])
    b = min(region.height, int(y1) - region.bbox[1])
    x0 = max(0, int(xa) - region.bbox[0])
    x1 = min(region.width, int(xb) - region.bbox[0])
    if b <= a or x1 <= x0:
        return {'support': 0.0, 'ink': 0.0, 'rows': max(0, b - a)}
    sub = region.mask[a:b, x0:x1]
    rows = float(sub.any(axis=1).mean()) if sub.size else 0.0
    return {'support': round(rows, 4), 'ink': round(float(sub.mean()), 4),
            'rows': int(b - a)}
