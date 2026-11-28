"""从原图墨迹中提取可以单独核验的古籍栏线和版框。"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _directional_rule(ink: np.ndarray, length: int, width: int,
                      vertical: bool) -> np.ndarray:
    kernel = np.ones((length, 1) if vertical else (1, length), np.uint8)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    depth, span = (w, h) if vertical else (h, w)
    qualified = (depth <= width) & (span >= length)
    qualified[0] = False
    return qualified[labels]


def vertical_rule_mask(ink: np.ndarray, H: float, *, min_height: float = 3.0
                       ) -> np.ndarray:
    """只取横向很窄、纵向连续超过多个字高的印刷墨迹。"""
    return _directional_rule(ink, max(3, round(min_height * H)),
                             max(2, int(0.24 * H)), True)


def _sloping_vertical_rules(ink: np.ndarray, H: float) -> np.ndarray:
    """在原墨迹上验证轻微倾斜的长栏线，填上开运算留下的断点。"""
    length = max(3, round(3.1 * H))
    radius = max(1, round(0.1 * H))
    strokes = cv2.HoughLinesP(ink, 1, np.pi / 360, max(3, round(2.1 * H)),
                             minLineLength=length, maxLineGap=max(1, round(0.33 * H)))
    lines = np.zeros_like(ink)
    if strokes is None:
        return lines.astype(bool)
    for x1, y1, x2, y2 in strokes.reshape(-1, 4):
        span = abs(int(y2) - int(y1))
        if span < length or abs(int(x2) - int(x1)) > round(0.022 * span) + 1:
            continue
        xs = np.round(np.linspace(x1, x2, span + 1)).astype(int)
        ys = np.round(np.linspace(y1, y2, span + 1)).astype(int)
        if float(ink[ys, xs].mean()) < 0.6:
            continue
        cv2.line(lines, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2 * radius - 1)
    return (lines & ink).astype(bool)


def _collinear_dashed_vertical_rules(ink: np.ndarray, H: float,
                                     labels: np.ndarray,
                                     stats: np.ndarray) -> np.ndarray:
    """提取共线断续细纵栏线：仅作用于独立窄竖段，绝不侵蚀宽字形或单字长竖。"""
    max_w = max(3, int(0.16 * H))
    min_dash_h = max(5, round(0.22 * H))
    max_gap = max(12, round(3.6 * H))
    min_span = max(120, round(10.0 * H))
    min_sum_h = max(70, round(4.0 * H))
    n_labels = int(stats.shape[0])
    narrow = []
    for cid in range(1, n_labels):
        x, y, w, h, area = map(int, stats[cid])
        if w <= max_w and area >= 0.45 * h:
            strong = (h >= min_dash_h and h >= 2.0 * w and area >= 0.55 * h)
            narrow.append((cid, x, y, w, h, strong))
    out = np.zeros(ink.shape, dtype=bool)
    if len(narrow) < 8:
        return out
    keep_cids: set[int] = set()
    win_w = max_w
    cols = ink.shape[1]
    for x0 in range(0, cols - win_w + 1):
        x1 = x0 + win_w
        cands = [t for t in narrow if t[1] >= x0 and t[1] + t[3] <= x1]
        if len(cands) < 8:
            continue
        cands.sort(key=lambda t: t[2])
        chains = []
        cur = [cands[0]]
        cur_end = cands[0][2] + cands[0][4]
        for t in cands[1:]:
            if t[2] - cur_end <= max_gap:
                cur.append(t)
                cur_end = max(cur_end, t[2] + t[4])
            else:
                chains.append(cur)
                cur = [t]
                cur_end = t[2] + t[4]
        chains.append(cur)
        for chain in chains:
            if sum(1 for t in chain if t[5]) < 8:
                continue
            y_start = min(t[2] for t in chain)
            y_end = max(t[2] + t[4] for t in chain)
            span = y_end - y_start
            if span < min_span:
                continue
            cov = np.zeros(span, dtype=bool)
            for t in chain:
                cov[t[2] - y_start:t[2] + t[4] - y_start] = True
            sum_h = int(cov.sum())
            if sum_h >= min_sum_h and sum_h / span >= 0.28:
                for t in chain:
                    keep_cids.add(t[0])
    if keep_cids:
        lut = np.zeros(n_labels, dtype=bool)
        for cid in keep_cids:
            lut[cid] = True
        out |= lut[labels]
    return out


def _bridge_vertical_to_horizontal_rules(ink: np.ndarray, v_mask: np.ndarray,
                                         h_mask: np.ndarray, H: float) -> np.ndarray:
    """当已确认纵栏线在距横版框 <=0.65H 处因 1-2px 淡墨断开时，仅沿该纵栏线精确列坐标补齐桥接墨迹。"""
    out = np.zeros(ink.shape, dtype=bool)
    max_bridge = max(4, round(0.65 * H))
    min_run = max(12, round(1.5 * H))
    rows, cols = ink.shape
    ink_b = ink > 0
    for x in np.flatnonzero(v_mask.any(axis=0)):
        col_v = v_mask[:, x]
        col_h = h_mask[:, x]
        if not col_h.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[False, col_v, False])).reshape(-1, 2)
        h_ys = np.flatnonzero(col_h)
        for y0, y1 in edges:
            if y1 - y0 < min_run:
                continue
            below = h_ys[h_ys >= y1]
            if below.size and 0 < int(below[0]) - int(y1) <= max_bridge:
                yb = int(below[0])
                out[y1:yb, x] |= ink_b[y1:yb, x]
            above = h_ys[h_ys < y0]
            if above.size and 0 < int(y0) - int(above[-1]) - 1 <= max_bridge:
                ya = int(above[-1]) + 1
                out[ya:y0, x] |= ink_b[ya:y0, x]
    return out


def ruling_mask(ink: np.ndarray, H: float, labels: np.ndarray,
                stats: np.ndarray) -> np.ndarray:
    """标记原图墨迹里的细栏线、共线断续细纵栏线和长横版框，保留粘连字形所在原图。"""
    width = max(2, round(0.24 * H))
    vertical = max(3, round(2.2 * H))
    horizontal = max(3, round(3 * H))
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    isolated_v = (w <= width) & (h >= vertical)
    isolated_h = (h <= width) & (w >= horizontal)
    isolated_v[0] = False
    isolated_h[0] = False
    v_mask = (isolated_v[labels] | vertical_rule_mask(ink, H, min_height=2.2)
              | _sloping_vertical_rules(ink, H)
              | _collinear_dashed_vertical_rules(ink, H, labels, stats))
    h_mask = isolated_h[labels] | _directional_rule(ink, horizontal, width, False)
    bridge = _bridge_vertical_to_horizontal_rules(ink, v_mask, h_mask, H)
    return v_mask | h_mask | bridge


def masked_crop(image: np.ndarray, mask: np.ndarray
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """用本裁区的背景色替换掩膜像素，掩膜外像素逐值保留。"""
    if not mask.any():
        return None
    clean = image.copy()
    background = image[~mask]
    clean[mask] = np.percentile(background, 85, axis=0) if background.size else 255
    return clean, mask.astype(np.uint8) * 255
