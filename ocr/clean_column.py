"""Conservative, opt-in cleanup of bordered body crops."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _binary(crop):
    if (not isinstance(crop, np.ndarray) or crop.size == 0 or crop.dtype != np.uint8
            or not (crop.ndim == 2 or (crop.ndim == 3 and crop.shape[2] == 3))):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def _long_lines(binary):
    h, w = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * .16))))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    col_sum = np.count_nonzero(lines, axis=0)
    band = cv2.dilate((col_sum >= int(h * .35)).astype(np.uint8)[None, :],
                      np.ones((1, 3), np.uint8))
    active = band[0] != 0
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1),
                                              np.flatnonzero(edges == -1))
            if b - a <= max(8, int(w * .025))], lines


def _fill(crop, binary, mask):
    out = crop.copy()
    if not np.any(mask):
        return out
    # Per-component neighborhoods follow changes in paper color across a page.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    h, w = binary.shape
    all_background = crop[binary == 0]
    for label in range(1, count):
        x, y, bw, bh, _ = stats[label]
        x0, x1 = max(0, x - 15), min(w, x + bw + 15)
        y0, y1 = max(0, y - 15), min(h, y + bh + 15)
        region = crop[y0:y1, x0:x1]
        background = region[binary[y0:y1, x0:x1] == 0]
        if not len(background):
            background = all_background
        if len(background):
            out[labels == label] = np.median(background, axis=0).astype(crop.dtype)
    return out


def remove_vertical_grid_lines(crop: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    binary = _binary(crop)
    if binary is None:
        return crop, {"lines_removed": 0, "pixels_filled": 0}
    intervals, _ = _long_lines(binary)
    mask = np.zeros(binary.shape, dtype=bool)
    foreground = binary != 0
    for x0, x1 in intervals:
        mask[:, x0:x1] = foreground[:, x0:x1]
        for x in range(x0, x1):
            left = np.any(foreground[:, max(0, x - 3):max(0, x - 1)], axis=1)
            right = np.any(foreground[:, min(binary.shape[1], x + 2):min(binary.shape[1], x + 4)], axis=1)
            mask[:, x] &= ~(left & right)
    return _fill(crop, binary, mask), {"lines_removed": len(intervals), "pixels_filled": int(mask.sum())}


def strip_side_kunten(crop: np.ndarray, min_avg_per_col: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    binary = _binary(crop)
    no_trigger = {"triggered": False, "components_removed": 0}
    if binary is None:
        return crop, no_trigger
    h, w = binary.shape
    intervals, lines = _long_lines(binary)
    text = binary.copy()
    for x0, x1 in intervals:
        text[:, x0:x1] = cv2.subtract(text[:, x0:x1], lines[:, x0:x1])
    if len(intervals) >= 2:
        cuts = [0] + sorted((a + b) // 2 for a, b in intervals) + [w]
        spans = zip(cuts, cuts[1:])
    else:
        proj = np.count_nonzero(text, axis=0)
        positive = proj[proj > 0]
        if not len(positive):
            return crop, no_trigger
        smooth = np.convolve(proj, np.ones(3) / 3, mode='same')
        thr = max(1.0, float(np.median(positive)) * .12)
        active = smooth > thr
        # Narrow valleys after smoothing can separate a side mark from its
        # glyph; retain them while leaving wider inter-column valleys open.
        bridge = active[:-3] & active[3:]
        active[1:-2] |= bridge
        active[2:-1] |= bridge
        active[1:-1] |= active[:-2] & active[2:]
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        spans = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    cols = []
    for x0, x1 in spans:
        ys = np.flatnonzero(np.any(text[:, x0:x1] != 0, axis=1))
        if x1 - x0 >= 20 and len(ys) and ys[-1] - ys[0] + 1 >= h * .25:
            cols.append((int(x0), int(x1)))
    if len(cols) < 2:
        return crop, no_trigger
    mask = np.zeros(binary.shape, dtype=bool)
    candidates = 0
    for x0, x1 in cols:
        cw = x1 - x0
        column = text[:, x0:x1].copy()
        if len(intervals) >= 2:
            for a, b in intervals:
                center = (a + b) // 2
                if center == x0 or center == x1:
                    column[:, max(0, center - 2 - x0):min(cw, center + 3 - x0)] = 0
        count, labels, stats, centers = cv2.connectedComponentsWithStats(column, 8)
        for label in range(1, count):
            bx, _, bw, bh, area = stats[label]
            cx = centers[label][0]
            if (cx / cw >= .70 and bx >= cw * .60 and bw <= cw * .38
                    and bh <= cw * .60 and 2 <= area <= cw * cw * .18):
                mask[:, x0:x1] |= labels == label
                candidates += 1
    if candidates / len(cols) < min_avg_per_col:
        return crop, no_trigger
    expanded = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    mask = expanded & ((binary == 0) | mask)
    return _fill(crop, binary, mask), {"triggered": True, "columns": len(cols),
                                       "components_removed": candidates, "pixels_filled": int(mask.sum())}


def clean_body_crop(crop: np.ndarray, remove_grid: bool = False, strip_kunten: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    out, stats = crop, {}
    if remove_grid:
        out, stats["remove_grid"] = remove_vertical_grid_lines(out)
    if strip_kunten:
        out, stats["strip_kunten"] = strip_side_kunten(out)
    return out, stats
