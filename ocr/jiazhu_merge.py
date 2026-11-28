"""Verify dual-track annotations and splice them into vertical body text."""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Real
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from app.schemas import Block


def _cjk_count(text: str) -> int:
    return sum('\u4e00' <= char <= '\u9fff' for char in (text or ''))


def _recognizer(rec_fn):
    if rec_fn is not None:
        return rec_fn
    from .recognize import recognize_crop
    return lambda crop: recognize_crop(crop, min_conf=0.15)


def _valid_bbox(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value):
        return None
    x1, y1, x2, y2 = value
    return [x1, y1, x2, y2] if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height else None


def select_verified_jiazhu_groups(image: np.ndarray, evidence: Sequence[Dict[str, Any]], rec_fn=None) -> List[Dict[str, Any]]:
    height, width = image.shape[:2]
    grouped = defaultdict(dict)
    for item in evidence:
        gid, side = item.get('group_id'), item.get('side')
        if gid is not None and side in ('right', 'left'):
            grouped[gid][side] = item
    recognize = _recognizer(rec_fn)
    result = []
    for gid, sides in grouped.items():
        if 'right' not in sides or 'left' not in sides:
            continue
        right = _valid_bbox(sides['right'].get('bbox'), width, height)
        left = _valid_bbox(sides['left'].get('bbox'), width, height)
        if right is None or left is None:
            continue
        rx1, ry1, rx2, ry2 = right
        lx1, ly1, lx2, ly2 = left
        rw, lw = rx2 - rx1, lx2 - lx1
        bbox = [min(rx1, lx1), min(ry1, ly1), max(rx2, lx2), max(ry2, ly2)]
        if rw < 14 or lw < 10 or not 0.95 <= rw / lw <= 2.2 or bbox[2] - bbox[0] > 60 or bbox[3] - bbox[1] < 24:
            continue
        readings = []
        for x1, y1, x2, y2 in (right, left):
            crop = image[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            readings.append(recognize(np.ascontiguousarray(np.rot90(crop))))
        (rt, rc), (lt, lc) = readings
        if _cjk_count(rt) < 2 or _cjk_count(lt) < 2 or not rc >= 0.55 or not lc >= 0.55:
            continue
        result.append(dict(group_id=gid, bbox=bbox, right_bbox=right, left_bbox=left,
                           right_text=rt, left_text=lt, text=rt + lt, conf=min(rc, lc)))
    return sorted(result, key=lambda g: (-(g['bbox'][0] + g['bbox'][2]) / 2, g['bbox'][1]))


def splice_jiazhu_into_blocks(image: np.ndarray, blocks: List[Block], groups: List[Dict[str, Any]], rec_fn=None) -> Tuple[List[Block], List[Dict[str, Any]]]:
    recognize = _recognizer(rec_fn)
    traces = []
    for block in blocks:
        if block.kind != 'paragraph':
            continue
        bx1, by1, bx2, by2 = block.bbox
        matches = [g for g in groups if bx1 - 6 <= (g['bbox'][0] + g['bbox'][2]) / 2 <= bx2 + 6
                   and g['bbox'][1] < by2 and g['bbox'][3] > by1]
        kept = []
        for group in sorted(matches, key=lambda g: -g['conf']):
            y1, y2 = group['bbox'][1], group['bbox'][3]
            if any(max(0, min(y2, other['bbox'][3]) - max(y1, other['bbox'][1]))
                   > 0.5 * min(y2 - y1, other['bbox'][3] - other['bbox'][1]) for other in kept):
                continue
            kept.append(group)
        kept.sort(key=lambda g: g['bbox'][1])
        if not kept:
            continue
        pieces = []
        def append_body(y1, y2):
            if y2 - y1 < 18:
                return
            crop = image[y1:y2, bx1:bx2]
            if crop.shape[0] / crop.shape[1] >= 1.5:
                crop = np.ascontiguousarray(np.rot90(crop))
            text, _ = recognize(crop)
            pieces.append(text or '')
        cursor = by1
        for group in kept:
            gy1, gy2 = math.floor(group['bbox'][1]), math.ceil(group['bbox'][3])
            append_body(cursor, min(by2, max(by1, gy1)))
            pieces.append(group['right_text'] + group['left_text'])
            cursor = max(cursor, min(by2, gy2))
        append_body(cursor, by2)
        new_text = ''.join(pieces)
        if _cjk_count(new_text) > _cjk_count(block.text):
            old = block.text
            block.text = block.quote = new_text
            traces.append({'line_id': block.line_id, 'before': old, 'after': new_text,
                           'groups': [g['group_id'] for g in kept]})
    return blocks, traces
