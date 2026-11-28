"""Geometric isolation of bordered body text and upper margin notes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from app.config import CONFIG
from app.schemas import Block, source_hash_of
from .clean_column import clean_body_crop
from .recognize import RecognizeResult, _box_to_bbox
from .vertical import imread_unicode


def detect_body_frames(img: np.ndarray) -> List[Dict[str, Any]]:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        return []
    h, w = img.shape[:2]
    if not h or not w:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    regions = [("R", w // 2, w), ("L", 0, w // 2)] if w >= h * .98 else [("ALL", 0, w)]
    frames = []
    for half, hx0, hx1 in regions:
        hw = hx1 - hx0
        region = lines[:, hx0:hx1]
        proj = np.count_nonzero(region, axis=1)
        tops = np.flatnonzero(proj[int(h * .04):int(h * .45) + 1] > hw * .35) + int(h * .04)
        bottoms = np.flatnonzero(proj[int(h * .55):min(h, int(h * .98) + 1)] > hw * .35) + int(h * .55)
        if not len(tops) or not len(bottoms):
            continue
        ty, by = int(tops[-1]), int(bottoms[0])
        if by - ty < h * .45:
            continue
        band = np.concatenate((region[max(0, ty - 6):min(h, ty + 7)],
                               region[max(0, by - 6):min(h, by + 7)]), axis=0)
        xs = np.flatnonzero(np.any(band, axis=0))
        if not len(xs):
            continue
        fx0, fx1 = int(xs[0]) + hx0, int(xs[-1]) + hx0
        if fx1 - fx0 < hw * .68:
            continue
        if len(regions) == 1:
            if np.any(proj[0:max(0, ty - 20)] > hw * 0.35):
                return []
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 6)))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            for x in (fx0, fx1):
                border = vertical_lines[ty:by, max(0, x - 12):min(w, x + 13)]
                if not border.size or np.max(np.count_nonzero(border, axis=0)) < .55 * (by - ty):
                    return []
            inset = max(15, int((by - ty) * .08))
            interior = lines[ty + inset:by - inset + 1, fx0:fx1]
            if interior.size and np.any(np.count_nonzero(interior, axis=1) >= .50 * (fx1 - fx0)):
                return []
        frames.append({"half": half, "bbox": [fx0, ty, fx1, by], "top_y": ty, "bot_y": by})
    if len(regions) == 2:
        if len(frames) != 2:
            return []
        if (abs(frames[0]["top_y"] - frames[1]["top_y"]) > h * .15 or
                abs(frames[0]["bot_y"] - frames[1]["bot_y"]) > h * .15):
            return []
    return frames


def is_outer_spine_bbox(bbox: Sequence[int], page_w: int) -> bool:
    x1, _, x2, _ = bbox
    return (x2 - x1 <= max(36, int(page_w * .04)) and
            (x2 <= int(page_w * .038) or x1 >= int(page_w * .962)))


def recognize_with_frame_isolation(image_path: str, page_id: str, infer_fn) -> Optional[RecognizeResult]:
    img = imread_unicode(image_path)
    if img is None:
        return None
    frames = detect_body_frames(img)
    if not frames:
        return None
    h, w = img.shape[:2]
    blocks: List[Block] = []
    counts = {"paragraph": 0, "spine": 0, "margin_note": 0}

    def add(raw, dx, dy, margin=False):
        for box, text, conf in raw or []:
            x1, y1, x2, y2 = _box_to_bbox(box)
            bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            kind = "margin_note" if margin else ("spine" if is_outer_spine_bbox(bbox, w) else "paragraph")
            line_id = f"R{len(blocks) + 1:03d}"
            blocks.append(Block(page_id=page_id, bbox=bbox, line_id=line_id, quote=text,
                                source_hash=source_hash_of("ocr", page_id, line_id, text),
                                text=text, text_norm=text, kind=kind, conf=conf, col_id=0,
                                reading_order=len(blocks) + 1, block_ref=f"{page_id}:{line_id}"))
            counts[kind] += 1

    clean_stats = []
    for frame in frames:
        fx0, ty, fx1, by = frame["bbox"]
        cx0, cy0 = max(0, fx0 + 2), max(0, ty + 2)
        cx1, cy1 = min(w, fx1 - 2), min(h, by - 2)
        if cx1 > cx0 and cy1 > cy0:
            crop = img[cy0:cy1, cx0:cx1]
            if CONFIG.ocr.remove_grid or CONFIG.ocr.strip_kunten:
                crop, stats = clean_body_crop(crop, remove_grid=CONFIG.ocr.remove_grid,
                                              strip_kunten=CONFIG.ocr.strip_kunten)
                clean_stats.append(stats)
            add(infer_fn(crop), cx0, cy0)
    if not blocks:
        return None
    min_ty = min(f["top_y"] for f in frames)
    if min_ty >= int(h * .08):
        add(infer_fn(img[0:max(1, min_ty - 2), 0:w]), 0, 0, margin=True)
    return RecognizeResult(blocks=blocks, engine="rapidocr", model_loaded=True, is_demo=False,
                           width=w, height=h,
                           notes=[f"正文版框：{[f['bbox'] for f in frames]}",
                                  f"隔离区块数：{counts}"] + ([f"正文净化统计：{clean_stats}"] if clean_stats else []))
