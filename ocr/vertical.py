# -*- coding: utf-8 -*-
"""
ocr/vertical.py —— 竖排古籍逐列检测与识别（P1）

问题：rapidocr 的 det 模型是横排训练的，在木刻竖排页上会把竖列横切成 1–3 字的小框，
块数中位仅 17，覆盖率 19%，CER 0.89（见 docs/真实古籍页诊断.md）。

做法（顺序不能颠倒）：
  1. 灰度 → 自适应二值化 → 中值去噪（二值图**只用于找列边界**，识别仍用原图裁剪）；
  2. 用形态学长直线（版框 / 界行）从二值图里减掉，避免边框被当成列；
  3. 垂直投影找列间隙 → 过滤窄列与矮列 → **从右向左**编号 col_index；
  4. 列内水平投影切块（切点落在字间隙最低处，避免切断字）；
  5. 每块旋转 **90° 逆时针**（实测方向，见下）后调用 `recognize_crop()`；
  6. bbox 加列偏移映射回原图坐标系。

旋转方向实测（page_10.png 最右列，同图同模型）：
  - 90° 逆时针：conf 0.73，识别 20 字「中献三赋召試文章设河西尉薪不行政岩衡率府」
  - 90° 顺时针：conf 0.00，识别 0 字
  因此默认 `TCM_OCR_ROTATE=90`（逆时针）。该方向与 rapidocr 包内 `np.rot90` 一致。

已知未解决（不假装解决）：
  - 双行小字夹注：窄列合并后整列识别，`kind="annotation"` 标注，未做行内左右分离；
  - 页面倾斜：未做纠偏（测试页基本水平），倾斜超过 3° 会显著掉点；
  - 印章：由 `ocr/deseal.py` 处理（P3，可关闭）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import CONFIG
from app.schemas import Block, source_hash_of

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

VERTICAL_ASPECT = 3.0        # 列高/列宽 >= 3 视为竖排列
MIN_COL_RATIO = 0.12         # 列前景高度至少占页面高度的 20%
MAX_PIECE_H = 220            # 单块最大高度：块越短，rec 输入缩放越小、字形细节保留越多
                             # （420px 高块缩放到 rec 输入高 48 时压缩 8.7 倍，木刻字会糊）
MIN_PIECE_H = 70


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """OpenCV 在 Windows 上读不了中文路径，用 imdecode 兜底。"""
    if not _CV2_OK:
        return None
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, ValueError):
        return None


def _binarize(gray: np.ndarray) -> np.ndarray:
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.medianBlur(bw, 3)


def _red_mask(img: np.ndarray) -> np.ndarray:
    """朱丝栏 / 朱印的红色区域：从二值图里剔除，避免界行被当成列内容。

    注意：这一步会同时抹掉压在正文上的印章（P3 的取舍），因此识别仍用原图裁剪，
    只有「找列」用的是去红后的二值图。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255])) | \
        cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def _remove_lines(bw: np.ndarray) -> np.ndarray:
    """减掉长横线 / 长竖线（版框与界行），只影响找列，不影响识别。"""
    h, w = bw.shape[:2]
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25))))
    horz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1)))
    return cv2.subtract(bw, cv2.bitwise_or(vert, horz))


def _segments(mask: np.ndarray, min_w: int) -> List[Tuple[int, int]]:
    """一维布尔数组 → 连续 True 区间，过滤过窄段并合并过近的段。"""
    segs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    segs = [(a, b) for a, b in segs if b - a >= min_w]
    merged: List[Tuple[int, int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= 4:   # 极近的窄列合并（双行小字）
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def split_columns(img: np.ndarray) -> List[Dict[str, Any]]:
    """切列，返回 [{bbox, crop, col_index}]，按 x 中心**从右向左**排序。"""
    if not _CV2_OK or img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    bw[_red_mask(img) > 0] = 0          # 朱丝栏 / 朱印不参与找列
    clean = _remove_lines(bw)
    proj = clean.sum(axis=0) / 255.0
    if proj.max() <= 0:
        return []
    # 自适应谷阈值：以非零投影的中位数为基准，取 35%（比固定分位数更能适应密排页）
    nz = proj[proj > 0]
    med = float(np.median(nz)) if nz.size else 0.0
    thr = max(2.0, med * 0.35)
    smooth = np.convolve(proj, np.ones(5) / 5, mode="same")
    segs = _segments(smooth > thr, max(1, CONFIG.ocr.col_min_w))

    cols: List[Dict[str, Any]] = []
    for a, b in segs:
        band = clean[:, a:b]
        rows = np.where(band.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        if (y1 - y0) < h * MIN_COL_RATIO:      # 页眉 / 页脚 / 印章碎片
            continue
        cols.append({"bbox": [int(a), y0, int(b), y1],
                     "crop": img[y0:y1, a:b].copy()})
    # 从右向左
    cols.sort(key=lambda c: -((c["bbox"][0] + c["bbox"][2]) / 2))
    for i, c in enumerate(cols):
        c["col_index"] = i
    return cols


def is_vertical_page(img: np.ndarray) -> bool:
    """按列数 + 列平均高宽比判断是否竖排页。"""
    if img is None or img.size == 0:
        return False
    cols = split_columns(img)
    if len(cols) < 5:
        return False
    aspects = []
    for c in cols:
        x1, y1, x2, y2 = c["bbox"]
        if x2 > x1:
            aspects.append((y2 - y1) / (x2 - x1))
    if not aspects:
        return False
    return float(np.median(aspects)) >= VERTICAL_ASPECT


def rotate_bbox_back(bbox: List[float], crop_shape: Tuple[int, int],
                     rotate: int = 90) -> List[int]:
    """把**旋转后**图像中的 bbox 映射回旋转前坐标（供需要先旋转再切块的实现复用）。

    crop_shape: 旋转前的 (h, w)
    rotate=90  （逆时针）：原图 (x,y) → 旋转图 (y, W-1-x)；反变换 (x',y') → (W-1-y', x')
    rotate=270 （顺时针）：原图 (x,y) → 旋转图 (H-1-y, x)；反变换 (x',y') → (y', H-1-x')
    """
    h, w = int(crop_shape[0]), int(crop_shape[1])
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if int(rotate) == 90:
        back = [(w - 1 - y, x) for x, y in pts]
    else:
        back = [(y, h - 1 - x) for x, y in pts]
    xs = [p[0] for p in back]
    ys = [p[1] for p in back]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _cut_points(rowproj: np.ndarray, max_h: int) -> List[int]:
    """按 max_h 找切点，并在 ±40px 内取投影最低处，尽量落在字间隙。"""
    h = len(rowproj)
    if h <= max_h:
        return [0, h]
    pts = [0]
    y = max_h
    while y < h - MIN_PIECE_H:
        lo, hi = max(1, y - 40), min(h - 1, y + 40)
        if hi <= lo:
            break
        best = lo + int(np.argmin(rowproj[lo:hi]))
        if best - pts[-1] < MIN_PIECE_H:
            best = min(h, pts[-1] + max_h)
        pts.append(best)
        y = best + max_h
    # 末尾过短的一块并入前一块，避免出现 60px 的残块被丢弃
    while len(pts) >= 2 and (h - pts[-1]) < MIN_PIECE_H:
        pts.pop()
    pts.append(h)
    return pts


def ocr_column(crop: np.ndarray, max_h: int = MAX_PIECE_H,
               rotate: Optional[int] = None) -> List[Dict[str, Any]]:
    """列内切块 + 旋转 + 识别，返回 [{text, conf, bbox_in_col}]。"""
    if crop is None or crop.size == 0:
        return []
    from .recognize import recognize_crop

    rot = int(rotate if rotate is not None else CONFIG.ocr.rotate)
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    rowproj = _binarize(gray).sum(axis=1) / 255.0
    pts = _cut_points(rowproj, max(120, max_h))

    out: List[Dict[str, Any]] = []
    for y0, y1 in zip(pts[:-1], pts[1:]):
        if y1 - y0 < MIN_PIECE_H:
            continue
        piece = crop[y0:y1]
        # 高度远大于宽度时先旋转，让文字方向变成横向
        if (y1 - y0) / max(1, w) > 2.0:
            code = cv2.ROTATE_90_COUNTERCLOCKWISE if rot == 90 else cv2.ROTATE_90_CLOCKWISE
            piece = cv2.rotate(piece, code)
        text, conf = recognize_crop(piece)
        if not text:
            continue
        out.append({"text": text, "conf": conf,
                    "bbox_in_col": [0, int(y0), int(w), int(y1)]})
    return out


def ocr_page(image_path: str, page_id: str) -> Tuple[List[Block], Dict[str, Any]]:
    """整页竖排识别：切列 → 逐列切块识别 → 坐标映射回原图。"""
    img = imread_unicode(image_path)
    if img is None:
        return [], {"width": 0, "height": 0, "columns": 0, "notes": ["图像读取失败"]}
    h, w = img.shape[:2]
    cols = split_columns(img)
    widths = [(c["bbox"][2] - c["bbox"][0]) for c in cols] or [1]
    med_w = float(np.median(widths))

    blocks: List[Block] = []
    order = 0
    for c in cols:
        x1, y1, x2, y2 = c["bbox"]
        kind = "annotation" if (x2 - x1) < med_w * 0.7 else "paragraph"
        for i, piece in enumerate(ocr_column(c["crop"]), start=1):
            py0 = y1 + piece["bbox_in_col"][1]
            py1 = y1 + piece["bbox_in_col"][3]
            line_id = f"V{c['col_index']:02d}B{i:02d}"
            order += 1
            blocks.append(Block(
                page_id=page_id, bbox=[x1, py0, x2, py1],
                line_id=line_id, quote=piece["text"],
                source_hash=source_hash_of("ocr-v", page_id, line_id, piece["text"]),
                text=piece["text"], text_norm=piece["text"], kind=kind,
                conf=piece["conf"], col_id=c["col_index"], reading_order=order,
                block_ref=f"{page_id}:{line_id}",
            ))
    return blocks, {
        "width": w, "height": h, "columns": len(cols),
        "rotate": CONFIG.ocr.rotate,
        "notes": [f"竖排逐列：{len(cols)} 列，识别 {len(blocks)} 块，旋转 {CONFIG.ocr.rotate}°（逆时针）"],
    }


def describe() -> Dict[str, Any]:
    return {
        "enabled": CONFIG.ocr.vertical_split,
        "col_min_w": CONFIG.ocr.col_min_w,
        "rotate": CONFIG.ocr.rotate,
        "policy": "竖排逐列检测；横排页仍走整页识别；TCM_OCR_VERTICAL_SPLIT=0 可一键回退",
    }
