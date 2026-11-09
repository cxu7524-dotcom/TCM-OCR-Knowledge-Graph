# -*- coding: utf-8 -*-
"""
ocr/vertical.py —— 竖排古籍逐列检测与识别（P1）

问题：rapidocr 的 det 模型是横排训练的，在木刻竖排页上会把竖列横切成 1–3 字的小框，
覆盖率低、CER 高（见 docs/真实古籍页诊断.md）。

做法（顺序不能颠倒）：
  1. **书脊检测**（`split_spread`）：对开双页先切成两个半页，右半页先读；
     判据三路并用（列间隙 / 装订阴影 / 亮纸带）+ 物理先验（高宽比 > 1.2 视为竖长单页，不切）；
  2. 灰度 → 自适应二值化 → 中值去噪（二值图**只用于找列边界**，识别仍用原图裁剪）；
  3. 用形态学长直线（版框 / 界行）从二值图里减掉，避免边框被当成列；
  4. 垂直投影找列间隙 → 过滤窄列与矮列 → **跨列框抑制**（宽度 > 1.8× 中位的段按内部次峰再切一刀）
     → **从右向左**编号 col_index（跨半页全局连续）；
  5. 列内水平投影切块（切点落在字间隙最低处，避免切断字），裁剪时左右各留 `COL_PAD` 像素，
     避免字缘笔画被切掉；
  6. 每块旋转 **90° 逆时针**（实测方向，见下）后调用 `recognize_crop()`；
  7. bbox 加列偏移（含半页偏移）映射回原图坐标系。

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
GUTTER_WINDOW = (0.38, 0.62)  # 书脊候选窗口：页面中线附近
GUTTER_MIN_RATIO = 0.02       # 中缝宽度下限（占页宽），小于此视为列间隙而非书脊
WIDE_COL_FACTOR = 1.8         # 列宽 > 1.8× 中位列宽 → 判为跨列框（两列被合并）
COL_PAD = 5                   # 列裁剪左右留白：切列点落在投影最低处，字缘笔画可能被切掉；
                              # 留 5px（约列宽 15%，仍小于列间隙）实测逐列 CER 中位 0.5700→0.4873


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


def _profiles(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (原始前景投影, 去长线后前景投影, 灰度列均值)，均按 x 轴（列方向）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    bw[_red_mask(img) > 0] = 0
    clean = _remove_lines(bw)
    return (bw.sum(axis=0) / 255.0, clean.sum(axis=0) / 255.0, gray.mean(axis=0))


def _vertical_line_peaks(img: np.ndarray) -> List[Tuple[int, int, float]]:
    """贯穿全页的长竖线峰值 [(x0, x1, peak)]：版框边、界行、书脊都算。

    医书对开页左右各有一个矩形版框，两框相邻处有一条（或一对紧挨的）长竖线，
    落在页面中线附近——这是比「列间隙」更强的书脊证据（《丹方彚编》《合併脉訣》
    这类宽高比只有 1.06~1.15 的对开页，列间隙法常常抓不到）。
    """
    if not _CV2_OK:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = _binarize(gray)
    bw[_red_mask(img) > 0] = 0
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25))))
    proj = vert.sum(axis=0) / 255.0
    if proj.max() <= 0:
        return []
    thr = 0.45 * float(proj.max())
    peaks: List[Tuple[int, int, float]] = []
    i = 0
    while i < w:
        if proj[i] > thr:
            j = i
            while j < w and proj[j] > thr:
                j += 1
            peaks.append((i, j, float(proj[i:j].max())))
            i = j
        else:
            i += 1
    return peaks


def _gutter_candidates(img: np.ndarray) -> List[Tuple[float, int, int, str]]:
    """书脊候选区间 (score, x0, x1, 依据)，四路证据取并集。

    ① 列间隙法：把全页切列后，中窗内**明显宽于常态**的列间隙——对开页两半的间距
       通常是普通列间隙的 1.5 倍以上（实测 page_10 1.93×、page_11 6.25×、page_13(1) 3.49×）；
    ② 阴影型中缝：装订阴影是贯穿全页的长暗带，灰度显著低于纸张中位，
       但它会被 `_remove_lines` 当成竖线减掉，因此去线后前景≈0；
    ③ 空白型中缝：中窗内去线后前景持续接近 0 的亮纸带；
    ④ 中缝竖线（阶段六新增）：版框边 / 书脊在页面中线附近形成的贯穿全页长竖线，
       对宽高比 1.06~1.15 的医书对开页最有效。
    """
    raw, clean, gray = _profiles(img)
    h, w = img.shape[:2]
    lo, hi = int(GUTTER_WINDOW[0] * w), int(GUTTER_WINDOW[1] * w)
    out: List[Tuple[float, int, int, str]] = []

    # ④ 中缝竖线（先算，权重最高）
    nlo, nhi = int(0.42 * w), int(0.58 * w)
    mid = [p for p in _vertical_line_peaks(img) if nlo <= (p[0] + p[1]) / 2 <= nhi]
    if mid:
        a, b, _peak = max(mid, key=lambda t: t[2])
        out.append((6.0, a, b, "中缝竖线"))

    # ① 列间隙法
    cols = sorted(split_columns(img), key=lambda c: c["bbox"][0])
    if len(cols) >= 6:
        gaps = [(cols[i]["bbox"][2], cols[i + 1]["bbox"][0]) for i in range(len(cols) - 1)]
        widths = [b - a for a, b in gaps]
        med_gap = float(np.median(widths)) or 1.0
        for a, b in gaps:
            mid = (a + b) / 2.0
            width = b - a
            if not (lo <= mid <= hi):
                continue
            if width >= max(1.4 * med_gap, 0.014 * w):
                out.append((width / med_gap, a, b, "列间隙"))
    # ② 阴影型
    med_gray = float(np.median(gray)) or 1.0
    dark = np.convolve((gray < med_gray * 0.82).astype(np.int8), np.ones(3) / 3.0, mode="same")
    for a, b in _segments(dark[lo:hi] > 0.6, 4):
        a, b = a + lo, b + lo
        if float(clean[a:b].mean()) < 0.35 * (float(np.median(clean[clean > 0])) or 1.0):
            out.append((3.0, a, b, "装订阴影"))
    # ③ 空白型
    nz = clean[clean > 0]
    thr = max(2.0, (float(np.median(nz)) if nz.size else 0.0) * 0.10)
    smooth = np.convolve(clean, np.ones(3) / 3.0, mode="same")
    for a, b in _segments(smooth[lo:hi] < thr, 4):
        a, b = a + lo, b + lo
        if (b - a) >= max(6, int(w * GUTTER_MIN_RATIO * 0.6)):
            out.append((2.0, a, b, "亮纸带"))
    return out


def split_spread(img: np.ndarray) -> List[Tuple[int, int]]:
    """检测对开双页的书脊，返回**按阅读顺序**排列的半页 x 区间。

    古籍对开扫描有两种中缝形态，判据要能同时覆盖：
      ① 装订阴影型（如《杜工部集》）：中缝是一条暗带，原始前景投影很高，
         但它是**贯穿全页的长竖条**，会被 `_remove_lines` 整条减掉 → 去线后投影≈0；
      ② 空白型（如《兵鏡》）：两页之间是亮纸带 → 原始与去线后投影都≈0。
    因此统一用「去长线后的垂直投影在页面中线附近出现持续低谷」作主判据，
    再用「原始投影远高于去线后投影」识别阴影型，放宽其宽度下限。

    安全阀（检测不到就返回整页，**不强行切**）：
      - 切分后左右两半各自至少能切出 3 列（否则可能是单页中间的空白）；
      - 两半的前景质量各占全页 ≥ 15%；
      - 两半宽度差不超过页宽 20%（对开页两半大致等宽）。
    """
    if not _CV2_OK or img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    if w < 120 or h < 120:
        return [(0, w)]
    # 物理先验：单页古籍是竖长版式（高/宽 > 1.2），对开页横向更宽。
    # 这条判据单独就能挡掉绝大多数「单页中间空白被误判成书脊」的误切（实测天文問答 1.40）。
    if h / float(w) > 1.2:
        return [(0, w)]
    raw, clean, _ = _profiles(img)
    total_mass = float(clean.sum()) or 1.0
    for score, a, b, why in sorted(_gutter_candidates(img), key=lambda t: -t[0]):
        # 竖线型候选本身就是一条细线（1~8px），宽度下限按证据类型区分。
        # 实测有页的中缝线只有 1px 宽（合併 page_12），下限设 2 会把它整条丢掉。
        min_w = 1 if why == "中缝竖线" else max(8, int(0.02 * w))
        if b - a < min_w or a < 10 or b > w - 10:
            continue
        if abs(a - (w - b)) > 0.2 * w:            # 两半宽度差过大 → 不像对开页
            continue
        # 竖线证据强，不要求两半墨量均衡（医书目录页右半页字少是常态）；
        # 其他证据仍要求两半各占 ≥15% 前景，避免把单页中间的空白当书脊
        if why != "中缝竖线":
            if min(float(clean[:a].sum()), float(clean[b:].sum())) / total_mass < 0.15:
                continue
        # 阶段七 P2：中缝竖线是「左右两个独立版框」的直接物理证据，
        # 此时放宽列数安全阀到「每半 ≥1 列」——医书对开页的右半常是目录/插图，
        # 只能切出 1~2 列（阶段六 3/6 误判即由此而来）。其他证据仍要求 ≥3 列。
        min_half_cols = 1 if why == "中缝竖线" else 3
        if (len(split_columns(img[:, :a])) < min_half_cols
                or len(split_columns(img[:, b:])) < min_half_cols):
            continue
        # 右半页先读（古籍从右往左翻）
        return [(b, w), (0, a)]
    return [(0, w)]




def _split_wide_column(img: np.ndarray, clean: np.ndarray, col: Dict[str, Any],
                       med_w: float) -> List[Dict[str, Any]]:
    """跨列框抑制：宽度 > 1.8× 中位列宽的列，按内部投影次峰再切一刀。

    截图问题「一个框横跨两列」在逐列路径里的成因是**两列被并进同一个段**
    （列间隙太窄或书脊两侧列粘连）。切分点取列内 30%~70% 区间投影最低处，
    切完两半都不得窄于 col_min_w，否则放弃切分（宁可不切，也不切坏字）。
    """
    x1, y1, x2, y2 = col["bbox"]
    wdt = x2 - x1
    if wdt <= med_w * WIDE_COL_FACTOR or wdt < 20:
        return [col]
    band = clean[y1:y2, x1:x2]
    if band.size == 0:
        return [col]
    prof = band.sum(axis=0) / 255.0
    smooth = np.convolve(prof, np.ones(3) / 3.0, mode="same")
    lo, hi = int(wdt * 0.30), int(wdt * 0.70)
    if hi - lo < 4:
        return [col]
    k = lo + int(np.argmin(smooth[lo:hi]))
    min_w = max(8, CONFIG.ocr.col_min_w)
    if min(k, wdt - k) < min_w:
        return [col]

    pieces: List[Dict[str, Any]] = []
    W = img.shape[1]
    for a, b in ((0, k), (k, wdt)):
        sub_clean = clean[y1:y2, x1 + a:x1 + b]
        rows = np.where(sub_clean.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        yy0, yy1 = y1 + int(rows[0]), y1 + int(rows[-1]) + 1
        cx1 = max(0, x1 + a - COL_PAD)
        cx2 = min(W, x1 + b + COL_PAD)
        pieces.append({"bbox": [x1 + a, yy0, x1 + b, yy1],
                       "crop": img[yy0:yy1, cx1:cx2].copy(),
                       "wide_split": True})
    return pieces if len(pieces) == 2 else [col]


def split_columns(img: np.ndarray) -> List[Dict[str, Any]]:
    """切列，返回 [{bbox, crop, col_index}]，按 x 中心**从右向左**排序。

    切完会做一次「跨列框抑制」：宽度异常（> 1.8× 中位）的列按内部次峰再切一刀，
    被切过的列带 `wide_split=True`，供 `ocr_page()` 统计跨列框率。
    """
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
                     "crop": img[y0:y1, max(0, a - COL_PAD):min(w, b + COL_PAD)].copy()})
    if cols:
        med_w = float(np.median([c["bbox"][2] - c["bbox"][0] for c in cols]))
        fixed: List[Dict[str, Any]] = []
        for c in cols:
            fixed.extend(_split_wide_column(img, clean, c, med_w))
        cols = fixed
    # 从右向左
    cols.sort(key=lambda c: -((c["bbox"][0] + c["bbox"][2]) / 2))
    for i, c in enumerate(cols):
        c["col_index"] = i
    return cols


def page_layout(image_path: str) -> Dict[str, Any]:
    """整页列结构（含双页切分），供评测计算列数准确率与跨列框率。

    返回的 cols 已按阅读顺序排列（右半页 → 左半页，半页内从右向左），
    bbox 为**整页坐标系**。
    """
    img = imread_unicode(image_path)
    if img is None:
        return {"width": 0, "height": 0, "halves": [], "cols": []}
    h, w = img.shape[:2]
    halves = split_spread(img)
    cols: List[Dict[str, Any]] = []
    for hx0, hx1 in halves:
        for c in split_columns(img[:, hx0:hx1]):
            x1, y1, x2, y2 = c["bbox"]
            cols.append({"bbox": [x1 + hx0, y1, x2 + hx0, y2],
                         "half": [hx0, hx1], "wide_split": bool(c.get("wide_split"))})
    return {"width": w, "height": h, "halves": halves, "cols": cols}


def cross_column_rate(blocks, cols: List[Dict[str, Any]]) -> Dict[str, Any]:
    """跨列框率：一个框横跨 ≥2 列（按 x 区间重叠 ≥30% 框宽判定）的框数 / 总框数。

    逐列路径下框=列片段，正常情况下该值为 0；整页路径（det 输出）常出现横跨框，
    这个数字就是「修前/修后」最直接的对照。
    """
    xs = [(c["bbox"][0], c["bbox"][2]) for c in cols]
    total = 0
    cross = 0
    examples: List[Dict[str, Any]] = []
    for b in blocks:
        try:
            bx1, _, bx2, _ = [float(v) for v in b.bbox[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        total += 1
        bw = max(1.0, bx2 - bx1)
        hit = sum(1 for a, c in xs if min(bx2, c) - max(bx1, a) > 0.3 * bw)
        if hit >= 2:
            cross += 1
            if len(examples) < 5:
                examples.append({"line_id": b.line_id, "bbox": list(b.bbox),
                                 "columns_hit": hit, "text": (b.text or "")[:20]})
    return {"total": total, "cross": cross,
            "rate": round(cross / total, 4) if total else None,
            "examples": examples}


def is_vertical_page(img: np.ndarray) -> bool:
    """按列数 + 列平均高宽比判断是否竖排页（对开页按半页判，避免书脊粘连拉低比例）。"""
    if img is None or img.size == 0:
        return False
    for hx0, hx1 in split_spread(img):
        cols = split_columns(img[:, hx0:hx1])
        if len(cols) < 5:
            continue
        aspects = []
        for c in cols:
            x1, y1, x2, y2 = c["bbox"]
            if x2 > x1:
                aspects.append((y2 - y1) / (x2 - x1))
        if aspects and float(np.median(aspects)) >= VERTICAL_ASPECT:
            return True
    return False


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
    """整页竖排识别：书脊检测 → 半页切列 → 逐列切块识别 → 坐标映射回原图。

    对开双页按「右半页先、左半页后」阅读；`col_index` 跨半页全局连续编号，
    保证 `reading_order` 与古籍阅读顺序一致（评测按 reading_order 拼文本）。
    """
    img = imread_unicode(image_path)
    if img is None:
        return [], {"width": 0, "height": 0, "columns": 0, "notes": ["图像读取失败"]}
    h, w = img.shape[:2]
    halves = split_spread(img)
    spread = len(halves) > 1

    blocks: List[Block] = []
    order = 0
    gcol = 0
    wide_fixed = 0
    col_counts: List[int] = []
    for hx0, hx1 in halves:
        sub = img[:, hx0:hx1]
        cols = split_columns(sub)
        col_counts.append(len(cols))
        wide_fixed += sum(1 for c in cols if c.get("wide_split"))
        widths = [(c["bbox"][2] - c["bbox"][0]) for c in cols] or [1]
        med_w = float(np.median(widths))
        for c in cols:
            x1, y1, x2, y2 = c["bbox"]
            kind = "annotation" if (x2 - x1) < med_w * 0.7 else "paragraph"
            for i, piece in enumerate(ocr_column(c["crop"]), start=1):
                py0 = y1 + piece["bbox_in_col"][1]
                py1 = y1 + piece["bbox_in_col"][3]
                line_id = f"V{gcol:02d}B{i:02d}"
                order += 1
                blocks.append(Block(
                    page_id=page_id, bbox=[x1 + hx0, py0, x2 + hx0, py1],
                    line_id=line_id, quote=piece["text"],
                    source_hash=source_hash_of("ocr-v", page_id, line_id, piece["text"]),
                    text=piece["text"], text_norm=piece["text"], kind=kind,
                    conf=piece["conf"], col_id=gcol, reading_order=order,
                    block_ref=f"{page_id}:{line_id}",
                ))
            gcol += 1

    total_cols = sum(col_counts)
    notes = [f"竖排逐列：{total_cols} 列，识别 {len(blocks)} 块，旋转 {CONFIG.ocr.rotate}°（逆时针）"]
    if spread:
        notes.append(f"对开双页：检测到书脊，按半页分别切列（右 {col_counts[0]} 列 / "
                     f"左 {col_counts[1]} 列），右半页先读")
    if wide_fixed:
        notes.append(f"跨列框抑制：{wide_fixed} 个超宽列按内部次峰二次切分")
    return blocks, {
        "width": w, "height": h, "columns": total_cols,
        "halves": halves, "spread": spread, "col_counts": col_counts,
        "wide_split": wide_fixed,
        "rotate": CONFIG.ocr.rotate, "notes": notes,
    }


def describe() -> Dict[str, Any]:
    return {
        "enabled": CONFIG.ocr.vertical_split,
        "col_min_w": CONFIG.ocr.col_min_w,
        "rotate": CONFIG.ocr.rotate,
        "col_pad": COL_PAD,
        "wide_col_factor": WIDE_COL_FACTOR,
        "policy": ("竖排逐列检测 + 双页对开切分（书脊检测，右页先读）+ 跨列框抑制；"
                   "横排页与单页仍走整页识别；TCM_OCR_VERTICAL_SPLIT=0 可一键回退"),
    }
