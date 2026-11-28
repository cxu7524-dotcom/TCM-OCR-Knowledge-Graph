# -*- coding: utf-8 -*-
"""正文尺度：H（字高）、G（字间留白）、glyph_pitch（字心距）。

三者不是一回事，不能互推：
  H     = 字形带高度的中位数
  G     = 相邻字形带之间 a_next − b_i 的中位数
  pitch = 相邻字形带中心差的中位数

「H + G ≈ pitch」只在所有字等高、间距均匀时近似成立，实测里经常差几像素，
所以契约要求分别保存、分别读取。

v2 的关键修正——**墨迹单位统一**：
前景掩膜可以是 0/1，也可以是 0/255。行活跃度若直接对像素值求和，0/255 下一个
墨点就顶 255，阈值形同虚设；0/1 下又几乎永远达不到阈值。两者会得到完全不同的
字形带。这里一律先归一为布尔掩膜，再用**墨点个数**（= 覆盖比例 × 列宽）判定，
保证同一条前景掩膜用 0/1 或 0/255 表示时结果逐字节相同。

另外「继承」与「实测」必须分开记：区域自己样本不足时，可以继承同页尺度，
但 `value_origin` 会标 `inherited_from_page`、`confidence_status` 会标 `inherited`，
绝不把继承值伪装成实测值。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ScaleEstimate:
    H: Optional[float] = None
    G: Optional[float] = None
    glyph_pitch: Optional[float] = None
    page_height: Optional[float] = None
    small_glyph_height: Optional[float] = None
    small_glyph_width: Optional[float] = None
    small_glyph_gap: Optional[float] = None
    small_glyph_pitch: Optional[float] = None
    unit: str = 'px'
    source: str = 'none'
    sample_count: Dict[str, int] = field(default_factory=dict)
    dispersion: Dict[str, Optional[float]] = field(default_factory=dict)
    confidence_status: str = 'failed'      # confident / provisional / inherited / failed
    measured_confidence: str = 'failed'    # 只看本区域自己的样本
    value_origin: str = 'none'             # measured / inherited_from_page / none
    inherited_fields: List[str] = field(default_factory=list)
    inherited_note: Optional[str] = None
    failure_reason: Optional[str] = None
    small_glyph_status: str = 'not_estimated'
    excluded_samples: Dict[str, Any] = field(default_factory=dict)
    filter_conditions: Dict[str, Any] = field(default_factory=dict)
    per_column: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ('H', 'G', 'glyph_pitch', 'page_height'):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        d['inherited_fields'] = list(self.inherited_fields)
        return d

    # ------------------------------------------------------------------ 继承
    def inherit_missing_from(self, page: 'ScaleEstimate', note: str) -> 'ScaleEstimate':
        """只填自己**缺**的字段；填了就把来源标成继承，并逐字段记账。

        自己已经算出来的字段（哪怕只有 H）不算继承——`value_origin` 会区分
        `inherited_from_page`（全继承）与 `partially_inherited`（部分继承）。
        """
        if page is None:
            return self
        own = [f for f in ('H', 'G', 'glyph_pitch') if getattr(self, f) is not None]
        fills: List[str] = []
        for name in ('H', 'G', 'glyph_pitch'):
            if getattr(self, name) is None and getattr(page, name) is not None:
                setattr(self, name, getattr(page, name))
                fills.append(name)
        if not fills:
            return self
        self.value_origin = 'inherited_from_page' if not own else 'partially_inherited'
        self.inherited_fields = fills
        self.inherited_note = note
        self.confidence_status = 'inherited'
        self.source = 'inherited_from_page' if not own else 'partially_inherited'
        self.excluded_samples.setdefault('inherited_sample_count', dict(page.sample_count))
        self.excluded_samples.setdefault('own_measured_fields', own)
        self.filter_conditions.setdefault('inherited_from_page_scale', True)
        return self


def as_ink_mask(bw: np.ndarray) -> np.ndarray:
    """把任意前景编码归一为布尔掩膜（0/1 与 0/255 走同一条路）。

    只做「>0」，不做阈值：编码差异不该由尺度模块猜。
    """
    if bw is None:
        return np.zeros((0, 0), dtype=bool)
    arr = np.asarray(bw)
    if arr.dtype == bool:
        return arr
    return arr > 0


def glyph_bands(bw: np.ndarray, x1: int, x2: int, y0: int, y1: int,
                cfg: Dict[str, Any],
                exclude_x: Sequence[Tuple[int, int]] = ()) -> List[Dict[str, Any]]:
    """在一列内按行投影切出字形带，返回 [{y0,y1,h,cx,width,ink_rows}]。

    判据沿用 ocr/small_columns.py 的经验（行活跃度 ≥ 列宽的 11%、带高不超过列宽 1.4 倍），
    但把「宽 ≥ 列宽 45%」作为独立门槛，避免把标点、栏线残渣当成字。
    exclude_x 里是高置信**独立栏线**的 x 区间：这些像素先置零再投影，
    既不参与行活跃度，也不计入字宽；贴字栏线不在此列（它本身就是笔画的一部分）。
    """
    x1, x2 = int(x1), int(x2)
    col_w = x2 - x1
    if col_w < 6 or y1 - y0 < 30:
        return []
    band = as_ink_mask(bw[int(y0):int(y1), x1:x2]).copy()
    if band.size == 0:
        return []
    # 排除独立栏线像素（框内任意位置，不只两端；4 元组时只排除该 y 区间，
    # 这样断线的断口与贴字段仍保留原图墨迹）
    applied: List[List[int]] = []
    for item in exclude_x:
        a, b = int(item[0]), int(item[1])
        ya = int(item[2]) if len(item) >= 4 else int(y0)
        yb = int(item[3]) if len(item) >= 4 else int(y1)
        ia = max(0, a - x1)
        ib = min(col_w, b - x1)
        ra = max(0, ya - int(y0))
        rb = min(int(y1) - int(y0), yb - int(y0))
        if ib > ia and rb > ra:
            band[ra:rb, ia:ib] = False
            applied.append([a, b, ya, yb])
    active_w = col_w - sum(x[1] - x[0] for x in applied)
    if active_w < 4:
        return []

    support = max(int(cfg.get('band_min_support_px', 2)),
                  int(np.ceil(active_w * float(cfg.get('band_support_ratio', 0.11)))))
    ink_per_row = band.sum(axis=1)
    active = (ink_per_row >= support).astype(np.uint8)
    active = cv2.morphologyEx(active[:, None], cv2.MORPH_CLOSE,
                              np.ones((3, 1), np.uint8)).ravel()
    out = []
    idx = np.flatnonzero(np.diff(np.r_[0, active, 0]).astype(np.int16))
    for a, b in zip(idx[::2], idx[1::2]):
        hh = int(b - a)
        if hh < 10 or hh > int(col_w * 1.4):
            continue
        seg = band[a:b]
        cols = np.flatnonzero(seg.sum(axis=0) >= 2)
        if not cols.size:
            continue
        width = int(cols[-1] - cols[0] + 1)
        if width < active_w * float(cfg['band_min_width_ratio']):
            continue
        out.append({'y0': int(y0) + int(a), 'y1': int(y0) + int(b), 'h': hh,
                    'cx': round(float(x1 + (cols[0] + cols[-1]) / 2.0), 2),
                    'width': width, 'col_w': col_w,
                    'ink_rows': int(ink_per_row[a:b].max()) if hh else 0,
                    'row_support_px': int(support),
                    'excluded_rule_bands': applied})
    return out


def _median(vals: Sequence[float]) -> Optional[float]:
    return float(np.median(vals)) if len(vals) else None


def _iqr(vals: Sequence[float]) -> Optional[float]:
    if len(vals) < 4:
        return None
    q = np.percentile(np.asarray(vals, dtype=float), [25, 75])
    return float(q[1] - q[0])


def estimate_scale(bw: np.ndarray, columns: Sequence[Dict[str, Any]],
                   y_range: Tuple[int, int], image_size: Tuple[int, int],
                   cfg: Dict[str, Any],
                   exclude_rules: Sequence[Tuple[int, int]] = (),
                   narrow_columns: Optional[int] = None) -> ScaleEstimate:
    """按「可信栏 → 同半页正文证据」的顺序估尺度。

    narrow_columns 由调用方给：列数 <= narrow_width_cols（或总宽 <= narrow_width_px）
    时按窄裁块放宽样本门槛，但**不降低判定标准**，只是允许更少样本参与。
    """
    est = ScaleEstimate(page_height=float(image_size[1]))
    widths = [int(c['bbox'][2]) - int(c['bbox'][0]) for c in columns]
    total_w = sum(widths)
    est.filter_conditions = {
        'y_band': [int(y_range[0]), int(y_range[1])],
        'band_min_width_ratio': cfg['band_min_width_ratio'],
        'band_support_ratio': cfg.get('band_support_ratio', 0.11),
        'band_min_support_px': cfg.get('band_min_support_px', 2),
        'ink_unit': 'binary_gt0',
        'height_trim': list(cfg['height_trim']),
        'gap_max_px': cfg['gap_max_px'],
        'excluded_rule_columns': [list(map(int, r)) for r in exclude_rules],
    }
    n_cols = (narrow_columns if narrow_columns is not None else len(columns))
    narrow = (n_cols <= cfg['narrow_width_cols']
              or (columns and total_w <= int(cfg.get('narrow_width_px', 60))))
    min_samples = int(cfg['min_samples_narrow'] if narrow else cfg['min_samples_H'])
    est.filter_conditions['narrow_block'] = bool(narrow)
    est.filter_conditions['min_samples'] = min_samples

    bands: List[Dict[str, Any]] = []
    per_col = []
    for i, c in enumerate(columns):
        b = [int(v) for v in c['bbox']]
        got = glyph_bands(bw, b[0], b[2], y_range[0], y_range[1], cfg, exclude_rules)
        per_col.append({'col_index': i, 'bbox': b, 'bands': len(got),
                        'rule_bands_excluded': sum(1 for g in got if g['excluded_rule_bands']) and
                        len(got[0]['excluded_rule_bands']) if got else 0})
        for g in got:
            bands.append({**g, 'col_index': i})
    est.per_column = per_col

    if not bands:
        est.failure_reason = ('no_glyph_band：该范围内没有满足「高 10px~1.4 列宽、'
                              f'宽 ≥ {cfg["band_min_width_ratio"]:.0%} 列宽、行活跃度 ≥ '
                              f'{cfg.get("band_support_ratio", 0.11):.0%} 列宽」的字形带')
        return est

    heights = [g['h'] for g in bands]
    h0 = _median(heights) or 0.0
    lo, hi = cfg['height_trim'][0] * h0, cfg['height_trim'][1] * h0
    kept = [g for g in bands if lo <= g['h'] <= hi]
    trimmed = [g for g in bands if not (lo <= g['h'] <= hi)]
    est.excluded_samples['height_trimmed'] = [
        {'y': [g['y0'], g['y1']], 'h': g['h'], 'col': g['col_index']} for g in trimmed[:20]]
    if len(kept) < min_samples:
        est.failure_reason = (f'insufficient_samples：去离群后 {len(kept)} 个 < {min_samples}'
                              f'（原始 {len(bands)}，剔除 {len(trimmed)}）')
        est.sample_count = {'H': len(kept), 'G': 0, 'glyph_pitch': 0}
        return est

    est.H = _median([g['h'] for g in kept])
    est.sample_count['H'] = len(kept)
    est.dispersion['H_iqr'] = _iqr([g['h'] for g in kept])

    # G 与 pitch 只在**同列相邻**带之间算（跨列没有「字间距」语义）
    gaps, pitches = [], []
    excluded_gaps = []
    for i in range(len(kept) - 1):
        a, b = kept[i], kept[i + 1]
        if a['col_index'] != b['col_index']:
            continue
        gap = b['y0'] - a['y1']
        if gap < 0:
            excluded_gaps.append({'y': [a['y1'], b['y0']], 'gap': gap,
                                  'why': '字形带重叠（连笔或切分错误）'})
            continue
        if gap > cfg['gap_max_px']:
            excluded_gaps.append({'y': [a['y1'], b['y0']], 'gap': gap,
                                  'why': '留白过大，按段落间隔剔除'})
            continue
        gaps.append(gap)
        pitches.append((b['y0'] + b['y1']) / 2.0 - (a['y0'] + a['y1']) / 2.0)
    est.excluded_samples['gaps_excluded'] = excluded_gaps[:20]

    if len(gaps) >= max(3, min_samples - 2):
        est.G = _median(gaps)
        est.sample_count['G'] = len(gaps)
        est.dispersion['G_iqr'] = _iqr(gaps)
    else:
        est.excluded_samples['G_failure'] = f'有效留白样本 {len(gaps)} 不足'
    if len(pitches) >= max(3, min_samples - 2):
        est.glyph_pitch = _median(pitches)
        est.sample_count['glyph_pitch'] = len(pitches)
        est.dispersion['pitch_iqr'] = _iqr(pitches)
    else:
        est.excluded_samples['pitch_failure'] = f'有效字心距样本 {len(pitches)} 不足'

    if est.G is None or est.glyph_pitch is None:
        est.confidence_status = 'failed'
        est.measured_confidence = 'failed'
        est.failure_reason = ('partial：H 有值，但 '
                              + ('G ' if est.G is None else '')
                              + ('glyph_pitch ' if est.glyph_pitch is None else '')
                              + '样本不足')
        est.source = 'glyph_bands_partial'
        est.value_origin = 'measured'
        return est

    est.measured_confidence = ('confident' if len(kept) >= int(cfg['min_samples_H'])
                               else 'provisional')
    est.confidence_status = est.measured_confidence
    est.value_origin = 'measured'
    est.source = 'glyph_bands_same_halfpage'
    if narrow:
        est.source = 'glyph_bands_narrow_block'
    return est
