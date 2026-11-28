# -*- coding: utf-8 -*-
"""stage8 修正一：列槽切分（只补「半页被并成一列」这一个环节）。

定位依据（tools/dual_track_stage8_calib.py + 页级审计，2026-11-25）：
  - 19 页里只有 丹方彚编/page_12 的右半页被 page_layout 并成**一列宽 446px**，
    温疫方論/page_12 也出现 130px / 96px 的过宽列；这两页的已标注组占 13/38。
  - 这两页的相邻列之间**几乎没有空白沟**（p12 的列框首尾相接：885–934 与 934–986），
    所以按「投影低于阈值」切列必然失败——这是纯空白判据的固有上限，不是参数问题。
  - 但列槽间距是强周期的：自相关峰值 p12 右半 = 49（标注实测间距 48.7）、
    温疫 = 44（标注槽宽 43）、思濟堂 = 49。周期可靠，相位可以靠墨量低谷吸附。

所以本模块只做一件事：**把过宽列按周期 + 低谷吸附切成列槽**，
不动列内检测（列内仍是 ocr.small_columns 的既有逻辑）。

判据约束：
  - 只切宽度 > split_ratio × 该半页周期的列（默认 1.45），窄列一律不动，
    保证已在历史回归集上有效的 19/20 不被波及；
  - 切点必须落在平滑投影的局部低谷 ±0.30 周期内，落不到低谷就用周期格点；
  - 任一子槽宽度 < 0.45 周期即放弃该列（宁可退回整列，也不产生碎槽）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- 投影与周期

def ink_projection(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (proj, clean_bw)：与 ocr.vertical.split_columns 完全同一套预处理。"""
    import cv2
    from ocr.vertical import _binarize, _red_mask, _remove_lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = _binarize(gray)
    if img.ndim == 3:
        bw[_red_mask(img) > 0] = 0
    clean = _remove_lines(bw)
    return clean.sum(axis=0) / 255.0, clean


def estimate_pitch(proj: np.ndarray, H: float,
                   lo_ratio: float = 1.3, hi_ratio: float = 3.2) -> Optional[int]:
    """列槽周期：投影自相关在半页内的第一个主峰。"""
    x = np.asarray(proj, dtype=float)
    if x.size < int(hi_ratio * H) + 2 or x.std() <= 0:
        return None
    x = x - x.mean()
    ac = np.correlate(x, x, 'full')[x.size - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo = max(2, int(round(lo_ratio * H)))
    hi = min(ac.size, int(round(hi_ratio * H)))
    if hi <= lo:
        return None
    k = lo + int(np.argmax(ac[lo:hi]))
    return int(k) if ac[k] > 0.15 else None


# ---------------------------------------------------------------- 切点

def _local_minima(s: np.ndarray) -> List[int]:
    if s.size < 3:
        return []
    return [i for i in range(1, s.size - 1) if s[i] <= s[i - 1] and s[i] <= s[i + 1]]


def drift_cuts(proj: np.ndarray, x1: int, x2: int, pitch: int,
               snap_ratio: float = 0.30) -> List[int]:
    """在 [x1,x2) 内按周期推进切点，每一步吸附到最近的低谷。

    逐步推进（而不是铺一张全局格点）是为了跟住木刻版的**漂移**：
    p12 下半栏的槽位比上半栏整体右移约 10px，全局格点会切错。
    """
    sm = np.convolve(proj, np.ones(5) / 5, mode='same')
    cand = [i for i in _local_minima(sm) if x1 < i < x2]
    tol = max(2, int(round(snap_ratio * pitch)))
    cuts: List[int] = []
    cur = float(x1)
    while cur + pitch * 1.45 < x2:
        target = cur + pitch
        best, best_v = None, None
        for c in cand:
            if abs(c - target) > tol:
                continue
            v = float(sm[c])
            if best_v is None or v < best_v:
                best, best_v = c, v
        cut = int(best) if best is not None else int(round(target))
        if not cuts or cut - cuts[-1] >= pitch * 0.45:
            cuts.append(cut)
        cur = cut
    return cuts


def best_phase_cuts(proj: np.ndarray, x1: int, x2: int, pitch: int,
                    min_w: float = 0.45) -> Optional[Dict[str, Any]]:
    """在一段投影上**枚举相位**，取「切点落在墨最少处」的那一组。

    与 drift_cuts 的区别：drift_cuts 从左边界按周期推进、每步只在 ±0.3 周期内
    吸附，所以相位被左边界钉死；一旦整段文字的版式相对列框整体平移（木刻版下半
    栏常见），推进法永远吸到错的低谷。这里反过来，把 0..pitch-1 的相位全试一遍，
    用切点处的墨量当分数，谁的切点最空就用谁。
    """
    sm = np.convolve(np.asarray(proj, dtype=float), np.ones(5) / 5, mode='same')
    best = None
    for p in range(pitch):
        cuts = [x1 + p + pitch * k for k in range(0, 16)]
        cuts = [c for c in cuts if x1 < c < x2]
        if not cuts:
            continue
        edges = [x1] + cuts + [x2]
        if min(b - a for a, b in zip(edges, edges[1:])) < min_w * pitch:
            continue
        sc = float(np.mean([sm[max(0, c - 1):c + 2].min() for c in cuts]))
        if best is None or sc < best['score']:
            best = {'score': round(sc, 3), 'phase': int(p), 'cuts': cuts}
    return best


def band_boundary(rows: np.ndarray, y1: int, y2: int) -> int:
    """上下两栏的分界行：中部 30%–70% 之间墨量最少的一行。"""
    h = y2 - y1
    a, b = y1 + int(0.30 * h), y1 + int(0.70 * h)
    if b - a < 2:
        return (y1 + y2) // 2
    seg = rows[a:b]
    return int(a + int(np.argmin(seg)))


def split_wide_columns(cols: Sequence[Dict[str, Any]], proj: np.ndarray, H: float,
                       cfg: Optional[Dict[str, Any]] = None,
                       clean: Optional[np.ndarray] = None
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把过宽列按周期切成列槽；返回 (新列表, 审计记录)。"""
    c = (cfg or {}).get('slots') or {}
    ratio = float(c.get('split_ratio', 1.45))
    min_w = float(c.get('min_slot_ratio', 0.45))
    banded = bool((cfg or {}).get('band_slots'))
    drift_min = float((cfg or {}).get('band_drift_ratio', 0.20))
    out: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for col in cols:
        x1, y1, x2, y2 = (int(v) for v in col['bbox'])
        w = x2 - x1
        pitch = estimate_pitch(proj[max(0, x1 - 40):min(proj.size, x2 + 40)], H)
        pitch = pitch or int(round(2.2 * H))
        if w <= ratio * pitch:
            out.append(col)
            continue

        # 分栏相位：只对「本来就切得开」的过宽列生效，窄列一律不动
        if banded and clean is not None:
            yb = band_boundary(clean[:, max(0, x1):min(clean.shape[1], x2)].sum(axis=1), y1, y2)
            # 传整幅宽的投影：best_phase_cuts 里的切点是**绝对 x**，切片会错位
            pu = best_phase_cuts(clean[y1:yb].sum(axis=0) / 255.0, x1, x2, pitch, min_w)
            pl = best_phase_cuts(clean[yb:y2].sum(axis=0) / 255.0, x1, x2, pitch, min_w)
            if pu and pl:
                gap = abs(pu['phase'] - pl['phase'])
                gap = min(gap, pitch - gap)
                if gap >= drift_min * pitch:
                    rec = {'column': [x1, y1, x2, y2], 'pitch': pitch, 'action': 'band_split',
                           'band_y': int(yb), 'phase': [pu['phase'], pl['phase']],
                           'phase_gap': int(gap), 'cuts': [pu['cuts'], pl['cuts']],
                           'score': [pu['score'], pl['score']]}
                    audit.append(rec)
                    for (ya, yb2), ph in (((y1, yb), pu), ((yb, y2), pl)):
                        edges = [x1] + ph['cuts'] + [x2]
                        for a, b in zip(edges, edges[1:]):
                            sub = dict(col)
                            sub['bbox'] = [int(a), int(ya), int(b), int(yb2)]
                            sub['slot_of'] = [x1, x2]
                            sub['band'] = [int(y1), int(y2)]
                            out.append(sub)
                    continue

        cuts = drift_cuts(proj, x1, x2, pitch)
        edges = [x1] + cuts + [x2]
        widths = [b - a for a, b in zip(edges, edges[1:])]
        rec = {'column': [x1, y1, x2, y2], 'pitch': pitch, 'n_cut': len(cuts),
               'cuts': cuts, 'widths': widths}
        if not cuts or min(widths) < min_w * pitch:
            rec['action'] = 'kept_whole'
            audit.append(rec)
            out.append(col)
            continue
        rec['action'] = 'split'
        audit.append(rec)
        for a, b in zip(edges, edges[1:]):
            sub = dict(col)
            sub['bbox'] = [int(a), int(y1), int(b), int(y2)]
            sub['slot_of'] = [x1, x2]
            out.append(sub)
    out.sort(key=lambda d: -((d['bbox'][0] + d['bbox'][2]) / 2))
    for i, d in enumerate(out):
        d['col_index'] = i
    return out, audit


# ---------------------------------------------------------------- 收紧

def tighten_slots(cols: Sequence[Dict[str, Any]], clean: np.ndarray,
                  min_keep: float = 0.55) -> List[Dict[str, Any]]:
    """把列槽的 x 边界收到**去线后**的墨迹外接框。

    为什么需要：自相关只定周期，切点落在低谷附近，实测系统性偏 +4~5px
    （p12 切在 594/641/690…，标注在 589/637/686…）。而标注框的构造规则本身
    就是「列窗 + 中缝测量后取墨迹外接框」，所以按墨迹收紧是在对齐构造口径，
    不是把预测往标注上凑。去线（clean）已排除版框/界行，不会把直线算成墨迹。
    """
    out: List[Dict[str, Any]] = []
    for c in cols:
        x1, y1, x2, y2 = (int(v) for v in c['bbox'])
        x1c, x2c = max(0, x1), min(clean.shape[1], x2)
        if x2c - x1c < 4:
            out.append(c)
            continue
        colsum = clean[max(0, y1):y2, x1c:x2c].sum(axis=0)
        xs = np.flatnonzero(colsum > 0)
        if xs.size == 0:
            out.append(c)
            continue
        a, b = x1c + int(xs.min()), x1c + int(xs.max()) + 1
        if b - a < min_keep * (x2 - x1):
            out.append(c)
            continue
        d = dict(c)
        d['bbox'] = [a, y1, b, y2]
        d['slot_untightened'] = [x1, x2]
        out.append(d)
    return out


# ---------------------------------------------------------------- 页面级

def layout_slots(img: np.ndarray, H: float,
                 cfg: Optional[Dict[str, Any]] = None,
                 tighten: bool = False) -> Dict[str, Any]:
    """整页列槽：先走 page_layout，再只对过宽列做周期切分。"""
    from ocr.vertical import split_spread, split_columns
    h, w = img.shape[:2]
    cols: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for hx0, hx1 in split_spread(img):
        sub = img[:, hx0:hx1]
        proj, clean = ink_projection(sub)
        base = split_columns(sub)
        parts, aud = split_wide_columns(base, proj, H, cfg, clean=clean)
        if tighten:
            parts = tighten_slots(parts, clean)
        for p in parts:
            b = list(p['bbox'])
            b[0] += hx0; b[2] += hx0
            p = dict(p); p['bbox'] = b
            if p.get('slot_untightened'):
                p['slot_untightened'] = [p['slot_untightened'][0] + hx0,
                                         p['slot_untightened'][1] + hx0]
            cols.append(p)
        for a in aud:
            a['half'] = [int(hx0), int(hx1)]
            audits.append(a)
    cols.sort(key=lambda d: -((d['bbox'][0] + d['bbox'][2]) / 2))
    for i, d in enumerate(cols):
        d['col_index'] = i
    return {'width': w, 'height': h, 'cols': cols, 'slot_audit': audits}
