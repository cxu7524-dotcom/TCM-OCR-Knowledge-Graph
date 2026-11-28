from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np


@dataclass
class SegmentFeatures:
    h_profile_32: List[float]
    valley_depth: float
    valley_prominence: float
    valley_pos: float
    peak_distance: float
    peak_area_ratio: float
    peak_width_ratio: float
    width_ratio: float
    centroid_x: float
    ink_x1_rel: float
    ink_x2_rel: float
    height_ratio: float
    left_sub_blobs: int
    right_sub_blobs: int
    ink_density: float
    seam_ink_ratio: float = 0.0
    left_ink_height: float = 0.0
    right_ink_height: float = 0.0


@dataclass
class ColumnSegment:
    seg_type: str
    y1: int
    y2: int
    x1: int
    x2: int
    right_box: Optional[Tuple[int, int, int, int]] = None
    left_box: Optional[Tuple[int, int, int, int]] = None
    split_x: Optional[int] = None
    alignment: str = "center"
    score: float = 0.0
    features: Optional[SegmentFeatures] = None


@dataclass
class ColumnSegmentAnalysis:
    width: int
    height: int
    inner_x1: int
    inner_x2: int
    large_char_h: float
    segments: List[ColumnSegment]
    clean_bw: Optional[np.ndarray] = None


class SegmentClassifier(Protocol):
    def classify(self, features: SegmentFeatures, *, adjacent_dual: bool = False) -> Tuple[str, float, str]: ...


class RuleSegmentClassifier:
    def classify(self, features: SegmentFeatures, *, adjacent_dual: bool = False) -> Tuple[str, float, str]:
        f = features
        clean_seam = f.valley_depth >= .84 and f.seam_ink_ratio <= .14
        sparse = (f.valley_depth >= .90 and f.seam_ink_ratio <= .10
                  and f.peak_width_ratio >= .45)
        peaks = (0.32 <= f.valley_pos <= 0.68 and clean_seam
                 and f.peak_distance >= 0.28
                 and f.peak_area_ratio >= (.10 if sparse else .25)
                 and f.peak_width_ratio >= (.45 if sparse else .35))
        count = max(f.left_sub_blobs, f.right_sub_blobs)
        small_pair = (0.22 <= f.height_ratio <= 0.78
                      and f.peak_width_ratio >= .50
                      and f.left_ink_height <= .78 and f.right_ink_height <= .78)
        multiple = count >= 2 and f.height_ratio / count <= .75
        if peaks and (multiple or (count <= 1 and small_pair)):
            return "dual_track", min(1.0, 0.6 + 0.4 * f.valley_depth), "center"
        if f.width_ratio < 0.62:
            if f.centroid_x >= 0.56 and f.ink_x1_rel >= 0.28:
                return "single_track", 0.85, "right"
            if f.centroid_x <= 0.44 and f.ink_x2_rel <= 0.72:
                return "single_track", 0.85, "left"
            if (0.40 < f.centroid_x < 0.60 and f.width_ratio < 0.58
                    and (max(f.left_sub_blobs, f.right_sub_blobs) >= 2
                         or (f.height_ratio <= 0.70 and adjacent_dual))):
                return "single_track", 0.75, "center"
        return "large", 0.75, "center"


def _gray(crop: np.ndarray) -> np.ndarray:
    a = np.asarray(crop)
    if a.ndim == 2:
        return a.astype(np.uint8, copy=False)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return cv2.cvtColor(a.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY if a.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    raise ValueError("crop must be a grayscale, BGR, or BGRA image")


def sauvola_binarize(gray: np.ndarray, *, window: int | None = None, k: float = 0.25, r: float = 128.0) -> np.ndarray:
    gray = _gray(gray)
    if not gray.size:
        return np.zeros(gray.shape, np.uint8)
    if r <= 0:
        raise ValueError("r must be positive")
    win = max(1, int(window)) | 1 if window is not None else max(15, int(round(gray.shape[1] * .65)) | 1)
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(f * f, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    mask = f < mean * (1 + k * (std / r - 1))
    p10, p85, p90 = np.percentile(f, [10, 85, 90])
    if p90 - p10 >= 35:
        mask &= f < p85 - max(12.0, .22 * (p90 - p10))
    return mask.astype(np.uint8) * 255


def strip_column_borders(bw: np.ndarray) -> Tuple[np.ndarray, int, int]:
    clean = (np.asarray(bw) > 0).astype(np.uint8) * 255
    if clean.ndim != 2:
        raise ValueError("bw must be two-dimensional")
    h, w = clean.shape
    if not h or not w:
        return clean, 0, w
    edge = min(w, max(2, int(round(w * .24))))
    dilated = cv2.dilate(clean, np.ones((1, 3), np.uint8))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                              np.ones((max(9, int(round(.06*h))), 1), np.uint8))
    long_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                  np.ones((max(1, int(np.ceil(.30*h))), 1), np.uint8))
    candidates = (np.mean(dilated > 0, axis=0) >= .28) & np.any(long_lines, axis=0)
    # Reject broad foreground components; a border remains narrow after dilation.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((long_lines > 0).astype(np.uint8), 8)
    narrow = np.zeros(w, bool)
    for x, y, cw, ch, area in stats[1:]:
        if cw <= max(6, .12*w):
            narrow[x:x+cw] = True
    candidates &= narrow
    # Use original pixels to avoid trimming the extra dilation radius.
    line_cols = candidates & np.any(clean > 0, axis=0)
    left = np.flatnonzero(line_cols[:edge])
    right = np.flatnonzero(line_cols[w-edge:]) + w-edge
    x1 = int(left[-1]+1) if left.size else 0
    x2 = int(right[0]) if right.size else w
    if x2 > x1:
        clean[:, :x1] = 0
        clean[:, x2:] = 0
    else:
        x1, x2 = 0, w
    inner = clean[:, x1:x2]
    vertical = cv2.dilate(inner, np.ones((3, 1), np.uint8))
    row = np.mean(vertical > 0, axis=1) if inner.shape[1] else np.zeros(h)
    margin = max(2, int(round(w * .22)))
    rows = np.flatnonzero((row >= .65) & ((np.arange(h) < margin) | (np.arange(h) >= h-margin)))
    # Clear the actual line pixels as well as the dilated row candidates.
    clean[rows, :] = 0
    projection = np.count_nonzero(clean, axis=0)
    active = np.flatnonzero(projection >= 2)
    if not active.size:
        active = np.flatnonzero(projection)
    if active.size:
        x1, x2 = int(active[0]), int(active[-1]+1)
    else:
        x1, x2 = 0, w
    return clean, x1, x2


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    a = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    d = np.diff(a)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _bands(bw: np.ndarray, width: int) -> List[Tuple[int, int]]:
    h = bw.shape[0]
    if not h or not width:
        return []
    proj = np.count_nonzero(bw, axis=1).astype(float)
    kernel = max(1, int(round(width * .035)))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8)) if kernel > 1 else bw
    op = np.count_nonzero(opened, axis=1).astype(float)
    bands = []
    min_h = max(1, int(round(.32 * width)))
    def split(a: int, b: int) -> None:
        if b - a < max(2 * min_h, int(.8 * width)):
            bands.append((a, b)); return
        best = None
        for y in range(a + min_h, b - min_h + 1):
            rad = max(1, int(.16 * width))
            l, r = proj[max(a, y-rad):y], proj[y:min(b, y+rad)]
            if not l.size or not r.size:
                continue
            local = min(float(np.max(l)), float(np.max(r)))
            if local <= 0:
                continue
            depth = 1 - min(proj[y], op[y]) / local
            # Structure changes reinforce a shallower valley when the two
            # neighboring portions differ in width, balance, or centroid.
            def structure(part: np.ndarray) -> tuple[float, float, float]:
                yy, xx = np.nonzero(part)
                if not xx.size:
                    return (0., .5, .5)
                return ((int(xx.max()) - int(xx.min()) + 1) / width,
                        float(xx.mean()) / max(width-1, 1),
                        float(np.mean(xx < width / 2)))
            ls = structure(bw[max(a, y-rad):y])
            rs = structure(bw[y:min(b, y+rad)])
            change = max(abs(ls[0]-rs[0]), abs(ls[1]-rs[1]), abs(ls[2]-rs[2]))
            evidence = depth + .2 * change
            strong = depth >= .72 or (depth >= .52 and change >= .3 and b-a >= 1.35*width)
            if strong and (best is None or evidence > best[0]):
                best = (evidence, y)
        if best is None:
            bands.append((a, b))
        else:
            split(a, best[1]); split(best[1], b)
    for a, b in _runs(proj > 0):
        split(a, b)
    return bands


def _count_track_chars(part: np.ndarray, width: int) -> int:
    if not part.size:
        return 0
    kernel = np.ones((max(2, int(round(.065 * width))), 1), np.uint8)
    closed = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    minimum = max(3, int(round(.18 * width)))
    return sum(b-a >= minimum for a, b in _runs(np.any(closed, axis=1)))


def _features(mask: np.ndarray, width: int) -> SegmentFeatures:
    h = mask.shape[0]
    binary = mask > 0
    ys, xs = np.nonzero(binary)
    profile = binary.sum(axis=0).astype(float)
    scale = float(profile.max()) if profile.size else 0.0
    norm = profile / scale if scale else profile
    p32 = cv2.resize(norm.reshape(1, -1).astype(np.float32), (32, 1), interpolation=cv2.INTER_AREA).ravel().tolist() if width else [0.0] * 32
    if not xs.size:
        return SegmentFeatures(p32, 0, 0, .5, 0, 0, 0, 0, .5, 0, 0, h/max(width,1), 0, 0, 0)
    lo, hi = int(xs.min()), int(xs.max()) + 1
    # Locate the deepest central separation, then each side's strongest peak.
    start = lo + max(1, int((hi-lo)*.32)); end = lo + max(2, int((hi-lo)*.68))
    end = min(end, hi-1)
    cut = start + int(np.argmin(profile[start:end])) if start < end else (lo+hi)//2
    left, right = profile[lo:cut], profile[cut:hi]
    if left.size and right.size and left.max() and right.max():
        p1, p2 = int(np.argmax(left))+lo, int(np.argmax(right))+cut
        valley = float(profile[cut]); peak = min(float(profile[p1]), float(profile[p2]))
        depth = max(0., 1 - valley/peak)
        a1, a2 = left.sum(), right.sum()
        w1, w2 = np.count_nonzero(left), np.count_nonzero(right)
        area = min(a1,a2)/max(a1,a2); wr = min(w1,w2)/max(w1,w2)
        distance = abs(p2-p1)/max(width,1)
    else:
        depth = area = wr = distance = 0.
    def ink_height(part: np.ndarray) -> float:
        rows = np.flatnonzero(np.any(part, axis=1))
        return (rows[-1] - rows[0] + 1) / max(width, 1) if rows.size else 0.0
    return SegmentFeatures(p32, float(depth), float(depth * min(1., scale/max(h,1))),
        (cut-lo)/max(hi-lo,1), float(distance), float(area), float(wr),
        (hi-lo)/max(width,1), float(xs.mean()/max(width-1,1)), lo/max(width,1), hi/max(width,1),
        h/max(width,1), _count_track_chars(binary[:, :cut], width),
        _count_track_chars(binary[:, cut:], width), float(binary.mean()),
        float(profile[cut]/max(h, 1)), ink_height(binary[:, :cut]), ink_height(binary[:, cut:]))


def _make_segment(kind: str, y1: int, y2: int, x1: int, x2: int, bw: np.ndarray,
                  score: float, align: str, features: SegmentFeatures) -> ColumnSegment:
    region = bw[y1:y2, x1:x2]
    yy, xx = np.nonzero(region)
    if kind == "large" or not xx.size:
        return ColumnSegment(kind, y1, y2, x1, x2, alignment=align, score=score, features=features)
    ink1, ink2 = x1 + int(xx.min()), x1 + int(xx.max()) + 1
    if kind != "dual_track":
        return ColumnSegment(kind, y1, y2, ink1, ink2, alignment=align, score=score, features=features)
    split = x1 + int(round(features.ink_x1_rel * (x2-x1) + features.valley_pos * (features.ink_x2_rel-features.ink_x1_rel) * (x2-x1)))
    split = min(max(split, ink1+1), ink2-1)
    def box(a: int, b: int) -> Optional[Tuple[int,int,int,int]]:
        ys, xs = np.nonzero(bw[y1:y2, a:b])
        if not xs.size: return None
        return (a+int(xs.min()), y1+int(ys.min()), a+int(xs.max())+1, y1+int(ys.max())+1)
    return ColumnSegment(kind, y1, y2, ink1, ink2, box(split, x2), box(x1, split), split, align, score, features)


def segment_column(crop: np.ndarray, *, classifier: Optional[SegmentClassifier] = None, keep_bw: bool = False) -> ColumnSegmentAnalysis:
    gray = _gray(crop)
    h, w = gray.shape
    if not h or not w:
        return ColumnSegmentAnalysis(w, h, 0, w, float(w), [], np.zeros((h,w), np.uint8) if keep_bw else None)
    clean, x1, x2 = strip_column_borders(sauvola_binarize(gray))
    wi = x2-x1
    bands = _bands(clean[:, x1:x2], wi)
    bands = [(a, b) for a, b in bands
             if not (b-a < max(3, int(round(.14*wi)))
                     and np.count_nonzero(np.any(clean[a:b, x1:x2], axis=0)) < max(4, int(round(.24*wi))))
             and np.count_nonzero(clean[a:b, x1:x2]) >= max(6, int(round(.010*wi*wi)))]
    classifier = classifier or RuleSegmentClassifier()
    feats = [_features(clean[a:b, x1:x2], wi) for a,b in bands]
    results = [classifier.classify(f, adjacent_dual=False) for f in feats]
    # Revisit centered small glyphs with neighboring dual-track or centered rhythm.
    for i, f in enumerate(feats):
        adjacent = any(results[j][0] == "dual_track" for j in (i-1,i+1) if 0 <= j < len(results))
        if adjacent:
            results[i] = classifier.classify(f, adjacent_dual=True)
    if isinstance(classifier, RuleSegmentClassifier):
        # Reassess adjacent rows at the confirmed neighbor's seam.
        for i, f in enumerate(feats):
            if results[i][0] == 'dual_track' or f.height_ratio > .78:
                continue
            for j in (i-1, i+1):
                if not (0 <= j < len(results)) or results[j][0] != 'dual_track':
                    continue
                gap = max(bands[i][0]-bands[j][1], bands[j][0]-bands[i][1])
                if gap > .45*wi:
                    continue
                g = feats[j]
                cut = int(round((g.ink_x1_rel + g.valley_pos*(g.ink_x2_rel-g.ink_x1_rel))*wi))
                part = clean[bands[i][0]:bands[i][1], x1:x2] > 0
                if 0 < cut < wi and part[:, :cut].any() and part[:, cut:].any():
                    profile = part.sum(axis=0)
                    peak = min(profile[:cut].max(), profile[cut:].max())
                    depth = 1 - profile[cut]/peak if peak else 0
                    if depth >= .88 and profile[cut]/max(len(part), 1) <= .14:
                        results[i] = ('dual_track', .85, 'center')
                        break
        for i, f in enumerate(feats):
            if results[i][0] != 'large' or not (.4 < f.centroid_x < .6 and f.width_ratio < .58 and f.height_ratio <= .7): continue
            for j in (i-1,i+1):
                if 0 <= j < len(feats):
                    g = feats[j]
                    if .4 < g.centroid_x < .6 and g.width_ratio < .58 and g.height_ratio <= .7 and max(bands[i][0]-bands[j][1], bands[j][0]-bands[i][1]) <= .3*wi:
                        results[i] = ('single_track', .7, 'center'); break
    large_heights = [b-a for (a,b), r, f in zip(bands, results, feats)
                     if r[0] == 'large' and f.width_ratio >= .62
                     and .55*wi <= b-a <= 1.35*wi]
    large_char_h = float(np.median(large_heights)) if large_heights else float(wi)

    # Attach a one-sided continuation only when its ink fits a neighboring
    # dual row's actual track, on either side of that row.
    for i, ((a, b), res) in enumerate(zip(bands, results)):
        if res[0] != 'single_track':
            continue
        xs = np.flatnonzero(np.any(clean[a:b, x1:x2] > 0, axis=0))
        for j in (i-1, i+1):
            if not (0 <= j < len(results)) or results[j][0] != 'dual_track':
                continue
            if max(a-bands[j][1], bands[j][0]-b) > .45*wi:
                continue
            g = feats[j]
            cut = int(round((g.ink_x1_rel + g.valley_pos *
                             (g.ink_x2_rel-g.ink_x1_rel))*wi))
            row = clean[bands[j][0]:bands[j][1], x1:x2] > 0
            for track in (np.flatnonzero(np.any(row[:, :cut], axis=0)),
                          np.flatnonzero(np.any(row[:, cut:], axis=0)) + cut):
                if track.size and xs.size and track[0] <= xs[0] and xs[-1] <= track[-1]:
                    results[i] = ('dual_track', min(res[1], results[j][1]), 'center')
                    break
            if results[i][0] == 'dual_track':
                break

    # Merge dual rows before judging whether a clean seam belongs to one
    # large character. Recompute features over the complete merged group.
    groups = []
    for (a, b), res in zip(bands, results):
        if (groups and res[0] == 'dual_track' and groups[-1][2][0] == 'dual_track'
                and a-groups[-1][1] <= .45*wi):
            prev = groups[-1]
            groups[-1] = (prev[0], b, ('dual_track', min(res[1], prev[2][1]), 'center'))
        else:
            groups.append((a, b, res))
    reviewed = []
    for a, b, res in groups:
        if res[0] == 'dual_track':
            f = _features(clean[a:b, x1:x2], wi)
            if (max(f.left_sub_blobs, f.right_sub_blobs) <= 1
                    and (b-a >= .76*large_char_h or f.height_ratio > .68)):
                res = ('large', .75, 'center')
        est_chars = max(1, _count_track_chars(clean[a:b, x1:x2] > 0, wi))
        reviewed.append((a, b, res, 1, est_chars, b-a, (b-a) / est_chars))

    groups = []
    for a, b, res, band_count, est_chars, ink_h_sum, max_char_h in reviewed:
        gap_limit = .75 if res[0] == 'single_track' else .45
        if (groups and res[0] == groups[-1][2][0]
                and (res[0] != 'single_track' or res[2] == groups[-1][2][2])
                and a-groups[-1][1] <= gap_limit*wi):
            prev = groups[-1]
            groups[-1] = (prev[0], b, (res[0], min(res[1], prev[2][1]), res[2]),
                          prev[3] + band_count, prev[4] + est_chars,
                          prev[5] + ink_h_sum, max(prev[6], max_char_h))
        else:
            groups.append((a, b, res, band_count, est_chars, ink_h_sum, max_char_h))
    if isinstance(classifier, RuleSegmentClassifier):
        reviewed = []
        for a, b, res, band_count, est_chars, ink_h_sum, max_char_h in groups:
            mean_char_h = ink_h_sum / max(1, est_chars)
            if (res[0] == 'single_track' and
                    (mean_char_h / max(large_char_h, 1.0) > .78
                     or max_char_h / max(large_char_h, 1.0) > 1.15
                     or (res[2] != 'right' and band_count <= 1 and est_chars <= 1))):
                res = ('large', .75, 'center')
            if (reviewed and res[0] == 'large' and reviewed[-1][2][0] == 'large'
                    and a-reviewed[-1][1] <= .45*wi):
                prev = reviewed[-1]
                reviewed[-1] = (prev[0], b, ('large', min(res[1], prev[2][1]), 'center'))
            else:
                reviewed.append((a, b, res))
    else:
        reviewed = [(a, b, res) for a, b, res, _, _, _, _ in groups]
    segments = []
    for a,b,(kind,score,align) in reviewed:
        f = _features(clean[a:b, x1:x2], wi)
        segments.append(_make_segment(kind,a,b,x1,x2,clean,float(np.clip(score,0,1)),align,f))
    return ColumnSegmentAnalysis(w,h,x1,x2,large_char_h,segments,clean if keep_bw else None)


def render_column_segments_debug(crop: np.ndarray, analysis: ColumnSegmentAnalysis, out_path: str | Path | None = None) -> np.ndarray:
    gray = _gray(crop)
    h,w = gray.shape
    if not h or not w:
        return np.zeros((h, w*3, 3), np.uint8)
    src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop[:, :, :3].copy()
    bw = analysis.clean_bw
    if bw is None:
        bw, _, _ = strip_column_borders(sauvola_binarize(gray))
    binary = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    overlay = src.copy()
    colors = {'large': (0,180,0), 'dual_track': (0,0,255), 'single_track': (255,100,0)}
    for s in analysis.segments:
        color = colors.get(s.seg_type, (255,0,255))
        cv2.rectangle(overlay, (s.x1,s.y1), (max(s.x1,s.x2-1),max(s.y1,s.y2-1)), color, 1)
        if s.split_x is not None:
            cv2.line(overlay,(s.split_x,s.y1),(s.split_x,s.y2-1),color,1)
        cv2.putText(overlay, s.seg_type, (s.x1, max(8,s.y1+9)), cv2.FONT_HERSHEY_SIMPLEX, max(.25,min(.5,w/200)), color, 1, cv2.LINE_AA)
    result = np.concatenate((src,binary,overlay), axis=1)
    if out_path is not None:
        ok, encoded = cv2.imencode('.png', result)
        if not ok: raise ValueError('PNG encoding failed')
        Path(out_path).write_bytes(encoded.tobytes())
    return result
