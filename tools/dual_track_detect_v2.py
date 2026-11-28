"""Strict page-level detector for local dual-track small characters.

A region qualifies only when, inside one column strip:
  * two ink lanes of width 6-17 px separated by a near-blank valley >= 3 px
  * valley ink <= 18% of the weaker lane ink, over a vertical run >= 22 px
  * both lanes carry ink over >= 65% of the run height and each holds >= 2 glyph cells
  * small glyph height 8-16 px, i.e. 0.25-0.6 of the page's large-glyph height
  * a large glyph sits directly above the run inside the same column

No OCR, no reference text; output is a candidate list requiring visual check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / '古籍评测集/医家类'
OUT = ROOT / 'output/双轨小字页检测'


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def mask_of(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
    return cv2.subtract(bw, cv2.dilate(vert, np.ones((3, 3), np.uint8)))


def glyph_height(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    hs = [s[3] for s in stats[1:] if s[4] > 60 and s[2] >= 5]
    if not hs:
        return None
    hs.sort()
    return float(np.median(hs[int(len(hs) * 0.8):]))


def cells(mask, x0, x1, y0, y1, hmin):
    """Count vertical ink runs (glyph cells) in a lane band."""
    band = mask[y0:y1, x0:x1]
    rows = (band.sum(axis=1) > max(1, 0.02 * (x1 - x0) * 255)).astype(np.uint8)
    runs, cur = [], 0
    for v in rows:
        if v:
            cur += 1
        elif cur:
            if cur >= hmin:
                runs.append(cur)
            cur = 0
    if cur >= hmin:
        runs.append(cur)
    return len(runs), runs


def scan_region(mask, x0, x1, y0, y1, large_h):
    """Search a column strip for one dual-track run; return best or None."""
    hs, ws = mask.shape
    x0, x1 = max(0, x0), min(ws, x1)
    y0, y1 = max(0, y0), min(hs, y1)
    if x1 - x0 < 18 or y1 - y0 < 24:
        return None
    strip = mask[y0:y1, x0:x1]
    H = y1 - y0
    best = None
    # candidate valleys: local minima of column ink profile, 25-75% of the strip
    prof = strip.sum(axis=0).astype(float) / 255.0
    for xv in range(int((x1 - x0) * 0.30), int((x1 - x0) * 0.70) + 1):
        for lw in (9, 11, 13):
            for rw in (9, 11, 13):
                lx0, lx1 = xv - lw - 1, xv - 1
                rx0, rx1 = xv + 1, xv + rw + 1
                if lx0 < 0 or rx1 > x1 - x0:
                    continue
                # sliding vertical window of 24 px, extend while valid
                ink = strip / 255.0
                lcol = ink[:, lx0:lx1].mean(axis=1)
                rcol = ink[:, rx0:rx1].mean(axis=1)
                vcol = ink[:, max(0, xv - 1):xv + 2].mean(axis=1)
                valid = (lcol > 0.06) & (rcol > 0.06) & (vcol < 0.02)
                run_start, run_len, cur = 0, 0, 0
                best_run = (0, 0)
                for i, v in enumerate(valid):
                    if v:
                        if cur == 0:
                            run_start = i
                        cur += 1
                        if cur > run_len:
                            run_len = cur
                            best_run = (run_start, run_start + cur)
                    else:
                        cur = 0
                a, b = best_run
                if b - a < 22:
                    continue
                span = b - a
                lfill = float(lcol[a:b].astype(bool).mean())
                rfill = float(rcol[a:b].astype(bool).mean())
                if min(lfill, rfill) < 0.65:
                    continue
                lvin = float(ink[a:b, lx0:lx1].sum())
                rvin = float(ink[a:b, rx0:rx1].sum())
                vvin = float(ink[a:b, max(0, xv - 1):xv + 2].sum())
                if vvin > 0.18 * min(lvin, rvin):
                    continue
                nh_l, hl = cells(mask, x0 + lx0, x0 + lx1, y0 + a, y0 + b, 5)
                nh_r, hr = cells(mask, x0 + rx0, x0 + rx1, y0 + a, y0 + b, 5)
                if nh_l < 2 or nh_r < 2:
                    continue
                small_h = float(np.median(hl + hr))
                if not (0.25 * large_h <= small_h <= 0.62 * large_h):
                    continue
                score = span * min(lfill, rfill) * (1 - vvin / max(1e-6, min(lvin, rvin)))
                if best is None or score > best['score']:
                    best = {'score': round(float(score), 2),
                            'bbox': [int(x0 + lx0), int(y0 + a), int(x0 + rx1), int(y0 + b)],
                            'valley_x': int(x0 + xv), 'lane_w': [lw, rw], 'run_h': int(span),
                            'lane_fill': [round(lfill, 3), round(rfill, 3)],
                            'valley_ratio': round(vvin / max(1e-6, min(lvin, rvin)), 3),
                            'cells': [nh_l, nh_r], 'small_h': round(small_h, 1)}
    return best


def columns_of(mask, large_h):
    """Column strips from the ink projection, split at 界行 gaps."""
    prof = (mask > 0).sum(axis=0).astype(float) / max(1, mask.shape[0])
    on = prof > 0.03
    strips, cur = [], None
    for x, v in enumerate(on):
        if v and cur is None:
            cur = x
        elif not v and cur is not None:
            if x - cur >= 16:
                strips.append((cur, x))
            cur = None
    if cur is not None and mask.shape[1] - cur >= 16:
        strips.append((cur, mask.shape[1]))
    return strips


def detect_page(path):
    im = imread_u(path)
    if im is None:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = mask_of(gray)
    H = glyph_height(mask)
    if not H:
        return None
    strips = columns_of(mask, H)
    regions = []
    for (x0, x1) in strips:
        # column content vertical extent
        sub = mask[:, x0:x1]
        rows = np.flatnonzero(sub.sum(axis=1) > 0)
        if not len(rows):
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        r = scan_region(mask, x0, x1, y0, y1, H)
        if r:
            # require a large glyph right above the run
            a, b = r['bbox'][1], r['bbox'][3]
            band = mask[max(0, a - int(1.3 * H)):a, x0:x1]
            n, _, st, _ = cv2.connectedComponentsWithStats(band, 8)
            above = any(0.8 * H <= s[3] <= 1.3 * H and s[4] > 60 for s in st[1:])
            r['main_above'] = bool(above)
            r['column'] = [int(x0), int(x1)]
            regions.append(r)
    return {'image_size': [im.shape[1], im.shape[0]], 'large_h': round(H, 1),
            'columns': len(strips), 'regions': regions}


def main():
    pages = sorted((b, p) for b in BOOKS.iterdir() if b.is_dir() for p in b.glob('page_*.png'))
    res = {}
    for book, path in pages:
        r = detect_page(path)
        key = f'{book.name}/{path.stem}'
        r['source'] = str(path.relative_to(ROOT)).replace('\\', '/')
        res[key] = r
        strong = [x for x in r['regions'] if x['main_above']]
        print(f"{key:28s} large_h={r['large_h']:5.1f} regions={len(r['regions']):2d} anchored={len(strong):2d}", flush=True)
    (OUT / 'detect_v2.json').write_text(json.dumps({'method': __doc__, 'pages': res}, ensure_ascii=False, indent=1), encoding='utf8')
    hit = [k for k, v in res.items() if any(x['main_above'] for x in v['regions'])]
    print('pages with anchored dual-track:', len(hit))
    for k in hit:
        print('  ', k)


if __name__ == '__main__':
    main()
