"""Dual-track scanner v3: calibrated on the 20 positive / 22 negative annotations.

Region rule (all must hold):
  * two ink lanes, each 10-30 px wide, valley between them 1-4 px
  * valley column ink <= 12% of the weaker lane peak, over a run >= 12 px
  * both lanes inked over that run (row fill >= 5% each, valley < 2%)
  * small glyph cells in each lane: 7-20 px tall, >= 2 cells in the longer lane
  * a main glyph of 0.8-1.4x page glyph height sits directly above, overlapping x
  * two lanes together 28-48 px wide (one column of large glyphs is wider or
    shows no valley at all)

Modes: `eval` scores the annotated samples, `scan` walks every page.
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


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def mask_of(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)


def glyph_height(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    hs = [s[3] for s in st[1:] if s[4] > 80 and s[2] >= 6]
    if not hs:
        return None
    hs.sort()
    return float(np.median(hs[int(len(hs) * 0.85):]))


def longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def cells(band, min_h=6):
    r = band.mean(axis=1) > 0.05
    out, cur = [], 0
    for v in r:
        cur = cur + 1 if v else 0
        if not v and cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return [c for c in out if c >= min_h]


def test_box(mask, box, large_h, verbose=False):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    sub = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    if sub.size < 200:
        return None
    W = sub.shape[1]
    colp = sub.mean(axis=0)
    best = None
    for lw in range(10, 31, 2):
        for rw in range(10, 31, 2):
            if not (28 <= lw + rw + 2 <= 52):
                continue
            for xv in range(lw + 1, W - rw - 1):
                L = sub[:, xv - lw - 1:xv - 1]
                R = sub[:, xv + 1:xv + rw + 1]
                V = sub[:, max(0, xv - 1):xv + 2]
                lp, rp = colp[xv - lw - 1:xv - 1].max(), colp[xv + 1:xv + rw + 1].max()
                if min(lp, rp) < 0.12:
                    continue
                vratio = colp[xv] / min(lp, rp)
                if vratio > 0.12:
                    continue
                flags = (L.mean(axis=1) > 0.05) & (R.mean(axis=1) > 0.05) & (V.mean(axis=1) < 0.02)
                run = longest_run(flags)
                if run < 12:
                    continue
                # locate the longest run extent
                bestf = cur = 0
                s = e = 0
                for i, f in enumerate(flags):
                    if f:
                        cur += 1
                        if cur > bestf:
                            bestf, e, s = cur, i + 1, i + 1 - cur
                    else:
                        cur = 0
                cl = cells(L[s:e])
                cr = cells(R[s:e])
                if max(len(cl), len(cr)) < 2:
                    continue
                gh = float(np.median(cl + cr)) if (cl or cr) else 0
                if not (7 <= gh <= 20):
                    continue
                if not (0.22 * large_h <= gh <= 0.62 * large_h):
                    continue
                score = run * min(lp, rp) / max(1e-6, vratio + 0.02)
                cand = {'bbox': [x0 + xv - lw - 1, y0 + s, x0 + xv + rw + 1, y0 + e],
                        'valley_x': x0 + xv, 'lane_w': [lw, rw], 'run': run, 'vratio': round(float(vratio), 3),
                        'cells': [len(cl), len(cr)], 'gh': round(gh, 1), 'score': round(float(score), 1)}
                if best is None or cand['score'] > best['score']:
                    best = cand
    return best


def big_glyph_above(mask, region, large_h, col_span):
    x0, y0, x1, y1 = region['bbox']
    top = max(0, y0 - int(1.5 * large_h))
    band = (mask[top:y0, max(0, col_span[0]):col_span[1]] > 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(band, 8)
    for s in st[1:]:
        if 0.78 * large_h <= s[3] <= 1.45 * large_h and s[4] > 80:
            return [int(max(0, col_span[0]) + s[0]), int(top + s[1]),
                    int(max(0, col_span[0]) + s[0] + s[2]), int(top + s[1] + s[3])]
    return None


def strips(mask):
    prof = (mask > 0).sum(axis=0)
    on = prof > 0.02 * mask.shape[0]
    out, cur = [], None
    for x, v in enumerate(on):
        if v and cur is None:
            cur = x
        elif not v and cur is not None:
            if x - cur >= 18:
                out.append((cur, x))
            cur = None
    if cur is not None and mask.shape[1] - cur >= 18:
        out.append((cur, mask.shape[1]))
    return out


def scan_page(path, large_h_override=None):
    im = rd(path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = mask_of(gray)
    H = large_h_override or glyph_height(mask)
    if not H:
        return None
    regions = []
    for (a, b) in strips(mask):
        rows = np.flatnonzero((mask[:, a:b] > 0).sum(axis=1) > 0)
        if not len(rows):
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        r = test_box(mask, [a, y0, b, y1], H)
        if not r:
            continue
        m = big_glyph_above(mask, r, H, (a, b))
        if not m:
            continue
        r['main_bbox'] = m
        r['column'] = [int(a), int(b)]
        regions.append(r)
    return {'image_size': [im.shape[1], im.shape[0]], 'large_h': round(H, 1),
            'strips': len(strips(mask)), 'regions': regions}


def evaluate():
    ann = json.loads((ROOT / 'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'))
    dev = json.loads((ROOT / 'output/small_columns_step1_dev/annotation.json').read_text(encoding='utf8'))
    truth = []
    for s in ann['samples']:
        if s['label'] == 'positive':
            bx = [min(c['bbox'][0] for c in s['columns']), min(c['bbox'][1] for c in s['columns']),
                  max(c['bbox'][2] for c in s['columns']), max(c['bbox'][3] for c in s['columns'])]
            truth.append((s['id'], str(s['page']), bx, 1))
        elif s.get('columns') and len(s['columns']) == 2:
            bx = [min(c['bbox'][0] for c in s['columns']), min(c['bbox'][1] for c in s['columns']),
                  max(c['bbox'][2] for c in s['columns']), max(c['bbox'][3] for c in s['columns'])]
            truth.append((s['id'], str(s['page']), bx, 0))
    s = dev['sample']
    bx = [min(c['bbox'][0] for c in s['columns']), min(c['bbox'][1] for c in s['columns']),
          max(c['bbox'][2] for c in s['columns']), max(c['bbox'][3] for c in s['columns'])]
    truth.append((s['id'], '35', bx, 1))
    cache = {}
    tp = fp = fn = tn = 0
    for sid, page, bx, label in truth:
        if page not in cache:
            cache[page] = scan_page(BOOKS / '思濟堂方書' / f'page_{page}.png')
        regs = cache[page]['regions']
        hit = any(min(r['bbox'][2], bx[2]) - max(r['bbox'][0], bx[0]) > 4 and
                  min(r['bbox'][3], bx[3]) - max(r['bbox'][1], bx[1]) > 4 for r in regs)
        ok = (hit and label == 1) or ((not hit) and label == 0)
        tp += hit and label == 1
        fn += (not hit) and label == 1
        fp += hit and label == 0
        tn += (not hit) and label == 0
        print(('OK ' if ok else 'ERR'), sid, 'label', label, 'hit', hit, 'regions_on_page', len(regs))
    print(f'positives {tp}/{tp + fn} detected, negatives wrongly hit {fp}/{fp + tn}')


def scan_all():
    pages = sorted((b, p) for b in BOOKS.iterdir() if b.is_dir() for p in b.glob('page_*.png'))
    res = {}
    for book, path in pages:
        r = scan_page(path)
        key = f'{book.name}/{path.stem}'
        r['source'] = str(path.relative_to(ROOT)).replace('\\', '/')
        res[key] = r
        print(f"{key:30s} regions={len(r['regions'])}", flush=True)
    (OUT / 'detect_v3.json').write_text(json.dumps({'pages': res}, ensure_ascii=False, indent=1), encoding='utf8')
    hits = [k for k, v in res.items() if v['regions']]
    print('pages with regions', len(hits))
    for k in hits:
        print('  ', k, len(res[k]['regions']))


if __name__ == '__main__':
    {'eval': evaluate, 'scan': scan_all}[sys.argv[1]]()
