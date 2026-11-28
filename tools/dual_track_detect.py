"""Page-level detector for dual-track small characters (大字下左右两轨小字).

Pipeline (no OCR, no reference text):
  1. ink mask from adaptive threshold, long vertical rules removed
  2. connected components -> page-level large-glyph height (median of tall comps)
  3. small comps: height 0.22-0.62 * large height
  4. group small comps into vertical runs inside a column strip; require two
     x-clusters separated by a white valley, each covering >=2 cells, with a
     large glyph directly above the run inside the same column
  5. emit region bbox + evidence, page accepted when >=1 region survives

Output: output/双轨小字页检测/detect.json, crops and contact sheets.
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
OUT.mkdir(parents=True, exist_ok=True)


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    Path(path).write_bytes(cv2.imencode('.png', img)[1].tobytes())


def ink_mask(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    # remove long vertical rules (column separators) and page frame
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45)))
    return cv2.subtract(bw, cv2.dilate(vert, np.ones((3, 3), np.uint8)))


def comps(mask, min_px=6):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_px or h < 5 or w < 3:
            continue
        out.append({'bbox': [int(x), int(y), int(x + w), int(y + h)], 'area': int(a), 'c': cent[i]})
    return out


def page_metrics(cs):
    hs = sorted((c['bbox'][3] - c['bbox'][1]) for c in cs if c['area'] > 25)
    if not hs:
        return None
    large = float(np.median(hs[int(len(hs) * 0.85):])) if len(hs) > 5 else float(hs[-1])
    return {'large_h': large}


def detect_page(path):
    im = imread_u(path)
    if im is None:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = ink_mask(gray)
    cs = comps(mask)
    m = page_metrics(cs)
    if not m:
        return None
    H = m['large_h']
    small = [c for c in cs if 0.22 * H <= (c['bbox'][3] - c['bbox'][1]) <= 0.62 * H and c['area'] >= 12]
    big = [c for c in cs if 0.85 * H <= (c['bbox'][3] - c['bbox'][1]) <= 1.25 * H and c['area'] >= 60]
    # cluster small comps by x into candidate lanes (within ~ column width)
    small.sort(key=lambda c: (c['c'][0], c['c'][1]))
    clusters = []
    for c in small:
        placed = False
        for cl in clusters:
            if abs(cl['x'] - c['c'][0]) <= 4.5:
                cl['items'].append(c)
                cl['x'] = float(np.mean([i['c'][0] for i in cl['items']]))
                placed = True
                break
        if not placed:
            clusters.append({'x': c['c'][0], 'items': [c]})
    clusters = [cl for cl in clusters if len(cl['items']) >= 2]
    # merge lanes into pairs: lanes whose x distance is 8-20 px and vertical spans overlap
    clusters.sort(key=lambda cl: cl['x'])
    pairs = []
    for i, a in enumerate(clusters):
        for b in clusters[i + 1:]:
            dx = b['x'] - a['x']
            if dx < 7:
                continue
            if dx > 21:
                break
            ya = [c['bbox'][1] for c in a['items']] + [c['bbox'][3] for c in a['items']]
            yb = [c['bbox'][1] for c in b['items']] + [c['bbox'][3] for c in b['items']]
            ov = min(max(ya), max(yb)) - max(min(ya), min(yb))
            if ov < 18:
                continue
            y0, y1 = max(min(ya), min(yb)), min(max(ya), max(yb))
            left, right = (a, b) if a['x'] < b['x'] else (b, a)
            lspan = sum(c['bbox'][3] - c['bbox'][1] for c in left['items'])
            rspan = sum(c['bbox'][3] - c['bbox'][1] for c in right['items'])
            cover = (lspan + rspan) / max(1, 2 * (y1 - y0))
            # main glyph directly above, horizontally overlapping the pair
            main = [g for g in big if g['bbox'][3] <= y0 + 6 and g['bbox'][1] >= y0 - 1.6 * H
                    and min(g['bbox'][2], right['x'] + 8) - max(g['bbox'][0], left['x'] - 8) > 6]
            area = [min(c['bbox'][0] for c in left['items'] + right['items']) - 3, y0 - 3,
                    max(c['bbox'][2] for c in left['items'] + right['items']) + 3, y1 + 3]
            pairs.append({'bbox': [int(v) for v in area], 'y': [int(y0), int(y1)],
                          'lanes_x': [round(left['x'], 1), round(right['x'], 1)],
                          'cells': [len(left['items']), len(right['items'])],
                          'ink_cover': round(float(cover), 3),
                          'main_above': bool(main),
                          'main_bbox': [int(v) for v in main[-1]['bbox']] if main else None,
                          'main_h': int(main[-1]['bbox'][3] - main[-1]['bbox'][1]) if main else None})
    # de-duplicate overlapping pairs, keep the tallest run of each x band
    pairs.sort(key=lambda p: (p['lanes_x'][0], -(p['y'][1] - p['y'][0])))
    kept = []
    for p in pairs:
        dup = False
        for q in kept:
            if abs(p['lanes_x'][0] - q['lanes_x'][0]) <= 6 and min(p['y'][1], q['y'][1]) - max(p['y'][0], q['y'][0]) > 0.5 * (p['y'][1] - p['y'][0]):
                dup = True
                break
        if not dup:
            kept.append(p)
    return {'image_size': [im.shape[1], im.shape[0]], 'large_h': round(H, 1),
            'comps': len(cs), 'small_comps': len(small), 'pairs': kept}


def main():
    pages = sorted((b, p) for b in BOOKS.iterdir() if b.is_dir() for p in b.glob('page_*.png'))
    res = {}
    for book, path in pages:
        r = detect_page(path)
        key = f'{book.name}/{path.stem}'
        r['source'] = str(path.relative_to(ROOT)).replace('\\', '/')
        res[key] = r
        print(key, 'large_h', r['large_h'], 'pairs', len(r['pairs']),
              'with_main', sum(p['main_above'] for p in r['pairs']), flush=True)
    (OUT / 'detect.json').write_text(json.dumps({'method': __doc__, 'pages': res}, ensure_ascii=False, indent=1), encoding='utf8')
    hits = {k: len([p for p in v['pairs'] if p['main_above']]) for k, v in res.items()}
    print('pages with main-anchored dual track:', sum(1 for v in hits.values() if v), '/', len(hits))


if __name__ == '__main__':
    main()
