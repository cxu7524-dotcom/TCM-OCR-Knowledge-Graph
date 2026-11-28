"""Measure main glyph + two lanes for a given page/column and y window."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.vertical import page_layout, imread_unicode

BOOKS = ROOT / '古籍评测集/医家类'
OUT = ROOT / 'output/双轨小字页检测/坐标核对'
OUT.mkdir(parents=True, exist_ok=True)


def col_mask(im, box):
    x0, y0, x1, y1 = box
    sub = im[max(0, y0):y1, max(0, x0):x1]
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    return (m > 0).astype(np.float32), max(0, x0), max(0, y0)


def valley_rows(m, y_from, y_to):
    """Rows (local) whose ink splits into two x clusters with a blank middle."""
    H, W = m.shape
    out = []
    for y in range(y_from, min(y_to, H)):
        row = m[max(0, y - 3):y + 4].mean(axis=0)
        if row.sum() < 0.15:
            continue
        a, b = int(W * 0.2), int(W * 0.8)
        k = int(np.argmin(row[a:b])) + a
        left, right = row[:max(1, k - 1)], row[k + 2:]
        if not left.size or not right.size:
            continue
        if row[k] < 0.20 * max(left.max(), right.max(), 1e-6) and left.max() > 0.05 and right.max() > 0.05:
            out.append(y)
    return out


def measure(book, page, col, y_from, y_to, pad=0):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    im = imread_unicode(str(path))
    box = cols[col - 1]
    m, ox, oy = col_mask(im, box)
    rows = valley_rows(m, y_from - oy, y_to - oy)
    if not rows:
        return None
    a, b = min(rows), max(rows)
    # extend the dual run downward while a lane still has ink and no full-width glyph starts
    W = m.shape[1]
    probe_y = b + 1
    gap = 0
    while probe_y < min(m.shape[0], b + 40) and gap <= 2:
        row = m[probe_y]
        ink = np.flatnonzero(row > 0)
        wide = ink.size and (ink[-1] - ink[0]) > 0.72 * W
        if wide:
            break
        if row.sum() >= 2:
            b = probe_y
            gap = 0
        else:
            gap += 1
        probe_y += 1
    band = m[a:b + 1]
    prof = band.mean(axis=0)
    xs = np.flatnonzero(prof > 0.03)
    L, R = int(xs[0]), int(xs[-1]) + 1
    k = int(np.argmin(prof[L:R])) + L
    lx = np.flatnonzero(prof[L:k] > 0.03)
    rx = np.flatnonzero(prof[k + 1:R] > 0.03)
    if not lx.size or not rx.size:
        return None  # single track: one lane only
    def lane(xa, xb):
        band = m[a:b + 1, xa:xb]
        rows_l = np.flatnonzero(band.sum(axis=1) > 0)
        ya, yb = (a + int(rows_l[0]), a + int(rows_l[-1]) + 1) if rows_l.size else (a, b + 1)
        return [int(xa + ox) - pad, ya + oy - pad, int(xb + ox) + pad, yb + oy + pad]
    dual = {'y': [a + oy, b + oy],
            'right': lane(k + 1 + int(rx[0]), k + 1 + int(rx[-1]) + 1),
            'left': lane(L + int(lx[0]), L + int(lx[-1]) + 1)}
    # main glyph run: contiguous ink above the dual run (a 2-3 character drug name)
    top = a - 1
    gap = 0
    while top > max(0, a - 115) and gap <= 2:
        if m[top].sum() >= 2:
            gap = 0
        else:
            gap += 1
        top -= 1
    main = [box[0], top + 1 + oy, box[2], a - 1 + oy] if a - 1 - top > 8 else None
    return {'book': book, 'page': page, 'column': col, 'column_bbox': box,
            'main': main, 'right': dual['right'], 'left': dual['left'], 'dual_y': dual['y']}


def overlay(res, tag):
    path = BOOKS / res['book'] / f'page_{res["page"]}.png'
    im = imread_unicode(str(path))
    vis = im.copy()
    for box, color in ((res['main'], (0, 210, 255)), (res['right'], (0, 0, 255)), (res['left'], (0, 150, 0))):
        if box:
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 1)
    x0, y0 = res['left'][0], res['main'][1] - 30
    x1, y1 = res['right'][2], res['left'][3] + 40
    crop = vis[max(0, y0):y1, max(0, x0):x1]
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    out = OUT / f'overlay_{res["book"][:3]}_{res["page"]}_c{res["column"]}{tag}.png'
    out.write_bytes(cv2.imencode('.png', big)[1].tobytes())
    return out


if __name__ == '__main__':
    book, page, col = sys.argv[1], sys.argv[2], int(sys.argv[3])
    y0, y1 = int(sys.argv[4]), int(sys.argv[5])
    r = measure(book, page, col, y0, y1)
    if not r:
        print('no rows')
    else:
        print(json.dumps(r, ensure_ascii=False))
        print(overlay(r, sys.argv[6] if len(sys.argv) > 6 else ''))
