"""List all two-lane runs on a page, numbered, with overlay for manual selection."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.vertical import page_layout, imread_unicode
from tools.dual_track_measure import col_mask, valley_rows

BOOKS = ROOT / '古籍评测集/医家类'
OUT = ROOT / 'output/双轨小字页检测/坐标核对'


def candidates(book, page, min_run=18):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    im = imread_unicode(str(path))
    found = []
    for ci, box in enumerate(cols, 1):
        m, ox, oy = col_mask(im, box)
        rs = valley_rows(m, 0, m.shape[0])
        if not rs:
            continue
        groups, st, prev = [], rs[0], rs[0]
        for y in rs[1:]:
            if y - prev <= 6:
                prev = y
            else:
                groups.append((st, prev))
                st = prev = y
        groups.append((st, prev))
        W = m.shape[1]
        for a, b in groups:
            if b - a < min_run:
                continue
            band = m[a:b + 1]
            prof = band.mean(axis=0)
            xs = np.flatnonzero(prof > 0.03)
            if not xs.size:
                continue
            L, R = int(xs[0]), int(xs[-1]) + 1
            k = int(np.argmin(prof[L:R])) + L
            lx = np.flatnonzero(prof[L:k] > 0.03)
            rx = np.flatnonzero(prof[k + 1:R] > 0.03)
            if not lx.size or not rx.size or (lx.size < 3) or (rx.size < 3):
                continue

            def lane(xa, xb):
                bl = m[a:b + 1, xa:xb]
                rl = np.flatnonzero(bl.sum(axis=1) > 0)
                ya, yb = (a + int(rl[0]), a + int(rl[-1]) + 1) if rl.size else (a, b + 1)
                return [int(xa + ox), int(ya + oy), int(xb + ox), int(yb + oy)]
            main_top = max(0, a - 100)
            seg = m[main_top:a]
            rr = np.flatnonzero(seg.sum(axis=1) >= 2)
            main = [box[0], main_top + int(rr[0]) + oy, box[2], a - 1 + oy] if rr.size else None
            found.append({'col': ci, 'main': main, 'right': lane(k + 1 + int(rx[0]), k + 1 + int(rx[-1]) + 1),
                          'left': lane(L + int(lx[0]), L + int(lx[-1]) + 1), 'run': [a + oy, b + oy]})
    return im, found


def draw(book, page, min_run=18):
    im, found = candidates(book, page, min_run)
    vis = im.copy()
    for n, c in enumerate(found, 1):
        for box, color in ((c['main'], (0, 200, 255)), (c['right'], (0, 0, 255)), (c['left'], (0, 150, 0))):
            if box:
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv2.putText(vis, str(n), (c['left'][0], c['left'][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    out = OUT / f'all_{book[:3]}_{page}.png'
    out.write_bytes(cv2.imencode('.png', vis)[1].tobytes())
    (OUT / f'all_{book[:3]}_{page}.json').write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding='utf8')
    for n, c in enumerate(found, 1):
        print(n, 'col', c['col'], 'main', c['main'], 'R', c['right'], 'L', c['left'])
    return out


if __name__ == '__main__':
    print(draw(sys.argv[1], sys.argv[2]))
