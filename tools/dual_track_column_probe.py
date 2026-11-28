"""Probe candidate dual-track bands inside each column and draw numbered overlays.

A band is a vertical run of ink rows. A dual-track candidate is a tall band (large
glyph) followed by a band whose ink splits into two x-clusters separated by a valley.
"""
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


def bands(mask, min_gap=2, min_ink=3):
    rows = (mask > 0).sum(axis=1)
    on = rows >= min_ink
    out, cur = [], None
    gap = 0
    for y, v in enumerate(on):
        if v:
            if cur is None:
                cur = y
            gap = 0
        elif cur is not None:
            gap += 1
            if gap >= min_gap:
                out.append((cur, y - gap + 1))
                cur = None
    if cur is not None:
        out.append((cur, len(on) - 1))
    return [(a, b) for a, b in out if b - a >= 4]


def split(mask, y0, y1):
    sub = (mask[y0:y1] > 0)
    prof = sub.mean(axis=0)
    xs = np.flatnonzero(prof > 0.02)
    if not len(xs):
        return None
    a, b = int(xs[0]), int(xs[-1]) + 1
    if b - a < 12:
        return None
    mid = prof[a:b]
    k = int(np.argmin(mid[int(len(mid) * 0.3):int(len(mid) * 0.7)]) + len(mid) * 0.3)
    xv = a + k
    left = prof[a:xv]
    right = prof[xv + 1:b]
    if not len(left) or not len(right):
        return None
    lx = np.flatnonzero(left > 0.02)
    rx = np.flatnonzero(right > 0.02)
    if not len(lx) or not len(rx):
        return None
    lbbox = [int(a + lx[0]), y0, int(a + lx[-1]) + 1, y1]
    rbbox = [int(xv + 1 + rx[0]), y0, int(xv + 1 + rx[-1]) + 1, y1]
    gap = lbbox[2] - rbbox[0]
    return {'valley_x': xv, 'left': lbbox, 'right': rbbox, 'gap': int(gap),
            'lw': lbbox[2] - lbbox[0], 'rw': rbbox[2] - rbbox[0]}


def probe(book, page, draw=True):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    im = imread_unicode(str(path))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    cands = []
    for ci, (x0, y0, x1, y1) in enumerate(cols, 1):
        sub = mask[max(0, y0):y1, max(0, x0):x1]
        bs = bands(sub)
        for i, (a, b) in enumerate(bs[:-1]):
            h = b - a
            if not (20 <= h <= 44):
                continue
            na, nb = bs[i + 1]
            nh = nb - na
            if not (10 <= nh <= 90):
                continue
            sp = split(sub, na, nb)
            if not sp or sp['gap'] < 2 or sp['lw'] < 6 or sp['rw'] < 6:
                continue
            main = [max(0, x0), max(0, y0) + a, x1, max(0, y0) + b]
            cands.append({'col': ci, 'main': main, 'right': [sp['right'][0] + max(0, x0), sp['right'][1] + max(0, y0),
                                                             sp['right'][2] + max(0, x0), sp['right'][3] + max(0, y0)],
                          'left': [sp['left'][0] + max(0, x0), sp['left'][1] + max(0, y0),
                                   sp['left'][2] + max(0, x0), sp['left'][3] + max(0, y0)],
                          'run_h': nh, 'lw': sp['lw'], 'rw': sp['rw'], 'gap': sp['gap']})
    if draw and cands:
        vis = im.copy()
        for n, c in enumerate(cands, 1):
            for box, color in ((c['main'], (0, 200, 255)), (c['right'], (0, 0, 255)), (c['left'], (0, 160, 0))):
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.putText(vis, str(n), (c['right'][0] + 1, c['main'][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        out = OUT / f'probe_{book[:3]}_{page}.png'
        out.write_bytes(cv2.imencode('.png', vis)[1].tobytes())
        print('overlay', out)
    (OUT / f'probe_{book[:3]}_{page}.json').write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding='utf8')
    for n, c in enumerate(cands, 1):
        print(n, 'col', c['col'], 'main', c['main'], 'R', c['right'], 'L', c['left'], 'run', c['run_h'], 'w', c['lw'], c['rw'], 'gap', c['gap'])
    return cands


if __name__ == '__main__':
    probe(sys.argv[1], sys.argv[2])
