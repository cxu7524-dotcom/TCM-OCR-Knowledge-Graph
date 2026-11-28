"""Column-strip review sheets: split a page at its 界行 and enlarge every column.

Each strip is a single text column at ~5x, so 小字 tracks under a 大字 are legible.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def rule_x(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    ink = (bw > 0).mean(axis=0)
    thr = 0.80
    xs = np.flatnonzero(ink > thr)
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [int(np.mean(g)) for g in groups]


def columns(path):
    im = rd(path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    xs = rule_x(gray)
    if len(xs) < 2:
        return []
    out = []
    for a, b in zip(xs, xs[1:]):
        if b - a < 14:
            continue
        sub = (bw[:, a + 2:b - 1] > 0).sum(axis=1)
        rows = np.flatnonzero(sub > 0)
        if not len(rows) or rows[-1] - rows[0] < 120:
            continue
        out.append([a + 2, int(rows[0]), b - 1, int(rows[-1])])
    return out


def sheet(path, out, col_h=760, gap=8):
    im = rd(path)
    cols = columns(path)
    if not cols:
        return None
    tiles = []
    for x0, y0, x1, y1 in cols:
        c = im[y0:y1, x0:x1]
        s = col_h / c.shape[0]
        tiles.append(cv2.resize(c, (max(1, int(c.shape[1] * s)), col_h), interpolation=cv2.INTER_CUBIC))
    W = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((col_h + 22, W, 3), 250, np.uint8)
    x = 0
    for i, t in enumerate(tiles):
        canvas[22:22 + t.shape[0], x:x + t.shape[1]] = t
        cv2.putText(canvas, str(i + 1), (x + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
        x += t.shape[1] + gap
    Path(out).write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    return out


if __name__ == '__main__':
    BOOKS = ROOT / '古籍评测集/医家类'
    OUT = ROOT / 'output/双轨小字页检测/列条审阅'
    OUT.mkdir(parents=True, exist_ok=True)
    book = sys.argv[1]
    for stem in sys.argv[2].split(','):
        print(sheet(BOOKS / book / f'page_{stem}.png', OUT / f'{book}_{stem}_cols.png'))
