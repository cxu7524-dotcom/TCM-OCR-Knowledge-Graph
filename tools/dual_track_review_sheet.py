"""Crop the printed plate area of a page (ink bounding box) to raise review zoom."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def plate_box(im, pad=8):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    colsum = (bw > 0).sum(axis=0)
    rowsum = (bw > 0).sum(axis=1)
    xt = max(6, int(0.012 * im.shape[0]))
    yt = max(6, int(0.012 * im.shape[1]))
    xs = np.flatnonzero(colsum > xt)
    ys = np.flatnonzero(rowsum > yt)
    if not len(xs) or not len(ys):
        return im
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(im.shape[1], x1 + pad), min(im.shape[0], y1 + pad)
    return im[y0:y1, x0:x1]


def review_sheet(paths, out, width=720, gap=6, head=22):
    crops = []
    for p in paths:
        im = imread_u(p)
        crops.append((p, plate_box(im)))
    hs = max(int(c.shape[0] * width / c.shape[1]) for _, c in crops)
    canvas = np.full((hs + head, len(crops) * (width + gap), 3), 245, np.uint8)
    for i, (p, c) in enumerate(crops):
        c = cv2.resize(c, (width, int(c.shape[0] * width / c.shape[1])), interpolation=cv2.INTER_AREA)
        x = i * (width + gap)
        canvas[head:head + c.shape[0], x:x + c.shape[1]] = c
        cv2.putText(canvas, f'{p.parent.name}_{p.stem}', (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    Path(out).write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    return out


if __name__ == '__main__':
    import sys
    BOOKS = ROOT / '古籍评测集/医家类'
    book, stems = sys.argv[1], sys.argv[2].split(',')
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    OUT = ROOT / 'output/双轨小字页检测/审阅'
    OUT.mkdir(parents=True, exist_ok=True)
    print(review_sheet([BOOKS / book / f'page_{s}.png' for s in stems], OUT / f'{book}_{stems[0]}.png', w))
