"""Render one column of a page at high zoom with a pixel ruler (original coordinates)."""
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


def render(book, page, col_idx, scale=6, y0=None, y1=None):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    x0, y0c, x1, y1c = cols[col_idx - 1]
    im = imread_unicode(str(path))
    y0 = y0c if y0 is None else y0
    y1 = y1c if y1 is None else y1
    crop = im[y0:y1, max(0, x0 - 6):min(im.shape[1], x1 + 6)]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = 60
    canvas = np.full((big.shape[0] + 30, big.shape[1] + pad, 3), 250, np.uint8)
    canvas[15:15 + big.shape[0], pad:pad + big.shape[1]] = big
    for y in range(y0, y1, 10):
        yy = 15 + (y - y0) * scale
        if yy >= canvas.shape[0]:
            break
        major = y % 50 == 0
        cv2.line(canvas, (pad - 8 if major else pad - 4, yy), (pad, yy), (0, 0, 220) if major else (150, 150, 150), 1)
        if major:
            cv2.putText(canvas, str(y), (2, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
    for x in range(x0 - 6, x1 + 6, 10):
        xx = pad + (x - (x0 - 6)) * scale
        if 0 <= xx < canvas.shape[1]:
            cv2.line(canvas, (xx, 15), (xx, 15 + 6), (0, 150, 0), 1)
    out = OUT / f'{book[:3]}_{page}_col{col_idx}.png'
    out.write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    return out, (x0, y0c, x1, y1c)


if __name__ == '__main__':
    book, page, col = sys.argv[1], sys.argv[2], int(sys.argv[3])
    scale = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    y0 = int(sys.argv[5]) if len(sys.argv) > 5 else None
    y1 = int(sys.argv[6]) if len(sys.argv) > 6 else None
    out, box = render(book, page, col, scale, y0, y1)
    print(out, 'column bbox', box)
