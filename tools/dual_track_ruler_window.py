"""Zoomed coordinate ruler for a y-window of one column (for manual coordinate reading)."""
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


def ruler(book, page, col, y0, y1, scale=8, tag=''):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    x0, _, x1, _ = cols[col - 1]
    im = imread_unicode(str(path))
    X0, X1 = max(0, x0 - 8), min(im.shape[1], x1 + 8)
    crop = im[y0:y1, X0:X1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    margin, top = 96, 26
    canvas = np.full((big.shape[0] + top, big.shape[1] + margin, 3), 252, np.uint8)
    canvas[top:, margin:] = big
    for y in range((y0 // 10) * 10, y1, 10):
        yy = top + (y - y0) * scale
        if yy < top or yy >= canvas.shape[0]:
            continue
        major = y % 20 == 0
        cv2.line(canvas, (margin - (14 if major else 6), yy), (margin, yy), (0, 0, 200) if major else (170, 170, 170), 1)
        if major:
            cv2.line(canvas, (margin, yy), (canvas.shape[1], yy), (225, 225, 225), 1)
            cv2.putText(canvas, str(y), (4, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
    for x in range(X0, X1, 10):
        xx = margin + (x - X0) * scale
        cv2.line(canvas, (xx, top - 5), (xx, top), (0, 140, 0), 1)
        if x % 20 == 0:
            cv2.putText(canvas, str(x), (xx - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 0), 1)
    cv2.putText(canvas, f'{book[:3]}-p{page}-col{col}{tag}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    out = OUT / f'ruler_{book[:3]}_{page}_col{col}{tag}.png'
    out.write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    return out


if __name__ == '__main__':
    b, p, c = sys.argv[1], sys.argv[2], int(sys.argv[3])
    y0, y1 = int(sys.argv[4]), int(sys.argv[5])
    sc = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    print(ruler(b, p, c, y0, y1, sc))
