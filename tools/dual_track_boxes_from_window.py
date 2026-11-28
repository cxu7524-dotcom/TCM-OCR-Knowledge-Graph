"""Tight bboxes for a main glyph run and its two lanes, given column + y ranges."""
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


def tight(mask, x0, x1, y0, y1):
    sub = (mask[y0:y1, x0:x1] > 0)
    rows = np.flatnonzero(sub.sum(axis=1) > 0)
    cols = np.flatnonzero(sub.sum(axis=0) > 0)
    if not rows.size or not cols.size:
        return None
    return [int(x0 + cols[0]), int(y0 + rows[0]), int(x0 + cols[-1]) + 1, int(y0 + rows[-1]) + 1]


def boxes(book, page, col, main_y, lane_y, split=None):
    path = BOOKS / book / f'page_{page}.png'
    cols = [c['bbox'] for c in page_layout(str(path))['cols']]
    x0, _, x1, _ = cols[col - 1]
    im = imread_unicode(str(path))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    mid = split if split else (x0 + x1) // 2
    res = {'book': book, 'page': page, 'column': col, 'column_bbox': [x0, cols[col - 1][1], x1, cols[col - 1][3]],
           'main': tight(mask, x0, x1, main_y[0], main_y[1]),
           'left': tight(mask, x0, mid - 1, lane_y[0], lane_y[1]),
           'right': tight(mask, mid + 1, x1, lane_y[0], lane_y[1]), 'split_x': mid}
    return res, im


def draw(res, im, tag=''):
    vis = im.copy()
    for key, color in (('main', (0, 210, 255)), ('right', (0, 0, 255)), ('left', (0, 150, 0))):
        b = res[key]
        if b:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), color, 1)
    ys = [b[1] for b in (res['main'], res['left'], res['right']) if b]
    ye = [b[3] for b in (res['main'], res['left'], res['right']) if b]
    xs = [b[0] for b in (res['main'], res['left'], res['right']) if b]
    xe = [b[2] for b in (res['main'], res['left'], res['right']) if b]
    crop = vis[max(0, min(ys) - 20):min(vis.shape[0], max(ye) + 20), max(0, min(xs) - 20):min(vis.shape[1], max(xe) + 20)]
    big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    out = OUT / f'box_{res["book"][:3]}_{res["page"]}_c{res["column"]}{tag}.png'
    out.write_bytes(cv2.imencode('.png', big)[1].tobytes())
    return out


if __name__ == '__main__':
    book, page, col = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main_y = [int(sys.argv[4]), int(sys.argv[5])]
    lane_y = [int(sys.argv[6]), int(sys.argv[7])]
    split = int(sys.argv[8]) if len(sys.argv) > 8 else None
    r, im = boxes(book, page, col, main_y, lane_y, split)
    print(json.dumps(r, ensure_ascii=False))
    print(draw(r, im))
