# -*- coding: utf-8 -*-
"""stage8 判据标定：两侧轨的「字高中位数」能否分开真双轨与单轨误检。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocr.vertical import imread_unicode  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


_cache = {}


def gray_of(rel):
    rel = str(rel)
    if rel not in _cache:
        im = imread_unicode(str(ROOT / rel))
        _cache[rel] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    return _cache[rel]


def glyph_heights(rel, box):
    g = gray_of(rel)
    x1, y1, x2, y2 = [int(v) for v in box]
    w = x2 - x1
    if w < 4 or y2 - y1 < 6:
        return []
    sub = g[max(0, y1):y2, max(0, x1):x2]
    bg = cv2.GaussianBlur(sub, (31, 31), 0).astype(np.int16)
    delta = bg - sub.astype(np.int16)
    bw = (delta >= 16).astype(np.uint8)
    rows = bw.sum(axis=1)
    act = (rows >= max(2, int(w * 0.12))).astype(np.uint8)
    act = cv2.morphologyEx(act[:, None], cv2.MORPH_CLOSE,
                           np.ones((3, 1), np.uint8)).ravel().astype(np.int16)
    e = np.diff(np.r_[np.int16(0), act, np.int16(0)])
    st = np.flatnonzero(e == 1)
    en = np.flatnonzero(e == -1)
    return [int(b - a) for a, b in zip(st, en) if b - a >= 5]


def main() -> int:
    ann = load(S2 / 'annotations.json')
    img_of = {g['group_id']: g['image_path'] for g in ann['groups']}

    print('=== A. 已标注 38 组：两侧字高中位数比 ===')
    rs = []
    for g in ann['groups']:
        bl = glyph_heights(g['image_path'], g['left_bbox'])
        br = glyph_heights(g['image_path'], g['right_bbox'])
        if not bl or not br:
            print(f"{g['region']:10s} 无字带 L={len(bl)} R={len(br)}")
            continue
        hl, hr = float(np.median(bl)), float(np.median(br))
        r = max(hl, hr) / min(hl, hr)
        rs.append(r)
        print(f"{g['region']:10s} L={hl:5.1f}({len(bl):2d}) R={hr:5.1f}({len(br):2d}) "
              f"ratio={r:.2f}  Lw={g['left_bbox'][2]-g['left_bbox'][0]:2d} "
              f"Rw={g['right_bbox'][2]-g['right_bbox'][0]:2d}")
    if rs:
        print(f'GT ratio: min={min(rs):.2f} p50={np.median(rs):.2f} '
              f'p90={np.percentile(rs, 90):.2f} max={max(rs):.2f}  n={len(rs)}')

    print()
    print('=== B. 修正版预测中落在负例区的组：同量 ===')
    neg = load(S2 / 'negative_regions.json')
    pred = load(S8 / 'predictions' / 'path_fix.json')
    for p in pred['predictions']:
        gb = p.get('group_bbox')
        if not gb:
            continue
        for r in neg['regions']:
            if r['page_id'] != p['page_id']:
                continue
            ix1, iy1 = max(gb[0], r['bbox'][0]), max(gb[1], r['bbox'][1])
            ix2, iy2 = min(gb[2], r['bbox'][2]), min(gb[3], r['bbox'][3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            cov = (ix2 - ix1) * (iy2 - iy1) / area if area else 0
            bl = glyph_heights(p['page_id'] and next(
                x['image_path'] for x in ann['groups'] if x['page_id'] == p['page_id']),
                p['left_bbox'])
            br = glyph_heights(next(
                x['image_path'] for x in ann['groups'] if x['page_id'] == p['page_id']),
                p['right_bbox'])
            hl = float(np.median(bl)) if bl else 0.0
            hr = float(np.median(br)) if br else 0.0
            r2 = max(hl, hr) / min(hl, hr) if hl and hr else float('nan')
            print(f"{r['negative_region_id']:18s} {p['prediction_id'].split('#')[-1]} "
                  f"cov={cov:.2f} box={gb} L={hl:.1f}({len(bl)}) R={hr:.1f}({len(br)}) "
                  f"ratio={r2:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
