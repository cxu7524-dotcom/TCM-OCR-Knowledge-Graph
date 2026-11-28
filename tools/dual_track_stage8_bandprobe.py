# -*- coding: utf-8 -*-
"""原型：分带列槽（band slots），只用于验证 p12 下半栏能否恢复。"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dual_track_fix.slots import ink_projection, estimate_pitch
from ocr.small_columns import propose_in_column, estimate_page_typography
from ocr.vertical import imread_unicode


def fit_lattice(proj, x1, x2, lo, hi, step=1):
    """在 [x1,x2) 内拟合 (pitch, phase)：切点处墨量均值最小。"""
    best = None
    for d in range(lo, hi + 1, step):
        for a in range(x1, min(x2, x1 + d)):
            cuts = [x for x in range(a, x2, d) if x1 + 0.3 * d < x < x2 - 0.3 * d]
            if len(cuts) < 3:
                continue
            vals = [proj[c] for c in cuts]
            score = float(np.mean(vals)) + 0.5 * float(np.std(vals))
            if best is None or score < best[0]:
                best = (score, d, a, cuts)
    return best


def band_proj(clean, ya, yb, x1, x2):
    return clean[ya:yb, x1:x2].sum(axis=0) / 255.0


def main():
    img = imread_unicode('古籍评测集/医家类/丹方彚编/page_12.png')
    h, w = img.shape[:2]
    X0 = 540
    sub = img[:, X0:1000]
    proj, clean = ink_projection(sub)
    pitch = estimate_pitch(proj, 21.0)
    print('pitch', pitch)
    # 全局格点（整列）
    g = fit_lattice(proj, 5, 451, pitch - 4, pitch + 6)
    print('global lattice pitch=%d phase=%d cuts=%s' % (g[1], g[2] + X0, [c + X0 for c in g[3]]))
    H = 21.0
    win = int(8 * H); step = int(4 * H)
    marks = []
    for ya in range(0, h - win + 1, step):
        pb = band_proj(clean, ya, ya + win, 5, 451)
        lb = fit_lattice(pb, 5, 451, pitch - 4, pitch + 6)
        shift = np.median([b - a for a, b in zip(g[3], lb[3])]) if lb else 0
        marks.append((ya, ya + win, lb[1] if lb else None, lb[2] + X0 if lb else None,
                      [c + X0 for c in (lb[3] if lb else [])], float(shift)))
    for m in marks:
        print(f'y[{m[0]},{m[1]}) pitch={m[2]} phase={m[3]} shift={m[5]:+.1f} cuts={m[4][:4]}...')
    return marks, g, (proj, clean, X0)


if __name__ == '__main__':
    main()
