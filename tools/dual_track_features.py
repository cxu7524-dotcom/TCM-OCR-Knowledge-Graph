"""Evaluate simple pixel features of labelled dual-track regions (20 pos / 22 neg)."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = json.loads((ROOT / 'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'))
DEV = json.loads((ROOT / 'output/small_columns_step1_dev/annotation.json').read_text(encoding='utf8'))


def rd(p):
    return cv2.imdecode(np.fromfile(str(ROOT / p), dtype=np.uint8), cv2.IMREAD_COLOR)


def mask_of(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    return bw


def features(gray, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = gray[y0:y1, x0:x1]
    if sub.size < 100:
        return None
    m = (mask_of(sub) > 0).astype(np.float32)
    colp = m.mean(axis=0)
    W = len(colp)
    if W < 10:
        return None
    mid = range(int(W * 0.25), int(W * 0.75) + 1)
    xv = min(mid, key=lambda k: colp[k])
    lb, rb = m[:, :max(1, xv - 1)], m[:, min(W, xv + 2):]
    if lb.size == 0 or rb.size == 0:
        return None
    lp, rp = colp[:max(1, xv - 1)].max(), colp[min(W, xv + 2):].max()
    vratio = float(colp[xv] / max(1e-6, min(lp, rp)))
    rows = (lb.mean(axis=1) > 0.05) & (rb.mean(axis=1) > 0.05)
    val = m[:, max(0, xv - 1):xv + 2].mean(axis=1) < 0.02
    good = rows & val
    run, best = 0, 0
    for g in good:
        run = run + 1 if g else 0
        best = max(best, run)
    # glyph height via vertical run lengths on each side
    def runs(side):
        r = (side.mean(axis=1) > 0.04).astype(np.uint8)
        out, cur = [], 0
        for v in r:
            cur = cur + 1 if v else 0
            if not v and cur:
                out.append(cur); cur = 0
        if cur:
            out.append(cur)
        return out
    rl = runs(lb) + runs(rb)
    gh = float(np.median(rl)) if rl else 0
    return {'w': x1 - x0, 'h': y1 - y0, 'vratio': round(vratio, 3), 'run': best,
            'fill_l': round(float(lb.mean()), 3), 'fill_r': round(float(rb.mean()), 3),
            'gh': round(gh, 1), 'xv_frac': round(xv / W, 2)}


def main():
    rows = []
    for s in H['samples']:
        if s['label'] != 'positive':
            continue
        im = rd(H['pages'][str(s['page'])]['path'])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        x0 = min(c['bbox'][0] for c in s['columns'])
        x1 = max(c['bbox'][2] for c in s['columns'])
        y0 = min(c['bbox'][1] for c in s['columns'])
        y1 = max(c['bbox'][3] for c in s['columns'])
        f = features(gray, [x0, y0, x1, y1])
        rows.append(('POS', s['id'], f))
    for s in H['samples']:
        if s['label'] != 'negative' or not s.get('columns'):
            continue
        im = rd(H['pages'][str(s['page'])]['path'])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if len(s['columns']) == 2:
            x0 = min(c['bbox'][0] for c in s['columns']); x1 = max(c['bbox'][2] for c in s['columns'])
            y0 = min(c['bbox'][1] for c in s['columns']); y1 = max(c['bbox'][3] for c in s['columns'])
            f = features(gray, [x0, y0, x1, y1])
        else:
            f = features(gray, s['columns'][0]['bbox'])
        rows.append(('NEG', s['id'], f))
    for tag, sid, f in rows:
        if f:
            print(f"{tag} {sid:10s} w={f['w']:3d} h={f['h']:3d} vratio={f['vratio']:.3f} run={f['run']:3d} "
                  f"fillL={f['fill_l']:.2f} fillR={f['fill_r']:.2f} gh={f['gh']:.1f} xv={f['xv_frac']:.2f}")
        else:
            print(tag, sid, 'no-features')


if __name__ == '__main__':
    main()
