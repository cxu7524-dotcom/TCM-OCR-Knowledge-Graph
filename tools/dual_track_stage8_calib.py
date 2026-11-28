# -*- coding: utf-8 -*-
"""stage8 阈值标定：量「中缝」的可测特征，不看模型猜。

对 38 个已标注组：量 GT 缝（left.x2 .. right.x0）在组高度内的空白占比与最长空白行
对 10 个负例区：量区内最像缝的那一列（空白占比、最长空白行、缝宽）

输出两族分布，用来定 min_run / max_corridor 两个阈值。
只在 development 分集上定值；historical_regression / independent_acceptance 只报不调。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ocr.vertical import imread_unicode  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def ink_of(path):
    import cv2
    img = imread_unicode(str(ROOT / path))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 14)
    return bw > 0


def rule_of(lines, shape):
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=bool)
    for ln in lines or []:
        for s in (ln.get('segments') or [ln]):
            x0 = max(0, int(s.get('x0', 0))); x1 = min(w, int(s.get('x1', 0)))
            y0 = max(0, int(s.get('y0', 0))); y1 = min(h, int(s.get('y1', 0)))
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = True
    return out


def col_stats(white, ink_norule, x, y0, y1):
    """单列在 [y0,y1) 内的空白统计。"""
    col = white[y0:y1, x]
    n = col.size
    if n == 0:
        return None
    # 最长连续空白（行内）
    best = cur = 0
    for v in col:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    # 两端是否被字形墨（非栏线）夹住
    top_ink = bool(ink_norule[max(0, y0 - 1):y0, x].any())
    bot_ink = bool(ink_norule[y1:y1 + 1, x].any())
    return {'white_frac': float(col.mean()), 'max_blank': int(best),
            'blank_frac': best / n, 'top_ink': top_ink, 'bot_ink': bot_ink}


def main() -> int:
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    sr = load(S4 / 'search_regions.json')
    masks = {p['page_id']: p.get('lines') or [] for p in load(S4 / 'rule_masks.json')['pages']}
    meta = {p['page_id']: p for p in sr['pages']}

    ink_cache = {}
    rule_cache = {}

    def get(pid, path):
        if pid not in ink_cache:
            ink_cache[pid] = ink_of(path)
            rule_cache[pid] = rule_of(masks.get(pid, []), ink_cache[pid].shape)
        return ink_cache[pid], rule_cache[pid]

    print('=== A. 已标注组：GT 缝的空白结构（组高度内） ===')
    print(f"{'group':24s} {'split':9s} {'H':>5s} {'seamw':>5s} {'gh/H':>6s} "
          f"{'wfrac':>6s} {'blank':>6s} {'bfrac':>6s} {'top':>4s} {'bot':>4s}")
    rows_a = []
    for g in ann['groups']:
        pid = g['page_id']
        H = float((meta[pid].get('page_scale') or {}).get('H') or 20)
        ink, rule = get(pid, g['image_path'])
        white = ~ink
        ik = ink & ~rule
        l, r, gb = g['left_bbox'], g['right_bbox'], g['group_bbox']
        sx0, sx1 = l[2], r[0]
        y0, y1 = gb[1], gb[3]
        cols = []
        for x in range(max(0, sx0 - 1), min(ink.shape[1], sx1 + 1)):
            s = col_stats(white, ik, x, y0, y1)
            if s:
                cols.append(s)
        if not cols:
            continue
        wf = float(np.mean([c['white_frac'] for c in cols]))
        mb = int(max(c['max_blank'] for c in cols))
        bf = float(max(c['blank_frac'] for c in cols))
        top = all(c['top_ink'] for c in cols)
        bot = all(c['bot_ink'] for c in cols)
        rows_a.append({'group_id': g['group_id'], 'split': g['split_membership'],
                       'H': H, 'seam_w': sx1 - sx0, 'gh_H': (y1 - y0) / H,
                       'wfrac': wf, 'blank': mb, 'bfrac': bf, 'top': top, 'bot': bot})
        print(f"{g['group_id'].split('/',1)[1]:24s} {g['split_membership'][:9]:9s} "
              f"{H:5.1f} {sx1-sx0:5d} {(y1-y0)/H:6.2f} {wf:6.3f} {mb:6d} {bf:6.3f} "
              f"{str(top):>4s} {str(bot):>4s}")

    print()
    print('=== B. 负例区：区内最像缝的那一列 ===')
    print(f"{'neg':18s} {'cat':13s} {'H':>5s} {'bestx':>6s} {'w':>3s} {'wfrac':>6s} "
          f"{'blank':>6s} {'bfrac':>6s} {'blank/H':>7s} {'top':>4s} {'bot':>4s}")
    rows_b = []
    for rg in neg['regions']:
        pid = rg['page_id']
        H = float((meta[pid].get('page_scale') or {}).get('H') or 20)
        ink, rule = get(pid, rg['image_path'])
        white = ~ink
        ik = ink & ~rule
        x1, y1, x2, y2 = rg['bbox']
        best = None
        for x in range(max(0, x1), min(ink.shape[1], x2)):
            s = col_stats(white, ik, x, y1, y2)
            if not s:
                continue
            if best is None or s['max_blank'] > best[1]['max_blank']:
                best = (x, s)
        if best is None:
            continue
        x, s = best
        # 该列所在的连续空白列簇宽度
        w = 1
        xx = x - 1
        while xx >= x1 and white[y1:y2, xx].mean() > 0.9:
            w += 1; xx -= 1
        xx = x + 1
        while xx < x2 and white[y1:y2, xx].mean() > 0.9:
            w += 1; xx += 1
        rows_b.append({'neg': rg['negative_region_id'], 'cat': rg['category'], 'H': H,
                       'x': x, 'w': w, 'wfrac': s['white_frac'],
                       'blank': s['max_blank'], 'bfrac': s['blank_frac'],
                       'bh': s['max_blank'] / H, 'top': s['top_ink'], 'bot': s['bot_ink']})
        print(f"{rg['negative_region_id']:18s} {rg['category'][:13]:13s} {H:5.1f} "
              f"{x:6d} {w:3d} {s['white_frac']:6.3f} {s['max_blank']:6d} "
              f"{s['blank_frac']:6.3f} {s['max_blank']/H:7.2f} "
              f"{str(s['top_ink']):>4s} {str(s['bot_ink']):>4s}")

    out = ROOT / 'output' / 'dual_tracks' / 'stage8' / 'calib_seam_stats.json'
    out.write_text(json.dumps({'gt': rows_a, 'neg': rows_b}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('\nwrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
