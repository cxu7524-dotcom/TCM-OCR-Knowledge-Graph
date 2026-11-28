# -*- coding: utf-8 -*-
"""原型：按「本组自己的 y 区间」修正列槽左右边界，再重跑列内检测。

判据：列槽边界应当落在**空白**处。若边界落在墨上（本组 y 区间内的墨量高），
说明这个槽被相邻列侵入了，需要在该 y 区间内把边界外推到最近的空隙。
只影响边界落在墨上的槽——上半栏的槽边界本来就在空隙里，因此逐字不变。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from ocr.vertical import imread_unicode
from ocr.small_columns import propose_in_column, estimate_page_typography
from dual_track_fix.slots import layout_slots

PAGE = '古籍评测集/医家类/丹方彚编/page_12.png'
PID = '丹方彚编/page_12'


def ink_mask(gray):
    bg = cv2.GaussianBlur(gray, (31, 31), 0).astype(np.int16)
    return ((bg - gray.astype(np.int16)) >= 14)


def col_frac(ink, x, y0, y1):
    return float(ink[y0:y1, x].sum()) / max(1, (y1 - y0))


def snap_edge(ink, x, y0, y1, pitch, direction):
    """从 x 出发沿 direction 走，直到找到空隙（墨量低于地板）；返回空隙位置。

    只在「起点在墨上」时才修；找不到空隙就原样返回。
    """
    floor = 0.08
    h = y1 - y0
    if h < 8:
        return x, False
    if col_frac(ink, min(max(x, 0), ink.shape[1] - 1), y0, y1) < floor:
        return x, False                      # 本来就在空隙里
    span = int(round(0.6 * pitch))
    for d in range(1, span + 1):
        xx = x + direction * d
        if xx <= 0 or xx >= ink.shape[1] - 1:
            return x, False
        if col_frac(ink, xx, y0, y1) < floor:
            return xx, True
    return x, False


def main():
    img = imread_unicode(PAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = ink_mask(gray)
    lay = layout_slots(img, 21.0)
    cols = lay['cols']
    scale = estimate_page_typography(img, cols)
    w = img.shape[1]
    pitch = 50

    ann = json.load(open('output/dual_tracks/stage2/annotations.json', encoding='utf-8'))
    G = {g['group_id'].split('/')[-1]: g for g in ann['groups'] if g['page_id'] == PID}

    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        i = (ix1 - ix0) * (iy1 - iy0)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
        return i / u if u else 0.0

    # 第一遍：当前槽
    groups = []
    for i, col in enumerate(cols):
        slot = [int(v) for v in col['bbox']]
        x1, _, x2, _ = slot
        left = max([b['bbox'][2] for j, b in enumerate(cols) if j != i and b['bbox'][2] <= x1], default=0)
        right = min([b['bbox'][0] for j, b in enumerate(cols) if j != i and b['bbox'][0] >= x2], default=w)
        lim = ((left + x1) // 2, (right + x2) // 2)
        for p in propose_in_column(img, slot, lim, scale, span_ratio=0.9) or []:
            lanes = {c['side']: list(c['bbox']) for c in p['columns']}
            groups.append({'slot': slot, 'lim': lim, 'lanes': lanes,
                           'main': p['main_bbox'], 'ev': p['evidence'], 'prop': p})

    # 第二遍：边界落在墨上的槽，按本组 y 区间外推 + 重跑
    out = []
    for g in groups:
        rb, lb = g['lanes'].get('right'), g['lanes'].get('left')
        if not rb or not lb:
            continue
        y0, y1 = min(lb[1], rb[1]), max(lb[3], rb[3])
        x1, _, x2, _ = g['slot']
        nl, chl = snap_edge(ink, x1, y0, y1, pitch, -1)
        nr, chr_ = snap_edge(ink, x2 - 1, y0, y1, pitch, +1)
        nr = nr + 1 if chr_ else x2
        if not (chl or chr_):
            out.append({'group': g, 'slot2': g['slot'], 'changed': False, 'lanes': g['lanes']})
            continue
        slot2 = [nl, g['slot'][1], nr, g['slot'][3]]
        lim2 = g['lim']
        ps = propose_in_column(img, slot2, lim2, scale, span_ratio=0.9) or []
        if not ps:
            out.append({'group': g, 'slot2': slot2, 'changed': True, 'lanes': g['lanes'],
                        'note': 'no_proposal'})
            continue
        # 取 y 重叠最大的那条
        best = max(ps, key=lambda p: min(p['columns'][0]['bbox'][3], y1) - max(p['columns'][0]['bbox'][1], y0))
        lanes2 = {c['side']: list(c['bbox']) for c in best['columns']}
        out.append({'group': g, 'slot2': slot2, 'changed': True, 'lanes': lanes2,
                    'main2': best['main_bbox'], 'ev2': best['evidence']})

    print('changed slots:', sum(1 for o in out if o['changed']), '/', len(out))
    for o in out:
        if not o['changed']:
            continue
        g = o['group']
        lt = o['lanes']
        gb = [min(lt['left'][0], lt['right'][0]), min(lt['left'][1], lt['right'][1]),
              max(lt['left'][2], lt['right'][2]), max(lt['left'][3], lt['right'][3])]
        print('slot', g['slot'], '->', o['slot2'], 'group', gb)

    # 评测
    groups2 = []
    for o in out:
        lt = o['lanes']
        if 'right' not in lt or 'left' not in lt:
            continue
        gb = [min(lt['left'][0], lt['right'][0]), min(lt['left'][1], lt['right'][1]),
              max(lt['left'][2], lt['right'][2]), max(lt['left'][3], lt['right'][3])]
        groups2.append(gb)
    print('\n--- p12 IoU（第二遍 vs GT）---')
    hits = 0
    for name, gg in G.items():
        best = max(groups2, key=lambda b: iou(b, gg['group_bbox']))
        v = iou(best, gg['group_bbox'])
        hits += v >= 0.5
        print(f'{name} {v:.3f} {best}')
    print('hits@0.5', hits)


if __name__ == '__main__':
    main()
