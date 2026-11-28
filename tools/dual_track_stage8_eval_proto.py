# -*- coding: utf-8 -*-
"""stage8 原型评测：跑新检测器，逐组报最佳 IoU（诊断用，不是评测口径）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_eval.geometry import iou  # noqa: E402
from dual_track_fix.config import DEFAULT_CONFIG  # noqa: E402
from dual_track_fix.pairs import detect_page, page_ink, rule_mask  # noqa: E402
from ocr.vertical import imread_unicode  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'


def load(p: Path):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def run_page(page_id: str, image_path: str, H: float, halves, lines,
             cfg=None) -> dict:
    import cv2
    img = imread_unicode(str(ROOT / image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink0 = page_ink(gray, cfg)
    bad = rule_mask(lines, ink0.shape)
    ink = ink0
    out = detect_page(ink, H, halves, cfg, bad)
    for i, g in enumerate(out['groups'], 1):
        g['prediction_id'] = f'{page_id}#g{i:03d}'
    return {'page_id': page_id, 'groups': out['groups'], 'rejected': out['rejected']}


def main() -> int:
    ann = load(S2 / 'annotations.json')
    sr = load(S4 / 'search_regions.json')
    masks = {p['page_id']: p.get('lines') or [] for p in load(S4 / 'rule_masks.json')['pages']}
    meta = {p['page_id']: p for p in sr['pages']}
    groups_by_page = {}
    for g in ann['groups']:
        groups_by_page.setdefault(g['page_id'], []).append(g)
    rows, hits = [], {0.5: 0, 0.75: 0}
    total_pred = 0
    for pid, gs in groups_by_page.items():
        m = meta[pid]
        H = float((m.get('page_scale') or {}).get('H') or 20)
        res = run_page(pid, gs[0]['image_path'], H, m['halves'], masks.get(pid, []))
        total_pred += len(res['groups'])
        for g in gs:
            best = max(res['groups'], key=lambda p: iou(p['group_bbox'], g['group_bbox']),
                       default=None)
            v = iou(best['group_bbox'], g['group_bbox']) if best else 0.0
            r = iou(best['right'], g['right_bbox']) if best else 0.0
            l = iou(best['left'], g['left_bbox']) if best else 0.0
            for tk in hits:
                hits[tk] += 1 if v >= tk else 0
            rows.append({'group_id': g['group_id'], 'H': H, 'iou': round(v, 4),
                         'R': round(r, 4), 'L': round(l, 4),
                         'pred': best['prediction_id'] if best else None,
                         'pred_box': best['group_bbox'] if best else None,
                         'gt': g['group_bbox']})
        print(f"{pid:22s} H={H:4.1f} preds={len(res['groups']):2d} "
              f"iou={[round(x['iou'],2) for x in rows if x['group_id'].startswith(pid[:3]) and x['group_id'] in [y['group_id'] for y in gs]]}")
    print('hits', hits, 'preds', total_pred)
    for r in rows:
        print(f"{r['group_id'][:30]:32s} iou={r['iou']:.3f} R={r['R']:.3f} L={r['L']:.3f} {r['pred']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
