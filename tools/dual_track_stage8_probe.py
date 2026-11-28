# -*- coding: utf-8 -*-
"""stage8 诊断探针：对任一预测集，逐组给出「最佳组 IoU / 对应预测 / 分侧 IoU」。

不是评测口径，只是定位用：把「差多少」摊开，避免在阈值附近猜。
用法：
  python tools/dual_track_stage8_probe.py --pred <predictions.json> [--out diag.json]
预测集合可以是 path_c.json / stage6 groups（自动转 refined）等。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_eval.geometry import iou  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def load_preds(path: Path) -> List[Dict[str, Any]]:
    doc = _load(path)
    if isinstance(doc, list):
        return doc
    if 'predictions' in doc:
        return doc['predictions']
    # stage5/6 groups.json → 精修后预测
    from dual_track_refine import refined_predictions
    return refined_predictions(doc, use_before=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', default='')
    args = ap.parse_args(argv)
    ann = _load(S2 / 'annotations.json')['groups']
    preds = load_preds(Path(args.pred))
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        by_page.setdefault(p['page_id'], []).append(p)
    rows = []
    for g in ann:
        best = {'group_iou': 0.0, 'prediction_id': None, 'right_iou_fixed': None,
                'left_iou_fixed': None, 'swap': None, 'overlap_gt_right': None,
                'overlap_gt_left': None}
        for p in by_page.get(g['page_id'], []):
            v = iou(p['group_bbox'], g['group_bbox'])
            if v <= best['group_iou']:
                continue
            r = iou(p['right_bbox'], g['right_bbox'])
            l = iou(p['left_bbox'], g['left_bbox'])
            best = {'group_iou': round(v, 4), 'prediction_id': p['prediction_id'],
                    'right_iou_fixed': round(r, 4), 'left_iou_fixed': round(l, 4),
                    'swap': round(max(iou(p['right_bbox'], g['left_bbox']),
                                      iou(p['left_bbox'], g['right_bbox'])), 4),
                    'overlap_gt_right': round(iou(p['group_bbox'], g['right_bbox']), 4),
                    'overlap_gt_left': round(iou(p['group_bbox'], g['left_bbox']), 4)}
        rows.append({'group_id': g['group_id'], 'page_id': g['page_id'],
                     'main_text': g['main_text'], 'gt_group': g['group_bbox'],
                     'gt_right': g['right_bbox'], 'gt_left': g['left_bbox'],
                     **best})
    hits = {tk: sum(1 for r in rows if r['group_iou'] >= tk) for tk in (0.5, 0.75)}
    out = {'pred': str(Path(args.pred).resolve().relative_to(ROOT)).replace('\\', '/'),
           'n_predictions': len(preds), 'hits': hits, 'rows': rows}
    print('pred', out['pred'], 'n', len(preds), 'best-iou hits', hits)
    for r in rows:
        print(f"{r['group_id']:32s} {r['main_text']:5s} iou={r['group_iou']:.3f} "
              f"R={r['right_iou_fixed']} L={r['left_iou_fixed']} pred={r['prediction_id']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                                  encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
