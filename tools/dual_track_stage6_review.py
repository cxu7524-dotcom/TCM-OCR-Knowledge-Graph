# -*- coding: utf-8 -*-
"""stage6 38 组复评：冻结匹配、轨级分类、固定侧别 IoU 与证据索引。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_refine.api import S6, load  # noqa: E402
from dual_track_track.inputs import S2  # noqa: E402


def _index(metrics: Dict[str, Any], tag: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    scores = (metrics.get('scores', {}).get(tag, {}) or {})
    for page in (scores.get('by_page') or {}).values():
        for tk, spec in (page.get('thresholds') or {}).items():
            for pred, gid, iou in spec.get('pairs') or []:
                out.setdefault(gid, {})[f'hit_{tk}'] = {'pred': pred, 'iou': round(float(iou), 6)}
        for row in (scores.get('summary', {}).get('lane_rows') or []):
            if row.get('threshold') == '0.5':
                out.setdefault(row['group_id'], {})['lane'] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default=str(S6 / 'metrics.json'))
    ap.add_argument('--out', default=str(S6 / 'review_38.json'))
    args = ap.parse_args(argv)
    m = load(Path(args.metrics))
    ann = load(S2 / 'annotations.json')
    before = _index(m, 'v4_tentative')
    v1 = _index(m, 'stage6_v3')
    after = _index(m, 'stage6_refined')
    rows: List[Dict[str, Any]] = []
    for a in ann['groups']:
        gid = a['group_id']
        b, x = before.get(gid, {}), after.get(gid, {})
        v = v1.get(gid, {})
        lane_b, lane_a = b.get('lane') or {}, x.get('lane') or {}
        pred = (x.get('hit_0.5') or {}).get('pred') or (b.get('hit_0.5') or {}).get('pred')
        ev = ''
        if pred:
            p = S6 / 'evidence' / 'per_group' / f'{pred.replace("/", "__")}.jpg'
            if p.exists():
                ev = str(p.relative_to(ROOT)).replace('\\', '/')
        change = ('改善' if lane_a.get('category') == 'pairing_success'
                  and lane_b.get('category') != 'pairing_success'
                  else '退化' if lane_b.get('category') == 'pairing_success'
                  and lane_a.get('category') != 'pairing_success'
                  else '命中增加' if x.get('hit_0.5') and not b.get('hit_0.5')
                  else '命中减少' if b.get('hit_0.5') and not x.get('hit_0.5')
                  else '持平')
        rows.append({
            'group_id': gid, 'page_id': a['page_id'], 'main_text': a.get('main_text'),
            'v4': {k: b.get(k) for k in ('hit_0.5', 'hit_0.75', 'lane')},
            'stage6_v3': {k: v.get(k) for k in ('hit_0.5', 'hit_0.75', 'lane')},
            'stage6': {k: x.get(k) for k in ('hit_0.5', 'hit_0.75', 'lane')},
            'change': change,
            'fixed_iou_delta': {
                s: round(float(lane_a.get(f'{s}_iou_fixed') or 0)
                         - float(lane_b.get(f'{s}_iou_fixed') or 0), 4)
                for s in ('right', 'left')},
            'evidence': ev,
        })
    agg = {}
    for tag in ('v4_tentative', 'stage6_v3', 'stage6_refined'):
        lanes = [v.get('lane') or {} for v in
                 ({'v4_tentative': before, 'stage6_v3': v1,
                   'stage6_refined': after}[tag]).values()]
        agg[tag] = {s: round(sum(float(r.get(f'{s}_iou_fixed') or 0) for r in lanes)
                             / max(1, sum(1 for r in lanes
                                          if r.get(f'{s}_iou_fixed') is not None)), 4)
                    for s in ('right', 'left')}
    out = {
        'stage': 'stage6_review_38', 'release': m.get('release'),
        'protocol': m.get('scope', {}).get('protocol'),
        'fixed_side_mean_iou': agg,
        'counts': {'annotations': len(rows),
                   'change': {k: sum(1 for r in rows if r['change'] == k)
                              for k in ('改善', '退化', '命中增加', '命中减少', '持平')}},
        'rows': rows,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print('复评', len(rows), '组；固定侧别均 IoU', agg,
          '；变化', out['counts']['change'])
    print('→', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
