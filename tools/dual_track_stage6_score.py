# -*- coding: utf-8 -*-
"""stage6 冻结评分：同一 19 页/38 组，比较 v4 暂定框与精修框。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_eval import score_predictions  # noqa: E402
import dual_track_stage3_score as S3  # noqa: E402
from dual_track_refine import refined_predictions  # noqa: E402
from dual_track_refine.api import S6, load  # noqa: E402
from dual_track_track.inputs import S2, S3 as S3DIR, canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))


def _summary(res: Dict[str, Any]) -> Dict[str, Any]:
    out = {tk: {} for tk in res['summary']['overall']}
    for tk, raw in res['summary']['overall'].items():
        out[tk] = {k: raw.get(k) for k in ('group_hits', 'tp', 'fp', 'fn',
                                           'new_candidates', 'lane_counts',
                                           'lane_denominator', 'hit_rate_on_known')}
    rows = []
    for pg in res.get('pages', []):
        for tk, t in (pg.get('thresholds') or {}).items():
            for row in (t.get('lane') or {}).get('rows') or []:
                rows.append({'page_id': pg['page_id'], 'threshold': tk, **row})
    return {'overall': out, 'lane_rows': rows}


def _by_page(res: Dict[str, Any]) -> Dict[str, Any]:
    return {pg['page_id']: {tk: {'group_hits': t.get('group_hits'),
                                 'pairs': t.get('pairs') or []}
                            for tk, t in (pg.get('thresholds') or {}).items()}
            for pg in res.get('pages', [])}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage6 冻结评分')
    ap.add_argument('--groups', default=str(S6 / 'groups.json'))
    ap.add_argument('--out', default=str(S6 / 'metrics.json'))
    args = ap.parse_args(argv)
    payload = load(Path(args.groups))
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    cfg_doc = load(S3DIR / 'evaluation_config.json')
    runtime = S3.runtime_config(cfg_doc)
    pages = sorted({r['page_id'] for r in payload['pages']})
    tables = S3.build_tables(ann, neg, pages, S3.region_thresholds(runtime))
    split = S3.page_split_map(load(S2 / 'dataset_splits.json'))
    sizes = S3.page_sizes_map(ann)
    v1_path = S6 / 'history' / 'stage6-v1' / 'groups.json'
    v1_path = S6 / 'history' / 'stage6-v3' / 'groups.json'
    v1_payload = load(v1_path) if v1_path.exists() else None
    scored = {}
    for tag, use_before in (('v4_tentative', True), ('stage6_refined', False)):
        preds = refined_predictions(payload, use_before=use_before)
        res = score_predictions(preds, ann['groups'], tables, runtime, pages=pages,
                                page_split=split, page_sizes=sizes)
        scored[tag] = {'summary': _summary(res), 'by_page': _by_page(res)}
    if v1_payload is not None:
        preds = refined_predictions(v1_payload, use_before=False)
        res = score_predictions(preds, ann['groups'], tables, runtime, pages=pages,
                                page_split=split, page_sizes=sizes)
        scored['stage6_v3'] = {'summary': _summary(res), 'by_page': _by_page(res)}

    before = scored['v4_tentative']['summary']['overall']
    after = scored['stage6_refined']['summary']['overall']
    delta = {tk: {k: (after[tk].get(k) or 0) - (before[tk].get(k) or 0)
                  for k in ('group_hits', 'fp', 'fn')} for tk in ('0.5', '0.75')}
    if 'stage6_v3' in scored:
        v1 = scored['stage6_v3']['summary']['overall']
        delta['v3_to_v4'] = {tk: {k: (after[tk].get(k) or 0) - (v1[tk].get(k) or 0)
                               for k in ('group_hits', 'fp', 'fn')}
                            for tk in ('0.5', '0.75')}

    # 原 v4 五个轨级子框定位失败组逐条变化
    problem_ids = ['丹方彚编/p12/DF12-03', '丹方彚编/p12/DF12-04',
                   '丹方彚编/p12/DF12-11', '思濟堂方書/p35/dev-35-001',
                   '思濟堂方書/p46/P46-08']
    problem = []
    for gid in problem_ids:
        row = {'group_id': gid}
        for tag in ('v4_tentative', 'stage6_v3', 'stage6_refined'):
            rows = [r for r in scored[tag]['summary']['lane_rows']
                    if r['group_id'] == gid and r['threshold'] == '0.5']
            row[tag] = {'category': rows[0].get('category') if rows else None,
                        'prediction_id': rows[0].get('prediction_id') if rows else None,
                        'group_iou': rows[0].get('group_iou') if rows else None,
                        'right_iou_fixed': rows[0].get('right_iou_fixed') if rows else None,
                        'left_iou_fixed': rows[0].get('left_iou_fixed') if rows else None}
        b = row.get('stage6_v3') or row['v4_tentative']
        a = row['stage6_refined']
        row['change'] = ('改善' if a['category'] == 'pairing_success'
                         and b['category'] != 'pairing_success'
                         else '退化' if b['category'] == 'pairing_success'
                         and a['category'] != 'pairing_success' else '持平')
        problem.append(row)

    out = {
        'stage': 'stage6_frozen_comparison', 'release': payload.get('release'),
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'config': {'file': 'output/dual_tracks/stage3/evaluation_config.json',
                   'sha256': sha256_of(S3DIR / 'evaluation_config.json'),
                   'release': cfg_doc.get('release')},
        'scope': {'pages': len(pages), 'annotations': len(ann['groups']),
                  'protocol': 'stage3-v1 / min_cost_max_flow / 0.5 与 0.75 独立匹配'},
        'inputs': {'stage6_groups_sha256': sha256_of(Path(args.groups)),
                   'stage6_content_sha256': payload['determinism']['content_sha256'],
                   'stage6_v3_content_sha256': (v1_payload or {}).get('determinism', {}).get('content_sha256')},
        'scores': scored, 'delta': delta, 'problem_groups': problem,
        'predictions_content_sha256': {
            'v4_tentative': canon_sha256(refined_predictions(payload, True)),
            'stage6_refined': canon_sha256(refined_predictions(payload, False))},
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print('v4 @0.5/0.75', {tk: before[tk]['group_hits'] for tk in ('0.5', '0.75')},
          '→ stage6', {tk: after[tk]['group_hits'] for tk in ('0.5', '0.75')})
    print('轨级 @0.5', before['0.5']['lane_counts'], '→', after['0.5']['lane_counts'])
    if 'stage6_v3' in scored:
        v1 = scored['stage6_v3']['summary']['overall']
        print('v3→v4 @0.5', v1['0.5']['group_hits'], '→', after['0.5']['group_hits'])
    print('逐组变化', {x['group_id'].split('/')[-1]: x['change'] for x in problem})
    print('→', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
