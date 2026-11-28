# -*- coding: utf-8 -*-
"""第五阶段评分：用**阶段 3 冻结配置**给本阶段的预测打分，并与冻结基线对照。

口径全部沿用阶段 3：
  - 组匹配 IoU 0.5 / 0.75（两个阈值独立求最优匹配）
  - 轨级：right/left 子框 IoU 0.5，分母排除 ambiguous_lane_*
  - 重复/碎组/并组：诊断标签，不免除基础 FP
  - 端点敏感性：0.75 下 pending 端点两种解释各算一遍
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_eval import (RegionTable, endpoint_sensitivity,  # noqa: E402
                             score_predictions)
import dual_track_stage3_score as S3                              # noqa: E402
from dual_track_track.api import groups_as_predictions            # noqa: E402
from dual_track_track.inputs import S2, S3 as S3DIR, S5, canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage5 评分')
    ap.add_argument('--groups', default=str(S5 / 'groups.json'))
    ap.add_argument('--out', default=str(S5 / 'metrics.json'))
    args = ap.parse_args(argv)

    payload = load(Path(args.groups))
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    cfg_doc = load(S3DIR / 'evaluation_config.json')
    runtime = S3.runtime_config(cfg_doc)
    pages = sorted({p['page_id'] for p in payload['pages']})
    tables = S3.build_tables(ann, neg, pages, S3.region_thresholds(runtime))
    split = S3.page_split_map(load(S2 / 'dataset_splits.json'))
    sizes = S3.page_sizes_map(ann)
    preds = groups_as_predictions(payload)

    res = score_predictions(preds, ann['groups'], tables, runtime, pages=pages,
                            page_split=split, page_sizes=sizes)
    overall = {tk: {k: v for k, v in res['summary']['overall'][tk].items()
                    if k in ('group_hits', 'annotations_total', 'hit_rate_on_known',
                             'tp', 'fp', 'fn', 'new_candidates', 'lane_counts',
                             'lane_denominator', 'lane_hit_rate')}
               for tk in res['summary']['overall']}

    baseline = load(S3DIR / 'metrics' / 'c.json')
    base_scores = baseline['scores']['overall']
    comparison = {}
    for tk in ('0.5', '0.75'):
        comparison[tk] = {
            'stage5_group_hits': overall.get(tk, {}).get('group_hits'),
            'baseline_c_group_hits': base_scores.get(tk, {}).get('group_hits'),
            'delta': (overall.get(tk, {}).get('group_hits') or 0)
                     - (base_scores.get(tk, {}).get('group_hits') or 0),
        }

    sens = endpoint_sensitivity(preds, ann['groups'], tables, runtime)
    diag = res.get('diagnostics') or {}
    by_page, per_group = _page_and_group_detail(res, pages)
    attribution = _attribution(payload, res, ann)

    out = {
        'stage': 'stage5_dual_track_metrics',
        'release': payload.get('release'),
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'config': {'file': 'output/dual_tracks/stage3/evaluation_config.json',
                   'sha256': sha256_of(S3DIR / 'evaluation_config.json'),
                   'release': cfg_doc.get('release'),
                   'runtime': {'group_iou_thresholds': runtime['group_iou_thresholds'],
                               'lane_child_thresholds': runtime['lane_child_thresholds'],
                               'excluded_certainty': runtime.get(
                                   'lane_eligible_exclude_certainty')}},
        'predictions': {'n': len(preds),
                        'content_sha256': canon_sha256(preds),
                        'source': 'output/dual_tracks/stage5/groups.json'},
        'groups_in_artifact': payload['summary']['groups'],
        'scores': {'overall': overall,
                   'by_split': res.get('summary', {}).get('by_split'),
                   'by_page': by_page,
                   'per_group': per_group,
                   'diagnostics': diag,
                   'metric_limits': res.get('metric_limits')},
        'baseline_comparison': {'baseline_id': baseline.get('baseline_snapshot_id')
                                or 'baseline-848bb88d3e30',
                                'path': 'C（旧布局 + propose_page）', **comparison},
        'endpoint_sensitivity': sens,
        'failure_attribution': attribution,
        'pending_groups': [g['group_id'] for p in payload['pages']
                           for g in p.get('groups') or [] if g.get('pending')],
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print('预测', len(preds), '组；overall', {tk: overall[tk].get('group_hits')
                                              for tk in overall})
    print('对照基线 C', {tk: comparison[tk] for tk in comparison})
    print('→', args.out)
    return 0


def _page_and_group_detail(res: Dict[str, Any], pages: Sequence[str]
                           ) -> tuple:
    """冻结评分的逐页 / 逐组明细：命中对、IoU、一对一匹配次数都可复查。"""
    by_page: Dict[str, Any] = {}
    per_group: Dict[str, Any] = {}
    for pg in res.get('pages', []):
        pid = pg['page_id']
        ths = pg.get('thresholds') or {}
        by_page[pid] = {
            'annotations': (pg.get('counts') or {}).get('annotations'),
            'predictions': (pg.get('counts') or {}).get('valid_predictions'),
            'thresholds': {tk: {'group_hits': t.get('group_hits'),
                                'tp': t.get('tp'), 'fp': t.get('fp'),
                                'fn': t.get('fn'),
                                'new_candidates': t.get('new_candidates'),
                                'pairs': t.get('pairs') or [],
                                'lane_rows': (t.get('lane') or {}).get('rows') or []}
                           for tk, t in ths.items() if isinstance(t, dict)}}
        for tk, t in ths.items():
            if not isinstance(t, dict):
                continue
            for pred_id, gid, iou in (t.get('pairs') or []):
                row = per_group.setdefault(gid, {'page_id': pid, 'matches': {}})
                row['matches'].setdefault(tk, []).append(
                    {'prediction_id': pred_id, 'iou': round(float(iou), 6)})
    for gid, row in per_group.items():
        row['matched_prediction_count'] = {tk: len(v)
                                           for tk, v in row['matches'].items()}
    return by_page, per_group


def _attribution(payload: Dict[str, Any], res: Dict[str, Any],
                 ann: Dict[str, Any]) -> Dict[str, Any]:
    """把失败分四类：搜索范围不足 / 成对拒绝 / 断组错误 / 暂定框偏差。"""
    pages = {p['page_id']: p for p in payload['pages']}
    gt_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in ann['groups']:
        gt_by_page.setdefault(g['page_id'], []).append(g)

    search_gap, pair_reject, split_err, box_bias = [], [], [], []
    for pid, p in pages.items():
        covered = 0
        for gt in gt_by_page.get(pid, []):
            b = gt['group_bbox']
            hit = [g for g in p.get('groups') or []
                   if _overlap_ratio(g.get('group_bbox'), b) > 0.3]
            if hit:
                covered += 1
                continue
            region = _region_containing(p, b)
            if region is None:
                search_gap.append({'page_id': pid, 'group': gt['group_id'],
                                   'why': '标注组不落在任何搜索区域内'})
            else:
                reasons = region.get('candidate_runs') or []
                rejected = [r for r in reasons if not r.get('confirmed')]
                pair_reject.append({'page_id': pid, 'group': gt['group_id'],
                                    'region': region['region_id'],
                                    'region_source': region.get('source'),
                                    'rejected_runs': len(rejected),
                                    'top_reject': (rejected[0].get('rejected')
                                                   if rejected else None)})
    for pid, p in pages.items():
        for g in p.get('groups') or []:
            stops = {e['type'] for e in g.get('stop_events') or []}
            if 'seam_missing' in stops and 'blank_gap' not in stops:
                split_err.append({'page_id': pid, 'group_id': g['group_id'],
                                  'why': 'seam_missing 结束，可能本可继续',
                                  'bbox': g.get('group_bbox')})
    return {'search_range_gaps': search_gap,
            'pairing_rejections': pair_reject,
            'split_errors': split_err,
            'box_bias': box_bias,
            'note': ('搜索范围不足=标注组不在任何区域内；成对拒绝=区域在但没确认；'
                     '断组/暂定框偏差取评分器诊断字段')}


def _overlap_ratio(a: Optional[List[int]], b: List[int]) -> float:
    if not a:
        return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / area


def _region_containing(page: Dict[str, Any], bbox: List[int]) -> Optional[Dict[str, Any]]:
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    best = None
    for r in page.get('regions') or []:
        rb = r.get('region_bbox')
        if not rb:
            continue
        if rb[0] <= cx < rb[2] and rb[1] <= cy < rb[3]:
            best = r
            break
    return best


if __name__ == '__main__':
    raise SystemExit(main())
