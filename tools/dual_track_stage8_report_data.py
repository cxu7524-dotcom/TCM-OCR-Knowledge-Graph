# -*- coding: utf-8 -*-
"""stage8 报数据汇总：把三份快照的逐组/逐页/分集/轨级/负例/端点敏感性摊平成一张表。

只做汇总与核对，不重算指标——指标一律来自 `tools/dual_track_stage3_score.py`
用冻结配置算出的 metrics json，避免「报告另起一套口径」。

输出：output/dual_tracks/stage8/report_data.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_eval.geometry import iou  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
S6 = ROOT / 'output' / 'dual_tracks' / 'stage6'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'

SNAPS = ['C', 'stage7v2', 'fix']


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def overlap_ratio(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return (ix2 - ix1) * (iy2 - iy1) / area


def main() -> int:
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    masks = {p['page_id']: p for p in load(S4 / 'rule_masks.json')['pages']}
    labels = load(S3 / 'frozen_layout_labels.json')
    s6 = load(S6 / 'groups.json')

    preds = {s: load(S8 / 'snapshots' / s / 'predictions.json')['predictions'] for s in SNAPS}
    met = {s: load(S8 / 'metrics' / f'{s}.json') for s in SNAPS}
    by_page = {s: {} for s in SNAPS}
    for s in SNAPS:
        for p in preds[s]:
            if p.get('group_bbox'):
                by_page[s].setdefault(p['page_id'], []).append(p)

    # ---------------- 逐组
    rows: List[Dict[str, Any]] = []
    for g in ann['groups']:
        row = {'group_id': g['group_id'], 'region': g['region'],
               'page_id': g['page_id'], 'split': g['split_membership'],
               'group_bbox': g['group_bbox'],
               'certainty': g.get('certainty'),
               'annotation_confidence': g.get('certainty'),
               'wide_box': ((g.get('labels') or {}).get('wide_box') or {}).get('state'),
               'rule_state': ((g.get('labels') or {}).get('rule_state')),
               'hits': {}}
        for s in SNAPS:
            ps = by_page[s].get(g['page_id'], [])
            best = max(ps, key=lambda p: iou(p['group_bbox'], g['group_bbox']), default=None)
            v = iou(best['group_bbox'], g['group_bbox']) if best else 0.0
            row['hits'][s] = {'iou': round(v, 4),
                              'hit50': v >= 0.5, 'hit75': v >= 0.75,
                              'pred': best['prediction_id'] if best else None,
                              'pred_bbox': best['group_bbox'] if best else None}
        rows.append(row)

    # ---------------- 逐页
    pages: List[Dict[str, Any]] = []
    for p in met['fix']['pages']:
        pid = p['page_id']
        n_ann = p['counts']['annotations']
        rec = {'page_id': pid, 'annotations': n_ann,
               'coverage': p['coverage'],
               'rule_state': None, 'rule_counts': None,
               'preds': {s: len(by_page[s].get(pid, [])) for s in SNAPS},
               'hits50': {}, 'hits75': {},
               'negFP': {}, 'candidates': {}}
        m = masks.get(pid)
        if m:
            cnt = m.get('counts') or {}
            rec['rule_counts'] = {k: cnt.get(k) for k in
                                  ('all', 'qualified', 'ambiguous', 'touching', 'segments', 'in_group')}
            rec['rule_state'] = rule_state(cnt)
        for s in SNAPS:
            sp = next((q for q in met[s]['pages'] if q['page_id'] == pid), None)
            d5 = (sp or {}).get('thresholds', {}).get('0.5', {})
            d75 = (sp or {}).get('thresholds', {}).get('0.75', {})
            rec['hits50'][s] = d5.get('group_hits')
            rec['hits75'][s] = d75.get('group_hits')
            rec['negFP'][s] = d5.get('negative_confirmed_fp')
            rec['candidates'][s] = d5.get('new_candidates')
        pages.append(rec)

    # ---------------- 轨级
    lane = {}
    for s in SNAPS:
        o5 = met[s]['scores']['overall']['0.5']
        o75 = met[s]['scores']['overall']['0.75']
        lane[s] = {'0.5': {'counts': o5['lane_counts'], 'denom': o5['lane_denominator'],
                           'excluded': o5['lane_excluded'],
                           'accuracy': o5['pairing_accuracy'],
                           'swap_rate': o5['swap_rate'],
                           'child_failure_rate': o5['child_failure_rate']},
                   '0.75': {'counts': o75['lane_counts'], 'denom': o75['lane_denominator'],
                            'excluded': o75['lane_excluded'],
                            'accuracy': o75['pairing_accuracy']}}

    # ---------------- 负例覆盖
    negatives = []
    for r in neg['regions']:
        item = {'negative_region_id': r['negative_region_id'], 'page_id': r['page_id'],
                'category': r['category'], 'status': r['status'], 'bbox': r['bbox'],
                'cover': {}}
        for s in SNAPS:
            hit = []
            for pp in by_page[s].get(r['page_id'], []):
                c = overlap_ratio(pp['group_bbox'], r['bbox'])
                if c > 0:
                    hit.append({'prediction_id': pp['prediction_id'],
                                'group_bbox': pp['group_bbox'], 'cover': round(c, 3)})
            item['cover'][s] = {'n': len(hit), 'hits': hit}
        negatives.append(item)

    # ---------------- 端点敏感性
    sens = {}
    for s in SNAPS:
        e = met[s]['endpoint_sensitivity']
        sens[s] = {}
        for k in ('nominal', 'inclusive'):
            if k in e:
                o = e[k]['overall']
                sens[s][k] = {tk: {'hits': o[tk]['group_hits'],
                                   'negfp': o[tk]['negative_confirmed_fp']} for tk in o}

    # ---------------- 旧版落入单列负例的候选组
    legacy = []
    for gid in ('丹方彚编/page_12#g002', '丹方彚编/page_12#g014',
                '丹方彚编/page_12#g024', '丹方彚编/page_12#g026'):
        r = None
        for pg in s6.get('pages', []):
            for rec in pg.get('records') or []:
                if rec.get('group_id') == gid:
                    r = rec
        if r is None:
            continue
        b = r['after']['group_bbox']
        after = []
        for pp in by_page['fix'].get('丹方彚编/page_12', []):
            c = overlap_ratio(pp['group_bbox'], b)
            if c > 0.2:
                after.append({'prediction_id': pp['prediction_id'],
                              'group_bbox': pp['group_bbox'], 'cover': round(c, 3),
                              'in_negative': any(overlap_ratio(pp['group_bbox'], nr['bbox']) > 0
                                                 for nr in neg['regions']
                                                 if nr['page_id'] == '丹方彚编/page_12')})
        legacy.append({'legacy_group_id': gid, 'legacy_bbox': b,
                       'legacy_in_negative_ratio': round(
                           overlap_ratio(b, [536, 210, 585, 815]), 3),
                       'fix_overlapping': after})

    # ---------------- 冻结布局标签
    wide = Counter()
    for g in ann['groups']:
        wide[((g.get('labels') or {}).get('wide_box') or {}).get('state') or 'unset'] += 1

    out = {
        'generated_at': None,
        'snapshots': {s: {
            'n_predictions': met[s]['n_predictions'],
            'content_sha256': met[s]['predictions_content_sha256'][:16],
            'overall': {tk: {k: met[s]['scores']['overall'][tk][k] for k in
                             ('group_hits', 'annotations_total', 'tp', 'fp', 'fn',
                              'negative_confirmed_fp', 'negative_suspected_fp',
                              'duplicate_extra_fp', 'new_candidates', 'invalid_predictions',
                              'lane_denominator', 'lane_excluded', 'pairing_accuracy')}
                        for tk in ('0.5', '0.75')},
            'by_split': {k: {tk: {'hits': v['thresholds'][tk]['group_hits'],
                                  'total': v['thresholds'][tk]['annotations_total']}
                             for tk in ('0.5', '0.75')}
                         for k, v in met[s]['scores']['by_split'].items()},
        } for s in SNAPS},
        'groups': rows, 'pages': pages, 'lane': lane,
        'negatives': negatives, 'endpoint_sensitivity': sens,
        'legacy_single_track_candidates': legacy,
        'wide_box_labels': dict(wide),
        'rule_state_by_page': {k: v for k, v in
                               ((p['page_id'], p['rule_state']) for p in pages)},
    }
    (S8 / 'report_data.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                                         encoding='utf-8')
    print('wrote', S8 / 'report_data.json')

    # 控制台摘要
    print('\n分组命中（@0.5 / @0.75）')
    for s in SNAPS:
        o = out['snapshots'][s]['overall']
        print(f"  {s:10s} {o['0.5']['group_hits']:2d}/38  {o['0.75']['group_hits']:2d}/38  "
              f"negFP {o['0.5']['negative_confirmed_fp']}  候选 {o['0.5']['new_candidates']}")
    print('\n负例（修正版覆盖）')
    for n in negatives:
        c = n['cover']['fix']
        print(f"  {n['negative_region_id']:18s} {n['category']:13s} C={n['cover']['C']['n']} "
              f"s7={n['cover']['stage7v2']['n']} fix={c['n']}")
    print('\n旧版 4 个候选组在修正版的重叠预测')
    for l in legacy:
        print(f"  {l['legacy_group_id'].split('#')[-1]} bbox={l['legacy_bbox']} "
              f"负例占比={l['legacy_in_negative_ratio']} -> {len(l['fix_overlapping'])} 条")
    return 0


def rule_state(cnt: Dict[str, Any]) -> str:
    """栏线状态四分类（完整 / 部分或断裂 / 无栏线 / 待核）。"""
    all_ = cnt.get('all') or 0
    qual = cnt.get('qualified') or 0
    amb = cnt.get('ambiguous') or 0
    touch = cnt.get('touching') or 0
    if all_ == 0:
        return 'none'
    if amb and not qual:
        return 'pending'
    if qual and qual == all_:
        return 'complete'
    if qual or touch:
        return 'partial_or_broken'
    return 'none'


if __name__ == '__main__':
    raise SystemExit(main())
