# -*- coding: utf-8 -*-
"""stage6 自检：冻结组 ID、原图坐标、证据字段与评分口径。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_refine.api import S6, load  # noqa: E402
from dual_track_track.inputs import canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(S6 / 'validation_report.json'))
    args = ap.parse_args(argv)
    payload = load(S6 / 'groups.json')
    s5 = load(S6.parent / 'stage5' / 'groups.json')
    metrics = load(S6 / 'metrics.json') if (S6 / 'metrics.json').exists() else {}
    checks: List[Dict[str, Any]] = []

    def add(cid, title, ok, detail):
        checks.append({'id': cid, 'title': title,
                       'status': 'pass' if ok else 'fail', 'detail': detail})

    recs = [r for p in payload['pages'] for r in p['records']]
    s5g = {g['group_id'] for p in s5['pages'] for g in p.get('groups') or []}
    add('S01', '冻结组 ID 与 stage5-v4 一一对应',
        len(recs) == len(s5g) and {r['group_id'] for r in recs} == s5g,
        f'stage6 {len(recs)} / stage5 {len(s5g)}')

    bad = []
    for r in recs:
        if r['before'] != {k: next(g[k] for p in s5['pages'] for g in p['groups']
                                   if g['group_id'] == r['group_id'])
                           for k in ('right_bbox', 'left_bbox', 'group_bbox')}:
            bad.append(r['group_id'])
    add('S02', '精修前框逐组等于冻结 v4 暂定框', not bad, f'异常 {len(bad)}', )

    bad = []
    for r in recs:
        a = r['after']
        rb, lb, gb = a['right_bbox'], a['left_bbox'], a['group_bbox']
        if not (rb and lb and gb):
            bad.append(f'{r["group_id"]}:missing')
        elif rb[0] < lb[0] or rb[1] > lb[3] + 40 or lb[1] > rb[3] + 40:
            bad.append(f'{r["group_id"]}:side_order')
        elif not (gb[0] <= min(rb[0], lb[0]) and gb[2] >= max(rb[2], lb[2])):
            bad.append(f'{r["group_id"]}:union')
        if 'scale_used' not in r or 'region_id' not in r:
            bad.append(f'{r["group_id"]}:trace')
    add('S03', '原图框与侧别可追溯，精修框并集正确', not bad,
        f'{len(recs)} 组；异常 {len(bad)}')

    bad = []
    for r in recs:
        if not r.get('region_source') or r.get('low_confidence') is None:
            bad.append(r['group_id'])
        for side in ('right', 'left'):
            s = r['sides'].get(side)
            if not s or 'edge' not in s or 'supplement' not in s:
                bad.append(f'{r["group_id"]}:{side}:audit')
    add('S04', '区域来源、低置信与逐侧逐边审计齐全', not bad,
        f'异常 {len(bad)}')

    low = [r for r in recs if r['horizontal_low_confidence']
           or any(s['supplement_limited'] for s in r['sides'].values())]
    bad = [r['group_id'] for r in low if not r['flags'] or
           not ('horizontal_low_confidence' in r['flags'] or
                'supplement_limited' in r['flags'])]
    add('S05', '横向低置信与补笔受限均可在产物中定位', not bad,
        f'{len(low)} 组；异常 {len(bad)}')

    # p12 两组冻结一对一命中与标题隔离
    pairs = {}
    for tag in ('v4_tentative', 'stage6_refined'):
        for page in (metrics.get('scores', {}).get(tag, {}).get('by_page') or {}).values():
            for tk in ('0.5', '0.75'):
                for p in page.get(tk, {}).get('pairs') or []:
                    pairs[(tag, tk, p[1])] = (p[0], p[2])
    ids = ('丹方彚编/p12/DF12-03', '丹方彚编/p12/DF12-11')
    ok = all(('stage6_refined', '0.5', gid) in pairs
             and pairs[('stage6_refined', '0.5', gid)][1] >= 0.5 for gid in ids)
    rec_by_id = {r['group_id']: r for r in recs}
    matched = {gid: pairs.get(('stage6_refined', '0.5', gid), (None,))[0]
               for gid in ids}
    ok = ok and all(matched[gid] in rec_by_id for gid in ids)
    if ok:
        ok = rec_by_id[matched[ids[0]]]['after']['group_bbox'][3] <= 536
        ok = ok and rec_by_id[matched[ids[1]]]['after']['group_bbox'][1] >= 655
        ok = ok and all(rec_by_id[matched[gid]]['region_source'] == 'halfpage_fallback'
                        and rec_by_id[matched[gid]]['low_confidence'] for gid in ids)
    add('S06', 'p12 两组一对一命中、标题隔离、来源/低置信保留', ok,
        {gid: pairs.get(('stage6_refined', '0.5', gid)) for gid in ids})

    content = [{'page_id': p['page_id'], 'records': p['records']}
               for p in payload['pages']]
    add('S07', '内容哈希可重算',
        canon_sha256(content) == payload['determinism']['content_sha256'],
        payload['determinism']['content_sha256'][:16])

    add('S08', '冻结评分产物存在且为同一 19 页/38 组',
        metrics.get('scope', {}).get('pages') == 19
        and metrics.get('scope', {}).get('annotations') == 38
        and bool(metrics.get('scores', {}).get('stage6_refined')),
        str(metrics.get('scope')))

    # S09 114 条分侧最终校验状态与冲突证据一致性
    side_rows: List[Dict[str, Any]] = []
    bad = []
    for p in payload['pages']:
        for r in p['records']:
            for side in ('right', 'left'):
                v = r['sides'].get(side)
                if not v:
                    continue
                fc = v.get('final_validation') or {}
                notes = v.get('edge', {}).get('notes') or []
                has_existing = any(
                    'existing_boundary_conflict' in str(n.get('kind', ''))
                    or 'existing_boundary_conflict' in str(n.get('reason', ''))
                    for n in notes) or any(
                        'existing_boundary_conflict' in x
                        for x in v.get('pending_reasons') or [])
                ok = fc.get('ok')
                row = {'group_id': r['group_id'], 'side': side,
                       'final_ok': ok, 'reason': fc.get('reason'),
                       'fallback_to_tentative': fc.get('fallback_to_tentative'),
                       'status': fc.get('status'),
                       'existing_boundary_conflict': has_existing,
                       'horizontal_low_confidence': v.get('horizontal_low_confidence')}
                side_rows.append(row)
                if has_existing and ok is not False:
                    bad.append(f'{r["group_id"]}:{side}:conflict_marked_pass')
                if ok is False and not v.get('horizontal_low_confidence'):
                    bad.append(f'{r["group_id"]}:{side}:failed_without_low_confidence')
                if ok is True and has_existing:
                    bad.append(f'{r["group_id"]}:{side}:pass_with_conflict')
    add('S09', '114 条分侧 final_validation 与冲突/低置信状态一致', not bad,
        f'分侧 {len(side_rows)}；失败 {sum(1 for r in side_rows if r["final_ok"] is False)}；'
        f'冲突 {sum(1 for r in side_rows if r["existing_boundary_conflict"])}；异常 {len(bad)}')

    summary = {'checks': len(checks),
               'pass': sum(1 for c in checks if c['status'] == 'pass'),
               'fail': sum(1 for c in checks if c['status'] == 'fail')}
    out = {'stage': 'stage6_validation', 'release': payload.get('release'),
           'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
           'groups_sha256': sha256_of(S6 / 'groups.json'),
           'summary': summary, 'checks': checks,
           'side_status': side_rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(f"自检 {summary['checks']} 项：通过 {summary['pass']}、失败 {summary['fail']}")
    for c in checks:
        print(f"  [{'PASS' if c['status'] == 'pass' else 'FAIL'}] {c['id']} "
              f"{c['title']} — {c['detail']}")
    return 0 if summary['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
