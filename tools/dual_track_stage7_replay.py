# -*- coding: utf-8 -*-
"""stage7：在线框选 vs 冻结回放，逐步定位差异发生在哪一步。

三步依次比：
  S1 搜索区域   live: stage4 plan_page 现算      frozen: stage4 search_regions.json
  S2 成对确认   live: 阶段5 detect_page          frozen: stage5 groups.json（v4）
  S3 分侧精修   live: 阶段6 refine_group         frozen: stage6 groups.json（v4）

差异一律给出「发生在哪一步 + 对应输入/输出」，不做模糊归因。
输出：output/dual_tracks/stage7/replay_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

TZ = timezone(timedelta(hours=8))
S5 = ROOT / 'output' / 'dual_tracks' / 'stage5'
S6 = ROOT / 'output' / 'dual_tracks' / 'stage6'
OUT = ROOT / 'output' / 'dual_tracks' / 'stage7'


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def _norm_reason(v: Any) -> Any:
    return '' if v is None else v


def compare(page_id: str, image_path: str) -> Dict[str, Any]:
    from ocr.dual_track import (_contract, _frozen_context, _live_context,
                                _neighbor_bounds, _sha256_file)
    from ocr.vertical import imread_unicode
    from dual_track_track import DEFAULT_CONFIG as S5_CFG, PageInput, detect_page, page_ink
    from dual_track_track import rule_segments_for_region
    from dual_track_refine import DEFAULT_CONFIG as S6_CFG, refine_group

    img = imread_unicode(image_path)
    assert img is not None, image_path
    h, w = img.shape[:2]
    live = _live_context(img, image_path, page_id)
    froz = _frozen_context(page_id, img)

    # ---------------- S1 搜索区域
    lr = {r['region_id']: list(r['bbox']) for r in live['regions']}
    fr = {r['region_id']: list(r['bbox']) for r in froz['regions']}
    s1 = {'live_regions': len(lr), 'frozen_regions': len(fr),
          'ids_equal': sorted(lr) == sorted(fr),
          'bbox_diffs': [{'region_id': k, 'live': lr[k], 'frozen': fr[k]}
                         for k in sorted(set(lr) & set(fr)) if lr[k] != fr[k]],
          'live_only': sorted(set(lr) - set(fr)),
          'frozen_only': sorted(set(fr) - set(lr)),
          'scale_live': live['page_scale'].get('H'),
          'scale_frozen': froz['page_scale'].get('H')}
    s1['equal'] = (s1['ids_equal'] and not s1['bbox_diffs']
                   and not s1['live_only'] and not s1['frozen_only']
                   and s1['scale_live'] == s1['scale_frozen'])

    # ---------------- S2 成对确认（两条路径各自跑阶段5）
    def _detect(ctx) -> List[Dict[str, Any]]:
        page = PageInput(page_id=page_id, image_path=image_path,
                         image_sha256=_sha256_file(Path(image_path)),
                         image_size=[w, h], halves=ctx['halves'],
                         regions=ctx['regions'], page_scale=ctx['page_scale'],
                         rule_lines=ctx['rule_lines'], audit={})
        contract = _contract()
        return detect_page(page, page_ink(img, contract), S5_CFG, contract)['groups']

    g_live = _detect(live)
    g_froz = _detect(froz)
    frozen_s5 = {}
    try:
        payload5 = _load(S5 / 'groups.json')
        page5 = next((p for p in payload5['pages'] if p['page_id'] == page_id), None)
        frozen_s5 = {g['group_id']: g for g in (page5 or {}).get('groups') or []}
    except (OSError, ValueError):
        frozen_s5 = {}
    lmap = {g['group_id']: g for g in g_live}
    fmap = {g['group_id']: g for g in g_froz}
    gid_diff = [{'group_id': k, 'live': lmap[k]['group_bbox'], 'frozen': fmap[k]['group_bbox']}
                for k in sorted(set(lmap) & set(fmap)) if lmap[k]['group_bbox'] != fmap[k]['group_bbox']]
    s5_diff = [{'group_id': k, 'replay': fmap[k]['group_bbox'],
                'stage5_v4': frozen_s5[k]['group_bbox']}
               for k in sorted(set(fmap) & set(frozen_s5))
               if fmap[k]['group_bbox'] != frozen_s5[k]['group_bbox']]
    s2 = {'replay_groups': len(g_froz), 'live_groups': len(g_live),
          'ids_equal': sorted(lmap) == sorted(fmap),
          'bbox_diffs': gid_diff,
          'stage5_v4_groups': len(frozen_s5),
          'replay_vs_stage5_v4': s5_diff,
          'live_only': sorted(set(lmap) - set(fmap)),
          'frozen_only': sorted(set(fmap) - set(lmap))}
    s2['equal'] = (s2['ids_equal'] and not gid_diff and not s5_diff
                   and not s2['live_only'] and not s2['frozen_only'])

    # ---------------- S3 分侧精修
    def _refine(ctx, groups) -> Dict[str, Dict[str, Any]]:
        contract = _contract()
        ink = page_ink(img, contract)
        region_by_id = {r['region_id']: r for r in ctx['regions']}
        out: Dict[str, Dict[str, Any]] = {}
        for g in sorted(groups, key=lambda x: (x['group_bbox'] or [0, 0, 0, 0])[1]):
            reg = region_by_id[g['region_id']]
            segs = rule_segments_for_region(ctx['rule_lines'], reg['bbox'])
            nb = _neighbor_bounds(g, groups, reg['bbox'])
            rec = refine_group(g, reg['bbox'], ink, segs, S6_CFG, nb)
            out[g['group_id']] = rec
        return out

    r_live = _refine(live, g_live)
    r_froz = _refine(froz, g_froz)
    frozen_s6 = {}
    try:
        payload6 = _load(S6 / 'groups.json')
        page6 = next((p for p in payload6['pages'] if p['page_id'] == page_id), None)
        frozen_s6 = {r['group_id']: r for r in (page6 or {}).get('records') or []}
    except (OSError, ValueError):
        frozen_s6 = {}

    def _side_view(rec: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for side in ('right', 'left'):
            s = (rec.get('sides') or {}).get(side) or {}
            fv = s.get('final_validation') or {}
            out[side] = {'bbox': s.get('bbox_after'), 'ok': bool(fv.get('ok')),
                         'status': fv.get('status'), 'reason': _norm_reason(fv.get('reason')),
                         'flags': list(s.get('flags') or []),
                         'accepted': len(s.get('accepted_glyphs') or [])}
        return out

    side_diffs = []
    for gid in sorted(set(r_live) & set(r_froz)):
        a, b = _side_view(r_live[gid]), _side_view(r_froz[gid])
        if a != b:
            side_diffs.append({'group_id': gid, 'live': a, 'replay': b})
    vs_frozen6 = []
    for gid in sorted(set(r_froz) & set(frozen_s6)):
        a = _side_view(r_froz[gid])
        b = {side: {'bbox': (frozen_s6[gid]['sides'].get(side) or {}).get('bbox_after'),
                    'ok': bool(((frozen_s6[gid]['sides'].get(side) or {})
                                .get('final_validation') or {}).get('ok')),
                    'status': ((frozen_s6[gid]['sides'].get(side) or {})
                               .get('final_validation') or {}).get('status'),
                    'reason': _norm_reason(((frozen_s6[gid]['sides'].get(side) or {})
                                            .get('final_validation') or {}).get('reason')),
                    'flags': list((frozen_s6[gid]['sides'].get(side) or {}).get('flags') or []),
                    'accepted': len((frozen_s6[gid]['sides'].get(side) or {})
                                    .get('accepted_glyphs') or [])}
              for side in ('right', 'left')}
        if a != b:
            vs_frozen6.append({'group_id': gid, 'replay': a, 'stage6_v4': b})
    s3 = {'groups_compared': len(set(r_live) & set(r_froz)),
          'side_diffs': side_diffs, 'replay_vs_stage6_v4': vs_frozen6,
          'stage6_v4_groups': len(frozen_s6)}
    s3['equal'] = not side_diffs and not vs_frozen6

    steps = []
    if not s1['equal']:
        steps.append('S1 搜索区域')
    if not s2['equal']:
        steps.append('S2 成对确认')
    if not s3['equal']:
        steps.append('S3 分侧精修')
    return {
        'page_id': page_id, 'image_path': image_path,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'S1_region': s1, 'S2_pairing': s2, 'S3_refine': s3,
        'first_diff_step': steps[0] if steps else None,
        'different_steps': steps,
        'verdict': ('在线框选与冻结回放逐步一致（区域 / 组 / 分侧框与状态全同）'
                    if not steps else f'差异出现在 {steps[0]}'),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage7：在线 vs 冻结回放逐步对比')
    ap.add_argument('--page-id', default='丹方彚编/page_12')
    ap.add_argument('--image', default='古籍评测集/医家类/丹方彚编/page_12.png')
    ap.add_argument('--out', default=str(OUT / 'replay_compare.json'))
    args = ap.parse_args(argv)
    image = str(ROOT / args.image) if not Path(args.image).is_absolute() else args.image
    res = compare(args.page_id, image)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('page_id', 'first_diff_step', 'different_steps',
                                          'verdict')}, ensure_ascii=False, indent=1))
    for k in ('S1_region', 'S2_pairing', 'S3_refine'):
        v = res[k]
        print(k, 'equal=', v['equal'],
              {kk: vv for kk, vv in v.items() if kk not in ('equal', 'bbox_diffs',
                                                            'side_diffs', 'live_only',
                                                            'frozen_only', 'replay_vs_stage5_v4',
                                                            'replay_vs_stage6_v4')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
