# -*- coding: utf-8 -*-
"""stage8 修正版跑批：列槽切分 + 旧列内检测 → 冻结格式预测文件。

v2（2026-11-25）：参数不再由命令行开关直接拼，而是
  `dual_track_fix.settings.effective_options()`（与正常 OCR 入口同一份来源）
+ 少量 `--set-*` 覆盖（只用于消融对照）。
每页写 `effective`（含配置指纹 fingerprint），供离线/在线逐页对账。

用法：
  python tools/dual_track_stage8_run.py --tag fix
输出：
  output/dual_tracks/stage8/predictions/path_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_fix.settings import detect_options, effective_options  # noqa: E402
from ocr.vertical import imread_unicode  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
OUT = ROOT / 'output' / 'dual_tracks' / 'stage8'
TZ = timezone(timedelta(hours=8))


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='fix')
    ap.add_argument('--source', default='FIX')
    # 默认一律读「同一份有效配置」；下面这些开关只用于消融对照（关掉某一个动作）。
    ap.add_argument('--no-slot-split', action='store_true', help='消融：关闭列槽切分')
    ap.add_argument('--no-clip', action='store_true', help='消融：子框不裁到列槽')
    ap.add_argument('--no-tighten-lanes', action='store_true', help='消融：子框不按墨迹收紧 x')
    ap.add_argument('--nolim', action='store_true', help='消融：不把列槽 x 限位传给检测')
    ap.add_argument('--no-relax-span', action='store_true', help='消融：跨度上限回到 150px')
    ap.add_argument('--dedupe', action='store_true', help='消融：开启同槽去重（负结果）')
    ap.add_argument('--band-slots', action='store_true',
                    help='实验：上下分栏各自定相位切槽（p12 下半栏右移的成因）')
    ap.add_argument('--set', action='append', default=[], metavar='K=V',
                    help='直接覆盖任一配置项（消融/复现用）')
    args = ap.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.no_slot_split:
        overrides['slots'] = False
    if args.no_clip:
        overrides['clip'] = False
    if args.no_tighten_lanes:
        overrides['tighten'] = False
    if args.nolim:
        overrides['use_limits'] = False
    if args.no_relax_span:
        overrides['relax_span'] = False
    if args.dedupe:
        overrides['dedupe_slot'] = True
    if getattr(args, 'band_slots', False):
        overrides['band_slots'] = True
    for kv in args.set:
        k, _, v = kv.partition('=')
        overrides[k.strip()] = v.strip()
    overrides.setdefault('detector', 'stage8')      # 本脚本就是 stage8 跑批
    eff = effective_options(overrides)

    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    sr = load(S4 / 'search_regions.json')
    meta = {p['page_id']: p for p in sr['pages']}

    pages = {g['page_id']: g['image_path'] for g in ann['groups']}
    for r in neg['regions']:
        pages.setdefault(r['page_id'], r['image_path'])
    order = list(pages)

    from dual_track_fix.detect import detect as detect_stage8

    preds: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for pid in order:
        t0 = time.time()
        img = imread_unicode(str(ROOT / pages[pid]))
        if img is None:
            runs.append({'page_id': pid, 'status': 'failed', 'error': 'image_unreadable',
                         'effective': eff})
            continue
        H = float((meta[pid].get('page_scale') or {}).get('H') or 20)
        d8 = detect_stage8(img, pid, H, options=detect_options(eff))
        cols = d8['regions']
        audit = d8['slot_audit']
        for g in d8['groups']:
            preds.append({
                'page_id': pid, 'prediction_id': f"{g['group_id']}",
                'group_bbox': g['group_bbox'], 'right_bbox': g['right_bbox'],
                'left_bbox': g['left_bbox'], 'main_bbox': g.get('main_bbox'),
                'parent_bbox': g.get('region_bbox'), 'slot_bbox': g.get('region_bbox'),
                'region_ref': {'kind': 'slot', 'region_id': g['region_id'],
                               'bbox': g.get('region_bbox')},
                'source': args.source, 'run_status': 'ok', 'error': '',
                'evidence': {k: g['evidence'].get(k) for k in
                             ('windows', 'valley_x', 'min_valley_ratio', 'core_y')},
                'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框'})
        runs.append({'page_id': pid, 'status': 'ok',
                     'effective': eff,
                     'H': H,
                     'span_ratio_used': d8.get('span_ratio'),
                     'n_columns': len(cols),
                     'n_slots_split': sum(1 for a in audit if a.get('action') == 'split'),
                     'n_predictions': len(d8['groups']),
                     'joint_candidates': d8.get('joint_candidates') or [],
                     'elapsed_s': round(time.time() - t0, 2)})
        audits.append({'page_id': pid, 'H': H, 'columns': len(cols),
                       'effective': eff, 'slot_audit': audit})
        print(f"{pid:22s} H={H:4.1f} cols={len(cols):2d} "
              f"split={sum(1 for a in audit if a.get('action')=='split')} "
              f"preds={len(d8['groups']):2d}")

    doc = {'path': args.source, 'entry': 'layout_propose_page+slot_split',
           'effective': eff, 'effective_fingerprint': eff['fingerprint'],
           'effective_by_page': {r['page_id']: r.get('effective') for r in runs},
           'predictions': preds, 'page_runs': runs,
           'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
           'timing_s': round(sum(r.get('elapsed_s') or 0 for r in runs), 2)}
    d = OUT / 'predictions'
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'path_{args.tag}.json'
    f.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (OUT / f'slot_audit_{args.tag}.json').write_text(
        json.dumps({'effective': eff, 'pages': audits}, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8')
    print('effective', eff['fingerprint'], eff['options'], 'span=', eff['effective_span'])
    print('preds', len(preds), '->', f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
