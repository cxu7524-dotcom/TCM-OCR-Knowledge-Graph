# -*- coding: utf-8 -*-
"""第五阶段批量运行：19 页 → 组、轨框、中缝轨迹、断点审计。

输入只走 `dual_track_track.inputs`（内部只调 `dual_track_search` 公开读取入口）。
产物：
  output/dual_tracks/stage5/groups.json   组 + 逐页审计 + 运行清单
  output/dual_tracks/stage5/windows.json  逐窗审计（仅含有候选段的区域）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from ocr.vertical import imread_unicode                       # noqa: E402

from dual_track_track import DEFAULT_CONFIG, detect_page, load_bundle, page_ink  # noqa: E402
from dual_track_track.inputs import S5, canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage5：双轨检测与追踪')
    ap.add_argument('--out-dir', default=str(S5))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT_CONFIG
    b = load_bundle()
    pages = b.pages()
    if args.only:
        keep = set(args.only)
        pages = [p for p in pages if p.page_id in keep]

    rows, windows = [], {}
    t0 = time.time()
    for i, pg in enumerate(pages, 1):
        img = imread_unicode(str(ROOT / pg.image_path))
        if img is None:
            rows.append({'page_id': pg.page_id, 'status': 'failed',
                         'status_reason': 'image_unreadable', 'groups': []})
            continue
        ink = page_ink(img, b.contract)
        rec = detect_page(pg, ink, cfg, b.contract)
        rows.append(rec)
        wins = {}
        for rid, ws in rec['windows'].items():
            n_valid = sum(1 for w in ws if w['valid'])
            if not n_valid:
                continue
            wins[rid] = [_slim_window(w) for w in ws]
        if wins:
            windows[pg.page_id] = wins
        rec.pop('windows', None)
        if not args.quiet:
            print(f"  {i}/{len(pages)} {pg.page_id:22s} groups={rec['group_count']:2d} "
                  f"t={rec['timing_s']}s", flush=True)

    content = [{k: v for k, v in r.items() if k != 'timing_s'} for r in rows]
    payload = {
        'stage': 'stage5_dual_track_tracking',
        'release': cfg['version'],
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage5_run.py',
        'config': cfg,
        'inputs': b.fingerprints,
        'scope': {'pages': len(rows),
                  'note': '与阶段 3/4 相同的 19 页（标注页 ∪ 负例页）'},
        'summary': _summary(rows),
        'pages': rows,
        'determinism': {'content_sha256': canon_sha256(content),
                        'hash_rule': 'pages 去掉 timing_s 后规范化 JSON 再 sha256'},
        'timing': {'total_s': round(time.time() - t0, 2)},
    }
    (out / 'groups.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
                                     encoding='utf-8')
    (out / 'windows.json').write_text(
        json.dumps({'stage': 'stage5_window_audit', 'release': cfg['version'],
                    'pages': windows}, ensure_ascii=False, indent=0) + '\n',
        encoding='utf-8')
    if not args.quiet:
        s = payload['summary']
        print(f"页 {s['pages']}｜组 {s['groups']}｜有组页 {s['pages_with_groups']}")
        print(f"断点 {s['stop_events']}")
        print(f"content_sha256 {payload['determinism']['content_sha256'][:16]}")
        print(f"用时 {payload['timing']['total_s']}s → {out}")
    return 0


def _slim_window(w: Dict[str, Any]) -> Dict[str, Any]:
    keep = ('window_id', 'candidate_seam', 'index', 'y', 'state', 'valid', 'seam_x',
            'L', 'R', 'V', 'r', 'left_support', 'right_support',
            'left_glyphs', 'right_glyphs', 'left_bbox', 'right_bbox', 'reason',
            'blank_run_px', 'big_band', 'alternatives', 'best_rejected')
    out = {k: w.get(k) for k in keep if k in w}
    if out.get('big_band'):
        out['big_band'] = {k: out['big_band'][k] for k in ('kind', 'y')}
    return out


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stops: Dict[str, int] = {}
    pend = 0
    by_src: Dict[str, int] = {}
    for r in rows:
        for g in r.get('groups') or []:
            pend += 1 if g.get('pending') else 0
            by_src[g.get('region_source')] = by_src.get(g.get('region_source'), 0) + 1
            for e in g.get('stop_events') or []:
                stops[e['type']] = stops.get(e['type'], 0) + 1
    return {'pages': len(rows),
            'groups': sum(r.get('group_count', 0) for r in rows),
            'pages_with_groups': sum(1 for r in rows if r.get('group_count')),
            'pending_groups': pend,
            'by_region_source': dict(sorted(by_src.items())),
            'stop_events': dict(sorted(stops.items())),
            'skipped_regions': sum(1 for r in rows for x in r.get('regions') or []
                                   if x.get('skipped')),
            'degraded_regions': sum(1 for r in rows for x in r.get('regions') or []
                                    if x.get('degraded'))}


if __name__ == '__main__':
    raise SystemExit(main())
