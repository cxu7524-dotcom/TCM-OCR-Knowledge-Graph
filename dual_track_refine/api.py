# -*- coding: utf-8 -*-
"""阶段 6 编排与读取入口：只读 stage5-v4 分组，输出精修框与审计。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEFAULT_CONFIG
from .refine import refine_group

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys  # noqa: E402
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))

from ocr.vertical import imread_unicode  # noqa: E402
from dual_track_track import load_bundle, page_ink, rule_segments_for_region  # noqa: E402
from dual_track_track.inputs import canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))
S5 = ROOT / 'output' / 'dual_tracks' / 'stage5'
S6 = ROOT / 'output' / 'dual_tracks' / 'stage6'


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def refine_all(out_dir: Path = S6, only: Optional[List[str]] = None,
               cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or DEFAULT_CONFIG
    stage5_path = S5 / 'groups.json'
    payload = load(stage5_path)
    bundle = load_bundle()
    bundle.pages()
    pages5 = payload['pages']
    if only:
        keep = set(only)
        pages5 = [p for p in pages5 if p['page_id'] in keep]
    bundle_pages = {p.page_id: p for p in bundle.pages()}
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for p5 in pages5:
        pid = p5['page_id']
        bp = bundle_pages.get(pid)
        if bp is None:
            continue
        img = imread_unicode(str(ROOT / bp.image_path))
        if img is None:
            continue
        ink = page_ink(img, bundle.contract)
        region_by_id = {r['region_id']: r for r in bp.regions}
        records = []
        groups = sorted(p5.get('groups') or [], key=lambda x: x['group_bbox'][1])
        for gi, g in enumerate(groups):
            reg = region_by_id.get(g['region_id'])
            if reg is None:
                continue
            segs = rule_segments_for_region(bp.rule_lines, reg['bbox'])
            gb = g['group_bbox']
            head_y, tail_y = int(reg['bbox'][1]), int(reg['bbox'][3])
            for other in groups:
                if other is g or other['region_id'] != g['region_id']:
                    continue
                ob = other['group_bbox']
                xo = min(gb[2], ob[2]) - max(gb[0], ob[0])
                if xo <= 0:
                    continue
                if ob[3] <= gb[1]:
                    head_y = max(head_y, int(ob[3]))
                if ob[1] >= gb[3]:
                    tail_y = min(tail_y, int(ob[1]))
            rec = refine_group(g, reg['bbox'], ink, segs, cfg,
                               {'head_y': head_y, 'tail_y': tail_y})
            rec['neighbor_bounds'] = {'head_y': head_y, 'tail_y': tail_y}
            rec['scale_used'] = g.get('scale_used')
            rec['confirmed_span'] = g.get('confirmed_span')
            rec['seam'] = g.get('seam')
            rec['stop_events'] = g.get('stop_events')
            records.append(rec)
        rows.append({'page_id': pid, 'image_path': bp.image_path,
                     'image_sha256': bp.image_sha256,
                     'image_size': bp.image_size, 'records': records,
                     'group_count': len(records)})

    fps = dict(bundle.fingerprints)
    fps['stage5_groups'] = {'path': 'output/dual_tracks/stage5/groups.json',
                            'sha256': sha256_of(stage5_path), 'exists': stage5_path.exists()}
    fps['stage5_content_sha256'] = payload['determinism']['content_sha256']
    content = [{'page_id': r['page_id'], 'records': r['records']} for r in rows]
    out = {
        'stage': 'stage6_dual_track_refinement',
        'release': cfg['version'],
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'dual_track_refine.api.refine_all',
        'config': cfg,
        'inputs': fps,
        'frozen_pairing': {'release': payload.get('release'),
                           'group_ids': [g['group_id'] for p in pages5
                                         for g in p.get('groups') or []]},
        'summary': _summary(rows),
        'pages': rows,
        'determinism': {'content_sha256': canon_sha256(content),
                        'hash_rule': 'pages 去掉运行元信息后规范化 JSON sha256'},
        'timing': {'total_s': round(time.time() - t0, 2)},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'groups.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return out


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    recs = [r for p in rows for r in p['records']]
    flags: Dict[str, int] = {}
    for r in recs:
        for f in r['flags']:
            flags[f] = flags.get(f, 0) + 1
    return {
        'pages': len(rows), 'groups': len(recs),
        'pages_with_groups': sum(1 for p in rows if p['group_count']),
        'changed_groups': sum(1 for r in recs if r['changed']),
        'pending_groups': sum(1 for r in recs if r['pending']),
        'low_confidence_groups': sum(1 for r in recs
                                     if r.get('low_confidence')),
        'flags': dict(sorted(flags.items())),
        'supplement_accepted': sum(len(s['supplement']['head']['accepted'])
                                   + len(s['supplement']['tail']['accepted'])
                                   for r in recs for s in r['sides'].values()),
    }


def refined_predictions(payload: Dict[str, Any], use_before: bool = False
                        ) -> List[Dict[str, Any]]:
    key = 'before' if use_before else 'after'
    out: List[Dict[str, Any]] = []
    for p in payload.get('pages', []):
        for r in p.get('records') or []:
            b = r[key]
            out.append({
                'page_id': r['page_id'], 'prediction_id': r['group_id'],
                'group_bbox': b['group_bbox'],
                'right_bbox': b['right_bbox'], 'left_bbox': b['left_bbox'],
                'parent_bbox': None,
                'region_ref': {'kind': 'stage6_region', 'region_id': r['region_id'],
                               'source': r.get('region_source')},
                'source': 'stage6_refined' if not use_before else 'stage5_tentative',
                'run_status': 'ok', 'error': '',
                'pending': bool(r.get('pending') or r.get('stage5_pending')),
                'stop_events': r.get('stop_events'),
            })
    return out


def groups_for_page(payload: Dict[str, Any], page_id: str) -> List[Dict[str, Any]]:
    for p in payload.get('pages', []):
        if p['page_id'] == page_id:
            return p.get('records') or []
    raise KeyError(f'page not found in stage6 output: {page_id}')


def audit_for_group(payload: Dict[str, Any], group_id: str) -> Optional[Dict[str, Any]]:
    for p in payload.get('pages', []):
        for r in p.get('records') or []:
            if r['group_id'] == group_id:
                return r
    return None
