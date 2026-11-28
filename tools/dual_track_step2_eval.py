# -*- coding: utf-8 -*-
"""第二步 19 页正常入口评估与基线对照脚本。

所有产物写入 output/dual_tracks/stage8/step2_fix/，不覆盖 closure/ 基线。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_fix.settings import effective_options  # noqa: E402
from ocr.pipeline import run_ocr  # noqa: E402
from tools.dual_track_stage8_online import (  # noqa: E402
    _image_path,
    _pages,
    evidence_to_predictions,
)
from tools.dual_track_stage8_accept import IMG as P12_IMG, edge_ink, ink_mask  # noqa: E402
from tools.dual_track_stage8_ink_review import review as run_ink_review  # noqa: E402

OUT = ROOT / 'output/dual_tracks/stage8/step2_fix'
BASELINE_PRED = ROOT / 'output/dual_tracks/stage8/closure/predictions/path_online_closure.json'
S2 = ROOT / 'output/dual_tracks/stage2'
TZ = timezone(timedelta(hours=8))


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def run_all() -> int:
    pred_dir = OUT / 'predictions'
    crop_dir = OUT / 'crops'
    acc_dir = OUT / 'acceptance'
    m_frozen = OUT / 'metrics/neg_frozen'
    m_v2 = OUT / 'metrics/neg_v2'
    for d in (pred_dir, crop_dir, acc_dir, m_frozen, m_v2):
        d.mkdir(parents=True, exist_ok=True)

    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    os.environ['TCM_OCR_DUAL_TRACK_MODE'] = 'live'
    os.environ['TCM_OCR_DUAL_TRACK_DETECTOR'] = 'stage8'
    os.environ['TCM_OCR_DUAL_TRACK_CROP_DIR'] = str(crop_dir)
    import app.config as C
    from importlib import reload
    reload(C)
    import ocr.pipeline as PL
    reload(PL)
    eff = effective_options({'detector': C.CONFIG.ocr.dual_track_detector})

    pages = _pages()
    preds: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    t0 = time.time()
    for pid in pages:
        img_p = _image_path(pid)
        t1 = time.time()
        res = PL.run_ocr(str(img_p), page_id=pid)
        ev = list(res.dual_track_evidence or [])
        plan_eff = None
        cand_audit = []
        for a in (res.dual_track_audit or []):
            if not isinstance(a, dict):
                continue
            if a.get('kind') == 'plan' and a.get('effective'):
                plan_eff = a['effective']
            if a.get('kind') in ('window_superseded', 'cross_slot_joint_superseded',
                                 'insufficient_pair_shape_support', 'rule_attached_pair_recovered'):
                cand_audit.append(a)
        runs.append({
            'page_id': pid,
            'status': res.dual_track_status,
            'effective': plan_eff or eff,
            'n_evidence': len(ev),
            'n_groups': len({e.get('group_id') for e in ev if e.get('group_id')}),
            'n_side_ok': sum(1 for e in ev if e.get('side_status') == 'ok'),
            'candidate_audit': cand_audit,
            'elapsed_s': round(time.time() - t1, 2),
        })
        preds.extend(evidence_to_predictions(pid, ev, source='ONLINE'))

    pred_path = pred_dir / 'path_online_step2.json'
    pred_doc = {
        'path': 'ONLINE',
        'entry': 'ocr.pipeline.run_ocr(detector=stage8)',
        'git_head': _git_head(),
        'effective': eff,
        'page_runs': runs,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'timing_s': round(time.time() - t0, 2),
        'predictions': preds,
    }
    pred_path.write_text(json.dumps(pred_doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # Run stage3_score on both frozen & v2 negative calibers
    py = sys.executable
    for cal_name, neg_file, out_m in [
        ('frozen', S2 / 'negative_regions.json', m_frozen),
        ('v2', ROOT / 'output/dual_tracks/stage8/negative_regions_v2.json', m_v2),
    ]:
        subprocess.check_call([
            py, str(ROOT / 'tools/dual_track_stage3_score.py'),
            '--file', str(pred_path),
            '--neg', str(neg_file),
            '--out-dir', str(out_m),
        ])

    # Run A4 on p12
    p12_img = cv2.imdecode(np.fromfile(str(P12_IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY))
    H = 21.0
    p12_preds = [p for p in preds if p.get('page_id') == '丹方彚编/page_12']
    checked_items, cut_items = [], []
    for p in p12_preds:
        for side in ('right', 'left'):
            b = p.get(f'{side}_bbox')
            if not b:
                continue
            if b[3] - b[1] < 2.5 * H:
                continue
            ov = edge_ink(ink, b)
            br = (p.get('boundary_review') or {}).get(side) or {}
            row = {
                'evidence_id': f"{p['prediction_id']}:{side}",
                'group_id': p['prediction_id'],
                'side': side,
                'bbox': b,
                'top': ov['top'],
                'bottom': ov['bottom'],
                'side_status': (p.get('side_status') or {}).get(side),
                'status_reason': (p.get('status_reason') or {}).get(side),
                'review_actions': br.get('actions') or [],
                'verdict': 'cut' if (ov['top'] >= 0.5 or ov['bottom'] >= 0.5) else 'ok',
            }
            checked_items.append(row)
            if row['verdict'] == 'cut':
                cut_items.append(row)
    a4_doc = {
        'prediction_source': str(pred_path),
        'prediction_sha256': _sha256(pred_path),
        'n_checked': len(checked_items),
        'n_cut': len(cut_items),
        'checked_items': checked_items,
        'cut_items': cut_items,
    }
    a4_path = acc_dir / 'a4_detail.json'
    a4_path.write_text(json.dumps(a4_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # Compare against baseline page by page
    base_doc = json.loads(BASELINE_PRED.read_text(encoding='utf-8'))
    base_by_page: Dict[str, Dict[str, Any]] = {}
    for p in base_doc['predictions']:
        base_by_page.setdefault(p['page_id'], {})[p['prediction_id']] = p
    new_by_page: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        new_by_page.setdefault(p['page_id'], {})[p['prediction_id']] = p

    page_diffs = []
    for pid in pages:
        bm = base_by_page.get(pid, {})
        nm = new_by_page.get(pid, {})
        removed = sorted(set(bm) - set(nm))
        added = sorted(set(nm) - set(bm))
        changed = []
        for gid in sorted(set(bm) & set(nm)):
            if (bm[gid].get('group_bbox') != nm[gid].get('group_bbox')
                    or bm[gid].get('right_bbox') != nm[gid].get('right_bbox')
                    or bm[gid].get('left_bbox') != nm[gid].get('left_bbox')):
                changed.append({
                    'group_id': gid,
                    'before': {'R': bm[gid].get('right_bbox'), 'L': bm[gid].get('left_bbox')},
                    'after': {'R': nm[gid].get('right_bbox'), 'L': nm[gid].get('left_bbox')},
                })
        page_diffs.append({
            'page_id': pid,
            'baseline_groups': len(bm),
            'step2_groups': len(nm),
            'removed': removed,
            'added': added,
            'changed': changed,
        })

    # Render p12 overview
    ov_img = p12_img[180:850, 520:1010].copy()
    for p in p12_preds:
        for side, col in [('right', (0, 0, 220)), ('left', (220, 120, 0))]:
            b = p.get(f'{side}_bbox')
            if b:
                cv2.rectangle(ov_img, (b[0] - 520, b[1] - 180), (b[2] - 520, b[3] - 180), col, 1)
    cv2.imencode('.png', ov_img)[1].tofile(str(acc_dir / 'p12_step2_overview.png'))

    comp_doc = {
        'baseline_prediction': str(BASELINE_PRED),
        'baseline_sha256': _sha256(BASELINE_PRED),
        'step2_prediction': str(pred_path),
        'step2_sha256': _sha256(pred_path),
        'effective_fingerprint': eff['fingerprint'],
        'git_head': _git_head(),
        'p12_a4_checked': len(checked_items),
        'p12_a4_cut': len(cut_items),
        'page_diffs': page_diffs,
    }
    (OUT / 'step2_comparison.json').write_text(
        json.dumps(comp_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Step2 pred sha256:', _sha256(pred_path))
    print(f'p12 A4: {len(cut_items)} / {len(checked_items)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(run_all())
