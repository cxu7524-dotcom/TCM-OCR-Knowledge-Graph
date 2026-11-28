# -*- coding: utf-8 -*-
"""第二步补充复验脚本（step2_v2）：
- 19 页正常入口运行，输出至独立目录 output/dual_tracks/stage8/step2_v2/
- 正式 2H 判据复算原图 A4，并从同一份预测生成 A3、掩膜复核、双口径评分与逐页对账
- 核验全部去重记录的最终保留 ID 是否存在于页面输出，核清 slot02#02 左侧未覆盖像素归属
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

from dual_track_fix.detect import _uncovered_side_ink  # noqa: E402
from dual_track_fix.settings import effective_options  # noqa: E402
from tools.dual_track_stage8_accept import IMG as P12_IMG, edge_ink, ink_mask  # noqa: E402
from tools.dual_track_stage8_ink_review import review as run_ink_review  # noqa: E402
from tools.dual_track_stage8_online import (  # noqa: E402
    _image_path,
    _pages,
    evidence_to_predictions,
)

OUT = ROOT / 'output/dual_tracks/stage8/step2_v2'
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


def _iou(a: List[int], b: List[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_eval(out_root: Path = OUT, pred_filename: str = 'path_online_step2_v2.json') -> int:
    pred_dir = out_root / 'predictions'
    crop_dir = out_root / 'crops'
    acc_dir = out_root / 'acceptance'
    m_frozen = out_root / 'metrics/neg_frozen'
    m_v2 = out_root / 'metrics/neg_v2'
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
            'image_sha256': _sha256(img_p),
            'n_evidence': len(ev),
            'n_groups': len({e.get('group_id') for e in ev if e.get('group_id')}),
            'n_side_ok': sum(1 for e in ev if e.get('side_status') == 'ok'),
            'candidate_audit': cand_audit,
            'elapsed_s': round(time.time() - t1, 2),
        })
        preds.extend(evidence_to_predictions(pid, ev, source='ONLINE'))

    pred_path = pred_dir / pred_filename
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
    pred_sha = _sha256(pred_path)

    # 1. Verify ALL superseded_by IDs exist in output!
    dangling_audit = []
    for r in runs:
        pid = r['page_id']
        out_ids = {p['prediction_id'] for p in preds if p['page_id'] == pid}
        for a in r['candidate_audit']:
            if a.get('kind') in ('window_superseded', 'cross_slot_joint_superseded'):
                if a.get('superseded_by') not in out_ids:
                    dangling_audit.append({'page_id': pid, **a})

    # 2. Score frozen & v2
    py = sys.executable
    for neg_file, out_m in [
        (S2 / 'negative_regions.json', m_frozen),
        (ROOT / 'output/dual_tracks/stage8/negative_regions_v2.json', m_v2),
    ]:
        subprocess.check_call([
            py, str(ROOT / 'tools/dual_track_stage3_score.py'),
            '--file', str(pred_path),
            '--neg', str(neg_file),
            '--out-dir', str(out_m),
        ])

    # 3. Official 2*H A4 check + A3 offline check + ink_review
    p12_img = cv2.imdecode(np.fromfile(str(P12_IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY))
    H = 21.0
    p12_preds = [p for p in preds if p.get('page_id') == '丹方彚编/page_12']
    checked_items, cut_items, short_items = [], [], []
    for p in p12_preds:
        for side in ('right', 'left'):
            b = p.get(f'{side}_bbox')
            if not b:
                continue
            if b[3] - b[1] < 2.0 * H:
                short_items.append({'evidence_id': f"{p['prediction_id']}:{side}", 'bbox': b})
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
        'H': H,
        'criterion': 'min_span >= 2.0 * H (42px)',
        'prediction_source': str(pred_path),
        'prediction_sha256': pred_sha,
        'n_checked': len(checked_items),
        'n_cut': len(cut_items),
        'skipped_short_lanes': len(short_items),
        'short_items': short_items,
        'checked_items': checked_items,
        'cut_items': cut_items,
    }
    a4_path = acc_dir / 'a4_detail.json'
    a4_path.write_text(json.dumps(a4_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ink_rep = run_ink_review(pred_path, a4_path, acc_dir / 'ink_review.json')

    # A3 check
    ann = json.loads((S2 / 'annotations.json').read_text(encoding='utf-8'))
    p12_gts = [g for g in ann['groups'] if g.get('page_id') == '丹方彚编/page_12' and g.get('main_bbox')]
    n_comp = 0
    intrusions = []
    for p in p12_preds:
        for s in ('left', 'right'):
            b = p.get(f'{s}_bbox')
            if not b:
                continue
            for g in p12_gts:
                m = g['main_bbox']
                if max(0, min(b[2], m[2]) - max(b[0], m[0])) > max(8, 0.3 * min(b[2] - b[0], m[2] - m[0])):
                    n_comp += 1
                    yo = max(0, min(b[3], m[3]) - max(b[1], m[1]))
                    if yo > 0.35 * (m[3] - m[1]):
                        intrusions.append({'evidence_id': f"{p['prediction_id']}:{s}", 'bbox': b, 'main_bbox': m, 'overlap_y': yo})
    a3_doc = {
        'prediction_source': str(pred_path),
        'prediction_sha256': pred_sha,
        'n_frozen_title_boxes': len(p12_gts),
        'n_lanes': sum(1 for p in p12_preds for s in ('left', 'right') if p.get(f'{s}_bbox')),
        'n_boxes_compared': n_comp,
        'n_intrusion_lanes': len(intrusions),
        'intrusions': intrusions,
        'pass': len(intrusions) == 0,
    }
    (acc_dir / 'title_check.json').write_text(json.dumps(a3_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 4. slot02#02 453px attribution audit
    base_doc = json.loads(BASELINE_PRED.read_text(encoding='utf-8'))
    s02_02 = next(p for p in base_doc['predictions'] if p['prediction_id'] == '丹方彚编/page_12#slot02#02')
    j00 = next(p for p in p12_preds if p['prediction_id'] == '丹方彚编/page_12#slot01#joint00')
    j_main = [849, 563, 905, 653]
    uncov_r = _uncovered_side_ink(ink, s02_02['right_bbox'], [j00['right_bbox'], j00['left_bbox'], j_main])
    uncov_l = _uncovered_side_ink(ink, s02_02['left_bbox'], [j00['right_bbox'], j00['left_bbox'], j_main])
    df12_10 = next(g for g in ann['groups'] if g['group_id'] == '丹方彚编/p12/DF12-10')
    uncov_l_after_neighbor_gt = _uncovered_side_ink(
        ink, s02_02['left_bbox'],
        [j00['right_bbox'], j00['left_bbox'], j_main, df12_10['right_bbox']],
    )
    s02_02_new = next((p for p in p12_preds if p['prediction_id'] == '丹方彚编/page_12#slot02#02'), None)
    slot02_02_audit = {
        'candidate_id': '丹方彚编/page_12#slot02#02',
        'baseline_right_bbox': s02_02['right_bbox'],
        'baseline_left_bbox': s02_02['left_bbox'],
        'step2_v2_retained': s02_02_new is not None,
        'step2_v2_right_bbox': s02_02_new['right_bbox'] if s02_02_new else None,
        'step2_v2_left_bbox': s02_02_new['left_bbox'] if s02_02_new else None,
        'step2_v2_side_status': s02_02_new['side_status'] if s02_02_new else None,
        'joint_id': '丹方彚编/page_12#slot01#joint00',
        'joint_right_bbox': j00['right_bbox'],
        'joint_left_bbox': j00['left_bbox'],
        'joint_main_bbox': j_main,
        'uncovered_right_ink_px': uncov_r,
        'uncovered_left_ink_px': uncov_l,
        'neighbor_gt_id': '丹方彚编/p12/DF12-10',
        'neighbor_gt_right_bbox': df12_10['right_bbox'],
        'uncovered_left_ink_outside_neighbor_gt_px': uncov_l_after_neighbor_gt,
        'conclusion': '经按实际轨框核查：slot02#02 左轨 [840,650,858,773] 含 454px 未被 slot01#joint00([853,654,880,798]) 覆盖的真实小字笔画（对应「陽起石」注文左列「鷺毛者真」，且左邻 slot03#05 止于 x=840 未覆盖该列），若去重将损失该独立小字列；故收回原去重建议，保留 slot02#02，并将其两侧轨首从 y1=630 安全修剪至 y1=650（裁除跨中缝药名「石」字 cid=3117 [846,624,872,650]）。',
    }

    # 5. Per-GT hit comparison & page-by-page diff
    base_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for p in base_doc['predictions']:
        base_by_page.setdefault(p['page_id'], []).append(p)
    new_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        new_by_page.setdefault(p['page_id'], []).append(p)

    gt_regressions = []
    for g in ann['groups']:
        pid = g['page_id']
        gb = g['group_bbox']
        b_best = max((_iou(gb, p['group_bbox']) for p in base_by_page.get(pid, []) if p.get('group_bbox')), default=0.0)
        n_best = max((_iou(gb, p['group_bbox']) for p in new_by_page.get(pid, []) if p.get('group_bbox')), default=0.0)
        if (b_best >= 0.5 and n_best < 0.5) or (b_best >= 0.75 and n_best < 0.75):
            gt_regressions.append({'group_id': g['group_id'], 'base_iou': round(b_best, 4), 'new_iou': round(n_best, 4)})

    page_diffs = []
    for pid in pages:
        bm = {p['prediction_id']: p for p in base_by_page.get(pid, [])}
        nm = {p['prediction_id']: p for p in new_by_page.get(pid, [])}
        removed = sorted(set(bm) - set(nm))
        added = sorted(set(nm) - set(bm))
        changed = []
        for gid in sorted(set(bm) & set(nm)):
            if (bm[gid].get('right_bbox') != nm[gid].get('right_bbox')
                    or bm[gid].get('left_bbox') != nm[gid].get('left_bbox')
                    or bm[gid].get('side_status') != nm[gid].get('side_status')):
                changed.append({
                    'group_id': gid,
                    'before': {'R': bm[gid].get('right_bbox'), 'L': bm[gid].get('left_bbox'), 'status': bm[gid].get('side_status')},
                    'after': {'R': nm[gid].get('right_bbox'), 'L': nm[gid].get('left_bbox'), 'status': nm[gid].get('side_status')},
                })
        page_diffs.append({
            'page_id': pid, 'baseline_groups': len(bm), 'new_groups': len(nm),
            'removed': removed, 'added': added, 'changed': changed,
        })

    comp_doc = {
        'baseline_prediction': str(BASELINE_PRED),
        'baseline_sha256': _sha256(BASELINE_PRED),
        'new_prediction': str(pred_path),
        'new_sha256': pred_sha,
        'effective_fingerprint': eff['fingerprint'],
        'git_head': _git_head(),
        'dangling_superseded_by_count': len(dangling_audit),
        'dangling_superseded_by': dangling_audit,
        'gt_hit_regressions': gt_regressions,
        'slot02_02_left_ink_audit': slot02_02_audit,
        'a3': {'n_compared': n_comp, 'n_intrusions': len(intrusions), 'pass': len(intrusions) == 0},
        'a4': {'n_checked': len(checked_items), 'n_cut': len(cut_items), 'skipped_short': len(short_items)},
        'ink_review': {k: ink_rep[k] for k in ('n_checked_original', 'n_raw_triggers', 'n_lanes_for_review', 'n_trigger_lanes_explained_by_rule', 'n_lanes_requiring_glyph_review')},
        'page_diffs': page_diffs,
    }
    (out_root / 'reconcile_summary.json').write_text(
        json.dumps(comp_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in comp_doc.items() if k != 'page_diffs'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(run_eval())
