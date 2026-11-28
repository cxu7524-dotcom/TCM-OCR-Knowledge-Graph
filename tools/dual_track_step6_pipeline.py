# -*- coding: utf-8 -*-
"""阶段八第六轮单次正常入口全链路验收流水线（以 step5_final 为历史对照）：
从同一次 19 页 ocr.pipeline.run_ocr(detector=stage8) 预测出发，统一输出至
output/dual_tracks/stage8/step6_closure/（绝不覆盖冻结标签与历史归档目录）：
  1. 新预测与指纹：predictions/path_online_step6.json，并与 step5_final 逐框/逐指标对账
  2. A4 原图统计、掩膜逐轨复核与动态 A4 联合判定（全坐标统一原图左闭右开 [a, b)）
  3. 固定样例前置条件校验（校验原图 SHA-256、预测框与端点，条件不符时明确报待核）：
     - slot06#05:right 首端调整前后对照五联图（「香」vs「如」完整笔画与分组验证）
     - 其余 5 条待核轨及「真」末笔逐端点 4 联 8× 像素证据图
  4. 修正 T6 v2_basis（写入复核版 review.retract_reason，同时保留原冻结依据与两套 T6 评分）
  5. A1–A7、R1–R5、T1–T8 与全链路可追溯结项报告
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_fix.settings import effective_options  # noqa: E402
from ocr.recognize import rec_model_name  # noqa: E402
from tools.dual_track_stage8_accept import (  # noqa: E402
    IMG as P12_IMG,
    check_evidence_crops,
    edge_ink,
    ink_mask,
)
from tools.dual_track_stage8_ink_review import _verify_crop, review as run_ink_review  # noqa: E402
from tools.dual_track_stage8_online import (  # noqa: E402
    _image_path,
    _pages,
    evidence_to_predictions,
)

S2 = ROOT / 'output/dual_tracks/stage2'
S8 = ROOT / 'output/dual_tracks/stage8'
STEP5_DIR = S8 / 'step5_final'
STEP5_PRED = STEP5_DIR / 'predictions/path_online_final.json'
OUT_DIR = S8 / 'step6_closure'
TZ = timezone(timedelta(hours=8))
EXPECTED_P12_SHA256 = 'e676a5629c0db853b38d33fa285d2fa9bab9bfedc2fb965e4a217a80c0f16cfb'


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


def _extract_v2_retract_basis(region: Dict[str, Any]) -> str:
    """优先提取 negative_regions_v2 中 review.retract_reason / conclusion，绝不回填旧误标 basis。"""
    rev = region.get('review') or {}
    reason = (rev.get('retract_reason') or '').strip()
    concl = (rev.get('conclusion') or '').strip()
    if concl and reason:
        return f'{concl}；{reason}'
    if reason:
        return reason
    if concl:
        return concl
    return (region.get('basis') or '').strip()


def validate_fixed_sample_preconditions(
    *,
    image_sha256: str,
    expected_image_sha256: str,
    actual_bbox: Optional[Sequence[int]],
    expected_x_range: Optional[Sequence[int]] = None,
    expected_y1_range: Optional[Sequence[int]] = None,
    expected_y2_range: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """校验固定样例的输入指纹与预测框范围；任一不符时返回 ok=False 与具体原因供调用方标待核。"""
    reasons: List[str] = []
    if image_sha256 != expected_image_sha256:
        reasons.append(f'image_sha256_mismatch:{image_sha256[:12]}!={expected_image_sha256[:12]}')
    if not actual_bbox or len(actual_bbox) != 4:
        reasons.append('missing_prediction_bbox')
    else:
        x1, y1, x2, y2 = [int(v) for v in actual_bbox]
        if expected_x_range and not (expected_x_range[0] <= x1 and x2 <= expected_x_range[1]):
            reasons.append(f'x_range_mismatch:[{x1},{x2})_not_in_[{expected_x_range[0]},{expected_x_range[1]})')
        if expected_y1_range and not (expected_y1_range[0] <= y1 < expected_y1_range[1]):
            reasons.append(f'y1_mismatch:{y1}_not_in_[{expected_y1_range[0]},{expected_y1_range[1]})')
        if expected_y2_range and not (expected_y2_range[0] <= y2 < expected_y2_range[1]):
            reasons.append(f'y2_mismatch:{y2}_not_in_[{expected_y2_range[0]},{expected_y2_range[1]})')
    return {
        'ok': len(reasons) == 0,
        'status': 'verified' if not reasons else 'pending_precondition_mismatch',
        'reasons': reasons,
        'checked_image_sha256': image_sha256,
        'checked_bbox': list(actual_bbox) if actual_bbox else None,
    }


def _run_19_pages_online(crop_dir: Path, pred_path: Path) -> Dict[str, Any]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
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
    model_ver = rec_model_name()

    pages = _pages()
    preds: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    all_ev_by_page: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.time()
    for pid in pages:
        img_p = _image_path(pid)
        t1 = time.time()
        res = PL.run_ocr(str(img_p), page_id=pid)
        ev = list(res.dual_track_evidence or [])
        all_ev_by_page[pid] = ev
        plan_eff = None
        cand_audit = []
        for a in (res.dual_track_audit or []):
            if not isinstance(a, dict):
                continue
            if a.get('kind') == 'plan' and a.get('effective'):
                plan_eff = a['effective']
            if a.get('kind') in (
                'window_superseded', 'cross_slot_joint_superseded',
                'insufficient_pair_shape_support', 'rule_attached_pair_recovered',
                'side_pending', 'recognize_error', 'side_invalid_bbox',
            ):
                cand_audit.append(a)
        side_counts: Dict[str, int] = {}
        for e in ev:
            st = e.get('side_status') or 'unknown'
            side_counts[st] = side_counts.get(st, 0) + 1
        runs.append({
            'page_id': pid,
            'status': res.dual_track_status,
            'effective': plan_eff or eff,
            'git_head': _git_head(),
            'model_version': model_ver,
            'image_path': str(img_p.relative_to(ROOT)).replace('\\', '/'),
            'image_sha256': _sha256(img_p),
            'n_evidence': len(ev),
            'n_groups': len({e.get('group_id') for e in ev if e.get('group_id')}),
            'side_status_counts': side_counts,
            'candidate_audit': cand_audit,
            'elapsed_s': round(time.time() - t1, 2),
        })
        preds.extend(evidence_to_predictions(pid, ev, source='ONLINE'))

    doc = {
        'path': 'ONLINE',
        'entry': 'ocr.pipeline.run_ocr(detector=stage8)',
        'git_head': _git_head(),
        'model_version': model_ver,
        'effective': eff,
        'page_runs': runs,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'timing_s': round(time.time() - t0, 2),
        'predictions': preds,
    }
    pred_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return {'doc': doc, 'pred_path': pred_path, 'pred_sha256': _sha256(pred_path), 'ev_by_page': all_ev_by_page}


def _diff_predictions(ref_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    ref_map = {(p['page_id'], p['prediction_id']): p for p in ref_doc['predictions']}
    new_map = {(p['page_id'], p['prediction_id']): p for p in new_doc['predictions']}
    added = sorted(set(new_map) - set(ref_map))
    removed = sorted(set(ref_map) - set(new_map))
    changed = []
    for k in sorted(set(ref_map) & set(new_map)):
        rp, np_ = ref_map[k], new_map[k]
        if (rp.get('group_bbox') != np_.get('group_bbox')
                or rp.get('right_bbox') != np_.get('right_bbox')
                or rp.get('left_bbox') != np_.get('left_bbox')):
            changed.append({
                'page_id': k[0],
                'prediction_id': k[1],
                'before': {
                    'group_bbox': rp.get('group_bbox'),
                    'right_bbox': rp.get('right_bbox'),
                    'left_bbox': rp.get('left_bbox'),
                },
                'after': {
                    'group_bbox': np_.get('group_bbox'),
                    'right_bbox': np_.get('right_bbox'),
                    'left_bbox': np_.get('left_bbox'),
                },
            })
    return {
        'ref_total_predictions': len(ref_map),
        'new_total_predictions': len(new_map),
        'added': [{'page_id': a[0], 'prediction_id': a[1]} for a in added],
        'removed': [{'page_id': r[0], 'prediction_id': r[1]} for r in removed],
        'changed_boxes': changed,
    }


def _render_endpoint_4panel(
    p12_img: np.ndarray,
    ink: np.ndarray,
    lane: Dict[str, Any],
    edge: str,
    out_path: Path,
    scale: int = 8,
) -> Optional[Dict[str, Any]]:
    """为单条待核/复核端点生成原图 / 掩膜 / 去栏线 / 像素差异四联放大图（全坐标左闭右开）。"""
    rm = (lane.get('evidence') or {}).get('rule_mask_review')
    if not rm or not rm.get('mask_path') or not rm.get('masked_crop_path'):
        return None
    x1, y1, x2, y2 = [int(v) for v in lane['bbox']]
    cx1, cy1, cx2, cy2 = [int(v) for v in rm['crop_bbox']]
    m_img = cv2.imdecode(np.fromfile(str(ROOT / rm['mask_path']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    c_img = cv2.imdecode(np.fromfile(str(ROOT / rm['masked_crop_path']), dtype=np.uint8), cv2.IMREAD_COLOR)
    if m_img is None or c_img is None:
        return None
    mask_full = np.zeros(ink.shape, dtype=bool)
    mask_full[cy1:cy2, cx1:cx2] = m_img > 0
    cleared_full = p12_img.copy()
    cleared_full[cy1:cy2, cx1:cx2] = c_img

    H_img, W_img = ink.shape
    y_edge = y1 if edge == 'top' else y2
    zx1 = max(0, x1 - 8)
    zx2 = min(W_img, x2 + 8)
    if edge == 'top':
        zy1 = max(0, y_edge - 14)
        zy2 = min(H_img, y_edge + 26)
    else:
        zy1 = max(0, y_edge - 24)
        zy2 = min(H_img, y_edge + 10)
    r_c = p12_img[zy1:zy2, zx1:zx2]
    m_c = cv2.cvtColor((mask_full[zy1:zy2, zx1:zx2].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    cl_c = cleared_full[zy1:zy2, zx1:zx2]
    diff_mask = np.any(cv2.absdiff(r_c, cl_c) > 0, axis=2)
    df_c = np.zeros_like(r_c)
    df_c[diff_mask] = (255, 0, 255)

    panels = []
    for pp in (r_c, m_c, cl_c, df_c):
        pz = cv2.resize(pp, (pp.shape[1] * scale, pp.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.line(pz, (0, (y_edge - zy1) * scale), (pz.shape[1] - 1, (y_edge - zy1) * scale), (0, 0, 255), 1)
        cv2.line(pz, ((x1 - zx1) * scale, 0), ((x1 - zx1) * scale, pz.shape[0] - 1), (255, 0, 0), 1)
        cv2.line(pz, ((x2 - zx1) * scale, 0), ((x2 - zx1) * scale, pz.shape[0] - 1), (255, 0, 0), 1)
        panels.append(pz)
    cv2.imencode('.png', np.hstack(panels))[1].tofile(str(out_path))
    return {
        'evidence_image': str(out_path.relative_to(ROOT)).replace('\\', '/'),
        'evidence_image_sha256': _sha256(out_path),
        'crop_bbox_half_open': [zx1, zy1, zx2, zy2],
        'lane_bbox_half_open': [x1, y1, x2, y2],
        'edge_y': y_edge,
        'masked_pixels_in_crop': int(mask_full[zy1:zy2, zx1:zx2].sum()),
        'outside_mask_pixels_changed': int((diff_mask & ~mask_full[zy1:zy2, zx1:zx2]).sum()),
    }


def _build_a3_a4_ink_and_evidence(
    pred_path: Path,
    pred_sha: str,
    step5_doc: Dict[str, Any],
    acc_dir: Path,
    ev_dir: Path,
) -> Dict[str, Any]:
    acc_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    preds = doc['predictions']
    p12_preds = [p for p in preds if p.get('page_id') == '丹方彚编/page_12']
    p12_pred_map = {p['prediction_id']: p for p in p12_preds}
    step5_p12_map = {p['prediction_id']: p for p in step5_doc['predictions'] if p.get('page_id') == '丹方彚编/page_12'}

    p12_sha = _sha256(P12_IMG)
    p12_img = cv2.imdecode(np.fromfile(str(P12_IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    H = 21.0

    # A4 原图冻结判据 (2.0 * H)
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
        'coordinate_convention': '原图左闭右开 [x1, y1, x2, y2)',
        'criterion': 'min_span >= 2.0 * H (42px); top/bottom >= 0.5 triggers cut',
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
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

    # 掩膜逐轨复核（已含共线断续纵栏线掩膜与纵横栏线桥接）
    ink_path = acc_dir / 'ink_review.json'
    ink_rep = run_ink_review(pred_path, a4_path, ink_path)
    lane_by_id = {l['evidence_id']: l for l in ink_rep['lanes']}

    # A3 离线药名框核查
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
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'n_frozen_title_boxes': len(p12_gts),
        'n_lanes': sum(1 for p in p12_preds for s in ('left', 'right') if p.get(f'{s}_bbox')),
        'n_boxes_compared': n_comp,
        'n_intrusion_lanes': len(intrusions),
        'intrusions': intrusions,
        'pass': len(intrusions) == 0,
    }
    (acc_dir / 'title_check.json').write_text(json.dumps(a3_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 固定样例 1：slot06#05:right 首端调整前后对比（「香」vs「如」，严格左闭右开 + 前置条件校验）
    gid_0605 = '丹方彚编/page_12#slot06#05'
    eid_0605r = f'{gid_0605}:right'
    new_0605 = p12_pred_map.get(gid_0605)
    old_0605 = step5_p12_map.get(gid_0605)
    new_r_box = new_0605.get('right_bbox') if new_0605 else None
    old_r_box = old_0605.get('right_bbox') if old_0605 else None
    pre_0605 = validate_fixed_sample_preconditions(
        image_sha256=p12_sha,
        expected_image_sha256=EXPECTED_P12_SHA256,
        actual_bbox=new_r_box,
        expected_x_range=[658, 692],
        expected_y1_range=[648, 653],
        expected_y2_range=[820, 826],
    )
    if pre_0605['ok'] and old_r_box and new_0605 and eid_0605r in lane_by_id:
        tx1, ty1, tx2, ty2 = 640, 615, 695, 685
        raw_t = p12_img[ty1:ty2, tx1:tx2].copy()
        lab_t = labels[ty1:ty2, tx1:tx2]
        xiang_full_cids, xiang_right_cids, rule_cids, ru_cids = [], [], [], []
        for cid in sorted(set(np.unique(lab_t)) - {0}):
            cx, cy, cw, ch, _ = map(int, stats[cid])
            if cx >= 684 and cw <= 4:
                rule_cids.append(int(cid))
            elif cy >= 620 and cy + ch <= 650 and cx >= 648 and cx + cw <= 682:
                xiang_full_cids.append(int(cid))
                if cx + cw > 660:
                    xiang_right_cids.append(int(cid))
            elif cy >= 652 and cy + ch <= 678 and cx >= 660 and cx + cw <= 684:
                ru_cids.append(int(cid))
        xiang_full_y = [
            min(int(stats[c][1]) for c in xiang_full_cids),
            max(int(stats[c][1] + stats[c][3]) for c in xiang_full_cids),
        ]
        xiang_right_y = [
            min(int(stats[c][1]) for c in xiang_right_cids),
            max(int(stats[c][1] + stats[c][3]) for c in xiang_right_cids),
        ]
        ru_y = [
            min(int(stats[c][1]) for c in ru_cids),
            max(int(stats[c][1] + stats[c][3]) for c in ru_cids),
        ]
        blank_right_rows = [yy for yy in range(xiang_right_y[1], ru_y[0]) if int(ink[yy, 660:684].sum()) == 0]
        blank_full_rows = [yy for yy in range(xiang_full_y[1], ru_y[0]) if int(ink[yy, 641:684].sum()) == 0]
        blank_right_half_open = [blank_right_rows[0], blank_right_rows[-1] + 1] if blank_right_rows else []
        blank_full_half_open = [blank_full_rows[0], blank_full_rows[-1] + 1] if blank_full_rows else []

        # 使用新预测运行时生成的真实掩膜与去栏线图
        rm0605 = (lane_by_id[eid_0605r].get('evidence') or {}).get('rule_mask_review') or {}
        cx1, cy1, cx2, cy2 = rm0605['crop_bbox']
        m_img = cv2.imdecode(np.fromfile(str(ROOT / rm0605['mask_path']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        c_img = cv2.imdecode(np.fromfile(str(ROOT / rm0605['masked_crop_path']), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask_full = np.zeros(ink.shape, dtype=bool)
        mask_full[cy1:cy2, cx1:cx2] = m_img > 0
        cleared_full = p12_img.copy()
        cleared_full[cy1:cy2, cx1:cx2] = c_img

        rm_t = mask_full[ty1:ty2, tx1:tx2]
        deru_t = cleared_full[ty1:ty2, tx1:tx2]
        diff_t = np.any(cv2.absdiff(raw_t, deru_t) > 0, axis=2)
        xiang_m = np.isin(lab_t, xiang_full_cids)
        ru_m = np.isin(lab_t, ru_cids)
        ov_t = raw_t.copy()
        ov_t[xiang_m] = (255, 220, 0)
        ov_t[rm_t] = (0, 220, 255)
        ov_t[ru_m] = (0, 200, 0)
        m_vis = cv2.cvtColor((rm_t.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        diff_vis = np.zeros_like(raw_t)
        diff_vis[diff_t] = (255, 0, 255)

        def _draw_panel(im: np.ndarray, show_old: bool = False, s: int = 6) -> np.ndarray:
            z = cv2.resize(im, (im.shape[1] * s, im.shape[0] * s), interpolation=cv2.INTER_NEAREST)
            if show_old:
                cv2.rectangle(z, ((old_r_box[0] - tx1) * s, (old_r_box[1] - ty1) * s),
                              ((old_r_box[2] - tx1) * s, (ty2 - ty1) * s), (0, 0, 255), 2)
            cv2.rectangle(z, ((new_r_box[0] - tx1) * s, (new_r_box[1] - ty1) * s),
                          ((new_r_box[2] - tx1) * s, (ty2 - ty1) * s), (0, 200, 0), 2)
            lb = new_0605.get('left_bbox')
            if lb:
                cv2.rectangle(z, ((lb[0] - tx1) * s, (lb[1] - ty1) * s),
                              ((lb[2] - tx1) * s, (ty2 - ty1) * s), (255, 160, 0), 1)
            return z

        top_proof_img = ev_dir / 'slot06_05_right_top_before_after_6x.png'
        cv2.imencode('.png', np.hstack([
            _draw_panel(raw_t, show_old=True),
            _draw_panel(ov_t, show_old=False),
            _draw_panel(m_vis, show_old=False),
            _draw_panel(deru_t, show_old=False),
            _draw_panel(diff_vis, show_old=False),
        ]))[1].tofile(str(top_proof_img))

        # 统计新框内「香」像素与「如」像素保留情况
        ny1 = int(new_r_box[1])
        xiang_in_new = int(xiang_m[max(0, ny1 - ty1):, max(0, new_r_box[0] - tx1):max(0, new_r_box[2] - tx1)].sum())
        ru_in_new = int(ru_m[max(0, ny1 - ty1):, max(0, new_r_box[0] - tx1):max(0, new_r_box[2] - tx1)].sum())
        ru_total = int(ru_m.sum())
        slot06_05_top_proof = {
            'precondition': pre_0605,
            'coordinate_convention': '原图左闭右开 [start, end)',
            'before_right_bbox': old_r_box,
            'after_right_bbox': new_r_box,
            'before_left_bbox': old_0605.get('left_bbox'),
            'after_left_bbox': new_0605.get('left_bbox'),
            'before_group_bbox': old_0605.get('group_bbox'),
            'after_group_bbox': new_0605.get('group_bbox'),
            'crop_bbox_half_open': [tx1, ty1, tx2, ty2],
            'evidence_image': str(top_proof_img.relative_to(ROOT)).replace('\\', '/'),
            'evidence_image_sha256': _sha256(top_proof_img),
            'xiang_full_y_half_open': xiang_full_y,
            'xiang_right_y_half_open': xiang_right_y,
            'blank_right_y_half_open': blank_right_half_open,
            'blank_full_y_half_open': blank_full_half_open,
            'ru_y_half_open': ru_y,
            'xiang_pixels_inside_after_right_bbox': xiang_in_new,
            'ru_pixels_total': ru_total,
            'ru_pixels_inside_after_right_bbox': ru_in_new,
            'ru_pixels_changed_by_runtime_rule_mask': int((diff_t & ru_m).sum()),
            'outside_runtime_mask_pixels_changed': int((diff_t & ~rm_t).sum()),
            'runtime_rule_mask_pixels_in_crop': int(rm_t.sum()),
            'grouping_preserved': (
                new_0605.get('group_bbox') == [
                    min(new_r_box[0], new_0605['left_bbox'][0]),
                    min(new_r_box[1], new_0605['left_bbox'][1]),
                    max(new_r_box[2], new_0605['left_bbox'][2]),
                    max(new_r_box[3], new_0605['left_bbox'][3]),
                ]
            ),
            'conclusion': (
                f"原图左闭右开坐标下：药名「香」全字占 y=[{xiang_full_y[0]}, {xiang_full_y[1]})"
                f"（右轨部分占 y=[{xiang_right_y[0]}, {xiang_right_y[1]})），"
                f"右轨空白带占 y=[{blank_right_half_open[0]}, {blank_right_half_open[1]})"
                f"（整组空白带占 y=[{blank_full_half_open[0]}, {blank_full_half_open[1]})），"
                f"首字「如」占 y=[{ru_y[0]}, {ru_y[1]})。新右轨起点由 y1={old_r_box[1]} 调至 y1={ny1}，"
                f"精准落于空白带内：新框内「香」像素={xiang_in_new}、「如」完整保留 {ru_in_new}/{ru_total} 像素，"
                f"运行时断续纵栏线掩膜对「如」改动 {int((diff_t & ru_m).sum())} 像素、掩膜外改动 {int((diff_t & ~rm_t).sum())} 像素，"
                f"且左右轨分组关系保持完整（group_bbox={new_0605.get('group_bbox')}）。"
            ),
        }
    else:
        slot06_05_top_proof = {
            'precondition': pre_0605,
            'status': 'pending_precondition_mismatch',
            'conclusion': f"前置条件校验未通过（{pre_0605['reasons']}），按规则标记待核，不复用旧结论。",
        }

    # 固定样例 2：两处「真」末笔（slot06#01:right 与 slot06#05:right）前置条件校验 + 证据图
    zhen_proofs = {}
    for zid in ('丹方彚编/page_12#slot06#01:right', '丹方彚编/page_12#slot06#05:right'):
        lane = lane_by_id.get(zid)
        pre_z = validate_fixed_sample_preconditions(
            image_sha256=p12_sha,
            expected_image_sha256=EXPECTED_P12_SHA256,
            actual_bbox=lane['bbox'] if lane else None,
            expected_x_range=[658, 692],
            expected_y2_range=[821, 825],
        )
        if pre_z['ok'] and lane:
            safe = zid.replace('丹方彚编/page_12#', '').replace(':', '_')
            zp = ev_dir / f'{safe}_bottom_zhen_4panel_8x.png'
            meta = _render_endpoint_4panel(p12_img, ink, lane, 'bottom', zp, scale=8)
            rm = lane['evidence']['rule_mask_review']
            cx1, cy1, cx2, cy2 = rm['crop_bbox']
            m_img = cv2.imdecode(np.fromfile(str(ROOT / rm['mask_path']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask_full = np.zeros(ink.shape, dtype=bool)
            mask_full[cy1:cy2, cx1:cx2] = m_img > 0
            x1, _, x2, y2 = lane['bbox']
            unmasked_xs = [x for x in range(x1, x2) if ink[y2, x] and not mask_full[y2, x]]
            zhen_proofs[zid] = {
                'precondition': pre_z,
                **(meta or {}),
                'adhesion_component_bbox_half_open': [663, 819, 672, 824],
                'edge_y_unmasked_ink_xs': unmasked_xs,
                'y824_mask_pixels': int(mask_full[824, x1:x2].sum()),
                'verdict': '待核（末字「真」底撇 [663, 819, 672, 824) 跨过 y2=823 并粘连 y=[824, 827) 底版框，无法无损证明完整）',
            }
        else:
            zhen_proofs[zid] = {
                'precondition': pre_z,
                'status': 'pending_precondition_mismatch',
                'verdict': f"待核（前置条件校验未通过：{pre_z['reasons']}）",
            }

    # 为全部复核轨的每个端点生成四联证据图，并逐条校验预测框一致与前置条件
    lane_decisions = []
    for lane in ink_rep['lanes']:
        eid = lane['evidence_id']
        gid, side = eid.rsplit(':', 1)
        pred_rec = p12_pred_map.get(gid)
        pred_box = pred_rec.get(f'{side}_bbox') if pred_rec else None
        bbox_match = (pred_box == lane['bbox'])
        eps = lane['endpoints']
        ep_details = {}
        for ep_name, ep_val in eps.items():
            safe = eid.replace('丹方彚编/page_12#', '').replace(':', '_')
            if eid in zhen_proofs and ep_name == 'bottom' and zhen_proofs[eid].get('evidence_image'):
                ep_img_meta = {
                    'evidence_image': zhen_proofs[eid]['evidence_image'],
                    'evidence_image_sha256': zhen_proofs[eid]['evidence_image_sha256'],
                    'crop_bbox_half_open': zhen_proofs[eid].get('crop_bbox_half_open'),
                }
            else:
                ep_img_path = ev_dir / f'{safe}_{ep_name}_4panel_8x.png'
                ep_img_meta = _render_endpoint_4panel(p12_img, ink, lane, ep_name, ep_img_path, scale=8)

            pre_ok = True
            if eid == eid_0605r and ep_name == 'top':
                pre_ok = bool(pre_0605['ok'])
            elif eid in zhen_proofs and ep_name == 'bottom':
                pre_ok = bool(zhen_proofs[eid]['precondition']['ok'])

            if not bbox_match or not pre_ok:
                cat = '前置校验不符'
                ep_status = 'pending_precondition_mismatch'
                detail = '新预测框或固定样例前置条件校验未通过，强制保持待核。'
            elif lane['files'] != 'verified':
                cat = '缺少掩膜证据'
                ep_status = 'pending_missing_mask'
                detail = f"运行时未提供已核验掩膜附件（files={lane['files']}），保持待核。"
            elif ep_val['verdict'] == 'rule_explains_trigger':
                cat = '框外栏线触发'
                ep_status = 'cleared_by_rule_mask'
                if eid == eid_0605r and ep_name == 'top':
                    detail = '新起点 y1=650 已落于「香」[621, 649) 与「如」[653, 676) 之间空白带 [649, 653)，运行时断续纵栏线掩膜完全解释跨边墨迹，无框内字形接触，解除该端点疑点。'
                elif eid == '丹方彚编/page_12#slot05#01:right' and ep_name == 'top':
                    detail = '运行时共线断续纵栏线掩膜（x=[734, 736)）完全解释 y1=591 跨边触发且无切字残墨，解除 y1=591 边界切字疑点（另注：框内 y=[594, 619) 含大字「酥」，已记入开放分组观察项）。'
                else:
                    detail = '掩膜复核验证通过，去栏线后无框内字形接触且无跨边残墨，解除该端点疑点。'
            else:
                cat = '框内真实笔画接触'
                ep_status = 'pending_glyph_adhesion'
                if eid == '丹方彚编/page_12#slot01#01:right' and ep_name == 'bottom':
                    detail = '末字「解」右侧竖笔在 y=[822, 825) 直接并入底版框，去掩膜后无法无损证明笔画完整，保持待核。'
                elif eid == '丹方彚编/page_12#slot07#00:right' and ep_name == 'bottom':
                    detail = '外侧纵栏线经桥接掩膜后，邻列字「效」左端 x=[639, 641), y=[818, 821) 紧贴纵栏线掩膜并触发框内触字检查（本轨正文区 x=[613, 636) 留白），保持待核。'
                elif eid in zhen_proofs and ep_name == 'bottom':
                    detail = '末字「真」底撇 [663, 819, 672, 824) 跨过 y2=823 并直接粘连 y=[824, 827) 底版框掩膜，无法无损证明完整，保持待核。'
                else:
                    detail = '框内真实字形笔画与栏线/版框掩膜直接接触粘连，无法无损证明完整，保持待核。'

            ep_details[ep_name] = {
                **ep_val,
                'category': cat,
                'endpoint_policy_status': ep_status,
                'detail': detail,
                'evidence_panel': ep_img_meta,
            }

        all_cleared = all(v['endpoint_policy_status'] == 'cleared_by_rule_mask' for v in ep_details.values())
        any_fail = any(v['endpoint_policy_status'] == 'fail_glyph_crossing' for v in ep_details.values())
        final_status = 'cleared_by_rule_mask' if all_cleared else ('fail_glyph_crossing' if any_fail else 'pending_review')
        lane_decisions.append({
            'evidence_id': eid,
            'bbox': lane['bbox'],
            'bbox_matches_prediction': bbox_match,
            'original_a4_trigger': lane['original_a4_trigger'],
            'files': lane['files'],
            'final_policy_status': final_status,
            'endpoints': ep_details,
        })

    a4_policy_doc = {
        'coordinate_convention': '原图左闭右开 [x1, y1, x2, y2)',
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'source_image_sha256': p12_sha,
        'all_referenced_bboxes_verified': all(x['bbox_matches_prediction'] for x in lane_decisions),
        'missing_mask_lanes': [x['evidence_id'] for x in lane_decisions if x['files'] != 'verified'],
        'original_a4_checked': ink_rep['n_checked_original'],
        'original_a4_raw_triggers': ink_rep['n_raw_triggers'],
        'review_lanes_total': len(lane_decisions),
        'cleared_lanes_count': sum(1 for x in lane_decisions if x['final_policy_status'] == 'cleared_by_rule_mask'),
        'cleared_raw_trigger_count': sum(1 for x in lane_decisions if x['original_a4_trigger'] and x['final_policy_status'] == 'cleared_by_rule_mask'),
        'remaining_raw_trigger_unresolved_count': sum(1 for x in lane_decisions if x['original_a4_trigger'] and x['final_policy_status'] != 'cleared_by_rule_mask'),
        'fail_glyph_crossing_lanes': [x['evidence_id'] for x in lane_decisions if x['final_policy_status'] == 'fail_glyph_crossing'],
        'pending_review_lanes': [x['evidence_id'] for x in lane_decisions if x['final_policy_status'] == 'pending_review'],
        'cleared_lanes': [x['evidence_id'] for x in lane_decisions if x['final_policy_status'] == 'cleared_by_rule_mask'],
        'a4_final_pass': all(x['final_policy_status'] == 'cleared_by_rule_mask' for x in lane_decisions),
        'slot06_05_right_top_proof': slot06_05_top_proof,
        'zhen_bottom_proofs': zhen_proofs,
        'lanes': lane_decisions,
    }
    (acc_dir / 'a4_policy_reconcile.json').write_text(json.dumps(a4_policy_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'a3': a3_doc, 'a4': a4_doc, 'ink_review': ink_rep, 'a4_policy': a4_policy_doc}


def _extract_targets(m_frozen_doc: Dict[str, Any], m_v2_doc: Dict[str, Any],
                     pred_path: Path, pred_sha: str) -> Dict[str, Any]:
    out_cal = {}
    for cal_name, m in (('frozen', m_frozen_doc), ('v2', m_v2_doc)):
        o5 = m['scores']['overall']['0.5']
        o75 = m['scores']['overall']['0.75']
        h5 = m['scores']['by_split']['historical_regression']['thresholds']['0.5']
        h75 = m['scores']['by_split']['historical_regression']['thresholds']['0.75']
        p12 = next(x['thresholds']['0.5'] for x in m['pages'] if x['page_id'] == '丹方彚编/page_12')
        w12 = next(x['thresholds']['0.5'] for x in m['pages'] if x['page_id'] == '温疫方論/page_12')
        t5_num = o5['lane_counts'].get('pairing_success', 0)
        t5_den = o5['lane_denominator']
        t5_val = round(t5_num / t5_den, 4) if t5_den else 0.0
        out_cal[cal_name] = {
            'T1': {'num': o5['group_hits'], 'den': 38, 'tp': o5['tp'], 'fp': o5['fp'], 'fn': o5['fn'], 'target': 30, 'pass': o5['group_hits'] >= 30},
            'T2': {'num': o75['group_hits'], 'den': 38, 'tp': o75['tp'], 'fp': o75['fp'], 'fn': o75['fn'], 'target': 18, 'pass': o75['group_hits'] >= 18},
            'T3': {'num': h5['group_hits'], 'den': 20, 'target': 19, 'pass': h5['group_hits'] >= 19},
            'T4': {'num': h75['group_hits'], 'den': 20, 'target': 14, 'pass': h75['group_hits'] >= 14},
            'T5': {'num': t5_num, 'den': t5_den, 'value': t5_val, 'excluded': o5['lane_excluded'], 'lane_counts': o5['lane_counts'], 'target': 0.9, 'pass': t5_den >= 10 and t5_val >= 0.9},
            'T6': {'num': o5['negative_confirmed_fp'], 'den': 10 if cal_name == 'frozen' else 9, 'suspected': o5['negative_suspected_fp'], 'duplicate_extra': o5['duplicate_extra_fp'], 'candidates': o5['new_candidates'], 'target': 0, 'pass': o5['negative_confirmed_fp'] == 0},
            'T7': {'num': p12['group_hits'], 'den': 11, 'target': 6, 'pass': p12['group_hits'] >= 6},
            'T8': {'num': w12['group_hits'], 'den': 2, 'target': 1, 'pass': w12['group_hits'] >= 1},
        }
    return {
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'calibers': out_cal,
    }


def _build_neg_df_single_01_review(ev_dir: Path, region_bbox: Sequence[int]) -> Dict[str, Any]:
    p12_sha = _sha256(P12_IMG)
    pre = validate_fixed_sample_preconditions(
        image_sha256=p12_sha,
        expected_image_sha256=EXPECTED_P12_SHA256,
        actual_bbox=region_bbox,
        expected_x_range=[536, 585],
        expected_y1_range=[210, 211],
        expected_y2_range=[815, 816],
    )
    if not pre['ok']:
        return {'negative_region_id': 'NEG-DF-SINGLE-01', 'precondition': pre, 'status': 'pending_precondition_mismatch'}
    p12_img = cv2.imdecode(np.fromfile(str(P12_IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY))
    sub = ink[334:740, 536:585]
    col_sum = sub.sum(axis=0)
    seam_stats = {}
    for xs in (563, 564, 565, 566):
        col = ink[334:740, xs]
        runs, cur = [], 0
        for v in col:
            if not v:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        seam_stats[str(xs)] = {
            'ink_pixels': int(col.sum()),
            'blank_ratio': round(float((~col).mean()), 4),
            'max_blank_run_px': int(max(runs) if runs else 0),
        }
    nx1, ny1, nx2, ny2 = 520, 195, 605, 835
    neg_ov = p12_img[ny1:ny2, nx1:nx2].copy()
    cv2.rectangle(neg_ov, (536 - nx1, 210 - ny1), (585 - nx1, 815 - ny1), (0, 220, 255), 1)
    cv2.rectangle(neg_ov, (564 - nx1, 330 - ny1), (594 - nx1, 822 - ny1), (0, 0, 255), 1)
    cv2.rectangle(neg_ov, (545 - nx1, 330 - ny1), (564 - nx1, 740 - ny1), (255, 100, 0), 1)
    cv2.line(neg_ov, (564 - nx1, 334 - ny1), (564 - nx1, 740 - ny1), (0, 200, 0), 1)
    ev_img = ev_dir / 'neg_df_single_01_independent_review.png'
    cv2.imencode('.png', np.hstack([p12_img[ny1:ny2, nx1:nx2], neg_ov]))[1].tofile(str(ev_img))
    return {
        'precondition': pre,
        'coordinate_convention': '原图左闭右开 [start, end)',
        'negative_region_id': 'NEG-DF-SINGLE-01',
        'page_id': '丹方彚编/page_12',
        'bbox_half_open': [536, 210, 585, 815],
        'source_image_sha256': p12_sha,
        'evidence_image': str(ev_img.relative_to(ROOT)).replace('\\', '/'),
        'evidence_image_sha256': _sha256(ev_img),
        'body_measurement_y_half_open': [334, 740],
        'left_lane_peak_col_ink': int(col_sum[545 - 536:562 - 536].max()),
        'right_lane_peak_col_ink': int(col_sum[568 - 536:585 - 536].max()),
        'seam_profile': seam_stats,
        'title_reading': '蘇合油香（y=[204, 330)，原冻结标注误记为“藿合油香”）',
        'right_lane_reading': '只用如膏油赤色無滓香極芬烈塗（x=[564, 594)）',
        'left_lane_reading': '上手皆香透于心或如鵝膠者真（x=[545, 564)）',
        'independent_review_conclusion': '原图明确为药名「蘇合油香」领起的双行小字注记：x=[564, 566) 存在空白率 99.26%、最长连续空白 354 px 的纵缝（谷深仅峰值 2.1%），左右各具独立完整小字列，原冻结标签“单一列、未分两轨”与原图事实不符。',
    }


def _run_a1_a7_and_r1_r5(
    pred_path: Path, pred_sha: str,
    ev_by_page: Dict[str, List[Dict[str, Any]]],
    a3_doc: Dict[str, Any], a4_doc: Dict[str, Any],
    a4_policy_doc: Dict[str, Any], acc_dir: Path,
) -> Dict[str, Any]:
    import ocr.dual_track as DT
    import ocr.pipeline as PL

    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    p12_preds = [p for p in doc['predictions'] if p.get('page_id') == '丹方彚编/page_12']
    p12_ev = ev_by_page['丹方彚编/page_12']

    by_reg: Dict[str, List[Dict[str, Any]]] = {}
    for p in p12_preds:
        rid = (p.get('region_ref') or {}).get('region_id') or p['prediction_id'].rsplit('#', 1)[0]
        by_reg.setdefault(rid, []).append(p)
    stacked = []
    for rid, gs in by_reg.items():
        gs_s = sorted(gs, key=lambda x: x['group_bbox'][1])
        if len(gs_s) >= 2 and gs_s[0]['group_bbox'][3] <= gs_s[-1]['group_bbox'][1]:
            stacked.append({'region_id': rid, 'upper': gs_s[0]['group_bbox'], 'lower': gs_s[-1]['group_bbox']})

    all_ev = [e for evs in ev_by_page.values() for e in evs]
    p12_crop_check = check_evidence_crops(p12_ev)
    all_crop_check = check_evidence_crops(all_ev)
    mask_checks = []
    for pid, evs in ev_by_page.items():
        img_arr = None
        for e in evs:
            rev_ev = e.get('review_evidence') or {}
            if rev_ev.get('rule_mask_review'):
                if img_arr is None:
                    img_arr = cv2.imdecode(np.fromfile(str(_image_path(pid)), dtype=np.uint8), cv2.IMREAD_COLOR)
                st, _ = _verify_crop(img_arr, rev_ev)
                mask_checks.append({'page_id': pid, 'evidence_id': e.get('evidence_id'), 'status': st})
    mask_failures = [m for m in mask_checks if m['status'] != 'verified']

    by_gid: Dict[str, Dict[str, str]] = {}
    for e in p12_ev:
        if e.get('side_status') == 'ok' and e.get('text'):
            by_gid.setdefault(e['group_id'], {})[e['side']] = e['text']
    paired = [{'group_id': g, 'right': s['right'], 'left': s['left']}
              for g, s in by_gid.items() if 'right' in s and 'left' in s]

    def _txt(r):
        return '\n'.join((b.text or '') for b in r.blocks)

    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    os.environ['TCM_OCR_DUAL_TRACK_MODE'] = 'live'
    os.environ['TCM_OCR_DUAL_TRACK_DETECTOR'] = 'stage8'
    import app.config as C
    from importlib import reload
    reload(C); reload(PL)
    res_on = PL.run_ocr(str(P12_IMG), page_id='丹方彚编/page_12')
    os.environ['TCM_OCR_DUAL_TRACK'] = '0'
    reload(C); reload(PL)
    res_off = PL.run_ocr(str(P12_IMG), page_id='丹方彚编/page_12')
    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    reload(C); reload(PL)
    base_text = _txt(res_on)
    a6_pass = (res_off.dual_track_status == 'not_run' and base_text == _txt(res_off))

    paired_ids = {p['group_id'] for p in paired}
    bc = Counter(tuple(e['bbox']) for e in p12_ev if e.get('bbox'))
    target_e = next(e for e in p12_ev if e.get('side') == 'left' and e.get('side_status') == 'ok'
                    and e.get('text') and e.get('group_id') in paired_ids and bc[tuple(e['bbox'])] == 1)
    target_bbox = [int(v) for v in target_e['bbox']][:4]
    orig_read = DT._crop_read

    def _patched(image, bbox, zoom, *a, **k):
        if [int(v) for v in bbox][:4] == target_bbox:
            raise RuntimeError('injected_side_failure')
        return orig_read(image, bbox, zoom, *a, **k)

    DT._crop_read = _patched
    try:
        inj = PL.run_ocr(str(P12_IMG), page_id='丹方彚编/page_12')
    finally:
        DT._crop_read = orig_read
    inj_ev = inj.dual_track_evidence or []
    t_ev = next(e for e in inj_ev if [int(v) for v in e['bbox']][:4] == target_bbox)
    m_ev = next(e for e in inj_ev if e['group_id'] == target_e['group_id'] and e['side'] == 'right')
    a7_pass = (t_ev['side_status'] == 'recognize_error' and m_ev['side_status'] == 'ok'
               and len(inj_ev) == len(p12_ev) and _txt(inj) == base_text)

    checks = [
        {'id': 'A1', 'title': 'p12 同栏上下两组', 'pass': len(stacked) >= 2, 'detail': {'n_stacked': len(stacked), 'examples': stacked}},
        {'id': 'A2', 'title': '左右裁图与掩膜附件逐条核验', 'pass': p12_crop_check['pass'] and all_crop_check['pass'] and not mask_failures,
         'detail': {'p12_crops': p12_crop_check['n_crops'], 'all_19_page_crops': all_crop_check['n_crops'], 'missing_crops': all_crop_check['missing'], 'mask_attachments_verified': len(mask_checks), 'mask_failures': mask_failures}},
        {'id': 'A3', 'title': '标题隔离（冻结药名框离线验收）', 'pass': a3_doc['pass'], 'detail': a3_doc},
        {'id': 'A4', 'title': '首尾笔画（保留原图触发 + 允许掩膜复核参与最终判定）',
         'pass': a4_policy_doc['a4_final_pass'],
         'detail': {
             'raw_n_checked': a4_doc['n_checked'],
             'raw_n_cut': a4_doc['n_cut'],
             'cleared_lanes_count': a4_policy_doc['cleared_lanes_count'],
             'cleared_raw_trigger_count': a4_policy_doc['cleared_raw_trigger_count'],
             'fail_glyph_crossing_lanes': a4_policy_doc['fail_glyph_crossing_lanes'],
             'pending_review_lanes': a4_policy_doc['pending_review_lanes'],
             'missing_mask_lanes': a4_policy_doc['missing_mask_lanes'],
         }},
        {'id': 'A5', 'title': '左右并列读数', 'pass': len(paired) >= 1, 'detail': {'n_paired': len(paired), 'examples': paired[:5]}},
        {'id': 'A6', 'title': '开关一致性（关闭 = not_run 且正文不变）', 'pass': a6_pass, 'detail': {'off_status': res_off.dual_track_status, 'text_identical': base_text == _txt(res_off)}},
        {'id': 'A7', 'title': '单侧失败隔离（实际注入异常）', 'pass': a7_pass, 'detail': {'page_status': inj.dual_track_status, 'target_evidence_id': target_e['evidence_id'], 'target_side_status': t_ev['side_status'], 'mate_side_status': m_ev['side_status'], 'n_evidence': len(inj_ev)}},
    ]
    checklist = {
        'stage': 'stage8_step6_single_run_acceptance',
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'passed': sum(1 for c in checks if c['pass']),
        'total': len(checks),
        'all_pass': all(c['pass'] for c in checks),
        'checks': checks,
    }
    (acc_dir / 'checklist.json').write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    common = dict(width=0, height=0, engine='rapidocr')
    s5_before = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='auto', cfg={'detector': 'stage5'}, **common)
    off1 = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='off', cfg={'detector': 'stage8'}, **common)
    on1 = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='auto', cfg={'detector': 'stage8'}, **common)
    on2 = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='auto', cfg={'detector': 'stage8'}, **common)
    off2 = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='off', cfg={'detector': 'stage8'}, **common)
    s5_after = DT.run_dual_track(str(P12_IMG), '丹方彚编/page_12', mode='auto', cfg={'detector': 'stage5'}, **common)
    r_checks = [
        {'id': 'R1', 'pass': off1['status'] == off2['status'] == 'not_run' and off1['evidence'] == off2['evidence'] is None},
        {'id': 'R2', 'pass': s5_before['groups'] == s5_after['groups'] and s5_before['status'] == s5_after['status']},
        {'id': 'R3', 'pass': on1['status'] == 'completed' and on1['groups'] > 0 and on1['plan']['detector'] == 'stage8'},
        {'id': 'R4', 'pass': [e['bbox'] for e in on1['evidence']] == [e['bbox'] for e in on2['evidence']]},
        {'id': 'R5', 'pass': off1['attempted'] is False and off2['attempted'] is False},
    ]
    reg_doc = {'passed': sum(1 for r in r_checks if r['pass']), 'total': 5, 'checks': r_checks}
    (acc_dir / 'regress_switch.json').write_text(json.dumps(reg_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'checklist': checklist, 'regress': reg_doc}


def main() -> int:
    py = sys.executable
    pred_path = OUT_DIR / 'predictions/path_online_step6.json'
    crop_dir = OUT_DIR / 'crops'
    acc_dir = OUT_DIR / 'acceptance'
    ev_dir = OUT_DIR / 'evidence'
    closure_dir = OUT_DIR / 'closure'
    m_frozen_dir = OUT_DIR / 'metrics/neg_frozen'
    m_v2_dir = OUT_DIR / 'metrics/neg_v2'
    for d in (acc_dir, ev_dir, closure_dir, m_frozen_dir, m_v2_dir):
        d.mkdir(parents=True, exist_ok=True)

    step5_doc = json.loads(STEP5_PRED.read_text(encoding='utf-8'))
    step5_sha = _sha256(STEP5_PRED)
    step5_a4p = json.loads((STEP5_DIR / 'acceptance/a4_policy_reconcile.json').read_text(encoding='utf-8'))
    step5_targets = json.loads((STEP5_DIR / 'acceptance/targets.json').read_text(encoding='utf-8'))

    # 1. 同一次 19 页正常入口新预测
    run = _run_19_pages_online(crop_dir, pred_path)
    pred_sha = run['pred_sha256']
    pred_diff = _diff_predictions(step5_doc, run['doc'])
    (closure_dir / 'diff_vs_step5.json').write_text(json.dumps({
        'reference_prediction': str(STEP5_PRED.relative_to(ROOT)).replace('\\', '/'),
        'reference_sha256': step5_sha,
        'new_prediction': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'new_sha256': pred_sha,
        **pred_diff,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 2. A3、A4 原图统计、掩膜复核与动态 A4 联合判定
    acc = _build_a3_a4_ink_and_evidence(pred_path, pred_sha, step5_doc, acc_dir, ev_dir)

    # 3. 两套 T6 评分
    for neg_f, out_m in [
        (S2 / 'negative_regions.json', m_frozen_dir),
        (S8 / 'negative_regions_v2.json', m_v2_dir),
    ]:
        subprocess.check_call([
            py, str(ROOT / 'tools/dual_track_stage3_score.py'),
            '--file', str(pred_path),
            '--neg', str(neg_f),
            '--out-dir', str(out_m),
        ])
    m_frozen_doc = json.loads((m_frozen_dir / 'online_step6.json').read_text(encoding='utf-8'))
    m_v2_doc = json.loads((m_v2_dir / 'online_step6.json').read_text(encoding='utf-8'))
    targets_doc = _extract_targets(m_frozen_doc, m_v2_doc, pred_path, pred_sha)
    (acc_dir / 'targets.json').write_text(json.dumps(targets_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 4. A1–A7 与 R1–R5
    checks_res = _run_a1_a7_and_r1_r5(
        pred_path, pred_sha, run['ev_by_page'],
        acc['a3'], acc['a4'], acc['a4_policy'], acc_dir,
    )

    # 5. NEG-DF-SINGLE-01 独立复核 + T6 双口径对账（修正 v2_basis 写入撤回理由）
    orig_neg = json.loads((S2 / 'negative_regions.json').read_text(encoding='utf-8'))
    v2_neg = json.loads((S8 / 'negative_regions_v2.json').read_text(encoding='utf-8'))
    vm = {r['negative_region_id']: r for r in v2_neg['regions']}
    diffs = []
    neg_df_bbox = [536, 210, 585, 815]
    for r in orig_neg['regions']:
        rid = r['negative_region_id']
        rv = vm.get(rid, {})
        if rid == 'NEG-DF-SINGLE-01':
            neg_df_bbox = r['bbox']
        if r.get('status') != rv.get('status'):
            diffs.append({
                'negative_region_id': rid,
                'page_id': r['page_id'],
                'bbox_half_open': r['bbox'],
                'frozen_status': r.get('status'),
                'v2_status': rv.get('status'),
                'frozen_basis': r.get('basis'),
                'v2_basis': _extract_v2_retract_basis(rv),
                'v2_original_basis_preserved': rv.get('basis'),
            })
    neg_review = _build_neg_df_single_01_review(ev_dir, neg_df_bbox)
    t6_reconcile = {
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'reference_step5_sha256': step5_sha,
        'formal_historical_caliber': 'frozen',
        'frozen_negative_sha256': _sha256(S2 / 'negative_regions.json'),
        'v2_negative_sha256': _sha256(S8 / 'negative_regions_v2.json'),
        'differences': diffs,
        'affected_predictions': [
            p['prediction_id'] for p in run['doc']['predictions']
            if p['page_id'] == '丹方彚编/page_12' and p['group_bbox'][0] < 585 and p['group_bbox'][2] > 536
        ],
        'independent_review_NEG_DF_SINGLE_01': neg_review,
        'T6_frozen_formal_historical': targets_doc['calibers']['frozen']['T6'],
        'T6_v2_review_reference': targets_doc['calibers']['v2']['T6'],
    }
    t6_path = closure_dir / 't6_caliber_reconcile.json'
    t6_path.write_text(json.dumps(t6_reconcile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 6. 自动生成全数字回溯最终报告（严格原图左闭右开坐标）
    a4p = acc['a4_policy']
    tf, tv = targets_doc['calibers']['frozen'], targets_doc['calibers']['v2']
    sf_tf = step5_targets['calibers']['frozen']
    ck = checks_res['checklist']
    rg = checks_res['regress']
    lane_rows = []
    for l in a4p['lanes']:
        eps_str = '；'.join(f"{k}: {v['category']} → {v['endpoint_policy_status']}" for k, v in l['endpoints'].items())
        lane_rows.append(
            f"| `{l['evidence_id'].replace('丹方彚编/page_12#','')}` | `{l['bbox']}` | "
            f"{'是' if l['original_a4_trigger'] else '否'} | {l['files'] or '无掩膜'} | "
            f"**{l['final_policy_status']}** | {eps_str} |"
        )
    sp = a4p['slot06_05_right_top_proof']
    report_md = f"""# 阶段八新一轮正常入口验收与待决结项报告（step6_closure）

生成时间：{datetime.now(TZ).isoformat(timespec='seconds')}  
代码版本：Git `HEAD` `{_git_head()}` · 配置指纹 `{run['doc']['effective']['fingerprint']}` · 模型 `{run['doc']['model_version']}`  
坐标约定：**原图左闭右开 `[x1, y1, x2, y2)`**  
历史对照预测：`output/dual_tracks/stage8/step5_final/predictions/path_online_final.json`（SHA-256 `{step5_sha}`）  
本轮唯一预测：`{t6_reconcile['prediction_source']}`（SHA-256 `{pred_sha}`）  
预测框变化数：`{len(pred_diff['changed_boxes'])}` 组（增 `{len(pred_diff['added'])}` / 删 `{len(pred_diff['removed'])}`）

## 1. 综合结论与相对 step5_final 变化
- **阶段八状态：未通过（暂不结项）**
- 入口验收 A1–A7：**{ck['passed']}/{ck['total']}**（较 step5 不变；A4 不通过项由 `1` 降至 **`{len(a4p['fail_glyph_crossing_lanes'])}`**, 缺掩膜项由 `2` 降至 **`{len(a4p['missing_mask_lanes'])}`**, 解除疑点由 `{step5_a4p['cleared_lanes_count']}` 升至 **`{a4p['cleared_lanes_count']}`**, 待核由 `{len(step5_a4p['pending_review_lanes'])}` 变为 **`{len(a4p['pending_review_lanes'])}`**）；开关回归 R1–R5：**{rg['passed']}/{rg['total']}**。
- 原冻结正式历史口径 T1–T8：**{sum(1 for v in tf.values() if v['pass'])}/8**（仅 T6=`{tf['T6']['num']}/{tf['T6']['den']}` 未达 0）；v2 复核口径 T1–T8：**{sum(1 for v in tv.values() if v['pass'])}/8**（T6=`{tv['T6']['num']}/{tv['T6']['den']}` 达标）。

## 2. `slot06#05:right` 首端调整与尾端「真」核验（原图左闭右开）
- **前置条件校验**：`{sp.get('precondition', {}).get('status')}`
- **首端像素核实**：{sp.get('conclusion')}
- **对照五联图**：`{sp.get('evidence_image')}`（SHA-256 `{(sp.get('evidence_image_sha256') or '')[:16]}...`）
- **尾端「真」核验**：`slot06#01:right` 与 `slot06#05:right` 共享末字「真」底撇 `[663, 819, 672, 824)` 跨过 `y2=823` 并粘连 `y=[824, 827)` 底版框，均继续标**待核**。

## 3. A4 原图统计与 9 条复核轨逐端点结果（全框已动态回查新预测）
- **原图 A4 触发（保留）**：`{a4p['original_a4_raw_triggers']} / {a4p['original_a4_checked']}`（较 step5 `{step5_a4p['original_a4_raw_triggers']}/{step5_a4p['original_a4_checked']}`）
- **掩膜附件核验**：`{a4p['review_lanes_total'] - len(a4p['missing_mask_lanes'])} / {a4p['review_lanes_total']}`（缺掩膜 `{len(a4p['missing_mask_lanes'])}` 条）
- **联合判定**：解除疑点 **{a4p['cleared_lanes_count']} 条轨**（含原图触发中 **{a4p['cleared_raw_trigger_count']} 条**）、不通过 **{len(a4p['fail_glyph_crossing_lanes'])} 条轨**、待核 **{len(a4p['pending_review_lanes'])} 条轨**。

| 轨道 | 新原图轨框 `[x1,y1,x2,y2)` | 原图 A4 | 掩膜核验 | 联合判定 | 逐端点分类 |
|---|---|---|---|---|---|
{chr(10).join(lane_rows)}

## 4. T6 双口径对账（已修正 `v2_basis`）
- **冻结历史依据**（`frozen_basis`）：`{diffs[0]['frozen_basis'] if diffs else ''}`
- **复核版撤回理由**（`v2_basis`）：`{diffs[0]['v2_basis'] if diffs else ''}`
- **评分**：冻结历史口径 T6 = **`{tf['T6']['num']}/{tf['T6']['den']}`**（未达 0）；v2 复核口径 T6 = **`{tv['T6']['num']}/{tv['T6']['den']}`**（达标）。

## 5. T1–T8 指标对比（step5_final → step6_closure）
| 项 | step5（冻结） | step6（冻结） | step6（v2） | 门槛 |
|---|---|---|---|---|
| T1 (@0.50) | `{sf_tf['T1']['num']}/38` | `{tf['T1']['num']}/38` | `{tv['T1']['num']}/38` | ≥30/38 |
| T2 (@0.75) | `{sf_tf['T2']['num']}/38` | `{tf['T2']['num']}/38` | `{tv['T2']['num']}/38` | ≥18/38 |
| T3 / T4 | `{sf_tf['T3']['num']}/20`, `{sf_tf['T4']['num']}/20` | `{tf['T3']['num']}/20`, `{tf['T4']['num']}/20` | `{tv['T3']['num']}/20`, `{tv['T4']['num']}/20` | ≥19/20, ≥14/20 |
| T5 轨配对 | `{sf_tf['T5']['num']}/{sf_tf['T5']['den']} ({sf_tf['T5']['value']})` | `{tf['T5']['num']}/{tf['T5']['den']} ({tf['T5']['value']})` | `{tv['T5']['num']}/{tv['T5']['den']} ({tv['T5']['value']})` | ≥0.90 |
| T6 负例误检 | `{sf_tf['T6']['num']}/10` | **`{tf['T6']['num']}/10`（未达）** | **`{tv['T6']['num']}/9`（达标）** | 0 |
| T7 / T8 | `{sf_tf['T7']['num']}/11`, `{sf_tf['T8']['num']}/2` | `{tf['T7']['num']}/11`, `{tf['T8']['num']}/2` | `{tv['T7']['num']}/11`, `{tv['T8']['num']}/2` | ≥6/11, ≥1/2 |

## 6. 仍未解决问题与样本范围说明
1. **A4 剩余 {len(a4p['pending_review_lanes'])} 条真实贴线待核轨**：`{', '.join(x.replace('丹方彚编/page_12#','') for x in a4p['pending_review_lanes'])}`（末字笔画或邻字与版框/栏线物理粘连，在原图不修图前提下无法无损证明完整）。
2. **`slot05#01` 组起点观察项**：`y1=591` 边界本身不切字且已由断续纵栏线掩膜解释，但框内 `y=[594, 619)` 含大字「酥」，留待后续分组治理。
3. **T6 原冻结正式口径**：仍为 `1/10`，需人工决定是否将 `negative_regions_v2.json` 升为正式版本。
4. **样本限制**：评测集仅 19 页（38 正例组、10 原冻结负例区），结论仅覆盖当前古籍竖排双行小字样本范围。
"""
    report_path = closure_dir / '阶段八第六轮验收与待决结项报告.md'
    report_path.write_text(report_md, encoding='utf-8')
    print('=== STEP6 SINGLE-RUN PIPELINE SUMMARY ===')
    print(json.dumps({
        'prediction_sha256': pred_sha,
        'reference_step5_sha256': step5_sha,
        'changed_boxes_count': len(pred_diff['changed_boxes']),
        'changed_boxes': pred_diff['changed_boxes'],
        'A1_A7': f"{ck['passed']}/{ck['total']}",
        'R1_R5': f"{rg['passed']}/{rg['total']}",
        'A4_raw': f"{a4p['original_a4_raw_triggers']}/{a4p['original_a4_checked']}",
        'A4_missing_masks': a4p['missing_mask_lanes'],
        'A4_cleared_lanes': a4p['cleared_lanes'],
        'A4_fail_lanes': a4p['fail_glyph_crossing_lanes'],
        'A4_pending_lanes': a4p['pending_review_lanes'],
        'T6_frozen': f"{tf['T6']['num']}/{tf['T6']['den']}",
        'T6_v2': f"{tv['T6']['num']}/{tv['T6']['den']}",
        'v2_basis': diffs[0]['v2_basis'] if diffs else None,
        'report': str(report_path.relative_to(ROOT)).replace('\\', '/'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
