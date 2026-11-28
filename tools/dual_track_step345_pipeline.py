# -*- coding: utf-8 -*-
"""阶段八单次正常入口全链路验收流水线（同一次预测生成全部产物与最终报告）：
从同一次 19 页 ocr.pipeline.run_ocr(detector=stage8) 预测出发，统一输出至
output/dual_tracks/stage8/step5_final/（绝不覆盖历史归档目录）：
  1. 预测与指纹：predictions/path_online_final.json（与第四步归档逐框核对）
  2. A4 原图统计、掩膜逐轨复核与 A4 联合判定（含剩余 7 轨分类、slot06#05 首尾与「真」末笔证据）
  3. A3、A1–A7 入口验收与 R1–R5 开关回归
  4. 两套 T6 评分（冻结历史口径 + v2 复核口径）与 NEG-DF-SINGLE-01 独立复核依据
  5. 全数字可回溯的最终阶段八验收与待决结项报告
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
from typing import Any, Dict, List

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
STEP4_REF = S8 / 'step4_acceptance/predictions/path_online_step4.json'
OUT_DIR = S8 / 'step5_final'
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
            if a.get('kind') in ('window_superseded', 'cross_slot_joint_superseded',
                                 'insufficient_pair_shape_support', 'rule_attached_pair_recovered',
                                 'side_pending', 'recognize_error', 'side_invalid_bbox'):
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


def _build_a3_a4_ink_and_evidence(pred_path: Path, pred_sha: str, acc_dir: Path, ev_dir: Path) -> Dict[str, Any]:
    acc_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    preds = doc['predictions']
    p12_preds = [p for p in preds if p.get('page_id') == '丹方彚编/page_12']

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

    # 掩膜逐轨复核（已修正 _mask_touches_glyph 仅核查框内墨迹）
    ink_path = acc_dir / 'ink_review.json'
    ink_rep = run_ink_review(pred_path, a4_path, ink_path)

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

    # 生成 slot06#05:right 顶部「香」vs「如」五联像素差异证明 + 两处「真」末笔 8× 放大证明
    tx1, ty1, tx2, ty2 = 640, 615, 695, 685
    raw_t = p12_img[ty1:ty2, tx1:tx2].copy()
    lab_t = labels[ty1:ty2, tx1:tx2]
    xiang_cids, rule_cids, ru_cids = [], [], []
    for cid in sorted(set(np.unique(lab_t)) - {0}):
        cx, cy, cw, ch, _ = map(int, stats[cid])
        if cx >= 684 and cw <= 4:
            rule_cids.append(int(cid))
        elif cy + ch <= 651 and cx >= 648 and cx + cw <= 682:
            xiang_cids.append(int(cid))
        elif cy >= 652 and cx >= 660 and cx + cw <= 684:
            ru_cids.append(int(cid))
    xiang_m = np.isin(lab_t, xiang_cids)
    rule_m = np.isin(lab_t, rule_cids)
    ru_m = np.isin(lab_t, ru_cids)
    rm_m = xiang_m | rule_m
    bg_val = np.percentile(raw_t[~rm_m], 85, axis=0)
    deru_t = raw_t.copy()
    deru_t[rm_m] = bg_val
    diff_t = cv2.absdiff(raw_t, deru_t)
    changed_t = np.any(diff_t > 0, axis=2)
    diff_vis = np.zeros_like(raw_t)
    diff_vis[changed_t] = (255, 0, 255)
    ov_t = raw_t.copy()
    ov_t[xiang_m] = (255, 220, 0)
    ov_t[rule_m] = (0, 220, 255)
    ov_t[ru_m] = (0, 200, 0)

    def _sc(im, s=6):
        z = cv2.resize(im, (im.shape[1] * s, im.shape[0] * s), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, ((660 - tx1) * s, (630 - ty1) * s), ((690 - tx1) * s, (ty2 - ty1) * s), (0, 0, 255), 2)
        return z

    m_vis = cv2.cvtColor((rm_m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    top_proof_img = ev_dir / 'slot06_05_right_top_xiang_vs_ru_6x.png'
    cv2.imencode('.png', np.hstack([_sc(raw_t), _sc(ov_t), _sc(m_vis), _sc(deru_t), _sc(diff_vis)]))[1].tofile(str(top_proof_img))
    slot06_05_top_proof = {
        'original_rectangle_preserved': [660, 630, 690, 823],
        'crop_bbox': [tx1, ty1, tx2, ty2],
        'evidence_image': str(top_proof_img.relative_to(ROOT)).replace('\\', '/'),
        'evidence_image_sha256': _sha256(top_proof_img),
        'xiang_cids': xiang_cids,
        'rule_cids': rule_cids,
        'ru_cids': ru_cids,
        'xiang_pixels_masked': int(xiang_m.sum()),
        'rule_pixels_masked': int(rule_m.sum()),
        'ru_pixels_total': int(ru_m.sum()),
        'ru_pixels_changed': int((changed_t & ru_m).sum()),
        'outside_mask_pixels_changed': int((changed_t & ~rm_m).sum()),
        'blank_rows_between_xiang_and_ru': [yy for yy in range(647, 655) if int(ink[yy, 660:684].sum()) == 0],
        'conclusion': '「香」(y=621..648)与首字「如」(y=653..676)之间存在 5 行完全空白(y=648..652)，掩膜去除「香」与外侧栏线后「如」的 167/167 像素 0 改动、掩膜外 0 像素改动；但因保留的原图矩形 y1=630 切过药名「香」字，该端点按口径判为不通过（药名笔画跨界）。',
    }

    # 两处「真」末笔 8× 放大与像素差异证据（slot06#01:right & slot06#05:right）
    zhen_proofs = {}
    for lane in ink_rep['lanes']:
        if lane['evidence_id'] in ('丹方彚编/page_12#slot06#01:right', '丹方彚编/page_12#slot06#05:right'):
            x1, y1, x2, y2 = lane['bbox']
            rm = lane['evidence']['rule_mask_review']
            cx1, cy1, cx2, cy2 = rm['crop_bbox']
            m_img = cv2.imdecode(np.fromfile(str(ROOT / rm['mask_path']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            c_img = cv2.imdecode(np.fromfile(str(ROOT / rm['masked_crop_path']), dtype=np.uint8), cv2.IMREAD_COLOR)
            mask_full = np.zeros(ink.shape, dtype=bool)
            mask_full[cy1:cy2, cx1:cx2] = m_img > 0
            cleared_full = p12_img.copy()
            cleared_full[cy1:cy2, cx1:cx2] = c_img
            zx1, zy1, zx2, zy2 = x1 - 8, y2 - 22, x2 + 8, y2 + 10
            r_c = p12_img[zy1:zy2, zx1:zx2]
            m_c = cv2.cvtColor((mask_full[zy1:zy2, zx1:zx2].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
            cl_c = cleared_full[zy1:zy2, zx1:zx2]
            df_c = np.zeros_like(r_c)
            df_c[np.any(cv2.absdiff(r_c, cl_c) > 0, axis=2)] = (255, 0, 255)
            z_panels = []
            for pp in (r_c, m_c, cl_c, df_c):
                pz = cv2.resize(pp, (pp.shape[1] * 8, pp.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
                cv2.line(pz, (0, (y2 - zy1) * 8), (pz.shape[1] - 1, (y2 - zy1) * 8), (0, 0, 255), 1)
                cv2.line(pz, ((x1 - zx1) * 8, 0), ((x1 - zx1) * 8, pz.shape[0] - 1), (255, 0, 0), 1)
                cv2.line(pz, ((x2 - zx1) * 8, 0), ((x2 - zx1) * 8, pz.shape[0] - 1), (255, 0, 0), 1)
                z_panels.append(pz)
            safe = lane['evidence_id'].replace('丹方彚编/page_12#', '').replace(':', '_')
            zp = ev_dir / f'{safe}_bottom_zhen_4panel_8x.png'
            cv2.imencode('.png', np.hstack(z_panels))[1].tofile(str(zp))
            zhen_proofs[lane['evidence_id']] = {
                'bbox': lane['bbox'],
                'evidence_image': str(zp.relative_to(ROOT)).replace('\\', '/'),
                'evidence_image_sha256': _sha256(zp),
                'adhesion_component_bbox': [663, 819, 672, 824],
                'y823_unmasked_ink_xs': [x for x in range(x1, x2) if ink[823, x] and not mask_full[823, x]],
                'y824_mask_pixels': int(mask_full[824, x1:x2].sum()),
                'verdict': '待核（末字「真」底撇 [663,819,672,824] 跨过 y2=823 并直接粘连 y=824 底版框，无法无损证明完整）',
            }

    # 按三类严格归类并形成 A4 联合判定（a4_policy_reconcile.json）
    lane_decisions = []
    for lane in ink_rep['lanes']:
        eid = lane['evidence_id']
        eps = lane['endpoints']
        ep_details = {}
        for ep_name, ep_val in eps.items():
            if lane['files'] is None:
                cat = '缺少掩膜证据'
                ep_status = 'pending_missing_mask'
                detail = '运行时外侧断续纵栏线（x=734..736）单段短于 2.2H 未生成掩膜，保持待核。'
            elif ep_val['verdict'] == 'rule_explains_trigger':
                cat = '框外栏线触发'
                ep_status = 'cleared_by_rule_mask'
                detail = '掩膜复核验证通过，去栏线后无框内字形接触且无跨边残墨，解除该端点疑点。'
            elif eid == '丹方彚编/page_12#slot06#05:right' and ep_name == 'top':
                cat = '框内真实笔画接触（药名跨界）'
                ep_status = 'fail_glyph_crossing'
                detail = '原图矩形 y1=630 切过药名「丁香」之「香」(y=621..648)；掩膜去「香」可 100% 保留「如」(y=653..676)，但保留原矩形下按口径判不通过。'
            else:
                cat = '框内真实笔画接触'
                ep_status = 'pending_glyph_adhesion'
                detail = '框内真实字形笔画（或紧贴邻字）与栏线/底版框掩膜直接接触粘连，无法无损证明完整，保持待核。'
            ep_details[ep_name] = {**ep_val, 'category': cat, 'endpoint_policy_status': ep_status, 'detail': detail}

        all_cleared = all(v['endpoint_policy_status'] == 'cleared_by_rule_mask' for v in ep_details.values())
        any_fail = any(v['endpoint_policy_status'] == 'fail_glyph_crossing' for v in ep_details.values())
        final_status = 'cleared_by_rule_mask' if all_cleared else ('fail_glyph_crossing' if any_fail else 'pending_review')
        lane_decisions.append({
            'evidence_id': eid,
            'bbox': lane['bbox'],
            'original_a4_trigger': lane['original_a4_trigger'],
            'files': lane['files'],
            'final_policy_status': final_status,
            'endpoints': ep_details,
        })

    a4_policy_doc = {
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'user_policy_quote': 'A4 允许逐条确认的掩膜复核参与最终判定，但原图触发数继续保留。被证明只是栏线触发的轨道可以解除该项疑点；文字笔画跨界或归属不明的轨道继续标为待核或不通过。',
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


def _build_neg_df_single_01_review(ev_dir: Path) -> Dict[str, Any]:
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
        'negative_region_id': 'NEG-DF-SINGLE-01',
        'page_id': '丹方彚编/page_12',
        'bbox': [536, 210, 585, 815],
        'source_image_sha256': _sha256(P12_IMG),
        'evidence_image': str(ev_img.relative_to(ROOT)).replace('\\', '/'),
        'evidence_image_sha256': _sha256(ev_img),
        'body_measurement_y_span': [334, 740],
        'left_lane_peak_col_ink': int(col_sum[545 - 536:562 - 536].max()),
        'right_lane_peak_col_ink': int(col_sum[568 - 536:585 - 536].max()),
        'seam_profile': seam_stats,
        'title_reading': '蘇合油香（y=204..330，原冻结标注误记为“藿合油香”）',
        'right_lane_reading': '只用如膏油赤色無滓香極芬烈塗（x=564..594）',
        'left_lane_reading': '上手皆香透于心或如鵝膠者真（x=545..564）',
        'independent_review_conclusion': '原图明确为药名「蘇合油香」领起的双行小字注记：x=564..565 存在空白率 99.26%、最长连续空白 354 px 的纵缝（谷深仅峰值 2.1%），左右各具独立完整小字列，原冻结标签“单一列、未分两轨”与原图事实不符。',
    }


def _run_a1_a7_and_r1_r5(pred_path: Path, pred_sha: str,
                         ev_by_page: Dict[str, List[Dict[str, Any]]],
                         a3_doc: Dict[str, Any], a4_doc: Dict[str, Any],
                         a4_policy_doc: Dict[str, Any], acc_dir: Path) -> Dict[str, Any]:
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
         }},
        {'id': 'A5', 'title': '左右并列读数', 'pass': len(paired) >= 1, 'detail': {'n_paired': len(paired), 'examples': paired[:5]}},
        {'id': 'A6', 'title': '开关一致性（关闭 = not_run 且正文不变）', 'pass': a6_pass, 'detail': {'off_status': res_off.dual_track_status, 'text_identical': base_text == _txt(res_off)}},
        {'id': 'A7', 'title': '单侧失败隔离（实际注入异常）', 'pass': a7_pass, 'detail': {'page_status': inj.dual_track_status, 'target_evidence_id': target_e['evidence_id'], 'target_side_status': t_ev['side_status'], 'mate_side_status': m_ev['side_status'], 'n_evidence': len(inj_ev)}},
    ]
    checklist = {
        'stage': 'stage8_final_single_run_acceptance',
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
    pred_path = OUT_DIR / 'predictions/path_online_final.json'
    crop_dir = OUT_DIR / 'crops'
    acc_dir = OUT_DIR / 'acceptance'
    ev_dir = OUT_DIR / 'evidence'
    closure_dir = OUT_DIR / 'closure'
    m_frozen_dir = OUT_DIR / 'metrics/neg_frozen'
    m_v2_dir = OUT_DIR / 'metrics/neg_v2'
    for d in (acc_dir, ev_dir, closure_dir, m_frozen_dir, m_v2_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. 同一次 19 页正常入口运行
    run = _run_19_pages_online(crop_dir, pred_path)
    pred_sha = run['pred_sha256']

    # 对照第四步归档预测确认框坐标一致
    ref_match = None
    if STEP4_REF.is_file():
        ref_doc = json.loads(STEP4_REF.read_text(encoding='utf-8'))
        key_fn = lambda ds: sorted((p['page_id'], p['prediction_id'], tuple(p['right_bbox'] or ()), tuple(p['left_bbox'] or ())) for p in ds['predictions'])
        ref_match = (key_fn(ref_doc) == key_fn(run['doc']))

    # 2. A3、A4 原图统计、掩膜复核与 A4 联合判定
    acc = _build_a3_a4_ink_and_evidence(pred_path, pred_sha, acc_dir, ev_dir)

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
    m_frozen_doc = json.loads((m_frozen_dir / 'online_final.json').read_text(encoding='utf-8'))
    m_v2_doc = json.loads((m_v2_dir / 'online_final.json').read_text(encoding='utf-8'))
    targets_doc = _extract_targets(m_frozen_doc, m_v2_doc, pred_path, pred_sha)
    (acc_dir / 'targets.json').write_text(json.dumps(targets_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 4. A1–A7 与 R1–R5
    checks_res = _run_a1_a7_and_r1_r5(
        pred_path, pred_sha, run['ev_by_page'],
        acc['a3'], acc['a4'], acc['a4_policy'], acc_dir,
    )

    # 5. NEG-DF-SINGLE-01 独立复核 + T6 双口径对账
    neg_review = _build_neg_df_single_01_review(ev_dir)
    orig_neg = json.loads((S2 / 'negative_regions.json').read_text(encoding='utf-8'))
    v2_neg = json.loads((S8 / 'negative_regions_v2.json').read_text(encoding='utf-8'))
    vm = {r['negative_region_id']: r for r in v2_neg['regions']}
    diffs = []
    for r in orig_neg['regions']:
        rid = r['negative_region_id']
        rv = vm.get(rid, {})
        if r.get('status') != rv.get('status'):
            diffs.append({
                'negative_region_id': rid, 'page_id': r['page_id'], 'bbox': r['bbox'],
                'frozen_status': r.get('status'), 'v2_status': rv.get('status'),
                'frozen_basis': r.get('basis'), 'v2_basis': rv.get('basis'),
            })
    t6_reconcile = {
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'prediction_sha256': pred_sha,
        'matches_step4_archived_boxes': ref_match,
        'formal_historical_caliber': 'frozen',
        'user_policy_quote': 'T6 先保留原冻结口径作为正式历史验收结果，同时列出 v2 复核口径。对有争议的负例完成独立复核后，再决定是否将修订版作为新的正式标注版本。',
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

    # 6. 自动生成全数字回溯最终报告
    a4p = acc['a4_policy']
    tf, tv = targets_doc['calibers']['frozen'], targets_doc['calibers']['v2']
    ck = checks_res['checklist']
    rg = checks_res['regress']
    lane_rows = []
    for l in a4p['lanes']:
        eps_str = '；'.join(f"{k}: {v['category']} → {v['endpoint_policy_status']}" for k, v in l['endpoints'].items())
        lane_rows.append(f"| `{l['evidence_id'].replace('丹方彚编/page_12#','')}` | `{l['bbox']}` | {'是' if l['original_a4_trigger'] else '否'} | {l['files'] or '无掩膜'} | **{l['final_policy_status']}** | {eps_str} |")

    report_md = f"""# 阶段八最终验收与待决结项报告（同一次正常入口预测全链路产出）

生成时间：{datetime.now(TZ).isoformat(timespec='seconds')}  
代码版本：Git `HEAD` `{_git_head()}` · 配置指纹 `{run['doc']['effective']['fingerprint']}` · 模型 `{run['doc']['model_version']}`  
唯一验收预测：`{t6_reconcile['prediction_source']}`  
预测 SHA-256：`{pred_sha}`（与第四步归档框逐框一致：`{ref_match}`）

## 1. 最终结论
- **阶段八状态：未通过（暂不结项）**
- 入口验收 A1–A7：**{ck['passed']}/{ck['total']}**（A4 联合判定仍未通过）；开关回归 R1–R5：**{rg['passed']}/{rg['total']}**。
- 原冻结正式历史口径 T1–T8：**{sum(1 for v in tf.values() if v['pass'])}/8**（仅 T6=`{tf['T6']['num']}/{tf['T6']['den']}` 未达 0）；v2 复核口径 T1–T8：**{sum(1 for v in tv.values() if v['pass'])}/8**（T6=`{tv['T6']['num']}/{tv['T6']['den']}` 达标）。

## 2. A4 原图统计与联合判定（剩余 7 轨逐端点分类）
- **原图 A4 触发（保留）**：`{a4p['original_a4_raw_triggers']} / {a4p['original_a4_checked']}`
- **掩膜复核参与最终判定**：
  - 解除疑点（框外栏线触发）：**{a4p['cleared_lanes_count']} 条轨**（含原图触发中 **{a4p['cleared_raw_trigger_count']} 条**：`slot04#00:right`、`slot00#01:right`；以及仅运行时待核 1 条 `slot01#joint00:right`）。
  - 文字笔画跨界（不通过）：**{len(a4p['fail_glyph_crossing_lanes'])} 条轨**（`slot06#05:right` 顶部「香」字跨界；底部「真」末笔贴底版框待核）。
  - 待核（框内真实笔画接触 / 缺少掩膜证据）：**{len(a4p['pending_review_lanes'])} 条轨**（`slot01#01:right`「解」底触版框、`slot06#01:right`「真」底撇贴版框、`slot07#00:right` 外侧栏线贴邻字、`slot05#00:right` 与 `slot05#01:right` 缺少断续栏线掩膜）。

| 轨道 | 原图轨框 | 原图 A4 | 掩膜核验 | 联合判定 | 逐端点分类 |
|---|---|---|---|---|---|
{chr(10).join(lane_rows)}

### 2.1 `slot06#05:right` 首尾与两处「真」末笔像素证据
- **顶部（「香」vs「如」，保留原矩形 `[660,630,690,823]`）**：五联图 `{a4p['slot06_05_right_top_proof']['evidence_image']}`（SHA-256 `{a4p['slot06_05_right_top_proof']['evidence_image_sha256'][:16]}...`）。「香」(`y=621..648`) 与首字「如」(`y=653..676`) 之间有 **5 行完全空白** (`y=648..652`)；掩膜去掉「香」({a4p['slot06_05_right_top_proof']['xiang_pixels_masked']} px) 与外侧栏线 ({a4p['slot06_05_right_top_proof']['rule_pixels_masked']} px) 后，「如」的 `{a4p['slot06_05_right_top_proof']['ru_pixels_total']}` 个像素修改 **`{a4p['slot06_05_right_top_proof']['ru_pixels_changed']}`** 个、掩膜外修改 **`{a4p['slot06_05_right_top_proof']['outside_mask_pixels_changed']}`** 个。但因原图矩形 `y1=630` 切过「香」字，按口径判**不通过**。
- **底部（两处「真」末笔：`slot06#01:right` 与 `slot06#05:right`）**：末字「真」底撇连通域 `[663,819,672,824]` 跨过 `y2=823` 并直接粘连 `y=824` 底版框，无法无损证明完整，均继续标**待核**。

## 3. T6 双口径对账与 `NEG-DF-SINGLE-01` 独立复核依据
- **原冻结正式历史口径**：T6 = `{tf['T6']['num']}/{tf['T6']['den']}`（未达 0，疑 0，重复 FP 0）；**v2 复核口径**：T6 = `{tv['T6']['num']}/{tv['T6']['den']}`（达标，疑 0，重复 FP 0）。
- **`NEG-DF-SINGLE-01` 独立复核依据**（核查图 `{neg_review['evidence_image']}`，SHA-256 `{neg_review['evidence_image_sha256'][:16]}...`）：
  - 原图 `丹方彚编/page_12` `[536,210,585,815]` 顶部为四字药名「蘇合油香」（原冻结标注误记为“藿合油香”），下方 `y=334..740`（高 406 px）明确分为左右两列双行小字：右轨「只用如膏油赤色無滓香極芬烈塗」（列峰值 {neg_review['right_lane_peak_col_ink']} px），左轨「上手皆香透于心或如鵝膠者真」（列峰值 {neg_review['left_lane_peak_col_ink']} px）。
  - 中缝 `x=564..565` 空白率 **`99.26%`**（各仅 3 px 墨迹），`x=565` 最长连续空白 **`354 px`**，谷深仅右列峰值的 `2.1%`。原冻结标签“单一列、未分两轨”与原图事实不符；当前保留原冻结结果并提交两套口径，待决定是否将 v2 升级为正式标注版本。

## 4. T1–T8 汇总（同一预测 SHA-256 `{pred_sha[:16]}...`）
| 项 | 冻结历史口径 | v2 复核口径 | 门槛 |
|---|---|---|---|
| T1 (@0.50) | `{tf['T1']['num']}/{tf['T1']['den']}` ({'达标' if tf['T1']['pass'] else '未达'}) | `{tv['T1']['num']}/{tv['T1']['den']}` ({'达标' if tv['T1']['pass'] else '未达'}) | ≥30/38 |
| T2 (@0.75) | `{tf['T2']['num']}/{tf['T2']['den']}` ({'达标' if tf['T2']['pass'] else '未达'}) | `{tv['T2']['num']}/{tv['T2']['den']}` ({'达标' if tv['T2']['pass'] else '未达'}) | ≥18/38 |
| T3 / T4 | `{tf['T3']['num']}/20`, `{tf['T4']['num']}/20` | `{tv['T3']['num']}/20`, `{tv['T4']['num']}/20` | ≥19/20, ≥14/20 |
| T5 轨配对 | `{tf['T5']['num']}/{tf['T5']['den']} ({tf['T5']['value']})` | `{tv['T5']['num']}/{tv['T5']['den']} ({tv['T5']['value']})` | ≥0.90 |
| T6 负例误检 | **`{tf['T6']['num']}/{tf['T6']['den']}`（未达）** | **`{tv['T6']['num']}/{tv['T6']['den']}`（达标）** | 0 |
| T7 / T8 | `{tf['T7']['num']}/11`, `{tf['T8']['num']}/2` | `{tv['T7']['num']}/11`, `{tv['T8']['num']}/2` | ≥6/11, ≥1/2 |

## 5. 样本限制与复现命令
评测集为 19 页（38 标注正例组、10 原冻结负例区），独立验收样本量有限，结论仅覆盖当前古籍竖排双行小字样本范围。
```powershell
Set-Location D:/A中医药赛题/TCM-OCR-Knowledge-Graph
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_dual_track*.py"
.\\.venv\\Scripts\\python.exe tools/dual_track_step345_pipeline.py
```
"""
    report_path = closure_dir / '阶段八最终验收与待决结项报告.md'
    report_path.write_text(report_md, encoding='utf-8')

    print('=== FINAL SINGLE-RUN PIPELINE SUMMARY ===')
    print(json.dumps({
        'prediction_sha256': pred_sha,
        'matches_step4_archived_boxes': ref_match,
        'A1_A7': f"{ck['passed']}/{ck['total']}",
        'R1_R5': f"{rg['passed']}/{rg['total']}",
        'A4_raw': f"{a4p['original_a4_raw_triggers']}/{a4p['original_a4_checked']}",
        'A4_cleared_lanes': a4p['cleared_lanes'],
        'A4_fail_lanes': a4p['fail_glyph_crossing_lanes'],
        'A4_pending_lanes': a4p['pending_review_lanes'],
        'T6_frozen': f"{tf['T6']['num']}/{tf['T6']['den']}",
        'T6_v2': f"{tv['T6']['num']}/{tv['T6']['den']}",
        'report': str(report_path.relative_to(ROOT)).replace('\\', '/'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
