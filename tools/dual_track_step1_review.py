# -*- coding: utf-8 -*-
"""第一步补充复核脚本：集合对账、缺掩膜归因、掩膜留边像素核查与去重候选复核。

只读现有原图、预测与验收产物，不修改冻结标注与 v1 复核文件。
输出：output/dual_tracks/stage8/step1_review/endpoint_review_v2.json 及局部放大图。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dual_track_stage8_accept import IMG, ink_mask  # noqa: E402

DEFAULT_PRED = ROOT / 'output/dual_tracks/stage8/closure/predictions/path_online_closure.json'
DEFAULT_A4 = ROOT / 'output/dual_tracks/stage8/closure/acceptance/a4_detail.json'
DEFAULT_INK = ROOT / 'output/dual_tracks/stage8/closure/acceptance/ink_review.json'
DEFAULT_ANN = ROOT / 'output/dual_tracks/stage2/annotations.json'
DEFAULT_OUT_DIR = ROOT / 'output/dual_tracks/stage8/step1_review'
PAGE_ID = '丹方彚编/page_12'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def _imdecode(path: Path, flags: int) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return img


def _x_ov(a: List[int], b: List[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def _y_ov(a: List[int], b: List[int]) -> int:
    return max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _build_judgments() -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        ('丹方彚编/page_12#slot05#00:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 274, 'y2': 824},
            'basis': '原图可见「輕粉」之「粉」字末笔下探至 y≈270 跨过 y1=269；小字首字「明」自 y≈274 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot04#00:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 270, 'y2': 540},
            'basis': '掩膜移除右侧纵栏线（x≈787）后端点处无连续笔画，末字「真」在 y≈533 完整结束，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot06#01:right', 'bottom'): {
            'attribution': '粘连不清',
            'suggested_range': {'y1': 270, 'y2': 823},
            'basis': '8×像素检查：末字「真」底撇位于 y=821–822（x=664–669），y=823 存在 x=[663,664] 与「真」底撇相连且紧邻底版框（y=824–826 全掩膜，仅 x=660 留 2px 缺口）；1–2px 粘连归属无法可靠拆分，标为待核。',
            'confidence_status': '待核',
        },
        ('丹方彚编/page_12#slot07#00:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 272, 'y2': 822},
            'basis': '8×像素检查：x=[613,641] 内 y=818–822 仅右侧纵栏线 x=[636,637,639,640] 有墨迹，y=823 为未掩膜底版框上沿（ink_cnt=26, mask_cnt=0），框内无字形笔画。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot01#01:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 296, 'y2': 823},
            'basis': '掩膜移除底版框后端点处无跨界笔画，末字「解」完整，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot01#01:left', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 305, 'y2': 533},
            'basis': '上方药名「翠膽礬」之「礬」字底部笔画下探至 y≈303 跨过 y1=296；小字首字「皆」自 y≈305 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot00#01:right', 'top'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 298, 'y2': 821},
            'basis': '掩膜移除右侧纵栏线后无跨界笔画，首字「必」完整，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot00#01:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 298, 'y2': 821},
            'basis': '掩膜移除底版框及右侧纵栏线后无跨界笔画，框底为空白，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot08#00:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 334, 'y2': 822},
            'basis': '上方未标注药名「蘇合油香」之「香」字下部「日」跨过 y1=314（至 y≈332）；小字「只」自 y≈334 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot08#00:left', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 334, 'y2': 740},
            'basis': '同一未标注药名「蘇合油香」之「香」字斜笔/「日」部跨过 y1=314；左轨小字「上」自 y≈334 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot00#02:right', 'top'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 412, 'y2': 821},
            'basis': '掩膜移除右侧纵栏线后无跨界笔画，首字「冬」完整，掩膜未遮掉真实字形（注：该组为 slot00#01 内部重复短窗口）。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot00#02:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 412, 'y2': 821},
            'basis': '掩膜移除底版框及右侧纵栏线后无跨界笔画，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot08#02:right', 'top'): {
            'attribution': '小字笔画',
            'suggested_range': None,
            'basis': 'y1=482 切过小字「赤」字顶部笔画；slot08#02 为 slot08#00 内部嵌套的重复短窗口，非独立组起点。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot08#02:left', 'top'): {
            'attribution': '小字笔画',
            'suggested_range': None,
            'basis': 'y1=482 切过小字「于」字竖笔；slot08#02 为 slot08#00 内部嵌套的重复短窗口，非独立组起点。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot05#01:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 630, 'y2': 824},
            'basis': 'y1=591 切过未标注药名「丁香」之「香」字；小字自 y≈630 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot03#05:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 628, 'y2': 824},
            'basis': '上方「硃砂」之「砂」字末笔微跨 y1=620；小字「有」自 y≈628 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot03#05:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 628, 'y2': 824},
            'basis': '掩膜移除底版框后无跨界笔画，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot06#05:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 635, 'y2': 823},
            'basis': 'y1=630 切过未标注药名「丁香」之「香」字底笔；小字自 y≈635 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot06#05:right', 'bottom'): {
            'attribution': '粘连不清',
            'suggested_range': {'y1': 635, 'y2': 823},
            'basis': '8×像素检查：与 slot06#01:right 同处底版框位置，y=823 存在 x=[663,664] 与「真」底撇相连且紧邻底版框（x=660, y=824 留 2px 缺口）；1–2px 粘连归属无法可靠拆分，标为待核。',
            'confidence_status': '待核',
        },
        ('丹方彚编/page_12#slot06#05:left', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': {'y1': 635, 'y2': 823},
            'basis': 'y1=630 切过未标注药名「丁香」之「香」字底笔；左轨小字自 y≈635 起。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot02#02:right', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': None,
            'basis': 'y1=630 切过「陽起石」之「石」字横撇；小字自 y≈653 起。且 slot02#02 为与 slot01#joint00 冲突的错位单槽候选。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot02#02:left', 'top'): {
            'attribution': '药名笔画',
            'suggested_range': None,
            'basis': 'y1=630 切过「陽起石」之「石」字（与右轨同字）；且 slot02#02 为与 slot01#joint00 冲突的错位单槽候选。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot04#01:left', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 649, 'y2': 824},
            'basis': '掩膜移除底版框后无跨界笔画，末字「假」完整，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
        ('丹方彚编/page_12#slot01#joint00:right', 'bottom'): {
            'attribution': '独立栏线',
            'suggested_range': {'y1': 652, 'y2': 824},
            'basis': '掩膜移除底版框后无跨界笔画，末字「辨」完整，掩膜未遮掉真实字形。',
            'confidence_status': '已确认',
        },
    }


def run_review(pred_path: Path, a4_path: Path, ink_path: Path, ann_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir = out_dir / 'mask_residue_zoom'
    zoom_dir.mkdir(parents=True, exist_ok=True)

    pred_doc = json.loads(pred_path.read_text(encoding='utf-8'))
    a4_doc = json.loads(a4_path.read_text(encoding='utf-8'))
    ink_doc = json.loads(ink_path.read_text(encoding='utf-8'))
    ann_doc = json.loads(ann_path.read_text(encoding='utf-8'))

    p12_preds = {p['prediction_id']: p for p in pred_doc['predictions'] if p.get('page_id') == PAGE_ID}
    p12_run = next(r for r in pred_doc.get('page_runs', []) if r.get('page_id') == PAGE_ID)
    gts = [g for g in ann_doc.get('groups') or [] if g.get('page_id') == PAGE_ID]

    # 1. Sets A, P, B
    set_a = [item['evidence_id'] for item in a4_doc.get('cut_items', [])]
    set_p = [
        f"{pid}:{side}"
        for pid, p in p12_preds.items()
        for side in ('right', 'left')
        if p.get(f'{side}_bbox') and (p.get('side_status') or {}).get(side) == 'pending_validation'
    ]
    a_s, p_s = set(set_a), set(set_p)
    set_b = [eid for eid in set_p if eid not in a_s]
    b_s = set(set_b)

    sets_reconcile = {
        'A_raw_a4_trigger': {'count': len(set_a), 'evidence_ids': set_a},
        'P_runtime_pending_validation': {'count': len(set_p), 'evidence_ids': set_p},
        'B_pending_without_a4_trigger': {'count': len(set_b), 'evidence_ids': set_b},
        'A_intersect_P': {'count': len(a_s & p_s), 'evidence_ids': [x for x in set_a if x in p_s]},
        'A_intersect_B': {'count': len(a_s & b_s), 'evidence_ids': sorted(a_s & b_s)},
        'P_minus_A': {'count': len(p_s - a_s), 'evidence_ids': [x for x in set_p if x not in a_s]},
        'A_minus_P': {'count': len(a_s - p_s), 'evidence_ids': [x for x in set_a if x not in p_s]},
    }

    # 2. Missing mask reasons + 8x pixel inspection for mask-residue endpoints
    img = _imdecode(IMG, cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    missing_mask_audit = []
    mask_residue_audit = []
    residue_targets = {
        '丹方彚编/page_12#slot06#01:right',
        '丹方彚编/page_12#slot07#00:right',
        '丹方彚编/page_12#slot06#05:right',
    }
    for lane in ink_doc['lanes']:
        eid = lane['evidence_id']
        ev = lane['evidence'] or {}
        if lane['files'] is None:
            rm = ev.get('rule_mask_review')
            if rm is None:
                cat = '无栏线需要掩膜'
                why = 'ocr/dual_track.py::_write_rule_evidence 在待核端点 ±H(21px) 范围内统计 rule_mask 像素和为 0，故未生成掩膜附件。'
            elif not rm.get('mask_path') or not rm.get('masked_crop_path'):
                cat = '运行时未生成掩膜'
                why = 'rule_mask_review 存在但路径为空。'
            else:
                cat = '附件缺失'
                why = '掩膜路径记录存在但磁盘文件缺失或损坏。'
            missing_mask_audit.append({
                'evidence_id': eid,
                'bbox': lane['bbox'],
                'runtime_reason': ev.get('reason'),
                'category': cat,
                'detail': why,
            })

        if eid in residue_targets:
            x1, y1, x2, y2 = lane['bbox']
            rm = ev['rule_mask_review']
            cx1, cy1, cx2, cy2 = rm['crop_bbox']
            m_img = _imdecode(ROOT / rm['mask_path'], cv2.IMREAD_GRAYSCALE)
            c_img = _imdecode(ROOT / rm['masked_crop_path'], cv2.IMREAD_COLOR)
            mask_full = np.zeros(ink.shape, dtype=bool)
            mask_full[cy1:cy2, cx1:cx2] = m_img > 0
            cleared_full = img.copy()
            cleared_full[cy1:cy2, cx1:cx2] = c_img

            row_stats = []
            for yy in range(y2 - 4, y2 + 5):
                ink_xs = [xx for xx in range(x1, x2) if ink[yy, xx]]
                mask_xs = [xx for xx in range(x1, x2) if mask_full[yy, xx]]
                res_xs = [xx for xx in range(x1, x2) if ink[yy, xx] and not mask_full[yy, xx]]
                row_stats.append({
                    'y': yy, 'ink_pixels': len(ink_xs), 'mask_pixels': len(mask_xs),
                    'unmasked_ink_xs': res_xs,
                })

            zx1, zy1, zx2, zy2 = x1 - 10, y2 - 22, x2 + 10, y2 + 12
            p1 = img[zy1:zy2, zx1:zx2].copy()
            p2 = cv2.cvtColor((mask_full[zy1:zy2, zx1:zx2].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
            p3 = cleared_full[zy1:zy2, zx1:zx2].copy()
            panels = []
            for pp in (p1, p2, p3):
                pz = cv2.resize(pp, (pp.shape[1] * 8, pp.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
                ry = (y2 - zy1) * 8
                bx1, bx2 = (x1 - zx1) * 8, (x2 - zx1) * 8
                cv2.line(pz, (0, ry), (pz.shape[1] - 1, ry), (0, 0, 255), 1)
                cv2.line(pz, (bx1, 0), (bx1, pz.shape[0] - 1), (255, 0, 0), 1)
                cv2.line(pz, (bx2, 0), (bx2, pz.shape[0] - 1), (255, 0, 0), 1)
                panels.append(pz)
            safe = eid.replace('丹方彚编/page_12#', '').replace(':', '_')
            z_rel = f'output/dual_tracks/stage8/step1_review/mask_residue_zoom/{safe}_bottom_8x.png'
            cv2.imencode('.png', np.hstack(panels))[1].tofile(str(ROOT / z_rel))
            mask_residue_audit.append({
                'evidence_id': eid,
                'bbox': lane['bbox'],
                'zoom_crop_bbox': [zx1, zy1, zx2, zy2],
                'zoom_image': z_rel,
                'row_stats': row_stats,
            })

    # 3. Three duplicate candidate pairs
    dup_pairs_spec = [
        ('丹方彚编/page_12#slot00#02', '丹方彚编/page_12#slot00#01',
         '同槽同中缝内部嵌套短窗口（100%纵向被覆盖，底边同为821），属于同一组「滴乳石」小字；去除不损失独立小字段落。'),
        ('丹方彚编/page_12#slot08#02', '丹方彚编/page_12#slot08#00',
         '同槽同中缝内部嵌套短窗口（100%纵向被覆盖，底边同为822），y1=482腰斩小字「赤」「于」；属于同一组「蘇合油香」小字；去除不损失独立小字段落。'),
        ('丹方彚编/page_12#slot02#02', '丹方彚编/page_12#slot01#joint00',
         '单槽错位候选：slot02#02 在 slot02 内误取假缝 x=858，与跨槽联合组 slot01#joint00（真中缝 x=882，「陽起石」DF12-11）纵向重叠 87.1% 且顶部切过「石」字；去除不损失独立小字段落。'),
    ]
    duplicate_candidates_audit = []
    for cand_id, keep_id, reason in dup_pairs_spec:
        c, k = p12_preds[cand_id], p12_preds[keep_id]
        cg, kg = c['group_bbox'], k['group_bbox']
        y_ov = _y_ov(cg, kg)
        c_h, k_h = cg[3] - cg[1], kg[3] - kg[1]

        def _support(p_obj: Dict[str, Any]) -> Dict[str, Any]:
            rb, lb = p_obj['right_bbox'], p_obj['left_bbox']
            return {
                'gutter_x': (lb[2] + rb[0]) / 2.0,
                'right_bbox': rb,
                'left_bbox': lb,
                'right_ink_pixels': int(ink[rb[1]:rb[3], rb[0]:rb[2]].sum()),
                'left_ink_pixels': int(ink[lb[1]:lb[3], lb[0]:lb[2]].sum()),
                'side_status': p_obj.get('side_status'),
            }

        duplicate_candidates_audit.append({
            'candidate_id': cand_id,
            'keep_id': keep_id,
            'candidate_group_bbox': cg,
            'keep_group_bbox': kg,
            'y_overlap_px': y_ov,
            'y_overlap_ratio_of_candidate': round(y_ov / c_h, 4),
            'y_overlap_ratio_of_keep': round(y_ov / k_h, 4),
            'candidate_support': _support(c),
            'keep_support': _support(k),
            'keep_joint_evidence': k.get('joint_evidence'),
            'same_group_or_conflict_reason': reason,
            'loses_independent_paragraph_if_removed': False,
        })

    # 4. Build endpoint records (v2)
    judgments = _build_judgments()
    records = []
    for idx, lane in enumerate(ink_doc['lanes'], 1):
        eid = lane['evidence_id']
        pid, side = eid.split(':')
        p = p12_preds[pid]
        ev = lane['evidence']
        mate_side = 'left' if side == 'right' else 'right'
        rev_box = lane['bbox']
        nearby_titles = [
            {'group_id': g['group_id'], 'main_text': g.get('main_text') or g.get('text', ''), 'main_bbox': g['main_bbox']}
            for g in gts if g.get('main_bbox') and _x_ov(rev_box, g['main_bbox']) > 0
        ]
        nearby_groups = [
            {'prediction_id': opid, 'group_bbox': op['group_bbox']}
            for opid, op in p12_preds.items() if opid != pid and _x_ov(rev_box, op['group_bbox']) > 0
        ]
        ep_records = {}
        for ep_name, ep_val in lane['endpoints'].items():
            j = judgments[(eid, ep_name)]
            safe_id = eid.replace('丹方彚编/page_12#', '').replace(':', '_')
            panel_rel = f'output/dual_tracks/stage8/step1_review/crops/{idx:02d}_{safe_id}_{ep_name}.png'
            ep_records[ep_name] = {
                'raw_a4_overlap': ep_val['raw_overlap'],
                'clean_overlap': ep_val['clean_overlap'],
                'remaining_crossing_components': ep_val['remaining_crossing_components'],
                'marked_pixels_at_boundary': ep_val['marked_pixels_at_boundary'],
                'ink_review_verdict': ep_val['verdict'],
                'attribution': j['attribution'],
                'suggested_range': j['suggested_range'],
                'basis': j['basis'],
                'confidence_status': j['confidence_status'],
                'panel_image': panel_rel,
            }
        rm = ev.get('rule_mask_review') or {}
        records.append({
            'index': idx,
            'evidence_id': eid,
            'page_id': PAGE_ID,
            'group_id': pid,
            'slot_id': pid.split('#')[1],
            'side': side,
            'in_A_raw_a4_trigger': eid in a_s,
            'in_P_runtime_pending': eid in p_s,
            'in_B_pending_only': eid in b_s,
            'original_bbox': ev.get('original_bbox') or rev_box,
            'reviewed_bbox': rev_box,
            'mate_bbox': p.get(f'{mate_side}_bbox'),
            'nearby_title_boxes': nearby_titles,
            'nearby_groups': nearby_groups,
            'runtime_pending_reason': ev.get('reason'),
            'review_actions': ev.get('actions') or [],
            'evidence_images': {
                'original_review_crop': ev.get('image_path'),
                'rule_mask': rm.get('mask_path'),
                'deruled_crop': rm.get('masked_crop_path'),
            },
            'endpoints': ep_records,
        })

    attr_counts: Dict[str, int] = {}
    conf_counts: Dict[str, int] = {}
    pending_endpoints: List[str] = []
    for r in records:
        for ep_name, ep in r['endpoints'].items():
            attr_counts[ep['attribution']] = attr_counts.get(ep['attribution'], 0) + 1
            conf_counts[ep['confidence_status']] = conf_counts.get(ep['confidence_status'], 0) + 1
            if ep['confidence_status'] != '已确认':
                pending_endpoints.append(f"{r['evidence_id']}:{ep_name}")

    out_doc = {
        'version': 'step1_endpoint_review_v2',
        'annotator': 'Antigravity',
        'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'git_head': _git_head(),
        'runtime_effective_config': p12_run.get('effective'),
        'coordinate_system': '原图像素坐标 [x1, y1, x2, y2]，左闭右开（x 向右递增，y 向下递增）',
        'inputs': {
            'source_image': str(IMG.relative_to(ROOT)).replace('\\', '/'),
            'source_image_sha256': _sha256(IMG),
            'prediction_source': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
            'prediction_sha256': _sha256(pred_path),
            'a4_source': str(a4_path.relative_to(ROOT)).replace('\\', '/'),
            'a4_sha256': _sha256(a4_path),
            'ink_review_source': str(ink_path.relative_to(ROOT)).replace('\\', '/'),
            'ink_review_sha256': _sha256(ink_path),
        },
        'sets_reconcile': sets_reconcile,
        'missing_mask_audit': missing_mask_audit,
        'mask_residue_audit': mask_residue_audit,
        'duplicate_candidates_audit': duplicate_candidates_audit,
        'summary': {
            'total_checked_lanes': a4_doc['n_checked'],
            'count_A': len(set_a),
            'count_P': len(set_p),
            'count_B': len(set_b),
            'count_A_intersect_P': len(a_s & p_s),
            'count_A_intersect_B': len(a_s & b_s),
            'count_P_minus_A': len(p_s - a_s),
            'total_review_endpoints': sum(len(r['endpoints']) for r in records),
            'attribution_counts_by_endpoint': attr_counts,
            'confidence_counts_by_endpoint': conf_counts,
            'pending_endpoints': pending_endpoints,
        },
        'records': records,
    }

    out_file = out_dir / 'endpoint_review_v2.json'
    out_file.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return out_file


def main() -> int:
    ap = argparse.ArgumentParser(description='第一步补充复核脚本')
    ap.add_argument('--pred', type=Path, default=DEFAULT_PRED)
    ap.add_argument('--a4', type=Path, default=DEFAULT_A4)
    ap.add_argument('--ink', type=Path, default=DEFAULT_INK)
    ap.add_argument('--ann', type=Path, default=DEFAULT_ANN)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out_file = run_review(args.pred, args.a4, args.ink, args.ann, args.out_dir)
    doc = json.loads(out_file.read_text(encoding='utf-8'))
    print('Wrote:', out_file)
    print('Output sha256:', _sha256(out_file))
    print('Summary:', json.dumps(doc['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
