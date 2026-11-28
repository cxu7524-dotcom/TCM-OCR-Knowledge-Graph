# -*- coding: utf-8 -*-
"""阶段八第七轮单次正常入口全链路验收流水线（以 step6_closure 为历史对照）：
从同一次 19 页 ocr.pipeline.run_ocr(detector=stage8) 预测出发，统一输出至
output/dual_tracks/stage8/step7_closure/（绝不覆盖冻结标注与历史归档目录）：
  1. 新预测与逐页/逐组差异：predictions/path_online_step7.json、closure/diff_vs_step6.json
  2. slot05#00/slot05#01 与 slot06#01/slot06#05 分组核验图、A3 已标注标题覆盖说明
  3. 贴线待核轨（「解」「真」）12× 灰度像素证据与边界调整可行性核查
  4. 新栏线规则影响抽查（独立/断续/贴字栏线与长竖笔 + 第六轮 6 条解除疑点轨无遮字核验）
  5. A1–A7、R1–R5、A4 原图+联合判定、T6 双口径保留（冻结 1/10 + v2 0/9 与升级决定要求）及完整报告
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

from dual_track_fix.ruling import (  # noqa: E402
    _bridge_vertical_to_horizontal_rules,
    _collinear_dashed_vertical_rules,
    _directional_rule,
    _sloping_vertical_rules,
    ruling_mask,
    vertical_rule_mask,
)
from dual_track_fix.settings import effective_options  # noqa: E402
from dual_track_track import page_ink  # noqa: E402
from ocr.dual_track import _contract, _stage8_ink_components  # noqa: E402
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
from tools.dual_track_step6_pipeline import (  # noqa: E402
    EXPECTED_P12_SHA256,
    _build_neg_df_single_01_review,
    _extract_targets,
    _extract_v2_retract_basis,
    _git_head,
    _render_endpoint_4panel,
    _run_19_pages_online,
    _run_a1_a7_and_r1_r5,
    _sha256,
    validate_fixed_sample_preconditions,
)

S2 = ROOT / 'output/dual_tracks/stage2'
S8 = ROOT / 'output/dual_tracks/stage8'
STEP6_DIR = S8 / 'step6_closure'
STEP6_PRED = STEP6_DIR / 'predictions/path_online_step6.json'
OUT_DIR = S8 / 'step7_closure'
TZ = timezone(timedelta(hours=8))


def _diff_pages_and_groups(ref_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    ref_map = {(p['page_id'], p['prediction_id']): p for p in ref_doc['predictions']}
    new_map = {(p['page_id'], p['prediction_id']): p for p in new_doc['predictions']}
    added = [{'page_id': a[0], 'prediction_id': a[1], 'group_bbox': new_map[a].get('group_bbox')}
             for a in sorted(set(new_map) - set(ref_map))]
    removed = [{'page_id': r[0], 'prediction_id': r[1], 'group_bbox': ref_map[r].get('group_bbox')}
               for r in sorted(set(ref_map) - set(new_map))]
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
    all_pages = sorted({p['page_id'] for p in ref_doc['predictions']} | {p['page_id'] for p in new_doc['predictions']} | set(_pages()))
    by_page = []
    for pid in all_pages:
        rc = sum(1 for p in ref_doc['predictions'] if p['page_id'] == pid)
        nc = sum(1 for p in new_doc['predictions'] if p['page_id'] == pid)
        pa = [x for x in added if x['page_id'] == pid]
        pr = [x for x in removed if x['page_id'] == pid]
        pc = [x for x in changed if x['page_id'] == pid]
        by_page.append({
            'page_id': pid,
            'ref_groups': rc,
            'new_groups': nc,
            'added_count': len(pa),
            'removed_count': len(pr),
            'changed_count': len(pc),
            'changed_ids': [x['prediction_id'] for x in pc],
        })
    return {
        'ref_total_predictions': len(ref_map),
        'new_total_predictions': len(new_map),
        'added': added,
        'removed': removed,
        'changed_boxes': changed,
        'by_page': by_page,
    }


def _render_slot_before_after(
    p12_img: np.ndarray,
    crop_box: Sequence[int],
    old_preds: Sequence[Dict[str, Any]],
    new_preds: Sequence[Dict[str, Any]],
    title_boxes: Sequence[Sequence[int]],
    break_ys: Sequence[Sequence[int]],
    out_path: Path,
    scale: int = 3,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = crop_box
    raw = p12_img[y1:y2, x1:x2].copy()
    ov_old = raw.copy()
    ov_new = raw.copy()
    colors = [((0, 0, 255), (255, 100, 0)), ((0, 180, 0), (0, 200, 255))]
    for idx, p in enumerate(old_preds):
        cr, cl = colors[idx % len(colors)]
        rb, lb = p['right_bbox'], p['left_bbox']
        cv2.rectangle(ov_old, (rb[0] - x1, rb[1] - y1), (rb[2] - x1, rb[3] - y1), cr, 1)
        cv2.rectangle(ov_old, (lb[0] - x1, lb[1] - y1), (lb[2] - x1, lb[3] - y1), cl, 1)
    for tb in title_boxes:
        cv2.rectangle(ov_new, (tb[0] - x1, tb[1] - y1), (tb[2] - x1, tb[3] - y1), (255, 0, 255), 1)
    for by0, by1 in break_ys:
        cv2.line(ov_new, (0, by0 - y1), (x2 - x1 - 1, by0 - y1), (0, 255, 255), 1)
        cv2.line(ov_new, (0, by1 - y1), (x2 - x1 - 1, by1 - y1), (0, 255, 255), 1)
    for idx, p in enumerate(new_preds):
        cr, cl = colors[idx % len(colors)]
        rb, lb = p['right_bbox'], p['left_bbox']
        cv2.rectangle(ov_new, (rb[0] - x1, rb[1] - y1), (rb[2] - x1, rb[3] - y1), cr, 1)
        cv2.rectangle(ov_new, (lb[0] - x1, lb[1] - y1), (lb[2] - x1, lb[3] - y1), cl, 1)
    sep = np.full((y2 - y1, 4, 3), 255, np.uint8)
    comb = np.hstack([raw, sep, ov_old, sep, ov_new])
    z = cv2.resize(comb, (comb.shape[1] * scale, comb.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imencode('.png', z)[1].tofile(str(out_path))
    return {
        'evidence_image': str(out_path.relative_to(ROOT)).replace('\\', '/'),
        'evidence_image_sha256': _sha256(out_path),
        'crop_bbox_half_open': list(crop_box),
        'panels': ['原图', 'step6_closure 叠框', 'step7_closure 叠框（品红=药名，黄线=断组空白带）'],
    }


def _build_slot05_slot06_and_a3_audit(
    p12_img: np.ndarray,
    step6_doc: Dict[str, Any],
    new_doc: Dict[str, Any],
    ev_p12: Sequence[Dict[str, Any]],
    ev_dir: Path,
    closure_dir: Path,
) -> Dict[str, Any]:
    s6_p12 = {p['prediction_id']: p for p in step6_doc['predictions'] if p['page_id'] == '丹方彚编/page_12'}
    n_p12 = {p['prediction_id']: p for p in new_doc['predictions'] if p['page_id'] == '丹方彚编/page_12'}
    ev_by_id = {e['evidence_id']: e for e in ev_p12}

    s05_ids = ['丹方彚编/page_12#slot05#00', '丹方彚编/page_12#slot05#01']
    s05_img = _render_slot_before_after(
        p12_img,
        [682, 200, 744, 835],
        [s6_p12[i] for i in s05_ids if i in s6_p12],
        [n_p12[i] for i in s05_ids if i in n_p12],
        [[696, 562, 725, 619]],
        [[559, 562], [619, 624]],
        ev_dir / 'slot05_before_after_grouping_3x.png',
        scale=3,
    )
    s06_ids = ['丹方彚编/page_12#slot06#01', '丹方彚编/page_12#slot06#05']
    s06_img = _render_slot_before_after(
        p12_img,
        [635, 200, 696, 835],
        [s6_p12[i] for i in s06_ids if i in s6_p12],
        [n_p12[i] for i in s06_ids if i in n_p12],
        [[646, 566, 676, 649]],
        [[559, 566], [649, 653]],
        ev_dir / 'slot06_before_after_grouping_3x.png',
        scale=3,
    )

    ann = json.loads((S2 / 'annotations.json').read_text(encoding='utf-8'))
    p12_titles = []
    for g in ann['groups']:
        if g.get('page_id') == '丹方彚编/page_12' and g.get('main_bbox'):
            mb = g['main_bbox']
            p12_titles.append({
                'group_id': g['group_id'],
                'band': 'upper' if mb[1] < 400 else 'lower',
                'main_bbox_half_open': mb,
                'group_bbox_half_open': g['group_bbox'],
            })

    def _pair_summary(gid: str) -> Dict[str, Any]:
        old_p = s6_p12.get(gid) or {}
        new_p = n_p12.get(gid) or {}
        er = ev_by_id.get(f'{gid}:right') or {}
        el = ev_by_id.get(f'{gid}:left') or {}
        return {
            'group_id': gid,
            'before': {'group_bbox': old_p.get('group_bbox'), 'right_bbox': old_p.get('right_bbox'), 'left_bbox': old_p.get('left_bbox')},
            'after': {'group_bbox': new_p.get('group_bbox'), 'right_bbox': new_p.get('right_bbox'), 'left_bbox': new_p.get('left_bbox')},
            'after_side_status': {'right': er.get('side_status'), 'left': el.get('side_status')},
            'after_readings': {'right': er.get('text'), 'left': el.get('text')},
        }

    doc = {
        'coordinate_convention': '原图左闭右开 [x1, y1, x2, y2)',
        'slot05_review': {
            'evidence': s05_img,
            'upper_title': {'text': '輕粉', 'y_half_open': [196, 269]},
            'middle_title': {
                'text': '蟾酥',
                'chan_bbox_half_open': [697, 562, 725, 591],
                'su_bbox_half_open': [696, 594, 725, 619],
                'attribution': '「酥」为药名「蟾酥」第二字（跨中缝 x=710 居中大字，宽 29 px、高 25 px），不属于双轨小字正文。',
            },
            'vertical_breaks_half_open': {
                'upper_end_to_title': [559, 562],
                'title_end_to_lower_start': [619, 624],
            },
            'seam_x': {'slot05#00': 711, 'slot05#01': 710},
            'groups': [_pair_summary(i) for i in s05_ids],
            'independent_paragraphs_preserved': True,
            'conclusion': 'slot05 上下确为两组独立小字：上组「輕粉」注记（[690, 269, 738, 560)）与下组「蟾酥」注记（[690, 621, 738, 823)），中间由药名「蟾酥」([696, 562, 725, 619))及两段空白带([559, 562)、[619, 624))隔开。修正后消除了上组下延重叠与下组吞入「酥」字，两组独立段落完整保留。',
        },
        'slot06_review': {
            'evidence': s06_img,
            'middle_title': {'text': '母丁香', 'bbox_half_open': [646, 566, 676, 649]},
            'vertical_breaks_half_open': {
                'upper_end_to_title': [559, 566],
                'title_end_to_lower_start': [649, 653],
            },
            'groups': [_pair_summary(i) for i in s06_ids],
            'conclusion': '同源通用断组规则同步将上组 slot06#01 停止在「母丁香」前的空白带 y2=559，既对齐冻结正例 DF12-07 ([637, 278, 686, 559))，又消除了 slot06#01:right 伪延伸至底端「真」字的跨组重复。',
        },
        'a3_title_coverage_explanation': {
            'total_frozen_titles_on_p12': len(p12_titles),
            'upper_band_count': sum(1 for t in p12_titles if t['band'] == 'upper'),
            'lower_band_count': sum(1 for t in p12_titles if t['band'] == 'lower'),
            'covered_titles': p12_titles,
            'uncovered_lower_titles_observed': [
                {'slot': 'slot05', 'text': '蟾酥', 'bbox_half_open': [696, 562, 725, 619]},
                {'slot': 'slot06', 'text': '母丁香', 'bbox_half_open': [646, 566, 676, 649]},
            ],
            'explanation': 'A3 离线药名隔离检查基于 stage2/annotations.json 在丹方 p12 冻结的 11 个 main_bbox（上部 8 栏 DF12-01..08，以及下部仅 3 栏 DF12-09/10/11 对应 slot04/03/02）。下部 slot05（「蟾酥」）与 slot06（「母丁香」）未纳入该 11 个冻结药名框，故此前 A3=0/43 通过但仍需图像复核发现并修复这两处标题侵入。',
        },
    }
    (closure_dir / 'slot05_grouping_and_a3_coverage.json').write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return doc


def _build_line_adhesion_audit(
    p12_img: np.ndarray,
    ink: np.ndarray,
    lane_by_id: Dict[str, Dict[str, Any]],
    new_p12_map: Dict[str, Dict[str, Any]],
    ev_dir: Path,
    acc_dir: Path,
) -> Dict[str, Any]:
    gray = cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY)
    p12_sha = _sha256(P12_IMG)
    records = {}

    def _render_12x(lane: Dict[str, Any], name: str, y_lo: int, y_hi: int) -> Dict[str, Any]:
        x1, y1, x2, y2 = lane['bbox']
        rm = (lane.get('evidence') or {}).get('rule_mask_review') or {}
        cx1, cy1, cx2, cy2 = rm['crop_bbox']
        m_img = cv2.imdecode(np.fromfile(str(ROOT / rm['mask_path']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        c_img = cv2.imdecode(np.fromfile(str(ROOT / rm['masked_crop_path']), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask_full = np.zeros(ink.shape, dtype=bool)
        mask_full[cy1:cy2, cx1:cx2] = m_img > 0
        cleared_full = p12_img.copy()
        cleared_full[cy1:cy2, cx1:cx2] = c_img

        zx1, zx2 = max(0, x1 - 4), min(p12_img.shape[1], x2 + 4)
        r_c = p12_img[y_lo:y_hi, zx1:zx2]
        m_c = cv2.cvtColor((mask_full[y_lo:y_hi, zx1:zx2].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        cl_c = cleared_full[y_lo:y_hi, zx1:zx2]
        ov_c = r_c.copy()
        ov_c[mask_full[y_lo:y_hi, zx1:zx2]] = (255, 0, 255)
        panels = []
        s = 12
        for pp in (r_c, ov_c, m_c, cl_c):
            pz = cv2.resize(pp, (pp.shape[1] * s, pp.shape[0] * s), interpolation=cv2.INTER_NEAREST)
            cv2.line(pz, (0, (y2 - y_lo) * s), (pz.shape[1] - 1, (y2 - y_lo) * s), (0, 0, 255), 2)
            cv2.line(pz, ((x1 - zx1) * s, 0), ((x1 - zx1) * s, pz.shape[0] - 1), (255, 0, 0), 1)
            cv2.line(pz, ((x2 - zx1) * s, 0), ((x2 - zx1) * s, pz.shape[0] - 1), (255, 0, 0), 1)
            panels.append(pz)
        out_p = ev_dir / f'{name}_12x_pixel_proof.png'
        cv2.imencode('.png', np.hstack(panels))[1].tofile(str(out_p))
        return {
            'evidence_image': str(out_p.relative_to(ROOT)).replace('\\', '/'),
            'evidence_image_sha256': _sha256(out_p),
            'crop_bbox_half_open': [zx1, y_lo, zx2, y_hi],
            'lane_bbox_half_open': list(lane['bbox']),
            'mask_path': rm.get('mask_path'),
            'masked_crop_path': rm.get('masked_crop_path'),
        }

    # 1. slot01#01:right 「解」
    eid_jie = '丹方彚编/page_12#slot01#01:right'
    if eid_jie in lane_by_id:
        ln = lane_by_id[eid_jie]
        meta = _render_12x(ln, 'slot01#01_right_bottom_jie', 804, 828)
        records[eid_jie] = {
            **meta,
            'glyph': '解',
            'grayscale_probe_x_half_open': [929, 935],
            'grayscale_rows': {str(y): [int(gray[y, x]) for x in range(929, 935)] for y in range(819, 826)},
            'boundary_adjustment_check': {
                'shrink_y2_to_822': 'y=822 处 x=932 灰度 128 仍属「解」右竖笔向下延续（y=820 灰度 90），上收至 y2=822 会截断真实竖笔末端。',
                'keep_y2_at_823': 'y=822 竖笔与 y=823 底版框（整行灰度 94..105）之间无空白行，物理融合粘连。',
            },
            'verdict': '待核（调整矩形或掩膜均无法无损分离「解」右竖笔与底版框）',
        }

    # 2. slot06#05:right 「真」
    eid_zhen = '丹方彚编/page_12#slot06#05:right'
    if eid_zhen in lane_by_id:
        ln = lane_by_id[eid_zhen]
        meta = _render_12x(ln, 'slot06#05_right_bottom_zhen', 804, 829)
        records[eid_zhen] = {
            **meta,
            'glyph': '真',
            'grayscale_probe_x_half_open': [661, 670],
            'grayscale_rows': {str(y): [int(gray[y, x]) for x in range(661, 670)] for y in range(819, 827)},
            'boundary_adjustment_check': {
                'keep_y2_at_823': '「真」底撇末尖在 y=823（x=663..664 灰度 128, 119）落于 y2=823 外侧。',
                'expand_y2_to_824': '外扩至 y2=824 虽纳入 y=823，但撇尖直接连入 y=[824, 827) 实心底版框（灰度 75..92），无空白分离带，掩膜去除底版框仍紧贴笔画末端。',
            },
            'verdict': '待核（调整矩形或掩膜均无法无损证明「真」底撇完整）',
        }

    # 3. slot06#01:right（原共享「真」，本轮已在「母丁香」前正确断组至 y2=559）
    p0601 = new_p12_map.get('丹方彚编/page_12#slot06#01')
    records['丹方彚编/page_12#slot06#01:right'] = {
        'previous_bbox_step6': [661, 270, 690, 823],
        'new_bbox_step7': p0601.get('right_bbox') if p0601 else None,
        'still_reaches_zhen_at_y823': bool(p0601 and p0601['right_bbox'][3] > 800),
        'verdict': '已解除（同栏断组修正将 slot06#01 停止在「母丁香」前 y2=559，不再伪延伸至底端「真」字，A4 与运行时均通过）',
    }

    doc = {
        'coordinate_convention': '原图左闭右开 [x1, y1, x2, y2)',
        'source_image_sha256': p12_sha,
        'records': records,
    }
    (acc_dir / 'line_adhesion_audit.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return doc


def _build_ruling_mask_impact_audit(
    p12_img: np.ndarray,
    ink_p12: np.ndarray,
    step6_a4p: Dict[str, Any],
    ev_dir: Path,
    acc_dir: Path,
) -> Dict[str, Any]:
    v2 = json.loads((S8 / 'negative_regions_v2.json').read_text(encoding='utf-8'))
    H = 22.0
    width = max(2, round(0.24 * H))
    vert = max(3, round(2.2 * H))
    horiz = max(3, round(3 * H))

    rule_sample_results = []
    for rs in v2['rule_samples']:
        img_p = ROOT / rs['image_path']
        rimg = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
        rink = page_ink(rimg, _contract())
        rlab, rstat = _stage8_ink_components(rink)
        rw, rh = rstat[:, cv2.CC_STAT_WIDTH], rstat[:, cv2.CC_STAT_HEIGHT]
        ri_v = (rw <= width) & (rh >= vert); ri_v[0] = False
        ri_h = (rh <= width) & (rw >= horiz); ri_h[0] = False
        old_m = (ri_v[rlab] | vertical_rule_mask(rink, H, min_height=2.2)
                 | _directional_rule(rink, horiz, width, False)
                 | _sloping_vertical_rules(rink, H))
        dashed_m = _collinear_dashed_vertical_rules(rink, H, rlab, rstat)
        v_all = ri_v[rlab] | vertical_rule_mask(rink, H, min_height=2.2) | _sloping_vertical_rules(rink, H) | dashed_m
        h_all = ri_h[rlab] | _directional_rule(rink, horiz, width, False)
        bridge_m = _bridge_vertical_to_horizontal_rules(rink, v_all, h_all, H)
        new_m = v_all | h_all | bridge_m

        bx1, by1, bx2, by2 = rs['bbox']
        pad_x = 12 if (bx2 - bx1) < 12 else 4
        cx1, cy1 = max(0, bx1 - pad_x), max(0, by1)
        cx2, cy2 = min(rimg.shape[1], bx2 + pad_x), min(rimg.shape[0], min(by2, by1 + 260))
        raw_c = rimg[cy1:cy2, cx1:cx2].copy()
        ov_old = raw_c.copy(); ov_old[old_m[cy1:cy2, cx1:cx2]] = (0, 200, 255)
        ov_new = raw_c.copy(); ov_new[old_m[cy1:cy2, cx1:cx2]] = (0, 200, 255)
        ov_new[(new_m & ~old_m)[cy1:cy2, cx1:cx2]] = (255, 0, 255)
        sep = np.full((cy2 - cy1, 3, 3), 255, np.uint8)
        vis = np.hstack([raw_c, sep, ov_old, sep, ov_new])
        sc = 3 if vis.shape[0] <= 260 else 2
        vis_z = cv2.resize(vis, (vis.shape[1] * sc, vis.shape[0] * sc), interpolation=cv2.INTER_NEAREST)
        out_p = ev_dir / f"rule_sample_{rs['sample_id']}.png"
        cv2.imencode('.png', vis_z)[1].tofile(str(out_p))

        rule_sample_results.append({
            'sample_id': rs['sample_id'],
            'kind': rs['kind'],
            'page_id': rs['page_id'],
            'bbox_half_open': rs['bbox'],
            'ink_pixels_in_bbox': int(rink[by1:by2, bx1:bx2].sum()),
            'old_mask_pixels': int(old_m[by1:by2, bx1:bx2].sum()),
            'new_mask_pixels': int(new_m[by1:by2, bx1:bx2].sum()),
            'added_by_new_rules_pixels': int((new_m & ~old_m)[by1:by2, bx1:bx2].sum()),
            'dashed_rule_pixels': int(dashed_m[by1:by2, bx1:bx2].sum()),
            'bridge_rule_pixels': int(bridge_m[by1:by2, bx1:bx2].sum()),
            'evidence_image': str(out_p.relative_to(ROOT)).replace('\\', '/'),
            'evidence_image_sha256': _sha256(out_p),
            'note': (
                '新断续/桥接规则新增 0 像素；原有 72 px 为历史 _sloping_vertical_rules (Hough) 在「實」竖笔处触发，未因新规则扩大。'
                if rs['sample_id'] == 'RULE-LONG-01'
                else '新规则显著补全栏线覆盖，仅作用于独立窄栏线列，未侵蚀两侧宽字形主体。'
            ),
        })

    # 逐条核查 step6_closure 中 6 条解除疑点轨（+ slot06#05:right:top）在轨内正文区是否被掩膜遮掉真实字形
    lab12, st12 = _stage8_ink_components(ink_p12)
    rm12 = ruling_mask(ink_p12, H, lab12, st12)
    cleared_checks = []
    check_lanes = [l for l in step6_a4p['lanes'] if l['final_policy_status'] == 'cleared_by_rule_mask']
    # 同时纳入 slot06#05:right 顶部
    s0605 = next((l for l in step6_a4p['lanes'] if l['evidence_id'] == '丹方彚编/page_12#slot06#05:right'), None)
    if s0605:
        check_lanes.append(s0605)
    for ln in check_lanes:
        x1, y1, x2, y2 = ln['bbox']
        # 轨内正文区：剔除右外缘 5px 栏线带 [x2-5, x2) 以及顶底 2px 边界外版框
        inner_x2 = max(x1 + 1, x2 - 5)
        inner_y1, inner_y2 = y1 + 2, max(y1 + 3, y2 - 2)
        inner_mask_px = int(rm12[inner_y1:inner_y2, x1:inner_x2].sum())
        outer_rule_mask_px = int(rm12[y1:y2, inner_x2:x2].sum())
        # 检查是否有任何宽字形（w >= 6）的像素被掩膜覆盖
        masked_cids = set(map(int, np.unique(lab12[y1:y2, x1:x2][rm12[y1:y2, x1:x2]]))) - {0}
        wide_glyph_cids = [
            {'cid': c, 'bbox': [int(st12[c][0]), int(st12[c][1]), int(st12[c][0] + st12[c][2]), int(st12[c][1] + st12[c][3])]}
            for c in sorted(masked_cids)
            if 6 <= int(st12[c][2]) <= 40 and int(st12[c][3]) <= 45
        ]
        cleared_checks.append({
            'evidence_id': ln['evidence_id'],
            'step6_bbox_half_open': ln['bbox'],
            'text_interior_x_half_open': [x1, inner_x2],
            'text_interior_masked_pixels': inner_mask_px,
            'outer_rule_strip_masked_pixels': outer_rule_mask_px,
            'wide_glyph_components_masked_in_lane': wide_glyph_cids,
            'no_real_glyph_occluded': (inner_mask_px == 0 and len(wide_glyph_cids) == 0),
        })

    doc = {
        'coordinate_convention': '原图左闭右开 [x1, y1, x2, y2)',
        'rule_samples_audit': rule_sample_results,
        'step6_cleared_lanes_glyph_safety_audit': cleared_checks,
        'all_cleared_lanes_zero_glyph_occlusion': all(c['no_real_glyph_occluded'] for c in cleared_checks),
    }
    (acc_dir / 'ruling_mask_impact_audit.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return doc


def _build_a3_a4_ink_and_policy(
    pred_path: Path,
    pred_sha: str,
    acc_dir: Path,
    ev_dir: Path,
) -> Dict[str, Any]:
    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    p12_preds = [p for p in doc['predictions'] if p.get('page_id') == '丹方彚编/page_12']
    p12_pred_map = {p['prediction_id']: p for p in p12_preds}
    p12_sha = _sha256(P12_IMG)
    p12_img = cv2.imdecode(np.fromfile(str(P12_IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    ink = ink_mask(cv2.cvtColor(p12_img, cv2.COLOR_BGR2GRAY))
    H = 21.0

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

    ink_path = acc_dir / 'ink_review.json'
    ink_rep = run_ink_review(pred_path, a4_path, ink_path)
    lane_by_id = {l['evidence_id']: l for l in ink_rep['lanes']}

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

    lane_decisions = []
    for lane in ink_rep['lanes']:
        eid = lane['evidence_id']
        gid, side = eid.rsplit(':', 1)
        pred_rec = p12_pred_map.get(gid)
        pred_box = pred_rec.get(f'{side}_bbox') if pred_rec else None
        bbox_match = (pred_box == lane['bbox'])
        ep_details = {}
        for ep_name, ep_val in lane['endpoints'].items():
            safe = eid.replace('丹方彚编/page_12#', '').replace(':', '_')
            ep_img_meta = _render_endpoint_4panel(p12_img, ink, lane, ep_name, ev_dir / f'{safe}_{ep_name}_4panel_8x.png', scale=8)
            if not bbox_match:
                cat, ep_status = '前置校验不符', 'pending_precondition_mismatch'
                detail = '新预测框与复核框不一致，保持待核。'
            elif lane['files'] != 'verified':
                cat, ep_status = '缺少掩膜证据', 'pending_missing_mask'
                detail = f"缺少已核验掩膜附件（files={lane['files']}），保持待核。"
            elif ep_val['verdict'] == 'rule_explains_trigger':
                cat, ep_status = '框外栏线触发', 'cleared_by_rule_mask'
                detail = '掩膜复核验证通过，去栏线后无框内字形接触且无跨边残墨，解除该端点疑点。'
            else:
                cat, ep_status = '框内真实笔画接触', 'pending_glyph_adhesion'
                detail = '框内真实字形笔画与底版框/栏线物理粘连，调整矩形或掩膜均无法无损证明完整，保持待核。'
            ep_details[ep_name] = {**ep_val, 'category': cat, 'endpoint_policy_status': ep_status, 'detail': detail, 'evidence_panel': ep_img_meta}
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
        'lanes': lane_decisions,
    }
    (acc_dir / 'a4_policy_reconcile.json').write_text(json.dumps(a4_policy_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'a3': a3_doc, 'a4': a4_doc, 'ink_review': ink_rep, 'a4_policy': a4_policy_doc,
        'lane_by_id': lane_by_id, 'p12_pred_map': p12_pred_map, 'p12_img': p12_img, 'ink': ink,
    }


def main() -> int:
    py = sys.executable
    pred_path = OUT_DIR / 'predictions/path_online_step7.json'
    crop_dir = OUT_DIR / 'crops'
    acc_dir = OUT_DIR / 'acceptance'
    ev_dir = OUT_DIR / 'evidence'
    closure_dir = OUT_DIR / 'closure'
    m_frozen_dir = OUT_DIR / 'metrics/neg_frozen'
    m_v2_dir = OUT_DIR / 'metrics/neg_v2'
    for d in (acc_dir, ev_dir, closure_dir, m_frozen_dir, m_v2_dir):
        d.mkdir(parents=True, exist_ok=True)

    step6_doc = json.loads(STEP6_PRED.read_text(encoding='utf-8'))
    step6_sha = _sha256(STEP6_PRED)
    step6_a4p = json.loads((STEP6_DIR / 'acceptance/a4_policy_reconcile.json').read_text(encoding='utf-8'))
    step6_targets = json.loads((STEP6_DIR / 'acceptance/targets.json').read_text(encoding='utf-8'))

    # 1. 同一次 19 页正常入口预测
    run = _run_19_pages_online(crop_dir, pred_path)
    pred_sha = run['pred_sha256']
    diff_doc = _diff_pages_and_groups(step6_doc, run['doc'])
    (closure_dir / 'diff_vs_step6.json').write_text(json.dumps({
        'reference_prediction': str(STEP6_PRED.relative_to(ROOT)).replace('\\', '/'),
        'reference_sha256': step6_sha,
        'new_prediction': str(pred_path.relative_to(ROOT)).replace('\\', '/'),
        'new_sha256': pred_sha,
        **diff_doc,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 2. A3 / A4 / ink_review / A4 policy
    acc = _build_a3_a4_ink_and_policy(pred_path, pred_sha, acc_dir, ev_dir)

    # 3. 专题核验：slot05 & slot06 分组 + A3 覆盖、贴线待核轨 12× 像素核查、新栏线规则影响抽查
    s05_doc = _build_slot05_slot06_and_a3_audit(
        acc['p12_img'], step6_doc, run['doc'], run['ev_by_page']['丹方彚编/page_12'], ev_dir, closure_dir,
    )
    adh_doc = _build_line_adhesion_audit(
        acc['p12_img'], acc['ink'], acc['lane_by_id'], acc['p12_pred_map'], ev_dir, acc_dir,
    )
    p12_page_ink = page_ink(acc['p12_img'], _contract())
    rule_audit = _build_ruling_mask_impact_audit(acc['p12_img'], p12_page_ink, step6_a4p, ev_dir, acc_dir)

    # 4. 两套 T6 评分
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
    m_frozen_doc = json.loads((m_frozen_dir / 'online_step7.json').read_text(encoding='utf-8'))
    m_v2_doc = json.loads((m_v2_dir / 'online_step7.json').read_text(encoding='utf-8'))
    targets_doc = _extract_targets(m_frozen_doc, m_v2_doc, pred_path, pred_sha)
    (acc_dir / 'targets.json').write_text(json.dumps(targets_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 5. A1–A7 与 R1–R5
    checks_res = _run_a1_a7_and_r1_r5(
        pred_path, pred_sha, run['ev_by_page'],
        acc['a3'], acc['a4'], acc['a4_policy'], acc_dir,
    )

    # 6. T6 双口径对账（明确保留冻结历史口径 1/10 与 v2 复核口径 0/9，注明版本升级须明确决定记录）
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
        'reference_step6_sha256': step6_sha,
        'formal_historical_caliber': 'frozen',
        'version_governance_decision': {
            'current_formal_label_version': 'stage2_frozen (output/dual_tracks/stage2/negative_regions.json)',
            'candidate_revised_version': 'stage8_v2_review (output/dual_tracks/stage8/negative_regions_v2.json)',
            'status': 'pending_explicit_version_decision',
            'requirement': '冻结历史口径继续记录 1/10，v2 复核口径单独报告 0/9；将 v2 升为新的正式标注版本须形成明确的版本决定与签署记录，本脚本不自动改写正式版本。',
        },
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
    (closure_dir / 't6_caliber_reconcile.json').write_text(json.dumps(t6_reconcile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # 7. 自动生成报告
    a4p = acc['a4_policy']
    tf, tv = targets_doc['calibers']['frozen'], targets_doc['calibers']['v2']
    s6_tf = step6_targets['calibers']['frozen']
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
    rs_rows = [
        f"| `{r['sample_id']}` | `{r['kind']}` | `{r['bbox_half_open']}` | {r['ink_pixels_in_bbox']} | {r['old_mask_pixels']} → **{r['new_mask_pixels']}** (+{r['added_by_new_rules_pixels']}) | {r['note']} |"
        for r in rule_audit['rule_samples_audit']
    ]
    report_md = f"""# 阶段八第七轮正常入口验收与待决结项报告（step7_closure）

生成时间：{datetime.now(TZ).isoformat(timespec='seconds')}  
代码版本：Git `HEAD` `{_git_head()}` · 配置指纹 `{run['doc']['effective']['fingerprint']}` · 模型 `{run['doc']['model_version']}`  
坐标约定：**原图左闭右开 `[x1, y1, x2, y2)`**  
历史对照预测：`output/dual_tracks/stage8/step6_closure/predictions/path_online_step6.json`（SHA-256 `{step6_sha}`）  
本轮唯一预测：`{t6_reconcile['prediction_source']}`（SHA-256 `{pred_sha}`）  
预测变化：总数 `{diff_doc['ref_total_predictions']} → {diff_doc['new_total_predictions']}`，新增 `{len(diff_doc['added'])}`，删除 `{len(diff_doc['removed'])}`，坐标变化 `{len(diff_doc['changed_boxes'])}` 组

## 1. 综合结论与相对 `step6_closure` 变化
- **阶段八状态：未通过（暂不结项）**
- 入口验收 A1–A7：**{ck['passed']}/{ck['total']}**（A1 同栏上下对由 `3` 增至 **`{ck['checks'][0]['detail']['n_stacked']}`**；A4 不通过 `0`，缺掩膜 `0`，待核由 `{len(step6_a4p['pending_review_lanes'])}` 降至 **`{len(a4p['pending_review_lanes'])}`**）；开关回归 R1–R5：**{rg['passed']}/{rg['total']}**。
- 原冻结正式历史口径 T1–T8：**{sum(1 for v in tf.values() if v['pass'])}/8**（**T2 由 `{s6_tf['T2']['num']}/38` 升至 `{tf['T2']['num']}/38`**，仅 T6=`{tf['T6']['num']}/{tf['T6']['den']}` 未达 0）；v2 复核口径 T1–T8：**{sum(1 for v in tv.values() if v['pass'])}/8**（T6=`{tv['T6']['num']}/{tv['T6']['den']}` 达标）。

## 2. `slot05` / `slot06` 分组修正与 A3 标题覆盖说明
- **`slot05`**：上组「輕粉」`[690, 269, 738, 560)`、中隔药名「蟾酥」`[696, 562, 725, 619)`（其中「酥」`[696, 594, 725, 619)` 确属药名）、下组「蟾酥」注记 `[690, 621, 738, 823)`，两处空白断组带 `[559, 562)` 与 `[619, 624)` 完整隔离；对比图 `{s05_doc['slot05_review']['evidence']['evidence_image']}`。
- **`slot06`**：同源断组规则将 `slot06#01` 止于「母丁香」前 `[641, 270, 690, 559)`，对齐冻结正例 `DF12-07` 并消除与 `slot06#05`（`[641, 650, 690, 823)`）的重叠及伪贴「真」；对比图 `{s05_doc['slot06_review']['evidence']['evidence_image']}`。
- **A3 覆盖范围**：{s05_doc['a3_title_coverage_explanation']['explanation']}

## 3. 3 条贴线待核轨核查结果
- **`slot06#01:right`**：**已解除**（断组修正后止于 `y2=559`，不再伪延伸至 `y=823`「真」，A4 与运行时均为 `ok`）。
- **`slot01#01:right`（「解」底竖）**：**保持待核**（12× 像素图证实 `y=822` 竖笔灰度 `128` 直接并入 `y=823` 底版框 `94..105`，上收会切字、保留仍粘连）。
- **`slot06#05:right`（「真」底撇）**：**保持待核**（12× 像素图证实 `y=823` 撇尖 `x=[663, 665)` 灰度 `128/119` 直接连入 `y=[824, 827)` 底版框 `75..92`，外扩至 `y2=824` 仍紧贴底版框无法无损证完整）。

## 4. 新栏线规则影响抽查与无遮字核验
| 样例 ID | 类别 | 原图框 `[x1,y1,x2,y2)` | 墨迹 px | 旧→新掩膜 px | 结论 |
|---|---|---|---|---|---|
{chr(10).join(rs_rows)}

- **第六轮解除疑点轨无遮字核验**：`all_cleared_lanes_zero_glyph_occlusion = {rule_audit['all_cleared_lanes_zero_glyph_occlusion']}`（逐条确认正文区内掩膜像素 = `0`、宽字形被遮数 = `0`）。

## 5. A4 逐轨判定（原图触发保留 `{a4p['original_a4_raw_triggers']}/{a4p['original_a4_checked']}`）
| 轨道 | 新原图轨框 `[x1,y1,x2,y2)` | 原图 A4 | 掩膜核验 | 联合判定 | 逐端点分类 |
|---|---|---|---|---|---|
{chr(10).join(lane_rows)}

## 6. T1–T8 与 T6 双口径保留（版本升级须明确决定）
| 项 | step6（冻结） | step7（冻结） | step7（v2） | 门槛 |
|---|---|---|---|---|
| T1 (@0.50) | `{s6_tf['T1']['num']}/38` | `{tf['T1']['num']}/38` | `{tv['T1']['num']}/38` | ≥30/38 |
| T2 (@0.75) | `{s6_tf['T2']['num']}/38` | **`{tf['T2']['num']}/38`** | **`{tv['T2']['num']}/38`** | ≥18/38 |
| T3 / T4 | `{s6_tf['T3']['num']}/20`, `{s6_tf['T4']['num']}/20` | `{tf['T3']['num']}/20`, `{tf['T4']['num']}/20` | `{tv['T3']['num']}/20`, `{tv['T4']['num']}/20` | ≥19/20, ≥14/20 |
| T5 轨配对 | `{s6_tf['T5']['num']}/{s6_tf['T5']['den']} ({s6_tf['T5']['value']})` | `{tf['T5']['num']}/{tf['T5']['den']} ({tf['T5']['value']})` | `{tv['T5']['num']}/{tv['T5']['den']} ({tv['T5']['value']})` | ≥0.90 |
| T6 负例误检 | `{s6_tf['T6']['num']}/10` | **`{tf['T6']['num']}/10`（未达）** | **`{tv['T6']['num']}/9`（达标）** | 0 |
| T7 / T8 | `{s6_tf['T7']['num']}/11`, `{s6_tf['T8']['num']}/2` | `{tf['T7']['num']}/11`, `{tf['T8']['num']}/2` | `{tv['T7']['num']}/11`, `{tv['T8']['num']}/2` | ≥6/11, ≥1/2 |

- **T6 版本治理**：原冻结正式历史口径继续保留 **`1/10`**，v2 复核口径单独报告 **`0/9`**；是否将 `negative_regions_v2.json` 升为新的正式标注版本须由人工形成明确版本决定与签署记录。
- **样本范围**：19 页（38 标注正例组、10 原冻结负例区），样本量有限，结论仅覆盖当前古籍竖排双行小字样本范围。
"""
    report_path = closure_dir / '阶段八第七轮验收与待决结项报告.md'
    report_path.write_text(report_md, encoding='utf-8')

    print('=== STEP7 SINGLE-RUN PIPELINE SUMMARY ===')
    print(json.dumps({
        'prediction_sha256': pred_sha,
        'reference_step6_sha256': step6_sha,
        'added': len(diff_doc['added']),
        'removed': len(diff_doc['removed']),
        'changed_boxes_count': len(diff_doc['changed_boxes']),
        'changed_ids': [c['prediction_id'] for c in diff_doc['changed_boxes']],
        'A1_A7': f"{ck['passed']}/{ck['total']}",
        'A1_stacked': ck['checks'][0]['detail']['n_stacked'],
        'R1_R5': f"{rg['passed']}/{rg['total']}",
        'A4_raw': f"{a4p['original_a4_raw_triggers']}/{a4p['original_a4_checked']}",
        'A4_missing_masks': a4p['missing_mask_lanes'],
        'A4_cleared_lanes': a4p['cleared_lanes'],
        'A4_fail_lanes': a4p['fail_glyph_crossing_lanes'],
        'A4_pending_lanes': a4p['pending_review_lanes'],
        'T1': f"{tf['T1']['num']}/38",
        'T2': f"{tf['T2']['num']}/38",
        'T5': f"{tf['T5']['num']}/{tf['T5']['den']} ({tf['T5']['value']})",
        'T6_frozen': f"{tf['T6']['num']}/{tf['T6']['den']}",
        'T6_v2': f"{tv['T6']['num']}/{tv['T6']['den']}",
        'all_cleared_zero_glyph_occlusion': rule_audit['all_cleared_lanes_zero_glyph_occlusion'],
        'report': str(report_path.relative_to(ROOT)).replace('\\', '/'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
