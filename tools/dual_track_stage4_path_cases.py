# -*- coding: utf-8 -*-
"""第四阶段反例与路径判定实测（对应验收缺口 1、2、3 的指定样例）。

产出 `path_decisions.json`，每条都写：实测值 / 采取的路径 / 理由。

包含：
  A 列框校验三例  主体重叠超限、框内独立栏线、缺宽度参考的 230px 宽框
  B 半页隔离反例  只在右半页画栏线，确认左半页不复用右侧线组
  C 不合格间隔    一个过大间隔 + 一个尾段，其余范围继续成组
  D 掩膜编码      同一前景 0/1 与 0/255 得到同一 H/G/pitch；单点斜线被拒
  E p40 前后对照  v1 的对侧错框 vs v2 的 0 个跨界框
  F p12 前后对照  446px 仍被拒 + 11 组覆盖 + 尺度来源
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2                                        # noqa: E402
import numpy as np                                # noqa: E402

from dual_track_search import (DEFAULT_CONFIG, build_rule_mask, estimate_scale,  # noqa: E402
                               glyph_bands, group_rule_runs, line_eligibility,
                               mask_digest, page_mask_record, plan_page,
                               validate_columns, verify_record)
from dual_track_search.rules import (classify_rule_touching,  # noqa: E402
                                     detect_rule_candidates, merge_broken_rules)

S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
HIST = S4 / 'history'
CFG = DEFAULT_CONFIG


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


# --------------------------------------------------------------------- A 列框校验

def column_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    common = dict(book_stats={'cols': 310, 'median_width': 43.0}, global_stats=None,
                  cfg=CFG['columns'])
    # A1 主体重叠
    cols = [{'bbox': [30, 10, 70, 390]}, {'bbox': [45, 10, 85, 390]}]
    r = validate_columns(cols, (0, 1024), (1024, 600), rule_lines=[], **common)
    vs = [v.to_dict() for v in r['verdicts']]
    inter = min(70, 85) - max(30, 45)
    out.append({
        'case': 'A1_body_overlap',
        'input': {'columns': cols, 'half_range': [0, 1024], 'image_size': [1024, 600],
                  'reference_px': 43.0},
        'measured': {'overlap_px': inter, 'narrower_width_px': 40,
                     'overlap_ratio': round(inter / 40, 4),
                     'limit': CFG['columns']['max_overlap_ratio'],
                     'decisions': {v['col_index']: v['decision'] for v in vs}},
        'path': [v['decision'] for v in vs],
        'reason': vs[0]['reasons'][-1] if vs and vs[0]['reasons'] else '',
        'verdicts': vs,
    })
    # A2 框内独立栏线
    cols = [{'bbox': [30, 10, 70, 390]}]
    rules = [{'center': 50.0, 'kind': 'independent', 'confidence': 'high', 'coverage': 0.92}]
    r = validate_columns(cols, (0, 1024), (1024, 600), rule_lines=rules, **common)
    v = r['verdicts'][0]
    out.append({
        'case': 'A2_rule_inside_box',
        'input': {'columns': cols, 'rule_lines': rules, 'half_range': [0, 1024],
                  'reference_px': 43.0},
        'measured': {'rules_inside': v.measurements.get('rules_inside'),
                     'split_parts_px': v.measurements.get('split_parts_px'),
                     'split_min_width_px': r['reference']['split_min_width_px']},
        'path': v.decision,
        'reason': v.reasons[-1] if v.reasons else '',
        'verdicts': [v.to_dict()],
    })
    # A3 缺参考 + 230px 宽框
    cols = [{'bbox': [300, 10, 530, 390]}]
    r = validate_columns(cols, (0, 1024), (1024, 600), book_stats=None, global_stats=None,
                         cfg=CFG['columns'], page_stats=None, rule_lines=[])
    v = r['verdicts'][0]
    out.append({
        'case': 'A3_no_reference_wide_box',
        'input': {'columns': cols, 'book_stats': None, 'global_stats': None,
                  'page_stats': None},
        'measured': {'width_px': v.width, 'reference_px': r['reference']['value_px'],
                     'reference_source': r['reference']['source'],
                     'page_width_ratio': v.measurements.get('page_width_ratio'),
                     'width_reference_notes': r['reference']['notes']},
        'path': v.decision,
        'reason': v.reasons[-1] if v.reasons else '',
        'verdicts': [v.to_dict()],
    })
    # A4 同页替代参考（缺同书时用同页中位宽，来源可查）
    cols = [{'bbox': [100 + k * 50, 10, 140 + k * 50, 390]} for k in range(8)]
    cols.append({'bbox': [520, 10, 750, 390]})
    r = validate_columns(cols, (0, 1024), (1024, 600), book_stats={'cols': 5, 'median_width': 43.0},
                         global_stats={'cols': 1702, 'median_width': 37.0},
                         cfg=CFG['columns'],
                         page_stats={'cols': 9, 'median_width': 40.0}, rule_lines=[])
    v = r['verdicts'][-1]
    out.append({
        'case': 'A4_page_median_substitute',
        'input': {'book_cols': 5, 'global_median_width': 37.0, 'page_median_width': 40.0,
                  'box': [520, 10, 750, 390]},
        'measured': {'reference_px': r['reference']['value_px'],
                     'reference_source': r['reference']['source'],
                     'width_ratio_to_reference': v.measurements.get('width_ratio_to_reference'),
                     'threshold_px': r['reference']['threshold_px']},
        'path': v.decision,
        'reason': v.reasons[-1] if v.reasons else '',
        'verdicts': [v.to_dict()],
    })
    return out


# --------------------------------------------------------------------- B 半页隔离

def _draw_vline(bw, x, y0, y1, thick=1, value=255):
    bw[y0:y1, x:x + thick] = value


def _draw_glyphs(bw, x0, x1, ys, h=30, value=255):
    for y in ys:
        bw[y:y + h, x0:x1] = value


def half_isolation_case() -> Dict[str, Any]:
    """只在右半页画栏线（x=600..750），左半页只有列框字块。"""
    W, H = 1024, 800
    bw = np.zeros((H, W), np.uint8)
    xs = [600, 650, 700, 750]
    for x in xs:
        _draw_vline(bw, x, 60, 740)
    for a, b in zip(xs[:-1], xs[1:]):
        _draw_glyphs(bw, a + 12, b - 12, range(90, 700, 52), h=30)
    left_cols = [[100, 60, 140, 740], [160, 60, 200, 740], [220, 60, 260, 740]]
    for c in left_cols:
        _draw_glyphs(bw, c[0] + 4, c[2] - 4, range(90, 700, 52), h=30)
    columns = [{'bbox': c} for c in left_cols] + \
              [{'bbox': [x + 2, 60, x + 46, 740]} for x in xs[:-1]]
    halves = [(520, 1024), (0, 504)]
    plan = plan_page('synthetic/half_isolation', 'synthetic', bw, (W, H), halves, columns,
                     CFG).to_dict()
    per_half = {}
    for h in plan['halves_detail']:
        regs = [r for r in plan['regions'] if r['half_id'] == h['half_id']]
        per_half[h['half_id']] = {
            'half_range': h['half_range'],
            'n_regions': len(regs),
            'by_source': {s: sum(1 for r in regs if r['column_range_source'] == s)
                          for s in ('grouped_rules', 'validated_columns', 'halfpage_fallback')},
            'grouped_rules_ok': (h['paths'].get('grouped_rules') or {}).get('ok'),
            'rule_candidate_x': [c['x0'] for c in (h.get('rule_candidates') or [])],
            'regions': [{'region_id': r['region_id'], 'bbox': r['bbox'],
                         'source': r['column_range_source']} for r in regs],
            'all_inside_half': all(r['bbox'][0] >= h['half_range'][0]
                                   and r['bbox'][2] <= h['half_range'][1] for r in regs),
        }
    left = per_half.get('left', {})
    return {
        'case': 'B_half_isolation_rules_right_only',
        'input': {'rule_xs': xs, 'left_half_columns': left_cols, 'halves': halves},
        'measured': per_half,
        'path': ('右半页 grouped_rules；左半页无栏线 → 列框/回退'),
        'pass': (left.get('by_source', {}).get('grouped_rules', 0) == 0
                 and left.get('all_inside_half', False)
                 and per_half.get('right', {}).get('grouped_rules_ok') is True),
        'reason': ('左半页没有独立栏线，裁剪后半页内候选数为 0，不可能复用右半页的线组；'
                   '所有输出框都落在各自半页边界内'),
    }


# --------------------------------------------------------------------- C 不合格间隔

def _line_at(bw, x=100, thick=3, y0=20, y1=180, touching=False):
    bw[y0:y1, x:x + thick] = 255
    if touching:
        for y in range(y0, y1, 6):
            bw[y:y + 4, x + thick:x + thick + 5] = 255
            bw[y:y + 4, x - 5:x] = 255


def broken_merge_cases() -> Dict[str, Any]:
    """组合反例：上段贴字 + 下段独立，合并后必须保留歧义；两段都独立时仍合并。"""
    H, W = 400, 300
    out: Dict[str, Any] = {'case': 'B2_broken_merge_segments'}
    cases = []
    for name, mixed in (('mixed_touching_and_independent', True),
                        ('both_independent', False)):
        bw = np.zeros((H, W), np.uint8)
        _line_at(bw, 100, 3, 20, 180, touching=mixed)
        _line_at(bw, 100, 3, 190, 380, touching=False)
        cands = detect_rule_candidates(bw, (20, 380), CFG['rules'])
        cands = classify_rule_touching(cands, bw, CFG['rules'])
        before = [{'y': [c.y0, c.y1], 'touches_text': c.touches_text,
                   'confidence': c.confidence, 'kind': c.kind} for c in cands]
        merged, log = merge_broken_rules(cands, CFG['rules'], body_h=360)
        item: Dict[str, Any] = {'name': name, 'segments_before_merge': before,
                                'merged_count': len(merged),
                                'merge_log': [{'gap_px': x['gap_px'], 'gaps': x['gaps'],
                                               'status': x['status'],
                                               'ambiguous': x['ambiguous']} for x in log]}
        if merged:
            m = merged[0]
            e = line_eligibility(m, 360, CFG['rules'])
            m.qualified = e['qualified']
            item['merged'] = {
                'bbox': [m.x0, m.y0, m.x1, m.y1],
                'status': m.status, 'ambiguous': m.ambiguous,
                'eligible_coverage': m.eligible_coverage,
                'qualified': e['qualified'], 'gap_total_px': m.gap_total_px,
                'segments': [{'y': [s['y0'], s['y1']], 'touches_text': s['touches_text'],
                              'confidence': s['confidence'], 'eligible': s['eligible']}
                             for s in m.segments],
                'excluded_from_scale': bool(e['qualified']),
            }
            rec = page_mask_record({'page_id': f'synthetic/{name}',
                                    'image_size': [W, H], 'halves': [[0, W]],
                                    'rule_lines': [dict(m.to_dict())]})
            all_m = build_rule_mask(rec['lines'], [W, H], 'all')
            ind_m = build_rule_mask(rec['lines'], [W, H], 'independent')
            item['mask'] = {
                'all_ink_px': int(all_m.sum()),
                'independent_ink_px': int(ind_m.sum()),
                'touching_segment_set_px': int(all_m[100, 100:103].sum()),
                'gap_set_px': int(all_m[185, 100:103].sum()),
                'independent_segment_set_px': int(all_m[300, 100:103].sum()),
                'independent_mask_touching_px': int(ind_m[100, 100:103].sum()),
                'independent_mask_gap_px': int(ind_m[185, 100:103].sum()),
                'independent_mask_independent_px': int(ind_m[300, 100:103].sum()),
            }
        cases.append(item)
    mixed, both = cases[0], cases[1]
    mm = mixed.get('merged') or {}
    bm = both.get('merged') or {}
    ok = (mixed['merged_count'] == 1 and both['merged_count'] == 1
          and mm.get('status') == 'mixed_ambiguous' and mm.get('ambiguous') is True
          and mm.get('qualified') is False
          and [s['y'] for s in mm.get('segments', [])] == [[20, 181], [191, 380]]
          and [s['touches_text'] for s in mm.get('segments', [])] == [True, False]
          and mixed['mask']['touching_segment_set_px'] > 0
          and mixed['mask']['gap_set_px'] == 0
          and mixed['mask']['independent_mask_touching_px'] == 0
          and mixed['mask']['independent_mask_gap_px'] == 0
          and bm.get('status') == 'confirmed' and bm.get('qualified') is True
          and both['mask']['independent_mask_touching_px'] > 0
          and both['mask']['independent_mask_gap_px'] == 0
          and both['mask']['independent_mask_independent_px'] > 0
          and both['merge_log'] and both['merge_log'][0]['gap_px'] == 10)
    out.update({'input': {'x': 100, 'segments': [[20, 180, 'touching'], [190, 380, 'independent']],
                          'body_h': 360},
                'measured': {'cases': cases},
                'path': '两段合并成一条断线；逐段证据与资格判定见 measured',
                'pass': ok,
                'reason': ('混合段合并后 status=mixed_ambiguous、qualified=false，'
                           '独立掩膜在贴字段与断口都不置位（整条按歧义不用）；'
                           '两段均独立时 status=confirmed、qualified=true，'
                           '独立掩膜覆盖两段但断口 9px 留空')})
    return out


def bad_gap_case() -> Dict[str, Any]:
    W, H = 700, 620
    bw = np.zeros((H, W), np.uint8)
    xs = [80, 130, 180, 230, 400, 450, 500, 660]   # 230→400 间隔 170（> 上限 120）；660 为尾段
    for x in xs:
        _draw_vline(bw, x, 40, 580)
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a < 120:
            _draw_glyphs(bw, a + 10, b - 10, range(70, 540, 50), h=28)
    cands = detect_rule_candidates(bw, (40, 580), CFG['rules'])
    cands = classify_rule_touching(cands, bw, CFG['rules'])
    cands, _ = merge_broken_rules(cands, CFG['rules'])
    groups, log = group_rule_runs(cands, bw, (40, 580), CFG['rules'])
    # 整页编排：确认「被拒的尾段」由回退窗口接管，而不是消失。
    # 列框按两组栏的实体栏给出，这样页面级 H 可估（否则 fallback 会因缺尺度记 uncovered）。
    cols = [{'bbox': [a, 99, b, 570]} for a, b in
            [(81, 130), (131, 180), (181, 230), (401, 450), (451, 500)]]
    plan = plan_page('synthetic/bad_gap', 'synthetic', bw, (W, H), [(0, W)], cols,
                     CFG).to_dict()
    tail = [r for r in plan['regions'] if r['bbox'][2] > 520]
    return {
        'case': 'C_bad_gap_and_tail',
        'input': {'rule_xs': xs, 'pitch_max_px': CFG['rules']['group_pitch_max_px'],
                  'group_min_count': CFG['rules']['group_min_count']},
        'measured': {'n_groups': len(groups),
                     'groups': [{'span': g.span, 'columns': g.columns,
                                 'pitch_px': round(g.pitch_px, 2),
                                 'line_count': len(g.lines)} for g in groups],
                     'rejected_runs': [x for x in log if x.get('why') not in
                                       ('not_usable', 'column_ink_low')],
                     'pipeline': {'by_source': plan['coverage']['by_source'],
                                  'regions': [{'bbox': r['bbox'],
                                               'source': r['column_range_source']}
                                              for r in plan['regions']],
                                  'tail_area_regions': [{'bbox': r['bbox'],
                                                         'source': r['column_range_source']}
                                                        for r in tail],
                                  'uncovered_px': plan['coverage']['completion']
                                  ['uncovered_px']}},
        'path': '前两段继续成组；尾段单独记 run_single_line，再由回退窗口覆盖该 x 段',
        'pass': (len(groups) >= 2
                 and any(x.get('why') == 'run_single_line' for x in log)
                 and tail and plan['coverage']['completion']['uncovered_px'] == 0),
        'reason': ('170px 的间隔超出 pitch 上限，只断开该处，前面 4 条与后面 3 条各自成组；'
                   '孤立的尾线记 run_single_line，其 x 段由 halfpage_fallback 接管，'
                   '整页未覆盖 0px'),
    }


# --------------------------------------------------------------------- D 掩膜编码

def encoding_cases() -> Dict[str, Any]:
    H, W = 400, 44
    ones = np.zeros((H, W), np.uint8)
    for k in range(9):
        y = 10 + k * 35
        ones[y:y + 30, 4:W - 4] = 1
    full = ones * 255
    box = [{'bbox': [0, 0, W, H]}]
    a = estimate_scale(ones, box, (0, H), (W, H), CFG['scale'], narrow_columns=1)
    b = estimate_scale(full, box, (0, H), (W, H), CFG['scale'], narrow_columns=1)
    bands_a = glyph_bands(ones, 0, W, 0, H, CFG['scale'])
    bands_b = glyph_bands(full, 0, W, 0, H, CFG['scale'])

    # 单点斜线：每行只有 1 个墨点，行活跃度不足 → 不是字
    diag = np.zeros((300, 40), np.uint8)
    for i in range(250):
        diag[10 + i, 5 + (i % 30)] = 1
    d_est = estimate_scale(diag, [{'bbox': [0, 0, 40, 300]}], (0, 300), (40, 300),
                           CFG['scale'], narrow_columns=1)
    d_bands = glyph_bands(diag, 0, 40, 0, 300, CFG['scale'])
    return {
        'case': 'D_mask_encoding_and_degenerate',
        'input': {'strip': {'H': 30, 'G': 5, 'n': 9, 'width': 44},
                  'diagonal': {'rows': 250, 'ink_per_row': 1}},
        'measured': {
            'unit_0_1': {'H': a.H, 'G': a.G, 'glyph_pitch': a.glyph_pitch,
                         'bands': len(bands_a), 'status': a.confidence_status,
                         'support_px': bands_a[0]['row_support_px'] if bands_a else None},
            'unit_0_255': {'H': b.H, 'G': b.G, 'glyph_pitch': b.glyph_pitch,
                           'bands': len(bands_b), 'status': b.confidence_status,
                           'support_px': bands_b[0]['row_support_px'] if bands_b else None},
            'identical': (a.to_dict()['H'] == b.to_dict()['H']
                          and a.to_dict()['G'] == b.to_dict()['G']
                          and a.to_dict()['glyph_pitch'] == b.to_dict()['glyph_pitch']
                          and len(bands_a) == len(bands_b)),
            'diagonal': {'bands': len(d_bands), 'status': d_est.confidence_status,
                         'failure_reason': d_est.failure_reason, 'H': d_est.H},
        },
        'path': '0/1 与 0/255 走同一条布尔化路径；单点斜线被行活跃度拒掉',
        'pass': (a.to_dict()['H'] == b.to_dict()['H'] == 30.0
                 and a.to_dict()['G'] == b.to_dict()['G'] == 5.0
                 and a.to_dict()['glyph_pitch'] == b.to_dict()['glyph_pitch'] == 35.0
                 and len(d_bands) == 0 and d_est.confidence_status == 'failed'),
        'reason': ('行活跃度按墨点个数（≥ max(2, 11%×列宽)）判定，与像素值域无关；'
                   '单点斜线每行仅 1 个墨点，达不到支持度，故不产生字形带，也不给 H'),
    }


# --------------------------------------------------------------------- E/F 前后对照

def _crossing_regions(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    halves = page.get('halves') or []
    idx = {'right': 0, 'left': 1}
    out = []
    for r in page.get('regions') or []:
        k = idx.get(r['half_id'], 0)
        if k >= len(halves):
            continue
        hx0, hx1 = halves[k]
        if r['bbox'][0] < hx0 or r['bbox'][2] > hx1:
            out.append({'region_id': r['region_id'], 'bbox': r['bbox'],
                        'half': [hx0, hx1],
                        'overrun_px': [max(0, hx0 - r['bbox'][0]),
                                       max(0, r['bbox'][2] - hx1)]})
    return out


def coord_offset_case() -> Dict[str, Any]:
    """非零半页偏移：右半页从 x=400 起，孤立线在原图 x=780。

    验证候选、run 拒绝记录、fallback_cause 三处指向同一条线（原图坐标）。
    """
    W, H = 1024, 400
    bw = np.zeros((H, W), np.uint8)
    for a, b in [(40, 80), (110, 150), (180, 220)]:          # 左半页给 3 栏让页面 H 可估
        for y in range(40, 340, 40):
            bw[y:y + 28, a:b] = 255
    bw[40:380, 780:782] = 255                                 # 孤立线，仅此一条
    cols = [{'bbox': [40, 40, 80, 340]}, {'bbox': [110, 40, 150, 340]},
            {'bbox': [180, 40, 220, 340]}]
    plan = plan_page('synthetic/coord_offset', 'synthetic', bw, (W, H),
                     [(400, 1024), (0, 380)], cols, CFG).to_dict()
    right = next(h for h in plan['halves_detail'] if h['half_id'] == 'right')
    lines = [l for l in plan['rule_lines'] if l['half_id'] == 'right']
    runs = right['rule_runs']
    windows = (right['paths']['halfpage_fallback'] or {}).get('windows') or []
    cand_x = [round(l['center'], 1) for l in lines]
    run_spans = [r.get('span') for r in runs if r.get('span')]
    causes = [{'bbox': w['bbox'], 'cause': w['cause'].get('cause'),
               'run_span': w['cause'].get('run_span'),
               'coord_system': w['cause'].get('coord_system')} for w in windows]
    hit = [c for c in causes if c['bbox'][0] <= 780 <= c['bbox'][2]]
    leakage = [r for r in runs if (r.get('span') or [0])[0] < 400 or
               (r.get('x_list') and min(r['x_list']) < 400)]
    ok = (cand_x == [781.0]
          and run_spans and run_spans[0] == [781.0, 781.0]
          and all(r.get('coord_system') == 'original_image' for r in runs)
          and hit and all(c['cause'] == 'rule_run_rejected' for c in hit)
          and all(c['run_span'] == [781.0, 781.0] for c in hit)
          and not leakage)
    return {
        'case': 'B3_coord_offset',
        'input': {'halves': [[400, 1024], [0, 380]], 'isolated_line_x': 780,
                  'expected_center': 781},
        'measured': {'half_range': right['half_range'], 'crop_offset_x': right['crop_offset_x'],
                     'candidate_centers': cand_x,
                     'run_spans': run_spans,
                     'run_why': [r.get('why') for r in runs if r.get('span')],
                     'fallback_windows': causes,
                     'windows_containing_line': hit,
                     'crop_coord_leakage': leakage},
        'path': '候选 → run 拒绝记录 → fallback_cause 三处都用原图 x',
        'pass': ok,
        'reason': ('半页从 400 起、线在 780；三处记录都是原图 781（线心），'
                   '没有任何 381 这样的裁块坐标泄漏；覆盖该线的窗口 cause=rule_run_rejected'),
    }


def before_after(payload: Dict[str, Any]) -> Dict[str, Any]:
    old = load(HIST / 'search_regions_stage4-v1.json')
    old_pages = {p['page_id']: p for p in old['pages']}
    new_pages = {p['page_id']: p for p in payload['pages']}

    def snap(pid: str, pages: Dict[str, Any]) -> Dict[str, Any]:
        p = pages.get(pid)
        if p is None:
            return {'missing': True}
        cross = _crossing_regions(p)
        right = (p.get('column_validation') or {}).get('right', {})
        cov = {c['status'] for c in (p.get('group_coverage') or [])}
        return {'regions': len(p.get('regions') or []),
                'by_source': (p.get('coverage') or {}).get('by_source'),
                'crossing_regions': cross,
                'crossing_count': len(cross),
                'right_accepted': right.get('accepted'),
                'right_rejected': [(x['width'], x['reasons'][0]) for x in
                                   (right.get('rejected_detail') or [])],
                'max_region_width': max((r['bbox'][2] - r['bbox'][0]
                                         for r in p.get('regions') or []), default=0),
                'page_scale': {k: (p.get('page_scale') or {}).get(k)
                               for k in ('H', 'G', 'glyph_pitch', 'sample_count',
                                         'confidence_status', 'value_origin')},
                'group_coverage': sorted(cov),
                'n_groups': len(p.get('group_coverage') or [])}

    p40 = {'v1': snap('思濟堂方書/page_40', old_pages),
           'v2': snap('思濟堂方書/page_40', new_pages)}
    p12 = {'v1': snap('丹方彚编/page_12', old_pages),
           'v2': snap('丹方彚编/page_12', new_pages)}
    wy = {'v1': snap('温疫方論/page_12', old_pages),
          'v2': snap('温疫方論/page_12', new_pages)}
    return {
        'E_p40': {'case': 'E_p40_cross_half_boxes',
                  'v1': p40['v1'], 'v2': p40['v2'],
                  'pass': p40['v1']['crossing_count'] == 2
                          and p40['v2']['crossing_count'] == 0,
                  'reason': ('v1 的成组栏线在整页坐标里做，跨书脊的两条栏被配成一组，'
                             '左半页出现 [516,547]、右半页出现 [480,512] 两个对侧框；'
                             'v2 逐半页裁剪后两半各自只有本侧候选，跨界框归零')},
        'F_p12': {'case': 'F_p12_446_and_groups',
                  'v1': p12['v1'], 'v2': p12['v2'],
                  'pass': (p12['v2']['right_accepted'] == 0
                           and p12['v2']['max_region_width'] < 400
                           and p12['v2']['n_groups'] == 11
                           and p12['v2']['group_coverage'] == ['fully_covered']),
                  'reason': ('446px 框仍被宽度校验拒绝，右半页全部走 fallback；'
                             '11 个标注组仍全部落在搜索区域内；尺度改为逐半页实测')},
        'G_wenyi': {'case': 'G_wenyi_p12_scale',
                    'v1': wy['v1'], 'v2': wy['v2'],
                    'reason': ('v1 因行活跃度按 0/255 求和，几乎每行都「有墨」，'
                               '字形带被并成长条后按「高 > 1.4 列宽」全部丢弃，'
                               '页面尺度失败（只有 16 个样本、G/pitch 为空）；'
                               'v2 归一为布尔掩膜后重估，H/G/pitch 齐全')},
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='stage4 反例与路径判定')
    ap.add_argument('--regions', default=str(S4 / 'search_regions.json'))
    ap.add_argument('--out', default=str(S4 / 'path_decisions.json'))
    args = ap.parse_args(argv)

    payload = load(Path(args.regions))
    cases = {
        'stage': 'stage4_path_decisions',
        'release': payload.get('release'),
        'config_version': CFG['version'],
        'column_cases': column_cases(),
        'half_isolation': half_isolation_case(),
        'broken_merge': broken_merge_cases(),
        'coord_offset': coord_offset_case(),
        'bad_gap': bad_gap_case(),
        'encoding': encoding_cases(),
        'before_after': before_after(payload),
    }
    # 掩膜可重建性：拿 p40 的记录重建并比对哈希
    p40 = next(p for p in payload['pages'] if p['page_id'] == '思濟堂方書/page_40')
    rec = page_mask_record(p40)
    chk = verify_record(rec)
    cases['mask_rebuild'] = {
        'case': 'H_mask_rebuild',
        'page_id': p40['page_id'],
        'recorded': {k: v['sha256'] for k, v in rec['masks'].items()},
        'verify': chk,
        'pass': all(v['match'] for v in chk.values()),
        'reason': ('只凭 rule_masks.json 里的线清单与编码说明重建 0/1 掩膜，'
                   '两种口径的 sha256 都与记录一致'),
    }
    ok = (cases['half_isolation']['pass'] and cases['bad_gap']['pass']
          and cases['broken_merge']['pass'] and cases['coord_offset']['pass']
          and cases['encoding']['pass'] and cases['before_after']['E_p40']['pass']
          and cases['before_after']['F_p12']['pass'] and cases['mask_rebuild']['pass'])
    cases['summary'] = {'all_pass': ok,
                        'cases': {k: cases[k].get('pass') for k in
                                  ('half_isolation', 'broken_merge', 'coord_offset',
                                   'bad_gap', 'encoding', 'mask_rebuild')}}
    Path(args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(json.dumps(cases['summary'], ensure_ascii=False))
    print('列框样例路径：', [(c['case'], c['path']) for c in cases['column_cases']])
    print('E p40 跨界框 v1/v2：', cases['before_after']['E_p40']['v1']['crossing_count'],
          '→', cases['before_after']['E_p40']['v2']['crossing_count'])
    print('F p12 组数/覆盖：', cases['before_after']['F_p12']['v2']['n_groups'],
          cases['before_after']['F_p12']['v2']['group_coverage'])
    print('→', Path(args.out))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
