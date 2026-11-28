# -*- coding: utf-8 -*-
"""三级选择编排：逐半页决定「用哪条路径切出搜索区域」。

  路径一 grouped_rules      成组栏线（最强的证据）
  路径二 validated_columns  校验通过的列框（含按栏线拆分）
  路径三 halfpage_fallback  低置信局部窗口（前两条覆盖不到的范围）

一条半页可以同时用三条：左侧有界行就用路径一，右侧列框被拒绝就退回路径三。
这不是「三选一」，而是「按 x 段选」——协议要求「每段正文范围都能追溯到
已选搜索区域或明确的未处理原因」。

v2 的两条硬约束（验收缺口 1）：
  1. **半页隔离**：栏线检测、成组、列框合并、正文纵向范围、回退窗口全部在
     半页裁块内计算；输出时 x += offset_x 回映原图，并逐个框对照 half 边界校验。
     这样书脊两侧的线不会被配成一组，p40 那种「对侧错框」不会再有。
  2. **范围外 ≠ 应处理未覆盖**：
     out_of_scope   中缝带、页边留白、半页之外——本来就不该出区域；
     uncovered      应处理（半页正文范围）却没有任何区域覆盖——是缺口，要计入完成度。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .columns import merge_split_columns, reject_wide_boxes, validate_columns
from .rules import (classify_rule_touching, detect_rule_candidates, group_rule_runs,
                    merge_broken_rules)
from .scale import ScaleEstimate, estimate_scale


@dataclass
class SearchRegion:
    page_id: str
    half_id: str
    region_id: str
    bbox: List[int]                 # 原图 [x1,y1,x2,y2)
    column_range_source: str        # grouped_rules / validated_columns / halfpage_fallback
    low_confidence: bool
    scale: Dict[str, Any]
    validation_measurements: Dict[str, Any] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    parent_column_bbox: Optional[List[int]] = None
    overlap_with: List[str] = field(default_factory=list)
    transform: Dict[str, Any] = field(default_factory=lambda: {'crop': None, 'scale': 1.0,
                                                               'rotate': 0,
                                                               'offset_x': 0})
    uses_reference_position: bool = False
    clipped: bool = False
    fallback_cause: Optional[str] = None
    rule_group_id: Optional[str] = None
    half_bounds: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegionPlan:
    page_id: str
    image_size: List[int]
    halves: List[List[int]]
    regions: List[SearchRegion]
    rule_candidates: List[Dict[str, Any]]
    rule_groups: List[Dict[str, Any]]
    column_validation: Dict[str, Any]
    fallback: List[Dict[str, Any]]
    unprocessed: List[Dict[str, Any]]
    page_scale: Dict[str, Any]
    coverage: Dict[str, Any]
    status: str
    status_reason: str = ''
    timing_s: float = 0.0
    out_of_scope: List[Dict[str, Any]] = field(default_factory=list)
    halves_detail: List[Dict[str, Any]] = field(default_factory=list)
    rule_lines: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['regions'] = [r.to_dict() for r in self.regions]
        return d


# --------------------------------------------------------------------------- 工具

def _merge_intervals(segments: Sequence[Tuple[int, int]], a: int, b: int
                     ) -> List[Tuple[int, int]]:
    segs = sorted((max(a, s0), min(b, s1)) for s0, s1 in segments if s1 > a and s0 < b)
    out: List[List[int]] = []
    for s, e in segs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(int(s), int(e)) for s, e in out if e > s]


def _gaps(segments: Sequence[Tuple[int, int]], a: int, b: int,
          min_len: int = 24) -> List[Tuple[int, int]]:
    """未被覆盖的 x 段。"""
    out, cur = [], a
    for s0, s1 in _merge_intervals(segments, a, b):
        if s0 - cur >= min_len:
            out.append((cur, s0))
        cur = max(cur, s1)
    if b - cur >= min_len:
        out.append((cur, b))
    return out


def _covered(segments: Sequence[Tuple[int, int]], a: int, b: int,
             min_overlap: float = 0.6) -> bool:
    for s0, s1 in segments:
        inter = max(0, min(b, s1) - max(a, s0))
        if inter >= min_overlap * max(1, b - a):
            return True
    return False


def _body_y_range(columns: Sequence[Dict[str, Any]], bw: np.ndarray,
                  image_size: Tuple[int, int], cfg: Dict[str, Any]
                  ) -> Tuple[int, int, str]:
    """半页内的正文纵向范围：优先用本半页列框的纵向中位数。"""
    H = image_size[1]
    if columns:
        y0 = int(np.median([c['bbox'][1] for c in columns]))
        y1 = int(np.median([c['bbox'][3] for c in columns]))
        if y1 - y0 > H * 0.5:
            return max(0, y0), min(H, y1), 'half_columns_median'
    lo, hi = cfg['scale']['height_band_ratio']
    return int(H * lo), int(H * hi), 'image_height_ratio'


def _region_scale(bw: np.ndarray, x1: int, x2: int, y_range: Tuple[int, int],
                  image_size: Tuple[int, int], cfg: Dict[str, Any],
                  page_scale: Optional[ScaleEstimate],
                  columns: Sequence[Dict[str, Any]] = (),
                  narrow: bool = True,
                  exclude_rules: Sequence[Tuple[int, int]] = ()) -> Dict[str, Any]:
    """区域级尺度：先用区域内的真实列样本，不够再继承页面级（标明来源）。"""
    boxes = []
    for c in columns:
        b = [int(v) for v in c['bbox']]
        a, bb = max(int(x1), b[0]), min(int(x2), b[2])
        if bb - a >= 12:
            boxes.append([a, b[1], bb, b[3]])
    if not boxes:
        boxes = [[int(x1), int(y_range[0]), int(x2), int(y_range[1])]]
    est = estimate_scale(bw, [{'bbox': b} for b in boxes], y_range, image_size,
                         cfg['scale'], exclude_rules=exclude_rules,
                         narrow_columns=len(boxes) if narrow else 10 ** 6)
    if est.confidence_status == 'failed' and page_scale is not None:
        est.inherit_missing_from(
            page_scale,
            f'本区域自身样本不足（{est.failure_reason}），按字段继承同页尺度')
    return est.to_dict()


def _patch(effective: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """深拷贝生效配置的浅层补丁（小工具，避免到处 dict(...)）。"""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in effective.items()}
    for k, v in kwargs.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- 半页

def plan_halfpage(page_id: str, half_id: str, half_range: Tuple[int, int],
                  bw: np.ndarray, columns: Sequence[Dict[str, Any]],
                  image_size: Tuple[int, int], cfg: Dict[str, Any],
                  book_stats: Optional[Dict[str, Any]] = None,
                  global_stats: Optional[Dict[str, Any]] = None,
                  page_scale: Optional[ScaleEstimate] = None,
                  page_stats: Optional[Dict[str, Any]] = None
                  ) -> Tuple[List[SearchRegion], Dict[str, Any]]:
    """半页内按 x 段选路径，返回 (regions, diagnostics)。"""
    W, H = image_size
    hx0 = max(0, min(W - 1, int(half_range[0])))
    hx1 = max(hx0 + 1, min(W, int(half_range[1])))
    transform = {'crop': [hx0, 0, hx1, H], 'offset_x': hx0, 'scale': 1.0, 'rotate': 0,
                 'note': '半页裁块内计算，输出 x += offset_x 回映原图'}
    half_cols = [c for c in columns
                 if hx0 - 2 <= (c['bbox'][0] + c['bbox'][2]) / 2.0 < hx1 + 2]
    y0, y1, y_basis = _body_y_range(half_cols, bw, image_size, cfg)
    body_h = max(1, int(y1) - int(y0))
    diag: Dict[str, Any] = {
        'half_id': half_id, 'half_range': [hx0, hx1], 'transform': transform,
        'body_y': [y0, y1], 'body_y_basis': y_basis, 'body_h_px': body_h,
        'coord_system': 'original_image [x1,y1,x2,y2) 半开区间；'
                        f'裁块计算时已加 offset_x={hx0}',
        'crop_offset_x': hx0,
        'columns_in_half': len(half_cols), 'paths': {}, 'audit': [],
        'coverage': {}}
    regions: List[SearchRegion] = []
    seq = {'grouped_rules': 0, 'validated_columns': 0, 'halfpage_fallback': 0}
    crop = bw[:, hx0:hx1]

    # ---------------- 路径一：成组栏线（裁剪块内检测 + 成组，坐标回映）
    cands = detect_rule_candidates(crop, (y0, y1), cfg['rules'])
    cands = classify_rule_touching(cands, crop, cfg['rules'])
    cands, broken_log = merge_broken_rules(cands, cfg['rules'], body_h=body_h)
    groups, run_log = group_rule_runs(cands, crop, (y0, y1), cfg['rules'])

    for c in cands:                                   # 回映原图（只加一次）
        c.x0 += hx0
        c.x1 += hx0
        for s in c.segments:                          # 逐段证据也回映
            s['x0'] += hx0
            s['x1'] += hx0
    for g in groups:
        # 组的 lines 与 cands 是同一批对象，上面已回映；这里只回映栏区间
        g.columns = [(a + hx0, b + hx0) for a, b in g.columns]
        g.span = (g.span[0] + hx0, g.span[1] + hx0)
        if isinstance(g.diagnostics.get('run_span'), (list, tuple)):
            g.diagnostics['run_span'] = [g.diagnostics['run_span'][0] + hx0,
                                         g.diagnostics['run_span'][1] + hx0]
    # 审计记录统一回映原图：断线合并、run 切分都按同一坐标系
    for item in broken_log:
        item['x'] = int(item.get('x', 0)) + hx0
        if 'x1' in item:
            item['x1'] = int(item['x1']) + hx0
        for s in item.get('segment_detail') or []:
            s['x0'] += hx0
            s['x1'] += hx0
    for item in run_log:
        if isinstance(item.get('span'), (list, tuple)):
            item['span'] = [round(float(item['span'][0]) + hx0, 1),
                            round(float(item['span'][1]) + hx0, 1)]
        if 'x' in item:
            item['x'] = int(item['x']) + hx0
        if 'x1' in item:
            item['x1'] = int(item['x1']) + hx0
        if isinstance(item.get('x_list'), list):
            item['x_list'] = [round(float(v) + hx0, 1) for v in item['x_list']]
        item['coord_system'] = 'original_image'
    for gi, group in enumerate(groups):              # 归属标记（供掩膜记录使用）
        gid = f'{page_id}#{half_id}#rg{gi:02d}'
        for ln in group.lines:
            ln.group_id = gid
    rule_lines = [dict(c.to_dict(), half_id=half_id, coord_system='original_image')
                  for c in cands]
    diag['rule_candidates'] = rule_lines
    diag['broken_merge'] = broken_log
    diag['rule_runs'] = run_log
    diag['paths']['grouped_rules'] = {
        'ok': bool(groups), 'groups': [g.to_dict() for g in groups],
        'group_count': len(groups), 'run_log': run_log,
        'coord_system': 'original_image'}

    def eligible_bands(lines: Sequence[Any], lo: int, hi: int) -> List[Tuple[int, int, int, int]]:
        """统一资格判据下的排除带：只取合格线的**独立段**（4 元组，带 y 区间）。"""
        out = []
        for ln in lines:
            if not ln.qualified:
                continue
            if not (lo <= ln.center <= hi):
                continue
            for s in ln.segment_list():
                if s.get('eligible'):
                    out.append((int(s['x0']), int(s['x1']), int(s['y0']), int(s['y1'])))
        return out

    used_segments: List[Tuple[int, int]] = []
    for gi, group in enumerate(groups):
        gid = f'{page_id}#{half_id}#rg{gi:02d}'
        for k, (a, b) in enumerate(group.columns):
            a, b = max(hx0, int(a)), min(hx1, int(b))
            if b - a < 12:
                diag['audit'].append({'kind': 'rule_column_skipped',
                                      'bbox': [a, y0, b, y1],
                                      'why': f'栏宽 {b - a}px < 12px'})
                continue
            seq['grouped_rules'] += 1
            excl = eligible_bands(group.lines, a, b)
            sc = _region_scale(bw, a, b, (y0, y1), image_size, cfg, page_scale,
                               columns=[c for c in half_cols
                                        if hx0 <= (c['bbox'][0] + c['bbox'][2]) / 2 < hx1
                                        and c['bbox'][0] < b and c['bbox'][2] > a],
                               narrow=True, exclude_rules=excl)
            regions.append(SearchRegion(
                page_id=page_id, half_id=half_id,
                region_id=f'{page_id}#{half_id}#grouped_rules#{seq["grouped_rules"]:03d}',
                bbox=[a, y0, b, y1],
                column_range_source='grouped_rules', low_confidence=False, scale=sc,
                validation_measurements={'rule_pitch_px': round(group.pitch_px, 2),
                                         'rule_pitch_cv': group.pitch_cv,
                                         'line_kind': [ln.kind for ln in group.lines],
                                         'column_index_in_group': k,
                                         'rule_group_id': gid,
                                         'run_index': group.run_index},
                rejection_reasons=[], rule_group_id=gid, half_bounds=[hx0, hx1],
                transform=dict(transform)))
            used_segments.append((a, b))

    # ---------------- 路径二：校验列框（半页内合并 + 校验，判定 accept/split/fallback）
    ref_for_merge = (book_stats or {}).get('median_width') or \
                    (page_stats or {}).get('median_width') or \
                    (global_stats or {}).get('median_width')
    merged_cols, merge_log = merge_split_columns(half_cols, ref_for_merge, cfg['columns'])
    colres = validate_columns(merged_cols, (hx0, hx1), image_size, book_stats, global_stats,
                              cfg=cfg['columns'], page_stats=page_stats,
                              rule_lines=[c.to_dict() for c in cands])
    diag['column_merge'] = {'before': len(half_cols), 'after': len(merged_cols),
                            'log': merge_log}
    diag['column_validation'] = {
        'accepted': len(colres['accepted']), 'rejected': len(colres['rejected']),
        'split': sum(1 for v in colres['accepted'] if v.decision == 'split'),
        'reference': colres['reference'], 'notes': colres['notes'],
        'tight_gutters': colres['tight_gutters'],
        'verdicts': [v.to_dict() for v in colres['verdicts']]}
    diag['paths']['validated_columns'] = diag['column_validation']

    pad = max(3, int(0.35 * page_scale.H)) if (page_scale and page_scale.H) \
        else max(2, int(0.12 * max(1, colres['reference']['value_px'] or 30)))
    for v in colres['accepted']:
        parts = v.outputs or [{'bbox': list(v.bbox), 'role': 'body', 'split_index': 0}]
        last = len(parts) - 1
        for part in parts:
            pb = [int(x) for x in part['bbox']]
            # padding 只加在本体最外侧：拆分段之间不加，避免自己压到自己；
            # 也不越过与邻栏间隙的中点，避免两个区域互相压住对方主体
            prev_edge = max([c['bbox'][2] for c in merged_cols
                             if c['bbox'][2] <= pb[0]], default=None)
            next_edge = min([c['bbox'][0] for c in merged_cols
                             if c['bbox'][0] >= pb[2]], default=None)
            pad_l = pad if prev_edge is None else min(pad, max(0, (pb[0] - prev_edge) // 2))
            pad_r = pad if next_edge is None else min(pad, max(0, (next_edge - pb[2]) // 2))
            rx1 = pb[0] - (pad_l if part['split_index'] == 0 else 0)
            rx2 = pb[2] + (pad_r if part['split_index'] == last else 0)
            rx1, rx2 = max(hx0, rx1), min(hx1, rx2)
            if _covered(used_segments, rx1, rx2, min_overlap=0.7):
                diag['audit'].append({'kind': 'column_skipped',
                                      'bbox': [rx1, y0, rx2, y1],
                                      'why': '已被成组栏线覆盖，不重复出区域'})
                continue
            seq['validated_columns'] += 1
            excl = eligible_bands(cands, rx1, rx2)
            sc = _region_scale(bw, rx1, rx2, (y0, y1), image_size, cfg, page_scale,
                               columns=[c for c in half_cols
                                        if c['bbox'][0] < rx2 and c['bbox'][2] > rx1],
                               narrow=True, exclude_rules=excl)
            measurements = {**{k: val for k, val in v.measurements.items()},
                            'col_index': v.col_index, 'width_px': v.width,
                            'column_bbox': list(v.bbox), 'padding_px': pad,
                            'decision': v.decision, 'role': part['role'],
                            'split_index': part['split_index'],
                            'split_count': len(parts),
                            'rule_bands_excluded': excl}
            regions.append(SearchRegion(
                page_id=page_id, half_id=half_id,
                region_id=f'{page_id}#{half_id}#validated_columns#%03d'
                          % seq['validated_columns'],
                bbox=[rx1, y0, rx2, y1],
                column_range_source='validated_columns', low_confidence=False, scale=sc,
                validation_measurements=measurements,
                rejection_reasons=list(v.reasons) if v.decision == 'split' else [],
                parent_column_bbox=list(v.bbox), half_bounds=[hx0, hx1],
                transform=dict(transform)))
            used_segments.append((rx1, rx2))

    for v in colres['rejected']:
        diag['audit'].append({'kind': 'column_rejected', 'bbox': v.bbox,
                              'decision': v.decision, 'reasons': list(v.reasons)})

    # ---------------- 路径三：局部窗口（覆盖剩余 x 段；逐段记原因）
    wide_rejected = reject_wide_boxes(half_cols, image_size)
    diag['wide_rejected'] = wide_rejected
    fb_cfg = cfg['windows']
    base_scale = page_scale
    min_len = int(cfg['scope']['uncovered_min_len_px'])
    gaps = _gaps(used_segments, hx0, hx1, min_len=min_len)
    diag['used_segments'] = [list(map(int, s)) for s in used_segments]
    diag['uncovered_gaps_before_fallback'] = [list(g) for g in gaps]

    def cause_of(g0: int, g1: int) -> Dict[str, Any]:
        for v in colres['rejected']:
            inter = max(0, min(g1, v.bbox[2]) - max(g0, v.bbox[0]))
            if inter > 0.5 * max(1, g1 - g0):
                return {'cause': 'rejected_column', 'col_index': v.col_index,
                        'column_bbox': list(v.bbox), 'decision': v.decision,
                        'reasons': list(v.reasons),
                        'coord_system': 'original_image'}
        for rl in run_log:
            span = rl.get('span')
            if not span:
                continue
            mid = (float(span[0]) + float(span[1])) / 2.0
            inside = g0 <= mid <= g1
            overlap = max(0, min(g1, float(span[1])) - max(g0, float(span[0])))
            if inside or overlap > 0.5 * max(1, g1 - g0):
                return {'cause': 'rule_run_rejected', 'why': rl.get('why'),
                        'run_span': [round(float(span[0]), 1), round(float(span[1]), 1)],
                        'detail': rl.get('detail'),
                        'coord_system': 'original_image'}
        return {'cause': 'no_rule_or_column',
                'detail': '该段既无可信栏线，也无校验通过的列框',
                'coord_system': 'original_image'}

    fb_log: Dict[str, Any] = {'width_px': None, 'step_px': None, 'windows': [],
                              'gaps': [], 'narrow_windows': 0}
    uncovered: List[Dict[str, Any]] = []
    gutters: List[Dict[str, Any]] = []
    if gaps and (base_scale is None or base_scale.H is None):
        for g0, g1 in gaps:
            uncovered.append({'range': [int(g0), int(g1)],
                              'kind': 'uncovered_should_process',
                              'why': 'fallback 失败：没有可信 H，无法确定窗口宽度',
                              **cause_of(g0, g1)})
        diag['paths']['halfpage_fallback'] = {
            'ok': False, 'reason': 'no_scale：没有可信 H，无法确定窗口宽度与步长',
            'gaps': [list(g) for g in gaps]}
        diag['audit'].append({'kind': 'fallback_failed', 'why': 'no_scale',
                              'gaps': [list(g) for g in gaps]})
    elif gaps:
        width = int(np.clip(base_scale.H * fb_cfg['width_multiplier'],
                            fb_cfg['min_width_px'], fb_cfg['max_width_px']))
        step = max(12, int(width * fb_cfg['step_ratio']))
        hh = int((y1 - y0) * fb_cfg['min_height_ratio'])
        win_y0 = max(y0, int((y0 + y1) / 2 - hh / 2))
        fb_log.update({'width_px': width, 'step_px': step, 'height_px': hh,
                       'width_basis': f"H={base_scale.H:.1f}px × {fb_cfg['width_multiplier']}"
                                      f"(clamp {fb_cfg['min_width_px']}~{fb_cfg['max_width_px']})",
                       'uncovered_gaps': [list(map(int, g)) for g in gaps]})
        for g0, g1 in gaps:
            cause = cause_of(g0, g1)
            gw = g1 - g0
            starts: List[int] = []
            if gw <= width:
                starts = [g0]
            else:
                x = g0
                while x + width < g1:
                    starts.append(x)
                    x += step
                starts.append(g1 - width)
            # 去掉几乎重合的窗口（尾窗对齐会把起点挤到上一个窗口身上）
            clean: List[int] = []
            for s in starts:
                if clean and s - clean[-1] < max(1, step // 2):
                    continue
                clean.append(s)
            for wi, x in enumerate(clean):
                xx1 = max(hx0, x)
                xx2 = min(hx1, g1, x + width)      # 窗口不得越过本段末尾
                # 最后一段若差不到一个步长，就把它补到段末，避免留下几十像素的尾巴
                if wi == len(clean) - 1 and g1 - xx2 <= step:
                    xx2 = min(hx1, g1)
                if xx2 - xx1 < int(fb_cfg['narrow_gap_min_px']):
                    # 窄于窗口下限：记「残余栏缝」，不算应处理未覆盖（这不是漏掉的栏）
                    gutters.append({'range': [int(xx1), int(xx2)],
                                    'kind': 'gutter_residual',
                                    'why': f'仅 {xx2 - xx1}px < 窗口下限 '
                                           f'{fb_cfg["narrow_gap_min_px"]}px，'
                                           f'不足以容纳一栏，也未生成窗口',
                                    **cause})
                    continue
                seq['halfpage_fallback'] += 1
                rid = f'{page_id}#{half_id}#halfpage_fallback#%03d' % seq['halfpage_fallback']
                overlap = []
                for prev in regions:
                    if prev.column_range_source != 'halfpage_fallback':
                        continue
                    inter = max(0, min(xx2, prev.bbox[2]) - max(xx1, prev.bbox[0]))
                    if inter > 0:
                        overlap.append({'region_id': prev.region_id,
                                        'overlap_px': int(inter),
                                        'ratio_of_width': round(inter / max(1, xx2 - xx1), 3)})
                sc = _region_scale(bw, xx1, xx2, (win_y0, win_y0 + hh), image_size, cfg,
                                   page_scale, columns=[], narrow=True,
                                   exclude_rules=[])
                narrow_win = (xx2 - xx1) < int(fb_cfg['min_width_px'])
                if narrow_win:
                    fb_log['narrow_windows'] += 1
                regions.append(SearchRegion(
                    page_id=page_id, half_id=half_id, region_id=rid,
                    bbox=[xx1, win_y0, xx2, win_y0 + hh],
                    column_range_source='halfpage_fallback', low_confidence=True, scale=sc,
                    validation_measurements={'window_width_px': xx2 - xx1,
                                             'window_height_px': hh, 'step_px': step,
                                             'gap_covered': [int(g0), int(g1)],
                                             'fallback_cause': cause['cause'],
                                             'narrow_gap_window': bool(narrow_win)},
                    rejection_reasons=[cause.get('detail') or
                                       f"来源：{cause['cause']}"],
                    overlap_with=[o['region_id'] for o in overlap],
                    fallback_cause=cause['cause'], half_bounds=[hx0, hx1],
                    transform=dict(transform)))
                fb_log['windows'].append({'region_id': rid,
                                          'bbox': [xx1, win_y0, xx2, win_y0 + hh],
                                          'cause': cause, 'overlap': overlap})
                used_segments.append((xx1, xx2))
        diag['paths']['halfpage_fallback'] = {'ok': bool(fb_log['windows']), **fb_log}
    else:
        diag['paths']['halfpage_fallback'] = {
            'ok': False, 'reason': 'nothing_left：前两条路径已覆盖整个半页',
            'windows': []}

    # ---------------- 逐框对照 half 边界校验（不信任任何来源）
    for r in regions:
        if r.bbox[0] < hx0 or r.bbox[2] > hx1:
            before = list(r.bbox)
            r.bbox[0] = max(hx0, r.bbox[0])
            r.bbox[2] = min(hx1, r.bbox[2])
            r.clipped = True
            r.validation_measurements['bbox_before_half_clip'] = before
            diag['audit'].append({'kind': 'half_clip', 'region_id': r.region_id,
                                  'before': before, 'after': list(r.bbox)})
        if not (hx0 <= r.bbox[0] < r.bbox[2] <= hx1):
            diag['audit'].append({'kind': 'half_violation', 'region_id': r.region_id,
                                  'bbox': list(r.bbox), 'half': [hx0, hx1]})
    regions = [r for r in regions if r.bbox[2] > r.bbox[0]]

    # ---------------- 半页完成度：范围外 vs 应处理未覆盖
    margin = max(int(cfg['scope']['edge_margin_min_px']),
                 int(cfg['scope']['edge_margin_ratio'] * (hx1 - hx0)))
    sx0, sx1 = hx0 + margin, hx1 - margin
    segs = [(r.bbox[0], r.bbox[2]) for r in regions]
    covered = _merge_intervals(segs, sx0, sx1)
    covered_px = sum(e - s for s, e in covered)
    should_px = max(1, sx1 - sx0)
    leftover = _gaps(segs, sx0, sx1, min_len=1)
    for g0, g1 in leftover:
        known = any(u['range'] == [g0, g1] for u in uncovered) or \
            any(u['range'] == [g0, g1] for u in gutters)
        if known:
            continue
        bounded = (any(r.bbox[2] <= g0 + 1 for r in regions)
                   and any(r.bbox[0] >= g1 - 1 for r in regions))
        if (g1 - g0) < int(fb_cfg['narrow_gap_min_px']) or (
                bounded and (g1 - g0) <= int(cfg['scope'].get('gutter_residual_max_px', 24))):
            # 两个搜索区域之间的残余栏缝：padding 已按半间隙截断，这里本就不该有区域
            gutters.append({'range': [int(g0), int(g1)], 'kind': 'gutter_residual',
                            'why': (f'两侧都有搜索区域，残余 {g1 - g0}px 为栏间空白'
                                    if bounded else
                                    f'仅 {g1 - g0}px，窄于窗口下限 '
                                    f'{fb_cfg["narrow_gap_min_px"]}px'),
                            **cause_of(g0, g1)})
            continue
        uncovered.append({'range': [int(g0), int(g1)],
                          'kind': 'uncovered_should_process',
                          'why': '该段没有任何搜索区域覆盖', **cause_of(g0, g1)})
    fb_by_cause: Dict[str, int] = {}
    for r in regions:
        if r.column_range_source == 'halfpage_fallback':
            fb_by_cause[r.fallback_cause or 'unknown'] = \
                fb_by_cause.get(r.fallback_cause or 'unknown', 0) + 1
    # 账本闭合：covered + gutter + uncovered 必须等于应处理长度（按区间并集算，不重复计）
    gutter_segs = [(g['range'][0], g['range'][1]) for g in gutters]
    accounted = _merge_intervals(segs + gutter_segs, sx0, sx1)
    accounted_px = sum(e - s for s, e in accounted)
    gutter_px = max(0, accounted_px - covered_px)
    uncovered_px = max(0, should_px - accounted_px)
    if uncovered_px > 0:
        for g0, g1 in _gaps(segs + gutter_segs, sx0, sx1, min_len=1):
            if not any(u['range'] == [g0, g1] for u in uncovered):
                uncovered.append({'range': [int(g0), int(g1)],
                                  'kind': 'uncovered_should_process',
                                  'why': '该段没有任何搜索区域覆盖',
                                  **cause_of(g0, g1)})
    diag['coverage'] = {
        'should_process_px': int(should_px), 'covered_px': int(covered_px),
        'processed_ratio': round(covered_px / should_px, 4),
        'accounted_ratio': round((covered_px + gutter_px) / should_px, 4),
        'margin_px': margin, 'should_process_range': [int(sx0), int(sx1)],
        'out_of_scope': [{'range': [int(hx0), int(sx0)], 'kind': 'out_of_scope',
                          'why': '半页左页边留白'} if sx0 > hx0 else None,
                         {'range': [int(sx1), int(hx1)], 'kind': 'out_of_scope',
                          'why': '半页右页边留白'} if hx1 > sx1 else None],
        'uncovered': uncovered,
        'uncovered_px': int(uncovered_px),
        'gutter_residual': gutters, 'gutter_px': int(gutter_px),
        'fallback_by_cause': fb_by_cause,
        'regions': len(regions), 'by_source': {k: v for k, v in seq.items() if v}}
    diag['coverage']['out_of_scope'] = [o for o in diag['coverage']['out_of_scope'] if o]
    return regions, diag


# --------------------------------------------------------------------------- 整页

def _page_scope(W: int, halves: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """整页范围之外：左右页边、半页之间的中缝带。"""
    out: List[Dict[str, Any]] = []
    hs = sorted((max(0, int(a)), min(W, int(b))) for a, b in halves)
    if hs and hs[0][0] > 0:
        out.append({'range': [0, hs[0][0]], 'kind': 'out_of_scope',
                    'why': '页面最左侧不在任何半页范围内（版框外/页边）'})
    for a, b in zip(hs, hs[1:]):
        if b[0] > a[1]:
            out.append({'range': [int(a[1]), int(b[0])], 'kind': 'out_of_scope',
                        'why': '两个半页之间的中缝带（书脊），不属于任何一侧正文'})
    if hs and hs[-1][1] < W:
        out.append({'range': [hs[-1][1], W], 'kind': 'out_of_scope',
                    'why': '页面最右侧不在任何半页范围内（版框外/页边）'})
    return out


def plan_page(page_id: str, image_path: str, bw: np.ndarray, image_size: Tuple[int, int],
              halves: Sequence[Tuple[int, int]], columns: Sequence[Dict[str, Any]],
              cfg: Dict[str, Any], book_stats: Optional[Dict[str, Any]] = None,
              global_stats: Optional[Dict[str, Any]] = None
              ) -> RegionPlan:
    """整页：逐半页走三级选择，并给出页面级尺度与完成度。"""
    import time
    t0 = time.time()
    W, H = image_size
    page_cols = [{'bbox': list(c['bbox'])} for c in columns]
    y0, y1, y_basis = _body_y_range(page_cols, bw, image_size, cfg)
    page_scale = estimate_scale(bw, page_cols, (y0, y1), image_size, cfg['scale'],
                                narrow_columns=len(page_cols))
    widths = [int(c['bbox'][2]) - int(c['bbox'][0]) for c in page_cols if
              int(c['bbox'][2]) > int(c['bbox'][0])]
    page_stats = {'cols': len(widths),
                  'median_width': round(float(np.median(widths)), 3) if widths else None}

    all_regions: List[SearchRegion] = []
    half_diags: List[Dict[str, Any]] = []
    for k, (hx0, hx1) in enumerate(halves):
        if len(halves) == 1:
            hid = 'full'
        elif k == 0:
            hid = 'right'
        elif k == 1:
            hid = 'left'
        else:
            hid = f'h{k}'
        regs, diag = plan_halfpage(page_id, hid, (hx0, hx1), bw, columns, image_size, cfg,
                                   book_stats, global_stats, page_scale, page_stats)
        all_regions.extend(regs)
        half_diags.append(diag)

    out_of_scope = _page_scope(W, halves)
    for hd in half_diags:
        out_of_scope.extend([{**o, 'half_id': hd['half_id']}
                             for o in (hd['coverage'].get('out_of_scope') or [])])

    unprocessed: List[Dict[str, Any]] = []
    for hd in half_diags:
        for u in hd['coverage'].get('uncovered') or []:
            unprocessed.append({'half_id': hd['half_id'], **u})

    covered_px = sum((hd['coverage'].get('covered_px') or 0) for hd in half_diags)
    should_px = sum((hd['coverage'].get('should_process_px') or 0) for hd in half_diags)
    uncovered_px = sum((hd['coverage'].get('uncovered_px') or 0) for hd in half_diags)
    gutter_px = sum((hd['coverage'].get('gutter_px') or 0) for hd in half_diags)

    if not all_regions:
        status, reason = 'failed', '没有产生任何搜索区域'
    elif uncovered_px > 0:
        status, reason = 'completed_with_gaps', \
            f'有 {len(unprocessed)} 段应处理范围无区域覆盖（{uncovered_px}px）'
    elif any(r.column_range_source == 'grouped_rules' for r in all_regions):
        status, reason = 'completed', '至少一个半页有成组栏线，且无覆盖缺口'
    else:
        status, reason = 'completed_with_fallback', '无成组栏线，靠列框与局部窗口覆盖'

    by_source: Dict[str, int] = {}
    for r in all_regions:
        by_source[r.column_range_source] = by_source.get(r.column_range_source, 0) + 1
    fb_by_cause: Dict[str, int] = {}
    for r in all_regions:
        if r.column_range_source == 'halfpage_fallback':
            key = r.fallback_cause or 'unknown'
            fb_by_cause[key] = fb_by_cause.get(key, 0) + 1
    cov = {'regions': len(all_regions), 'by_source': by_source,
           'low_confidence_regions': sum(1 for r in all_regions if r.low_confidence),
           'clipped_regions': sum(1 for r in all_regions if r.clipped),
           'halves': len(halves),
           'halves_with_rules': sum(1 for h in half_diags
                                    if (h.get('paths', {}).get('grouped_rules') or {}).get('ok')),
           'halves_with_columns': sum(1 for h in half_diags
                                      if (h.get('column_validation') or {}).get('accepted')),
           'unprocessed_areas': len(unprocessed),
           'fallback_by_cause': fb_by_cause,
           'completion': {
               'should_process_px': int(should_px), 'covered_px': int(covered_px),
               'processed_ratio': round(covered_px / should_px, 4) if should_px else None,
               'gutter_px': int(gutter_px),
               'accounted_ratio': round((covered_px + gutter_px) / should_px, 4)
                                  if should_px else None,
               'uncovered_px': int(uncovered_px),
               'uncovered_areas': len(unprocessed),
               'out_of_scope_areas': len(out_of_scope),
               'out_of_scope_px': int(sum(b - a for a, b in
                                          (o['range'] for o in out_of_scope)))}}
    return RegionPlan(page_id=page_id, image_size=[int(W), int(H)],
                      halves=[[int(a), int(b)] for a, b in halves],
                      regions=all_regions,
                      rule_candidates=[c for h in half_diags for c in h.get('rule_candidates', [])],
                      rule_groups=[g for h in half_diags
                                   for g in (h['paths'].get('grouped_rules') or {}).get('groups', [])],
                      column_validation={h['half_id']: h.get('column_validation', {})
                                         for h in half_diags},
                      fallback=[h['paths'].get('halfpage_fallback', {}) for h in half_diags],
                      unprocessed=unprocessed,
                      page_scale=page_scale.to_dict(),
                      coverage=cov, status=status, status_reason=reason,
                      timing_s=round(time.time() - t0, 3),
                      out_of_scope=out_of_scope, halves_detail=half_diags,
                      rule_lines=[c for h in half_diags for c in h.get('rule_candidates', [])])


# --------------------------------------------------------------------------- 阶段 5 读取

def _page_of(payload: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    page = next((p for p in payload.get('pages', []) if p['page_id'] == page_id), None)
    if page is None:
        raise KeyError(f'page not found in stage4 output: {page_id}')
    return page


def regions_for_page(payload: Dict[str, Any], page_id: str) -> List[Dict[str, Any]]:
    """阶段 5 的固定读取入口：给 page_id，拿回该页全部搜索区域（含尺度）。"""
    return _page_of(payload, page_id)['regions']


def scale_for_page(payload: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    return _page_of(payload, page_id).get('page_scale') or {}


def rule_lines_for_page(payload: Dict[str, Any], page_id: str) -> List[Dict[str, Any]]:
    """栏线记录（含 kind / confidence / excluded_from_scale / group 归属）。"""
    page = _page_of(payload, page_id)
    if 'rule_lines' in page:
        return page['rule_lines']
    return page.get('rule_candidates') or []


def audit_for_page(payload: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    """逐页审计：范围外、应处理未覆盖、拒绝原因、回退来源。"""
    page = _page_of(payload, page_id)
    return {'status': page.get('status'), 'status_reason': page.get('status_reason'),
            'coverage': page.get('coverage'), 'unprocessed': page.get('unprocessed'),
            'out_of_scope': page.get('out_of_scope'),
            'halves': [{k: v for k, v in h.items()
                        if k in ('half_id', 'half_range', 'body_y', 'body_y_basis',
                                 'coverage', 'rule_runs', 'column_validation',
                                 'paths', 'audit', 'columns_in_half')}
                       for h in page.get('halves_detail', [])]}
