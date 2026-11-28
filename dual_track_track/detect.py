# -*- coding: utf-8 -*-
"""状态机：搜索 → 候选 → 已确认 → 独立追踪 → 完成/拒绝。

两遍扫描（本轮的关键结构）：
  第一遍：每个窗口全位置扫，记录**该窗最优缝**；
  聚类：把跨窗口出现次数够多的缝位置聚成「候选缝」（贯穿性，不是某一窗的偶然极小）；
  第二遍：对每条候选缝单独纵向扫描，走状态机，产出组。

这样做的原因：双轨小字的缝只有 4~8px，逐窗独立找局部极小会被「左右偏旁之间的
空隙」骗到，缝位一窗一跳。先聚类再按缝扫描，既稳定又天然支持
「回退窗口里有多条实体栏 → 每条缝各自成组」。

停止判据全部是枚举 + 原图 y + 「扫描在哪确认」：
  page_boundary / region_boundary / rule_boundary / next_title / blank_gap /
  seam_missing / seam_drift / track_ink_end / ambiguous_ink
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .projection import RegionInk, glyph_components
from .seam import SeamEval, evaluate_window


@dataclass
class Win:
    y0: int
    y1: int
    ev: SeamEval
    big: Optional[Dict[str, Any]] = None
    blank: int = 0
    candidate_seam: Optional[int] = None
    window_id: str = ''
    index: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.ev.valid)

    def to_dict(self, state: str = 'searching') -> Dict[str, Any]:
        d = self.ev.to_dict()
        d.update({'y': [self.y0, self.y1], 'state': state,
                  'big_band': self.big, 'blank_run_px': self.blank,
                  'candidate_seam': self.candidate_seam,
                  'window_id': self.window_id, 'index': self.index})
        return d


def _fit_seam(points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(points) < 2:
        xs = [p['x'] for p in points] or [0]
        return {'slope': 0.0, 'intercept': float(xs[0]),
                'resid_std': 0.0, 'max_resid': 0.0}
    ys = np.asarray([p['y'] for p in points], dtype=float)
    xs = np.asarray([p['x'] for p in points], dtype=float)
    A = np.vstack([ys, np.ones_like(ys)]).T
    slope, intercept = np.linalg.lstsq(A, xs, rcond=None)[0]
    resid = xs - (slope * ys + intercept)
    return {'slope': float(slope), 'intercept': float(intercept),
            'resid_std': float(resid.std()), 'max_resid': float(np.abs(resid).max())}


def _seam_at(fit: Dict[str, Any], y: float) -> float:
    return float(fit['slope']) * float(y) + float(fit['intercept'])


def _median(vals: Sequence[float]) -> Optional[float]:
    return float(np.median(vals)) if len(vals) else None


def _side_scale(comps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    comps = sorted(comps, key=lambda c: c['bbox'][1])
    gaps, pitches = [], []
    for a, b in zip(comps, comps[1:]):
        g = b['bbox'][1] - a['bbox'][3]
        if g >= 0:
            gaps.append(g)
            pitches.append((b['bbox'][1] + b['bbox'][3]) / 2.0
                           - (a['bbox'][1] + a['bbox'][3]) / 2.0)
    return {'n': len(comps),
            'height': _median([c['h'] for c in comps]),
            'width': _median([c['w'] for c in comps]),
            'gap': _median(gaps), 'pitch': _median(pitches),
            'y_span': ([comps[0]['bbox'][1], comps[-1]['bbox'][3]] if comps else None)}


def _union_box(boxes: Sequence[Sequence[int]]) -> Optional[List[int]]:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]


def _unique_boxes(boxes: Sequence[Sequence[int]]) -> List[List[int]]:
    """按完整字形框去重；同一字在重叠窗里出现多次只能算一个。"""
    out: List[List[int]] = []
    seen: set = set()
    for b in boxes:
        if not b:
            continue
        t = tuple(int(v) for v in b)
        if t in seen:
            continue
        seen.add(t)
        out.append(list(t))
    return out


def _n_unique(run: Sequence[Win], side: str) -> int:
    src = 'right_boxes' if side == 'right' else 'left_boxes'
    return len(_unique_boxes([b for w in run for b in getattr(w.ev, src)]))


def _blank_run(region: RegionInk, y0: int, y1: int) -> int:
    a = max(0, int(y0) - region.bbox[1])
    b = min(region.height, int(y1) - region.bbox[1])
    if b <= a:
        return 0
    rows = region.mask[a:b].any(axis=1)
    n = 0
    for v in rows:
        if v:
            break
        n += 1
    return n


def _big_band(glyphs: Sequence[Dict[str, Any]], y0: int, y1: int, H: float,
              cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    st = cfg['stop']
    big = []
    for g in glyphs:
        if g['bbox'][3] <= y0 or g['bbox'][1] >= y1:
            continue
        if g['w'] >= st['rule_width_H'] * H and g['h'] <= st['rule_height_H'] * H:
            return {'kind': 'rule_boundary', 'y': int(g['bbox'][1]),
                    'evidence': {'bbox': g['bbox'], 'w': g['w'], 'h': g['h']}}
        if g['h'] >= st['big_glyph_H'] * H and g['w'] >= st['big_glyph_w_H'] * H:
            big.append(g)
    if len(big) >= int(st['big_glyph_min_count']):
        g = big[0]
        return {'kind': 'next_title', 'y': int(g['bbox'][1]),
                'evidence': {'bbox': g['bbox'], 'h': g['h'], 'H': round(H, 1),
                             'count': len(big)}}
    return None


def _scan(region_ink: RegionInk, ys: Sequence[int], win_h: int, ry1: int,
          glyphs: Sequence[Dict[str, Any]], pitch: float, H: float,
          rule_xs: Sequence[float], cfg: Dict[str, Any],
          seam_filter: Optional[Sequence[int]] = None,
          prefer: Optional[int] = None,
          all_glyphs: Optional[Sequence[Dict[str, Any]]] = None,
          collect_alternatives: bool = False) -> List[Win]:
    out: List[Win] = []
    gap_thr = max(int(cfg['stop']['blank_gap_min_px']),
                  int(round(float(cfg['stop']['blank_gap_G']) * pitch)))
    for y0 in ys:
        y1 = min(ry1, y0 + win_h)
        big = _big_band(all_glyphs, y0, y1, H, cfg)
        blank = _blank_run(region_ink, y0, y1)
        ev = evaluate_window(region_ink, y0, y1, glyphs, pitch, H, rule_xs, cfg,
                             prefer_x=prefer, seam_candidates=seam_filter,
                             collect_alternatives=collect_alternatives)
        out.append(Win(y0=y0, y1=y1, ev=ev, big=big, blank=blank))
    return out


def _cluster_seams(values: Sequence[int], tol: float) -> List[Tuple[int, int]]:
    """一维聚类：返回 [(代表缝位, 出现窗数)]，按出现次数降序。"""
    if not values:
        return []
    vals = sorted(values)
    clusters: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    out = [(int(round(float(np.median(c)))), len(c)) for c in clusters]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def detect_region(page_id: str, region: Dict[str, Any], ink: np.ndarray,
                  rule_segments: Sequence[Sequence[int]], scale: Dict[str, Any],
                  cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bbox = region['bbox']
    H = scale.get('H')
    audit: Dict[str, Any] = {
        'region_id': region['region_id'], 'region_bbox': list(bbox),
        'source': region.get('column_range_source'),
        'low_confidence': bool(region.get('low_confidence')),
        'scale_status': {}, 'degraded': [], 'windows': [], 'seam_clusters': [],
        'candidate_runs': [], 'groups': [], 'skipped': None}
    if not H or H < cfg['scale']['min_H_px']:
        audit['skipped'] = 'no_usable_scale'
        return [], audit

    H = float(H)
    pitch = scale.get('glyph_pitch')
    G = scale.get('G')
    sc_status = {'H': {'value': round(H, 2), 'source': scale.get('value_origin'),
                       'confidence': scale.get('confidence_status'),
                       'status': 'measured'}}
    if pitch:
        pitch = float(pitch)
        sc_status['glyph_pitch'] = {'value': round(pitch, 2),
                                    'source': scale.get('value_origin'),
                                    'status': 'measured'}
    else:
        pitch = H * float(cfg['scale']['pitch_fallback_multiple'])
        sc_status['glyph_pitch'] = {'value': round(pitch, 2), 'source': 'fallback_1.0H',
                                    'status': 'fallback'}
        audit['degraded'].append('glyph_pitch_fallback')
    if G:
        G = float(G)
        sc_status['G'] = {'value': round(G, 2), 'source': scale.get('value_origin'),
                          'status': 'measured'}
    else:
        G = H * float(cfg['scale']['G_fallback_multiple'])
        sc_status['G'] = {'value': round(G, 2), 'source': 'fallback_0.25H',
                          'status': 'fallback'}
        audit['degraded'].append('G_fallback')
    iqr = (scale.get('dispersion') or {}).get('H_iqr')
    if iqr is not None and iqr > cfg['scale']['max_H_iqr_ratio'] * H:
        audit['degraded'].append('H_dispersion_high')
        sc_status['H']['status'] = 'measured_but_dispersed'
    audit['scale_status'] = sc_status
    audit['scale_origin'] = scale.get('value_origin')

    region_ink = RegionInk(bbox=list(bbox),
                           mask=(ink[bbox[1]:bbox[3], bbox[0]:bbox[2]] > 0)
                           .astype(np.uint8), raw=np.zeros((1, 1), np.uint8))
    for seg in rule_segments or []:
        sx0, sx1, sy0, sy1 = [int(v) for v in seg[:4]]
        a, b = max(0, sx0 - bbox[0]), min(region_ink.width, sx1 - bbox[0])
        c, d = max(0, sy0 - bbox[1]), min(region_ink.height, sy1 - bbox[1])
        if b > a and d > c:
            region_ink.mask[c:d, a:b] = 0
            region_ink.rule_bands.append([sx0, sx1, sy0, sy1])
    region_ink.components()
    all_glyphs = glyph_components(region_ink, cfg, H)
    # 大字带单独判，不参与轨框与中缝证据（否则标题字会被算进轨框）
    glyphs = [g for g in all_glyphs if g['h'] < cfg['stop']['big_glyph_H'] * H]
    rule_xs = [float((s[0] + s[1]) / 2.0) for s in (rule_segments or [])]

    win_h = int(round(float(cfg['window']['height_H']) * H))
    win_h = max(int(cfg['window']['min_height_px']),
                min(int(cfg['window']['max_height_px']), win_h))
    step = max(2, int(round(float(cfg['window']['step_H']) * H)))
    ry0, ry1 = int(bbox[1]), int(bbox[3])
    ys = list(range(ry0, max(ry0 + 1, ry1 - win_h + 1), step)) or [ry0]

    # ---------------- 第一遍：全位置扫描，收集每窗最优缝
    pass1 = _scan(region_ink, ys, win_h, ry1, glyphs, pitch, H, rule_xs, cfg,
                  all_glyphs=all_glyphs, collect_alternatives=True)
    for n, w in enumerate(pass1):
        w.candidate_seam = None
        w.index = n
        w.window_id = f"{region['region_id']}#p1#{w.y0:04d}"
    # 一个窗里可能有两条都合格的缝：都进聚类，各自独立跑状态机；
    # 不再让「本窗最优」把同窗另一条真缝永久压掉。
    pass1_xs: List[int] = []
    for w in pass1:
        if not w.ok:
            continue
        pass1_xs.append(int(w.ev.seam_x))
        pass1_xs.extend(int(a['seam_x']) for a in w.ev.alternatives)
    clusters = _cluster_seams(pass1_xs, tol=2.0)
    # 一条缝只要在足够多的窗口里出现（至少够确认的连续数）就算「贯穿」，
    # 不能用「全部窗口的百分比」——双轨区段常只占高区域的一小段
    min_windows = max(int(cfg['window']['min_consecutive']),
                      int(cfg['window']['cluster_min_windows']))
    keep = [(x, n) for x, n in clusters if n >= min_windows]
    audit['seam_clusters'] = [{'x': x, 'windows': n, 'kept': n >= min_windows}
                              for x, n in clusters]
    groups: List[Dict[str, Any]] = []
    if not keep:
        audit['windows'] = [w.to_dict('searching') for w in pass1]
        if clusters:
            audit['skipped'] = 'no_persistent_seam'
        return [], audit

    # ---------------- 第二遍：每条候选缝单独走状态机
    used_windows: Dict[Tuple[int, int], Win] = {}
    for seam_x, _n in keep:
        wins = _scan(region_ink, ys, win_h, ry1, glyphs, pitch, H, rule_xs, cfg,
                     seam_filter=[x for x in range(seam_x - 2, seam_x + 3)],
                     prefer=seam_x, all_glyphs=all_glyphs)
        for n, w in enumerate(wins):
            w.candidate_seam = int(seam_x)
            w.index = n
            w.window_id = f"{region['region_id']}#c{seam_x}#{w.y0:04d}"
            # 同一高度的多条候选缝各留一份通过/拒绝证据
            used_windows[(int(seam_x), w.y0)] = w
        gs, runs = _state_machine(page_id, region, wins, region_ink, glyphs,
                                  H, G, pitch, cfg, seam_x, rule_segments or [])
        groups.extend(gs)
        audit['candidate_runs'].extend(runs)
    audit['windows'] = [used_windows[k].to_dict()
                        for k in sorted(used_windows, key=lambda kv: (kv[0], kv[1]))]
    audit['groups'] = [g['group_id'] for g in groups]
    return groups, audit


def _lane_of(boxes: Sequence[Sequence[int]], H: float) -> Optional[Dict[str, float]]:
    """确认区段的轨横向范围；延伸只能在这条轨内接字形，不能跳到邻栏。"""
    if not boxes:
        return None
    x0 = min(float(b[0]) for b in boxes)
    x1 = max(float(b[2]) for b in boxes)
    return {'x0': x0, 'x1': x1, 'cx': (x0 + x1) / 2.0,
            'tol': max(0.8 * H, 0.5 * (x1 - x0))}


def _boxes_in_lane(w: Win, side: str, lane: Optional[Dict[str, float]],
                   seam_at_y: float, H: float,
                   rule_rects: Sequence[Sequence[int]] = (),
                   stop_y: Optional[int] = None,
                   hard_y: Optional[int] = None) -> List[List[int]]:
    """用完整字形框核对延伸资格：本轨边界内、不跨缝、不压栏线、不过断组点。

    v2 只看字形中心，半个框跨到邻栏/中缝仍会被接进轨；这里要求整个外接框
    落在确认轨的横向邻域内。延伸只更新**已确认的那一侧轨**，绝不新建轨。
    """
    src = w.ev.left_boxes if side == 'left' else w.ev.right_boxes
    lo = (lane['x0'] - lane['tol']) if lane is not None else -1e9
    hi = (lane['x1'] + lane['tol']) if lane is not None else 1e9
    out: List[List[int]] = []
    for b in src or []:
        x0, y0, x1, y1 = [int(v) for v in b]
        # 完整框必须在本轨横向范围内（含确认轨的允许漂移量）
        if x0 < lo or x1 > hi:
            continue
        # 不得跨中缝；右轨整体在缝右侧、左轨整体在缝左侧
        if side == 'right' and x0 < seam_at_y - 1.0:
            continue
        if side == 'left' and x1 > seam_at_y + 1.0:
            continue
        # 不得压在合格栏线上
        if any(sx0 < x1 and sx1 > x0 and sy0 < y1 and sy1 > y0
               for sx0, sx1, sy0, sy1 in rule_rects):
            continue
        # 不得越过该侧已确认的断组点
        if stop_y is not None and y0 >= stop_y:
            continue
        # 硬断点（标题/栏线/留白）按**整个字形框**约束：跨线即拒，不裁字
        if hard_y is not None and y1 > hard_y:
            continue
        out.append([x0, y0, x1, y1])
    return out


def _latest_glyph(boxes: Sequence[Sequence[int]]) -> Optional[List[int]]:
    return [int(v) for v in max(boxes, key=lambda b: int(b[3]))] if boxes else None


def _state_machine(page_id: str, region: Dict[str, Any], wins: Sequence[Win],
                   region_ink: RegionInk, glyphs: Sequence[Dict[str, Any]],
                   H: float, G: float, pitch: float,
                   cfg: Dict[str, Any], seam_x: int,
                   rule_rects: Sequence[Sequence[int]] = ()
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    N = int(cfg['window']['min_consecutive'])
    min_union = float(cfg['window']['min_union_H']) * H
    max_confirm = float(cfg['window']['max_confirm_H']) * H
    min_g = int(cfg['glyph']['min_per_side_group'])
    st = cfg['stability']
    i = 0
    while i < len(wins):
        if not wins[i].ok:
            i += 1
            continue
        run = [wins[i]]
        j = i
        while j + 1 < len(wins) and wins[j + 1].ok:
            med = float(np.median([w.ev.seam_x for w in run]))
            if abs(wins[j + 1].ev.seam_x - med) > 0.25 * H:
                break
            run.append(wins[j + 1])
            j += 1
            span = run[-1].y1 - run[0].y0
            unique_ok = all(_n_unique(run, side) >= min_g for side in ('right', 'left'))
            # 重叠窗里的同一个字不能重复计数：确认段要凑够两侧独立字形，
            # 到 max_confirm_H 仍不够就按证据不足拒绝。
            if (len(run) >= N and span >= min_union and unique_ok) or span >= max_confirm:
                break
        span = run[-1].y1 - run[0].y0
        unique_ok = all(_n_unique(run, side) >= min_g for side in ('right', 'left'))
        fit = _fit_seam([{'y': (w.y0 + w.y1) / 2.0, 'x': w.ev.seam_x} for w in run])
        stable = (abs(fit['slope']) <= st['max_abs_slope']
                  and fit['resid_std'] <= st['resid_std_H'] * H
                  and fit['max_resid'] <= st['max_resid_H'] * H)
        if len(run) < N or span < min_union or not stable or not unique_ok:
            reason = ('insufficient_windows' if len(run) < N else
                      'insufficient_span' if span < min_union else
                      'insufficient_unique_glyphs' if not unique_ok else
                      'seam_unstable')
            runs.append({'seam': seam_x, 'span': [run[0].y0, run[-1].y1],
                         'windows': len(run), 'confirmed': False,
                         'rejected': reason,
                         'unique_glyphs': {s: _n_unique(run, s)
                                           for s in ('right', 'left')},
                         'fit': {k: round(v, 4) for k, v in fit.items()}})
            i = j + 1
            continue

        conf_start, conf_end = run[0].y0, run[-1].y1
        points = [{'y': (w.y0 + w.y1) / 2.0, 'x': w.ev.seam_x} for w in run]
        # 确认前按完整字形框去重：重叠窗里的同一个字只能计一次，
        # 左右轨分别过 min_per_side_group 门槛（v2 曾把 1 个字算成 3 个）。
        boxes = {'right': _unique_boxes([b for w in run for b in w.ev.right_boxes]),
                 'left': _unique_boxes([b for w in run for b in w.ev.left_boxes])}
        for side in ('right', 'left'):
            if not boxes[side]:
                boxes[side] = _unique_boxes(
                    [w.ev.right_bbox if side == 'right' else w.ev.left_bbox
                     for w in run if (w.ev.right_bbox if side == 'right'
                                      else w.ev.left_bbox)])
        counts = {side: len(boxes[side]) for side in ('right', 'left')}
        conf_boxes = {side: [list(b) for b in boxes[side]] for side in ('right', 'left')}
        conf_counts = dict(counts)
        lanes = {side: _lane_of(boxes[side], H) for side in ('right', 'left')}
        seen = {side: {tuple(b) for b in boxes[side]} for side in ('right', 'left')}
        last_good = {'right': conf_end, 'left': conf_end}
        last_glyph = {side: _latest_glyph(boxes[side]) for side in ('right', 'left')}
        stopped = {'right': False, 'left': False}
        stop_y: Dict[str, Optional[int]] = {'right': None, 'left': None}
        side_streak = {'right': 0, 'left': 0}
        max_side = int(cfg['stop']['track_end_windows'])
        stops: List[Dict[str, Any]] = []
        pending: List[str] = []
        k = j + 1
        invalid = 0
        seam_out = 0
        gap_thr = max(int(cfg['stop']['blank_gap_min_px']),
                      int(round(float(cfg['stop']['blank_gap_G']) * G)))
        # 反向补齐：确认段之前仍属同两条轨的字形回填到确认框，
        # 避免「凑够独立字形」所需的窗把组框起点推到标题下方。
        back_miss = 0
        for w in reversed(wins[:j]):
            if w.big or w.blank >= gap_thr or w.y0 < conf_start - max_confirm:
                break
            pred = _seam_at(fit, (w.y0 + w.y1) / 2.0)
            # 回填只容得下短暂无效窗；缝位一旦漂移或连续失效就停，避免跨组回填
            if not w.ok or abs(w.ev.seam_x - pred) > float(st['max_resid_H']) * H + 2.0:
                back_miss += 1
                if back_miss > int(cfg['stop']['invalid_windows']):
                    break
                continue
            back_miss = 0
            back = {side: _boxes_in_lane(w, side, lanes[side], pred, H,
                                         rule_rects, None)
                    for side in ('right', 'left')}
            grew = False
            for side in ('right', 'left'):
                fresh = [b for b in back[side] if tuple(b) not in seen[side]]
                if fresh:
                    grew = True
                    counts[side] += len(fresh)
                    for b in fresh:
                        seen[side].add(tuple(b))
                        boxes[side].append(b)
                    last_glyph[side] = _latest_glyph(boxes[side])
            if grew:
                conf_start = w.y0
                if w.ok and abs(w.ev.seam_x - pred) <= float(st['max_resid_H']) * H + 2.0:
                    points.insert(0, {'y': (w.y0 + w.y1) / 2.0,
                                      'x': w.ev.seam_x})
        def _apply_hard_stop(kind: str, hard_y: int, scan_y: int,
                             base_ev: Optional[Dict[str, Any]] = None) -> None:
            """硬断点按整个字形框生效：跨线/越线字形直接拒，不裁字。"""
            rejected: Dict[str, List[List[int]]] = {}
            for side in ('right', 'left'):
                cross = [b for b in boxes[side] if int(b[3]) > hard_y]
                boxes[side] = [b for b in boxes[side] if int(b[3]) <= hard_y]
                counts[side] = len(boxes[side])
                last_glyph[side] = _latest_glyph(boxes[side])
                rejected[side] = cross
                stops.append({
                    'type': kind, 'side': side,
                    'at_y': int(hard_y), 'decided_at_y': int(scan_y),
                    'evidence': {**(base_ev or {}),
                                 'hard_boundary_y': int(hard_y),
                                 'scan_trigger_y': int(scan_y),
                                 'glyph_end_y': (int(last_glyph[side][3])
                                                 if last_glyph[side] else None),
                                 'rejected_crossing': rejected[side]}})

        while k < len(wins):
            w = wins[k]
            if w.big:
                _apply_hard_stop(w.big['kind'], int(w.big['y']), w.y0,
                                 w.big.get('evidence'))
                break
            if w.blank >= gap_thr:
                _apply_hard_stop('blank_gap', w.y0, w.y0,
                                 {'blank_px': w.blank, 'threshold_px': gap_thr})
                break
            pred = _seam_at(fit, (w.y0 + w.y1) / 2.0)
            drift = w.ok and abs(w.ev.seam_x - pred) > float(st['max_resid_H']) * H + 2.0
            accepted: Dict[str, List[List[int]]] = {
                side: _boxes_in_lane(w, side, lanes[side], pred, H,
                                     rule_rects, stop_y[side])
                for side in ('right', 'left')}
            if w.ok and not drift:
                invalid = 0
                seam_out = 0
                points.append({'y': (w.y0 + w.y1) / 2.0, 'x': w.ev.seam_x})
                for side in ('right', 'left'):
                    if stopped[side]:
                        continue
                    fresh = [b for b in accepted[side] if tuple(b) not in seen[side]]
                    if fresh:
                        last_good[side] = w.y1
                        counts[side] += len(fresh)
                        for b in fresh:
                            seen[side].add(tuple(b))
                            boxes[side].append(b)
                        last_glyph[side] = _latest_glyph(boxes[side])
                k += 1
                continue
            invalid += 1
            seam_out += 1
            if w.ev.reason == 'ambig_ink_crossing' and 'ambiguous_ink' not in pending:
                pending.append('ambiguous_ink')
                stops.append({'type': 'ambiguous_ink', 'side': 'pair',
                              'at_y': w.y0, 'decided_at_y': w.y0,
                              'evidence': {'crossing_glyphs': w.ev.crossing_glyphs,
                                           'seam_x': w.ev.seam_x}})
            for side in ('right', 'left'):
                if stopped[side]:
                    continue
                fresh = [b for b in accepted[side] if tuple(b) not in seen[side]]
                if fresh:
                    side_streak[side] = 0
                    last_good[side] = w.y1
                    counts[side] += len(fresh)
                    for b in fresh:
                        seen[side].add(tuple(b))
                        boxes[side].append(b)
                    last_glyph[side] = _latest_glyph(boxes[side])
                else:
                    side_streak[side] += 1
                    if side_streak[side] >= max_side:
                        stopped[side] = True
                        # at_y 用末字原图末端；decided_at_y 仍记录扫描确认位置
                        glyph_end = (int(last_glyph[side][3])
                                     if last_glyph[side] else int(last_good[side]))
                        stop_y[side] = glyph_end
                        stops.append({'type': 'track_ink_end', 'side': side,
                                      'at_y': glyph_end, 'decided_at_y': w.y1,
                                      'evidence': {'reason': w.ev.reason or 'no_glyph',
                                                   'windows_without_glyph':
                                                       side_streak[side],
                                                   'window': [w.y0, w.y1],
                                                   'scan_end_y': int(last_good[side]),
                                                   'glyph_end_y': glyph_end,
                                                   'last_glyph_bbox': last_glyph[side]}})
            if all(stopped.values()):
                break
            if seam_out >= int(cfg['stop']['seam_gap_windows']):
                stops.append({'type': 'seam_missing', 'side': 'pair',
                              'at_y': max(last_good.values()), 'decided_at_y': w.y1,
                              'evidence': {'seam_absent_windows': seam_out,
                                           'reason': w.ev.reason or 'no_seam'}})
                break
            if invalid >= int(cfg['stop']['invalid_windows']) and not any(
                    accepted[s]
                    for s in ('right', 'left') if not stopped[s]):
                stops.append({'type': 'seam_drift' if w.ev.reason == 'seam_drift'
                              else 'seam_missing', 'side': 'pair',
                              'at_y': max(last_good.values()), 'decided_at_y': w.y1,
                              'evidence': {'invalid_windows': invalid,
                                           'reason': w.ev.reason}})
                break
            k += 1
        end_y = max(last_good.values())
        end_idx = k - 1
        if not stops:
            at = min(region_ink.bbox[3], end_y)
            stops.extend(_stop_pair('region_boundary', at, at,
                                    {'region_y': [region_ink.bbox[1],
                                                  region_ink.bbox[3]]}))
        undone = None
        for side in ('right', 'left'):
            if conf_counts[side] < min_g:
                undone = f'confirmed_span_missing_glyphs_{side}'
                break
        span_g = [g for g in glyphs
                  if g['bbox'][3] > conf_start and g['bbox'][1] < end_y]
        if span_g:
            med_h = float(np.median([g['h'] for g in span_g]))
            if med_h > float(cfg['glyph']['small_max_H']) * H:
                undone = undone or 'not_small_text'
        rb = _union_box([b for b in boxes['right'] if b[1] < end_y])
        lb = _union_box([b for b in boxes['left'] if b[1] < end_y])
        if rb is None or lb is None:
            undone = undone or 'track_bbox_undefined'
        elif ((rb[2] - rb[0]) > float(cfg['track']['span_width_max_H']) * H
              or (lb[2] - lb[0]) > float(cfg['track']['span_width_max_H']) * H):
            undone = undone or 'track_bbox_too_wide'
        # 延伸段把框撑坏时退回确认段本身：宁可短，也不能把同一窗里后续的
        # 另一条真缝吞掉（v1 的 p12 右半页全部因此丢失）。
        if undone:
            cr = _union_box(conf_boxes['right'])
            cl = _union_box(conf_boxes['left'])
            if (cr and cl
                    and min(conf_counts.values()) >= int(cfg['glyph']['min_per_side_group'])
                    and max(cr[2] - cr[0], cl[2] - cl[0])
                    <= float(cfg['track']['max_width_H']) * H):
                rb, lb, undone = cr, cl, None
                end_y = conf_end
                # 退回确认段时同步裁剪字形与端点证据，保证轨框/停轨坐标一致
                boxes = {side: [list(b) for b in conf_boxes[side]]
                         for side in ('right', 'left')}
                counts = {side: len(boxes[side]) for side in ('right', 'left')}
                # 延伸段的停轨事件也已失效，避免留下与最终轨框不一致的坐标
                stops = [e for e in stops if int(e['at_y']) <= conf_end]
                if not stops:
                    stops.extend(_stop_pair('region_boundary', conf_end, conf_end,
                                            {'fallback_to': 'confirmed_span'}))
        groups.append(_make_group(page_id, region, conf_start, conf_end, rb, lb,
                                  points, fit, counts, boxes, stops, pending, undone,
                                  H, G, pitch, seam_x, glyphs, conf_counts))
        runs.append({'seam': seam_x, 'span': [conf_start, conf_end], 'windows': len(run),
                     'confirmed': undone is None, 'rejected': undone})
        # 撤销的段不能吞掉后面的窗口：从确认段之后重新扫，后一条缝才有机会
        i = (j + 1) if undone else max(end_idx + 1, j + 1)
    return groups, runs


def _stop_pair(kind: str, at_y: int, decided_at: int,
               evidence: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return [{'type': kind, 'side': side, 'at_y': int(at_y),
             'decided_at_y': int(decided_at),
             'evidence': dict(evidence or {})} for side in ('right', 'left')]


def _make_group(page_id: str, region: Dict[str, Any], conf_start: int, conf_end: int,
                rb: Optional[List[int]], lb: Optional[List[int]],
                points: Sequence[Dict[str, Any]], fit: Dict[str, Any],
                counts: Dict[str, int], boxes: Dict[str, List[List[int]]],
                stops: Sequence[Dict[str, Any]], pending: Sequence[str],
                undone: Optional[str], H: float, G: float, pitch: float,
                seam_x: int, glyphs: Sequence[Dict[str, Any]] = (),
                conf_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    group_bbox = ([min(lb[0], rb[0]), min(lb[1], rb[1]),
                   max(lb[2], rb[2]), max(lb[3], rb[3])] if rb and lb else None)
    end_y = max((lb[3] if lb else 0), (rb[3] if rb else 0), conf_end)
    span_glyphs = [g for g in glyphs
                   if g['bbox'][3] > conf_start and g['bbox'][1] < end_y]
    return {
        'group_id': f"{page_id}#{region['region_id'].split('#')[-1]}#{conf_start:04d}",
        'page_id': page_id,
        'region_id': region['region_id'],
        'region_source': region.get('column_range_source'),
        'low_confidence': bool(region.get('low_confidence')),
        'confirmed_span': [int(conf_start), int(conf_end)],
        'group_bbox': group_bbox,
        'right_bbox': rb, 'left_bbox': lb,
        'state': 'completed' if not undone else 'rejected',
        'seam': {'x_start': points[0]['x'] if points else seam_x,
                 'x_end': points[-1]['x'] if points else seam_x,
                 'slope': round(fit['slope'], 4),
                 'resid_std': round(fit['resid_std'], 3),
                 'max_resid': round(fit['max_resid'], 3),
                 'points': [{'y': round(p['y'], 1), 'x': int(p['x'])} for p in points]},
        'glyph_support': {
            'right': {**_side_scale([g for g in span_glyphs
                                     if g['cx'] >= _seam_at(fit, g['cy'])]),
                      'n_span': counts['right'],
                      'n_unique_confirm': (conf_counts or {}).get('right', 0),
                      'last_glyph_bbox': _latest_glyph(boxes['right']),
                      'last_glyph_end': (_latest_glyph(boxes['right']) or [0, 0, 0, 0])[3]},
            'left': {**_side_scale([g for g in span_glyphs
                                    if g['cx'] < _seam_at(fit, g['cy'])]),
                     'n_span': counts['left'],
                     'n_unique_confirm': (conf_counts or {}).get('left', 0),
                     'last_glyph_bbox': _latest_glyph(boxes['left']),
                     'last_glyph_end': (_latest_glyph(boxes['left']) or [0, 0, 0, 0])[3]}},
        'small_scale': _small_scale(span_glyphs, fit, H),
        'stop_events': list(stops),
        'pending': bool(pending), 'pending_reasons': list(pending),
        'scale_used': {'H': round(H, 2), 'G': round(G, 2),
                       'glyph_pitch': round(pitch, 2)},
        'undone': undone,
        'internal_seam': int(seam_x),
    }


def _small_scale(glyphs: Sequence[Dict[str, Any]], fit: Dict[str, Any],
                 H: float) -> Dict[str, Any]:
    """确认后的局部小字尺寸（来源与样本都记）。"""
    sides = {'right': [], 'left': []}
    for g in glyphs:
        sides['right' if g['cx'] >= _seam_at(fit, g['cy']) else 'left'].append(g)
    merged = sorted(glyphs, key=lambda c: c['bbox'][1])
    hs = [c['h'] for c in merged]
    ws = [c['w'] for c in merged]
    gaps, pitches = [], []
    for side in ('right', 'left'):
        comps = sorted(sides[side], key=lambda c: c['bbox'][1])
        for a, b in zip(comps, comps[1:]):
            g = b['bbox'][1] - a['bbox'][3]
            if g >= 0:
                gaps.append(g)
                pitches.append((b['bbox'][1] + b['bbox'][3]) / 2.0
                               - (a['bbox'][1] + a['bbox'][3]) / 2.0)
    return {'small_glyph_height': _median(hs), 'small_glyph_width': _median(ws),
            'small_glyph_gap': _median(gaps), 'small_glyph_pitch': _median(pitches),
            'sample_count': {'height': len(hs), 'gap': len(gaps), 'pitch': len(pitches)},
            'source': 'stage5_local_components' if merged else 'none',
            'parent_H': round(H, 2)}
