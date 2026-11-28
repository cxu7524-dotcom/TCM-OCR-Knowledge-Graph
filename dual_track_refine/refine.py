# -*- coding: utf-8 -*-
"""双轨子框精修：分侧回溯字形、收紧横边、独立补齐首尾。

所有坐标都是原图半开区间 [x1,y1,x2,y2)。stage5 的组 ID、成对关系、
确认区段和断点事件只读不改；stage4 的区域/尺度/栏线证据原样复用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dual_track_track import DEFAULT_CONFIG as S5_CFG
from dual_track_track.projection import build_region_ink, glyph_components

SIDES = ('right', 'left')


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _median(vals: Sequence[float]) -> Optional[float]:
    return float(np.median(vals)) if len(vals) else None


def _bbox_key(b: Sequence[int]) -> Tuple[int, int, int, int]:
    return tuple(int(v) for v in b)  # type: ignore[return-value]


def _seam_at(points: Sequence[Dict[str, Any]], y: float, fallback: float) -> float:
    if not points:
        return float(fallback)
    pts = sorted(points, key=lambda p: float(p['y']))
    if y <= pts[0]['y']:
        return float(pts[0]['x'])
    if y >= pts[-1]['y']:
        return float(pts[-1]['x'])
    for a, b in zip(pts, pts[1:]):
        if float(a['y']) <= y <= float(b['y']):
            t = (y - float(a['y'])) / max(1e-9, float(b['y']) - float(a['y']))
            return float(a['x']) + t * (float(b['x']) - float(a['x']))
    return float(fallback)


def _side_ok(g: Dict[str, Any], side: str, seam_pts, seam_fallback: float,
             margin: int) -> bool:
    sx = _seam_at(seam_pts, float(g['cy']), seam_fallback)
    b = g['bbox']
    return (b[0] >= sx - margin) if side == 'right' else (b[2] <= sx + margin)


def _stop_info(group: Dict[str, Any]) -> Tuple[Dict[str, List[int]], List[int]]:
    per_side: Dict[str, List[int]] = {s: [] for s in SIDES}
    hard: List[int] = []
    for e in group.get('stop_events') or []:
        ev = e.get('evidence') or {}
        hy = ev.get('hard_boundary_y')
        if hy is not None:
            hard.append(int(hy))
        at = int(e.get('at_y', 0))
        if e.get('side') in SIDES:
            per_side[e['side']].append(at)
        elif e.get('side') == 'pair':
            for s in SIDES:
                per_side[s].append(at)
    return per_side, hard


def _union(boxes: Sequence[Sequence[int]]) -> Optional[List[int]]:
    if not boxes:
        return None
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]


def _track_glyphs(all_glyphs: Sequence[Dict[str, Any]], side: str,
                  track: Sequence[int], cfg: Dict[str, Any]
                  ) -> List[Dict[str, Any]]:
    out = []
    thr = float(cfg['glyph']['overlap_with_track'])
    for g in all_glyphs:
        b = g['bbox']
        ox = _overlap(b[0], b[2], track[0], track[2]) / float(max(1, b[2] - b[0]))
        oy = _overlap(b[1], b[3], track[1], track[3]) / float(max(1, b[3] - b[1]))
        if (ox >= thr and oy >= thr
                and b[0] >= track[0] - 2 and b[2] <= track[2] + 2):
            out.append(g)
    return out


def _screen_glyphs(glyphs: Sequence[Dict[str, Any]], side: str, region_box,
                   rule_rects, seam_pts, seam_fallback: float,
                   med_h: float, med_w: float, cfg: Dict[str, Any],
                   pitch: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (可用于定边/补笔的字形, 被排除记录)。"""
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen = set()
    gcfg = cfg['glyph']
    near = float(gcfg['rule_near_px'])
    margin = int(cfg['edge']['seam_margin_px'])
    for g in glyphs:
        b = g['bbox']
        key = _bbox_key(b)
        if key in seen:
            excluded.append({'bbox': list(b), 'reason': 'duplicate'})
            continue
        seen.add(key)
        if not _side_ok(g, side, seam_pts, seam_fallback, margin):
            excluded.append({'bbox': list(b), 'reason': 'crosses_seam'})
            continue
        if b[0] <= region_box[0] + int(gcfg['boundary_touch_px']) or \
                b[2] >= region_box[2] - int(gcfg['boundary_touch_px']):
            excluded.append({'bbox': list(b), 'reason': 'boundary_clipped'})
            continue
        touching = [list(s) for s in rule_rects
                    if s[0] < b[2] and s[1] > b[0]
                    and s[2] < b[3] and s[3] > b[1]]
        if touching:
            excluded.append({'bbox': list(b), 'reason': 'ambiguous_rule_attached',
                             'rule': touching[0]})
            continue
        nearby = [list(s) for s in rule_rects
                  if s[0] < b[2] + near and s[1] > b[0] - near
                  and s[2] < b[3] and s[3] > b[1]]
        if nearby:
            # 邻域只触发复核：整框与栏线分离且已通过本轨侧别/尺寸判据，
            # 保留字形与首尾，把复核依据挂到字形上。
            g['_rule_proximity'] = nearby
        if b[2] - b[0] < float(gcfg['min_width_H']) * med_w:
            excluded.append({'bbox': list(b), 'reason': 'size_mismatch'})
            continue
        slender = float(g['h']) / float(max(1, g['w']))
        neighbors = [o for o in glyphs if o is not g
                     and abs(o['cy'] - g['cy']) <= max(pitch * 1.5, med_h * 2.0)
                     and _overlap(o['bbox'][0], o['bbox'][2], b[0], b[2]) > 0]
        if slender > float(gcfg['slender_ratio']) and not neighbors:
            excluded.append({'bbox': list(b), 'reason': 'ambiguous_long_stroke'})
            continue
        kept.append(g)
    return kept, excluded


def _horizontal_edges(kept: Sequence[Dict[str, Any]], side: str,
                      original: Sequence[int], region_box: Sequence[int],
                      rule_rects, seam_pts, seam_fallback: float,
                      H: float, cfg: Dict[str, Any]
                      ) -> Tuple[Optional[List[int]], List[Dict[str, Any]], bool]:
    boxes = [g['bbox'] for g in kept]
    if not boxes:
        return None, [], True
    xlo, xhi = min(b[0] for b in boxes), max(b[2] for b in boxes)
    notes: List[Dict[str, Any]] = []
    margin = int(cfg['edge']['seam_margin_px'])
    rmargin = int(cfg['edge']['rule_margin_px'])
    if side == 'left':
        limit = min(_seam_at(seam_pts, g['cy'], seam_fallback) for g in kept) - margin
        if xhi > limit:
            notes.append({'kind': 'seam', 'from': xhi, 'to': int(limit)})
            xhi = int(limit)
    else:
        limit = max(_seam_at(seam_pts, g['cy'], seam_fallback) for g in kept) + margin
        if xlo < limit:
            notes.append({'kind': 'seam', 'from': xlo, 'to': int(limit)})
            xlo = int(limit)
    for sx0, sx1, sy0, sy1 in rule_rects:
        if sy1 <= min(b[1] for b in boxes) or sy0 >= max(b[3] for b in boxes):
            continue
        if side == 'left' and sx0 >= xlo:
            if sx0 - rmargin < xhi:
                notes.append({'kind': 'rule', 'from': xhi, 'to': int(sx0 - rmargin),
                              'rule': [sx0, sx1, sy0, sy1]})
                xhi = int(sx0 - rmargin)
        if side == 'right' and sx1 <= xhi:
            if sx1 + rmargin > xlo:
                notes.append({'kind': 'rule', 'from': xlo, 'to': int(sx1 + rmargin),
                              'rule': [sx0, sx1, sy0, sy1]})
                xlo = int(sx1 + rmargin)
    xlo = max(int(region_box[0]) + int(cfg['edge']['region_margin_px']), int(xlo))
    xhi = min(int(region_box[2]) - int(cfg['edge']['region_margin_px']), int(xhi))
    low = False
    if xhi - xlo < float(cfg['edge']['min_width_H']) * H:
        low = True
        notes.append({'kind': 'too_narrow', 'width': int(xhi - xlo),
                      'min_px': round(float(cfg['edge']['min_width_H']) * H, 1)})
    if any(b[0] < xlo or b[2] > xhi for b in boxes):
        low = True
        notes.append({'kind': 'would_cut_confirmed_glyph'})
    if low:
        return None, notes, True
    return [int(xlo), int(xhi)], notes, False


def _whole_rect_ok(boxes: Sequence[Sequence[int]], side: str, region_box,
                   rule_rects, seam_pts, seam_fallback: float,
                   hard: Sequence[int], opposite_bbox: Optional[Sequence[int]],
                   cfg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """补笔后复验最终外接矩形：缝、栏线、硬断点、区域与邻轨全部一致。"""
    u = _union(boxes)
    if not u:
        return False, 'empty_rect'
    if u[0] < region_box[0] or u[2] > region_box[2]:
        return False, 'outside_region'
    ys = [u[1], max(u[1], u[3] - 1)]
    ys.extend(b[1] for b in boxes)
    ys.extend(max(b[1], b[3] - 1) for b in boxes)
    # 半开区间：纳入落在该轨框内的中缝折线端点与所有内部节点，
    # 包括字间空白处的转折；排除 y2 本身（不属于框内）。
    ys.extend(float(p['y']) for p in seam_pts
              if float(u[1]) <= float(p['y']) < float(u[3]))
    seams = [_seam_at(seam_pts, float(y), seam_fallback) for y in ys]
    margin = int(cfg['edge']['seam_margin_px'])
    if side == 'right' and u[0] < max(seams) + margin:
        return False, 'crosses_seam_in_final_rect'
    if side == 'left' and u[2] > min(seams) - margin:
        return False, 'crosses_seam_in_final_rect'
    for sx0, sx1, sy0, sy1 in rule_rects:
        if sx0 < u[2] and sx1 > u[0] and sy0 < u[3] and sy1 > u[1]:
            return False, 'crosses_rule_in_final_rect'
    if any(int(u[1]) < int(y) < int(u[3]) for y in hard):
        return False, 'crosses_hard_boundary_in_final_rect'
    if opposite_bbox:
        if side == 'right' and u[0] < int(opposite_bbox[2]) - 1:
            return False, 'overlaps_opposite_track'
        if side == 'left' and u[2] > int(opposite_bbox[0]) + 1:
            return False, 'overlaps_opposite_track'
    return True, None


def _search_supplement(anchor: Dict[str, Any], direction: str, side: str,
                       pool: Sequence[Dict[str, Any]], used: set,
                       current_boxes: Sequence[Sequence[int]],
                       opposite_bbox: Optional[Sequence[int]],
                       edges: Sequence[int], region_box, rule_rects,
                       seam_pts, seam_fallback: float,
                       stops: Sequence[int], hard: Sequence[int],
                       big_boxes: Sequence[Sequence[int]],
                       neighbor_head_y: int, neighbor_tail_y: int,
                       med_h: float, med_w: float, pitch_up: float,
                       cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    sc = cfg['supplement']
    lateral = float(sc['lateral_margin_H']) * med_h
    lo, hi = int(edges[0]) - int(round(lateral)), int(edges[1]) + int(round(lateral))
    accepted: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    qualified = 0
    hard_limit = min([y for y in hard if y > anchor['bbox'][1]] or [region_box[3]])
    hard_limit = min(hard_limit, neighbor_tail_y)
    stop_limit = min([y for y in stops if y > anchor['bbox'][1]] or [region_box[3]])
    for g in pool:
        b = g['bbox']
        if _bbox_key(b) in used:
            continue
        if direction == 'head':
            gap = int(anchor['bbox'][1]) - int(b[3])
        else:
            gap = int(b[1]) - int(anchor['bbox'][3])
        if gap < 0 or gap > pitch_up:
            excluded.append({'bbox': list(b), 'reason': 'gap_too_large', 'gap': gap})
            continue
        if (b[0] <= region_box[0] + int(cfg['glyph']['boundary_touch_px'])
                or b[2] >= region_box[2] - int(cfg['glyph']['boundary_touch_px'])):
            excluded.append({'bbox': list(b), 'reason': 'boundary_clipped'})
            continue
        if b[0] < lo or b[2] > hi:
            excluded.append({'bbox': list(b), 'reason': 'outside_track_lane'})
            continue
        if float(g['h']) / float(max(1, g['w'])) > \
                float(cfg['glyph']['slender_ratio']):
            excluded.append({'bbox': list(b), 'reason': 'ambiguous_long_stroke'})
            continue
        bm = int(round(2 * med_h))
        if any(bx0 - bm < b[2] and bx1 + bm > b[0]
               and by0 - bm < b[3] and by1 + bm > b[1]
               for bx0, by0, bx1, by1 in big_boxes):
            excluded.append({'bbox': list(b), 'reason': 'above_hard_boundary'})
            continue
        if not _side_ok(g, side,
                        seam_pts, seam_fallback,
                        int(cfg['edge']['seam_margin_px'])):
            excluded.append({'bbox': list(b), 'reason': 'crosses_seam'})
            continue
        if any(sx0 < b[2] and sx1 > b[0] and sy0 < b[3] and sy1 > b[1]
               for sx0, sx1, sy0, sy1 in rule_rects):
            excluded.append({'bbox': list(b), 'reason': 'crosses_rule'})
            continue
        if direction == 'tail' and (b[1] >= stop_limit or b[3] > hard_limit):
            excluded.append({'bbox': list(b), 'reason': 'crosses_stop',
                             'stop_limit': stop_limit, 'hard_limit': hard_limit})
            continue
        if direction == 'head' and (b[3] > hard_limit or b[1] < neighbor_head_y):
            excluded.append({'bbox': list(b), 'reason': 'above_hard_boundary',
                             'hard_limit': hard_limit,
                             'neighbor_head_y': neighbor_head_y})
            continue
        if not (float(sc['size_h_min']) * med_h <= g['h']
                <= float(sc['size_h_max']) * med_h
                and float(sc['size_w_min']) * med_w <= g['w']
                <= float(sc['size_w_max']) * med_w):
            excluded.append({'bbox': list(b), 'reason': 'size_mismatch'})
            continue
        ok, why = _whole_rect_ok(list(current_boxes) + [b], side, region_box,
                                 rule_rects, seam_pts, seam_fallback, hard,
                                 opposite_bbox, cfg)
        if not ok:
            excluded.append({'bbox': list(b), 'reason': why})
            continue
        qualified += 1
        if len(accepted) < int(sc['max_per_side_end']):
            accepted.append(g)
            current_boxes.append(b)
            used.add(_bbox_key(b))
    return accepted, excluded, qualified > len(accepted)


def _opposite(group: Dict[str, Any], side: str) -> Optional[List[int]]:
    other = 'left' if side == 'right' else 'right'
    b = group.get(f'{other}_bbox')
    return [int(v) for v in b] if b else None


def refine_side(group: Dict[str, Any], side: str, region_box, all_glyphs,
                rule_rects, big_boxes: Sequence[Sequence[int]],
                neighbor_head_y: int, neighbor_tail_y: int,
                cfg: Dict[str, Any]) -> Dict[str, Any]:
    original = [int(v) for v in group[f'{side}_bbox']]
    H = float(group['scale_used']['H'])
    G = float(group['scale_used'].get('G') or 0.25 * H)
    pitch = float(group['scale_used'].get('glyph_pitch') or H)
    small = group.get('small_scale') or {}
    seam_pts = (group.get('seam') or {}).get('points') or []
    seam_fallback = int(group.get('internal_seam') or (original[0] + original[2]) // 2)
    stops, hard = _stop_info(group)

    track_glyphs = _track_glyphs(all_glyphs, side, original, cfg)
    med_h = float(small.get('small_glyph_height') or
                  _median([g['h'] for g in track_glyphs]) or 0.6 * H)
    med_w = float(small.get('small_glyph_width') or
                  _median([g['w'] for g in track_glyphs]) or 0.6 * H)
    med_pitch = float(small.get('small_glyph_pitch') or pitch)
    pitch_up = min(float(cfg['supplement']['pitch_max_H']) * H,
                   max(float(cfg['supplement']['pitch_min_H']) * H,
                       float(cfg['supplement']['pitch_multiple']) * max(med_pitch, G * 2)))
    kept, excluded = _screen_glyphs(track_glyphs, side, region_box, rule_rects,
                                    seam_pts, seam_fallback, med_h, med_w, cfg,
                                    max(med_pitch, pitch))
    for g in kept:
        g['_side'] = side
    edges, edge_notes, low = _horizontal_edges(kept, side, original, region_box,
                                               rule_rects, seam_pts, seam_fallback,
                                               H, cfg)
    flags: List[str] = []
    pending: List[str] = []
    if low:
        flags.append('horizontal_low_confidence')
        pending.append('horizontal_low_confidence')
    for x in excluded:
        if x['reason'] in ('boundary_clipped', 'ambiguous_rule_attached',
                           'ambiguous_long_stroke', 'crosses_seam'):
            pending.append(f"{x['reason']}@{x['bbox'][1]}")
    base_ok, base_why = _whole_rect_ok(
        [g['bbox'] for g in kept], side, region_box, rule_rects, seam_pts,
        seam_fallback, hard, _opposite(group, side), cfg)
    if not base_ok:
        low = True
        flags.append('horizontal_low_confidence')
        pending.append(f'existing_boundary_conflict:{base_why}')
        edge_notes.append({'kind': 'existing_boundary_conflict', 'reason': base_why})

    if edges is None or not base_ok:
        edges = [original[0], original[2]]
        used = {_bbox_key(g['bbox']) for g in kept}
        fb = {'ok': False,
              'reason': base_why if not base_ok else 'no_confirmed_glyphs',
              'rolled_back': [], 'fallback_to_tentative': True,
              'status': 'rejected_to_tentative'}
        return _pack_side(original, original, edges, [], [], kept, excluded,
                          flags, pending, edge_notes, med_h, med_w, pitch_up,
                          low, False, cfg, final_check=fb)

    used = {_bbox_key(g['bbox']) for g in kept}
    by_y = sorted(kept, key=lambda g: g['bbox'][1])
    first, last = by_y[0], by_y[-1]
    pool = [g for g in all_glyphs if g not in track_glyphs]
    head, head_ex, head_limited = _search_supplement(
        first, 'head', side, pool, used, [x['bbox'] for x in kept],
        _opposite(group, side), edges, region_box, rule_rects, seam_pts,
        seam_fallback, stops[side], hard, big_boxes, neighbor_head_y,
        neighbor_tail_y, med_h, med_w, pitch_up, cfg)
    tail, tail_ex, tail_limited = _search_supplement(
        last, 'tail', side, pool, used,
        [x['bbox'] for x in kept] + [g['bbox'] for g in head],
        _opposite(group, side), edges, region_box, rule_rects, seam_pts,
        seam_fallback, stops[side], hard, big_boxes, neighbor_head_y,
        neighbor_tail_y, med_h, med_w, pitch_up, cfg)
    supplements = head + tail
    # 输出前最终核验：把首尾已接纳补笔放进同一个轨框，冲突则回退最后接纳者
    final_ok, final_why = _whole_rect_ok(
        [x['bbox'] for x in kept] + [g['bbox'] for g in supplements],
        side, region_box, rule_rects, seam_pts, seam_fallback, hard,
        _opposite(group, side), cfg)
    rolled_back: List[Dict[str, Any]] = []
    while supplements and not final_ok:
        gone = supplements.pop()
        rolled_back.append({'bbox': list(gone['bbox']), 'reason': final_why})
        final_ok, final_why = _whole_rect_ok(
            [x['bbox'] for x in kept] + [g['bbox'] for g in supplements],
            side, region_box, rule_rects, seam_pts, seam_fallback, hard,
            _opposite(group, side), cfg)
    if not final_ok:
        low = True
        flags.append('horizontal_low_confidence')
        pending.append(f'existing_boundary_conflict:{final_why}')
        edge_notes.append({'kind': 'existing_boundary_conflict', 'reason': final_why})
        # 回退所有补笔并回到暂定框，状态必须如实写为失败
        head, tail = [], []
        edges = [original[0], original[2]]
        final_check = {'ok': False, 'reason': final_why,
                       'rolled_back': rolled_back,
                       'fallback_to_tentative': True,
                       'status': 'rejected_to_tentative'}
    else:
        head = [g for g in head if g in supplements]
        tail = [g for g in tail if g in supplements]
        final_check = {'ok': True, 'reason': None,
                       'rolled_back': rolled_back,
                       'fallback_to_tentative': False,
                       'status': 'accepted'}
        edges = [min([x['bbox'][0] for x in kept]
                     + [g['bbox'][0] for g in head + tail]),
                 max([x['bbox'][2] for x in kept]
                     + [g['bbox'][2] for g in head + tail])]
    limited = head_limited or tail_limited or bool(rolled_back)
    if limited:
        flags.append('supplement_limited')
    accepted = sorted(kept + head + tail, key=lambda g: g['bbox'][1])
    y0 = min(g['bbox'][1] for g in accepted)
    y1 = max(g['bbox'][3] for g in accepted)
    refined = (list(original) if final_check.get('fallback_to_tentative')
               else [int(edges[0]), int(y0), int(edges[1]), int(y1)])
    return _pack_side(original, refined, edges, head, tail, kept,
                      excluded + head_ex + tail_ex, flags, pending, edge_notes,
                      med_h, med_w, pitch_up, low, limited, cfg,
                      head_limited=head_limited, tail_limited=tail_limited,
                      final_check=final_check)


def _pack_side(original: List[int], refined: List[int], edges: Sequence[int],
               head: Sequence[Dict[str, Any]], tail: Sequence[Dict[str, Any]],
               kept: Sequence[Dict[str, Any]], excluded: Sequence[Dict[str, Any]],
               flags: Sequence[str], pending: Sequence[str],
               edge_notes: Sequence[Dict[str, Any]], med_h: float, med_w: float,
               pitch_up: float, low: bool, limited: bool, cfg: Dict[str, Any],
               head_limited: bool = False, tail_limited: bool = False,
               final_check: Optional[Dict[str, Any]] = None
               ) -> Dict[str, Any]:
    box = [int(v) for v in refined]
    return {
        'bbox_before': list(original), 'bbox_after': box,
        'edge': {'x1': int(edges[0]), 'x2': int(edges[1]),
                 'source_glyphs': [list(g['bbox']) for g in kept],
                 'notes': list(edge_notes)},
        'supplement': {
            'head': {'accepted': [list(g['bbox']) for g in head],
                     'excluded': [x for x in excluded
                                  if x.get('reason') in ('gap_too_large', 'above_hard_boundary',
                                                         'boundary_clipped',
                                                         'ambiguous_long_stroke',
                                                         'crosses_seam_in_final_rect',
                                                         'crosses_rule_in_final_rect',
                                                         'crosses_hard_boundary_in_final_rect',
                                                         'overlaps_opposite_track')],
                     'limited': bool(head_limited)},
            'tail': {'accepted': [list(g['bbox']) for g in tail],
                     'excluded': [x for x in excluded
                                  if x.get('reason') in ('crosses_stop', 'gap_too_large',
                                                         'crosses_rule',
                                                         'boundary_clipped',
                                                         'ambiguous_long_stroke',
                                                         'crosses_seam_in_final_rect',
                                                         'crosses_rule_in_final_rect',
                                                         'crosses_hard_boundary_in_final_rect',
                                                         'overlaps_opposite_track')],
                     'limited': bool(tail_limited)},
            'max_per_end': int(cfg['supplement']['max_per_side_end']),
            'pitch_upper_px': round(float(pitch_up), 2),
        },
        'local_scale': {'glyph_height': round(float(med_h), 2),
                        'glyph_width': round(float(med_w), 2)},
        'accepted_glyphs': [list(g['bbox']) for g in list(kept) + list(head) + list(tail)],
        'final_validation': final_check or {'ok': True, 'reason': None,
                                            'rolled_back': []},
        'rule_proximity_review': [
            {'bbox': list(g['bbox']), 'rule': g['_rule_proximity']}
            for g in kept if g.get('_rule_proximity')],
        'excluded_glyphs': list(excluded),
        'flags': sorted(set(flags)), 'pending_reasons': sorted(set(pending)),
        'horizontal_low_confidence': bool(low), 'supplement_limited': bool(limited),
    }


def refine_group(group: Dict[str, Any], region_box: Sequence[int], ink: np.ndarray,
                 rule_segments: Sequence[Sequence[int]],
                 cfg: Dict[str, Any],
                 neighbor_bounds: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    H = float(group['scale_used']['H'])
    ri = build_region_ink(ink, region_box, rule_segments)
    all_glyphs = glyph_components(ri, S5_CFG, H)
    big_boxes: List[List[int]] = []
    if ri.stats is not None:
        for i in range(1, ri.stats.shape[0]):
            x, y, w, h, area = [int(v) for v in ri.stats[i]]
            if area < int(S5_CFG['glyph']['min_area_px']):
                continue
            if h >= float(H) and w >= 0.6 * float(H):
                big_boxes.append([x + region_box[0], y + region_box[1],
                                  x + region_box[0] + w, y + region_box[1] + h])
    nb = neighbor_bounds or {}
    neighbor_head_y = int(nb.get('head_y', region_box[1]))
    neighbor_tail_y = int(nb.get('tail_y', region_box[3]))
    rule_rects = [list(s[:4]) for s in rule_segments or []]
    sides = {s: refine_side(group, s, region_box, all_glyphs, rule_rects,
                            big_boxes, neighbor_head_y, neighbor_tail_y, cfg)
             for s in SIDES if group.get(f'{s}_bbox')}
    before_union = [min(g['bbox_before'][0] for g in sides.values()),
                    min(g['bbox_before'][1] for g in sides.values()),
                    max(g['bbox_before'][2] for g in sides.values()),
                    max(g['bbox_before'][3] for g in sides.values())] if sides else None
    after_union = [min(g['bbox_after'][0] for g in sides.values()),
                   min(g['bbox_after'][1] for g in sides.values()),
                   max(g['bbox_after'][2] for g in sides.values()),
                   max(g['bbox_after'][3] for g in sides.values())] if sides else None
    flags = sorted({f for g in sides.values() for f in g['flags']})
    pending = sorted({r for g in sides.values() for r in g['pending_reasons']})
    return {
        'group_id': group['group_id'], 'page_id': group['page_id'],
        'region_id': group['region_id'],
        'region_source': group.get('region_source'),
        'low_confidence': bool(group.get('low_confidence')),
        'stage5_pending': bool(group.get('pending')),
        'before': {'right_bbox': sides.get('right', {}).get('bbox_before'),
                   'left_bbox': sides.get('left', {}).get('bbox_before'),
                   'group_bbox': before_union},
        'after': {'right_bbox': sides.get('right', {}).get('bbox_after'),
                  'left_bbox': sides.get('left', {}).get('bbox_after'),
                  'group_bbox': after_union},
        'sides': sides,
        'flags': flags, 'pending_reasons': pending,
        'horizontal_low_confidence': any(
            s['horizontal_low_confidence'] for s in sides.values()),
        'supplement_limited': any(s['supplement_limited'] for s in sides.values()),
        'pending': bool(pending),
        'changed': bool(before_union != after_union),
    }
