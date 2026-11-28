# -*- coding: utf-8 -*-
"""重复组去重：重叠回退窗口会看到同一组两次，只能输出一次。

判据（全部来自原图坐标，不依赖窗口编号）：
  - 组框 IoU > 0.5；或
  - 横向重叠 > 0.6（较短框宽为分母）且纵向重叠 > 0.5（较短框高为分母）。

**同栏上下两组**的纵向重叠 ≤ 0.5，不会被并掉——这正是 p12「雲母／陽起石」
那种上下连续两组要保留的场景。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def _inter(a: Sequence[int], b: Sequence[int], axis: int) -> int:
    i0 = 0 if axis == 0 else 1
    i1 = 2 if axis == 0 else 3
    return max(0, min(a[i1], b[i1]) - max(a[i0], b[i0]))


def bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    iw, ih = _inter(a, b, 0), _inter(a, b, 1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def is_duplicate(a: Dict[str, Any], b: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    if not a.get('group_bbox') or not b.get('group_bbox'):
        return False, ''
    ba, bb = a['group_bbox'], b['group_bbox']
    iou = bbox_iou(ba, bb)
    if iou > float(cfg['iou']):
        return True, f'iou={iou:.3f}'
    xw = min(ba[2] - ba[0], bb[2] - bb[0])
    yh = min(ba[3] - ba[1], bb[3] - bb[1])
    xr = _inter(ba, bb, 0) / xw if xw > 0 else 0.0
    yr = _inter(ba, bb, 1) / yh if yh > 0 else 0.0
    ha, hb = ba[3] - ba[1], bb[3] - bb[1]
    ratio = max(ha, hb) / float(min(ha, hb)) if min(ha, hb) > 0 else 99.0
    if (xr > float(cfg['x_overlap']) and yr > float(cfg['y_overlap'])
            and ratio <= float(cfg.get('height_ratio_max', 1.8))):
        return True, f'x_overlap={xr:.3f},y_overlap={yr:.3f},height_ratio={ratio:.2f}'
    return False, ''


def _support(g: Dict[str, Any]) -> int:
    gs = g.get('glyph_support') or {}
    return int((gs.get('right') or {}).get('n') or 0) + int((gs.get('left') or {}).get('n') or 0)


def dedupe_groups(groups: Sequence[Dict[str, Any]], cfg: Dict[str, Any]
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按来源优先级 + 字形支持数排序后贪心保留；被并掉的组写进日志。"""
    order = {s: i for i, s in enumerate(cfg['source_priority'])}
    ranked = sorted(groups, key=lambda g: (order.get(g.get('region_source'), 9),
                                           -_support(g),
                                           g['group_bbox'][1] if g.get('group_bbox') else 0,
                                           g['group_bbox'][0] if g.get('group_bbox') else 0))
    kept: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []
    for g in ranked:
        hit = None
        why = ''
        for k in kept:
            dup, reason = is_duplicate(g, k, cfg)
            if dup:
                hit, why = k, reason
                break
        if hit is None:
            kept.append(g)
            continue
        # 被并掉的组：把证据挂到保留组上，便于复查
        hit.setdefault('merged_candidates', []).append({
            'group_id': g['group_id'], 'region_id': g.get('region_id'),
            'region_source': g.get('region_source'),
            'group_bbox': g.get('group_bbox'), 'reason': why,
            'glyph_support': {s: (g.get('glyph_support') or {}).get(s, {}).get('n')
                              for s in ('right', 'left')}})
        log.append({'kept': hit['group_id'], 'dropped': g['group_id'],
                    'kept_source': hit.get('region_source'),
                    'dropped_source': g.get('region_source'),
                    'overlap_reason': why,
                    'kept_bbox': hit.get('group_bbox'),
                    'dropped_bbox': g.get('group_bbox')})
    kept.sort(key=lambda g: (g['group_bbox'][1], g['group_bbox'][0]))
    return kept, log
