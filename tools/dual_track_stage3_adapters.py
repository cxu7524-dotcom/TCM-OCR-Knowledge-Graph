# -*- coding: utf-8 -*-
"""旧算法入口适配：把各入口的原始输出转成统一预测格式。

统一字段（协议 prediction 契约 + 本阶段补充）：
  page_id, prediction_id, group_bbox, right_bbox, left_bbox,
  parent_bbox / region_ref（检测依据）, source（哪个入口）, run_status, error,
  raw_ref（指向原始输出的键）, plus diagnostic fields.

原则：group_bbox 只由两组轨框并集得到，**不用** main_bbox；缺轨或非法框保留
诊断记录，不悄悄丢掉。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_track_eval.geometry import union_bbox  # noqa: E402


def _clean_bbox(b: Any) -> Optional[List[int]]:
    if not b or not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        v = [int(round(float(x))) for x in b]
    except (TypeError, ValueError, OverflowError):
        return None
    return v


def proposal_to_prediction(prop: Dict[str, Any], page_id: str, index: int,
                           source: str, offset: Tuple[int, int] = (0, 0),
                           raw_ref: str = '') -> Dict[str, Any]:
    """propose_page / propose_components_page 的单条 proposal → 统一预测。"""
    ox, oy = offset
    sides = {c['side']: _clean_bbox(c['bbox']) for c in prop.get('columns', [])}
    right, left = sides.get('right'), sides.get('left')
    if right:
        right = [right[0] + ox, right[1] + oy, right[2] + ox, right[3] + oy]
    if left:
        left = [left[0] + ox, left[1] + oy, left[2] + ox, left[3] + oy]
    group = union_bbox([b for b in (right, left) if b])
    parent = _clean_bbox(prop.get('parent_bbox'))
    main = _clean_bbox(prop.get('main_bbox'))
    if parent:
        parent = [parent[0] + ox, parent[1] + oy, parent[2] + ox, parent[3] + oy]
    if main:
        main = [main[0] + ox, main[1] + oy, main[2] + ox, main[3] + oy]

    issues = []
    if not right:
        issues.append('missing_right_lane')
    if not left:
        issues.append('missing_left_lane')
    if group is None:
        issues.append('group_bbox_undefined')

    ev = prop.get('evidence') or {}
    return {
        'page_id': page_id,
        'prediction_id': f'{page_id}#{source}#{index:02d}',
        'group_bbox': group,
        'right_bbox': right,
        'left_bbox': left,
        'main_bbox': main,
        'parent_bbox': parent,
        'region_ref': {'kind': 'detected_column', 'bbox': parent},
        'source': source,
        'run_status': 'ok' if not issues else 'degraded',
        'error': ';'.join(issues) if issues else '',
        'raw_ref': raw_ref,
        'evidence': {
            'windows': ev.get('windows'),
            'valley_x': ev.get('valley_x'),
            'min_valley_ratio': ev.get('min_valley_ratio'),
            'core_y': ev.get('core_y'),
            'component_boundary': ev.get('component_boundary'),
        },
        'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框',
    }


def failed_prediction(page_id: str, source: str, error: str, index: int = 0) -> Dict[str, Any]:
    return {'page_id': page_id, 'prediction_id': f'{page_id}#{source}#fail{index}',
            'group_bbox': None, 'right_bbox': None, 'left_bbox': None,
            'source': source, 'run_status': 'failed', 'error': error,
            'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框'}


def run_entry(name: str, image_path: str, page_id: str,
              ocr_blocks: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """跑一个已确认的旧入口；返回 {predictions, raw, layout, status, error, timings}。

    name:
      - layout_propose_page   : page_layout + propose_page（无 OCR 依赖，主基线）
      - components_page       : page_layout + propose_components_page（需 ocr_blocks，辅助基线）
    """
    import time
    from ocr.vertical import imread_unicode, page_layout

    out: Dict[str, Any] = {'entry': name, 'page_id': page_id, 'predictions': [],
                           'raw': [], 'layout': None, 'status': 'not_run',
                           'error': '', 'timings': {}}
    t0 = time.time()
    try:
        lay = page_layout(image_path)
        out['layout'] = lay
        out['timings']['layout_s'] = round(time.time() - t0, 3)
        img = imread_unicode(image_path)
        if img is None:
            out.update(status='failed', error='image_unreadable')
            return out
        out['timings']['image_size'] = [int(img.shape[1]), int(img.shape[0])]

        t1 = time.time()
        if name == 'layout_propose_page':
            from ocr.small_columns import propose_page
            props = propose_page(img, lay['cols']) or []
        elif name == 'components_page':
            from ocr.small_columns_components import propose_components_page
            if not ocr_blocks:
                out.update(status='failed', error='ocr_blocks_required_but_empty')
                return out
            res = propose_components_page(img, lay['cols'], list(ocr_blocks))
            props = res.get('predictions') or []
            out['component_audit'] = res.get('audit') or []
        else:
            out.update(status='failed', error=f'unknown_entry:{name}')
            return out
        out['timings']['detect_s'] = round(time.time() - t1, 3)

        out['raw'] = props
        out['predictions'] = [proposal_to_prediction(p, page_id, i, name, raw_ref=f'raw[{i}]')
                              for i, p in enumerate(props)]
        out['status'] = 'ok'
    except Exception as exc:                                    # noqa: BLE001
        out.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    out['timings']['total_s'] = round(time.time() - t0, 3)
    return out


def crop_context_region(group_bbox: Sequence[int], column_bbox: Optional[Sequence[int]],
                        image_size: Tuple[int, int],
                        up_ratio: float = 0.9, down_ratio: float = 0.5,
                        side_ratio: float = 0.25, pad_ratio: float = 0.35,
                        min_height: int = 380, min_width: int = 60) -> Dict[str, Any]:
    """路径 B 的上下文裁块：覆盖标题、轨道与相邻文字，但以「一栏」为基本单位。

    规则（写进配置，便于复现）：
      - 有列框时：纵向直接取该列框的 y 范围（整栏，标题和上下相邻条目都在里面），
        横向取列框左右各 pad_ratio × 列宽。这样裁块与旧算法「按列找双轨」的
        工作单位一致，不会因为裁得太窄让 detect 直接失效。
      - 无列框时：纵向用组高上下扩 up/down_ratio，横向用组宽左右扩 side_ratio。
      - 最小尺寸兜底：高 < min_height 或宽 < min_width 时以组为中心扩到下限。
      - 一律裁到图内。
    """
    W, H = image_size
    x1, y1, x2, y2 = [int(v) for v in group_bbox]
    h = max(1, y2 - y1)
    if column_bbox:
        cx1, cy1, cx2, cy2 = [int(v) for v in column_bbox]
        cw = max(1, cx2 - cx1)
        # 横向必须容纳若干列：旧算法 estimate_page_typography 会按「列宽/页宽」比例
        # 过滤列，只给 1 列宽的裁块会让它直接返回 None（实测 42px 宽的 crop 全部失效）。
        half = int(2.2 * cw)
        center = (cx1 + cx2) // 2
        left, right = max(0, center - half), min(W, center + half)
        top = max(0, int(y1 - up_ratio * h))
        bottom = min(H, int(y2 + down_ratio * h))
        side_source = 'column_bbox_width_x4.4'
    else:
        pad = int(side_ratio * max(1, x2 - x1))
        left, right = max(0, x1 - pad), min(W, x2 + pad)
        top = max(0, int(y1 - up_ratio * h))
        bottom = min(H, int(y2 + down_ratio * h))
        side_source = 'group_width_pad'

    if bottom - top < min_height:
        cy = (y1 + y2) // 2
        top = max(0, cy - min_height // 2)
        bottom = min(H, top + min_height)
    if right - left < min_width:
        cx = (x1 + x2) // 2
        left = max(0, cx - min_width // 2)
        right = min(W, left + min_width)
    return {'crop_bbox': [left, top, right, bottom],
            'offset': [left, top],
            'rule': {'up_ratio': up_ratio, 'down_ratio': down_ratio,
                     'pad_ratio': pad_ratio, 'column_width_multiplier': 4.4, 'side_source': side_source,
                     'side_ratio': side_ratio,
                     'min_height': min_height, 'min_width': min_width},
            'uses_reference_position': True,
            'note': '裁块由参考组所在列的位置生成（这是路径 B 的既定前提，需如实记录）'}
