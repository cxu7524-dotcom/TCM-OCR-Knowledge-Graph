# -*- coding: utf-8 -*-
"""第五阶段编排与读取入口。

对外只有三类东西：
  detect_page(...)        一页 → 组 + 逐窗/断点审计
  groups_for_page(...)    阶段 6 的读取入口（组、轨框、中缝轨迹、断点）
  windows_for_page(...)   逐窗审计（调试/验收）
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .dedupe import dedupe_groups
from .detect import detect_region


def page_ink(image: np.ndarray, contract: Dict[str, Any]) -> np.ndarray:
    """按契约的 ink_mask 参数生成 0/1 掩膜（adaptiveThreshold GAUSSIAN, invert）。"""
    cfg = (contract.get('seam_profile') or {}).get('config', {}).get('ink_mask', {})
    block = int(cfg.get('block_size', 31))
    c = int(cfg.get('c', 14))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, c)
    return (bw > 0).astype(np.uint8)


def rule_segments_for_region(rule_lines: Sequence[Dict[str, Any]],
                             bbox: Sequence[int]) -> List[List[int]]:
    """只取**合格线**的**独立段**（与阶段 4 的资格判据一致），且与区域相交。"""
    out: List[List[int]] = []
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for ln in rule_lines or []:
        if not ln.get('qualified'):
            continue
        for s in ln.get('eligible_segments') or []:
            if s['x1'] <= x1 or s['x0'] >= x2 or s['y1'] <= y1 or s['y0'] >= y2:
                continue
            out.append([int(s['x0']), int(s['x1']), int(s['y0']), int(s['y1'])])
    return out


def detect_page(page: Any, ink: np.ndarray, cfg: Dict[str, Any],
                contract: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    groups: List[Dict[str, Any]] = []
    audit_regions: List[Dict[str, Any]] = []
    for region in page.regions:
        scale = page.region_scale(region)
        segs = rule_segments_for_region(page.rule_lines, region['bbox'])
        gs, audit = detect_region(page.page_id, region, ink, segs, scale, cfg)
        for g in gs:
            if g.get('state') == 'completed':
                groups.append(g)
        audit_regions.append(audit)

    kept, dedupe_log = dedupe_groups(groups, cfg['dedupe'])
    # 稳定编号：按 y、x 排序后重新编号（内部 id 留在 detection_id）
    for i, g in enumerate(kept, 1):
        g['detection_id'] = g['group_id']
        g['group_id'] = f"{page.page_id}#g{i:03d}"

    by_region = {r['region_id']: r for r in audit_regions}
    for g in kept:
        r = by_region.get(g['region_id'])
        g['region_order'] = (audit_regions.index(r) + 1) if r is not None else None

    coverage = _coverage_note(page, kept)
    return {
        'page_id': page.page_id,
        'image_path': page.image_path,
        'image_sha256': page.image_sha256,
        'image_size': page.image_size,
        'halves': page.halves,
        'status': 'completed' if kept else 'no_group',
        'groups': kept,
        'group_count': len(kept),
        'regions': [{'region_id': r['region_id'], 'region_bbox': r['region_bbox'],
                     'source': r['source'],
                     'low_confidence': r['low_confidence'],
                     'skipped': r['skipped'],
                     'groups': r['groups'], 'windows': len(r['windows']),
                     'candidate_runs': r['candidate_runs'],
                     'degraded': r['degraded'],
                     'scale_status': r['scale_status']}
                    for r in audit_regions],
        'windows': {r['region_id']: r['windows'] for r in audit_regions},
        'dedupe': dedupe_log,
        'coverage': coverage,
        'timing_s': round(time.time() - t0, 3),
    }


def _coverage_note(page: Any, groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    regions = page.regions
    used = {g['region_id'] for g in groups}
    by_src: Dict[str, int] = {}
    for r in regions:
        by_src[r.get('column_range_source')] = by_src.get(r.get('column_range_source'), 0) + 1
    return {'regions': len(regions), 'regions_by_source': by_src,
            'regions_with_group': len(used),
            'regions_without_group': len(regions) - len(used)}


# ------------------------------------------------------------------ 阶段 6 读取

def _page_of(payload: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    for p in payload.get('pages', []):
        if p['page_id'] == page_id:
            return p
    raise KeyError(f'page not found in stage5 output: {page_id}')


def groups_for_page(payload: Dict[str, Any], page_id: str) -> List[Dict[str, Any]]:
    return _page_of(payload, page_id)['groups']


def windows_for_page(payload: Dict[str, Any], page_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """逐窗审计：主体在 windows.json（体积大），这里给内联摘要或空字典。

    需要完整逐窗记录时读 `windows.json` 的对应页；入口始终可用、不抛异常。
    """
    return _page_of(payload, page_id).get('windows') or {}


def audit_for_page(payload: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    p = _page_of(payload, page_id)
    return {'status': p['status'], 'coverage': p['coverage'], 'dedupe': p['dedupe'],
            'regions': p['regions']}


def groups_as_predictions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """转成阶段 3 的统一直觉预测格式（供冻结评分配置复用）。"""
    out = []
    for p in payload.get('pages', []):
        for g in p.get('groups') or []:
            out.append({
                'page_id': p['page_id'],
                'prediction_id': g['group_id'],
                'group_bbox': g.get('group_bbox'),
                'right_bbox': g.get('right_bbox'),
                'left_bbox': g.get('left_bbox'),
                'parent_bbox': (next((r['region_bbox'] for r in p.get('regions', [])
                                      if r['region_id'] == g['region_id']), None)),
                'region_ref': {'kind': 'stage5_region', 'region_id': g['region_id'],
                               'source': g.get('region_source')},
                'source': 'stage5_dual_track',
                'run_status': 'ok',
                'error': '',
                'pending': g.get('pending'),
                'stop_events': g.get('stop_events'),
            })
    return out
