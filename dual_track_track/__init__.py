# -*- coding: utf-8 -*-
"""双轨小字检测与追踪（阶段 5）。

输入只来自阶段 4 的公开产物：搜索区域、尺度、栏线证据（独立段）、审计。
输出给阶段 6：每个已确认组的暂定组框、右/左轨原图框、中缝轨迹、断点事件。

设计约束：
  1. 坐标一律原图 [x1,y1,x2,y2)；阶段 4 已回映，本阶段不再引入裁块坐标。
  2. 尺度先实测后继承，缺什么记什么，不允许无来源默认值。
  3. 每个断点都有枚举 + 原图 y + 「扫描在哪确认」的位置。
  4. 单轨、噪声、重叠窗口：要么拒绝并留原因，要么只输出一次。
"""

from __future__ import annotations

from .api import (audit_for_page, detect_page, groups_as_predictions,
                  groups_for_page, page_ink, rule_segments_for_region,
                  windows_for_page)
from .config import DEFAULT_CONFIG
from .dedupe import bbox_iou, dedupe_groups, is_duplicate
from .detect import detect_region
from .inputs import Bundle, PageInput, load_bundle
from .projection import (RegionInk, build_region_ink, glyph_components,
                         moving_average, projection, side_stats, smooth_window_px)
from .seam import SeamEval, evaluate_window

__all__ = [
    'DEFAULT_CONFIG', 'Bundle', 'PageInput', 'load_bundle',
    'detect_page', 'detect_region', 'page_ink', 'rule_segments_for_region',
    'groups_for_page', 'windows_for_page', 'audit_for_page', 'groups_as_predictions',
    'dedupe_groups', 'is_duplicate', 'bbox_iou',
    'RegionInk', 'build_region_ink', 'glyph_components', 'projection',
    'moving_average', 'side_stats', 'smooth_window_px',
    'SeamEval', 'evaluate_window',
]

__version__ = '1.0.0'
