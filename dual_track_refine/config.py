# -*- coding: utf-8 -*-
"""第六阶段全部可调参数（双轨子框收紧 + 首尾补笔）。

只调本阶段；stage4 搜索范围/尺度/栏线、stage5 组 ID 与成对结果全部冻结。
"""

from __future__ import annotations

DEFAULT_CONFIG = {
    'version': 'stage6-v4',

    'glyph': {
        'overlap_with_track': 0.60,   # 字形必须主要落在暂定轨框内才算本轨确认字形
        'edge_margin_H': 0.35,        # 回溯本轨字形时允许的横向扩展
        'min_width_H': 0.30,          # 小于此宽度的域视为异常细长/残笔
        'slender_ratio': 3.0,         # h/w 超过此值需与周边字形复核
        'boundary_touch_px': 1,       # 贴搜索区边缘即视为可能被裁断
        'rule_near_px': 3,            # 贴栏线需复核的横向距离
    },

    'edge': {
        'min_width_H': 0.60,          # 约束相交后轨宽下限（相对 H）
        'seam_margin_px': 1,
        'rule_margin_px': 2,
        'region_margin_px': 0,
    },

    'supplement': {
        'max_per_side_end': 2,        # 每轨每端最多补几个字形
        'pitch_multiple': 1.6,        # 纵向间距上界 = 局部 pitch × 此值
        'pitch_min_H': 0.8,
        'pitch_max_H': 3.0,
        'size_h_min': 0.55,           # 相对局部小字中位高度
        'size_h_max': 1.60,
        'size_w_min': 0.50,
        'size_w_max': 2.00,
        'lateral_margin_H': 0.30,     # 补笔允许超出当前边的横向余量
        'hard_boundary_margin_px': 1,
    },

    'states': {
        'low_confidence_reasons': [
            'horizontal_low_confidence', 'supplement_limited',
            'boundary_clipped_glyph', 'ambiguous_rule_attached',
            'ambiguous_long_stroke', 'seam_adjacent_ink',
        ],
        'exclusion_reasons': [
            'crosses_seam', 'crosses_rule', 'crosses_stop',
            'outside_region', 'boundary_clipped', 'size_mismatch',
            'gap_too_large', 'outside_track_lane', 'duplicate',
            'above_hard_boundary',
        ],
    },
}
