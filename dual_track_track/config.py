# -*- coding: utf-8 -*-
"""第五阶段全部可调参数的唯一来源。

参数分三类，别混：
  - **契约给定**（来自 `cross_stage_contract.json` 的 seam_profile / scale_object）；
  - **本轮开发初值**（在开发集上起步，进阶段 6 前可在配置里调）；
  - **结构常量**（状态机、枚举、去重优先级）。

`status` 字段说明来源：contract / development_initial / structural。
"""

from __future__ import annotations

DEFAULT_CONFIG = {
    'version': 'stage5-v4',

    # ---------------------------------------------------------------- 中缝证据
    'seam': {
        'epsilon': 0.02,                 # r = V / (min(L,R)+eps)；contract
        'max_r': 0.15,                   # 上限判据；contract
        'min_peak': 0.10,                # 归一化覆盖率峰值下限；contract
        'min_support_ratio': 0.30,       # 搜索带内有墨行占比下限；contract
        'band_pitch_multiple': 1.5,      # 搜索带宽 = 1.5 × glyph_pitch；contract
        'band_min_px': 6,
        'seam_width_pitch_multiple': 0.4,  # 中缝带宽 = 0.4 × glyph_pitch；contract
        'seam_width_min_px': 3,
        'peak_min_gap_px': 2,            # 峰值点距中缝带边界的最小距离；contract
        'search_step_px': 1,             # 候选 x 扫描步长
        'smooth_pitch_multiple': 0.12,   # 平滑窗 = 0.12 × pitch；contract
        'smooth_min_px': 3,
        'long_gap_max': 0.05,            # 区域级投影「持续低谷」阈值（贯穿整栏才当缝）
    },

    # ---------------------------------------------------------------- 窗口
    'window': {
        'height_H': 0.8,                 # 窗高 = 0.8H；development_initial
        'step_H': 0.25,                  # 步长 = 0.25H；development_initial
        'min_height_px': 12,
        'max_height_px': 160,
        'min_consecutive': 3,            # 连续有效窗 N；development_initial
        'min_union_H': 1.0,              # 确认区段联合覆盖；development_initial
        'max_confirm_H': 5.0,            # 确认段最多延伸到此高度，凑独立字形数用
        'cluster_min_windows': 3,        # 一条缝至少在这么多窗口里出现才算贯穿（不按比例）
    },

    # ---------------------------------------------------------------- 字形支持
    'glyph': {
        'min_height_H': 0.25,            # 有效字形块高下限（相对 H）
        'min_height_px': 6,
        'max_height_H': 2.0,             # 超过即视为大字带（标题/正文大字）
        'min_width_px': 3,
        'min_area_px': 12,
        'max_width_H': 1.6,              # 超宽连通域不当字形（横线/污点）
        'min_per_side_window': 1,        # 每个窗口每侧至少几个字形块
        'min_per_side_group': 3,         # 确认区段每侧至少几个**独立**字形（按完整框去重）
        'small_max_H': 0.80,             # 双轨小字：区内字形高度中位数上限
    },

    # ---------------------------------------------------------------- 轨几何
    'track': {
        'max_width_H': 1.60,             # 单轨宽上限（双轨小字轨宽≈1.0H）
        'min_width_H': 0.25,             # 单轨宽下限（太窄=污点/残笔）
        'pair_max_width_H': 3.2,         # 两轨合计宽上限
        'seam_gap_max_H': 1.0,           # 两轨之间（中缝）宽上限
        'seam_gap_min_px': 1,
        'span_width_max_H': 2.5,         # 跨多窗的轨外接框允许轻微漂移（逐窗宽度另由 max_width_H 管）
        'width_ratio_min': 0.45,         # 两轨宽度比下限（双轨近似等宽）
        'width_ratio_max': 2.60,
        'gap_to_track_max': 0.80,        # 缝宽 / 较窄轨宽 上限（双轨缝很窄）
    },

    # ---------------------------------------------------------------- 中缝稳定性
    'stability': {
        'resid_std_H': 0.05,             # 残差标准差上限；development_initial
        'max_resid_H': 0.10,             # 单点最大残差上限；development_initial
        'max_abs_slope': 0.03,           # |斜率| 上限；development_initial
        'min_points': 3,
    },

    # ---------------------------------------------------------------- 停止判据
    'stop': {
        'invalid_windows': 2,            # 中缝连续失效窗 M；development_initial
        'seam_gap_windows': 6,           # 中缝连续缺席超过此数即断组，跨标题/大留白另起一组
        'track_end_windows': 4,          # 单轨连续无字形窗数（比中缝宽松，避免碎组）
        'fragment_merge_gap_H': 0.6,     # 同栏碎片合并：纵向间隙上限
        'fragment_merge_x_overlap': 0.7,
        'blank_gap_G': 0.8,              # 连续留白阈值 = 0.8G；contract（纵向停止）
        'blank_gap_min_px': 6,
        'big_glyph_H': 1.3,              # 大字带：块高 ≥ 1.3H
        'big_glyph_min_count': 1,         # 窗口内至少几个大字块才算「大字带」
        'big_glyph_w_H': 1.0,             # 大字块还必须够宽（横跨两轨才是正文/标题）
        'rule_width_H': 3.0,             # 横线：宽 ≥ 3H 且高 ≤ 0.35H
        'rule_height_H': 0.35,
        'trim_after_gap_px': 2,          # 停在留白处时回退的像素
    },

    # ---------------------------------------------------------------- 去重
    'dedupe': {
        'iou': 0.5,
        'x_overlap': 0.6,
        'y_overlap': 0.5,                # 同栏上下两组要求 y 重叠 > 0.5 才算重复
        'height_ratio_max': 1.8,         # 高度差过大时不并（短组嵌在长组里 ≠ 同一组）
        'source_priority': ['grouped_rules', 'validated_columns', 'halfpage_fallback'],
    },

    # ---------------------------------------------------------------- 尺度使用
    'scale': {
        'min_H_px': 8,
        'max_H_iqr_ratio': 0.6,          # 离散度 > 0.6H → 待核（仍可执行）
        'pitch_fallback_multiple': 1.0,  # glyph_pitch 缺失时用 1.0H（记录来源）
        'G_fallback_multiple': 0.25,     # G 缺失时用 0.25H（记录来源）
    },

    # ---------------------------------------------------------------- 状态与枚举
    'states': {
        'order': ['searching', 'candidate', 'confirmed', 'tracking', 'completed', 'rejected'],
        'stop_types': ['page_boundary', 'region_boundary', 'rule_boundary', 'next_title',
                       'blank_gap', 'seam_missing', 'seam_drift', 'track_ink_end',
                       'ambiguous_ink'],
        'reject_reasons': ['insufficient_peak', 'insufficient_support', 'blank_window',
                           'no_glyph_support', 'seam_not_clear', 'rule_at_seam',
                           'ambig_ink_crossing', 'no_search_band', 'track_geometry',
                           'insufficient_unique_glyphs', 'no_usable_scale'],
    },

    'output': {
        'group_id_format': '{page_id}#g{seq:03d}',
        'pending_flag': 'pending_review',
    },
}
