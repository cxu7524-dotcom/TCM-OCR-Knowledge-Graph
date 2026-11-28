# -*- coding: utf-8 -*-
"""本阶段全部可调参数的唯一来源。

参数分两类，别混：
  - **从数据量出来的**（如 1.8 倍宽框阈值沿用阶段 3 冻结规则）——写清来源；
  - **经验初值**（如长竖线最小高度 35%）——标 provisional，阶段 5 起可调。

v2 相对 v1 的改动全部对应验收缺口，逐项注明来源：
  rules.run_split_jump_ratio   一个不合格间隔不能作废整段（V2 要求「其余范围继续处理」）
  columns.page_reference_min_cols / 参考链  宽度参考不足时要有「有来源的替代估计」
  columns.max_overlap_ratio    主体重叠第一次真正参与判定（v1 只在配置里存在）
  columns.rule_conflict_*      栏内独立栏线 → 拆分，而不是只记诊断
  scale.band_support_ratio     行活跃度按**墨点数量**算，0/1 与 0/255 等价
  scope.*                      区分「范围外」与「应处理但未覆盖」
"""

from __future__ import annotations

DEFAULT_CONFIG = {
    'version': 'stage4-v2',

    # ---------------------------------------------------------------- 栏线检测
    'rules': {
        'min_height_ratio': 0.35,        # 候选线高度 / 正文区高度；provisional
        'max_width_px': 6,               # 线宽上限（超过就当块状墨迹，不当线）
        'open_kernel_ratio': 0.25,       # 开运算核高 = 该比例 × 正文区高；provisional
        'touching_probe_px': 3,          # 线两侧探针宽度
        'touching_ink_ratio': 0.25,      # 探针列有墨行占比 > 此值 → 判「贴字」
        'broken_gap_px': 18,             # 同 x 相邻段间隙 <= 此值 → 视为断线可合并
        'broken_max_dx': 3,
        'group_min_count': 3,            # 成组至少几条线（不含页框/书脊）
        'group_pitch_cv_max': 0.25,      # 相邻间距的变异系数上限
        'group_pitch_min_px': 18,        # 相邻间距下限（太小就是双线）
        'group_pitch_max_px': 120,       # 相邻间距上限（太大就不是同一组栏）
        'group_min_coverage': 0.60,      # 每条线在正文区的最小纵向覆盖
        'group_min_ink_between': 0.02,   # 两线之间墨迹占比下限（栏内要有字）
        'run_split_jump_ratio': 1.6,     # 间距突增到中位数的该倍数 → 断成新的一组
        'dedupe_dx_px': 2,               # 同 x（±该值）重复候选只留覆盖最高的一条
    },

    # ---------------------------------------------------------------- 列框校验
    'columns': {
        'width_ref_source': 'same_book→same_page→global→none',
        'outlier_multiplier': 1.8,       # 沿用阶段 3 冻结规则
        'min_reference_cols': 30,        # 同书参考所需列数
        'page_reference_min_cols': 5,    # 同页替代参考所需列数（有来源，非默认值）
        'global_substitute': True,
        'max_overlap_ratio': 0.15,       # 相邻列框主体重叠 / 较窄列宽
        'min_gutter_ratio': 0.12,        # 邻栏最小间隙 / 参考宽；更小记诊断
        'edge_margin_ratio': 0.03,       # 列框距页边最小比例（只记诊断，不单独否决）
        'min_cols_per_half': 2,
        'max_pitch_cv': 0.35,            # 半页内列间距的变异系数上限
        'merge_gap_factor': 0.35,         # 相邻列间隙 <= 该比例×较窄列宽 → 可能是同栏被切开
        'merge_max_reference_multiple': 1.35,  # 合并后宽度上限（× 参考宽），防止并邻栏
        'rule_conflict_margin_px': 2,    # 独立栏线离框边至少这么远才算「框内」
        'split_min_width_multiple': 0.35,  # 拆分后每段最小宽 = 该倍数 × 参考宽
        'min_width_px': 8,               # 绝对下限（参考缺失时也用）
        'half_boundary_tolerance_px': 2,  # 列框越过半页边界超过该值 → 跨界
    },

    # ---------------------------------------------------------------- 局部窗口
    'windows': {
        'width_source': 'scale.H',       # 窗口宽度按字高推算
        'width_multiplier': 6.0,         # 宽 ≈ 6 倍字高（约 2-3 栏）
        'step_ratio': 0.6,               # 步长 = 宽 × 0.6，保证 40% 重叠
        'min_width_px': 60,
        'max_width_px': 260,
        'min_height_ratio': 1.0,         # 高 = 正文区高 × 此值；同栏多组交给阶段 5 分
        'narrow_gap_min_px': 12,         # 窄缝（<最小窗宽）也出窗口，避免无人负责
    },

    # ---------------------------------------------------------------- 尺度估计
    'scale': {
        'min_samples_H': 8,              # H/G/pitch 各自的最小样本数
        'min_samples_narrow': 5,         # 窄裁块放宽到 5
        'narrow_width_cols': 2,          # 宽度 <= 该列数视为窄裁块
        'narrow_width_px': 60,           # 或框本身宽度 <= 该值
        'height_band_ratio': [0.16, 0.92],   # 无列框时的正文带兜底（避开页眉页脚）
        'band_min_width_ratio': 0.45,    # 字形带宽度 / 列宽 下限
        'band_support_ratio': 0.11,      # 行活跃度：有墨像素数 >= 该比例 × 列宽
        'band_min_support_px': 2,        # 行活跃度的墨点绝对下限
        'height_trim': [0.5, 1.8],       # 高度离群裁剪（相对中位数）
        'gap_max_px': 40,                # 超过这个「留白」多半是段落间隔，不计入 G
        'ink_unit': 'binary_gt0',        # 前景掩膜统一为 0/1；0/255 输入先归一
    },

    # ---------------------------------------------------------------- 页面范围
    'scope': {
        'edge_margin_ratio': 0.015,      # 半页左右页边算「范围外」
        'edge_margin_min_px': 4,
        'uncovered_min_len_px': 8,       # 短于此的未覆盖段并入相邻段记账
        'gutter_residual_max_px': 24,    # 两侧都有区域的残余缝 <= 此值 → 栏间空白
    },

    # ---------------------------------------------------------------- 输出
    'output': {
        'region_id_format': '{page_id}#{half_id}#{source}#{seq:03d}',
        'low_confidence_flag': 'low_confidence',
        'mask_record': 'rule_masks.json',
    },
}
