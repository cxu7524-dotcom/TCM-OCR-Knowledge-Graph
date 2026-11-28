# -*- coding: utf-8 -*-
"""stage8 参数（唯一来源，只放本轮新增/被改的项）。

阈值只在开发集（丹方 p12 11 组、p33/p35/p36/p42 4 组）上标定；
历史回归集（8 页 20 组）与独立验收集（p38、温疫 p12）不参与调参。
"""

DEFAULT_CONFIG = {
    'version': 'stage8-v1',

    'seam_pair': {
        # 中缝走廊：纵向至少连续空出 0.85H（≈一个字），横向宽 ≤0.22H
        'min_run_H': 0.85,
        'max_corridor_H': 0.22,
        # 走廊两侧（紧邻 2px）墨行占比下限：排除栏间空档、页边空白
        'flank_px': 2,
        'flank_ratio': 0.15,
        # 走廊块本身的门槛（防止把碎点当走廊）
        'corridor_min_area': 20,
        'corridor_max_fill': 0.60,
        # 轨列窗：从缝向外最多找 1.6H，墨量降到峰值 35% 即轨结束
        'reach_H': 1.60,
        'track_edge_ratio': 0.35,
        # 轨框尺寸门槛（开发集：p12 轨宽 23–24px=1.1H，思濟堂 15–21px=0.7–0.9H）
        'track_width_min_H': 0.50,
        'track_width_max_H': 1.85,
        'width_ratio_max': 2.60,
        'side_min_height_H': 0.45,
        # 去重
        'dedupe_iou': 0.45,
        'dedupe_containment': 0.80,
    },

    'ink': {
        'block_size': 31,
        'c': 14,
        'glyph_min_area': 12,
    },
}
