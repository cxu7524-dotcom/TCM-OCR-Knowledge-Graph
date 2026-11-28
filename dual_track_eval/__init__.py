# -*- coding: utf-8 -*-
"""dual_track_eval —— 双轨小字组级评测器（纯几何 + 评分逻辑，不依赖 OCR/数据路径）。

设计原则
--------
1. 只依赖 numpy 与标准库，可在任何环境离线重算；不读图、不跑模型，输入是「预测 + 标注 + 协议」。
2. 优化过程保留双精度原值，展示层才格式化。
3. 几何、字形、文本三件事分开：本包只做几何与计数，不推断字形归属。
"""

from .geometry import (bbox_area, intersect_area, union_bbox, iou,
                       parse_bbox, BboxError)
from .matching import match_groups, solve_min_cost_max_flow, solve_exhaustive
from .lane import classify_lane_pairing, lane_scores
from .diagnostics import (duplicate_clusters, fragment_clusters, merge_clusters,
                          attach_diagnostics)
from .regions import (BASE_STATUSES, RegionTable, attribute_prediction,
                      negative_coverage_ratio)
from .scorer import (classify_invalid, endpoint_sensitivity, normalize_predictions,
                     score_predictions)

__all__ = [
    'bbox_area', 'intersect_area', 'union_bbox', 'iou', 'parse_bbox', 'BboxError',
    'match_groups', 'solve_min_cost_max_flow', 'solve_exhaustive',
    'classify_lane_pairing', 'lane_scores',
    'duplicate_clusters', 'fragment_clusters', 'merge_clusters', 'attach_diagnostics',
    'BASE_STATUSES', 'RegionTable', 'attribute_prediction', 'negative_coverage_ratio',
    'score_predictions', 'endpoint_sensitivity', 'classify_invalid',
    'normalize_predictions',
]

__version__ = '1.2.2'
