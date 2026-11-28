# -*- coding: utf-8 -*-
"""轨级评分：组匹配固定后，先查固定侧别，再查交叉侧别。

分母约定（协议 lane_scoring.denominators）：
- 配对正确率 = pairing_success / 具备轨级资格的已匹配组数
- 左右交换率、子框定位失败率用同一分母
- 因 certainty=ambiguous_* 被排除的组要单列数量，不混进分母
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .geometry import iou, parse_bbox

LANE_ELIGIBLE_EXCLUDE = ('ambiguous_lane_split', 'ambiguous_lane_extent')


def lane_scores(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    """算固定侧别与交叉 IoU；缺框时返回 None 并在 note 里说明。"""
    out: Dict[str, Any] = {'right_iou_fixed': None, 'left_iou_fixed': None,
                           'right_pred_vs_gt_left_iou': None, 'left_pred_vs_gt_right_iou': None,
                           'note': ''}
    try:
        pr = parse_bbox(pred.get('right_bbox'))
        pl = parse_bbox(pred.get('left_bbox'))
        gr = parse_bbox(gt.get('right_bbox'))
        gl = parse_bbox(gt.get('left_bbox'))
    except Exception as exc:
        out['note'] = f'轨框不可解析：{type(exc).__name__}'
        return out
    out['right_iou_fixed'] = iou(pr, gr)
    out['left_iou_fixed'] = iou(pl, gl)
    out['right_pred_vs_gt_left_iou'] = iou(pr, gl)
    out['left_pred_vs_gt_right_iou'] = iou(pl, gr)
    return out


def classify_lane_pairing(scores: Dict[str, Any], right_thr: float = 0.5,
                          left_thr: float = 0.5) -> Dict[str, Any]:
    """三分类：pairing_success / left_right_swap / child_localization_failure。"""
    rr, ll = scores.get('right_iou_fixed'), scores.get('left_iou_fixed')
    rl, lr = scores.get('right_pred_vs_gt_left_iou'), scores.get('left_pred_vs_gt_right_iou')
    if rr is None or ll is None or rl is None or lr is None:
        return {'category': 'child_localization_failure',
                'reason': 'missing_or_invalid_lane_bbox',
                'detail': scores.get('note', '')}
    if rr >= right_thr and ll >= left_thr:
        return {'category': 'pairing_success',
                'reason': '固定的 right-right 与 left-left 均达子框阈值'}
    if rl >= right_thr and lr >= left_thr:
        return {'category': 'left_right_swap',
                'reason': '固定侧别未同时通过，交叉对照两侧均达阈值'}
    return {'category': 'child_localization_failure',
            'reason': '固定侧别与交叉对照都未同时达阈值'}


def lane_eligible(gt: Dict[str, Any],
                  exclude: Optional[Sequence[str]] = None) -> bool:
    """是否进入轨级分母。

    exclude=None 用模块默认清单；exclude=[] 表示**不排除任何 certainty**
    （包括 ambiguous_lane_*）——空名单是有意义的配置，不是「没配」。
    """
    pool = LANE_ELIGIBLE_EXCLUDE if exclude is None else tuple(exclude)
    cert = str(gt.get('certainty') or '')
    return cert not in pool
