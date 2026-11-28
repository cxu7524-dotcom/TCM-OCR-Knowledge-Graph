# -*- coding: utf-8 -*-
"""诊断标签：重复、碎组、并组。

铁律：**先完成全局一对一匹配，再贴标签**。标签不产生也不撤销 TP/FP/FN。
禁止「另取局部 IoU 最大者计 TP」——那会推翻全局最优解（CASE-A 就是反例）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .geometry import bbox_area, intersect_area, iou, union_ratio

DEFAULTS = {
    'fragment_union_coverage': 0.50,   # 多预测并集覆盖标注组面积的比例下限
    'merge_containment': 0.60,         # 单预测覆盖单个标注组的面积比例下限
}


def duplicate_clusters(matrix: Sequence[Sequence[float]], thr: float,
                       matched: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """同一标注组有 >=2 个预测达到阈值：匹配上的那个是 TP，其余是多余项。"""
    matched_pairs = [(int(a), int(b)) for a, b in matched]
    matched_preds = {a for a, _ in matched_pairs}
    out = []
    m = len(matrix[0]) if matrix else 0
    for j in range(m):
        cands = [i for i in range(len(matrix)) if matrix[i][j] >= thr]
        if len(cands) < 2:
            continue
        chosen = next((i for i, jj in matched_pairs if jj == j), None)
        extras = [i for i in cands if i != chosen]
        out.append({'gt_index': j, 'candidate_preds': cands, 'matched_pred': chosen,
                    'extra_preds': extras,
                    'max_iou_among_extras': max([matrix[i][j] for i in extras], default=0.0),
                    'extra_preds_already_matched_elsewhere': [i for i in extras
                                                             if i in matched_preds],
                    'note': ('多余预测若已在全局解中匹配到别的标注，只记诊断、'
                             '不再另计 FP（它已经是某个 TP）')})
    return out


def fragment_clusters(matrix: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[int]],
                      gt_boxes: Sequence[Sequence[int]], matched: Sequence[Sequence[int]],
                      matched_gt: Sequence[int] = (), coverage: float = DEFAULTS['fragment_union_coverage']
                      ) -> List[Dict[str, Any]]:
    """多个未匹配预测拼起来才盖住某个未匹配标注组。"""
    matched_preds = {int(a) for a, _ in matched}
    matched_gts = {int(b) for _, b in matched} | {int(x) for x in matched_gt}
    out = []
    for j, gtb in enumerate(gt_boxes):
        if j in matched_gts:
            continue
        touching = [i for i, p in enumerate(pred_boxes)
                    if i not in matched_preds and intersect_area(p, gtb) > 0]
        if len(touching) < 2:
            continue
        cover = union_ratio(gtb, [pred_boxes[i] for i in touching])
        if cover >= coverage:
            out.append({'gt_index': j, 'pred_indices': touching,
                        'union_coverage_of_gt': round(cover, 4),
                        'threshold': coverage, 'labels': ['fragment']})
    return out


def merge_clusters(pred_boxes: Sequence[Sequence[int]], gt_boxes: Sequence[Sequence[int]],
                   containment: float = DEFAULTS['merge_containment']) -> List[Dict[str, Any]]:
    """单个预测把 >=2 个标注组各自盖住 60% 以上。"""
    out = []
    for i, pb in enumerate(pred_boxes):
        hits = []
        for j, gtb in enumerate(gt_boxes):
            area = bbox_area(gtb)
            if area <= 0:
                continue
            r = intersect_area(pb, gtb) / area
            if r >= containment:
                hits.append({'gt_index': j, 'containment': round(r, 4)})
        if len(hits) >= 2:
            out.append({'pred_index': i, 'gts': hits, 'threshold': containment,
                        'labels': ['merge']})
    return out


def attach_diagnostics(matrix: Sequence[Sequence[float]], thr: float,
                       pred_boxes: Sequence[Sequence[int]], gt_boxes: Sequence[Sequence[int]],
                       matched: Sequence[Sequence[int]],
                       thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """汇总三类诊断；每条预测/标注可带多个标签。阈值可外部传入（配置驱动）。"""
    th = dict(DEFAULTS, **(thresholds or {}))
    dups = duplicate_clusters(matrix, thr, matched)
    frags = fragment_clusters(matrix, pred_boxes, gt_boxes, matched,
                              coverage=th['fragment_union_coverage'])
    merges = merge_clusters(pred_boxes, gt_boxes, containment=th['merge_containment'])

    pred_labels: Dict[int, List[str]] = {}
    gt_labels: Dict[int, List[str]] = {}
    for d in dups:
        gt_labels.setdefault(d['gt_index'], []).append('duplicate_prediction')
        for i in d['extra_preds']:
            pred_labels.setdefault(i, []).append('duplicate_prediction')
    for f in frags:
        gt_labels.setdefault(f['gt_index'], []).append('fragment')
        for i in f['pred_indices']:
            pred_labels.setdefault(i, []).append('fragment')
    for mg in merges:
        pred_labels.setdefault(mg['pred_index'], []).append('merge')

    return {'duplicates': dups, 'fragments': frags, 'merges': merges,
            'pred_labels': {str(k): sorted(set(v)) for k, v in pred_labels.items()},
            'gt_labels': {str(k): sorted(set(v)) for k, v in gt_labels.items()},
            'counts': {'duplicate_clusters': len(dups), 'fragment_clusters': len(frags),
                       'merge_clusters': len(merges),
                       'preds_with_labels': len(pred_labels),
                       'gts_with_labels': len(gt_labels)},
            'thresholds': {'fragment_union_coverage': th['fragment_union_coverage'],
                           'merge_containment': th['merge_containment']},
            'rule': '标签在全局匹配之后追加，不改动 TP/FP/FN'}
