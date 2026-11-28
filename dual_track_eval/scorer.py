# -*- coding: utf-8 -*-
"""评分器：把预测、标注、区域表和协议配置组合成一份可复现的评测结果。

执行顺序（这个顺序本身就是协议的一部分）：
  1. 非法预测先分流（缺轨 / 组框非法 / 越界），单列计数，不进匹配；
  2. 有效预测做匹配前归属：ignore 直接排除，ambiguous 只在「够不到任何正例」时暂缓；
  3. 剩下的预测与标注做组级匹配（每个 IoU 阈值独立求解）；
  4. 匹配上的判 TP，未匹配的按负例/未覆盖规则定 FP 或候选；
  5. 诊断标签在全局匹配之后追加，不改基础状态。

每个预测最终只有一个 base_status ∈ {TP, FP, IGNORED, PENDING_AMBIGUOUS,
PENDING_CANDIDATE, INVALID}，加总必须等于预测总数。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .diagnostics import attach_diagnostics
from .geometry import bbox_area, intersect_area, iou, parse_bbox, try_parse_bbox, union_bbox
from .lane import classify_lane_pairing, lane_eligible, lane_scores
from .regions import BASE_STATUSES, RegionTable

DEFAULT_CONFIG = {
    'group_iou_thresholds': [0.5, 0.75],
    'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5},
    'lane_eligible_exclude_certainty': ['ambiguous_lane_split', 'ambiguous_lane_extent'],
    'endpoint_pending_inclusive_shift_px': 1,
    'region_thresholds': None,
    'numeric': {'tie_break_tolerance': 0.0,
                'note': '并列判定用浮点精确相等，不留容差'},
}


# ---------------------------------------------------------------- 适配与校验

def _valid_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    b, _code, _detail = try_parse_bbox(value)
    if b is None or bbox_area(b) <= 0:
        return None
    return b


def classify_invalid(p: Dict[str, Any], image_size: Optional[Sequence[int]] = None) -> List[str]:
    """非法预测的分类；空列表表示合法。"""
    issues = []
    if _valid_bbox(p.get('group_bbox')) is None:
        issues.append('invalid_group_bbox')
    if _valid_bbox(p.get('right_bbox')) is None:
        issues.append('missing_or_invalid_right_lane')
    if _valid_bbox(p.get('left_bbox')) is None:
        issues.append('missing_or_invalid_left_lane')
    if image_size and len(image_size) == 2 and p.get('group_bbox'):
        b = try_parse_bbox(p['group_bbox'])[0]
        if b is not None:
            W, H = int(image_size[0]), int(image_size[1])
            if b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H:
                issues.append('out_of_image')
    return issues


def normalize_predictions(preds: Iterable[Dict[str, Any]],
                          page_sizes: Optional[Dict[str, Sequence[int]]] = None
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """补齐 group_bbox（左右轨并集），把非法框标出来；返回 (有效, 非法)。"""
    valid, invalid = [], []
    for p in preds:
        q = dict(p)
        rb = _valid_bbox(q.get('right_bbox'))
        lb = _valid_bbox(q.get('left_bbox'))
        gb = _valid_bbox(q.get('group_bbox'))
        if gb is None and rb and lb:
            gb = union_bbox([rb, lb])
            q['group_bbox_source'] = 'union_of_lanes'
        elif gb is not None:
            q['group_bbox_source'] = q.get('group_bbox_source', 'provided')
        q['_bbox'] = gb
        q['_right'] = rb
        q['_left'] = lb
        size = (page_sizes or {}).get(q.get('page_id'))
        q['_issues'] = classify_invalid(q, size)
        (invalid if gb is None else valid).append(q)
    return valid, invalid


def summarize_issues(valid: Sequence[Dict[str, Any]],
                     invalid: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把问题分成两类：不该进匹配的（无有效组框）与进了匹配但要标记的（缺轨/越界）。"""
    by_issue: Dict[str, int] = {}
    for p in list(valid) + list(invalid):
        for issue in p.get('_issues') or []:
            by_issue[issue] = by_issue.get(issue, 0) + 1
    degraded = [p for p in valid if p.get('_issues')]
    unlocatable = [p for p in invalid if p.get('group_bbox') is None]
    return {'invalid_count': len(invalid), 'degraded_count': len(degraded),
            'unlocatable_count': len(unlocatable), 'by_issue': by_issue,
            'degraded_ids': [p.get('prediction_id') for p in degraded],
            'policy': 'base_status=INVALID 的预测不参与匹配；'
                      'degraded（有组框但缺轨/越界）仍参与匹配，但轨级会判失败、'
                      '越界视为位置可疑；unlocatable 单列，计入无效输出统计'
            }


def _sort_key(item: Dict[str, Any], bbox_key: str, id_key: str) -> Tuple[int, int, str]:
    b = item.get(bbox_key) or [0, 0, 0, 0]
    return (int(b[1]), int(b[0]), str(item.get(id_key) or ''))


# ---------------------------------------------------------------- 端点敏感性

def _shift_br(box: Sequence[int], px: int) -> List[int]:
    return [int(box[0]), int(box[1]), int(box[2]) + px, int(box[3]) + px]


def endpoint_variant(groups: Sequence[Dict[str, Any]], mode: str,
                     shift_px: int = 1) -> List[Dict[str, Any]]:
    """mode='nominal' 原框；mode='inclusive' 把 pending 组的右下角当包含端点（+1）。"""
    out = []
    for g in groups:
        h = dict(g)
        ep = g.get('endpoint_interpretation') or {}
        pending = str(ep.get('status')) == 'pending'
        if mode == 'inclusive' and pending:
            for k in ('group_bbox', 'right_bbox', 'left_bbox', 'main_bbox'):
                if g.get(k):
                    h[k] = _shift_br(g[k], shift_px)
            h['_endpoint_applied'] = 'inclusive'
        else:
            h['_endpoint_applied'] = 'nominal' if pending else 'resolved'
        out.append(h)
    return out


# ---------------------------------------------------------------- 单页评分

def score_page(page_id: str, preds: Sequence[Dict[str, Any]], gts: Sequence[Dict[str, Any]],
               table: RegionTable, config: Dict[str, Any],
               page_sizes: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, Any]:
    cfgt = config.get('group_iou_thresholds') or DEFAULT_CONFIG['group_iou_thresholds']
    lane_thr = config.get('lane_child_thresholds') or DEFAULT_CONFIG['lane_child_thresholds']

    if getattr(table, 'data_error', None):
        return {'page_id': page_id, 'coverage': table.coverage, 'status': 'data_error',
                'data_error': table.data_error, 'counts': {'predictions': 0,
                                                          'valid_predictions': 0,
                                                          'annotations': 0,
                                                          'invalid_predictions': 0,
                                                          'degraded_predictions': 0,
                                                          'unlocatable_predictions': 0},
                'issue_summary': {'invalid_count': 0, 'degraded_count': 0,
                                  'unlocatable_count': 0, 'by_issue': {},
                                  'degraded_ids': [], 'policy': 'data_error'},
                'invalid': [], 'thresholds': {}, 'page_geometry': {'gt_boxes': [],
                                                                  'pred_boxes': []}}

    valid, invalid = normalize_predictions([p for p in preds if p.get('page_id') == page_id],
                                           page_sizes)
    page_gts = [g for g in gts if g.get('page_id') == page_id]
    valid.sort(key=lambda p: _sort_key(p, '_bbox', 'prediction_id'))
    page_gts = sorted(page_gts, key=lambda g: _sort_key(g, 'group_bbox', 'group_id'))

    gt_boxes = [g['group_bbox'] for g in page_gts]
    pred_boxes = [p['_bbox'] for p in valid]
    matrix = [[iou(pb, gb) for gb in gt_boxes] for pb in pred_boxes]

    thresholds_out: Dict[str, Any] = {}
    for thr in cfgt:
        thresholds_out[str(thr)] = _match(matrix, thr, pred_boxes, gt_boxes, valid, page_gts,
                                          lane_thr, table, page_id, config)

    issues = summarize_issues(valid, invalid)
    return {
        'page_id': page_id,
        'coverage': table.coverage,
        'counts': {'predictions': len(valid) + len(invalid), 'valid_predictions': len(valid),
                   'annotations': len(page_gts), 'invalid_predictions': len(invalid),
                   'degraded_predictions': issues['degraded_count'],
                   'unlocatable_predictions': issues['unlocatable_count']},
        'issue_summary': issues,
        'invalid': [{'prediction_id': p.get('prediction_id'), 'page_id': page_id,
                     'source': p.get('source'), 'run_status': p.get('run_status'),
                     'error': p.get('error'), 'issues': p['_issues'],
                     'group_bbox': p.get('group_bbox'),
                     'locatable': p.get('group_bbox') is not None}
                    for p in invalid],
        'thresholds': thresholds_out,
        'page_geometry': {'gt_boxes': gt_boxes, 'pred_boxes': pred_boxes},
    }


def config_diag(config: Dict[str, Any]) -> Dict[str, float]:
    d = config.get('diagnostics') or {}
    return {'fragment_union_coverage': d.get('fragment_union_coverage',
                                             d.get('fragment', {}).get('union_coverage_of_gt_min', 0.5)),
            'merge_containment': d.get('merge_containment',
                                       d.get('merge', {}).get('containment_of_gt_min', 0.6))}


def _match(matrix, thr, pred_boxes, gt_boxes, valid, page_gts, lane_thr, table, page_id,
           config=None):
    config = config or {}
    limits = (config.get('matching') or {}).get('lex_search_limits')
    exclude = config.get('lane_eligible_exclude_certainty')
    if exclude is None:
        exclude = DEFAULT_CONFIG['lane_eligible_exclude_certainty']
    from .matching import match_groups

    # 匹配前归属：ignore 排除；ambiguous 只在够不到正例时暂缓
    pre = [table.precheck(b) for b in pred_boxes]
    ignored_idx = [i for i, x in enumerate(pre) if x['in_ignore']]
    ambiguous_idx = [i for i, x in enumerate(pre) if x['in_ambiguous']]

    def best_iou(i: int) -> float:
        return max([matrix[i][j] for j in range(len(gt_boxes))], default=0.0)

    # ambiguous 预测若能达到某个正例阈值，仍参与匹配（协议：正例优先于歧义掩膜）
    excluded_by_ambiguity = [i for i in ambiguous_idx if best_iou(i) < thr]
    cand_idx = [i for i in range(len(valid))
                if i not in set(ignored_idx) and i not in set(excluded_by_ambiguity)]

    sub_matrix = [[matrix[i][j] for j in range(len(gt_boxes))] for i in cand_idx]
    m = match_groups(sub_matrix, thr, limits=limits) if cand_idx and gt_boxes else {
        'cardinality': 0, 'total_iou': 0.0, 'pairs': [], 'method': 'no_candidates',
        'tie_break_status': 'no_match', 'tie_break_applied': False}
    pairs = [(cand_idx[int(a)], int(b)) for a, b in m['pairs']]
    pairs.sort()
    matched_pred = {i: j for i, j in pairs}
    matched_gt = {j: i for i, j in pairs}

    diag = attach_diagnostics(matrix, thr, pred_boxes, gt_boxes, pairs,
                              thresholds=config_diag(config))
    pred_labels = diag['pred_labels']
    gt_labels = diag['gt_labels']

    # 轨级
    lane_rows, excl = [], []
    for i, j in pairs:
        gt = page_gts[j]
        if not lane_eligible(gt, exclude):
            excl.append({'prediction_id': valid[i].get('prediction_id'), 'group_id': gt.get('group_id'),
                         'certainty': gt.get('certainty'),
                         'reason': 'certainty 属 ambiguous_lane_*，不进轨级分母'})
            lane_rows.append({'prediction_id': valid[i].get('prediction_id'),
                              'group_id': gt.get('group_id'), 'group_iou': matrix[i][j],
                              'category': 'excluded_ambiguous', 'eligible': False})
            continue
        sc = lane_scores(valid[i], gt)
        cls = classify_lane_pairing(sc, lane_thr['right_iou'], lane_thr['left_iou'])
        lane_rows.append({'prediction_id': valid[i].get('prediction_id'),
                          'group_id': gt.get('group_id'), 'group_iou': matrix[i][j],
                          'right_iou_fixed': sc['right_iou_fixed'],
                          'left_iou_fixed': sc['left_iou_fixed'],
                          'right_pred_vs_gt_left_iou': sc['right_pred_vs_gt_left_iou'],
                          'left_pred_vs_gt_right_iou': sc['left_pred_vs_gt_right_iou'],
                          'category': cls['category'], 'reason': cls['reason'], 'eligible': True})

    # 重复簇里「已在别处匹配」的多余项：只记诊断，不再计 FP（协议 A3）
    already_elsewhere = {idx for d in diag['duplicates']
                         for idx in d['extra_preds_already_matched_elsewhere']}

    # 逐预测定基础状态（每个预测恰好一个）
    rows = []
    for i, p in enumerate(valid):
        matched = i in matched_pred
        j = matched_pred.get(i)
        attr = table.attribute(pred_boxes[i],
                              matched_gt=(page_gts[j]['group_id'] if matched else None),
                              matched=matched)
        labels = pred_labels.get(str(i), [])
        base = attr['base_status']
        reason = attr['reason']
        # 兜底：如果归属说 TP 但实际没匹配上（阈值/过滤造成），以匹配结果为准
        if matched and base != 'TP':
            base = 'TP'
        # 重复簇的多余项：即使处在 partial 区，也按协议计 1 个 FP（单列，便于区分）
        if (not matched and 'duplicate_prediction' in labels and i not in already_elsewhere
                and base == 'PENDING_CANDIDATE'):
            base = 'FP'
            reason = (reason + '；同时属于重复簇的多余项，按 partial_area_duplicates 计 1 个 FP')
        rows.append({
            'prediction_id': p.get('prediction_id'), 'page_id': page_id,
            'group_bbox': pred_boxes[i], 'right_bbox': p.get('_right'),
            'left_bbox': p.get('_left'),
            'source': p.get('source'), 'run_status': p.get('run_status'),
            'matched': matched, 'matched_group_id': page_gts[j]['group_id'] if matched else None,
            'group_iou': matrix[i][j] if matched else None,
            'base_status': base,
            'label': base,
            'counts_as_fp': base == 'FP',
            'diagnostic_labels': labels,
            'review_category': attr['review_category'], 'review_reason': reason,
            'negative_region_ids': attr['negative_region_ids'],
            'negative_coverage_ratio': attr['negative_coverage_ratio'],
            'overlaps_negative_region': attr['overlaps_negative_region'],
            'annotation_coverage': attr['annotation_coverage'],
            'evaluation_region_id': attr.get('evaluation_region_id'),
        })

    gt_rows = []
    for j, g in enumerate(page_gts):
        gt_rows.append({'group_id': g.get('group_id'), 'matched': j in matched_gt,
                        'matched_prediction_id': valid[matched_gt[j]].get('prediction_id') if j in matched_gt else None,
                        'diagnostic_labels': gt_labels.get(str(j), []),
                        'certainty': g.get('certainty'),
                        'lane_eligible': lane_eligible(g, exclude)})

    eligible_lane = [r for r in lane_rows if r['eligible']]
    lane_counts = {}
    for r in eligible_lane:
        lane_counts[r['category']] = lane_counts.get(r['category'], 0) + 1

    by_status = {s: sum(1 for r in rows if r['base_status'] == s) for s in BASE_STATUSES}
    dup_flip = [r['prediction_id'] for r in rows
                if 'duplicate_prediction' in r['diagnostic_labels'] and r['base_status'] == 'TP']

    return {
        'threshold': thr,
        'cardinality': m['cardinality'],
        'total_iou': m['total_iou'],
        'solver': m.get('method'),
        'tie_break_applied': m.get('tie_break_applied', False),
        'tie_break_status': m.get('tie_break_status'),
        'tie_break_scope': m.get('tie_break_scope'),
        'pairs': [[valid[i].get('prediction_id'), page_gts[j].get('group_id'),
                    round(matrix[i][j], 6)] for i, j in pairs],
        'group_hits': m['cardinality'],
        'annotations_total': len(page_gts),
        'tp': by_status['TP'],
        'fp': by_status['FP'],
        'fn': len(page_gts) - m['cardinality'],
        'ignored': by_status['IGNORED'],
        'pending_ambiguous': by_status['PENDING_AMBIGUOUS'],
        'new_candidates': by_status['PENDING_CANDIDATE'],
        'priority_positive_kept': len([i for i in ambiguous_idx if i not in set(excluded_by_ambiguity)]),
        'status_sum': sum(by_status.values()),
        'status_partition_ok': sum(by_status.values()) == len(valid),
        'by_status': by_status,
        'duplicate_extra_fp': sum(1 for r in rows if r['base_status'] == 'FP'
                                   and 'duplicate_prediction' in r['diagnostic_labels']),
        'duplicate_without_flipping_tp': dup_flip,
        'negative_confirmed_fp': sum(1 for r in rows
                                      if r['review_category'] == 'confirmed_false_positive'),
        'negative_suspected_fp': sum(1 for r in rows
                                      if r['review_category'] == 'suspected_false_positive'),
        'boundary_conflicts': sum(1 for r in rows if r['review_category'] == 'boundary_conflict'),
        'boundary_conflict_fp': sum(1 for r in rows if r['review_category'] == 'boundary_conflict'
                                     and r['base_status'] == 'FP'),
        'invalid_predictions': len([p for p in valid if p['_issues']]),
        'lane': {'rows': lane_rows, 'counts': lane_counts,
                 'denominator': len(eligible_lane),
                 'excluded': excl,
                 'excluded_count': len(excl),
                 'pairing_accuracy': (lane_counts.get('pairing_success', 0) / len(eligible_lane)
                                      if eligible_lane else None),
                 'swap_rate': (lane_counts.get('left_right_swap', 0) / len(eligible_lane)
                               if eligible_lane else None),
                 'child_failure_rate': (lane_counts.get('child_localization_failure', 0) / len(eligible_lane)
                                        if eligible_lane else None)},
        'diagnostics': {k: v for k, v in diag.items() if k not in ('pred_labels', 'gt_labels')},
        'predictions': rows,
        'annotations': gt_rows,
    }


# ---------------------------------------------------------------- 汇总

def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 6) if den else None


def score_predictions(predictions: Sequence[Dict[str, Any]],
                      annotations: Sequence[Dict[str, Any]],
                      region_tables: Dict[str, RegionTable],
                      config: Optional[Dict[str, Any]] = None,
                      pages: Optional[Sequence[str]] = None,
                      page_split: Optional[Dict[str, str]] = None,
                      page_sizes: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, Any]:
    """对多页打分并汇总。pages 决定要评哪些页（默认取预测与标注的并集）。"""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    page_ids = list(pages) if pages else sorted(
        {p.get('page_id') for p in predictions} | {g.get('page_id') for g in annotations})
    thr_keys = [str(t) for t in cfg['group_iou_thresholds']]

    per_page, data_errors = [], []
    for pid in page_ids:
        table = region_tables.get(pid) or RegionTable(pid)
        page = score_page(pid, predictions, annotations, table, cfg, page_sizes)
        if page.get('status') == 'data_error':
            data_errors.append({'page_id': pid, 'error': page['data_error']})
        per_page.append(page)

    invalid_rows = [row for p in per_page for row in p['invalid']]
    invalid_by_issue: Dict[str, int] = {}
    degraded_total, unlocatable_total = 0, 0
    for page in per_page:
        for issue, n in page['issue_summary']['by_issue'].items():
            invalid_by_issue[issue] = invalid_by_issue.get(issue, 0) + n
        degraded_total += page['issue_summary']['degraded_count']
        unlocatable_total += page['issue_summary']['unlocatable_count']

    summary: Dict[str, Any] = {'thresholds': thr_keys, 'overall': {}, 'by_split': {}}
    for tk in thr_keys:
        agg = {k: 0 for k in ('tp', 'fp', 'fn', 'group_hits', 'annotations_total',
                              'new_candidates', 'duplicate_extra_fp',
                              'negative_confirmed_fp', 'negative_suspected_fp',
                              'boundary_conflicts', 'boundary_conflict_fp',
                              'ignored', 'pending_ambiguous')}
        lane_acc: Dict[str, int] = {}
        status_acc: Dict[str, int] = {s: 0 for s in BASE_STATUSES}
        partition_ok = True
        for page in per_page:
            t = page['thresholds'].get(tk)
            if not t:
                continue
            for k in agg:
                agg[k] += t[k]
            for k, v in t['lane']['counts'].items():
                lane_acc[k] = lane_acc.get(k, 0) + v
            for k, v in t['by_status'].items():
                status_acc[k] = status_acc.get(k, 0) + v
            partition_ok = partition_ok and t['status_partition_ok']
        lane_den = sum(lane_acc.values())
        agg['lane_counts'] = lane_acc
        agg['lane_denominator'] = lane_den
        agg['lane_excluded'] = sum(len(p['thresholds'][tk]['lane']['excluded'])
                                   for p in per_page if tk in p['thresholds'])
        agg['pairing_accuracy'] = _rate(lane_acc.get('pairing_success', 0), lane_den)
        agg['swap_rate'] = _rate(lane_acc.get('left_right_swap', 0), lane_den)
        agg['child_failure_rate'] = _rate(lane_acc.get('child_localization_failure', 0), lane_den)
        agg['hit_rate_on_known'] = _rate(agg['group_hits'], agg['annotations_total'])
        agg['recall'] = 'not_computable'      # 无 exhaustive 区域
        agg['precision'] = 'not_computable'
        agg['invalid_predictions'] = len(invalid_rows)
        agg['degraded_predictions'] = degraded_total
        agg['invalid_by_issue'] = invalid_by_issue
        agg['status_partition'] = status_acc
        agg['status_partition_ok'] = partition_ok
        agg['status_sum'] = sum(status_acc.values())
        summary['overall'][tk] = agg

    if page_split:
        for split in sorted(set(page_split.values())):
            pages_in = [p['page_id'] for p in per_page if page_split.get(p['page_id']) == split]
            sub: Dict[str, Any] = {'pages': pages_in, 'thresholds': {}}
            for tk in thr_keys:
                rows = [p['thresholds'][tk] for p in per_page
                    if p['page_id'] in pages_in and tk in p['thresholds']]
                hits = sum(r['group_hits'] for r in rows)
                total = sum(r['annotations_total'] for r in rows)
                lane_d = sum(r['lane']['denominator'] for r in rows)
                ok = sum(r['lane']['counts'].get('pairing_success', 0) for r in rows)
                sub['thresholds'][tk] = {
                    'pages': len(pages_in), 'group_hits': hits, 'annotations_total': total,
                    'hit_rate_on_known': _rate(hits, total),
                    'lane_denominator': lane_d,
                    'pairing_accuracy': _rate(ok, lane_d)}
            summary['by_split'][split] = sub

    summary['scope'] = {
        'pages_evaluated': len(per_page),
        'pages_scorable': len(per_page) - len({d['page_id'] for d in data_errors}),
        'pages_excluded': len({d['page_id'] for d in data_errors}),
        'excluded_pages': sorted({d['page_id'] for d in data_errors}),
        'note': '所有指标只在可评分页上统计；被排除的页在 data_errors 里给出原因',
    }
    if data_errors:
        affected = sorted({d['page_id'] for d in data_errors})
        for tk in thr_keys:
            summary['overall'][tk]['data_error_pages'] = affected
            summary['overall'][tk]['data_error_page_count'] = len(affected)
            summary['overall'][tk]['scorable_page_count'] = len(per_page) - len(affected)
            summary['overall'][tk]['status'] = 'not_computable_partial'
            summary['overall'][tk]['scope_note'] = (
                f'{len(affected)} 页因数据错误无法计分：'
                f'{", ".join(affected)}；其余 {len(per_page) - len(affected)} 页的指标正常，'
                f'但总体数字不含这些页，不得当成全量结果')
    return {'config': cfg, 'pages': per_page, 'summary': summary,
            'data_errors': data_errors,
            'invalid': {'count': len(invalid_rows), 'degraded': degraded_total,
                        'unlocatable': unlocatable_total, 'by_issue': invalid_by_issue,
                        'policy': 'base_status=INVALID（无有效组框）不参与匹配，'
                                  '计入 invalid 统计但不进 TP/FP；'
                                  'degraded（缺轨/越界，组框仍可用）参与匹配但轨级判失败；'
                                  'unlocatable 单列，穷尽页上的非法输出按协议并入错误统计'},
            'metric_limits': {
                'recall': 'not_computable：当前无 exhaustive_region/exhaustive_page',
                'precision': 'not_computable：同上，且负例只覆盖明确区域',
                'false_group_count': 'not_computable：部分标注区内未匹配预测只作候选',
                'note': '可报告：已标位置命中率、IoU 0.5/0.75 独立匹配、轨级分类分布、'
                        '确定负例区域内误检计数、无效预测分类计数'}}


def endpoint_sensitivity(predictions: Sequence[Dict[str, Any]],
                        annotations: Sequence[Dict[str, Any]],
                        region_tables: Dict[str, RegionTable],
                        config: Optional[Dict[str, Any]] = None,
                        pages: Optional[Sequence[str]] = None,
                        shift_px: int = 1,
                        page_sizes: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, Any]:
    """对 endpoint 待核的标注分别按 nominal / inclusive 各算一遍，报差值。"""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    nominal = score_predictions(predictions, endpoint_variant(annotations, 'nominal', shift_px),
                               region_tables, cfg, pages, page_sizes=page_sizes)
    inclusive = score_predictions(predictions, endpoint_variant(annotations, 'inclusive', shift_px),
                                 region_tables, cfg, pages, page_sizes=page_sizes)
    pending_groups = [g['group_id'] for g in annotations
                      if str((g.get('endpoint_interpretation') or {}).get('status')) == 'pending']
    # 两种解释必须在**同一个可评分页集合**上比较，否则差异里混进了「页数不同」
    nom_excl = set((nominal.get('summary') or {}).get('scope', {}).get('excluded_pages') or [])
    inc_excl = set((inclusive.get('summary') or {}).get('scope', {}).get('excluded_pages') or [])
    scorable = sorted({p['page_id'] for p in nominal['pages']}
                      - nom_excl - inc_excl)
    diffs: Dict[str, Any] = {'shift_px': shift_px, 'pending_groups': len(pending_groups),
                             'body_report_uses': 'nominal',
                             'interpretation': 'nominal 与 inclusive 是**未决**的两种端点解释，'
                                               '当前证据不足以判定哪个是来源侧的真实约定',
                             'scope': {'pages_scorable': len(scorable),
                                       'scorable_pages': scorable,
                                       'excluded_pages': sorted(nom_excl | inc_excl),
                                       'excluded_count': len(nom_excl | inc_excl),
                                       'note': 'nominal / inclusive / 总 IoU / 配对差异'
                                               '都只在上述可评分页集合上计算'},
                             'by_threshold': {}}
    for tk in nominal['summary']['thresholds']:
        n = nominal['summary']['overall'][tk]
        c = inclusive['summary']['overall'][tk]
        diffs['by_threshold'][tk] = {
            'pages': len(scorable),
            'group_hits': {'nominal': n['group_hits'], 'inclusive': c['group_hits'],
                           'delta': c['group_hits'] - n['group_hits']},
            'total_iou': {'nominal': round(_sum_iou(nominal, tk), 6),
                          'inclusive': round(_sum_iou(inclusive, tk), 6)},
            'lane_counts': {'nominal': n['lane_counts'], 'inclusive': c['lane_counts']},
            'hit_rate_on_known': {'nominal': n['hit_rate_on_known'],
                                  'inclusive': c['hit_rate_on_known']},
        }
    pair_delta = []
    scorable_set = set(scorable)
    for pn, pi in zip(nominal['pages'], inclusive['pages']):
        if pn['page_id'] not in scorable_set:
            continue
        for tk in nominal['summary']['thresholds']:
            if tk not in (pn.get('thresholds') or {}) or tk not in (pi.get('thresholds') or {}):
                continue
            a = {k: v for k, _, v in pn['thresholds'][tk]['pairs']}
            b = {k: v for k, _, v in pi['thresholds'][tk]['pairs']}
            for gid in sorted(set(a) & set(b)):
                if a[gid] != b[gid]:
                    pair_delta.append({'page': pn['page_id'], 'threshold': float(tk),
                                       'group_id': gid, 'iou_nominal': a[gid],
                                       'iou_inclusive': b[gid]})
    diffs['changed_pairs'] = pair_delta
    return {'nominal': nominal['summary'], 'inclusive': inclusive['summary'], 'delta': diffs}


def _scorable_pages(result: Dict[str, Any], tk: str) -> List[Dict[str, Any]]:
    """只取在阈值 tk 下真有结果的页；data_error 页的 thresholds 是空的。"""
    return [p for p in result['pages'] if tk in (p.get('thresholds') or {})]


def _sum_iou(result: Dict[str, Any], tk: str) -> float:
    return float(sum(p['thresholds'][tk]['total_iou'] for p in _scorable_pages(result, tk)))
