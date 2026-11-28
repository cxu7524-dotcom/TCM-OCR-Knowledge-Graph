# -*- coding: utf-8 -*-
"""区域归属与负例关联。

归属优先级（协议 coverage_reporting.prediction_attribution）：
  ignore 区域 → ambiguous 区域 → 正例匹配 → 负例区域 → 未覆盖区域

**归属必须在匹配之前做第一步**：被 ignore 排除的预测不能占用标注节点，
否则同一个框会既算 TP 又算 IGNORED（旧实现的缺陷）。

负例覆盖比例 = area(pred ∩ union(负例区域)) / area(pred)，先求并集再算交集，
结果落在 [0,1]。阈值当前未标定，provisional 值与原值一起输出。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import (bbox_area, center, contains_point, intersect_area,
                       union_ratio)

PROVISIONAL = {
    'boundary_conflict_max': 0.15,   # 仅边缘擦到
    'suspected_fp_min': 0.50,        # 大面积落入负例
    'confirmed_fp_min': 0.80,        # 几乎整块落在确定负例里
    'status': 'pending_calibration',
}

REVIEW_CATEGORIES = (
    'ignored',
    'pending_review_ambiguous',
    'matched_positive',
    'boundary_conflict',
    'confirmed_false_positive',
    'suspected_false_positive',
    'new_candidate_in_partial',
    'unmatched_uncovered',
    'unmatched_outside_region',
)

# 每个预测最终只有一个基础计分状态；诊断标签另存，不改状态
BASE_STATUSES = ('TP', 'FP', 'IGNORED', 'PENDING_AMBIGUOUS', 'PENDING_CANDIDATE', 'INVALID')

EXHAUSTIVE = ('exhaustive_region', 'exhaustive_page')


def _valid_box(b: Any) -> bool:
    if not b or not isinstance(b, (list, tuple)) or len(b) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in b]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


class RegionTable:
    """一页的区域集合；负例与掩膜都按原图像素坐标。"""

    def __init__(self, page_id: str, negatives: Sequence[Dict[str, Any]] = (),
                 ambiguous: Sequence[Dict[str, Any]] = (),
                 ignores: Sequence[Dict[str, Any]] = (),
                 coverage: str = 'partial_positions',
                 thresholds: Optional[Dict[str, float]] = None,
                 evaluation_regions: Optional[Sequence[Dict[str, Any]]] = None):
        self.page_id = page_id
        self.negatives = [r for r in negatives if r.get('page_id') == page_id]
        self.ambiguous = [r for r in ambiguous if r.get('page_id') == page_id]
        self.ignores = [r for r in ignores if r.get('page_id') == page_id]
        self.coverage = coverage
        self.thr = dict(PROVISIONAL, **(thresholds or {}))
        # 局部穷尽区域：每个区域自带原图 bbox 与自己的 coverage。
        # 区域之外的预测不能按穷尽规则算，否则等于把没标注的地方当穷尽了。
        #
        # page_id 的处理分三种：没给 → 继承父页；给了且相同 → 正常；
        # 给了且与父页冲突 → 记数据错误（不能静默丢弃，那会让「有区域」变成「没区域」）。
        self.evaluation_regions = []
        conflicts = []
        for r in evaluation_regions or []:
            rp = r.get('page_id')
            if rp is not None and str(rp) != str(page_id):
                conflicts.append({'region_id': r.get('region_id'), 'region_page_id': rp,
                                  'parent_page_id': page_id})
                continue
            self.evaluation_regions.append(dict(r, page_id=page_id))
        self.data_error = None
        self.data_errors: List[str] = []
        bad = [r for r in self.evaluation_regions if not _valid_box(r.get('bbox'))]
        if conflicts:
            detail = '；'.join(f"{c['region_id']} 的 page_id={c['region_page_id']!r} "
                              f"与父页 {c['parent_page_id']!r} 冲突" for c in conflicts)
            self.data_error = f'{len(conflicts)} 个评测区域的 page_id 与父页冲突：{detail}'
        elif coverage == 'exhaustive_region' and not self.evaluation_regions:
            self.data_error = ('coverage=exhaustive_region 缺少评测区域坐标：'
                               '无法判断哪些预测落在穷尽范围内，本页不可计分')
        elif bad:
            self.data_error = (f"{len(bad)} 个评测区域缺有效 bbox："
                               f"{[r.get('region_id') for r in bad]}")
        if self.data_error:
            self.data_errors = [self.data_error]

    @property
    def exhaustive(self) -> bool:
        return self.coverage in EXHAUSTIVE

    def region_at(self, box: Sequence[int]) -> Optional[Dict[str, Any]]:
        """按预测组框中心落在哪个评测区域；都不在返回 None。"""
        cx, cy = center(box)
        for r in self.evaluation_regions:
            b = r.get('bbox')
            if _valid_box(b) and contains_point(b, cx, cy):
                return r
        return None

    def coverage_at(self, box: Sequence[int]) -> Tuple[str, Optional[str]]:
        """返回 (生效 coverage, region_id)。区域自带 coverage 覆盖页级设置。"""
        r = self.region_at(box)
        if r is not None:
            return (str(r.get('coverage') or self.coverage), r.get('region_id'))
        if self.evaluation_regions:
            # 有局部穷尽区域时，区域之外不能沿用页级的 exhaustive 口径，
            # 否则等于把没标注的地方当穷尽——那里只能算「未覆盖」。
            return 'partial_positions', None
        return self.coverage, None

    # ---------------------------------------------------------------- 查询
    def _hit(self, regions: Iterable[Dict[str, Any]], box: Sequence[int]) -> Optional[Dict[str, Any]]:
        cx, cy = center(box)
        for r in regions:
            b = r.get('bbox')
            if not b:
                continue
            if contains_point(b, cx, cy) or intersect_area(box, b) / max(1, bbox_area(box)) >= 0.5:
                return r
        return None

    def confirmed_negatives(self) -> List[Dict[str, Any]]:
        return [r for r in self.negatives if str(r.get('status')) == 'confirmed']

    def unknown_negatives(self) -> List[Dict[str, Any]]:
        return [r for r in self.negatives if str(r.get('status')) != 'confirmed']

    def negative_coverage(self, box: Sequence[int], only_confirmed: bool = True) -> Dict[str, Any]:
        pool = self.confirmed_negatives() if only_confirmed else self.negatives
        boxes = [r['bbox'] for r in pool if r.get('bbox')]
        ratio = union_ratio(box, boxes)
        hit_ids = [r.get('negative_region_id') for r in pool
                   if r.get('bbox') and intersect_area(box, r['bbox']) > 0]
        cats = sorted({str(r.get('category')) for r in pool
                       if r.get('bbox') and intersect_area(box, r['bbox']) > 0})
        return {'ratio': round(ratio, 6), 'region_ids': hit_ids, 'categories': cats,
                'union_first': True, 'pool': 'confirmed' if only_confirmed else 'all'}

    # ---------------------------------------------------------------- 匹配前过滤
    def precheck(self, box: Sequence[int]) -> Dict[str, Any]:
        """匹配之前的第一步归属：决定这个预测能不能进入匹配候选。

        - ignore：无条件排除（它按定义就是不参与计分的区域）。
        - ambiguous：**先不排除**，交由调用方判断「能否达到任何正例的阈值」；
          协议要求「确定正例不得因宽泛歧义掩膜被排除」，所以这里只报"踩到了"。
        """
        out = {'excluded_by': None, 'ignore_region_id': None, 'ambiguous_region_id': None,
               'in_ignore': False, 'in_ambiguous': False}
        ign = self._hit(self.ignores, box)
        if ign:
            out.update(excluded_by='ignore', in_ignore=True,
                       ignore_region_id=ign.get('ignore_region_id'))
            return out
        amb = self._hit(self.ambiguous, box)
        if amb:
            out.update(in_ambiguous=True,
                       ambiguous_region_id=amb.get('ambiguous_region_id'))
        return out

    # ---------------------------------------------------------------- 归属
    def attribute(self, box: Sequence[int], matched_gt: Optional[str] = None,
                  matched: bool = False) -> Dict[str, Any]:
        """返回 {review_category, base_status, counts_as_fp, reason, ...}。

        每个预测只得到一个 base_status；标签（boundary_conflict / duplicate 等）
        可以叠加，但不改变 base_status。
        """
        cov_eff, region_id = self.coverage_at(box)
        out: Dict[str, Any] = {
            'page_id': self.page_id,
            'pred_bbox': list(box),
            'annotation_coverage': cov_eff,
            'evaluation_region_id': region_id,
            'negative_region_ids': [], 'overlaps_negative_region': False,
            'negative_coverage_ratio': 0.0, 'negative_categories': [],
            'review_category': None, 'base_status': None, 'counts_as_fp': False,
            'reason': '', 'thresholds': self.thr,
        }
        pre = self.precheck(box)
        if pre['in_ignore']:
            out.update(review_category='ignored', base_status='IGNORED',
                       ignore_region_id=pre['ignore_region_id'],
                       reason='落在 ignore 区域：不参与匹配、不计 TP/FP')
            return out
        if pre['in_ambiguous'] and not matched:
            out.update(review_category='pending_review_ambiguous', base_status='PENDING_AMBIGUOUS',
                       ambiguous_region_id=pre['ambiguous_region_id'],
                       reason='落在 ambiguous 区域且未匹配到任何正例：不判对错，进待人工复核')
            return out

        cov = self.negative_coverage(box, only_confirmed=True)
        out.update(negative_region_ids=cov['region_ids'],
                   overlaps_negative_region=bool(cov['region_ids']),
                   negative_coverage_ratio=cov['ratio'],
                   negative_categories=cov['categories'])
        ratio = cov['ratio']

        if matched:
            out['matched_gt'] = matched_gt
            if ratio > 0:
                out.update(review_category='boundary_conflict', base_status='TP',
                           reason=f"已匹配正例，但与负例重叠 {ratio:.3f}——保留冲突记录，不改判")
            else:
                out.update(review_category='matched_positive', base_status='TP',
                           reason='全局匹配到正例组')
            return out

        if cov['region_ids'] and ratio >= self.thr['confirmed_fp_min']:
            out.update(review_category='confirmed_false_positive', base_status='FP',
                       counts_as_fp=True,
                       reason=f"确定负例覆盖 {ratio:.3f} ≥ {self.thr['confirmed_fp_min']}")
        elif cov['region_ids'] and ratio >= self.thr['suspected_fp_min']:
            out.update(review_category='suspected_false_positive', base_status='FP',
                       counts_as_fp=True,
                       reason=f"负例覆盖 {ratio:.3f} ≥ {self.thr['suspected_fp_min']}（阈值 provisional）")
        elif cov_eff in EXHAUSTIVE:
            # 穷尽区域：未匹配就是误检。边缘相交只加诊断，不免除 FP。
            if cov['region_ids'] and ratio > 0:
                out.update(review_category='boundary_conflict', base_status='FP',
                           counts_as_fp=True,
                           reason=f"穷尽区域内未匹配；虽仅边缘相交 {ratio:.3f}，仍计 1 个误检"
                                  f"（boundary_conflict 是诊断，不是豁免）")
            else:
                out.update(review_category='unmatched_uncovered', base_status='FP',
                           counts_as_fp=True, reason='穷尽区域内未匹配且不触负例：计误组')
        elif self.evaluation_regions and region_id is None:
            # 有局部穷尽区域、但该预测不在任何区域内：不能按穷尽规则算
            out.update(review_category='unmatched_outside_region',
                       base_status='PENDING_CANDIDATE', counts_as_fp=False,
                       reason='落在所有评测区域之外：属于未覆盖位置，'
                              '不计 FP（要计误组必须先把它纳入某个穷尽区域）')
        else:
            out.update(review_category='new_candidate_in_partial', base_status='PENDING_CANDIDATE',
                       counts_as_fp=False,
                       reason=('部分标注区内未匹配；与负例边缘相交 '
                               f"{ratio:.3f}（< {self.thr['suspected_fp_min']}），按候选待核"
                               if cov['region_ids'] else
                               '部分标注区域内未匹配且未落入确定负例：新增候选待核'))
        return out


def attribute_prediction(table: RegionTable, box: Sequence[int],
                         matched_gt: Optional[str] = None,
                         matched: bool = False) -> Dict[str, Any]:
    return table.attribute(box, matched_gt=matched_gt, matched=matched)


def negative_coverage_ratio(box: Sequence[int], regions: Sequence[Sequence[int]]) -> float:
    """便捷函数：直接给框与负例框列表。"""
    return round(union_ratio(box, regions), 6)
