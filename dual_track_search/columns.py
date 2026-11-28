# -*- coding: utf-8 -*-
"""路径二：校验已有列框（validated_columns），让校验**真正决定路径**。

旧布局（page_layout）给出的列框不是都能用。这里逐条校验并给出可执行结论：

    accept    直接作为搜索区域
    split     框内有一条经确认的独立栏线 → 在栏线处拆成多段
    fallback  拒绝，交低置信局部窗口，并记下拒绝依据

判据与 v1 的区别（v1 大多只写诊断、不影响路径）：
  主体重叠  adjacent boxes 横向重叠 / 较窄列宽 > 0.15 → 双方都降级为 fallback
  邻栏间距  gap < 0.12×参考宽 → 记 gutter_tight；gap < 0 已属重叠
  半页边界  框体越过 half 边界 > 2px → fallback（典型是跨书脊的宽框）
  栏线冲突  框内独立栏线 → 拆分；拆出的段太窄 → fallback
  宽度参考  同书 → 同页 → 全库 →（都没有）低置信 fallback，绝不用默认值顶替
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ColumnVerdict:
    col_index: int
    bbox: List[int]                 # 原图 [x1,y1,x2,y2)
    width: int
    decision: str = 'accept'        # accept / split / fallback
    reasons: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision in ('accept', 'split')

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['reasons'] = list(self.reasons)
        d['accepted'] = self.accepted
        return d


def _median_width(columns: Sequence[Dict[str, Any]]) -> Optional[float]:
    ws = [int(c['bbox'][2]) - int(c['bbox'][0]) for c in columns]
    ws = [w for w in ws if w > 0]
    return float(np.median(ws)) if ws else None


def _reference_width(book_stats: Optional[Dict[str, Any]],
                     page_stats: Optional[Dict[str, Any]],
                     global_stats: Optional[Dict[str, Any]],
                     cfg: Dict[str, Any]) -> Tuple[Optional[float], str, List[str]]:
    """参考宽度链：同书 → 同页 → 全库 → 无。每一步都留来源与理由。"""
    notes: List[str] = []
    if book_stats and (book_stats.get('cols') or 0) >= cfg['min_reference_cols']:
        med = book_stats.get('median_width')
        if med:
            return float(med), 'same_book', notes
        notes.append('同书统计缺中位数')
    else:
        notes.append(f"同书列数 {book_stats.get('cols') if book_stats else 0} "
                     f"< {cfg['min_reference_cols']}，改用替代参考")
    if page_stats and (page_stats.get('cols') or 0) >= cfg.get('page_reference_min_cols', 5) \
            and page_stats.get('median_width'):
        notes.append(f"同页 {page_stats['cols']} 列，中位宽 "
                     f"{page_stats['median_width']:.1f}px 作替代参考")
        return float(page_stats['median_width']), 'same_page_median', notes
    notes.append(f"同页列数 {page_stats.get('cols') if page_stats else 0} "
                 f"< {cfg.get('page_reference_min_cols', 5)}，不能作替代参考")
    if cfg.get('global_substitute') and global_stats and global_stats.get('median_width'):
        notes.append('改用全库替代参考')
        return float(global_stats['median_width']), 'global_substitute', notes
    notes.append('无可用参考（同书/同页/全库都不足），宽度无法判定 → 低置信回退')
    return None, 'none', notes


def _inside_rules(x1: int, x2: int, rules: Sequence[Dict[str, Any]], margin: int
                  ) -> List[Dict[str, Any]]:
    """框内**经确认**的独立栏线（离框边至少 margin）。

    资格判据只有一份（`rules.line_eligibility`）：`qualified=False` 的线
    （纯贴字、或含贴字段且独立段不够长的混合断线）一律不参与拆分，
    只把它们的坐标记进 `rules_inside_unqualified` 供人工核对。
    """
    out = []
    for r in rules:
        xc = float(r.get('center', r.get('x', 0)))
        if not (x1 + margin < xc < x2 - margin):
            continue
        q = r.get('qualified')
        if q is False:
            continue
        if q is None and r.get('kind') not in ('independent', 'broken'):
            continue
        out.append(r)
    out.sort(key=lambda r: float(r.get('center', r.get('x', 0))))
    return out


def _unqualified_rules_inside(x1: int, x2: int, rules: Sequence[Dict[str, Any]], margin: int
                              ) -> List[Dict[str, Any]]:
    out = []
    for r in rules:
        xc = float(r.get('center', r.get('x', 0)))
        if x1 + margin < xc < x2 - margin and r.get('qualified') is False:
            out.append({'center': round(xc, 1), 'status': r.get('status'),
                        'ambiguous': r.get('ambiguous'),
                        'eligible_coverage': r.get('eligible_coverage')})
    return out


def validate_columns(columns: Sequence[Dict[str, Any]], half_range: Tuple[int, int],
                     image_size: Tuple[int, int], book_stats: Optional[Dict[str, Any]] = None,
                     global_stats: Optional[Dict[str, Any]] = None,
                     rule_xs: Sequence[float] = (), cfg: Optional[Dict[str, Any]] = None,
                     page_stats: Optional[Dict[str, Any]] = None,
                     rule_lines: Optional[Sequence[Dict[str, Any]]] = None
                     ) -> Dict[str, Any]:
    """逐条校验半页内的列框。返回 {verdicts, accepted, rejected, reference, notes}。"""
    cfg = cfg or {}
    W, H = image_size
    hx0, hx1 = int(half_range[0]), int(half_range[1])
    ref, ref_source, notes = _reference_width(book_stats, page_stats, global_stats, cfg)
    outlier_mult = float(cfg.get('outlier_multiplier', 1.8))
    thr = ref * outlier_mult if ref else None
    tol = int(cfg.get('half_boundary_tolerance_px', 2))
    margin_rule = int(cfg.get('rule_conflict_margin_px', 2))
    split_min = max(int(cfg.get('min_width_px', 8)),
                    int(ref * float(cfg.get('split_min_width_multiple', 0.35)))) if ref \
        else int(cfg.get('min_width_px', 8))
    min_w = max(int(cfg.get('min_width_px', 8)), int(ref * 0.35)) if ref \
        else int(cfg.get('min_width_px', 8))

    if rule_lines is None:
        rule_lines = [{'center': float(x), 'kind': 'independent', 'confidence': 'high'}
                      for x in rule_xs]

    in_half = []
    for i, c in enumerate(columns):
        b = [int(v) for v in c['bbox']]
        cx = (b[0] + b[2]) / 2.0
        if hx0 - 2 <= cx < hx1 + 2:
            in_half.append((i, b))

    verdicts: List[ColumnVerdict] = []
    for i, b in in_half:
        x1, y1, x2, y2 = b
        width = x2 - x1
        v = ColumnVerdict(col_index=i, bbox=b, width=width)
        half_over_l = max(0, hx0 - x1)
        half_over_r = max(0, x2 - hx1)
        v.measurements.update({
            'reference_px': ref, 'reference_source': ref_source,
            'outlier_threshold_px': round(thr, 2) if thr else None,
            'width_ratio_to_reference': round(width / ref, 3) if ref else None,
            'edge_margin_left_px': x1, 'edge_margin_right_px': W - x2,
            'half_id_bounds': [hx0, hx1],
            'half_boundary_overrun_px': [half_over_l, half_over_r],
            'height_px': y2 - y1,
        })

        # ---- 半页边界：越过 > tol 就是跨界框（跨书脊的宽框典型如此）
        if half_over_l > tol or half_over_r > tol:
            v.decision = 'fallback'
            v.reasons.append(
                f'框体越过半页边界：左出 {half_over_l}px、右出 {half_over_r}px '
                f'(半页 [{hx0},{hx1})，容差 {tol}px)，不允许跨半页配轨')
            verdicts.append(v)
            continue

        # ---- 宽度
        if thr and width > thr:
            v.decision = 'fallback'
            v.reasons.append(
                f'宽度 {width}px > {outlier_mult}×参考 {ref:.1f} = {thr:.1f}px'
                f'（{width / ref:.1f} 倍），横跨多个实体栏，拒绝直接配轨')
            verdicts.append(v)
            continue
        if width < min_w:
            v.decision = 'fallback'
            v.reasons.append(f'宽度 {width}px < 下限 {min_w}px（可能是界行或碎片）')
            verdicts.append(v)
            continue

        # ---- 参考缺失：宽度无从判定
        if ref is None:
            v.decision = 'fallback'
            v.measurements['no_reference'] = True
            v.measurements['page_width_ratio'] = round(width / W, 4)
            v.reasons.append(
                f'无宽度参考（同书/同页/全库均不足），{width}px 无法判定是否跨栏'
                f'（占页宽 {width / W:.1%}）→ 低置信回退，不当成合法列框')
            verdicts.append(v)
            continue

        # ---- 栏线冲突：框内独立栏线 → 拆分
        inside = _inside_rules(x1, x2, rule_lines, margin_rule)
        unq = _unqualified_rules_inside(x1, x2, rule_lines, margin_rule)
        if unq:
            v.measurements['rules_inside_unqualified'] = unq
            v.reasons.append(
                f'框内有 {len(unq)} 条不合格栏线（贴字或含贴字段的混合断线），'
                f'按统一资格判据不用于拆分：{[u["center"] for u in unq]}')
        if inside:
            cuts = [float(r.get('center', r.get('x', 0))) for r in inside]
            edges = [x1] + [int(round(c)) for c in cuts] + [x2]
            parts = [[edges[k], edges[k + 1]] for k in range(len(edges) - 1)]
            v.measurements['rules_inside'] = [round(c, 1) for c in cuts]
            v.measurements['split_parts_px'] = [p[1] - p[0] for p in parts]
            if all(p[1] - p[0] >= split_min for p in parts):
                v.decision = 'split'
                v.reasons.append(
                    f'框内有 {len(inside)} 条经确认的独立栏线 {[round(c, 1) for c in cuts]}，'
                    f'按栏线拆成 {len(parts)} 段（各 {[p[1] - p[0] for p in parts]}px '
                    f'≥ 下限 {split_min}px）')
                v.outputs = [{'bbox': [p[0], y1, p[1], y2], 'role': 'split_part',
                              'split_index': k} for k, p in enumerate(parts)]
            else:
                v.decision = 'fallback'
                v.reasons.append(
                    f'框内有独立栏线 {[round(c, 1) for c in cuts]}，但拆出的段宽 '
                    f'{[p[1] - p[0] for p in parts]} 有低于下限 {split_min}px，'
                    f'拆分不可靠 → 回退')
            verdicts.append(v)
            continue

        # ---- 页边只记诊断
        edge = cfg.get('edge_margin_ratio', 0.03) * W
        if x1 < edge or x2 > W - edge:
            v.measurements['edge_violation'] = True
            v.reasons.append(f'列框触及页边（x1={x1}, x2={x2}, 页边距 {edge:.0f}px）'
                             '—— 记诊断，不单独否决')
        v.outputs = [{'bbox': list(b), 'role': 'body', 'split_index': 0}]
        verdicts.append(v)

    # ---------------- 主体重叠：双方都降级（无第三方证据时不能替它们选一个）
    lim = float(cfg.get('max_overlap_ratio', 0.15))
    kept = [v for v in verdicts if v.accepted]
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            va, vb = kept[a], kept[b]
            inter = max(0, min(va.bbox[2], vb.bbox[2]) - max(va.bbox[0], vb.bbox[0]))
            if inter <= 0:
                continue
            ratio = inter / max(1, min(va.width, vb.width))
            for v, other in ((va, vb), (vb, va)):
                v.measurements.setdefault('overlaps', []).append(
                    {'peer_col_index': other.col_index, 'overlap_px': int(inter),
                     'ratio_of_narrower': round(ratio, 4)})
            if ratio > lim:
                for v, other in ((va, vb), (vb, va)):
                    v.decision = 'fallback'
                    v.outputs = []
                    v.reasons.append(
                        f'与列框 #{other.col_index} {other.bbox} 主体横向重叠 '
                        f'{inter}px（占较窄列宽 {ratio:.1%} > {lim:.1%}），'
                        f'两条都不是可信单栏 → 双方回退')

    # ---------------- 邻栏间距（在保留下来的框之间）
    good = sorted([v for v in verdicts if v.accepted], key=lambda v: v.bbox[0])
    tight = []
    for a, b in zip(good, good[1:]):
        gap = b.bbox[0] - a.bbox[2]
        if ref and gap < cfg.get('min_gutter_ratio', 0.12) * ref:
            tight.append({'pair': [a.col_index, b.col_index], 'gap_px': int(gap),
                          'threshold_px': round(cfg.get('min_gutter_ratio', 0.12) * ref, 2)})
            a.measurements['gutter_tight'] = True
            b.measurements['gutter_tight'] = True
            a.reasons.append(f'与 #{b.col_index} 间隙仅 {gap}px，记 gutter_tight（不否决）')
            b.reasons.append(f'与 #{a.col_index} 间隙仅 {gap}px，记 gutter_tight（不否决）')

    accepted = [v for v in verdicts if v.accepted]
    rejected = [v for v in verdicts if not v.accepted]
    pitch_cv = None
    if len(accepted) >= 3:
        centers = sorted((v.bbox[0] + v.bbox[2]) / 2.0 for v in accepted)
        gaps = np.diff(centers)
        if gaps.mean() > 0:
            pitch_cv = float(gaps.std() / gaps.mean())
    if pitch_cv is not None and pitch_cv > cfg.get('max_pitch_cv', 0.35):
        notes.append(f'半页列距 CV={pitch_cv:.3f} 偏大，列框可能含合并列')
    if len(accepted) < cfg.get('min_cols_per_half', 2):
        notes.append(f'接受列数 {len(accepted)} < {cfg.get("min_cols_per_half", 2)}，'
                     f'该半页不宜直接用列框定栏')

    return {'verdicts': verdicts, 'accepted': accepted, 'rejected': rejected,
            'reference': {'value_px': ref, 'source': ref_source, 'notes': notes,
                          'outlier_multiplier': outlier_mult,
                          'threshold_px': round(thr, 2) if thr else None,
                          'split_min_width_px': split_min},
            'pitch_cv': round(pitch_cv, 4) if pitch_cv is not None else None,
            'tight_gutters': tight,
            'decisions': {v.col_index: v.decision for v in verdicts},
            'notes': notes}


def merge_split_columns(columns: Sequence[Dict[str, Any]], ref_width: Optional[float],
                        cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把「同一条实体栏被切成两半」的相邻列框合并。

    旧布局在双轨页上常把一栏沿中缝切成两条（丹方 p12 列宽 25、组宽 49 就是这样）。
    直接拿半条去配轨，搜索区域必然装不下整组。

    判据（两条都要满足）：
      1. 间隙很小：gap <= max(5, 0.35 × 较窄列宽)——正常栏之间有明显空白；
      2. 合并后仍是「一栏」：宽 <= 1.35 × 参考宽度——否则会把两栏并成一条。
    第 2 条是防止在思濟堂这类「一列就是一栏」的页上把邻栏也并进来。

    调用方必须**先按半页筛列框**再调用，否则会把书脊两侧的框并成一条。
    """
    gap_factor = float(cfg.get('merge_gap_factor', 0.35))
    max_mult = float(cfg.get('merge_max_reference_multiple', 1.35))
    merged_log: List[Dict[str, Any]] = []
    cols = [dict(c, bbox=[int(v) for v in c['bbox']]) for c in columns]
    cols.sort(key=lambda c: c['bbox'][0])
    changed = True
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(cols):
            if i + 1 < len(cols):
                a, b = cols[i], cols[i + 1]
                wa, wb = a['bbox'][2] - a['bbox'][0], b['bbox'][2] - b['bbox'][0]
                gap = b['bbox'][0] - a['bbox'][2]
                merged_w = b['bbox'][2] - a['bbox'][0]
                gap_ok = 0 <= gap <= max(5, gap_factor * min(wa, wb))
                width_ok = (ref_width is None) or (merged_w <= max_mult * ref_width)
                if gap_ok and width_ok:
                    out.append({'bbox': [a['bbox'][0], min(a['bbox'][1], b['bbox'][1]),
                                         b['bbox'][2], max(a['bbox'][3], b['bbox'][3])],
                                'merged_from': [a['bbox'], b['bbox']]})
                    merged_log.append({'from': [a['bbox'], b['bbox']],
                                       'to': [a['bbox'][0], min(a['bbox'][1], b['bbox'][1]),
                                              b['bbox'][2], max(a['bbox'][3], b['bbox'][3])],
                                       'gap_px': gap, 'merged_width_px': merged_w,
                                       'reference_px': ref_width})
                    i += 2
                    changed = True
                    continue
            out.append(cols[i])
            i += 1
        cols = out
    return cols, merged_log


def reject_wide_boxes(columns: Sequence[Dict[str, Any]], image_size: Tuple[int, int],
                      page_width_ratio: float = 0.45) -> List[Dict[str, Any]]:
    """兜底：宽度超过页宽某一比例的列框一律记为「跨栏框」并保留证据。

    与 validate_columns 的宽度校验互补——后者依赖参考宽度，这里不依赖，
    防止参考本身失真时把 446px 这种框放过去。
    """
    W, _H = image_size
    out = []
    for i, c in enumerate(columns):
        b = [int(v) for v in c['bbox']]
        w = b[2] - b[0]
        if w > page_width_ratio * W:
            out.append({'col_index': i, 'bbox': b, 'width': w,
                        'page_width_ratio': round(w / W, 4),
                        'threshold_ratio': page_width_ratio,
                        'why': '宽度超过页宽的该比例，几乎必然跨越多条实体栏'})
    return out
