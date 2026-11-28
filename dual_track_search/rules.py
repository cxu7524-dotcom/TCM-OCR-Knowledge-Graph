# -*- coding: utf-8 -*-
"""路径一：成组栏线（grouped_rules）。

古籍的「界行」是贯穿正文区的一组等距竖线。判据要同时满足：
方向（竖直）→ 纵向覆盖 → 相邻间距稳定 → 组合数量 → 栏内有字。

单独一条线可能只是版框边、书脊、长竖笔，**不能**当栏线用。

v2 的两处修正：
1. **一次可以出多组**。间距序列里只要有一个不合格间隔（过大/突增），以前整页
   作废；现在从该处断开，前后各自成组，剩下的范围继续走列框/回退，并逐段记录
   被拒的 run（太短、间距不稳、栏内无墨）。
2. **半页内检测**。候选与成组都在半页裁块内做，坐标带 offset 回映原图，
   避免把书脊当栏线、把对侧半页的线组复用到本半页。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .scale import as_ink_mask


@dataclass
class RuleCandidate:
    """一条候选竖线；坐标是**原图** x（线宽方向）。

    `segments` 保留合并前的每一段（原图 y 边界 + 贴字标记 + 置信），
    单段线时为空列表，用自身 bbox 当唯一一段。合并后 `status` 记录整体资格：
      confirmed        全部线段都独立
      touching         全部线段都贴字
      mixed_ambiguous  同时含贴字段与独立段 —— 合并记录必须保留这个歧义
    """
    x0: int
    x1: int
    y0: int
    y1: int
    coverage: float          # (y1-y0) / 正文区高
    ink_density: float       # 线内墨迹占比
    width: int
    kind: str = 'candidate'  # candidate / independent / touching / broken / frame / gutter
    segments: List[Dict[str, Any]] = field(default_factory=list)
    touches_text: bool = False
    note: str = ''
    confidence: str = 'unknown'   # high（独立/断线）/ low（贴字，只作诊断）
    group_id: Optional[str] = None
    status: str = 'unknown'       # confirmed / touching / mixed_ambiguous
    ambiguous: bool = False
    eligible_coverage: float = 0.0
    gap_total_px: int = 0
    qualified: Optional[bool] = None   # 由 line_eligibility 统一判定，四处消费者共用

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def segment_list(self) -> List[Dict[str, Any]]:
        """归一化的线段清单（单段线返回自身 bbox 一条）。"""
        if self.segments:
            return list(self.segments)
        return [{'x0': self.x0, 'x1': self.x1, 'y0': self.y0, 'y1': self.y1,
                 'touches_text': self.touches_text,
                 'confidence': 'low' if self.touches_text else 'high',
                 'eligible': not self.touches_text, 'coverage': self.coverage,
                 'ink_density': self.ink_density,
                 'source': 'single'}]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['center'] = round(self.center, 2)
        segs = self.segment_list()
        d['segments'] = segs
        d['eligible_segments'] = [s for s in segs if s.get('eligible')]
        d['excluded_from_scale'] = bool(self.qualified)
        return d


def line_eligibility(line, body_h: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """**唯一资格判据**：成组定栏、列框拆分、尺度排除、独立掩膜都调这一个函数。

    资格 = 「非贴字段的纵向跨度 ≥ group_min_coverage × 正文区高」。
    - 纯独立线：全程可排除、可定栏；
    - 纯贴字线：全程不可用；
    - 混合线（既有贴字段又有独立段）：只有当独立段仍占够多时才够格，
      而且即使够格，能用的也只是**独立段那几截**（贴字段与断口不排除）。
    """
    segs = line.segment_list() if hasattr(line, 'segment_list') else \
        line.get('segments') or [line]
    eligible = [s for s in segs if s.get('eligible', not s.get('touches_text'))]
    total = sum(int(s['y1']) - int(s['y0']) for s in segs)
    elig = sum(int(s['y1']) - int(s['y0']) for s in eligible)
    cov = (elig / body_h) if body_h else 0.0
    if not eligible:
        status = 'touching'
    elif len(eligible) == len(segs):
        status = 'confirmed'
    else:
        status = 'mixed_ambiguous'
    qualified = bool(eligible) and cov >= float(cfg['group_min_coverage']) - 1e-9
    return {'status': status, 'ambiguous': status == 'mixed_ambiguous',
            'eligible_segments': eligible, 'eligible_coverage': round(cov, 4),
            'segment_count': len(segs), 'eligible_segment_count': len(eligible),
            'total_segment_px': total, 'eligible_px': elig,
            'qualified': qualified,
            'threshold': cfg['group_min_coverage'],
            'reason': (f'非贴字段 {elig}px / 正文区 {body_h}px = {cov:.3f} '
                       f'{"≥" if qualified else "<"} 门槛 {cfg["group_min_coverage"]}'
                       + ('（含贴字段，整条按歧义处理）' if status == 'mixed_ambiguous'
                          else ''))}


@dataclass
class RuleGroup:
    """一组可信栏线：相邻两线的区间就是一条栏。"""
    lines: List[RuleCandidate]
    pitch_px: float
    pitch_cv: float
    span: Tuple[int, int]
    columns: List[Tuple[int, int]]   # 每栏的 (x1, x2)
    confidence: str = 'high'
    run_index: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'lines': [l.to_dict() for l in self.lines],
                'pitch_px': round(self.pitch_px, 2),
                'pitch_cv': round(self.pitch_cv, 4),
                'span': list(self.span),
                'columns': [list(c) for c in self.columns],
                'confidence': self.confidence,
                'run_index': self.run_index,
                'diagnostics': self.diagnostics}


def _vertical_mask(bw: np.ndarray, y0: int, y1: int, open_ratio: float) -> np.ndarray:
    """只留竖直长线；横线、字块被开运算去掉。"""
    h = max(1, y1 - y0)
    ker_h = max(40, int(h * open_ratio))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ker_h))
    src = as_ink_mask(bw[y0:y1]).astype(np.uint8) * 255
    return cv2.morphologyEx(src, cv2.MORPH_OPEN, ker)


def detect_rule_candidates(bw: np.ndarray, y_range: Tuple[int, int],
                           cfg: Dict[str, Any],
                           x_range: Optional[Tuple[int, int]] = None) -> List[RuleCandidate]:
    """在正文 y（可选 x 窗口）范围内找竖线候选，坐标回映到 bw 的坐标系。"""
    y0, y1 = int(y_range[0]), int(y_range[1])
    if y1 - y0 < 50:
        return []
    ox = 0
    src = bw
    if x_range is not None:
        ox = max(0, int(x_range[0]))
        src = bw[:, ox:min(bw.shape[1], int(x_range[1]))]
    vm = _vertical_mask(src, y0, y1, cfg['open_kernel_ratio'])
    h = y1 - y0
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(vm, 8)
    out: List[RuleCandidate] = []
    for i in range(1, n):
        x, y, w, hh, area = [int(v) for v in stats[i]]
        if w > cfg['max_width_px'] or hh < h * cfg['min_height_ratio']:
            continue
        seg = vm[y:y + hh, x:x + w]
        density = float((seg > 0).mean()) if seg.size else 0.0
        out.append(RuleCandidate(x0=x + ox, x1=x + w + ox, y0=y0 + y, y1=y0 + y + hh,
                                 coverage=round(hh / h, 4),
                                 ink_density=round(density, 4), width=w))
    out.sort(key=lambda r: (r.x0, r.y0))
    return out


def classify_rule_touching(candidates: Sequence[RuleCandidate], bw: np.ndarray,
                           cfg: Dict[str, Any]) -> List[RuleCandidate]:
    """区分「独立栏线」与「与文字粘连的疑似栏线」。

    做法：在线两侧各取 probe 宽的竖条，看有多少行有墨。独立栏线两侧基本是空白，
    贴字栏线两侧几乎每行都有笔画 —— 这类线不能从字形聚合里直接剔除，
    要保留墨迹和归属标记供后续核对。
    """
    probe = int(cfg['touching_probe_px'])
    thr = float(cfg['touching_ink_ratio'])
    mask = as_ink_mask(bw)
    for c in candidates:
        left = mask[c.y0:c.y1, max(0, c.x0 - probe):c.x0]
        right = mask[c.y0:c.y1, c.x1:min(mask.shape[1], c.x1 + probe)]
        lr = float(left.any(axis=1).mean()) if left.size else 0.0
        rr = float(right.any(axis=1).mean()) if right.size else 0.0
        touch = max(lr, rr) > thr
        c.touches_text = touch
        c.kind = 'touching' if touch else 'independent'
        c.confidence = 'low' if touch else 'high'
        c.note = (f'两侧墨迹行占比 L={lr:.2f} R={rr:.2f}（阈值 {thr}）→ '
                  + ('贴字，保留墨迹与归属标记，不参与定栏与尺度排除'
                     if touch else '独立栏线，可参与定栏并可从字形聚合中排除'))
    return list(candidates)


def merge_broken_rules(candidates: Sequence[RuleCandidate], cfg: Dict[str, Any],
                       body_h: Optional[int] = None
                       ) -> Tuple[List[RuleCandidate], List[Dict[str, Any]]]:
    """同 x 附近被断开的段合并成一条「断线」，合并记录保留逐段证据。

    合并**不**抹掉每段原来的 y 边界、贴字标记与置信：`segments` 里逐段记原图 y 区间；
    一条断线若同时含贴字段与独立段，`status=mixed_ambiguous`、`ambiguous=true`，
    且 `eligible_coverage` 只按独立段算 —— 后续四处消费者拿到的都是同一份资格结论。
    """
    gap_max = int(cfg['broken_gap_px'])
    dx_max = int(cfg['broken_max_dx'])
    ordered = sorted(candidates, key=lambda c: (c.x0, c.y0))
    merged: List[RuleCandidate] = []
    log: List[Dict[str, Any]] = []
    used = [False] * len(ordered)
    for i, c in enumerate(ordered):
        if used[i]:
            continue
        chain = [c]
        used[i] = True
        chain.sort(key=lambda r: r.y0)
        changed = True
        while changed:
            changed = False
            for j, o in enumerate(ordered):
                if used[j]:
                    continue
                last = chain[-1]
                if abs(o.x0 - last.x0) <= dx_max and 0 <= o.y0 - last.y1 <= gap_max:
                    chain.append(o)
                    used[j] = True
                    changed = True
                    chain.sort(key=lambda r: r.y0)
        if len(chain) == 1:
            merged.append(c)
            continue
        y0 = min(x.y0 for x in chain)
        y1 = max(x.y1 for x in chain)
        gaps = [chain[k + 1].y0 - chain[k].y1 for k in range(len(chain) - 1)]
        gap = sum(gaps)
        span0 = max(1, chain[0].y1 - chain[0].y0)
        body = int(body_h) if body_h else int(span0 / max(1e-6, chain[0].coverage))
        segs: List[Dict[str, Any]] = []
        for x in chain:
            segs.append({'x0': int(x.x0), 'x1': int(x.x1), 'y0': int(x.y0), 'y1': int(x.y1),
                         'touches_text': bool(x.touches_text),
                         'confidence': x.confidence if x.confidence != 'unknown'
                         else ('low' if x.touches_text else 'high'),
                         'eligible': not bool(x.touches_text),
                         'coverage': round((x.y1 - x.y0) / body, 4) if body else 0.0,
                         'ink_density': x.ink_density,
                         'source': 'merged_segment'})
        n_elig = sum(1 for s in segs if s['eligible'])
        status = ('confirmed' if n_elig == len(segs)
                  else 'touching' if n_elig == 0 else 'mixed_ambiguous')
        elig_px = sum(s['y1'] - s['y0'] for s in segs if s['eligible'])
        m = RuleCandidate(
            x0=chain[0].x0, x1=max(x.x1 for x in chain), y0=y0, y1=y1,
            coverage=round(min(1.0, (y1 - y0) / body), 4) if body else 0.0,
            ink_density=round(float(np.mean([x.ink_density for x in chain])), 4),
            width=max(x.width for x in chain), kind='broken',
            segments=segs,
            touches_text=(n_elig == 0),
            confidence='low' if n_elig == 0 else 'high',
            status=status, ambiguous=(status == 'mixed_ambiguous'),
            eligible_coverage=round(elig_px / body, 4) if body else 0.0,
            gap_total_px=int(gap),
            note=(f'{len(chain)} 段合并，断口 {gaps}；'
                  + {'confirmed': '全段独立',
                     'touching': '全段贴字',
                     'mixed_ambiguous': f'含贴字段（{len(segs) - n_elig} 段），合并保留歧义'}
                  [status]))
        merged.append(m)
        log.append({'x': chain[0].x0, 'x1': max(x.x1 for x in chain),
                    'segments': len(chain), 'gap_px': gap, 'gaps': gaps,
                    'y_range': [y0, y1],
                    'status': status, 'ambiguous': m.ambiguous,
                    'eligible_coverage': m.eligible_coverage,
                    'segment_detail': segs,
                    'coord_system': 'original_image',
                    'action': 'merged_as_broken'})
    merged.sort(key=lambda r: (r.x0, r.y0))
    return merged, log


def _dedupe(cands: Sequence[RuleCandidate], dx: int) -> List[RuleCandidate]:
    """同一 x 附近的重复候选只留覆盖最高、最窄的一条（防止双线各算一栏）。"""
    ordered = sorted(cands, key=lambda c: (c.center, -c.coverage, c.width))
    out: List[RuleCandidate] = []
    for c in ordered:
        if out and abs(c.center - out[-1].center) <= dx:
            if c.coverage > out[-1].coverage:
                out[-1] = c
            continue
        out.append(c)
    return out


def _split_runs(xs: Sequence[float], gaps: Sequence[float], cfg: Dict[str, Any]
                ) -> List[Tuple[int, int]]:
    """把间距序列切成若干「等距可信」的 run（含单条线），返回 (i, j) 线索引区间。"""
    lo, hi = float(cfg['group_pitch_min_px']), float(cfg['group_pitch_max_px'])
    jump = float(cfg.get('run_split_jump_ratio', 1.6))
    runs: List[Tuple[int, int]] = []
    start = 0
    for k, g in enumerate(gaps):
        bad = g < lo or g > hi
        if not bad and k > start:
            ref = float(np.median(gaps[start:k]))
            bad = ref > 0 and g > ref * jump
        if bad:
            runs.append((start, k))       # 线 start..k
            start = k + 1
    runs.append((start, len(xs) - 1))
    return runs


def _ink_check(bw: np.ndarray, y_range: Tuple[int, int], columns: Sequence[Tuple[int, int]],
               cfg: Dict[str, Any]) -> Tuple[List[float], List[int]]:
    y0, y1 = int(y_range[0]), int(y_range[1])
    band = as_ink_mask(bw[y0:y1])
    ratios = []
    for a, b in columns:
        seg = band[:, max(0, int(a)):max(1, int(b))]
        ratios.append(float(seg.mean()) if seg.size else 0.0)
    empty = [k for k, r in enumerate(ratios) if r < cfg['group_min_ink_between']]
    return ratios, empty


def group_rule_runs(candidates: Sequence[RuleCandidate], bw: np.ndarray,
                    y_range: Tuple[int, int], cfg: Dict[str, Any],
                    x_range: Optional[Tuple[int, int]] = None
                    ) -> Tuple[List[RuleGroup], List[Dict[str, Any]]]:
    """把候选切成若干 run，每个 run 独立判「能不能成组」；返回 (groups, run_log)。

    一个不合格间隔只作废它所在的那一段，**其余范围继续处理**。
    """
    log: List[Dict[str, Any]] = []
    xlo, xhi = (int(x_range[0]), int(x_range[1])) if x_range else (0, bw.shape[1])
    body_h = max(1, int(y_range[1]) - int(y_range[0]))
    usable = []
    for c in candidates:
        e = line_eligibility(c, body_h, cfg)
        c.status, c.ambiguous = e['status'], e['ambiguous']
        c.eligible_coverage, c.qualified = e['eligible_coverage'], e['qualified']
        if c.qualified and xlo <= c.center <= xhi:
            usable.append(c)
        else:
            log.append({'x': c.x0, 'x1': c.x1, 'span': [round(c.center, 1), round(c.center, 1)],
                        'why': 'not_usable',
                        'kind': c.kind, 'status': e['status'],
                        'ambiguous': e['ambiguous'],
                        'coverage': c.coverage,
                        'eligible_coverage': e['eligible_coverage'],
                        'segments': len(e['eligible_segments']),
                        'detail': ('贴字/混合线不参与定栏' if not c.qualified
                                   else '超出本半页 x 范围')})
    usable = _dedupe(usable, int(cfg.get('dedupe_dx_px', 2)))
    if len(usable) < cfg['group_min_count']:
        span = ([round(min(c.center for c in usable), 1),
                 round(max(c.center for c in usable), 1)] if usable else None)
        log.append({'why': 'candidate_count_below_min',
                    'detail': f'合格竖线 {len(usable)} < {cfg["group_min_count"]}',
                    'span': span,
                    'x_list': [round(c.center, 1) for c in usable]})
        return [], log

    xs = sorted(c.center for c in usable)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    runs = _split_runs(xs, gaps, cfg)

    groups: List[RuleGroup] = []
    for ri, (i, j) in enumerate(runs):
        seg_xs = xs[i:j + 1]
        if j <= i:
            log.append({'run_index': ri, 'why': 'run_single_line',
                        'span': [round(seg_xs[0], 1), round(seg_xs[-1], 1)],
                        'detail': '单条线不成栏'})
            continue
        if len(seg_xs) < cfg['group_min_count']:
            log.append({'run_index': ri, 'why': 'run_too_short',
                        'span': [round(seg_xs[0], 1), round(seg_xs[-1], 1)],
                        'lines': len(seg_xs),
                        'detail': f'该段 {len(seg_xs)} 条 < {cfg["group_min_count"]}，交回退处理'})
            continue
        seg_gaps = [seg_xs[k + 1] - seg_xs[k] for k in range(len(seg_xs) - 1)]
        mean = float(np.mean(seg_gaps))
        cv = float(np.std(seg_gaps) / mean) if mean else 9.9
        if cv > cfg['group_pitch_cv_max']:
            log.append({'run_index': ri, 'why': 'pitch_unstable',
                        'span': [round(seg_xs[0], 1), round(seg_xs[-1], 1)],
                        'gaps': [round(g, 1) for g in seg_gaps], 'cv': round(cv, 4),
                        'detail': f'间距 CV={cv:.3f} > {cfg["group_pitch_cv_max"]}'})
            continue
        lines = [c for c in usable if c.center in seg_xs]
        lines.sort(key=lambda c: c.center)
        columns = [(int(lines[k].x1), int(lines[k + 1].x0)) for k in range(len(lines) - 1)]
        ratios, empty = _ink_check(bw, y_range, columns, cfg)
        if empty and len(empty) >= max(1, len(columns) // 2):
            log.append({'run_index': ri, 'why': 'columns_without_ink',
                        'span': [round(seg_xs[0], 1), round(seg_xs[-1], 1)],
                        'ink_ratios': [round(r, 4) for r in ratios],
                        'detail': f'{len(empty)}/{len(columns)} 栏墨迹不足'})
            continue
        g = RuleGroup(lines=lines, pitch_px=mean, pitch_cv=cv,
                      span=(int(lines[0].x0), int(lines[-1].x1)),
                      columns=columns, confidence='high', run_index=ri,
                      diagnostics={'ink_ratios': [round(r, 4) for r in ratios],
                                   'empty_columns': empty,
                                   'candidates_total': len(candidates),
                                   'candidates_usable': len(usable),
                                   'pitch_min': cfg['group_pitch_min_px'],
                                   'pitch_max': cfg['group_pitch_max_px'],
                                   'pitch_cv_max': cfg['group_pitch_cv_max'],
                                   'run_span': [round(seg_xs[0], 1), round(seg_xs[-1], 1)]})
        for k in empty:
            g.confidence = 'medium'
            log.append({'run_index': ri, 'why': 'column_ink_low',
                        'column': list(columns[k]), 'ink_ratio': round(ratios[k], 4),
                        'detail': '该栏墨迹偏少，可能是缺字或分栏错误，标 medium'})
        groups.append(g)
    if not groups:
        log.append({'why': 'no_group_from_runs', 'runs': len(runs),
                    'detail': '所有 run 均未通过成组判据，交列框与回退'})
    return groups, log


def group_rules(candidates: Sequence[RuleCandidate], bw: np.ndarray,
                y_range: Tuple[int, int], cfg: Dict[str, Any]
                ) -> Tuple[Optional[RuleGroup], List[Dict[str, Any]]]:
    """兼容入口：只要最强的一组（线的条数最多）。完整多组用 group_rule_runs。"""
    groups, log = group_rule_runs(candidates, bw, y_range, cfg)
    if not groups:
        return None, log
    best = max(groups, key=lambda g: (len(g.lines), -g.pitch_cv))
    return best, log
