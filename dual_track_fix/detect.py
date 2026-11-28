# -*- coding: utf-8 -*-
"""stage8 修正版检测器（冻结：stage8-v1）。

三个动作，按顺序、可单独关掉（便于归因）：

  1. `slots`   列槽切分：page_layout 出来的过宽列按周期 + 低谷切成列槽
               （p12 右半页原来被并成一列 446px，13/38 个标注组落在这里）
  2. `clip`    子框裁到所属列槽的 x 范围（旧入口 C 会给列槽左右各留 7px padding，
               右边缘因此多出 12px：986 → 998）
  3. `tighten` 子框按框内墨迹收紧 x（标注框的构造规则就是「列窗 + 中缝 + 墨迹外接框」，
               旧入口输出的是列窗框，所以 0.5 命中 24 组而 0.75 只剩 11 组）

列内检测一律复用 `ocr.small_columns.propose_in_column`；断组跨度上限通过该函数的
`span_ratio` **调用参数**传入（v1 用进程级 monkeypatch，已移除，见 relax.py）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .relax import DEFAULT_SPAN_RATIO
from .slots import layout_slots

DEFAULT = {'slots': True, 'clip': True, 'tighten': True, 'use_limits': True,
           'relax_span': True, 'dedupe_slot': False, 'band_slots': False,
           'span_ratio': DEFAULT_SPAN_RATIO,
           'gap_clip': False, 'bound_bottom': False, 'auto_joint': True}


def _ink_cols(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[Tuple[int, int]]:
    """框内墨迹的 x 外接（列方向），排除贯穿框高的竖线（栏线/版框）。"""
    import cv2
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(gray.shape[1], x2); y2 = min(gray.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    sub = gray[y1:y2, x1:x2]
    bg = cv2.GaussianBlur(sub, (31, 31), 0).astype(np.int16)
    ink = (bg - sub.astype(np.int16)) >= 14
    cs = ink.sum(axis=0)
    keep = cs < 0.70 * ink.shape[0]
    xs = np.flatnonzero((cs > 0) & keep)
    if xs.size == 0:
        return None
    return x1 + int(xs[0]), x1 + int(xs[-1]) + 1


def detect(image: np.ndarray, page_id: str, H: float,
           options: Optional[Dict[str, bool]] = None,
           gray: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """整页 → {regions, groups, slot_audit}，坐标原图半开区间。"""
    import cv2
    from ocr.small_columns import estimate_page_typography, propose_in_column

    opt = dict(DEFAULT)
    opt.update(options or {})
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 跨度上限只作为本次调用的参数传下去；关闭时传 None（=150px 历史行为）
    span_ratio = opt.get('span_ratio') if opt.get('relax_span') else None

    if opt['slots']:
        lay = layout_slots(image, H, cfg={'band_slots': opt.get('band_slots')})
        cols = lay['cols']
        audit = lay['slot_audit']
    else:
        lay = page_layout_from_image(image)
        cols = lay['cols']
        audit = []
    src = 'slot_split' if opt['slots'] else 'page_layout'

    scale = estimate_page_typography(image, cols)
    w = image.shape[1]
    # 版框既用于 bound_bottom，也用于 v4 自动跨槽区段的轨底；无条件先算。
    border_y = page_bottom_border(gray)
    regions: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for i, col in enumerate(cols):
        slot = [int(v) for v in col['bbox']]
        rid = f'{page_id}#slot{i:02d}'
        regions.append({'region_id': rid, 'bbox': slot, 'source': src,
                        'half': col.get('half') or [0, w], 'validated': True,
                        'slot_untightened': col.get('slot_untightened')})
        x1, _, x2, _ = slot
        left = max([b['bbox'][2] for j, b in enumerate(cols) if j != i and b['bbox'][2] <= x1], default=0)
        right = min([b['bbox'][0] for j, b in enumerate(cols) if j != i and b['bbox'][0] >= x2], default=w)
        lim = ((left + x1) // 2, (right + x2) // 2) if opt['use_limits'] else None
        frame_top = column_frame_top(gray, slot, float((scale or {}).get('height') or H))
        for j, prop in enumerate(propose_in_column(image, slot, lim, scale,
                                                   span_ratio=span_ratio,
                                                   bottom_override=frame_top) or []):
            lanes = {c['side']: list(c['bbox']) for c in prop.get('columns') or []}
            rb, lb = lanes.get('right'), lanes.get('left')
            if rb is None or lb is None:
                continue
            # 记录裁前框：只有超出列槽的边才允许按空隙外延（见 _extend_truncated）
            rb_raw, lb_raw = list(rb), list(lb)
            if opt['clip']:
                rb = _clip(rb, slot); lb = _clip(lb, slot)
                if opt['gap_clip']:
                    # 槽周期 ≈ 2.4×字高（p12 实测 H=21、槽距 50），外延上限半个周期
                    pitch = 2.4 * float(H)
                    rb = _extend_truncated(gray, rb, rb_raw, slot, pitch)
                    lb = _extend_truncated(gray, lb, lb_raw, slot, pitch)
            if opt['tighten']:
                rb = _tighten(gray, rb) or rb
                lb = _tighten(gray, lb) or lb
            if opt.get('bound_bottom'):
                col_h = float(slot[3] - slot[1])
                rb = _extend_bottom(gray, rb, image.shape[0], col_h, border_y)
                lb = _extend_bottom(gray, lb, image.shape[0], col_h, border_y)
            gb = [min(lb[0], rb[0]), min(lb[1], rb[1]),
                  max(lb[2], rb[2]), max(lb[3], rb[3])]
            groups.append({
                'group_id': f'{page_id}#{rid.split("#")[-1]}#{j:02d}',
                'page_id': page_id, 'region_id': rid,
                'group_bbox': gb, 'right_bbox': rb, 'left_bbox': lb,
                'main_bbox': prop.get('main_bbox'),
                'region_bbox': slot,
                'region_source': src,
                'detector': 'stage8-v1',
                'evidence': {**(prop.get('evidence') or {}), 'bottom_frame_y': frame_top},
            })
    # v4：从原图 + 上面这批列槽自动触发跨槽联合取样。冻结标注不参与。
    joint_candidates: List[Dict[str, Any]] = []
    if opt.get('auto_joint'):
        from .auto_joint import auto_joint_candidates
        joint_candidates = auto_joint_candidates(image, groups, regions, H, border_y)
        for k, c in enumerate(joint_candidates):
            right = next((r for r in regions
                          if r['region_id'] == c['trigger']['slot_right'][0]
                          or r['bbox'] == c['trigger']['slot_right']), None)
            rid = (right or regions[0])['region_id']
            groups.append({
                'group_id': f'{page_id}#{rid.split("#")[-1]}#joint{k:02d}',
                'page_id': page_id, 'region_id': rid,
                'group_bbox': c['group_bbox'],
                'right_bbox': c['right_bbox'], 'left_bbox': c['left_bbox'],
                'main_bbox': c['main_est'], 'region_bbox': (right or regions[0])['bbox'],
                'region_source': 'joint_auto',
                'detector': 'stage8-v4',
                'evidence': {'method': 'joint_sample_auto_from_image',
                             'valley_x': c['seam_x'], 'trigger': c['trigger']},
                'joint_evidence': c['evidence'],
            })
    all_proposals = [dict(g) for g in groups]
    gray_for_dedupe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if (gray is None and image.ndim == 3) else gray
    groups, dropped = _superseded_windows(groups, H, gray=gray_for_dedupe)
    if opt['dedupe_slot']:
        groups, extra = _dedupe_slot(groups)
        dropped.extend(extra)
    groups.sort(key=lambda g: (g['group_bbox'][1], g['group_bbox'][0]))
    for k, g in enumerate(groups):
        g['index'] = k
    return {'regions': regions, 'groups': groups, 'all_proposals': all_proposals,
            'slot_audit': audit, 'dropped_duplicates': dropped,
            'typography': scale, 'options': opt,
            'span_ratio': span_ratio, 'joint_candidates': joint_candidates}


def _overlap_1d(a: Sequence[int], b: Sequence[int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _has_title_break(ink: np.ndarray, labels: np.ndarray, stats: np.ndarray,
                     valley_x: float, x1: int, x2: int, y_top: int, H: float) -> bool:
    """检查候选起点上方是否存在跨中缝的药名大字断组信号。"""
    ya = max(0, int(y_top - 2.5 * H))
    yb = min(ink.shape[0], int(y_top + 0.5 * H))
    if yb <= ya or x2 <= x1:
        return False
    ids = set(np.unique(labels[ya:yb, x1:x2])) - {0}
    for cid in ids:
        cx, _, cw, ch, _ = map(int, stats[cid])
        if cw >= 0.65 * H and ch >= 0.75 * H and cx + 1 <= valley_x <= cx + cw - 1:
            return True
    return False


def _uncovered_side_ink(ink: np.ndarray, box: Sequence[int],
                        excl_boxes: Sequence[Optional[Sequence[int]]]) -> int:
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return 0
    sub = ink[y1:y2, x1:x2].copy()
    for excl in excl_boxes:
        if not excl:
            continue
        ex1 = max(x1, int(excl[0])) - x1
        ey1 = max(y1, int(excl[1])) - y1
        ex2 = min(x2, int(excl[2])) - x1
        ey2 = min(y2, int(excl[3])) - y1
        if ex2 > ex1 and ey2 > ey1:
            sub[ey1:ey2, ex1:ex2] = False
    return int(sub.sum())


def _superseded_windows(groups: List[Dict[str, Any]], H: float,
                        gray: Optional[np.ndarray] = None
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """排除已被同槽稳定轨道解释的短中缝及同侧全覆盖嵌套重复，保留上下独立注记。"""
    import cv2

    sustained: Dict[str, List[Tuple[Dict[str, Any], Sequence[int]]]] = {}
    for group in groups:
        core = group['evidence'].get('core_y')
        if core and core[1] - core[0] >= 4 * H:
            sustained.setdefault(group['region_id'], []).append((group, core))

    kept, audit = [], []
    for group in groups:
        core = group['evidence'].get('core_y')
        peers = sustained.get(group['region_id'], ())
        if not core or core[1] - core[0] > 2.5 * H:
            kept.append(group)
            continue
        bounds = group['group_bbox'][1::2]
        for peer, peer_core in peers:
            if abs(group['evidence']['valley_x'] - peer['evidence']['valley_x']) > 0.8 * H:
                continue
            note = peer['group_bbox'][1::2]
            title = peer['main_bbox'][1::2]
            enters_note = (_overlap_1d(bounds, note) >= H
                           and min(core[1] - peer_core[0], peer_core[1] - core[0]) >= -H)
            inside_title = (_overlap_1d(bounds, title) >= 0.55 * (bounds[1] - bounds[0])
                            and _overlap_1d(core, title) >= 0.3 * (core[1] - core[0])
                            and note[0] - bounds[1] <= 2 * H)
            if enters_note or inside_title:
                audit.append({'group_id': group['group_id'],
                              'superseded_by': peer['group_id'],
                              'reason': 'same_slot_sustained_pair',
                              'overlaps_title': inside_title, 'core_y': list(core),
                              'sustained_core_y': list(peer_core)})
                break
        else:
            kept.append(group)

    if gray is None or gray.size == 0:
        return kept, audit

    bg = cv2.GaussianBlur(gray, (31, 31), 0).astype(np.int16)
    ink = (bg - gray.astype(np.int16)) >= 14
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    min_sup = max(8, round(0.5 * H))

    by_reg: Dict[str, List[Dict[str, Any]]] = {}
    for g in kept:
        if g.get('region_source') != 'joint_auto':
            by_reg.setdefault(g['region_id'], []).append(g)

    drop_ids: set[str] = set()
    for rid, gs in by_reg.items():
        gs_s = sorted(gs, key=lambda x: -(x['group_bbox'][3] - x['group_bbox'][1]))
        for i in range(len(gs_s)):
            p = gs_s[i]
            if p['group_id'] in drop_ids:
                continue
            py = p['group_bbox'][1::2]
            for j in range(i + 1, len(gs_s)):
                g = gs_s[j]
                if g['group_id'] in drop_ids:
                    continue
                gy = g['group_bbox'][1::2]
                gh = gy[1] - gy[0]
                if gh <= 0 or (py[1] - py[0]) <= gh:
                    continue
                ov = _overlap_1d(py, gy)
                if ov < 0.85 * gh:
                    continue
                if abs(p['evidence']['valley_x'] - g['evidence']['valley_x']) > 0.8 * H:
                    continue
                uncov = [
                    _uncovered_side_ink(ink, g[f'{s}_bbox'], [p[f'{s}_bbox']])
                    for s in ('right', 'left')
                ]
                if max(uncov) >= min_sup:
                    continue
                if _has_title_break(ink, labels, stats, g['evidence']['valley_x'],
                                    g['group_bbox'][0], g['group_bbox'][2], gy[0], H):
                    continue
                drop_ids.add(g['group_id'])
                audit.append({
                    'group_id': g['group_id'],
                    'superseded_by': p['group_id'],
                    'reason': 'same_slot_nested_pair',
                    'y_overlap': ov,
                    'uncovered_side_ink': {'right': uncov[0], 'left': uncov[1]},
                    'core_y': list(g['evidence'].get('core_y') or ()),
                    'sustained_core_y': list(p['evidence'].get('core_y') or ()),
                })

    kept = [g for g in kept if g['group_id'] not in drop_ids]
    return kept, audit


def _y_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """同槽两组的纵向重叠（按较短的那个算），用于判重复。"""
    iy0, iy1 = max(a[1], b[1]), min(a[3], b[3])
    if iy1 <= iy0:
        return 0.0
    h = min(a[3] - a[1], b[3] - b[1])
    return (iy1 - iy0) / h if h > 0 else 0.0


def _dedupe_slot(groups: List[Dict[str, Any]], thr: float = 0.5
                 ) -> tuple:
    """同一列槽内纵向重叠 ≥thr 的组只留证据最强的一个。

    一个列槽（一栏）里，同一段标注只会有一对左右轨。旧入口对同一槽会给出
    多个窗口组（实测 p12 某槽一次出 4 条、彼此 y 重叠 80% 以上），这些属于
    重复输出，不是分别命中不同条目。排序依据取「跨度 → 窗口数」，都是检出侧
    自带的证据量，不看标注。

    **默认关闭（负结果）**：开启后负例误检 4→1、候选 96→108，但分组命中
    32→30、配对率 0.903→0.897（跌破 T5 的 0.90）、p12 6→4，净效果为负。
    根因是「长组里套着短组」时，被判定为重复的那一条往往才是与标注对齐的那条
    （p12 的长注 500px 与槽内 100px 碎片 y 重叠接近 1.0）。
    """
    by_slot: Dict[str, List[Dict[str, Any]]] = {}
    for g in groups:
        by_slot.setdefault(g['region_id'], []).append(g)
    kept_all: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for rid, gs in by_slot.items():
        order = sorted(gs, key=lambda g: (-(g['group_bbox'][3] - g['group_bbox'][1]),
                                          -(g['evidence'].get('windows') or 0)))
        kept: List[Dict[str, Any]] = []
        for g in order:
            if any(_y_overlap(g['group_bbox'], k['group_bbox']) >= thr for k in kept):
                d = dict(g); d['dropped'] = 'duplicate_in_slot'
                dropped.append(d)
            else:
                kept.append(g)
        kept_all.extend(kept)
    dropped.sort(key=lambda g: (g['group_bbox'][1], g['group_bbox'][0]))
    return kept_all, dropped


def page_layout_from_image(image: np.ndarray) -> Dict[str, Any]:
    """不落盘的 page_layout（与 ocr.vertical.page_layout 同一套切分）。"""
    from ocr.vertical import split_columns, split_spread
    h, w = image.shape[:2]
    cols = []
    for hx0, hx1 in split_spread(image):
        for c in split_columns(image[:, hx0:hx1]):
            b = list(c['bbox']); b[0] += hx0; b[2] += hx0
            cols.append({'bbox': b, 'half': [hx0, hx1], 'col_index': len(cols)})
    return {'width': w, 'height': h, 'cols': cols}


def _clip(box: Sequence[int], slot: Sequence[int]) -> List[int]:
    a = [max(int(box[0]), int(slot[0])), int(box[1]),
         min(int(box[2]), int(slot[2])), int(box[3])]
    return a if a[2] - a[0] >= 6 else [int(v) for v in box]


def _scan_to_gap(gray: np.ndarray, x_edge: int, direction: int, y1: int, y2: int,
                 span: int, ink_floor: float = 0.10, gap_floor: float = 0.06) -> int:
    """从 x_edge 沿 direction 走：跳过空白 → 跟完这一块墨 → 停在它后面的空隙。

    只用于「轨框被列槽截断」的边：真正的列边界一定落在空隙里，截断点则可能落在
    墨上或落在中缝上，直接按槽边硬切会把这一组的后半截丢掉（p12 陽起石 Right
    轨被槽边切在 882，而它的右轨实际到 905）。
    返回新的边界 x（半开区间右端 +1 由调用方处理）。
    """
    import cv2
    x_edge = int(x_edge)
    y1 = max(0, int(y1)); y2 = min(gray.shape[0], int(y2))
    if y2 - y1 < 6:
        return x_edge
    lo = max(0, min(x_edge, x_edge + direction * span) - 2)
    hi = min(gray.shape[1], max(x_edge, x_edge + direction * span) + 3)
    sub = gray[y1:y2, lo:hi]
    bg = cv2.GaussianBlur(sub, (31, 31), 0).astype(np.int16)
    ink = (bg - sub.astype(np.int16)) >= 14
    frac = ink.sum(axis=0) / max(1, ink.shape[0])
    i_edge = x_edge - lo
    if not (0 <= i_edge < len(frac)):
        return x_edge
    seen = False
    pos = i_edge
    for d in range(0, span + 1):
        i = i_edge + direction * d
        if not (0 <= i < len(frac)):
            break
        v = frac[i]
        if v > 0.95:
            continue          # 整列贯通 = 版框/界行，不是字，不能当作继续外延的依据
        if v > ink_floor:
            seen = True
            pos = i
        elif seen and v < gap_floor:
            return lo + i
        if seen and v <= ink_floor:
            pos = i          # 空隙已经出现但还没确认足够空，继续看下一列
    return lo + pos if seen else x_edge


def _extend_truncated(gray: np.ndarray, lane: Sequence[int], pre_clip: Sequence[int],
                      slot: Sequence[int], pitch: float) -> List[int]:
    """轨框超出列槽时，不在槽边硬切，而是切到最近的真实空隙。"""
    out = [int(v) for v in lane]
    span = max(6, int(round(0.5 * pitch)))
    if pre_clip[2] > slot[2] + 1 and out[2] >= slot[2] - 1:
        out[2] = _scan_to_gap(gray, slot[2], +1, out[1], out[3], span) + 1
    if pre_clip[0] < slot[0] - 1 and out[0] <= slot[0] + 1:
        out[0] = _scan_to_gap(gray, slot[0], -1, out[1], out[3], span)
    return out


def page_bottom_border(gray: np.ndarray) -> int:
    """页面底版框：墨迹横跨 ≥60% 页宽的最后一个行（找不到就用 0.92×页高）。"""
    import cv2
    h, w = gray.shape[:2]
    sub = gray[:, :]
    bg = cv2.GaussianBlur(sub, (31, 31), 0).astype(np.int16)
    ink = (bg - sub.astype(np.int16)) >= 14
    rows = np.flatnonzero(ink.sum(axis=1) >= 0.60 * w)
    # 排除图像自身的黑边（scanner 边框在 96% 页高之后）
    rows = rows[(rows > 0.30 * h) & (rows < 0.96 * h)]
    return int(rows[0]) if rows.size else int(0.92 * h)


def column_frame_top(gray: np.ndarray, box: Sequence[int], H: float) -> Optional[int]:
    """在本栏和邻栏都有长横墨迹时，用实际底版框约束跟踪。"""
    import cv2

    height, width = gray.shape
    x1, _, x2, y2 = box
    center = (x1 + x2) // 2
    half = int(2.7 * H)
    left, right = max(0, center - half), min(width, center + half)
    start, end = int(0.88 * height), min(int(0.96 * height), y2)
    if right - left < 4 * H or end <= start:
        return None
    ink = cv2.adaptiveThreshold(gray[start:end, left:right], 1,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 14)
    length = max(3, round(3.1 * H))
    rule = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
    own = rule[:, max(0, x1 - left):min(right - left, x2 - left)].sum(axis=1)
    rows = np.flatnonzero((rule.sum(axis=1) >= 3.1 * H) & (own >= 0.65 * (x2 - x1)))
    return start + int(rows[0]) if rows.size else None


def _extend_bottom(gray: np.ndarray, box: Sequence[int], page_h: int,
                   col_h: float, border: Optional[int] = None) -> List[int]:
    """贴到底边的那一组：尾端补到版框内的最后一行墨（只补不缩）。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    bottom = int(border) if border else page_h
    if bottom - y2 > 0.15 * max(1.0, col_h):
        return [x1, y1, x2, y2]
    if y2 >= bottom - 1:
        return [x1, y1, x2, y2]
    import cv2
    sub = gray[y2:bottom, x1:x2]
    if sub.size == 0:
        return [x1, y1, x2, y2]
    bg = cv2.GaussianBlur(sub, (31, 31), 0).astype(np.int16)
    ink = (bg - sub.astype(np.int16)) >= 14
    rows = np.flatnonzero(ink.sum(axis=1) / max(1, x2 - x1) > 0.02)
    if rows.size == 0:
        return [x1, y1, x2, y2]
    return [x1, y1, x2, min(bottom, y2 + int(rows[-1]) + 1)]


def _tighten(gray: np.ndarray, box: Sequence[int], min_keep: float = 0.45
             ) -> Optional[List[int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    r = _ink_cols(gray, x1, y1, x2, y2)
    if r is None:
        return None
    if r[1] - r[0] < min_keep * max(1, x2 - x1):
        return None
    return [r[0], y1, r[1], y2]
