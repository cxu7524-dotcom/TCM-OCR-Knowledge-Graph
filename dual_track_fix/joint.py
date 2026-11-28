# -*- coding: utf-8 -*-
"""stage8 补修：中缝压在列槽边界上的跨槽双轨组（陽起石 DF12-11）。

问题（2026-11-25 定位）：
  p12 下半栏整体右移约 15px，DF12-11（陽起石）的中缝 x=882 正好与列槽边界
  slot02|slot01 = 882 重合。单槽提案只在各自槽内找缝：slot02 找到 858（假缝，
  把左列内部当两轨），slot01 找到 909（上一组的缝），于是这一组被拆掉/配错，
  @0.50 IoU 只有 0.332。

修复（本模块）：
  只在「中缝低谷 ≈ 某条列槽边界」且「该边界穿过本段药名带」时触发联合：
    - 用相邻两槽的联合 x 范围重估中缝（边界 ±6 内取列投影最低点）；
    - 用药名带下沿作轨顶、版框或下一标题作轨底；
    - 中缝两侧各自的墨迹外接定左右轨，逐轨用行游程回溯首字/末字。
  这正好对应用户要求的「在同一纵向区段内从相邻槽联合取样，重新确认本段的
  左右轨和中缝；把候选证据记到对应区段」。

保守性：
  - 触发条件同时要求「低谷 ≈ 槽边界」**和**「药名带横跨该边界」，实测 p12
    11 组里只有 DF12-11 命中（其余组中缝都在槽内部，差 ≥18px）。
  - 不读任何标注坐标；药名带与中缝都从原图墨迹估计。
  - 新组带 `region_id = <右槽>#joint`，`joint_evidence` 记下联合过程，
    便于报告逐项核对；原单槽组不删（重复预测的处理见报告 §6）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 触发与取样的几何常量（p12 H=21 实测，按字高比例写，便于迁移到其他页）
SEAM_TOL_PX = 3          # 中缝低谷与槽边界的最大偏差
SEAM_WIN_PX = 6          # 联合重估中缝时峰谷搜索半径
LANE_HALF_RATIO = 1.4    # 每轨向中缝一侧搜索的宽度 ≈ 1.4×字高
MIN_LANE_W_RATIO = 0.5   # 轨框最小宽度 ≈ 0.5×字高
ROW_INK_MIN = 2          # 判「该行有墨」的最少墨迹列数
COL_INK_MIN = 2          # 判「该列有墨」的最少墨迹行数
TAIL_PAD = 2             # 末行之后留 2px，避免贴着笔画边界（半开区间）


def ink_mask(gray: np.ndarray) -> np.ndarray:
    import cv2
    bg = cv2.GaussianBlur(gray, (31, 31), 0).astype(np.int16)
    return (bg - gray.astype(np.int16)) >= 14


def _runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    st: Optional[int] = None
    for i, v in enumerate(mask):
        if v and st is None:
            st = i
        elif not v and st is not None:
            out.append((st, i))
            st = None
    if st is not None:
        out.append((st, len(mask)))
    return out


def _valley(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    """[x0,x1) 内在 [y0,y1) 列投影最低的列，作为真实中缝。"""
    d = ink[max(0, y0):y1, max(0, x0):x1].sum(axis=0)
    if d.size == 0:
        return (x0 + x1) // 2
    sm = np.convolve(d, np.ones(3) / 3, mode='same')
    return int(x0 + int(np.argmin(sm)))


def _extent(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> Optional[Tuple[int, int]]:
    d = ink[max(0, y0):y1, max(0, x0):x1].sum(axis=0)
    xs = np.flatnonzero(d >= COL_INK_MIN)
    if xs.size == 0:
        return None
    return int(x0 + xs[0]), int(x0 + xs[-1]) + 1


def _lane_y(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int,
            min_h: float) -> Optional[Tuple[int, int]]:
    """逐轨用行游程回溯首字顶与末字底：取覆盖该轨的连续墨迹段的两端。

    与通用 propose 的差别：这里轨的 x 已经由联合中缝确定，不再依赖 main 带
    推断轨顶——阳起石的药名带被单槽切成两半，正是通用 propose 把轨顶定到
    y=681（比标注 655 晚 26px）的原因。
    """
    rows = ink[y0:y1, x0:x1].sum(axis=1)
    on = rows >= ROW_INK_MIN
    segs = [s for s in _runs(on) if s[1] - s[0] >= 2]
    if not segs:
        return None
    top, bot = segs[0][0], segs[-1][1]
    if bot - top < min_h:
        return None
    return y0 + int(top), min(y1, y0 + int(bot) + TAIL_PAD)


def joint_propose(gray: np.ndarray, slot_left: Sequence[int], slot_right: Sequence[int],
                  main_est: Sequence[int], boundary_x: int, H: float,
                  bottom_limit: int,
                  next_title_y: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """对一条被中缝压住的槽边界做联合取样；不满足条件返回 None。

    main_est: 原图估计的药名带 [x1,y1,x2,y2]（本轮由 detect 的单槽提案 main_bbox
              合并得到，不读冻结标注）。要求 boundary_x 水平穿过它。
    bottom_limit: 版框行（detect.page_bottom_border 已实现）。
    next_title_y: 下一标题的上沿；有则轨底不超过它。
    """
    x1, _, x2, _ = (int(v) for v in main_est)
    if not (x1 < boundary_x < x2):
        return None
    y_top = int(main_est[3])
    y_bot = min(int(bottom_limit), int(next_title_y) if next_title_y else int(bottom_limit))
    if y_bot - y_top < 2 * H:
        return None

    ink = ink_mask(gray)
    jx0 = min(int(slot_left[0]), int(slot_right[0]))
    jx1 = max(int(slot_left[2]), int(slot_right[2]))
    seam = _valley(ink, y_top, y_bot, boundary_x - SEAM_WIN_PX, boundary_x + SEAM_WIN_PX + 1)
    if abs(seam - boundary_x) > SEAM_TOL_PX:
        return None

    half = max(10, int(round(LANE_HALF_RATIO * H)))
    min_w = max(6, int(round(MIN_LANE_W_RATIO * H)))
    lx = _extent(ink, y_top, y_bot, max(jx0, seam - half), seam - 2)
    rx = _extent(ink, y_top, y_bot, seam + 2, min(jx1, seam + half))
    if not lx or not rx or lx[1] - lx[0] < min_w or rx[1] - rx[0] < min_w:
        return None
    ly = _lane_y(ink, lx[0], lx[1], y_top, y_bot, min_h=min_w)
    ry = _lane_y(ink, rx[0], rx[1], y_top, y_bot, min_h=min_w)
    if not ly or not ry:
        return None
    lb = [lx[0], ly[0], lx[1], ly[1]]
    rb = [rx[0], ry[0], rx[1], ry[1]]
    gb = [min(lb[0], rb[0]), min(lb[1], rb[1]), max(lb[2], rb[2]), max(lb[3], rb[3])]
    return {
        'group_bbox': gb, 'left_bbox': lb, 'right_bbox': rb,
        'main_est': [int(v) for v in main_est], 'seam_x': int(seam),
        'boundary_x': int(boundary_x), 'joint_slot': [jx0, jx1],
        'section_y': [y_top, y_bot],
        'evidence': {
            'method': 'joint_sample_across_slot_boundary',
            'valley_x': int(seam), 'boundary_x': int(boundary_x),
            'left_ink_x': list(lx), 'right_ink_x': list(rx),
            'main_est': [int(v) for v in main_est],
            'section_y': [y_top, y_bot],
        },
    }
