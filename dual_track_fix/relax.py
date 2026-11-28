# -*- coding: utf-8 -*-
"""stage8 修正三：断组跨度上限——**改为调用参数**，不再做 monkeypatch。

历史（v1）
----------
`ocr/small_columns._trace_boundaries` 里有一条硬上限：

    if not 18 <= main_bottom - main_top <= 95 or core_end - core_start > 150:
        return None

这个 150px 是按思濟堂短注（1–3 字剂量注，实测组高 52–170px）标定的；丹方 p12 的
长注组高 254–535px（H=21），必然被判失败——DF12-08 槽在放宽前一条都出不来。

v1 的做法是用 `inspect.getsource` + 字符串替换，把整个函数换掉（进程级全局生效）。
它有两个问题，v2 起全部改掉：

  1. 补丁一装，**所有**调用点一起变，从调用点看不出这次跑的是哪种口径；
  2. 跑批（tools/）与正常入口（ocr/dual_track.py）只要有一个忘了装/多装一次，
     同一份「有效配置」会产出两种结果——这正是 stage8-v2 要消除的。

v2 的做法
--------
`propose_in_column(..., span_ratio=r)` 把比例作为**本次调用参数**传下去，
由 `propose_in_column` 换算成像素上限再传给 `_trace_boundaries(span_limit=...)`。
`span_ratio=None` 时用量表里的 `SPAN_LIMIT=150`，即既有调用的默认行为完全不变。

换算规则与 v1 完全一致（v1 用的是 `bw.shape[0]`，也就是列裁块的纵向高度）：

    span_limit = max(150, round(r × 列裁块高度))

本模块只保留换算与说明，不再有 install/restore 的状态。
"""

from __future__ import annotations

from typing import Optional

DEFAULT_SPAN_LIMIT = 150
DEFAULT_SPAN_RATIO = 0.9


def span_limit_for(column_height: float, ratio: Optional[float] = DEFAULT_SPAN_RATIO) -> int:
    """列高 → 该列的断组跨度上限（px）。ratio=None 表示沿用历史 150px。"""
    if ratio is None:
        return DEFAULT_SPAN_LIMIT
    try:
        h = float(column_height)
    except (TypeError, ValueError):
        return DEFAULT_SPAN_LIMIT
    return max(DEFAULT_SPAN_LIMIT, int(round(float(ratio) * h)))


def install(*_a, **_k):  # pragma: no cover - 只为拦住残留调用
    raise RuntimeError(
        'stage8-v2 起跨度上限改为 propose_in_column(span_ratio=...) 调用参数，'
        'monkeypatch 已移除；请改用 dual_track_fix.settings.effective_options()')
