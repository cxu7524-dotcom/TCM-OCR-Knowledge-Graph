# -*- coding: utf-8 -*-
"""stage8 有效配置的**唯一来源**：跑批与正常 OCR 入口都必须读这一份。

为什么要有这个模块
------------------
v1 的问题不是「参数不对」，是**两条路径各自拼参数**：
`tools/dual_track_stage8_run.py` 用命令行开关拼，
`ocr/dual_track.py` 用 `cfg.get('stage8')` 拼，跨度上限还靠一个进程级 monkeypatch。
于是「同一份有效配置」这句话无法验证——离线跑出来 146 条，在线可能 145 条，
而且看不出来差在哪个开关上。

v2 的做法：两边都调用 `effective_options()`，返回值是可序列化的字典，
逐页写进 `page_runs[].effective`，并给一个 `fingerprint`（12 位 sha1）。
对账时只要比 fingerprint，就能确定两次运行是否真的用了同一份配置。

字段含义
--------
detector      检测器：`stage5`（冻结基线）或 `stage8`（修正版）
options.slots      列槽切分（过宽列按周期 + 低谷切成槽）
options.clip       子框裁到所属列槽的 x 范围
options.tighten    子框按框内墨迹收紧 x
options.use_limits 把列槽的 x 限位传给列内检测
options.relax_span 跨度上限改为按列高（否则 150px）
options.dedupe_slot 同槽纵向重叠去重（**负结果，默认关**）
span_ratio     relax_span 打开时用的比例（None = 已关闭，沿用 150px）
effective_span 人读的上限描述（写进报告与逐页记录）
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from dual_track_fix.relax import DEFAULT_SPAN_LIMIT, DEFAULT_SPAN_RATIO

STAGE8_DETECTORS = ('stage8', 'stage8-v1', 'slot')
OPTION_KEYS = ('slots', 'clip', 'tighten', 'use_limits', 'relax_span', 'dedupe_slot',
               'gap_clip', 'bound_bottom', 'band_slots', 'auto_joint')
# 冻结默认（stage8-v4：v1 配置 + 从原图自动触发的跨槽联合取样）
FROZEN = {'slots': True, 'clip': True, 'tighten': True, 'use_limits': True,
          'relax_span': True, 'dedupe_slot': False,
          'gap_clip': False, 'bound_bottom': False, 'band_slots': False,
          'auto_joint': True}


def _as_bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on', 'y')


def _ocr_cfg():
    try:
        from app.config import CONFIG
        return CONFIG.ocr
    except Exception:            # 配置不可用时退回冻结默认，绝不因此改变行为
        return None


def effective_options(overrides: Optional[Dict[str, Any]] = None,
                      ocr_cfg: Any = None) -> Dict[str, Any]:
    """本次调用的有效配置（可序列化）。

    overrides 只用于**实验对照**（关掉某一个动作做消融）；
    正式跑批与线上入口一律不传，读 `app.config.CONFIG.ocr`。
    """
    ocr = ocr_cfg if ocr_cfg is not None else _ocr_cfg()

    def g(name: str, default):
        return getattr(ocr, name, default) if ocr is not None else default

    eff: Dict[str, Any] = {
        'detector': str(g('dual_track_detector', 'stage5')).strip().lower(),
        'options': {
            'slots': _as_bool(g('dual_track_slots', FROZEN['slots'])),
            'clip': _as_bool(g('dual_track_clip', FROZEN['clip'])),
            'tighten': _as_bool(g('dual_track_tighten', FROZEN['tighten'])),
            'use_limits': _as_bool(g('dual_track_use_limits', FROZEN['use_limits'])),
            'relax_span': _as_bool(g('dual_track_relax_span', FROZEN['relax_span'])),
            'dedupe_slot': _as_bool(g('dual_track_dedupe_slot', FROZEN['dedupe_slot'])),
            # 负结果（N-06/N-07）：无标注时无法判别「槽该往哪边外延」，默认关
            'gap_clip': _as_bool(g('dual_track_gap_clip', FROZEN['gap_clip'])),
            'bound_bottom': _as_bool(g('dual_track_bound_bottom', FROZEN['bound_bottom'])),
            # 上下分栏各自定相位（p12 下半栏整体右移 ~15px 的成因，见 §6.4）
            'band_slots': _as_bool(g('dual_track_band_slots', FROZEN['band_slots'])),
            # v4：从原图 + 列槽自动触发跨槽联合取样（不再依赖冻结框定位）
            'auto_joint': _as_bool(g('dual_track_auto_joint', FROZEN['auto_joint'])),
        },
        'span_ratio': float(g('dual_track_span_ratio', DEFAULT_SPAN_RATIO)),
        'source': 'app.config',
    }
    for k, v in (overrides or {}).items():
        if k == 'detector':
            eff['detector'] = str(v).strip().lower()
        elif k == 'span_ratio':
            eff['span_ratio'] = float(v)
        elif k in eff['options']:
            eff['options'][k] = _as_bool(v, eff['options'][k])
        else:
            raise KeyError(f'未知的 stage8 配置项：{k}')
    if overrides:
        eff['source'] = 'overrides'

    # relax_span 关掉时把比例置 None，下游只认「比例 or 150」这一条判断
    if not eff['options']['relax_span']:
        eff['span_ratio'] = None
    eff['span_limit_default'] = DEFAULT_SPAN_LIMIT
    eff['effective_span'] = (f'max(150, {eff["span_ratio"]:.2f}×列高)'
                             if eff['span_ratio'] else '150')
    eff['fingerprint'] = fingerprint(eff)
    return eff


def fingerprint(eff: Dict[str, Any]) -> str:
    """配置指纹：只覆盖影响结果的字段，用来判断两次运行是否同口径。"""
    payload = {'detector': eff.get('detector'),
               'options': {k: bool(eff['options'].get(k)) for k in OPTION_KEYS},
               'span_ratio': eff.get('span_ratio')}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:12]


def detect_options(eff: Dict[str, Any]) -> Dict[str, Any]:
    """把有效配置转成 `dual_track_fix.detect.detect(options=...)` 的形状。"""
    out = dict(eff['options'])
    out['span_ratio'] = eff.get('span_ratio')
    return out
