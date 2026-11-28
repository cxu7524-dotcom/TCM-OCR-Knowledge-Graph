# -*- coding: utf-8 -*-
"""把 T1–T8 摊成「分子 / 分母 / 差额」表，两种标注口径并排。

为什么单独一个脚本：
  T1–T8 里有整体、分集、页级、轨级、负例五类指标，分母各不相同（38 组 / 20 组 /
  11 组 / 2 组 / 31 轨 / 确定负例区数）。只在报告里写「32/38」会让人以为所有项
  同分母；这里强制每项都带分子分母，并且换口径后逐项算差额。

用法：
  python tools/dual_track_stage8_targets.py
输出：
  output/dual_tracks/stage8/acceptance/targets_v2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
VERSIONS = (('C', 'C/predictions.json', '旧算法 C'),
            ('stage7-v2', 'stage7v2/predictions.json', 'stage7-v2'),
            ('stage8-v2', 'stage8v2/stage8v2.json', '修正版 stage8-v2'),
            ('stage8-v3', 'stage8v3/stage8v3_joint.json', '标注引导补修 stage8-v3'),
            ('stage8-v4-online', 'online_v4/online_v4.json',
             '正常入口 stage8-v4（封版复评）'))
CALIBERS = (('frozen', '冻结口径（stage2 原标签）'),
            ('v2', '复核口径（NEG-DF-SINGLE-01 撤标）'))
TARGETS = {
    'T1': ('全体已标注组命中 @0.50', 0.50, 'overall', 'group_hits'),
    'T2': ('全体已标注组命中 @0.75', 0.75, 'overall', 'group_hits'),
    'T3': ('历史回归分集命中 @0.50', 0.50, 'historical_regression', 'group_hits'),
    'T4': ('历史回归分集命中 @0.75', 0.75, 'historical_regression', 'group_hits'),
    'T5': ('轨级配对正确 @0.50', 0.50, 'overall', 'pairing'),
    'T6': ('负例确定误检（越少越好）', 0.50, 'overall', 'negative_confirmed_fp'),
}


def load(cal: str, rel: str) -> Dict[str, Any]:
    return json.loads((S8 / 'metrics' / f'neg_{cal}' / rel).read_text(encoding='utf-8'))


def page_hits(doc: Dict[str, Any], page_id: str, tk: str) -> Tuple[int, int]:
    for p in doc['pages']:
        if p['page_id'] == page_id:
            t = p['thresholds'][tk]
            return int(t.get('group_hits') or 0), int(t.get('annotations_total') or 0)
    return 0, 0


def measure(doc: Dict[str, Any], key: str, tk: str) -> Dict[str, Any]:
    name, _, scope, field = TARGETS[key]
    tkey = '0.5' if tk == 0.50 else '0.75'
    if scope == 'overall':
        x = doc['scores']['overall'][tkey]
    else:
        x = doc['scores']['by_split'][scope]['thresholds'][tkey]
    if field == 'pairing':
        num = x['lane_counts'].get('pairing_success', 0)
        den = x['lane_denominator']
        return {'num': num, 'den': den, 'value': round(num / den, 4) if den else None,
                'excluded': x['lane_excluded']}
    if field == 'negative_confirmed_fp':
        return {'num': x['negative_confirmed_fp'], 'den': None,
                'value': x['negative_confirmed_fp'],
                'suspected': x['negative_suspected_fp'],
                'duplicate_extra': x['duplicate_extra_fp'],
                'candidates': x['new_candidates']}
    # 分集表只给命中/分母，tp/fp/fn 只在 overall 里有
    return {'num': int(x['group_hits']), 'den': int(x['annotations_total']),
            'value': int(x['group_hits']),
            'tp': x.get('tp'), 'fp': x.get('fp'), 'fn': x.get('fn')}


def main() -> int:
    out: Dict[str, Any] = {'generated_by': 'tools/dual_track_stage8_targets.py',
                           'calibers': {c: d for c, d in CALIBERS},
                           'targets': {}}
    docs = {(cal, v): load(cal, rel) for cal, _ in CALIBERS for v, rel, _ in VERSIONS}

    for key in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6'):
        name, tk, _, _ = TARGETS[key]
        row = {'title': name, 'threshold': tk, 'by_caliber': {}}
        for cal, cname in CALIBERS:
            row['by_caliber'][cal] = {v: measure(docs[(cal, v)], key, tk) for v, _, _ in VERSIONS}
        row['delta_stage8'] = {
            'frozen_to_v2': (row['by_caliber']['v2']['stage8-v3']['value']
                             - row['by_caliber']['frozen']['stage8-v3']['value']),
            'v2_to_v3_frozen': (row['by_caliber']['frozen']['stage8-v3']['value']
                                - row['by_caliber']['frozen']['stage8-v2']['value']),
        }
        row['delta_stage8']['frozen_to_v2'] = (row['by_caliber']['v2']['stage8-v3']['value']
                                               - row['by_caliber']['frozen']['stage8-v3']['value'])
        out['targets'][key] = row

    for key, page in (('T7', '丹方彚编/page_12'), ('T8', '温疫方論/page_12')):
        row = {'title': f'{page} 命中 @0.50', 'threshold': 0.50, 'page': page, 'by_caliber': {}}
        for cal, _ in CALIBERS:
            row['by_caliber'][cal] = {}
            for v, _, _ in VERSIONS:
                num, den = page_hits(docs[(cal, v)], page, '0.5')
                row['by_caliber'][cal][v] = {'num': num, 'den': den, 'value': num}
        out['targets'][key] = row

    dst = S8 / 'acceptance' / 'targets_v2.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # 控制台：一项一行，分子分母差额都写出来
    print(f"{'项':4s} {'指标':26s} {'版本':10s} {'冻结':>10s} {'复核':>10s} {'差额':>6s}")
    for key in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'):
        row = out['targets'][key]
        for v, _, label in VERSIONS:
            f = row['by_caliber']['frozen'][v]
            g = row['by_caliber']['v2'][v]
            fs = f"{f['num']}/{f['den']}" if f['den'] else str(f['num'])
            gs = f"{g['num']}/{g['den']}" if g['den'] else str(g['num'])
            if key == 'T5':
                fs += f"={f['value']:.3f}" if f['value'] is not None else ''
                gs += f"={g['value']:.3f}" if g['value'] is not None else ''
            d = g['value'] - f['value']
            print(f"{key:4s} {row['title'][:26]:26s} {label:10s} {fs:>10s} {gs:>10s} "
                  f"{('%+.4f' if key == 'T5' else '%+d') % d:>6s}")
    print(f'→ {dst.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
