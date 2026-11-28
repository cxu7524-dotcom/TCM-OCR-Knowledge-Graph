# -*- coding: utf-8 -*-
"""基于冻结旧布局生成异常宽框分层标签（frozen_layout_labels.json）。

测量对象：**旧布局 column bbox 的宽度**。人工 group_bbox 宽度只作目标尺寸描述，
不参与本标签（第二阶段已删掉「用人工组框宽度对照新参考」的固化路径）。

规则（冻结，版本 frozen-1）
--------------------------
书级参考 R = 该书所有页「页内列宽中位数」的中位数。
  - 用页内中位数再跨页取中位，避免单页几根合并列把参考抬高。
页级判定，按顺序：
  1. 页面列数 < 3            → unknown（样本不足）
  2. 页内列宽中位数 > 1.3R    → unknown（多数列框异常，参考在该页失真）
  3. 存在列宽 > 1.8R         → abnormal（个别列跨了实体栏）
  4. 其余                    → normal
1.8 沿用 page_layout 内部 wide_split 的经验倍数，但参考从「页内中位」换成
「书级参考」，避免自证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D          # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
TZ = timezone(timedelta(hours=8))

RULE = {
    'version': 'frozen-1',
    'measured_object': '旧布局 column bbox 宽度（px）',
    'reference': {
        'source': 'same_book_page_median_of_page_medians',
        'unit': 'px',
        'why': '页内中位数对单页内的合并列免疫；跨页再取中位避免个别畸形页主导',
    },
    'thresholds': {
        'outlier_multiplier': 1.8,
        'page_median_guard_multiplier': 1.3,
        'min_cols_per_page': 3,
        'min_reference_cols': 30,
    },
    'decision_order': [
        'n_cols < 3 -> unknown（样本不足）',
        'page_median_width > 1.3 * R -> unknown（多数列框异常）',
        'any col_width > 1.8 * R -> abnormal',
        'otherwise -> normal',
    ],
    'unknown_allowed': True,
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(layouts: Dict[str, Any]) -> Dict[str, Any]:
    books = list(layouts['by_book'].keys())
    ref: Dict[str, Dict[str, Any]] = {}
    for b in books:
        pages = [r for r in layouts['pages'] if r['book'] == b and r['status'] == 'ok' and r['cols']]
        medians = [statistics.median([c['width'] for c in r['cols']]) for r in pages]
        n_cols = sum(r['n_cols'] for r in pages)
        ref[b] = {
            'book_ref_px': round(statistics.median(medians), 3) if medians else None,
            'page_median_spread': {'min': round(min(medians), 1), 'max': round(max(medians), 1)}
            if medians else None,
            'pages_in_reference': len(pages), 'cols_in_reference': n_cols,
            'usable': bool(medians) and n_cols >= RULE['thresholds']['min_reference_cols'],
        }
    # 参考不可用时退化：用全库（有来源、写清楚）
    all_medians = [statistics.median([c['width'] for c in r['cols']])
                   for r in layouts['pages'] if r['status'] == 'ok' and r['cols']]
    global_ref = round(statistics.median(all_medians), 3) if all_medians else None

    rows = []
    for r in layouts['pages']:
        if r['status'] != 'ok':
            rows.append({'page_id': r['page_id'], 'book': r['book'], 'page': r['page'],
                         'state': 'unknown', 'reason': f"layout_status={r['status']}",
                         'measured': None})
            continue
        widths = [c['width'] for c in r['cols']]
        b_ref = ref.get(r['book'], {})
        if b_ref.get('usable'):
            R, ref_source = b_ref['book_ref_px'], 'same_book'
        elif global_ref:
            R, ref_source = global_ref, 'global_substitute'
        else:
            rows.append({'page_id': r['page_id'], 'book': r['book'], 'page': r['page'],
                         'state': 'unknown', 'reason': 'no_reference_available', 'measured': None})
            continue

        thr = round(RULE['thresholds']['outlier_multiplier'] * R, 3)
        guard = round(RULE['thresholds']['page_median_guard_multiplier'] * R, 3)
        page_median = round(statistics.median(widths), 3) if widths else None
        over = [c for c in r['cols'] if c['width'] > thr]
        reason = ''
        if r['n_cols'] < RULE['thresholds']['min_cols_per_page']:
            state, reason = 'unknown', f"n_cols={r['n_cols']} < {RULE['thresholds']['min_cols_per_page']}"
        elif page_median is not None and page_median > guard:
            state = 'unknown'
            reason = f'页内列宽中位 {page_median} > 守卫值 {guard}（多数列框异常，参考失真）'
        elif over:
            state = 'abnormal'
            reason = f'{len(over)}/{r["n_cols"]} 列宽 > {thr}'
        else:
            state = 'normal'
        rows.append({
            'page_id': r['page_id'], 'book': r['book'], 'page': r['page'],
            'state': state, 'reason': reason,
            'measured': {
                'layout_snapshot_id': layouts.get('layout_snapshot_id'),
                'n_cols': r['n_cols'],
                'page_median_width_px': page_median,
                'max_col_width_px': max(widths) if widths else None,
                'reference_px': R, 'reference_source': ref_source,
                'threshold_px': thr, 'page_median_guard_px': guard,
                'outlier_cols': [{'col_index': c['col_index'], 'bbox': c['bbox'],
                                  'width': c['width'],
                                  'ratio_to_reference': round(c['width'] / R, 3)}
                                 for c in over],
                'allowed_ratios': {'width_to_reference': 1.8, 'page_median_to_reference': 1.3},
            },
            'decision_status': 'frozen',
        })

    counts = {}
    for row in rows:
        counts[row['state']] = counts.get(row['state'], 0) + 1
    by_book: Dict[str, Dict[str, int]] = {}
    for row in rows:
        by_book.setdefault(row['book'], {})
        by_book[row['book']][row['state']] = by_book[row['book']].get(row['state'], 0) + 1

    return {
        'stage': 'stage3_dual_track_evaluation',
        'release': 'stage3-v1',
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage3_labels.py',
        'rule': RULE,
        'layout_source': {'file': 'output/dual_tracks/stage3/frozen_layouts.json',
                          'sha256': layouts.get('layout_sha256'),
                          'layout_snapshot_id': layouts.get('layout_snapshot_id'),
                          'entry': 'ocr.vertical.page_layout'},
        'reference': ref,
        'global_substitute_reference_px': global_ref,
        'counts': {'by_state': counts, 'by_book': by_book, 'pages': len(rows)},
        'pages': rows,
        'notes': [
            '标签只用于分层比较；abnormal 表示旧布局在该页有个别列框跨了实体栏',
            'unknown 有三类原因：列数不足、多数列框异常、无参考；报告里按原因分别计数',
            '本标签在阶段 3 冻结，阶段 8 沿用同一 layout_snapshot_id 与 rule.version',
            '人工 group_bbox 宽度仅作目标尺寸描述，不参与本标签',
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成冻结宽框分层标签')
    ap.add_argument('--layouts', default=str(OUT / 'frozen_layouts.json'))
    ap.add_argument('--out', default=str(OUT / 'frozen_layout_labels.json'))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    lp = Path(args.layouts)
    layouts = json.loads(lp.read_text(encoding='utf-8'))
    layouts['layout_sha256'] = sha256_of(lp)
    layouts['layout_snapshot_id'] = 'layout-' + layouts['layout_sha256'][:12]
    payload = build(layouts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not args.quiet:
        print(f"layout_snapshot_id = {layouts['layout_snapshot_id']}")
        print(f"标签分布：{payload['counts']['by_state']}")
        for b, c in payload['counts']['by_book'].items():
            print(f'  {b:20s} {c}')
        print(f'→ {D.rel(out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
