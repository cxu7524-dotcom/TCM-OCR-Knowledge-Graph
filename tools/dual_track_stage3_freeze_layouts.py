# -*- coding: utf-8 -*-
"""冻结旧布局：对评测相关页跑 ocr.vertical.page_layout，保存原始列框。

测量对象是**旧布局的 column bbox**——不是人工 group_bbox。人工框宽只作为
目标尺寸描述，两回事，报告里分开写。

输出 frozen_layouts.json：
  pages[]: page_id, book, page, image_path, sha256, image_size,
           cols[]（bbox/half/wide_split/width/index）, halves, n_cols,
           half_stats{}, status, timing_s
  by_book{}: 参考统计（列宽中位数、IQR、有效列数）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D          # noqa: E402
from ocr.vertical import page_layout         # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
TZ = timezone(timedelta(hours=8))
BOOKS = ['丹方彚编', '合併脉訣難經太素評林', '太醫院校註婦人良方大全', '思濟堂方書', '温疫方論']


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _iqr(vals: List[float]) -> Optional[float]:
    if len(vals) < 4:
        return None
    q = statistics.quantiles(vals, n=4, method='inclusive')
    return round(q[2] - q[0], 3)


def _stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {'n': 0, 'median': None, 'iqr': None, 'min': None, 'max': None,
                'mean': None, 'stdev': None}
    return {'n': len(vals), 'median': round(statistics.median(vals), 3),
            'iqr': _iqr(vals), 'min': round(min(vals), 3), 'max': round(max(vals), 3),
            'mean': round(statistics.fmean(vals), 3),
            'stdev': round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0}


def page_list() -> List[Dict[str, Any]]:
    out = []
    for book in BOOKS:
        d = D.BOOKS / book
        if not d.is_dir():
            continue
        for p in sorted(d.glob('page_*.png'), key=lambda x: int(x.stem.split('_')[1])):
            page = int(p.stem.split('_')[1])
            out.append({'book': book, 'page': page, 'page_id': D.page_id(book, page),
                        'image_path': str(p.relative_to(ROOT)).replace('\\', '/')})
    return out


def run_page(rec: Dict[str, Any]) -> Dict[str, Any]:
    path = ROOT / rec['image_path']
    t0 = time.time()
    row: Dict[str, Any] = dict(rec)
    if not path.exists():
        row.update(status='missing', error='file_not_found', cols=[], n_cols=0)
        return row
    row['sha256'] = sha256_of(path)
    try:
        lay = page_layout(str(path))
    except Exception as exc:                                  # noqa: BLE001
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}', cols=[], n_cols=0,
                   timing_s=round(time.time() - t0, 3))
        return row
    cols = []
    for i, c in enumerate(lay.get('cols') or []):
        b = [int(v) for v in c['bbox']]
        cols.append({'col_index': i, 'bbox': b, 'width': b[2] - b[0], 'height': b[3] - b[1],
                     'half': [int(v) for v in c.get('half') or []],
                     'half_side': ('right' if c.get('half') and
                                   (b[0] + b[2]) / 2 >= (sum(c['half']) / 2) else 'left'),
                     'wide_split': bool(c.get('wide_split'))})
    widths = [c['width'] for c in cols]
    right_w = [c['width'] for c in cols if c['half_side'] == 'right']
    left_w = [c['width'] for c in cols if c['half_side'] == 'left']
    halves = [[int(v) for v in h] for h in (lay.get('halves') or [])]
    row.update(status='ok' if cols else 'empty_layout',
               image_size=[int(lay.get('width') or 0), int(lay.get('height') or 0)],
               halves=halves, cols=cols, n_cols=len(cols),
               half_stats={'right': _stats(right_w), 'left': _stats(left_w),
                           'all': _stats(widths)},
               n_cols_by_half={'right': len(right_w), 'left': len(left_w)},
               timing_s=round(time.time() - t0, 3),
               layout_source='ocr.vertical.page_layout（未改动）')
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='冻结旧布局列框')
    ap.add_argument('--out', default=str(OUT / 'frozen_layouts.json'))
    ap.add_argument('--only', nargs='*', default=None, help='只跑这些 page_id')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    pages = page_list()
    if args.only:
        keep = set(args.only)
        pages = [p for p in pages if p['page_id'] in keep]
    rows = []
    t0 = time.time()
    for i, rec in enumerate(pages, 1):
        rows.append(run_page(rec))
        if not args.quiet and i % 20 == 0:
            print(f'  {i}/{len(pages)} …', flush=True)

    by_book: Dict[str, Any] = {}
    for book in BOOKS:
        rs = [r for r in rows if r['book'] == book and r['status'] == 'ok']
        all_w = [c['width'] for r in rs for c in r['cols']]
        right_w = [c['width'] for r in rs for c in r['cols'] if c['half_side'] == 'right']
        left_w = [c['width'] for r in rs for c in r['cols'] if c['half_side'] == 'left']
        by_book[book] = {'pages': len(rs), 'cols': len(all_w),
                         'all': _stats(all_w), 'right': _stats(right_w), 'left': _stats(left_w)}

    payload = {
        'stage': 'stage3_dual_track_evaluation',
        'release': 'stage3-v1',
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage3_freeze_layouts.py',
        'entry': 'ocr.vertical.page_layout',
        'measured_object': '旧布局 column bbox 的宽度（px）',
        'not_measured_object': '人工 group_bbox 宽度（仅作目标尺寸描述，不得用于宽框标签）',
        'coord_system': '原图像素，半开区间 [x1,y1,x2,y2)',
        'scope': {'books': BOOKS, 'pages': len(rows)},
        'status_counts': {s: sum(1 for r in rows if r['status'] == s)
                          for s in sorted({r['status'] for r in rows})},
        'by_book': by_book,
        'pages': rows,
        'timing': {'total_s': round(time.time() - t0, 2),
                   'mean_s_per_page': round((time.time() - t0) / max(1, len(rows)), 3)},
        'notes': [
            'page_layout 内部会切双页（halves），bbox 已换回整页坐标',
            'wide_split=True 表示该列宽度 >1.8× 页内中位，被二次切开过',
            '列框宽度包含左右 padding 之外的实际墨迹范围，不含栏线',
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not args.quiet:
        print(f'冻结 {len(rows)} 页 → {D.rel(out)}')
        print(f'状态：{payload["status_counts"]}  用时 {payload["timing"]["total_s"]}s')
        for b, s in by_book.items():
            if s['cols']:
                print(f"  {b:20s} 页{s['pages']:3d} 列{s['cols']:4d} "
                      f"中位{s['all']['median']:.1f} IQR={s['all']['iqr']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
