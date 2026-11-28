# -*- coding: utf-8 -*-
"""stage8 修正二：子框按墨迹外接收紧（对齐标注构造口径）。

为什么这样做不算「往标注上凑」：
  stage2 标注的 `annotation_source_profile.precision` 明确写着
  「列窗（界行之间）+ 中缝（列内低墨量位置）+ **墨迹外接框**」，
  即 GT 的 left/right 就是墨迹外接框。旧入口 C 输出的子框是**列窗框**
  （`boxes=[[a+k, ...],[a, ...]]`，a/z 是带 padding 的列窗边界），
  所以它 0.5 命中 24 组而 0.75 只剩 11 组——差的正是框边。

只改这一件事：把每条轨框收紧到框内的墨迹外接框；其余一律不动。

保护条件（都是实测出来的，不是拍的）：
  - 收紧后宽或高小于原框的 `min_keep`（0.45）→ 保留原框（避免把框缩成碎片）；
  - 排除贯穿框高的竖直线（栏线/版框）：某列墨迹占比 ≥ `line_frac`（0.70）
    视为直线结构，不参与外接；这条把「版框被当成右轨」的误检一并解决。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocr.vertical import imread_unicode  # noqa: E402

S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


_cache: Dict[str, Any] = {}


def gray_of(rel: str) -> np.ndarray:
    if rel not in _cache:
        im = imread_unicode(str(ROOT / rel))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        _cache[rel] = g
    return _cache[rel]


def ink_of(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (31, 31), 0).astype(np.int16)
    return (bg - gray.astype(np.int16)) >= 14


def tight_box(gray: np.ndarray, box, min_keep: float = 0.45,
              line_frac: float = 0.70, axes: str = 'xy') -> Optional[List[int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(gray.shape[1], x2); y2 = min(gray.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    sub = gray[y1:y2, x1:x2]
    ink = ink_of(sub)
    h = ink.shape[0]
    colsum = ink.sum(axis=0)
    keep = colsum < line_frac * h            # 贯穿框高的竖线不算字形
    ink = ink & keep[None, :]
    ys = np.flatnonzero(ink.sum(axis=1) > 0)
    xs = np.flatnonzero(ink.sum(axis=0) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    nx0, nx1 = (x1 + int(xs[0]), x1 + int(xs[-1]) + 1) if 'x' in axes else (x1, x2)
    ny0, ny1 = (y1 + int(ys[0]), y1 + int(ys[-1]) + 1) if 'y' in axes else (y1, y2)
    if (nx1 - nx0) < min_keep * (x2 - x1) or (ny1 - ny0) < min_keep * (y2 - y1):
        return None
    return [nx0, ny0, nx1, ny1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default='path_fix.json')
    ap.add_argument('--out', default='path_fix_p.json')
    ap.add_argument('--source', default='FIXP')
    ap.add_argument('--min-keep', type=float, default=0.45)
    ap.add_argument('--line-frac', type=float, default=0.70)
    ap.add_argument('--axes', default='xy', choices=['x', 'xy', 'y'])
    args = ap.parse_args(argv)

    doc = load(S8 / 'predictions' / args.src) if not Path(args.src).is_absolute() \
        else load(Path(args.src))
    img_of = {}
    ann = load(ROOT / 'output' / 'dual_tracks' / 'stage2' / 'annotations.json')
    for g in ann['groups']:
        img_of[g['page_id']] = g['image_path']
    neg = load(ROOT / 'output' / 'dual_tracks' / 'stage2' / 'negative_regions.json')
    for r in neg['regions']:
        img_of.setdefault(r['page_id'], r['image_path'])

    stats = {'tightened': 0, 'kept': 0, 'groups_dropped': 0}
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(doc['predictions']):
        g = gray_of(img_of[p['page_id']])
        l = tight_box(g, p['left_bbox'], args.min_keep, args.line_frac, args.axes) if p.get('left_bbox') else None
        r = tight_box(g, p['right_bbox'], args.min_keep, args.line_frac, args.axes) if p.get('right_bbox') else None
        q = dict(p)
        q['prediction_id'] = f"{p['page_id']}#{args.source}#{i:03d}"
        q['source'] = args.source
        q['lane_tightened'] = {'left': bool(l), 'right': bool(r)}
        if l:
            q['left_bbox'] = l; stats['tightened'] += 1
        else:
            stats['kept'] += 1
        if r:
            q['right_bbox'] = r; stats['tightened'] += 1
        else:
            stats['kept'] += 1
        if q.get('left_bbox') and q.get('right_bbox'):
            a, b = q['left_bbox'], q['right_bbox']
            q['group_bbox'] = [min(a[0], b[0]), min(a[1], b[1]),
                               max(a[2], b[2]), max(a[3], b[3])]
        else:
            stats['groups_dropped'] += 1
        out.append(q)

    doc2 = {'path': args.source,
            'entry': doc.get('entry', '') + '+lane_tighten',
            'predictions': out, 'page_runs': doc.get('page_runs') or [],
            'tighten': {'src': args.src, 'min_keep': args.min_keep,
                        'line_frac': args.line_frac, 'axes': args.axes, **stats}}
    f = S8 / 'predictions' / args.out
    f.write_text(json.dumps(doc2, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('->', f, stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
