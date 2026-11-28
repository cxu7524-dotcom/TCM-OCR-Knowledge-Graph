"""生成丹方 p12 修正前后的逐组对照图（左=修正前 v1，右=修正后 v2）。

用法：
  python tools/dual_track_stage2_diff.py
  python tools/dual_track_stage2_diff.py --only DF12-03 DF12-11

读取：
  output/dual_tracks/stage2/evidence/df12_annotations_before_fix.json（修正前快照）
  output/dual_tracks/stage2/annotations.json（修正后）
产物：
  output/dual_tracks/stage2/evidence/fix_df12/<region>_before_after.png
  output/dual_tracks/stage2/evidence/fix_df12/_montage_1.png（1-6）/_montage_2.png（7-11）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_track_stage1_data as D  # noqa: E402

OUT = D.ROOT / 'output/dual_tracks/stage2'
BEFORE = OUT / 'evidence/df12_annotations_before_fix.json'
AFTER = OUT / 'annotations.json'
EVID = OUT / 'evidence/fix_df12'


def load_groups(path: Path, book: str, page: int) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return {g['region']: g for g in data['groups']
            if g['book'] == book and int(g['page']) == page}


def render_pair(book: str, page: int, before: Dict[str, Any], after: Dict[str, Any],
                scale: float = 2.2) -> np.ndarray:
    src = D.BOOKS / book / D.page_file_name(page)
    im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    boxes = [b for g in (before, after)
             for b in (g['main_bbox'], g['right_bbox'], g['left_bbox'])]
    x0 = max(0, min(b[0] for b in boxes) - 8)
    x1 = min(im.shape[1], max(b[2] for b in boxes) + 8)
    y0 = max(0, min(b[1] for b in boxes) - 8)
    y1 = min(im.shape[0], max(b[3] for b in boxes) + 8)
    panels = []
    for g, label in ((before, 'BEFORE (v1)'), (after, 'AFTER (v2)')):
        c = im[y0:y1, x0:x1].copy()
        for key, color in (('main_bbox', (0, 215, 255)), ('right_bbox', (0, 0, 255)),
                           ('left_bbox', (0, 160, 0))):
            b = g[key]
            cv2.rectangle(c, (b[0] - x0, b[1] - y0), (b[2] - x0 - 1, b[3] - y0 - 1), color, 1)
        c = cv2.copyMakeBorder(c, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(c, f"{g['region']} {label}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1)
        panels.append(cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
              for p in panels]
    return np.hstack(panels)


def montage(images: List[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    images = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 10, cv2.BORDER_CONSTANT,
                                 value=(25, 25, 25)) for i in images]
    return np.hstack(images)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成丹方 p12 修正前后对照图')
    ap.add_argument('--book', default='丹方彚编')
    ap.add_argument('--page', type=int, default=12)
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args(argv)

    before = load_groups(BEFORE, args.book, args.page)
    after = load_groups(AFTER, args.book, args.page)
    regions = [r for r in sorted(after) if not args.only or r in args.only]
    EVID.mkdir(parents=True, exist_ok=True)
    made = []
    for r in regions:
        pair = render_pair(args.book, args.page, before[r], after[r])
        path = EVID / f'{r}_before_after.png'
        path.write_bytes(cv2.imencode('.png', pair)[1].tobytes())
        made.append(pair)
        print(D.rel(path))
    for idx, chunk in enumerate([made[:6], made[6:]], start=1):
        if not chunk:
            continue
        m = montage(chunk)
        p = EVID / f'_montage_{idx}.png'
        p.write_bytes(cv2.imencode('.png', m)[1].tobytes())
        print(D.rel(p))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
