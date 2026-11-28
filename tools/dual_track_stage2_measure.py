"""按「界行窗口」测量双轨组框：中缝取列内低墨量位置，上下界取墨迹极值。

思路：
  1. 列窗口 [x0,x1] 取两条界行之间；中缝 = 列宽中部 30%-70% 内墨量最小的 x；
  2. 双轨起点 = 中缝连续数行为空、且中缝两侧同时有墨的第一个 y；
  3. 主字框 = 起点以上的墨迹外接框；左右轨框 = 起点以下、中缝左右的墨迹外接框；
  4. 末行墨迹 +1 -> 半开区间 [x1,y1,x2,y2)。
只做测量，不做「这算不算一组」的判定。

用法：
  python tools/dual_track_stage2_measure.py 丹方彚编 12 --col 589,637 --y 200,824
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

EVID = D.ROOT / 'output/dual_tracks/stage2/evidence/measure'
_CACHE: Dict[str, np.ndarray] = {}


def _mask(book: str, page: int) -> np.ndarray:
    key = f'{book}/{page}'
    if key not in _CACHE:
        src = D.BOOKS / book / D.page_file_name(page)
        im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f'无法解码 {src}')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _CACHE[key] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 31, 14) > 0
    return _CACHE[key]


def _tight(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> Optional[List[int]]:
    sub = mask[y0:y1, x0:x1]
    rows = np.flatnonzero(sub.sum(axis=1) > 0)
    cols = np.flatnonzero(sub.sum(axis=0) > 0)
    if not rows.size or not cols.size:
        return None
    return [int(x0 + cols[0]), int(y0 + rows[0]), int(x0 + cols[-1]) + 1, int(y0 + rows[-1]) + 1]


def measure_group(book: str, page: int, x0: int, x1: int, y_top: int, y_bottom: int,
                  gutter_range: tuple = (0.30, 0.72), min_blank: int = 12) -> Dict[str, Any]:
    m = _mask(book, page)
    w = x1 - x0
    g0, g1 = x0 + int(w * gutter_range[0]), x0 + int(w * gutter_range[1])
    col = m[y_top:y_bottom, x0:x1].sum(axis=0)
    gutter = g0 + int(np.argmin(col[g0 - x0:g1 - x0]))

    def row_ink_side(y: int, a: int, b: int) -> int:
        return int(m[y, a:b].sum())

    lane_start = None
    for y in range(y_top + 40, y_bottom - 80):
        ok = True
        # 要求连续 min_blank 行都保持「中缝空 + 两侧各有墨」，避免把药名字间空隙当成双轨起点
        for k in range(min_blank):
            yy = y + k
            if yy >= y_bottom:
                ok = False
                break
            if row_ink_side(yy, gutter - 1, gutter + 2) > 0:
                ok = False
                break
            if row_ink_side(yy, x0, gutter - 1) < 3 or row_ink_side(yy, gutter + 2, x1) < 3:
                ok = False
                break
        if ok:
            lane_start = y
            break

    result: Dict[str, Any] = {
        'book': book, 'page': page,
        'column': [x0, y_top, x1, y_bottom], 'gutter_x': gutter,
        'lane_start': lane_start, 'status': 'ok',
        'main_bbox': None, 'right_bbox': None, 'left_bbox': None,
    }
    if lane_start is None:
        result['status'] = 'no_lane_start'
        result['main_bbox'] = _tight(m, x0, x1, y_top, y_bottom)
        return result

    # 末行：取「横向跨度不超过列宽 80%」的最后一行，排除版框横线（横线横跨全列）
    lane_bottom = lane_start
    for y in range(y_bottom - 1, lane_start, -1):
        cols = np.flatnonzero(m[y, x0:x1])
        if cols.size < 3:
            continue
        if (cols[-1] - cols[0] + 1) > 0.8 * w:
            continue  # 疑似版框横线/界行，跳过
        lane_bottom = y + 1
        break

    result.update({
        'lane_end': lane_bottom,
        'main_bbox': _tight(m, x0, x1, y_top, lane_start - 1),
        'right_bbox': _tight(m, gutter + 2, x1, lane_start, lane_bottom),
        'left_bbox': _tight(m, x0, gutter - 1, lane_start, lane_bottom),
    })
    return result


def draw(book: str, page: int, res: Dict[str, Any], out: Path, scale: float = 4.0,
         pad: int = 8) -> Path:
    src = D.BOOKS / book / D.page_file_name(page)
    im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = res['column']
    boxes = [res.get(k) for k in ('main_bbox', 'right_bbox', 'left_bbox')]
    boxes = [b for b in boxes if b]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    cy0, cy1 = max(0, min(ys) - pad), min(im.shape[0], max(ys) + pad)
    cx0, cx1 = max(0, x0 - pad), min(im.shape[1], x1 + pad)
    crop = cv2.resize(im[cy0:cy1, cx0:cx1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for key, color in (('main_bbox', (0, 215, 255)), ('right_bbox', (0, 0, 255)), ('left_bbox', (0, 160, 0))):
        b = res.get(key)
        if not b:
            continue
        p1 = (int((b[0] - cx0) * scale), int((b[1] - cy0) * scale))
        p2 = (int((b[2] - cx0) * scale), int((b[3] - cy0) * scale))
        cv2.rectangle(crop, p1, p2, color, 2)
    cv2.line(crop, (int((res['gutter_x'] - cx0) * scale), 0),
             (int((res['gutter_x'] - cx0) * scale), crop.shape[0]), (255, 0, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cv2.imencode('.png', crop)[1].tobytes())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='按界行窗口测量双轨组框')
    ap.add_argument('book')
    ap.add_argument('page', type=int)
    ap.add_argument('--col', nargs='+', required=True, help='x0,x1')
    ap.add_argument('--y', default='200,824')
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--draw', action='store_true')
    args = ap.parse_args(argv)
    y_top, y_bottom = [int(v) for v in args.y.split(',')]
    for c in args.col:
        x0, x1 = [int(v) for v in c.split(',')]
        res = measure_group(args.book, args.page, x0, x1, y_top, y_bottom)
        print(json.dumps(res, ensure_ascii=False))
        if args.draw and res['status'] == 'ok':
            name = f'{args.book}_p{args.page}_c{x0}_{x1}.png'
            print('  overlay', D.rel(draw(args.book, args.page, res, EVID / name, args.scale)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
