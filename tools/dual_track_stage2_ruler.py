"""标尺图生成器：把原图的指定区域放大，叠加像素网格与坐标刻度，供人工读坐标。

用法：
  python tools/dual_track_stage2_ruler.py 丹方彚编 12 --box 500,0,1024,887 --scale 1.8 --grid 50
  python tools/dual_track_stage2_ruler.py 思濟堂方書 40 --box 0,0,1024,805 --scale 1.2 --grid 100

只读原图，产物写到 output/dual_tracks/stage2/evidence/ruler/。
坐标一律是原图像素（左上角原点）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_track_stage1_data as D  # noqa: E402

OUT = D.ROOT / 'output/dual_tracks/stage2/evidence/ruler'


def render(book: str, page: int, box: Sequence[int], scale: float, grid: int,
           out: Path, boxes: Optional[List[Sequence[int]]] = None) -> Path:
    src = D.BOOKS / book / D.page_file_name(page)
    im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f'无法解码 {src}')
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = im[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = big.shape[:2]

    # 网格：每条线标原图坐标
    for gx in range(x0 - x0 % grid, x1 + 1, grid):
        px = int((gx - x0) * scale)
        if 0 <= px < w:
            cv2.line(big, (px, 0), (px, h), (200, 220, 255), 1)
            cv2.putText(big, str(gx), (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
    for gy in range(y0 - y0 % grid, y1 + 1, grid):
        py = int((gy - y0) * scale)
        if 0 <= py < h:
            cv2.line(big, (0, py), (w, py), (200, 220, 255), 1)
            cv2.putText(big, str(gy), (2, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)

    for b in boxes or []:
        p1 = (int((b[0] - x0) * scale), int((b[1] - y0) * scale))
        p2 = (int((b[2] - x0) * scale), int((b[3] - y0) * scale))
        cv2.rectangle(big, p1, p2, (0, 0, 255), 2)
        cv2.putText(big, f'{b[0]},{b[1]},{b[2]},{b[3]}', (p1[0] + 3, max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cv2.imencode('.png', big)[1].tobytes())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成带像素网格的标尺图')
    ap.add_argument('book')
    ap.add_argument('page', type=int)
    ap.add_argument('--box', required=True, help='x0,y0,x1,y1')
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--grid', type=int, default=50)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    box = [int(v) for v in args.box.split(',')]
    name = args.out or f"{args.book}_p{args.page}_{box[0]}_{box[1]}_{box[2]}_{box[3]}_s{args.scale}.png"
    print(D.rel(render(args.book, args.page, box, args.scale, args.grid, OUT / name)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
