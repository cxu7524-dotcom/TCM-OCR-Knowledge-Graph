"""生成规范化的叠框放大证据图（每组一张，6 倍），供人工核对字形与框边界。

用法：
  .venv/Scripts/python.exe tools/dual_track_stage1_evidence.py
  .venv/Scripts/python.exe tools/dual_track_stage1_evidence.py --only P35 dev-35-001

读取 output/dual_tracks/stage1/normalized_annotations.json，不改动来源图片；
产物写到 output/dual_tracks/stage1/evidence/<group_id>.png。
黄 = 主字，红 = 右轨，绿 = 左轨，蓝虚线 = 左轨与右轨的并集（group_bbox）。
半开区间按 x2-1 / y2-1 画最后一像素行，端点待核的按原值画。
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

SCALE = 6
PAD = 12
COLORS = {'main': (0, 215, 255), 'right': (0, 0, 255), 'left': (0, 160, 0), 'group': (230, 150, 40)}


def load_groups(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))['groups']


def draw_group(group: Dict[str, Any], out_dir: Path) -> Path:
    img_path = D.ROOT / group['image_path']
    buf = np.fromfile(str(img_path), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f'无法解码 {img_path}')

    boxes = {}
    for key in ('main_bbox', 'right_bbox', 'left_bbox', 'group_bbox'):
        x1, y1, x2, y2 = group[key]
        if group['coordinate_convention'] == D.CONV_HALF_OPEN:
            boxes[key] = (x1, y1, x2 - 1, y2 - 1)
        else:
            boxes[key] = (x1, y1, x2, y2)

    xs = [b[0] for b in boxes.values()] + [b[2] for b in boxes.values()]
    ys = [b[1] for b in boxes.values()] + [b[3] for b in boxes.values()]
    cx1, cx2 = max(0, min(xs) - PAD), min(im.shape[1], max(xs) + PAD + 1)
    cy1, cy2 = max(0, min(ys) - PAD), min(im.shape[0], max(ys) + PAD + 1)
    crop = im[cy1:cy2, cx1:cx2]
    big = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    def tr(b):
        return ((b[0] - cx1) * SCALE, (b[1] - cy1) * SCALE), ((b[2] - cx1) * SCALE, (b[3] - cy1) * SCALE)

    for key, color in (('group_bbox', COLORS['group']), ('main_bbox', COLORS['main']),
                       ('right_bbox', COLORS['right']), ('left_bbox', COLORS['left'])):
        p1, p2 = tr(boxes[key])
        if key == 'group_bbox':
            for x in range(p1[0], p2[0], 12):  # 虚线
                cv2.line(big, (x, p1[1]), (min(x + 6, p2[0]), p1[1]), color, 1)
                cv2.line(big, (x, p2[1]), (min(x + 6, p2[0]), p2[1]), color, 1)
            for y in range(p1[1], p2[1], 12):
                cv2.line(big, (p1[0], y), (p1[0], min(y + 6, p2[1])), color, 1)
                cv2.line(big, (p2[0], y), (p2[0], min(y + 6, p2[1])), color, 1)
        else:
            cv2.rectangle(big, p1, p2, color, 2)

    label = f"{group['region']}  {group['annotation_source']}  {group['coordinate_convention']}"
    cv2.rectangle(big, (0, 0), (min(big.shape[1], 12 * len(label) // 2 + 10), 22), (0, 0, 0), -1)
    cv2.putText(big, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    safe = group['group_id'].replace('/', '__')
    out = out_dir / f'{safe}.png'
    out.write_bytes(cv2.imencode('.png', big)[1].tobytes())
    return out


def render_half_page(book: str, page: int, out_dir: Path, scale: float = 1.6) -> Path:
    """右半页整幅（用于交代数据集口径未覆盖的页面形态，如 丹方彚編 p12）。"""
    src = D.BOOKS / book / D.page_file_name(page)
    im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f'无法解码 {src}')
    h, w = im.shape[:2]
    crop = cv2.resize(im[0:h, w // 2:w], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = out_dir / f'_scope_{book}_p{page}_右半頁.png'
    out.write_bytes(cv2.imencode('.png', crop)[1].tobytes())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成双轨规范化叠框放大证据图')
    ap.add_argument('--annotations', default=str(D.OUT_DIR / 'normalized_annotations.json'))
    ap.add_argument('--out-dir', default=str(D.OUT_DIR / 'evidence'))
    ap.add_argument('--only', nargs='*', default=None, help='按 region 或 group_id 过滤')
    ap.add_argument('--half-page', nargs=2, metavar=('BOOK', 'PAGE'), default=None,
                    help='额外输出某一页的右半页整幅（不画框）')
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.half_page:
        print('右半页：', D.rel(render_half_page(args.half_page[0], int(args.half_page[1]), out_dir)))

    groups = load_groups(Path(args.annotations))
    written = []
    for g in groups:
        if args.only and not any(k in g['region'] or k in g['group_id'] for k in args.only):
            continue
        written.append(D.rel(draw_group(g, out_dir)))
    print(f'输出 {len(written)} 张叠框放大图到 {D.rel(out_dir)}')
    for w in written:
        print(' ', w)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
