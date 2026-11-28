# -*- coding: utf-8 -*-
"""逐页核对图：标注（绿）与预测（红=命中 / 橙=错检 / 灰=待核）叠在原图上。

命令行：
  python tools/dual_track_stage3_evidence.py --path C
  python tools/dual_track_stage3_evidence.py --path C --pages 丹方彚编/page_12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D          # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
METRIC_FILE = {'B': 'b.json', 'C': 'c.json', 'X': 'x_components.json'}
COLORS = {'TP': (60, 60, 220), 'FP': (40, 140, 240), 'UNMATCHED_PENDING': (150, 150, 150)}


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def draw(page: Dict[str, Any], image: Any, scale: float = 1.3) -> Any:
    import cv2
    img = image.copy()
    # 标注框（绿）——坐标从 stage2 标注直接读，避免与 metrics 的索引顺序耦合
    for box in page.get('_gt_boxes') or []:
        cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), (40, 160, 40), 2)
    for r in page['thresholds']['0.5']['predictions']:
        b = r['group_bbox']
        if not b:
            continue
        col = COLORS.get(r['label'], (150, 150, 150))
        cv2.rectangle(img, (b[0], b[1]), (b[2] - 1, b[3] - 1), col, 2)
        if r['label'] == 'TP':
            cv2.putText(img, 'HIT', (b[0], max(10, b[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    t = page['thresholds']['0.5']
    head = (f"{page['page_id']}  GT {t['annotations_total']}  hit {t['group_hits']}  "
            f"pred {page['counts']['predictions']}  cand {t['new_candidates']}")
    cv2.rectangle(big, (0, 0), (big.shape[1], 20), (25, 25, 25), -1)
    cv2.putText(big, head, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    return big


def main(argv: Optional[List[str]] = None) -> int:
    import cv2
    from ocr.vertical import imread_unicode

    ap = argparse.ArgumentParser(description='生成逐页核对图')
    ap.add_argument('--path', default='C', choices=['B', 'C', 'X'])
    ap.add_argument('--pages', nargs='*', default=None)
    ap.add_argument('--out-dir', default=str(OUT / 'evidence' / 'pages'))
    ap.add_argument('--scale', type=float, default=1.3)
    args = ap.parse_args(argv)

    mpath = OUT / 'metrics' / METRIC_FILE[args.path.upper()]
    if not mpath.exists():
        print(f'缺 {mpath}，先跑 dual_track_stage3_score.py')
        return 2
    metrics = load(mpath)
    ann = load(S2 / 'annotations.json')
    gt_by_page: Dict[str, List[List[int]]] = {}
    for g in ann['groups']:
        gt_by_page.setdefault(g['page_id'], []).append(g['group_bbox'])

    out_dir = Path(args.out_dir) / args.path.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page in metrics['pages']:
        pid = page['page_id']
        if args.pages and pid not in args.pages:
            continue
        book, num = pid.split('/page_')
        img = imread_unicode(str(D.BOOKS / book / D.page_file_name(int(num))))
        if img is None:
            print(f'  跳过 {pid}：图不可读')
            continue
        page = dict(page, _gt_boxes=gt_by_page.get(pid, []))
        big = draw(page, img, args.scale)
        f = out_dir / f"{book}__p{num}.jpg"
        cv2.imencode('.jpg', big, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(f))
        written.append(str(f.relative_to(ROOT)))
    print(f'输出 {len(written)} 张 → {out_dir.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
