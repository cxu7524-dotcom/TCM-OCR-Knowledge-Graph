# -*- coding: utf-8 -*-
"""第五阶段关键组叠框图：组框、右/左轨、中缝轨迹、断点位置、标注组对照。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2                                     # noqa: E402

import dual_track_stage1_data as D             # noqa: E402
from ocr.vertical import imread_unicode        # noqa: E402

S5 = ROOT / 'output' / 'dual_tracks' / 'stage5'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'

COLOR = {'group': (60, 200, 60), 'right': (60, 60, 220), 'left': (220, 120, 0),
         'seam': (255, 0, 255), 'stop': (0, 220, 255), 'gt': (255, 200, 0)}


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def draw_page(page: Dict[str, Any], img, gts: List[Dict[str, Any]],
              box: Optional[List[int]] = None, scale: float = 2.0):
    x0, y0, x1, y1 = box if box else [0, 0, img.shape[1], img.shape[0]]
    canvas = img[y0:y1, x0:x1].copy()

    def rect(b, color, th=1, dash=False):
        a, c = max(0, b[0] - x0), max(0, b[1] - y0)
        e, f = min(canvas.shape[1] - 1, b[2] - x0), min(canvas.shape[0] - 1, b[3] - y0)
        if e <= a or f <= c:
            return
        if dash:
            for k in range(a, e, 8):
                cv2.line(canvas, (k, c), (min(k + 4, e), c), color, 1)
                cv2.line(canvas, (k, f), (min(k + 4, e), f), color, 1)
        else:
            cv2.rectangle(canvas, (a, c), (e, f), color, th)

    for gt in gts:
        rect(gt['group_bbox'], COLOR['gt'], 1, dash=True)
    for g in page.get('groups') or []:
        if g.get('left_bbox'):
            rect(g['left_bbox'], COLOR['left'], 1)
        if g.get('right_bbox'):
            rect(g['right_bbox'], COLOR['right'], 1)
        if g.get('group_bbox'):
            rect(g['group_bbox'], COLOR['group'], 2)
        pts = [(int(p['x']) - x0, int(p['y']) - y0) for p in g['seam']['points']]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, COLOR['seam'], 1)
        for e in g.get('stop_events') or []:
            yy = int(e['at_y']) - y0
            if 0 <= yy < canvas.shape[0]:
                cv2.line(canvas, (max(0, (g['group_bbox'][0] - x0)), yy),
                         (min(canvas.shape[1] - 1, g['group_bbox'][2] - x0), yy),
                         COLOR['stop'], 1)
                cv2.putText(canvas, e['type'][:12],
                            (max(1, g['group_bbox'][0] - x0 + 2), min(canvas.shape[0] - 3,
                                                                      yy + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR['stop'], 1)
    big = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    head = (f"{page['page_id']}  groups={len(page.get('groups') or [])}  "
            f"state={page['status']}")
    cv2.rectangle(big, (0, 0), (big.shape[1], 22), (28, 28, 28), -1)
    cv2.putText(big, head, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return big


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=str(S5 / 'groups.json'))
    ap.add_argument('--pages', nargs='*', default=None)
    ap.add_argument('--out-dir', default=str(S5 / 'evidence'))
    args = ap.parse_args(argv)
    payload = load(Path(args.groups))
    ann = load(S2 / 'annotations.json')
    gts: Dict[str, List[Dict[str, Any]]] = {}
    for g in ann['groups']:
        gts.setdefault(g['page_id'], []).append(g)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for page in payload['pages']:
        if args.pages and page['page_id'] not in args.pages:
            continue
        if not page.get('groups'):
            continue
        book, num = page['page_id'].split('/page_')
        img = imread_unicode(str(D.BOOKS / book / D.page_file_name(int(num))))
        if img is None:
            continue
        full = draw_page(page, img, gts.get(page['page_id'], []), scale=1.0)
        f = out / f"{book}__p{num}__groups.jpg"
        cv2.imencode('.jpg', full, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(str(f))
        n += 1
        H = img.shape[0]
        for k, name in ((0, 'right'), (1, 'left')):
            halves = page.get('halves') or []
            if k >= len(halves):
                continue
            hx = halves[k]
            big = draw_page(page, img, gts.get(page['page_id'], []),
                            box=[hx[0], 0, hx[1], H], scale=2.5)
            f2 = out / f"{book}__p{num}__{name}.jpg"
            cv2.imencode('.jpg', big, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(str(f2))
            n += 1
    print(f'输出 {n} 张 → {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
