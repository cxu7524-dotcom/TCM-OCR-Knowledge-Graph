# -*- coding: utf-8 -*-
"""stage6 证据：精修前后叠框、p12 两轨首尾放大、子框问题组证据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

import cv2  # noqa: E402
from ocr.vertical import imread_unicode  # noqa: E402
from dual_track_refine.api import S6, load  # noqa: E402
from dual_track_track.inputs import S2  # noqa: E402

E = S6 / 'evidence'
COL = {'before': (80, 80, 80), 'after': (0, 0, 255), 'gt': (0, 190, 0),
       'right': (255, 90, 0), 'left': (0, 200, 220)}


def _write(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        path.write_bytes(buf.tobytes())


def _rect(canvas, b, color, th=1, dash=False, off=(0, 0)):
    if not b:
        return
    x0, y0 = max(0, b[0] - off[0]), max(0, b[1] - off[1])
    x1, y1 = min(canvas.shape[1] - 1, b[2] - off[0]), min(canvas.shape[0] - 1, b[3] - off[1])
    if x1 <= x0 or y1 <= y0:
        return
    if dash:
        for k in range(x0, x1, 7):
            cv2.line(canvas, (k, y0), (min(k + 4, x1), y0), color, th)
            cv2.line(canvas, (k, y1), (min(k + 4, x1), y1), color, th)
        for k in range(y0, y1, 7):
            cv2.line(canvas, (x0, k), (x0, min(k + 4, y1)), color, th)
            cv2.line(canvas, (x1, k), (x1, min(k + 4, y1)), color, th)
    else:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, th)


def _crop(img, b, pad=22):
    x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
    x1, y1 = min(img.shape[1], b[2] + pad), min(img.shape[0], b[3] + pad)
    return img[y0:y1, x0:x1].copy(), (x0, y0)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=str(S6 / 'groups.json'))
    ap.add_argument('--out-dir', default=str(E))
    args = ap.parse_args(argv)
    payload = load(Path(args.groups))
    ann = load(S2 / 'annotations.json')
    gt_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in ann['groups']:
        gt_by_page.setdefault(g['page_id'], []).append(g)
    problem = {'丹方彚编/p12/DF12-03', '丹方彚编/p12/DF12-04',
               '丹方彚编/p12/DF12-11', '思濟堂方書/p35/dev-35-001',
               '思济堂方书/p46/P46-08', '思濟堂方書/p46/P46-08'}
    n = 0
    for page in payload['pages']:
        pid = page['page_id']
        img = imread_unicode(str(ROOT / page['image_path']))
        if img is None:
            continue
        for rec in page['records']:
            gid = rec['group_id']
            union = rec['before']['group_bbox'] or rec['after']['group_bbox']
            if not union:
                continue
            crop, off = _crop(img, union, pad=26)
            for key, color, dash in (('before', COL['before'], True),
                                     ('after', COL['after'], False)):
                for side in ('right', 'left'):
                    _rect(crop, rec[key][f'{side}_bbox'],
                          COL[side] if key == 'after' else color,
                          1 if key == 'after' else 1, dash, off)
            _rect(crop, rec['before']['group_bbox'], COL['before'], 1, True, off)
            _rect(crop, rec['after']['group_bbox'], COL['after'], 2, False, off)
            for gt in gt_by_page.get(pid, []):
                if _ov(gt['group_bbox'], union) > 0:
                    _rect(crop, gt['group_bbox'], COL['gt'], 1, True, off)
            label = f"{gid.split('/')[-1]} {'P' if rec['pending'] else '-'}"
            cv2.putText(crop, label, (3, crop.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (20, 20, 20), 1, cv2.LINE_AA)
            safe = gid.replace('/', '__')
            _write(Path(args.out_dir) / 'per_group' / f'{safe}.jpg', crop)
            n += 1
            if gid in problem:
                _write(Path(args.out_dir) / 'problem' / f'{safe}.jpg', crop)
        # p12 两轨首尾放大
        if pid == '丹方彚编/page_12':
            for gid, short in (('丹方彚编/page_12#g006', 'DF12-03'),
                               ('丹方彚编/page_12#g036', 'DF12-11')):
                rec = next((r for r in page['records'] if r['group_id'] == gid), None)
                if not rec:
                    continue
                for side in ('right', 'left'):
                    b = rec['after'][f'{side}_bbox']
                    if not b:
                        continue
                    for end, y in (('head', b[1]), ('tail', b[3])):
                        box = [max(0, b[0] - 18), max(0, y - 26),
                               min(img.shape[1], b[2] + 18), min(img.shape[0], y + 26)]
                        crop2, off2 = _crop(img, box, pad=0)
                        for key, color in (('before', COL['before']),
                                           ('after', COL[side])):
                            _rect(crop2, rec[key][f'{side}_bbox'], color, 2,
                                  key == 'before', off2)
                        cv2.putText(crop2, f'{short} {side} {end}',
                                    (3, crop2.shape[0] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20),
                                    1, cv2.LINE_AA)
                        _write(Path(args.out_dir) / 'p12_zoom' /
                               f'{short}__{side}__{end}.jpg', crop2)
                        n += 1
    print(f'输出 {n} 张 → {args.out_dir}')
    return 0


def _ov(a, b) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0, min(a[3], b[3]) - max(a[1], b[1]))


if __name__ == '__main__':
    raise SystemExit(main())
