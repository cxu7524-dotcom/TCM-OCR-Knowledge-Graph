# -*- coding: utf-8 -*-
"""第四阶段核对图：把半页边界、范围外、搜索区域、栏线、标注组画在同一张原图上。

只为人工核对，不参与算法。v2 新增：
  - 蓝色虚线 = 半页边界（区域内不该有框跨过去）
  - 灰色斜纹带 = 范围外（中缝/页边，本就不该出区域）
  - 栏线按类型上色：品红=独立（可定栏），橙=贴字（只作诊断）
  - 页眉写状态、区域数、完成度、页面 H
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

import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402

import dual_track_stage1_data as D             # noqa: E402
from ocr.vertical import imread_unicode        # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage4'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'

COLOR = {
    'grouped_rules': (60, 200, 60),        # 绿：成组栏线
    'validated_columns': (60, 60, 220),    # 红：校验列框
    'halfpage_fallback': (0, 200, 255),    # 黄：低置信窗口
    'rejected': (128, 0, 255),             # 紫：被拒绝的旧框
    'group': (255, 200, 0),                # 青：标注组
    'rule_independent': (255, 0, 255),     # 品红：独立栏线
    'rule_touching': (0, 140, 255),        # 橙：贴字栏线
    'half': (255, 90, 0),                  # 蓝：半页边界
    'out_of_scope': (150, 150, 150),       # 灰：范围外
}


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def draw(page: Dict[str, Any], img, groups: List[Dict[str, Any]],
         box: Optional[List[int]] = None, scale: float = 2.0,
         with_rules: bool = True, with_scope: bool = True):
    x0, y0, x1, y1 = box if box else [0, 0, img.shape[1], img.shape[0]]
    canvas = img[y0:y1, x0:x1].copy()

    def rect(b, color, thickness=2, dash=False):
        bx1, by1, bx2, by2 = [int(v) for v in b]
        bx1, by1 = max(0, bx1 - x0), max(0, by1 - y0)
        bx2, by2 = min(canvas.shape[1] - 1, bx2 - x0), min(canvas.shape[0] - 1, by2 - y0)
        if bx2 <= bx1 or by2 <= by1:
            return
        if dash:
            for k in range(bx1, bx2, 8):
                cv2.line(canvas, (k, by1), (min(k + 4, bx2), by1), color, 1)
                cv2.line(canvas, (k, by2), (min(k + 4, bx2), by2), color, 1)
            for k in range(by1, by2, 8):
                cv2.line(canvas, (bx1, k), (bx1, min(k + 4, by2)), color, 1)
                cv2.line(canvas, (bx2, k), (bx2, min(k + 4, by2)), color, 1)
        else:
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness)

    # 范围外：灰色斜纹（中缝/页边）
    if with_scope:
        overlay = canvas.copy()
        for o in page.get('out_of_scope') or []:
            a, b = o['range']
            a, b = max(0, a - x0), min(canvas.shape[1], b - x0)
            if b <= a:
                continue
            for y in range(0, canvas.shape[0], 6):
                cv2.line(overlay, (a, y), (min(b, a + 6), min(canvas.shape[0], y + 6)),
                         COLOR['out_of_scope'], 1)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    # 旧布局里被拒绝的框（紫）
    for hid, cvres in (page.get('column_validation') or {}).items():
        for v in cvres.get('rejected_detail') or []:
            b = v['bbox']
            if b[2] <= x0 or b[0] >= x1:
                continue
            rect(b, COLOR['rejected'], 2, dash=True)
            cv2.putText(canvas, f"rejected {v['width']}px",
                        (max(1, b[0] - x0 + 2), max(12, b[1] - y0 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR['rejected'], 1)
    # 栏线：独立（品红）/ 贴字（橙）
    if with_rules:
        for c in page.get('rule_lines') or page.get('rule_candidates') or []:
            cx = int((c['x0'] + c['x1']) // 2) - x0
            if not (0 <= cx < canvas.shape[1]):
                continue
            col = COLOR['rule_independent'] if c['kind'] in ('independent', 'broken') \
                else COLOR['rule_touching']
            cv2.line(canvas, (cx, max(0, c['y0'] - y0)),
                     (cx, min(canvas.shape[0], c['y1'] - y0)), col, 1)
    # 搜索区域
    for r in page['regions']:
        b = r['bbox']
        if b[2] <= x0 or b[0] >= x1:
            continue
        col = COLOR.get(r['column_range_source'], (200, 200, 200))
        rect(b, col, 1, dash=bool(r['low_confidence']))
        tag = {'grouped_rules': 'R', 'validated_columns': 'C',
               'halfpage_fallback': 'F'}[r['column_range_source']]
        if r.get('clipped'):
            tag += '*'
        cv2.putText(canvas, tag, (max(1, b[0] - x0 + 2), max(10, b[1] - y0 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    # 标注组
    for g in groups:
        b = g['group_bbox']
        if b[2] <= x0 or b[0] >= x1:
            continue
        rect(b, COLOR['group'], 2)
    # 半页边界
    halves = page.get('halves') or []
    for hx0, hx1 in halves:
        for hx in (hx0, hx1):
            if x0 <= hx <= x1:
                cx = hx - x0
                for k in range(0, canvas.shape[0], 10):
                    cv2.line(canvas, (cx, k), (cx, min(canvas.shape[0], k + 5)),
                             COLOR['half'], 2)

    big = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    comp = ((page.get('coverage') or {}).get('completion') or {})
    head = (f"{page['page_id']}  {page['status']}  regions={len(page['regions'])}  "
            f"H={((page.get('page_scale') or {}).get('H'))}  "
            f"proc={comp.get('processed_ratio')}  gaps={comp.get('uncovered_areas')}")
    cv2.rectangle(big, (0, 0), (big.shape[1], 24), (28, 28, 28), -1)
    cv2.putText(big, head, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    # 图例
    legend = [('R grouped_rules', COLOR['grouped_rules']),
              ('C validated_columns', COLOR['validated_columns']),
              ('F fallback', COLOR['halfpage_fallback']),
              ('| half boundary', COLOR['half']),
              ('| independent rule', COLOR['rule_independent']),
              ('| touching rule', COLOR['rule_touching'])]
    for i, (text, col) in enumerate(legend):
        cv2.putText(big, text, (5, big.shape[0] - 8 - i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    return big


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage4 核对图')
    ap.add_argument('--regions', default=str(OUT / 'search_regions.json'))
    ap.add_argument('--pages', nargs='*', default=None)
    ap.add_argument('--out-dir', default=str(OUT / 'evidence'))
    args = ap.parse_args(argv)

    payload = load(Path(args.regions))
    ann = load(S2 / 'annotations.json')
    g_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in ann['groups']:
        g_by_page.setdefault(g['page_id'], []).append(g)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for page in payload['pages']:
        pid = page['page_id']
        if args.pages and pid not in args.pages:
            continue
        book, num = pid.split('/page_')
        img = imread_unicode(str(D.BOOKS / book / D.page_file_name(int(num))))
        if img is None:
            continue
        groups = g_by_page.get(pid, [])
        full = draw(page, img, groups, scale=1.0)
        f1 = out / f"{book}__p{num}__overview.jpg"
        cv2.imencode('.jpg', full, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(f1))
        written.append(str(f1.relative_to(ROOT)))
        W = img.shape[1]
        halves = page.get('halves') or [[0, W]]
        for name, k in (('right', 0), ('left', 1)):
            if k >= len(halves):
                continue
            hx = halves[k]
            big = draw(page, img, groups, box=[hx[0], 0, hx[1], img.shape[0]], scale=3.0)
            f2 = out / f"{book}__p{num}__{name}.jpg"
            cv2.imencode('.jpg', big, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(str(f2))
            written.append(str(f2.relative_to(ROOT)))
    print(f'输出 {len(written)} 张 → {out.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
