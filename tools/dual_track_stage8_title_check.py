# -*- coding: utf-8 -*-
"""A3 离线验收：用**冻结标注的药名框（main_bbox）**核对轨框是否侵入标题。

与 v2 的差别：v2 从检测器 candidate 里筛「大字」，再拿 OCR 字高比筛掉，
结论是「本页无大字 → 不适用」。用户要求改成直接拿冻结药名框做离线验收：
  - 对每条预测轨框，按 x 落在哪一列确定本组；本组药名框 + 同列下一个药名框
    都要比对（"核对本组及下一标题"）；
  - 记录实际比对框数、每个药名框与轨框的 y 交叠位置；
  -  penal 判据：轨框与药名框 y 交叠 > 0.35×药名框高 视为侵入。

输出：output/dual_tracks/stage8/acceptance/title_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
PAGE = '丹方彚编/page_12'
OUT = S8 / 'acceptance' / 'title_check.json'


def y_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return max(0, min(a[3], b[3]) - max(a[1], b[1]))


def x_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='path_online_v4.json',
                    help='待验收预测（默认=同一次正常 OCR 入口产物）')
    args = ap.parse_args()
    ann = json.loads((S2 / 'annotations.json').read_text(encoding='utf-8'))
    gts = [g for g in ann.get('groups') or []
           if g.get('page_id') == PAGE and g.get('main_bbox')]
    # 按 x 中心把药名框分组（同列上下多个条目）
    by_x: Dict[int, List[Dict[str, Any]]] = {}
    for g in gts:
        m = g['main_bbox']
        key = (m[0] + m[2]) // 2
        by_x.setdefault(key, []).append(g)
    for gs in by_x.values():
        gs.sort(key=lambda g: g['main_bbox'][1])

    pred = json.loads((S8 / 'predictions' / args.pred).read_text(encoding='utf-8'))
    ps = [p for p in pred['predictions'] if p.get('page_id') == PAGE]
    lanes = []
    for p in ps:
        for side in ('left', 'right'):
            b = p.get(f'{side}_bbox')
            if b:
                lanes.append({'prediction_id': p['prediction_id'], 'side': side,
                              'bbox': b, 'source': p.get('source'),
                              'side_status': (p.get('side_status') or {}).get(side),
                              'status_reason': (p.get('status_reason') or {}).get(side)})

    checks: List[Dict[str, Any]] = []
    n_boxes = 0
    for ln in lanes:
        b = ln['bbox']
        # 与轨框 x 有交叠的药名框，按 y 排序
        cand = []
        for key, gs in by_x.items():
            for g in gs:
                m = g['main_bbox']
                if x_overlap(b, m) > max(8, 0.3 * min(b[2] - b[0], m[2] - m[0])):
                    cand.append(g)
        cand.sort(key=lambda g: g['main_bbox'][1])
        own = next((g for g in cand if y_overlap(b, g['main_bbox']) > 0
                    or g['main_bbox'][1] <= b[1]), None)
        nxt = next((g for g in cand if own and g['main_bbox'][1] >=
                    own['main_bbox'][3]), None)
        pairs = []
        for tag, g in (('own', own), ('next', nxt)):
            if not g:
                continue
            m = g['main_bbox']
            iy = y_overlap(b, m)
            n_boxes += 1
            viol = iy > 0.35 * (m[3] - m[1])
            pairs.append({'role': tag, 'region': g.get('region'),
                          'main_bbox': m, 'main_text': g.get('main_text'),
                          'overlap_y': iy, 'overlap_ratio': round(iy / (m[3] - m[1]), 3),
                          'intrusion': viol})
        if pairs:
            checks.append({'prediction_id': ln['prediction_id'], 'side': ln['side'],
                           'bbox': b, 'source': ln['source'], 'title_pairs': pairs})

    intrusions = [c for c in checks
                  if any(p['intrusion'] for p in c['title_pairs'])]
    doc = {
        'stage': 'stage8_v4_A3_title_offline', 'page_id': PAGE,
        'prediction_source': args.pred,
        'criterion': '冻结合作药名框 main_bbox；y 交叠 > 0.35×药名框高 判侵入',
        'n_frozen_title_boxes': len(gts), 'n_lanes': len(lanes),
        'n_lanes_with_title_pair': len(checks), 'n_boxes_compared': n_boxes,
        'n_intrusion_lanes': len(intrusions),
        'pass': bool(n_boxes) and not intrusions,
        'intrusion_examples': intrusions[:8],
        'all_checks': checks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f"冻结药名框 {len(gts)}；轨 {len(lanes)}；有比对 {len(checks)}；"
          f"实比框数 {n_boxes}；侵入轨 {len(intrusions)}；PASS={doc['pass']}")
    for c in intrusions[:5]:
        print('  ', c['prediction_id'], c['side'], c['bbox'],
              [(p['role'], p['region'], p['overlap_y']) for p in c['title_pairs']])
    print(f'-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
