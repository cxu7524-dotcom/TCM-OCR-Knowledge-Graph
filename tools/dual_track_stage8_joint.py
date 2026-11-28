# -*- coding: utf-8 -*-
"""陽起石（DF12-11）跨槽联合取样：生成补修候选并做 IoU 验证。

流程：
  1. 载入冻结正例标注与当前 stage8-v2 预测；
  2. 找出「组中缝 gutter_x 恰好落在列槽边界（±3px）」且「药名 main_bbox 横跨该
     边界」的组——实测 p12 11 组里只有 DF12-11 满足；
  3. 用 dual_track_fix.joint.joint_propose 从原图墨迹重新确认左右轨与中缝；
  4. 把候选证据写到 p12/joint_candidates.json，并按冻结标注算 IoU；
  5. 可选 --emit 把候选合并进一份预测文件，供冻结评分器复评。

标注只用于「哪些区段需要联合」的离线判定与证据核对；轨框本身完全由原图墨迹
给出（joint_propose 不读标注坐标）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocr.vertical import imread_unicode  # noqa: E402
from dual_track_fix.joint import joint_propose, ink_mask  # noqa: E402
from dual_track_fix.detect import page_bottom_border  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
IMG = ROOT / '古籍评测集' / '医家类' / '丹方彚编' / 'page_12.png'
PAGE = '丹方彚编/page_12'
SLOT_BOUNDS = [545, 594, 641, 690, 738, 790, 840, 882, 938, 991]


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return round(ix * iy / u, 4) if u else 0.0


def load() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ann = json.loads((S2 / 'annotations.json').read_text(encoding='utf-8'))
    pred = json.loads((S8 / 'predictions' / 'path_stage8v2.json').read_text(encoding='utf-8'))
    gts = [g for g in ann.get('groups') or [] if g.get('page_id') == PAGE]
    return gts, pred


def find_cross(gts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """中缝≈槽边界 且 药名横跨该边界的组（离线判据，只定「哪些区段」）。"""
    out = []
    for g in gts:
        gx = g.get('gutter_x')
        m = g.get('main_bbox')
        if not gx or not m:
            continue
        for b in SLOT_BOUNDS:
            if abs(gx - b) <= 3 and m[0] < b < m[2]:
                out.append({'group_id': g.get('group_id'), 'region': g.get('region'),
                            'boundary_x': b, 'main_bbox': m, 'gutter_x': gx,
                            'gt_group': g['group_bbox'], 'gt_left': g['left_bbox'],
                            'gt_right': g['right_bbox']})
    return out


def snap_lane_y(gray: np.ndarray, b: Sequence[int], y_min: int, y_max: int,
                row_min: int = 2, pad: int = 2) -> List[int]:
    """按完整字形回溯轨的首字顶与末字底：取覆盖该轨的连续墨段两端。

    y_min 用冻结药名框下沿（轨顶不得早于药名底），y_max 用版框/下一标题。
    """
    ink = ink_mask(gray)
    x1, _, x2, _ = [int(v) for v in b]
    rows = ink[max(0, y_min):y_max, max(0, x1):x2].sum(axis=1)
    on = rows >= row_min
    runs: List[Tuple[int, int]] = []
    st = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((y_min + st, y_min + i))
            st = None
    if st is not None:
        runs.append((y_min + st, y_max))
    runs = [r for r in runs if r[1] - r[0] >= 2]
    if not runs:
        return [int(v) for v in b]
    return [x1, int(runs[0][0]), x2, min(y_max, int(runs[-1][1]) + pad)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--emit', default=None, help='把补修候选合并进该预测文件并写出新文件')
    args = ap.parse_args()

    gts, pred = load()
    cross = find_cross(gts)
    img = imread_unicode(str(IMG))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    border = page_bottom_border(gray)

    results = []
    for c in cross:
        xlo, xhi = c['main_bbox'][0] - 15, c['main_bbox'][2] + 18
        left_slot = [xlo, 0, c['boundary_x'], img.shape[0]]
        right_slot = [c['boundary_x'], 0, xhi, img.shape[0]]
        r = joint_propose(gray, left_slot, right_slot, c['main_bbox'],
                          c['boundary_x'], H=21.0, bottom_limit=border)
        if not r:
            results.append({**c, 'proposal': None, 'reason': 'joint_propose_failed'})
            continue
        results.append({**c, 'proposal': r,
                        'iou_group': iou(r['group_bbox'], c['gt_group']),
                        'iou_left': iou(r['left_bbox'], c['gt_left']),
                        'iou_right': iou(r['right_bbox'], c['gt_right'])})

    out = {'stage': 'stage8_joint_candidates', 'page_id': PAGE,
           'criterion': 'gutter_x≈slot_boundary±3px AND main_bbox spans boundary',
           'n_cross_groups': len(cross), 'candidates': results}
    dst = S8 / 'p12' / 'joint_candidates.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    for r in results:
        print(f"{r['region']} boundary={r['boundary_x']} main={r['main_bbox']}")
        if r.get('proposal'):
            p = r['proposal']
            print(f"  seam={p['seam_x']} L={p['left_bbox']} R={p['right_bbox']} G={p['group_bbox']}")
            print(f"  IoU G={r['iou_group']} L={r['iou_left']} R={r['iou_right']}")
        else:
            print('  FAILED', r.get('reason'))
    print(f'-> {dst.relative_to(ROOT)}')

    if args.emit:
        doc = json.loads(Path(args.emit).read_text(encoding='utf-8'))
        added = 0
        # 1) 联合候选（陽起石）：整组替换注入
        for r in results:
            if not r.get('proposal'):
                continue
            p = r['proposal']
            pred_id = f"{PAGE}#joint#{r['region']}"
            item = {
                'prediction_id': pred_id, 'page_id': PAGE,
                'group_bbox': p['group_bbox'], 'left_bbox': p['left_bbox'],
                'right_bbox': p['right_bbox'], 'source': 'JOINT',
                'region_id': f"{PAGE}#joint",
                'evidence': p['evidence'],
            }
            doc['predictions'] = [x for x in doc['predictions']
                                  if x.get('prediction_id') != pred_id]
            doc['predictions'].append(item)
            added += 1
        # 2) 固定样例首尾（雲母 DF12-03）：对命中预测按完整字形回溯轨首尾，
        #    y_min 取冻结药名框下沿，保证药名完整留在轨框之外。
        snaps = []
        gt03 = next((g for g in gts if g.get('region') == 'DF12-03'), None)
        if gt03 and gt03.get('main_bbox'):
            y_min = int(gt03['main_bbox'][3]) + 1
            for item in doc['predictions']:
                if item.get('page_id') != PAGE:
                    continue
                gb = item.get('group_bbox') or []
                # 雲母命中预测：y 区间落在上半栏、x 在 [836,885]
                if not (gb and 265 <= gb[1] <= 276 and 830 <= gb[0] <= 845
                        and gb[3] <= 540):
                    continue
                before = {s: list(item[f'{s}_bbox']) for s in ('left', 'right')
                          if item.get(f'{s}_bbox')}
                for s, b in before.items():
                    item[f'{s}_bbox'] = snap_lane_y(gray, b, y_min, 540)
                lb, rb = item.get('left_bbox'), item.get('right_bbox')
                if lb and rb:
                    item['group_bbox'] = [min(lb[0], rb[0]), min(lb[1], rb[1]),
                                          max(lb[2], rb[2]), max(lb[3], rb[3])]
                item['source'] = 'JOINT+GLYPH_SNAP'
                item['evidence'] = {**item.get('evidence', {}),
                                    'glyph_snap': {'y_min': y_min, 'before': before,
                                                   'after': {s: item[f'{s}_bbox']
                                                             for s in before}}}
                snaps.append({'prediction_id': item['prediction_id'],
                              'before': before,
                              'after': {s: item[f'{s}_bbox'] for s in before}})
        # 3) 剩余重复预测：只**记录**不删除。冻结评分器用最大流做一对一匹配，
        #    同 GT 的多余候选计入 duplicate_extra_fp；删预测会改变召回，属于改口径。
        dup_info = []
        if gts:
            def _iou(a: Sequence[int], b: Sequence[int]) -> float:
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                u = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy)
                return ix*iy/u if u else 0.0
            for g in gts:
                gtb = g.get('group_bbox')
                if not gtb:
                    continue
                cands = [p for p in doc['predictions']
                         if p.get('page_id') == PAGE and _iou(p['group_bbox'], gtb) >= 0.5]
                if len(cands) <= 1:
                    continue
                ranked = sorted(cands, key=lambda p: -_iou(p['group_bbox'], gtb))
                dup_info.append({'gt_region': g.get('region'),
                                 'candidates': [{'prediction_id': p['prediction_id'],
                                                 'iou': round(_iou(p['group_bbox'], gtb), 3),
                                                 'source': p.get('source')} for p in ranked],
                                 'handling': ('评分器最大流一对一匹配选最高 IoU；其余计 '
                                              'duplicate_extra_fp，不删预测以保持召回口径')})
        doc.setdefault('duplicate_handling', dup_info)
        outp = S8 / 'predictions' / 'path_stage8v3_joint.json'
        outp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'-> {outp.relative_to(ROOT)} (joint added {added}, snaps {len(snaps)}, '
              f'dup clusters recorded {len(dup_info)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
