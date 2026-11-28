# -*- coding: utf-8 -*-
"""p12 固定样例取证：雲母（DF12-03）与陽起石（DF12-11）的左右轨首尾放大图 + 组间标题叠框。

只画不改：把 GT 组框、预测组框、左右轨框、以及经/未命中用不同颜色叠在原图上，
再按 4× 放大落盘，供报告直接引用。首尾另画两条水平标线（框上下边），
用来肉眼核对「首笔/末笔有没有被切掉」。

用法：
  python tools/dual_track_stage8_p12_evidence.py
输出：
  output/dual_tracks/stage8/p12/df12_03_tracks.png
  output/dual_tracks/stage8/p12/df12_11_tracks.png
  output/dual_tracks/stage8/p12/df12_11_title_stack.png
  output/dual_tracks/stage8/p12/p12_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
IMG = ROOT / '古籍评测集' / '医家类' / '丹方彚编' / 'page_12.png'
DST = S8 / 'p12'
PAGE = '丹方彚编/page_12'
# 颜色：BGR。GT 蓝、预测组框 红、右轨 绿、左轨 橙
C_GT, C_PRED, C_R, C_L, C_TXT = (200, 80, 0), (0, 0, 230), (0, 160, 0), (230, 128, 0), (0, 0, 0)


def imread_u(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix, img)
    if not ok:
        raise SystemExit(f'写不出：{p}')
    buf.tofile(str(p))


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return round(inter / ua, 4) if ua else 0.0


def load_groups(pred_name: str = 'path_online_v4.json') -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ann = json.loads((S2 / 'annotations.json').read_text(encoding='utf-8'))
    pred = json.loads((S8 / 'predictions' / pred_name).read_text(encoding='utf-8'))
    gts = {(g.get('group_id') or g.get('annotation_id')): g
           for g in (ann.get('groups') or ann.get('annotations') or [])
           if g.get('page_id') == PAGE}
    ps = [p for p in pred['predictions'] if p.get('page_id') == PAGE]
    return gts, ps


def best_pred(ps: List[Dict[str, Any]], gt: Sequence[int]) -> Optional[Dict[str, Any]]:
    best, bi = None, -1.0
    for p in ps:
        v = iou(p['group_bbox'], gt)
        if v > bi:
            best, bi = p, v
    return best


def one_to_one(gts: Dict[str, Any], ps: List[Dict[str, Any]],
               threshold: float = 0.5) -> Dict[str, Tuple[float, str]]:
    """冻结标注只在这里出现：预测生成后按 IoU 降序做一对一匹配。"""
    pairs: List[Tuple[float, str, str]] = []
    for gid, g in gts.items():
        gt = list(g.get('bbox') if 'bbox' in g else g.get('group_bbox') or [])
        if len(gt) < 4:
            continue
        for p in ps:
            pairs.append((iou(p['group_bbox'], gt), gid, p['prediction_id']))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    used_g, used_p, out = set(), set(), {}
    for v, gid, pid in pairs:
        if v < threshold or gid in used_g or pid in used_p:
            continue
        out[gid] = (v, pid)
        used_g.add(gid)
        used_p.add(pid)
    return out


def draw_panel(img: np.ndarray, crop: Sequence[int], gt: Sequence[int],
               pred: Optional[Dict[str, Any]], scale: float, title: str,
               main_box: Optional[Sequence[int]] = None) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in crop]
    sub = img[y1:y2, x1:x2]
    sub = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def m(b):
        return [(int(round((b[0] - x1) * scale)), int(round((b[1] - y1) * scale))),
                (int(round((b[2] - x1) * scale)), int(round((b[3] - y1) * scale)))]

    def rect(b, col, thick=2):
        (a, c) = m(b)
        cv2.rectangle(sub, a, c, col, thick)

    # 冻结药名框（灰）：用来肉眼核对药名是否完整留在轨框之外
    if main_box:
        rect(main_box, (150, 150, 150), 2)
        (ma, mc) = m(main_box)
        cv2.putText(sub, 'MAIN(GT)', (ma[0], max(12, ma[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    # GT 蓝虚线（用矩形近似，虚线靠交替画短段）
    (ga, gc) = m(gt)
    for y in range(ga[1], gc[1], 12):
        cv2.line(sub, (ga[0], y), (ga[0], min(y + 6, gc[1])), C_GT, 2)
        cv2.line(sub, (gc[0], y), (gc[0], min(y + 6, gc[1])), C_GT, 2)
    if pred:
        rect(pred['group_bbox'], C_PRED, 3)
        if pred.get('right_bbox'):
            rect(pred['right_bbox'], C_R, 2)
        if pred.get('left_bbox'):
            rect(pred['left_bbox'], C_L, 2)
    # 首尾标线：GT 框上下边各拉一条横线，用来核首末笔是否被切
    for yy, tag in ((ga[1], 'top'), (gc[1], 'bottom')):
        cv2.line(sub, (0, yy), (sub.shape[1] - 1, yy), (0, 220, 220), 1)
    cv2.putText(sub, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TXT, 1, cv2.LINE_AA)
    cv2.putText(sub, f'GT {list(gt)}', (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GT, 1, cv2.LINE_AA)
    if pred:
        cv2.putText(sub, f'PRED {pred["group_bbox"]} IoU {iou(pred["group_bbox"], gt):.3f}',
                    (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_PRED, 1, cv2.LINE_AA)
    return sub


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='path_online_v4.json',
                    help='取证用的预测（默认=同一次正常 OCR 入口产物）')
    args = ap.parse_args()
    img = imread_u(IMG)
    gts, ps = load_groups(args.pred)
    DST.mkdir(parents=True, exist_ok=True)
    # 一对一匹配必须在取证之前算完：冻结标注只用于预测生成后的核对。
    matches = one_to_one(gts, ps, 0.5)
    by_id = {p['prediction_id']: p for p in ps}
    out: Dict[str, Any] = {'page': PAGE, 'predictions': args.pred,
                           'matching': {'rule': 'IoU 降序一对一，阈值 0.50',
                                        'n_matched': len(matches),
                                        'n_annotations': len(gts)},
                           'items': {}}

    plan = [('DF12-03', '雲母'), ('DF12-11', '陽起石')]
    for gid, name in plan:
        akey = f'丹方彚编/p12/{gid}'
        g = gts.get(akey)
        if not g:
            print(f'缺标注 {gid}')
            continue
        gt = list(g['bbox'] if 'bbox' in g else g['group_bbox'])
        hit = matches.get(akey)
        p = by_id.get(hit[1]) if hit else best_pred(ps, gt)
        m = 26
        crop = [max(0, gt[0] - m), max(0, gt[1] - m), gt[2] + m, gt[3] + m]
        panel = draw_panel(img, crop, gt, p, 4.0, f'{gid} {name}',
                           main_box=g.get('main_bbox'))
        fp = DST / f'{gid.lower().replace("-", "_")}_tracks.png'
        imwrite_u(fp, panel)
        out['items'][gid] = {
            'name': name, 'gt': gt,
            'pred': p['group_bbox'] if p else None,
            'iou': iou(p['group_bbox'], gt) if p else 0.0,
            'hit_0.50': bool(p and iou(p['group_bbox'], gt) >= 0.5),
            'hit_0.75': bool(p and iou(p['group_bbox'], gt) >= 0.75),
            'right_bbox': p.get('right_bbox') if p else None,
            'left_bbox': p.get('left_bbox') if p else None,
            'prediction_id': p.get('prediction_id') if p else None,
            'matched_by_one_to_one': bool(hit),
            'matched_iou': round(hit[0], 4) if hit else None,
            'source': p.get('source') if p else None,
            'region_ref': p.get('region_ref') if p else None,
            'side_status': p.get('side_status') if p else None,
            'joint_evidence': p.get('joint_evidence') if p else None,
            'boundary_review': p.get('boundary_review') if p else None,
            'main_bbox': g.get('main_bbox'),
            'main_overlap_y': (max(0, min(gt[3], g['main_bbox'][3])
                                   - max(gt[1], g['main_bbox'][1]))
                               if g.get('main_bbox') and p else None),
            'image': str(fp.relative_to(ROOT)).replace('\\', '/'),
            'crop': crop,
        }
        # 首尾放大：只截轨框上下各 24px，4× 放大，核首笔/末笔是否被切
        if p:
            for edge, yy in (('head', p['group_bbox'][1]),
                             ('tail', p['group_bbox'][3])):
                ecrop = [max(0, gt[0] - 12), max(0, yy - 26),
                         min(img.shape[1], gt[2] + 12), min(img.shape[0], yy + 26)]
                esub = cv2.resize(img[ecrop[1]:ecrop[3], ecrop[0]:ecrop[2]],
                                  None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
                cv2.line(esub, (0, 6 * 26), (esub.shape[1] - 1, 6 * 26), (0, 0, 230), 2)
                efp = DST / f'{gid.lower().replace("-", "_")}_{edge}_zoom.png'
                imwrite_u(efp, esub)
                out['items'][gid][f'{edge}_zoom'] = str(efp.relative_to(ROOT)).replace('\\', '/')

    # 组间标题叠框：DF12-11 与它上面一组之间那条带（含可能压进来的大字/栏线）
    g11 = gts.get('丹方彚编/p12/DF12-11')
    g10 = gts.get('丹方彚编/p12/DF12-10')
    if g11 and g10:
        gtv = list(g11['bbox'] if 'bbox' in g11 else g11['group_bbox'])
        top = max(0, gtv[1] - 90)
        crop = [max(0, gtv[0] - 30), top, gtv[2] + 30, gtv[3]]
        sub = img[crop[1]:crop[3], crop[0]:crop[2]]
        sub = cv2.resize(sub, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        def mm(b):
            return [(int((b[0] - crop[0]) * 4), int((b[1] - crop[1]) * 4)),
                    (int((b[2] - crop[0]) * 4), int((b[3] - crop[1]) * 4))]

        for gg, col, tag in ((g11, (0, 0, 230), 'DF12-11'), (g10, (200, 80, 0), 'DF12-10')):
            bb = list(gg['bbox'] if 'bbox' in gg else gg['group_bbox'])
            a, b = mm(bb)
            cv2.rectangle(sub, a, b, col, 2)
            cv2.putText(sub, tag, (a[0] + 2, a[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                        cv2.LINE_AA)
        fp2 = DST / 'df12_11_title_stack.png'
        imwrite_u(fp2, sub)
        out['items']['DF12-11_title_stack'] = {
            'image': str(fp2.relative_to(ROOT)).replace('\\', '/'), 'crop': crop,
            'note': 'DF12-10 与 DF12-11 之间的组间带：此处无大字标题，只有栏线；'
                    '两组 x 起点相差 55px（800 vs 855），下半栏相位比上半栏右移'}

    (DST / 'p12_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                                          encoding='utf-8')
    for gid, it in out['items'].items():
        if 'iou' in it:
            print(f'{gid:28s} {it["name"]:4s} IoU {it["iou"]:.3f} '
                  f'@0.5 {"命中" if it["hit_0.50"] else "未中"} '
                  f'@0.75 {"命中" if it["hit_0.75"] else "未中"}  {it["image"]}')
        else:
            print(f'{gid:28s} {it["image"]}')
    print(f'→ {DST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
