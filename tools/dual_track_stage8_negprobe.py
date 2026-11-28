# -*- coding: utf-8 -*-
"""第八阶段补修③：NEG-DF-SINGLE-01 按原图纵向区段复核。

只做取证，不改标注。输出：
  output/dual_tracks/stage8/negative_review_df_single.json

复核三件事（对应验收要求「按原图纵向区段复核单列、双轨及可疑重复组」）：
  1. 单列假设：区内每行墨迹是否是**一条**连续带
  2. 双轨假设：带内是否存在稳定纵向空隙，把墨迹分成左右两条
  3. 可疑重复组：现有预测在区内是否吐出重复框

判定不靠肉眼看图，靠行级剖面。空隙必须**在多行上稳定**才算双列，
单行偶然的笔画间空白不算——否则每个字都会被判成两列。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.vertical import _binarize  # noqa: E402  用与检测器同一套二值化，避免阈值口径不一致

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
REGION_ID = 'NEG-DF-SINGLE-01'

INK_THR = 96          # 占位：实际用 ocr.vertical._binarize 的同一套二值化
BAND = 40             # 行带高度
STRONG = 0.18         # 某列在带内覆盖率 ≥ 此值才算「有墨列」
WEAK = 0.06           # 覆盖率 < 此值算「缝」；介于两者是零散笔画
VALLEY_MIN = 3        # 缝最小宽度（px）
SIDE_MIN = 8          # 缝任一侧最小宽度（px）
RUN_MIN = 3           # 墨迹列段最小宽度（px），更窄的算零散笔画不参与判定
MEAN_MIN = 0.10       # 缝一侧在窗口内的平均列覆盖率下限，防「一列只写了两笔」
WIN = 160             # 主判定窗口高度（px）


def load_gray(rel: str) -> np.ndarray:
    p = ROOT / rel
    # Windows 下 cv2.imread 不认非 ASCII 路径，先读字节再解码
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f'读不到图：{p}')
    return img


def row_gap(ink: np.ndarray) -> Dict[str, Any]:
    """单行：在给定 x 窗口里找最宽的零墨迹空隙。ink 是 bool 数组。"""
    idx = np.flatnonzero(ink)
    if idx.size < 2:
        return {'ink': int(idx.size), 'gap': 0, 'gap_x0': None, 'gap_x1': None,
                'left': 0, 'right': 0}
    x0, x1 = int(idx[0]), int(idx[-1]) + 1
    seg = ink[x0:x1]
    # 找最长的连续 False 段
    best = (0, None, None)
    run_start = None
    for i, v in enumerate(seg):
        if not v:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start > best[0]:
                    best = (i - run_start, x0 + run_start, x0 + i)
                run_start = None
    if run_start is not None and len(seg) - run_start > best[0]:
        best = (len(seg) - run_start, x0 + run_start, x0 + len(seg))
    gap, gx0, gx1 = best
    return {'ink': int(idx.size), 'gap': gap, 'gap_x0': gx0, 'gap_x1': gx1,
            'left': (gx0 - x0) if gx0 is not None else 0,
            'right': (x1 - gx1) if gx1 is not None else 0}


def band_profile(ink: np.ndarray, y0: int, y1: int, xa: int) -> Dict[str, Any]:
    """行带级列密度剖面：某列在带内被墨覆盖的行占比。

    逐行判空隙不可靠——汉字横画会把两列的缝连起来，某列某行空笔也会让单行
    看起来只有一条带。所以判双列必须在**纵向带**上做：缝要在一段连续的行上
    都接近零密度，才是真的两列并排。
    """
    sub = ink[y0:y1]
    h = sub.shape[0]
    dens = sub.sum(axis=0) / max(1, h)
    on = dens >= WEAK
    runs: List[List[int]] = []
    st = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append([xa + st, xa + i])
            st = None
    if st is not None:
        runs.append([xa + st, xa + len(on)])
    runs = [r for r in runs if r[1] - r[0] >= RUN_MIN]
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    dual = False
    for i, g in enumerate(gaps):
        if (g >= VALLEY_MIN and runs[i][1] - runs[i][0] >= SIDE_MIN
                and runs[i + 1][1] - runs[i + 1][0] >= SIDE_MIN):
            dual = True
            break
    return {'y0': y0, 'y1': y1, 'rows': h,
            'ink_cols': int(np.flatnonzero(dens >= STRONG).size),
            'verdict': 'dual' if dual else ('blank' if not runs else 'single'),
            'runs': runs,
            'widths': [r[1] - r[0] for r in runs],
            'gaps': gaps}


def window_profile(ink: np.ndarray, y0: int, y1: int, xa: int) -> Dict[str, Any]:
    """长窗口汇总判定：把一段纵向窗口内所有行叠起来看列密度。

    40px 短带容易抖——某一列在这 40 行里恰好只写了一个字，短带就当成单列。
    判「这一整段是单列还是两列」必须在长窗口上汇总：两列各自在窗口内都持续
    有墨，中间那道缝在窗口内始终接近零，才是真并排。
    """
    h = y1 - y0
    dens = ink[y0:y1].sum(axis=0) / max(1, h)
    on = dens >= WEAK
    runs: List[List[int]] = []
    st = None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append([xa + st, xa + i])
            st = None
    if st is not None:
        runs.append([xa + st, xa + len(on)])
    runs = [r for r in runs if r[1] - r[0] >= RUN_MIN]

    def mean_d(a: int, b: int) -> float:
        return float(dens[a - xa:b - xa].mean())

    # 在「宽缝 + 两侧都有实质墨量」的候选里挑缝最宽的一处
    best = None
    for i in range(len(runs) - 1):
        g = runs[i + 1][0] - runs[i][1]
        lw, rw = runs[i][1] - runs[i][0], runs[i + 1][1] - runs[i + 1][0]
        if g < VALLEY_MIN or lw < SIDE_MIN or rw < SIDE_MIN:
            continue
        if mean_d(*runs[i]) < MEAN_MIN or mean_d(*runs[i + 1]) < MEAN_MIN:
            continue
        if best is None or g > best[0]:
            best = (g, runs[i], runs[i + 1])
    return {'y0': y0, 'y1': y1, 'rows': h,
            'verdict': 'dual' if best else ('blank' if not runs else 'single'),
            'runs': runs, 'gaps': [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)],
            'seam': None if not best else {'width': best[0],
                                           'left': best[1], 'right': best[2]},
            'seam_x': None if not best else (best[1][1], best[2][0])}


def analyse(gray: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    # 不给外扩：外扩 6px 会把 x=530–534 的竖界行拉进窗口，界行比真缝更宽，
    # 最宽空隙就被界行与正文之间的空白抢走，两列被判成单列（v1 原型即此错）。
    xa, xb = x1, x2
    ink = _binarize(gray)[y1:y2, xa:xb] > 0
    H = ink.shape[0]

    # 明细：40px 短带，只作抽查用
    bands: List[Dict[str, Any]] = []
    for y0 in range(0, H - BAND + 1, BAND // 2):
        p = window_profile(ink, y0, y0 + BAND, xa)
        p['y0'] += y1
        p['y1'] += y1
        bands.append(p)

    # 主判定：win=160px 长窗口，步长 40px
    wins: List[Dict[str, Any]] = []
    wlen = min(WIN, H)
    if H <= wlen:
        starts = [0]
    else:
        starts = list(range(0, H - wlen + 1, 40))
        if starts[-1] != H - wlen:
            starts.append(H - wlen)
    for y0 in starts:
        p = window_profile(ink, y0, y0 + wlen, xa)
        p['y0'] += y1
        p['y1'] += y1
        wins.append(p)

    counts = {'dual': 0, 'single': 0, 'blank': 0}
    for p in wins:
        counts[p['verdict']] += 1

    segs: List[Dict[str, Any]] = []
    cur = None
    for p in wins:
        v = p['verdict']
        if cur is None or cur['verdict'] != v:
            if cur is not None:
                segs.append(cur)
            cur = {'verdict': v, 'y0': p['y0'], 'y1': p['y1'], 'windows': 0, 'items': []}
        cur['y1'] = max(cur['y1'], p['y1'])
        cur['windows'] += 1
        if v == 'dual':
            cur['items'].append(p)
    if cur is not None:
        segs.append(cur)
    for s in segs:
        items = s.pop('items')
        if s['verdict'] != 'dual' or not items:
            continue
        seams = [p['seam_x'] for p in items if p['seam_x']]
        widths = [p['seam']['width'] for p in items if p['seam']]
        lefts = [p['seam']['left'] for p in items if p['seam']]
        rights = [p['seam']['right'] for p in items if p['seam']]
        s['seam_x'] = [int(np.median([x[0] for x in seams])), int(np.median([x[1] for x in seams]))]
        s['seam_width'] = int(np.median(widths))
        s['left_col'] = [int(np.median([c[0] for c in lefts])), int(np.median([c[1] for c in lefts]))]
        s['right_col'] = [int(np.median([c[0] for c in rights])), int(np.median([c[1] for c in rights]))]

    dual_wins = [p for p in wins if p['verdict'] == 'dual']
    stab: Dict[str, Any] = {}
    if dual_wins:
        arr = np.array([(p['seam_x'][0] + p['seam_x'][1]) / 2 for p in dual_wins])
        stab = {'windows_dual': len(dual_wins), 'windows_total': len(wins),
                'x_median': float(np.median(arr)),
                'x_min': float(arr.min()), 'x_max': float(arr.max()),
                'spread_px': float(arr.max() - arr.min())}

    return {'bbox': bbox, 'x_window': [xa, xb], 'window': wlen,
            'band': BAND, 'verdict_counts': counts, 'segments': segs,
            'stable_gap': stab, 'windows': wins, 'bands': bands}


def dup_scan(pred_path: Path, bbox: List[int]) -> Dict[str, Any]:
    """区内预测：按三层框分别算落在负例区内的覆盖率，重复判定用「x 高度重合 + y 有交叠」。"""
    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    x1, y1, x2, y2 = bbox
    items = doc.get('predictions') or []

    def hit(b: List[int]) -> float:
        ix = max(0, min(x2, b[2]) - max(x1, b[0]))
        iy = max(0, min(y2, b[3]) - max(y1, b[1]))
        ar = (b[2] - b[0]) * (b[3] - b[1])
        return ix * iy / ar if ar else 0.0

    groups = []
    for it in items:
        if it.get('page_id') != '丹方彚编/page_12':
            continue
        cov = {k: hit(it[k]) for k in ('group_bbox', 'right_bbox', 'left_bbox') if it.get(k)}
        best = max(cov.values()) if cov else 0.0
        if best >= 0.30:
            groups.append({'prediction_id': it.get('prediction_id'),
                           'group_bbox': it.get('group_bbox'),
                           'right_bbox': it.get('right_bbox'),
                           'left_bbox': it.get('left_bbox'),
                           'source': it.get('source'),
                           'coverage': {k: round(v, 3) for k, v in cov.items()},
                           'coverage_max': round(best, 3)})
    dups = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i]['group_bbox'], groups[j]['group_bbox']
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix > 0 and iy > 0:
                inter = ix * iy
                ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                dups.append({'a': groups[i]['prediction_id'], 'b': groups[j]['prediction_id'],
                             'iou': round(inter / ar, 3) if ar else 0.0,
                             'overlap_px': [ix, iy]})
    try:
        rel = str(pred_path.relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        rel = str(pred_path)
    return {'predictions': rel,
            'groups_in_region': groups, 'overlaps': dups, 'n_groups': len(groups)}


def main(argv: Optional[List[str]] = None) -> int:
    neg = json.loads((S2 / 'negative_regions.json').read_text(encoding='utf-8'))
    region = next(r for r in neg['regions'] if r['negative_region_id'] == REGION_ID)
    gray = load_gray(region['image_path'])
    res = analyse(gray, region['bbox'])
    res['region_id'] = REGION_ID
    res['page_id'] = region['page_id']
    res['image'] = region['image_path']
    res['image_size'] = [int(gray.shape[1]), int(gray.shape[0])]
    res['frozen_label'] = {'category': region['category'], 'status': region['status'],
                           'basis': region['basis'], 'evidence': region['evidence'],
                           'annotation_coverage': region['annotation_coverage']}

    vers = {
        'C': S8 / 'snapshots/C/predictions.json',
        'stage7-v2': S8 / 'snapshots/stage7v2/predictions.json',
        'stage8-v1': S8 / 'predictions' / 'path_stage8v1.json',
        'stage8-v2': S8 / 'predictions' / 'path_stage8v2.json',
    }
    res['versions'] = {k: dup_scan(v, region['bbox']) for k, v in vers.items() if v.exists()}
    res['zoom_images'] = [
        {'path': 'output/dual_tracks/stage8/_probe_neg_df_single.png',
         'crop': [515, 200, 600, 520], 'scale': 4},
        {'path': 'output/dual_tracks/stage8/_probe_neg_df_single_low.png',
         'crop': [515, 560, 600, 830], 'scale': 4},
    ]
    res['generated_at'] = datetime.now(TZ).isoformat(timespec='seconds')

    out = S8 / 'negative_review_df_single.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    print(f'区域 {REGION_ID}  框 {region["bbox"]}')
    print(f'  长窗口判定 {res["verdict_counts"]}（win={res["window"]}px，步长 40px）')
    if res['stable_gap']:
        g = res['stable_gap']
        print(f'  缝 x≈{g["x_median"]:.1f}  范围 {g["x_min"]:.1f}–{g["x_max"]:.1f}  '
              f'扩散 {g["spread_px"]:.1f}px  双列窗口 {g["windows_dual"]}/{g["windows_total"]}')
    print('  纵向区段（按长窗口）：')
    for s_ in res['segments']:
        extra = ''
        if s_.get('seam_x'):
            extra = (f'  缝 x {s_["seam_x"][0]}–{s_["seam_x"][1]} 宽 {s_["seam_width"]}px'
                     f'  左列 {s_["left_col"][0]}–{s_["left_col"][1]}'
                     f'  右列 {s_["right_col"][0]}–{s_["right_col"][1]}')
        print(f'    y {s_["y0"]:4d}-{s_["y1"]:4d}  {s_["verdict"]:6s} {s_["windows"]:2d} 窗{extra}')
    print('  区内预测（按版本）：')
    for k, v in res['versions'].items():
        covs = [g['coverage_max'] for g in v['groups_in_region']]
        print(f'    {k:10s} 区内 {v["n_groups"]:2d} 组  重叠对 {len(v["overlaps"])}'
              f'  最高覆盖 {max(covs) if covs else 0:.2f}'
              f'  {[g["group_bbox"] for g in v["groups_in_region"][:4]]}')
    print(f'  → {out.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
