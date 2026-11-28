# -*- coding: utf-8 -*-
"""第三阶段：跑三条实验路径 + 辅助基线，产出统一格式预测。

路径定义（同一套复识器 = ch_PP-OCRv5_rec_mobile.onnx，同一套预处理 = rot90 + 可选 2×）：
  A  recog_ref_lane_crop   参考右/左轨框裁图 → recognize_crop（只评文本，不产生框）
  B  ctx_propose_page      参考组上下文裁块 → page_layout + propose_page（框坐标回映射）
  C  full_page_propose_page 整页 → page_layout + propose_page（**主基线**）
  X  full_page_components  整页 OCR + page_layout + propose_components_page（辅助基线）

预测与指标分开：本脚本只产生预测与运行状态；评分由 dual_track_stage3_score.py 从
保存的预测重算，这样能区分「OCR 波动」与「评分逻辑变化」。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D          # noqa: E402
import dual_track_stage3_adapters as A     # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))

PATHS = {'A': 'recog_ref_lane_crop', 'B': 'ctx_propose_page',
         'C': 'full_page_propose_page', 'X': 'full_page_components'}


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def page_inventory(ann: Dict[str, Any], neg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """要跑的页 = 有标注的页 ∪ 有确定负例的页。"""
    pages: Dict[str, Dict[str, Any]] = {}
    for g in ann['groups']:
        pages.setdefault(g['page_id'], {'page_id': g['page_id'], 'reasons': set()})
        pages[g['page_id']]['reasons'].add('annotated')
    for r in neg['regions']:
        pages.setdefault(r['page_id'], {'page_id': r['page_id'], 'reasons': set()})
        pages[r['page_id']]['reasons'].add('negative_region')
    out = []
    for pid in sorted(pages):
        book, page = pid.split('/page_')
        out.append({'page_id': pid, 'book': book, 'page': int(page),
                    'image_path': str((D.BOOKS / book / D.page_file_name(int(page)))
                                      .relative_to(ROOT)).replace('\\', '/'),
                    'reasons': sorted(pages[pid]['reasons'])})
    return out


def column_for(box: Sequence[int], layout: Dict[str, Any]) -> Optional[List[int]]:
    """参考组落在旧布局的哪一列里（按中心 x）；找不到返回 None。"""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    best, best_area = None, -1
    for c in layout.get('cols') or []:
        b = c['bbox']
        if b[0] <= cx < b[2] and b[1] <= cy < b[3]:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if best is None or area < best_area:
                best, best_area = b, area
    return best


# ---------------------------------------------------------------- 路径 A

def run_path_a(pages: List[Dict[str, Any]], ann: Dict[str, Any],
               double_scale: int = 2) -> Dict[str, Any]:
    import cv2
    import numpy as np
    from ocr.recognize import rec_model_name, recognize_crop
    from ocr.vertical import imread_unicode

    records = []
    t_all = time.time()
    for pg in pages:
        img = imread_unicode(str(ROOT / pg['image_path']))
        page_groups = [g for g in ann['groups'] if g['page_id'] == pg['page_id']]
        if img is None:
            for g in page_groups:
                records.append({'page_id': pg['page_id'], 'group_id': g['group_id'],
                                'status': 'failed', 'error': 'image_unreadable'})
            continue
        for g in page_groups:
            rec = {'page_id': pg['page_id'], 'group_id': g['group_id'],
                   'text_status': g.get('text_status'),
                   'reference_text': {'right': g.get('right_text'), 'left': g.get('left_text')},
                   'eligible_for_text': str(g.get('text_status')) == 'transcribed',
                   'lanes': {}}
            for side in ('right', 'left'):
                box = g.get(f'{side}_bbox')
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    rec['lanes'][side] = {'status': 'empty_crop', 'bbox': box}
                    continue
                t0 = time.time()
                native, nconf = recognize_crop(np.ascontiguousarray(np.rot90(crop)), min_conf=0.15)
                big = cv2.resize(crop, None, fx=double_scale, fy=double_scale,
                                 interpolation=cv2.INTER_CUBIC)
                double, dconf = recognize_crop(np.ascontiguousarray(np.rot90(big)), min_conf=0.15)
                agree = bool(native) and native == double
                rec['lanes'][side] = {
                    'bbox': box, 'rotation': 'rot90_ccw', 'scale': 1,
                    'text': native or double, 'confidence': round(max(nconf, dconf), 4),
                    'native': {'text': native, 'conf': nconf, 'scale': 1},
                    'double': {'text': double, 'conf': dconf, 'scale': double_scale},
                    'stable': agree and min(nconf, dconf) >= 0.5,
                    'status': 'read' if (native or double) else 'unreadable',
                    'timing_s': round(time.time() - t0, 3),
                }
            rec['joined_text'] = (rec['lanes'].get('right', {}).get('text') or '') + \
                                 (rec['lanes'].get('left', {}).get('text') or '')
            records.append(rec)
        print(f"  A {pg['page_id']} 组 {len(page_groups)}", flush=True)
    return {'path': PATHS['A'], 'model': rec_model_name(),
            'preprocessing': {'rotation': 'rot90_ccw（竖排转横排）',
                              'scales': [1, double_scale],
                              'min_conf': 0.15,
                              'note': '与 ocr/small_columns_attach.py 的 native/double 两读法一致'},
            'records': records, 'timing_s': round(time.time() - t_all, 2),
            'counts': {'groups': len(records),
                       'lanes_read': sum(1 for r in records for s in ('right', 'left')
                                         if (r['lanes'].get(s) or {}).get('status') == 'read'),
                       'lanes_unreadable': sum(1 for r in records for s in ('right', 'left')
                                               if (r['lanes'].get(s) or {}).get('status') == 'unreadable'),
                       'unreadable_groups': sum(1 for r in records
                                                if all((r['lanes'].get(s) or {}).get('status') != 'read'
                                                       for s in ('right', 'left')))},
            'metric_limits': {'text_accuracy': 'only groups with text_status=transcribed; '
                                               'current dataset has none, so CER 不可计算'}}


# ---------------------------------------------------------------- 路径 B

def run_path_b(pages: List[Dict[str, Any]], ann: Dict[str, Any],
               layouts: Dict[str, Any], crop_dir: Path) -> Dict[str, Any]:
    import cv2
    from ocr.vertical import imread_unicode

    crop_dir.mkdir(parents=True, exist_ok=True)
    layout_by_page = {r['page_id']: r for r in layouts['pages']}
    preds, runs = [], []
    t_all = time.time()
    for pg in pages:
        page_groups = [g for g in ann['groups'] if g['page_id'] == pg['page_id']]
        if not page_groups:
            runs.append({'page_id': pg['page_id'], 'status': 'skipped',
                         'skip_reason': 'out_of_scope', 'reason': '本页无标注组，路径 B 无裁块依据',
                         'n_groups': 0, 'n_predictions': 0})
            continue
        img = imread_unicode(str(ROOT / pg['image_path']))
        if img is None:
            runs.append({'page_id': pg['page_id'], 'status': 'failed', 'error': 'image_unreadable'})
            continue
        H, W = img.shape[:2]
        lay = layout_by_page.get(pg['page_id'])
        n_pred, t_page = 0, time.time()
        page_runs = []
        for g in page_groups:
            col = column_for(g['group_bbox'], lay) if lay else None
            spec = A.crop_context_region(g['group_bbox'], col, (W, H))
            x1, y1, x2, y2 = spec['crop_bbox']
            crop = img[y1:y2, x1:x2]
            cpath = crop_dir / f"{pg['book']}__p{pg['page']}__{g['region']}.png"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.png', crop)[1].tofile(str(cpath))
            res = A.run_entry('layout_propose_page', str(cpath), pg['page_id'])
            mapped = []
            for i, p in enumerate(res['raw']):
                q = A.proposal_to_prediction(p, pg['page_id'], i, PATHS['B'],
                                             offset=(x1, y1),
                                             raw_ref=f"crop:{cpath.name}#raw[{i}]")
                q['prediction_id'] = f"{pg['page_id']}#B#{g['region']}#{i:02d}"
                q['crop_ref'] = {'group_id': g['group_id'], 'crop_bbox': spec['crop_bbox'],
                                 'crop_file': str(cpath.relative_to(ROOT)).replace('\\', '/'),
                                 'offset': [x1, y1], 'rule': spec['rule'],
                                 'uses_reference_position': True,
                                 'column_bbox_used': col}
                mapped.append(q)
            preds.extend(mapped)
            n_pred += len(mapped)
            page_runs.append({'group_id': g['group_id'], 'crop_bbox': spec['crop_bbox'],
                              'column_bbox_used': col, 'n_predictions': len(mapped),
                              'n_columns_detected': len((res.get('layout') or {}).get('cols') or []),
                              'crop_size': [spec['crop_bbox'][2] - spec['crop_bbox'][0],
                                            spec['crop_bbox'][3] - spec['crop_bbox'][1]],
                              'status': res['status'], 'error': res['error'],
                              'timing_s': res['timings'].get('total_s')})
        runs.append({'page_id': pg['page_id'], 'status': 'completed',
                     'n_groups': len(page_groups), 'n_predictions': n_pred,
                     'n_columns_detected': sum(g['n_columns_detected'] for g in page_runs),
                     'n_crops_detected_nothing': sum(1 for g in page_runs
                                                     if g['n_columns_detected'] == 0),
                     'groups': page_runs, 'timing_s': round(time.time() - t_page, 3)})
        print(f"  B {pg['page_id']} 组 {len(page_groups)} → 预测 {n_pred}", flush=True)
    return {'path': PATHS['B'], 'predictions': preds, 'page_runs': runs,
            'crop_rule': {'up_ratio': 0.9, 'down_ratio': 0.5, 'side_ratio': 0.25,
                          'side_source': 'column_bbox（旧布局列框）优先',
                          'uses_reference_position': True},
            'timing_s': round(time.time() - t_all, 2)}


# ---------------------------------------------------------------- 路径 C / X

def run_full_page(pages: List[Dict[str, Any]], entry: str, key: str) -> Dict[str, Any]:
    preds, runs = [], []
    t_all = time.time()
    for pg in pages:
        ocr_blocks = None
        t_page = time.time()
        ocr_note = {}
        if key == 'X':
            try:
                from ocr.pipeline import run_ocr
                t0 = time.time()
                res = run_ocr(image_path=str(ROOT / pg['image_path']), page_id=pg['page_id'])
                ocr_blocks = res.blocks
                ocr_note = {'ocr_engine': res.engine, 'model_loaded': res.model_loaded,
                            'is_demo': res.is_demo, 'blocks': len(res.blocks),
                            'timing_s': round(time.time() - t0, 3)}
                if res.is_demo or not res.model_loaded:
                    runs.append({'page_id': pg['page_id'], 'status': 'failed',
                                 'error': 'ocr_not_real', **ocr_note})
                    continue
            except Exception as exc:                              # noqa: BLE001
                runs.append({'page_id': pg['page_id'], 'status': 'failed',
                             'error': f'ocr_exception:{type(exc).__name__}: {exc}'})
                continue
        res = A.run_entry(entry, str(ROOT / pg['image_path']), pg['page_id'],
                          ocr_blocks=ocr_blocks)
        for p in res['predictions']:
            p['prediction_id'] = p['prediction_id'].replace(entry, key)
            p['source'] = key
        preds.extend(res['predictions'])
        status = res['status']
        runs.append({'page_id': pg['page_id'], 'status': status, 'entry': entry,
                     'n_predictions': len(res['predictions']),
                     'n_columns_detected': len((res.get('layout') or {}).get('cols') or []),
                     'error': res['error'], 'timings': res['timings'], **ocr_note,
                     'timing_s': round(time.time() - t_page, 3),
                     'n_component_audit': len(res.get('component_audit') or [])})
        print(f"  {key} {pg['page_id']} 列 {runs[-1]['n_columns_detected']} → "
              f"预测 {len(res['predictions'])}  {runs[-1]['timing_s']}s", flush=True)
    return {'path': key, 'entry': entry, 'predictions': preds, 'page_runs': runs,
            'timing_s': round(time.time() - t_all, 2)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='跑第三阶段实验路径')
    ap.add_argument('--paths', default='C,B,A,X', help='逗号分隔：A/B/C/X')
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    wanted = [p.strip().upper() for p in args.paths.split(',') if p.strip()]

    out = Path(args.out_dir)
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    (out / 'recognition').mkdir(parents=True, exist_ok=True)
    ann = load_json(S2 / 'annotations.json')
    neg = load_json(S2 / 'negative_regions.json')
    layouts = load_json(out / 'frozen_layouts.json')
    pages = page_inventory(ann, neg)
    if not args.quiet:
        print(f'页清单 {len(pages)} 页：'
              f'标注 {sum(1 for p in pages if "annotated" in p["reasons"])} / '
              f'负例 {sum(1 for p in pages if "negative_region" in p["reasons"])}')

    runs_summary: Dict[str, Any] = {}
    t0 = time.time()
    if 'C' in wanted:
        r = run_full_page(pages, 'layout_propose_page', 'C')
        (out / 'predictions' / 'path_c.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        runs_summary['C'] = {'pages': len(r['page_runs']),
                             'preds': len(r['predictions']), 'timing_s': r['timing_s']}
    if 'X' in wanted:
        r = run_full_page(pages, 'components_page', 'X')
        (out / 'predictions' / 'path_x_components.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        runs_summary['X'] = {'pages': len(r['page_runs']),
                             'preds': len(r['predictions']), 'timing_s': r['timing_s']}
    if 'B' in wanted:
        r = run_path_b(pages, ann, layouts, out / 'evidence' / 'crops')
        (out / 'predictions' / 'path_b.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        runs_summary['B'] = {'pages': len(r['page_runs']),
                             'preds': len(r['predictions']), 'timing_s': r['timing_s']}
    if 'A' in wanted:
        r = run_path_a(pages, ann)
        (out / 'recognition' / 'path_a.json').write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        runs_summary['A'] = {'records': len(r['records']),
                             'counts': r['counts'], 'timing_s': r['timing_s']}
    if not args.quiet:
        print('用时合计', round(time.time() - t0, 1), 's')
        print(json.dumps(runs_summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
