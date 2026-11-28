# -*- coding: utf-8 -*-
"""单因素对照：同一批原图、同一检测器、同一评分配置，只换「喂给检测器的列范围」。

  A 组：旧布局列框（page_layout.cols）
  B 组：阶段 4 搜索区域（stage4 regions 的 bbox）

两者都直接当 columns 交给 ocr.small_columns.propose_page——这是旧检测器唯一
支持的输入形态。若某类区域它消费不了（例如宽度超过页宽 45%），如实记录兼容限制。

产物单独保存，不覆盖阶段 3 的任何冻结结果。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2                                        # noqa: E402

import dual_track_stage3_adapters as A            # noqa: E402
import dual_track_stage3_score as SC              # noqa: E402
from dual_track_eval import RegionTable, score_predictions   # noqa: E402
from ocr.small_columns import propose_page        # noqa: E402
from ocr.vertical import imread_unicode, page_layout          # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def run_group(pages: List[Dict[str, Any]], stage4: Dict[str, Any], which: str
              ) -> Dict[str, Any]:
    regions_by_page = {p['page_id']: p for p in stage4['pages']}
    preds, runs = [], []
    t0 = time.time()
    for pg in pages:
        pid = pg['page_id']
        img = imread_unicode(str(ROOT / pg['image_path']))
        if img is None:
            runs.append({'page_id': pid, 'status': 'failed', 'error': 'image_unreadable'})
            continue
        lay = page_layout(str(ROOT / pg['image_path']))
        if which == 'old':
            columns = [{'bbox': [int(v) for v in c['bbox']]} for c in lay['cols']]
        else:
            columns = [{'bbox': [int(v) for v in r['bbox']]}
                       for r in regions_by_page[pid]['regions']]
            # 旧检测器要求列框不触页边（x1 < 3%W、x2 > 97%W 会直接 return []），
            # 这里裁一下并记录，否则整页静默为 0 预测
            W = img.shape[1]
            lo, hi = int(W * 0.031), int(W * 0.969)
            clipped = 0
            for c in columns:
                if c['bbox'][0] < lo or c['bbox'][2] > hi:
                    clipped += 1
                    c['bbox'] = [max(lo, c['bbox'][0]), c['bbox'][1],
                                 min(hi, c['bbox'][2]), c['bbox'][3]]
            columns = [c for c in columns if c['bbox'][2] - c['bbox'][0] >= 12]
            if clipped:
                runs.append({'page_id': pid, 'clipped_columns': clipped})
        t1 = time.time()
        props = propose_page(img, columns) or []
        got = [A.proposal_to_prediction(p, pid, i, f'stage4_control_{which}')
               for i, p in enumerate(props)]
        preds.extend(got)
        runs.append({'page_id': pid, 'status': 'completed',
                     'n_columns_in': len(columns), 'n_predictions': len(got),
                     'timing_s': round(time.time() - t1, 3)})
    return {'group': which, 'predictions': preds, 'page_runs': runs,
            'timing_s': round(time.time() - t0, 2)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage4 单因素对照')
    ap.add_argument('--out', default=str(S4 / 'control_experiment.json'))
    args = ap.parse_args(argv)

    stage4 = load(S4 / 'search_regions.json')
    pages = [{'page_id': p['page_id'], 'image_path': p['image_path']}
             for p in stage4['pages']]
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    cfg_doc = load(S3 / 'evaluation_config.json')
    runtime = SC.runtime_config(cfg_doc)
    sizes = SC.page_sizes_map(ann)
    split_map = SC.page_split_map(load(S2 / 'dataset_splits.json'))
    page_ids = sorted({p['page_id'] for p in stage4['pages']})
    tables = SC.build_tables(ann, neg, page_ids, SC.region_thresholds(runtime))

    out: Dict[str, Any] = {'stage': 'stage4_control_experiment',
                           'single_factor': '喂给检测器的列范围（旧布局 vs 阶段4区域）',
                           'regions_release': stage4.get('release'),
                           'regions_content_sha256': (stage4.get('determinism') or {})
                           .get('content_sha256'),
                           'regions_count': sum(len(p['regions']) for p in stage4['pages']),
                           'fixed': {'detector': 'ocr.small_columns.propose_page（未改）',
                                     'evaluator': 'dual_track_eval ' + str(runtime.get('numeric')),
                                     'config_version': cfg_doc.get('release'),
                                     'thresholds': runtime['group_iou_thresholds'],
                                     'pages': len(pages)},
                           'groups': {}}
    for which in ('old', 'new'):
        g = run_group(pages, stage4, which)
        res = score_predictions(g['predictions'], ann['groups'], tables, runtime,
                                pages=page_ids, page_split=split_map, page_sizes=sizes)
        out['groups'][which] = {
            'n_predictions': len(g['predictions']),
            'timing_s': g['timing_s'],
            'page_runs': g['page_runs'],
            'overall': {tk: {k: v for k, v in res['summary']['overall'][tk].items()
                             if k in ('group_hits', 'annotations_total', 'hit_rate_on_known',
                                      'tp', 'fp', 'fn', 'new_candidates', 'lane_denominator',
                                      'lane_counts')} for tk in res['summary']['overall']},
            'by_split': res['summary']['by_split'],
            'patched': 'stage4 regions 若触页边会被裁到 3.1%~96.9%（旧检测器硬约束）',
        }
    a = out['groups']['old']['overall']
    b = out['groups']['new']['overall']
    out['delta'] = {tk: {'group_hits': b[tk]['group_hits'] - a[tk]['group_hits'],
                         'predictions': out['groups']['new']['n_predictions']
                                        - out['groups']['old']['n_predictions']}
                    for tk in a}
    out['compatibility_notes'] = [
        'propose_page 只接受「列框」形态：必须有 bbox 且宽 ≥ 12px、不触页边 3%/97%',
        '它不看区域来源与置信度，所以低置信 fallback 窗口与高置信栏线区域在它眼里一样',
        '阶段的 446px 宽框若直接喂进去仍会被它当一列处理，因此本对照用的是阶段 4 拆开的区域',
    ]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print('old', {tk: a[tk]['group_hits'] for tk in a},
          'preds', out['groups']['old']['n_predictions'])
    print('new', {tk: b[tk]['group_hits'] for tk in b},
          'preds', out['groups']['new']['n_predictions'])
    print('delta', out['delta'])
    print('→', Path(args.out).relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
