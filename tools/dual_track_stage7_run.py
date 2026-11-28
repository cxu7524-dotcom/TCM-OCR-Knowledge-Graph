# -*- coding: utf-8 -*-
"""stage7：从正常 OCR 入口跑一页，对比「双轨开 / 关」的整页结果。

用法
----
  python tools/dual_track_stage7_run.py --dual both            # 开关两跑 + 对比（默认）
  python tools/dual_track_stage7_run.py --dual on --page-id X  # 只跑开
  TCM_OCR_DUAL_TRACK=0 ...                                     # 环境变量等价于 --dual off

产物（output/dual_tracks/stage7/run/）
  <slug>_on.json / <slug>_off.json   两次完整 OCRResult（含 dual_track_* 字段）
  compare_on_off.json                块 / 文本 / 坐标 / 阅读顺序 / 表格 / 版面逐项对比
  checklist_<slug>.json              标注组 × 证据的对账（含两轨读数）

硬约束：本工具**不改**任何冻结产物；只调用 ocr.pipeline.run_ocr 这个正常入口。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

TZ = timezone(timedelta(hours=8))
OUT = ROOT / 'output' / 'dual_tracks' / 'stage7' / 'run'
DEFAULT_PAGE = '丹方彚编/page_12'
DEFAULT_IMAGE = '古籍评测集/医家类/丹方彚编/page_12.png'


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def _sha(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]


def _slug(page_id: str) -> str:
    return page_id.replace('/', '__')


def _block_rows(ocr: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{'line_id': b.get('line_id'), 'block_ref': b.get('block_ref'),
             'reading_order': b.get('reading_order'), 'col_id': b.get('col_id'),
             'bbox': b.get('bbox'), 'text': b.get('text'),
             'text_norm': b.get('text_norm'), 'kind': b.get('kind'),
             'conf': b.get('conf')} for b in ocr.get('blocks') or []]
    rows.sort(key=lambda r: (r['reading_order'] or 0, r['line_id'] or ''))
    return rows


def _tables(ocr: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{'table_id': t.get('table_id'), 'rows': t.get('rows'),
             'cells': t.get('cells'), 'cell_refs': t.get('cell_refs')}
            for t in ocr.get('tables') or []]


def run_once(page_id: str, image: str, out_dir: Path, tag: str) -> Dict[str, Any]:
    """走正常入口跑一次，落盘。tag = on / off。"""
    from ocr.pipeline import run_ocr
    t0 = datetime.now(TZ)
    ocr = run_ocr(image, page_id)
    payload = {
        'tag': tag,
        'generated_at': t0.isoformat(timespec='seconds'),
        'env': {'TCM_OCR_DUAL_TRACK': os.environ.get('TCM_OCR_DUAL_TRACK'),
                'TCM_OCR_DUAL_TRACK_MODE': os.environ.get('TCM_OCR_DUAL_TRACK_MODE'),
                'TCM_OCR_DUAL_TRACK_ZOOM': os.environ.get('TCM_OCR_DUAL_TRACK_ZOOM')},
        'ocr': ocr.to_dict(),
        'summary': {
            'page_id': ocr.page_id, 'engine': ocr.engine,
            'model_loaded': ocr.model_loaded, 'is_demo': ocr.is_demo,
            'blocks': len(ocr.blocks),
            'tables': len(ocr.tables),
            'layout_cols': len(ocr.layout.cols or []),
            'direction': ocr.layout.direction,
            'dual_track_status': ocr.dual_track_status,
            'dual_track_run': ocr.dual_track_run,
            'dual_track_evidence': (None if ocr.dual_track_evidence is None
                                    else len(ocr.dual_track_evidence)),
            'dual_track_problems': ocr.validate_dual_track(),
            'block_sig': _sha(_block_rows(ocr.to_dict())),
            'table_sig': _sha(_tables(ocr.to_dict())),
            'layout_sig': _sha({'cols': ocr.layout.cols,
                                'order': ocr.layout.block_order,
                                'direction': ocr.layout.direction}),
            'notes_dual': [n for n in ocr.notes if '双轨' in n],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f'{_slug(page_id)}_{tag}.json'
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return payload


def compare(on: Dict[str, Any], off: Dict[str, Any]) -> Dict[str, Any]:
    """开 / 关两次的逐项对比：块、文本、坐标、阅读顺序、表格、版面。"""
    a = on['ocr']
    b = off['ocr']
    ar, br = _block_rows(a), _block_rows(b)
    diffs = []
    for i in range(max(len(ar), len(br))):
        x = ar[i] if i < len(ar) else None
        y = br[i] if i < len(br) else None
        if x != y:
            diffs.append({'index': i, 'on': x, 'off': y})
    return {
        'page_id': a.get('page_id'),
        'dual_on': {'status': on['summary']['dual_track_status'],
                    'evidence': on['summary']['dual_track_evidence']},
        'dual_off': {'status': off['summary']['dual_track_status'],
                     'evidence': off['summary']['dual_track_evidence']},
        'blocks': {'on': len(ar), 'off': len(br), 'identical': ar == br,
                   'diff_count': len(diffs), 'first_diffs': diffs[:3]},
        'block_sig': {'on': on['summary']['block_sig'], 'off': off['summary']['block_sig']},
        'texts': {'identical': [r['text'] for r in ar] == [r['text'] for r in br]},
        'bboxes': {'identical': [r['bbox'] for r in ar] == [r['bbox'] for r in br]},
        'reading_order': {'identical': [(r['line_id'], r['reading_order'], r['col_id'])
                                        for r in ar] == [(r['line_id'], r['reading_order'],
                                                          r['col_id']) for r in br]},
        'tables': {'on': len(_tables(a)), 'off': len(_tables(b)),
                   'identical': _tables(a) == _tables(b)},
        'layout': {'on_cols': len((a.get('layout') or {}).get('cols') or []),
                   'off_cols': len((b.get('layout') or {}).get('cols') or []),
                   'identical': on['summary']['layout_sig'] == off['summary']['layout_sig']},
        'source_hash': {'on': a.get('source_hash'), 'off': b.get('source_hash'),
                        'identical': a.get('source_hash') == b.get('source_hash')},
        'notes': {'on': on['summary']['notes_dual'], 'off': off['summary']['notes_dual']},
        'verdict': ('dual_track 开关不影响正文块/文本/坐标/阅读顺序/表格/版面'
                    if (ar == br and _tables(a) == _tables(b)
                        and on['summary']['layout_sig'] == off['summary']['layout_sig'])
                    else '存在差异，见 diffs'),
    }


def checklist(page_id: str, on: Dict[str, Any]) -> Dict[str, Any]:
    """把标注组（stage2）与本次证据对上账：命中的组逐侧列状态与读数。"""
    ocr = on['ocr']
    ev = ocr.get('dual_track_evidence') or []
    rows: List[Dict[str, Any]] = []
    ann = None
    try:
        ann = _load(ROOT / 'output' / 'dual_tracks' / 'stage2' / 'annotations.json')
    except (OSError, ValueError):
        ann = None
    groups = [g for g in ((ann or {}).get('groups') or []) if g['page_id'] == page_id]

    def iou(a: Any, b: Any) -> float:
        if not a or not b:
            return 0.0
        x = min(a[2], b[2]) - max(a[0], b[0])
        y = min(a[3], b[3]) - max(a[1], b[1])
        if x <= 0 or y <= 0:
            return 0.0
        inter = x * y
        ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return round(inter / ua, 6) if ua else 0.0

    pred_groups = sorted({e['group_id'] for e in ev})
    pred_box = {}
    for e in ev:
        pred_box.setdefault(e['group_id'], e.get('group_bbox'))
    for g in groups:
        best, best_iou = None, 0.0
        for gid in pred_groups:
            v = iou(g['group_bbox'], pred_box.get(gid))
            if v > best_iou:
                best, best_iou = gid, v
        item = {'annotation_id': g.get('group_id'), 'main_text': g.get('main_text'),
                'gt_bbox': g['group_bbox'], 'prediction_id': best,
                'group_iou': best_iou,
                'sides': {}}
        if best:
            for e in ev:
                if e['group_id'] != best:
                    continue
                item['sides'][e['side']] = {
                    'side_status': e['side_status'], 'bbox': e['bbox'],
                    'text': e['text'], 'conf': e['conf'], 'model': e['model'],
                    'crop_path': e['crop_path'], 'bbox_source': e['bbox_source'],
                    'pending_reasons': e['pending_reasons'][:4],
                }
        rows.append(item)
    return {'page_id': page_id, 'annotations': len(groups),
            'predicted_groups': len(pred_groups), 'evidence': len(ev),
            'sides': (on['summary'] or {}).get('dual_track_status'),
            'rows': rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage7：正常 OCR 入口 + 双轨开关对比')
    ap.add_argument('--page-id', default=DEFAULT_PAGE)
    ap.add_argument('--image', default=DEFAULT_IMAGE)
    ap.add_argument('--dual', choices=['on', 'off', 'both'], default='both')
    ap.add_argument('--out-dir', default=str(OUT))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    image = str(ROOT / args.image) if not Path(args.image).is_absolute() else args.image

    if args.dual in ('on', 'off'):
        os.environ['TCM_OCR_DUAL_TRACK'] = '1' if args.dual == 'on' else '0'
        payload = run_once(args.page_id, image, out_dir, args.dual)
        print(json.dumps(payload['summary'], ensure_ascii=False, indent=1))
        if args.dual == 'on':
            cl = checklist(args.page_id, payload)
            (out_dir / f'checklist_{_slug(args.page_id)}.json').write_text(
                json.dumps(cl, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        return 0

    # both：两个独立进程跑，环境变量在 import 时生效，避免同进程缓存干扰
    for tag, flag in (('on', '1'), ('off', '0')):
        env = dict(os.environ)
        env['TCM_OCR_DUAL_TRACK'] = flag
        if flag == '1':
            env.setdefault('TCM_OCR_DUAL_TRACK_CROP_DIR',
                           str(out_dir.parent / 'evidence' / 'crops'))
        else:
            env.pop('TCM_OCR_DUAL_TRACK_CROP_DIR', None)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--dual', tag,
               '--page-id', args.page_id, '--image', args.image,
               '--out-dir', str(out_dir)]
        print(f'[stage7] run {tag} ...', flush=True)
        subprocess.run(cmd, env=env, check=True)
    on = _load(out_dir / f'{_slug(args.page_id)}_on.json')
    off = _load(out_dir / f'{_slug(args.page_id)}_off.json')
    cmp = compare(on, off)
    (out_dir / 'compare_on_off.json').write_text(
        json.dumps(cmp, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    cl = checklist(args.page_id, on)
    (out_dir / f'checklist_{_slug(args.page_id)}.json').write_text(
        json.dumps(cl, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: cmp[k] for k in ('blocks', 'reading_order', 'tables', 'layout',
                                          'verdict')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
