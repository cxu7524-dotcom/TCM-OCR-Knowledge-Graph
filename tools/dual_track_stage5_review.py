# -*- coding: utf-8 -*-
"""38 个标注组的冻结评分复评：逐条命中/低 IoU/候选被拒/无候选 + 叠框证据。

口径：
  - 命中只认冻结评分器的最优一对一匹配（metrics.scores.by_page[].pairs）；
  - 未命中时另算"最佳候选 IoU"（同一组框、名义端点），区分候选未达门槛
    与候选在检测阶段就被拒；
  - 第四阶段覆盖直接读 search_regions.json 的 group_coverage，不重算；
  - seam_missing 且未经人工复核的候选一律标"疑似提前断组"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2  # noqa: E402
from ocr.vertical import imread_unicode  # noqa: E402

from dual_track_track.dedupe import bbox_iou  # noqa: E402
from dual_track_track.inputs import S2, S5  # noqa: E402

BEFORE = S5 / 'history' / 'stage5-v3'
EVIDENCE = S5 / 'evidence' / 'per_group'


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def _overlap(a: Optional[List[int]], b: List[int]) -> float:
    if not a:
        return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / float(max(1, (b[2] - b[0]) * (b[3] - b[1])))


def _match(metrics: Dict[str, Any], gid: str, tk: str) -> Dict[str, Any]:
    for page in (metrics.get('scores', {}).get('by_page') or {}).values():
        for spec in (page.get('thresholds') or {}).get(tk, {}).get('pairs') or []:
            if spec[1] == gid:
                return {'hit': True, 'prediction_id': spec[0], 'iou': round(float(spec[2]), 6)}
    return {'hit': False, 'prediction_id': None, 'iou': None}


def _lane_category(metrics: Dict[str, Any], gid: str, tk: str) -> Optional[str]:
    for page in (metrics.get('scores', {}).get('by_page') or {}).values():
        for row in (page.get('thresholds') or {}).get(tk, {}).get('lane_rows') or []:
            if row.get('group_id') == gid:
                return row.get('category')
    return None


def _region_for(s4: Dict[str, Any], gid: str) -> Optional[Dict[str, Any]]:
    for pg in s4.get('pages', []):
        for row in pg.get('group_coverage') or []:
            if row.get('group_id') == gid:
                return row
    return None


def _best_candidate(preds: List[Dict[str, Any]], bbox: List[int]
                    ) -> Tuple[Optional[Dict[str, Any]], float]:
    best, best_iou = None, 0.0
    for g in preds:
        iou = bbox_iou(g.get('group_bbox') or [0, 0, 0, 0], bbox)
        if iou > best_iou:
            best, best_iou = g, iou
    return best, best_iou


def _early_split(preds: List[Dict[str, Any]], bbox: List[int]) -> Optional[Dict[str, Any]]:
    cands = [g for g in preds if _overlap(g.get('group_bbox'), bbox) > 0.3]
    cands.sort(key=lambda g: bbox_iou(g.get('group_bbox') or [0, 0, 0, 0], bbox),
               reverse=True)
    for g in cands:
        stops = g.get('stop_events') or []
        if any(e['type'] == 'seam_missing' for e in stops) and not any(
                e['type'] == 'blank_gap' for e in stops):
            e = next(x for x in stops if x['type'] == 'seam_missing')
            return {'group_id': g['group_id'], 'at_y': e['at_y'],
                    'reason': e.get('evidence', {}).get('reason')}
    return None


def _evidence(ann: Dict[str, Any], pred: Optional[Dict[str, Any]],
              best: Optional[Dict[str, Any]], status: str, iou: float) -> str:
    img = imread_unicode(str(ROOT / ann['image_path']))
    if img is None:
        return ''
    b = ann['group_bbox']
    pad = 24
    x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
    x1, y1 = min(img.shape[1], b[2] + pad), min(img.shape[0], b[3] + pad)
    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (b[0] - x0, b[1] - y0), (b[2] - x0, b[3] - y0),
                  (0, 180, 0), 2)
    if best and best.get('group_bbox'):
        bb = best['group_bbox']
        cv2.rectangle(crop, (bb[0] - x0, bb[1] - y0), (bb[2] - x0, bb[3] - y0),
                      (255, 0, 255), 1)
    if pred and pred.get('group_bbox'):
        bb = pred['group_bbox']
        cv2.rectangle(crop, (bb[0] - x0, bb[1] - y0), (bb[2] - x0, bb[3] - y0),
                      (0, 0, 255), 2)
    label = f"{ann['group_id'].split('/')[-1]} {status} iou={iou:.3f}"
    cv2.putText(crop, label, (3, crop.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (20, 20, 20), 1, cv2.LINE_AA)
    safe = ann['group_id'].replace('/', '__')
    path = EVIDENCE / f'{safe}.jpg'
    # cv2.imwrite 在 Windows 下不支持中文路径，编码后由 Python 写文件
    ok, buf = cv2.imencode('.jpg', crop)
    if not ok:
        return ''
    path.write_bytes(buf.tobytes())
    return str(path.relative_to(ROOT)).replace('\\', '/')


def main() -> int:
    ann = load(S2 / 'annotations.json')
    s4 = load(S5.parent / 'stage4' / 'search_regions.json')
    before = load(BEFORE / 'metrics.json')
    v3 = load(S5 / 'metrics.json')
    groups = load(S5 / 'groups.json')
    preds_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for pg in groups['pages']:
        preds_by_page[pg['page_id']] = pg.get('groups') or []

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for a in ann['groups']:
        gid = a['group_id']
        preds = preds_by_page.get(a['page_id']) or []
        hit2 = {tk: _match(before, gid, tk) for tk in ('0.5', '0.75')}
        hit3 = {tk: _match(v3, gid, tk) for tk in ('0.5', '0.75')}
        best, best_iou = _best_candidate(preds, a['group_bbox'])
        cov = _region_for(s4, gid) or {}
        early = _early_split(preds, a['group_bbox'])
        if hit3['0.5']['hit']:
            status = '匹配命中'
        elif best_iou >= 0.05:
            status = '候选未达 IoU 门槛'
        elif (cov.get('candidate_regions') or []) and any(
                not r.get('confirmed') for pg in groups['pages']
                if pg['page_id'] == a['page_id']
                for r in (next((x['candidate_runs'] for x in pg['regions']
                                if x['region_id'] == cov['region_id']), []) or [])):
            status = '候选被拒'
        else:
            status = '未形成候选'
        lane = _lane_category(v3, gid, '0.5')
        confirmed_error = bool(hit3['0.5']['hit'] and lane == 'child_localization_failure')
        note = []
        if confirmed_error:
            note.append('命中但轨级子框失败（已复核）')
        if early:
            note.append(f"疑似提前断组 seam_missing@{early['at_y']}（未复核）")
        if cov.get('coverage_status', '').startswith('fully_covered'):
            coverage = '完全覆盖'
        elif cov.get('coverage_status'):
            coverage = '部分覆盖'
        else:
            coverage = '无覆盖记录'
        row = {
            'group_id': gid, 'page_id': a['page_id'], 'main_text': a.get('main_text'),
            'group_bbox': a['group_bbox'], 'stage4_coverage': coverage,
            'stage4_detail': cov.get('coverage_status'),
            'stage4_region_source': cov.get('region_source'),
            'v3': hit2, 'v4': hit3,
            'v3_hit_05': hit2['0.5']['hit'], 'v4_hit_05': hit3['0.5']['hit'],
            'v3_hit_075': hit2['0.75']['hit'], 'v4_hit_075': hit3['0.75']['hit'],
            'change': ('新增命中' if hit3['0.5']['hit'] and not hit2['0.5']['hit']
                       else '丢失命中' if hit2['0.5']['hit'] and not hit3['0.5']['hit']
                       else '保持命中' if hit3['0.5']['hit'] else '均未命中'),
            'status_v4': status, 'best_candidate_iou': round(best_iou, 6),
            'best_candidate_id': best.get('group_id') if best else None,
            'lane_category_v4': lane, 'suspected_early_split': early,
            'confirmed_error': confirmed_error, 'note': '；'.join(note),
            'evidence': _evidence(a, (best if hit3['0.5']['hit'] else None),
                                  best, status, best_iou),
        }
        rows.append(row)

    out = {
        'stage': 'stage5_frozen_review_38',
        'release': groups.get('release'),
        'before': {'release': before.get('release'),
                   'metrics': str((BEFORE / 'metrics.json').relative_to(ROOT))},
        'after': {'release': v3.get('release'), 'metrics': 'output/dual_tracks/stage5/metrics.json'},
        'protocol': 'stage3-v1 冻结评分；命中只取一对一最优匹配，未命中另算名义端点最佳 IoU',
        'counts': {
            'annotations': len(rows),
            'v3_hits': {tk: sum(1 for r in rows if r['v3'][tk]['hit'])
                        for tk in ('0.5', '0.75')},
            'v4_hits': {tk: sum(1 for r in rows if r['v4'][tk]['hit'])
                        for tk in ('0.5', '0.75')},
            'status_v4': {s: sum(1 for r in rows if r['status_v4'] == s)
                          for s in ('匹配命中', '候选未达 IoU 门槛', '候选被拒', '未形成候选')},
            'stage4_coverage': {s: sum(1 for r in rows if r['stage4_coverage'] == s)
                                for s in ('完全覆盖', '部分覆盖', '无覆盖记录')},
            'suspected_early_split': sum(1 for r in rows if r['suspected_early_split']),
            'confirmed_error': sum(1 for r in rows if r['confirmed_error']),
        },
        'rows': rows,
    }
    (S5 / 'review_38.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('复评', len(rows), '组；v3/v4 @0.5 =', out['counts']['v3_hits']['0.5'], '/',
          out['counts']['v4_hits']['0.5'], '；状态', out['counts']['status_v4'])
    print('→', S5 / 'review_38.json', '证据', EVIDENCE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
