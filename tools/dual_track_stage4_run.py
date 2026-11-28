# -*- coding: utf-8 -*-
"""第四阶段：批量产出搜索区域与尺度，并核对「已标组的搜索范围覆盖」。

与检测命中率严格分开：这里只回答「标注组落没落在我们准备搜索的范围内」，
不回答「检测器有没有找到它」。

v2 交付物：
  search_regions.json   逐页区域 + 尺度 + 完成度 + 审计（含半页明细）
  rule_masks.json       栏线掩膜的可重建记录（两种口径 + sha256）
  *_state.json          重复运行一致性（content hash）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2                                        # noqa: E402
import numpy as np                                # noqa: E402

import dual_track_stage1_data as D                # noqa: E402
from dual_track_search import (DEFAULT_CONFIG, payload_mask_record,  # noqa: E402
                               plan_page)
from ocr.vertical import imread_unicode, page_layout      # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
OUT = ROOT / 'output' / 'dual_tracks' / 'stage4'
TZ = timezone(timedelta(hours=8))

# 区域级尺度保留字段：per_column / excluded_samples 只留页面级，避免产物膨胀
REGION_SCALE_KEEP = ('H', 'G', 'glyph_pitch', 'page_height', 'unit', 'source',
                     'sample_count', 'dispersion', 'confidence_status',
                     'measured_confidence', 'value_origin', 'inherited_fields',
                     'inherited_note', 'failure_reason', 'small_glyph_status',
                     'small_glyph_height', 'small_glyph_width', 'small_glyph_gap',
                     'small_glyph_pitch', 'filter_conditions')


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon_sha256(obj: Any) -> str:
    """规范化 JSON 哈希：键排序、无空白，用于「同一输入两次运行是否一致」。"""
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def upstream_fingerprints() -> Dict[str, Any]:
    items = {
        'stage2_cross_stage_contract': S2 / 'cross_stage_contract.json',
        'stage2_annotations': S2 / 'annotations.json',
        'stage2_negative_regions': S2 / 'negative_regions.json',
        'stage3_baseline_snapshot': S3 / 'baseline_snapshot.json',
        'stage3_frozen_layouts': S3 / 'frozen_layouts.json',
        'stage3_frozen_layout_labels': S3 / 'frozen_layout_labels.json',
        'stage3_evaluation_config': S3 / 'evaluation_config.json',
        'stage3_evaluation_targets': S3 / 'evaluation_targets.json',
        'stage3_protocol_addendum': S3 / 'protocol_addendum.json',
        'stage3_metrics_c': S3 / 'metrics' / 'c.json',
        'stage3_report': S3 / '阶段报告.md',
    }
    out = {}
    for k, p in items.items():
        out[k] = {'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                  'sha256': sha256_of(p) if p.exists() else None,
                  'exists': p.exists()}
    return out


def book_reference(layouts: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """书级参考 = 该书各页「页内列宽中位数」的中位数（与阶段 3 一致）。"""
    by_book: Dict[str, Dict[str, Any]] = {}
    all_medians = []
    for book in layouts['by_book']:
        pages = [r for r in layouts['pages'] if r['book'] == book and r['status'] == 'ok'
                 and r.get('cols')]
        medians = [statistics.median([c['width'] for c in r['cols']]) for r in pages]
        all_medians.extend(medians)
        by_book[book] = {'cols': sum(r['n_cols'] for r in pages),
                         'median_width': round(statistics.median(medians), 3) if medians else None,
                         'pages': len(pages)}
    glob = {'cols': sum(v['cols'] for v in by_book.values()),
            'median_width': round(statistics.median(all_medians), 3) if all_medians else None}
    return by_book, glob


def load_labeled_pages() -> List[Dict[str, Any]]:
    """页清单 = 有标注的页 ∪ 有确定负例的页（与阶段 3 一致，便于对照）。"""
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    pages: Dict[str, set] = {}
    for g in ann['groups']:
        pages.setdefault(g['page_id'], set()).add('annotated')
    for r in neg['regions']:
        pages.setdefault(r['page_id'], set()).add('negative_region')
    out = []
    for pid in sorted(pages):
        book, num = pid.split('/page_')
        out.append({'page_id': pid, 'book': book, 'page': int(num),
                    'image_path': str((D.BOOKS / book / D.page_file_name(int(num)))
                                      .relative_to(ROOT)).replace('\\', '/'),
                    'reasons': sorted(pages[pid])})
    return out


def coverage_of_group(g: Dict[str, Any], regions: Sequence[Dict[str, Any]]
                      ) -> Dict[str, Any]:
    """一个标注组与搜索区域的关系：完整/部分/跨窗/无覆盖/歧义。"""
    b = g['group_bbox']
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    # 标注确定性是**附加信息**，不覆盖几何覆盖结论
    certainty_flag = ('ambiguous'
                      if (str(g.get('certainty') or '').startswith('ambiguous')
                          or str(g.get('certainty')) in ('待核', 'low')) else 'clear')

    def contains(r, box):
        return (r['bbox'][0] <= box[0] and box[2] <= r['bbox'][2]
                and r['bbox'][1] <= box[1] and box[3] <= r['bbox'][3])

    hits = [r for r in regions
            if r['bbox'][0] <= cx < r['bbox'][2] and r['bbox'][1] <= cy < r['bbox'][3]]
    full = [r for r in regions if contains(r, b)]
    if full:
        chosen = min(full, key=lambda r: (r['bbox'][2] - r['bbox'][0]))
        status = 'fully_covered'
    elif hits:
        chosen = min(hits, key=lambda r: (r['bbox'][2] - r['bbox'][0]))
        # 单个窗口装不下：如果相邻窗口的并集能盖住整组，算跨窗；否则部分覆盖
        touching = [r for r in regions
                    if not (r['bbox'][2] <= b[0] or r['bbox'][0] >= b[2]
                            or r['bbox'][3] <= b[1] or r['bbox'][1] >= b[3])]
        xs = sorted((r['bbox'][0], r['bbox'][2]) for r in touching)
        merged_ok = False
        if xs:
            cur_s, cur_e = xs[0]
            for s, e in xs[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    cur_s, cur_e = s, e
            merged_ok = (cur_s <= b[0] and cur_e >= b[2])
        status = 'cross_window' if (len(touching) > 1 and merged_ok) else 'partially_covered'
    else:
        chosen = None
        status = 'uncovered'
    return {'group_id': g['group_id'], 'certainty': g.get('certainty'),
            'certainty_flag': certainty_flag,
            'group_bbox': list(b), 'status': status,
            'region_id': chosen['region_id'] if chosen else None,
            'region_source': chosen['column_range_source'] if chosen else None,
            'region_bbox': chosen['bbox'] if chosen else None,
            'candidate_regions': [r['region_id'] for r in hits],
            'coverage_status': status if certainty_flag == 'clear'
                               else f'{status}+certainty_ambiguity'}


def _slim_scale(sc: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sc.get(k) for k in REGION_SCALE_KEEP if k in sc}


def _half_summary(h: Dict[str, Any]) -> Dict[str, Any]:
    """半页明细：成组依据、拒绝原因、回退来源、完成度，去掉重复的坐标清单。"""
    paths = h.get('paths') or {}
    gr = paths.get('grouped_rules') or {}
    cv = h.get('column_validation') or {}
    fb = paths.get('halfpage_fallback') or {}
    return {
        'half_id': h['half_id'], 'half_range': h['half_range'],
        'transform': h.get('transform'),
        'coord_system': h.get('coord_system'),
        'crop_offset_x': h.get('crop_offset_x'),
        'body_y': h.get('body_y'), 'body_y_basis': h.get('body_y_basis'),
        'body_h_px': h.get('body_h_px'),
        'columns_in_half': h.get('columns_in_half'),
        'rule_candidates': len(h.get('rule_candidates') or []),
        'rule_kinds': _count([c['kind'] for c in (h.get('rule_candidates') or [])]),
        'rule_runs': h.get('rule_runs'),
        'rule_groups': [{'group_id': ((g.get('lines') or [{}])[0].get('group_id')),
                         'span': g.get('span'), 'columns': g.get('columns'),
                         'pitch_px': g.get('pitch_px'), 'pitch_cv': g.get('pitch_cv'),
                         'confidence': g.get('confidence'), 'run_index': g.get('run_index'),
                         'line_count': len(g.get('lines') or [])}
                        for g in (gr.get('groups') or [])],
        'grouped_rules_ok': gr.get('ok'),
        'broken_merge': h.get('broken_merge'),
        'column_merge': h.get('column_merge'),
        'column_validation': {k: v for k, v in cv.items() if k != 'verdicts'},
        'column_decisions': [{'col_index': v['col_index'], 'bbox': v['bbox'],
                              'decision': v['decision'], 'reasons': v['reasons'],
                              'outputs': v.get('outputs')}
                             for v in (cv.get('verdicts') or [])],
        'fallback': {k: v for k, v in fb.items() if k != 'windows'},
        'fallback_causes': [{k: w.get(k) for k in ('bbox', 'cause')}
                            for w in (fb.get('windows') or [])],
        'coverage': h.get('coverage'),
        'audit': h.get('audit'),
    }


def _count(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def run_page(pg: Dict[str, Any], cfg: Dict[str, Any], book_stats: Dict[str, Any],
             glob_stats: Dict[str, Any],
             layout_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.time()
    path = ROOT / pg['image_path']
    rec: Dict[str, Any] = {'page_id': pg['page_id'], 'book': pg['book'],
                           'page': pg['page'], 'image_path': pg['image_path'],
                           'reasons': pg['reasons']}
    img = imread_unicode(str(path))
    if img is None:
        rec.update(status='failed', status_reason='image_unreadable', regions=[])
        return rec
    H, W = img.shape[:2]
    rec['image_sha256'] = sha256_of(path)
    if layout_row is not None and layout_row.get('sha256'):
        rec['image_sha256_matches_frozen_layout'] = (
            rec['image_sha256'] == layout_row['sha256'])
    bw = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 14)
    lay = page_layout(str(path))
    plan = plan_page(pg['page_id'], str(path), bw, (W, H), lay['halves'], lay['cols'], cfg,
                     book_stats.get(pg['book']), glob_stats)
    d = plan.to_dict()
    regions = [dict(r, scale=_slim_scale(r.get('scale') or {})) for r in d['regions']]
    rec.update({'status': d['status'], 'status_reason': d['status_reason'],
                'image_size': d['image_size'], 'halves': d['halves'],
                'regions': regions, 'page_scale': d['page_scale'],
                'coverage': d['coverage'], 'unprocessed': d['unprocessed'],
                'out_of_scope': d['out_of_scope'],
                'rule_lines': d['rule_lines'],
                'rule_groups_count': len(d['rule_groups']),
                'column_validation': {
                    hid: {'accepted': v.get('accepted'), 'rejected': v.get('rejected'),
                          'split': v.get('split'), 'reference': v.get('reference'),
                          'notes': v.get('notes'),
                          'rejected_detail': [x for x in (v.get('verdicts') or [])
                                              if not x.get('accepted')],
                          'decisions': {str(x['col_index']): x['decision']
                                        for x in (v.get('verdicts') or [])}}
                    for hid, v in (d['column_validation'] or {}).items()},
                'halves_detail': [_half_summary(h) for h in d['halves_detail']],
                'timing_s': round(time.time() - t0, 3)})
    return rec


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    cov_status: Dict[str, int] = {}
    fb_cause: Dict[str, int] = {}
    scale_state = {'confident': 0, 'provisional': 0, 'inherited': 0, 'failed': 0}
    region_origin = {'measured': 0, 'inherited_from_page': 0,
                     'partially_inherited': 0, 'none': 0}
    ratios = []
    for r in rows:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
        for k, v in (r.get('coverage') or {}).get('by_source', {}).items():
            by_source[k] = by_source.get(k, 0) + v
        for k, v in (r.get('coverage') or {}).get('fallback_by_cause', {}).items():
            fb_cause[k] = fb_cause.get(k, 0) + v
        ps = r.get('page_scale') or {}
        st = ps.get('confidence_status') or 'failed'
        scale_state[st] = scale_state.get(st, 0) + 1
        for reg in r.get('regions') or []:
            o = (reg.get('scale') or {}).get('value_origin') or 'none'
            region_origin[o] = region_origin.get(o, 0) + 1
        comp = (r.get('coverage') or {}).get('completion') or {}
        if comp.get('processed_ratio') is not None:
            ratios.append(comp['processed_ratio'])
        for c in r.get('group_coverage') or []:
            cov_status[c['status']] = cov_status.get(c['status'], 0) + 1
    return {'pages': len(rows),
            'regions': sum((r.get('coverage') or {}).get('regions', 0) for r in rows),
            'by_source': dict(sorted(by_source.items())),
            'by_status': dict(sorted(by_status.items())),
            'fallback_by_cause': dict(sorted(fb_cause.items())),
            'low_confidence_regions': sum((r.get('coverage') or {}).get('low_confidence_regions', 0)
                                          for r in rows),
            'clipped_regions': sum((r.get('coverage') or {}).get('clipped_regions', 0)
                                   for r in rows),
            'unprocessed_areas': sum((r.get('coverage') or {}).get('unprocessed_areas', 0)
                                     for r in rows),
            'out_of_scope_areas': sum(((r.get('coverage') or {}).get('completion') or {})
                                      .get('out_of_scope_areas', 0) for r in rows),
            'halves_with_rules': sum((r.get('coverage') or {}).get('halves_with_rules', 0)
                                     for r in rows),
            'processed_ratio_min': round(min(ratios), 4) if ratios else None,
            'processed_ratio_mean': round(sum(ratios) / len(ratios), 4) if ratios else None,
            'group_coverage': dict(sorted(cov_status.items())),
            'scale': scale_state, 'region_scale_origin': region_origin}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage4：搜索区域与尺度')
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT_CONFIG
    layouts = load(S3 / 'frozen_layouts.json')
    lay_by_page = {r['page_id']: r for r in layouts['pages']}
    book_stats, glob_stats = book_reference(layouts)
    pages = load_labeled_pages()
    if args.only:
        keep = set(args.only)
        pages = [p for p in pages if p['page_id'] in keep]
    ann = load(S2 / 'annotations.json')
    groups_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in ann['groups']:
        groups_by_page.setdefault(g['page_id'], []).append(g)

    rows = []
    t0 = time.time()
    for i, pg in enumerate(pages, 1):
        rec = run_page(pg, cfg, book_stats, glob_stats, lay_by_page.get(pg['page_id']))
        cov = [coverage_of_group(g, rec.get('regions') or [])
               for g in groups_by_page.get(pg['page_id'], [])]
        rec['group_coverage'] = cov
        rows.append(rec)
        if not args.quiet:
            sc = rec.get('coverage') or {}
            print(f"  {i}/{len(pages)} {pg['page_id']:24s} {rec['status']:24s} "
                  f"区域 {sc.get('regions', 0):3d} {sc.get('by_source')}", flush=True)

    # 重复运行一致性：内容哈希不含耗时
    content_rows = [{k: v for k, v in r.items() if k not in ('timing_s',)} for r in rows]
    payload = {
        'stage': 'stage4_search_regions_and_scale',
        'release': cfg['version'],
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage4_run.py',
        'config': cfg,
        'upstream': upstream_fingerprints(),
        'scope': {'pages': len(rows), 'note': '页清单与阶段 3 一致（标注页 ∪ 负例页）'},
        'summary': _summary(rows),
        'pages': rows,
        'determinism': {'content_sha256': canon_sha256(content_rows),
                        'hash_rule': 'pages 去掉 timing_s 后规范化 JSON（键排序、无空白）再 sha256',
                        'note': '同一图像 + 同一配置两次运行应得同一 content_sha256'},
        'timing': {'total_s': round(time.time() - t0, 2),
                   'mean_s_per_page': round((time.time() - t0) / max(1, len(rows)), 3)},
    }
    masks = payload_mask_record(payload)
    (out / 'rule_masks.json').write_text(
        json.dumps(masks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    payload['rule_masks'] = {'path': 'output/dual_tracks/stage4/rule_masks.json',
                             'sha256': sha256_of(out / 'rule_masks.json'),
                             'summary': masks['summary']}
    f = out / 'search_regions.json'
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not args.quiet:
        s = payload['summary']
        print(f"页 {s['pages']}｜区域 {s['regions']}｜来源 {s['by_source']}")
        print(f"已标组覆盖 {s['group_coverage']}")
        print(f"页面尺度 {s['scale']}｜区域尺度来源 {s['region_scale_origin']}")
        print(f"完成度 min/mean {s['processed_ratio_min']}/{s['processed_ratio_mean']}"
              f"｜缺口 {s['unprocessed_areas']} 段｜范围外 {s['out_of_scope_areas']} 段")
        print(f"content_sha256 {payload['determinism']['content_sha256'][:16]}")
        try:
            shown = out.resolve().relative_to(ROOT)
        except ValueError:
            shown = out
        print(f"用时 {payload['timing']['total_s']}s → {shown}"
              f"（{f.stat().st_size/1e6:.2f} MB）")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
