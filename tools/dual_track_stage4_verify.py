# -*- coding: utf-8 -*-
"""第四阶段自检：把「半页隔离 / 路径判定 / 掩膜编码 / 尺度状态」变成可重跑检查。

V01~V16 覆盖原八项要求 + 本轮四项修正。任何一项失败都返回非 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage4_run as R                    # noqa: E402
from dual_track_search import (DEFAULT_CONFIG, audit_for_page,  # noqa: E402
                               page_mask_record, regions_for_page,
                               rule_lines_for_page, scale_for_page, verify_record)

S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
TZ = timezone(timedelta(hours=8))


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def half_bounds(page: Dict[str, Any], half_id: str) -> List[int]:
    halves = page.get('halves') or []
    if len(halves) == 1:
        return halves[0]
    return halves[0] if half_id == 'right' else halves[1] if half_id == 'left' else halves[0]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(S4 / 'validation_report.json'))
    args = ap.parse_args(argv)

    payload = load(S4 / 'search_regions.json')
    masks = load(S4 / 'rule_masks.json')
    paths = load(S4 / 'path_decisions.json')
    control = load(S4 / 'control_experiment.json') if (S4 / 'control_experiment.json').exists() else {}
    pages = {p['page_id']: p for p in payload['pages']}
    checks: List[Dict[str, Any]] = []

    def add(cid: str, title: str, ok: bool, detail: str, items: Any = None) -> None:
        checks.append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                       'detail': detail, 'items': items or []})

    # ---------------- V01 p12：446 被拒 + 右半页 fallback + 不沿用该框
    p12 = pages.get('丹方彚编/page_12', {})
    right = (p12.get('column_validation') or {}).get('right', {})
    rej = right.get('rejected_detail') or []
    right_regs = [r for r in p12.get('regions', []) if r['half_id'] == 'right']
    ok = (right.get('accepted') == 0 and len(rej) == 1 and rej[0]['width'] == 446
          and right_regs and all(r['column_range_source'] == 'halfpage_fallback'
                                 for r in right_regs)
          and all(r['bbox'][2] - r['bbox'][0] < 400 for r in right_regs))
    add('V01', 'p12：446px 旧框仍被拒、右半页走 fallback、无宽于 400px 的区域', ok,
        f"accepted={right.get('accepted')} rejected={len(rej)} "
        f"right_regions={len(right_regs)} "
        f"max_w={max((r['bbox'][2]-r['bbox'][0] for r in right_regs), default=0)}")

    # ---------------- V02 逐框对照 half 边界
    bad = []
    clipped = 0
    for p in payload['pages']:
        for r in p['regions']:
            hb = half_bounds(p, r['half_id'])
            clipped += 1 if r.get('clipped') else 0
            if not (hb[0] <= r['bbox'][0] < r['bbox'][2] <= hb[1]):
                bad.append({'page': p['page_id'], 'region': r['region_id'],
                            'bbox': r['bbox'], 'half': hb})
    add('V02', '每个搜索区域都在其 half 边界内（逐框校验，不信任来源）', not bad,
        f'{sum(len(p["regions"]) for p in payload["pages"])} 个区域，越界 {len(bad)}，'
        f'经裁剪 {clipped}', bad[:5])

    # ---------------- V03 p40 对侧错框消失
    p40 = pages.get('思濟堂方書/page_40', {})
    p40_bad = []
    for r in p40.get('regions', []):
        hb = half_bounds(p40, r['half_id'])
        if not (hb[0] <= r['bbox'][0] and r['bbox'][2] <= hb[1]):
            p40_bad.append(r['region_id'])
    e = paths['before_after']['E_p40']
    add('V03', 'p40：v1 的两个对侧错框消失', (not p40_bad) and e['v1']['crossing_count'] == 2
        and e['v2']['crossing_count'] == 0,
        f"v1 跨界 {e['v1']['crossing_count']} → v2 跨界 {e['v2']['crossing_count']}；"
        f"v1 错框 {[c['region_id'] for c in e['v1']['crossing_regions']]}")

    # ---------------- V04 只在右半页画栏线的合成反例
    hi = paths['half_isolation']
    left_src = hi['measured'].get('left', {}).get('by_source', {})
    add('V04', '半页隔离：右半页栏线不被左半页复用', bool(hi.get('pass')),
        f"右侧 grouped={hi['measured'].get('right', {}).get('by_source', {}).get('grouped_rules')}"
        f"，左侧 grouped={left_src.get('grouped_rules')}，"
        f"左侧候选 x={hi['measured'].get('left', {}).get('rule_candidate_x')}")

    # ---------------- V05 一个不合格间隔/尾段不影响其余范围
    bg = paths['bad_gap']
    add('V05', '不合格间隔只断开该段，其余范围继续成组', bool(bg.get('pass')),
        f"成立 {bg['measured']['n_groups']} 组："
        + '；'.join(f"{g['span']}({g['line_count']} 线)" for g in bg['measured']['groups'])
        + f"；被拒 run {[x['why'] for x in bg['measured']['rejected_runs']]}")

    # ---------------- V06 范围外 vs 应处理未覆盖
    miss_fields = []
    kinds = {}
    unbalanced = []
    for p in payload['pages']:
        comp = (p.get('coverage') or {}).get('completion') or {}
        for k in ('should_process_px', 'covered_px', 'processed_ratio', 'gutter_px',
                  'accounted_ratio', 'uncovered_px', 'uncovered_areas', 'out_of_scope_areas'):
            if k not in comp:
                miss_fields.append(f"{p['page_id']}:{k}")
        for h in p.get('halves_detail') or []:
            c = h.get('coverage') or {}
            if c.get('should_process_px') != (
                    c.get('covered_px', 0) + c.get('gutter_px', 0) + c.get('uncovered_px', 0)):
                unbalanced.append({'page': p['page_id'], 'half': h['half_id'],
                                   'should': c.get('should_process_px'),
                                   'sum': c.get('covered_px', 0) + c.get('gutter_px', 0)
                                          + c.get('uncovered_px', 0)})
        for u in p.get('unprocessed') or []:
            kinds[u.get('kind')] = kinds.get(u.get('kind'), 0) + 1
            if not u.get('why') or 'cause' not in u:
                miss_fields.append(f"{p['page_id']}:uncovered_without_reason")
        for o in p.get('out_of_scope') or []:
            kinds[o.get('kind')] = kinds.get(o.get('kind'), 0) + 1
            if not o.get('why'):
                miss_fields.append(f"{p['page_id']}:out_of_scope_without_why")
    add('V06', '范围外与应处理未覆盖分开记账、完成度账本闭合', not miss_fields and not unbalanced,
        f"范围外/未覆盖段 {kinds}；完成度 min {payload['summary']['processed_ratio_min']}；"
        f"缺字段 {len(miss_fields)}；账本不平 {len(unbalanced)}", (miss_fields + unbalanced)[:6])

    # ---------------- V07 列框三例路径
    exp = {'A1_body_overlap': ['fallback', 'fallback'],
           'A2_rule_inside_box': 'split',
           'A3_no_reference_wide_box': 'fallback'}
    got = {c['case']: c['path'] for c in paths['column_cases']}
    ok = all(got.get(k) == v for k, v in exp.items())
    full = all(all(x in c for x in ('case', 'measured', 'path', 'reason'))
               for c in paths['column_cases'])
    add('V07', '列框校验三例：实测值 + 路径 + 理由齐备且路径符合预期', ok and full,
        '；'.join(f"{c['case']}={c['path']}" for c in paths['column_cases']))

    # ---------------- V08 p12 11 组覆盖
    cov12 = [c['status'] for c in p12.get('group_coverage') or []]
    add('V08', 'p12 的 11 个标注组仍全部落在搜索区域内', len(cov12) == 11
        and all(s == 'fully_covered' for s in cov12), f'{len(cov12)} 组：{sorted(set(cov12))}')

    # ---------------- V09 掩膜编码一致
    enc = paths['encoding']
    m = enc['measured']
    add('V09', '同一前景 0/1 与 0/255 得到同一 H/G/pitch；单点斜线被拒', bool(enc.get('pass')),
        f"0/1 {m['unit_0_1']['H']}/{m['unit_0_1']['G']}/{m['unit_0_1']['glyph_pitch']}，"
        f"0/255 {m['unit_0_255']['H']}/{m['unit_0_255']['G']}/{m['unit_0_255']['glyph_pitch']}，"
        f"一致={m['identical']}；斜线 bands={m['diagonal']['bands']} "
        f"status={m['diagonal']['status']}")

    # ---------------- V10 掩膜可重建
    rebuild_bad = []
    for rec in masks['pages']:
        chk = verify_record(rec)
        if not all(v['match'] for v in chk.values()):
            rebuild_bad.append(rec['page_id'])
    add('V10', '栏线掩膜记录可重建且 sha256 一致', not rebuild_bad,
        f"{len(masks['pages'])} 页；不一致 {len(rebuild_bad)}；"
        f"线数 {masks['summary']}", rebuild_bad[:5])

    # ---------------- V17 断线合并保留逐段证据（贴字/置信/原图 y 边界）
    seg_bad = []
    for p in payload['pages']:
        for ln in p.get('rule_lines') or []:
            segs = ln.get('segments') or []
            if not segs:
                seg_bad.append(ln.get('x0'))
                continue
            for s in segs:
                for k in ('x0', 'x1', 'y0', 'y1', 'touches_text', 'confidence'):
                    if k not in s:
                        seg_bad.append(f"{p['page_id']}:{ln['x0']}:{k}")
            elig = [s for s in segs if s.get('eligible')]
            expect = ('touching' if not elig else
                      'confirmed' if len(elig) == len(segs) else 'mixed_ambiguous')
            if ln.get('status') != expect:
                seg_bad.append(f"{p['page_id']}:{ln['x0']}:status={ln.get('status')}!={expect}")
            if bool(ln.get('ambiguous')) != (expect == 'mixed_ambiguous'):
                seg_bad.append(f"{p['page_id']}:{ln['x0']}:ambiguous_flag")
    bm = paths.get('broken_merge') or {}
    add('V17', '断线合并保留逐段原图边界/贴字/置信与歧义标记', not seg_bad and bool(bm.get('pass')),
        f"111 条线逐段字段异常 {len(seg_bad)}；合成混合断线 "
        f"status={(bm.get('measured') or {}).get('cases', [{}])[0].get('merged', {}).get('status')}",
        seg_bad[:5])

    # ---------------- V18 独立掩膜按段置位（断口/贴字段不置位）
    seg_mask_bad = []
    for rec in masks['pages']:
        expect = 0
        for ln in rec['lines']:
            if not ln.get('qualified'):
                continue
            for s in ln.get('eligible_segments') or []:
                expect += (int(s['x1']) - int(s['x0'])) * (int(s['y1']) - int(s['y0']))
        got = rec['masks']['independent']['ink_px']
        if got != expect:
            seg_mask_bad.append({'page': rec['page_id'], 'got': got, 'expect': expect})
    add('V18', '独立掩膜逐段绘制：断口与贴字段留空、只覆盖合格独立段',
        not seg_mask_bad and bool(bm.get('pass')),
        f"19 页按段面积核对，不一致 {len(seg_mask_bad)}；合格线 "
        f"{masks['summary'].get('qualified')} 条 / 全部 {masks['summary'].get('total')} 条，"
        f"歧义 {masks['summary'].get('ambiguous')} 条", seg_mask_bad[:5])

    # ---------------- V19 审计与回退归因坐标系一致
    coord_bad = []
    for p in payload['pages']:
        for h in p.get('halves_detail') or []:
            hx0, hx1 = h['half_range']
            if 'original_image' not in (h.get('coord_system') or ''):
                coord_bad.append(f"{p['page_id']}#{h['half_id']}:no_coord_label")
            for r in h.get('rule_runs') or []:
                if r.get('span') and r.get('coord_system') != 'original_image':
                    coord_bad.append(f"{p['page_id']}#{h['half_id']}:run_coord")
                if r.get('span') and not (hx0 - 2 <= r['span'][0] and r['span'][1] <= hx1 + 2):
                    coord_bad.append(f"{p['page_id']}#{h['half_id']}:run_out_of_half")
            for b in h.get('broken_merge') or []:
                if b.get('coord_system') != 'original_image':
                    coord_bad.append(f"{p['page_id']}#{h['half_id']}:merge_coord")
    co = paths.get('coord_offset') or {}
    add('V19', '线候选/断线合并/run 拒绝/回退归因统一用原图坐标', not coord_bad and bool(co.get('pass')),
        f"审计记录坐标异常 {len(coord_bad)}；非零偏移反例 候选="
        f"{(co.get('measured') or {}).get('candidate_centers')} run_span="
        f"{(co.get('measured') or {}).get('run_spans')}", coord_bad[:5])

    # ---------------- V11 尺度状态自洽
    issues = []
    for p in payload['pages']:
        ps = p['page_scale']
        if ps['confidence_status'] != 'failed':
            if None in (ps['H'], ps['G'], ps['glyph_pitch']):
                issues.append(p['page_id'] + ':page_scale_partial')
            elif len({ps['H'], ps['G'], ps['glyph_pitch']}) < 3:
                issues.append(p['page_id'] + ':page_scale_duplicate')
        for r in p['regions']:
            sc = r['scale']
            origin = sc.get('value_origin')
            if origin == 'measured':
                if sc.get('H') is None or sc.get('confidence_status') not in (
                        'confident', 'provisional'):
                    issues.append(r['region_id'] + ':measured_inconsistent')
            elif origin in ('inherited_from_page', 'partially_inherited'):
                if not sc.get('inherited_fields'):
                    issues.append(r['region_id'] + ':inherited_without_fields')
                for k in sc.get('inherited_fields') or []:
                    if sc.get(k) != ps.get(k):
                        issues.append(r['region_id'] + f':{k}_not_from_page')
                if sc.get('confidence_status') != 'inherited':
                    issues.append(r['region_id'] + ':inherited_status_wrong')
            elif origin == 'none':
                if sc.get('H') is not None:
                    issues.append(r['region_id'] + ':none_but_H_present')
            else:
                issues.append(r['region_id'] + ':unknown_origin')
        if ps.get('small_glyph_status') != 'not_estimated':
            issues.append(p['page_id'] + ':small_glyph_estimated')
    add('V11', '尺度来源/继承/失败状态自洽，small_glyph 未越权估计', not issues,
        f"页面尺度 {payload['summary']['scale']}；区域来源 "
        f"{payload['summary']['region_scale_origin']}；异常 {len(issues)}", issues[:6])

    # ---------------- V12 已标组覆盖
    cov = {}
    for p in payload['pages']:
        for c in p.get('group_coverage') or []:
            cov[c['status']] = cov.get(c['status'], 0) + 1
    total = sum(cov.values())
    add('V12', '38 个已标组全部落在搜索范围内（允许 partial）',
        cov.get('uncovered', 0) == 0 and total == 38, f'共 {total} 组：{cov}', cov)

    # ---------------- V13 阶段 5 入口
    try:
        regs = regions_for_page(payload, '丹方彚编/page_12')
        sc = scale_for_page(payload, '丹方彚编/page_12')
        rl = rule_lines_for_page(payload, '思濟堂方書/page_40')
        au = audit_for_page(payload, '思濟堂方書/page_40')
        keys = {'page_id', 'half_id', 'region_id', 'bbox', 'column_range_source',
                'low_confidence', 'scale', 'validation_measurements',
                'rejection_reasons', 'transform', 'half_bounds'}
        rec = page_mask_record(pages['思濟堂方書/page_40'])
        ok = bool(regs) and keys <= set(regs[0]) and 'H' in sc and rl and \
            'coverage' in au and rec['masks']['independent']['sha256']
    except Exception as exc:                       # noqa: BLE001
        ok, regs, sc, rl = False, [], {'error': str(exc)}, []
    add('V13', '阶段 5 可直接读取区域、尺度、栏线记录与审计', ok,
        f"regions={len(regs)}；rule_lines={len(rl)}；scale_keys={sorted(sc)[:5]}")

    # ---------------- V14 上游指纹 + 原图哈希
    up = payload['upstream']
    miss = [k for k, v in up.items() if not v.get('exists') or not v.get('sha256')]
    img_bad = [p['page_id'] for p in payload['pages']
               if p.get('image_sha256_matches_frozen_layout') is not True]
    add('V14', '上游冻结产物指纹齐全；原图 sha256 与冻结布局一致', not miss and not img_bad,
        f'上游 {len(up)} 项缺 {len(miss)}；原图不一致 {len(img_bad)}', miss + img_bad)

    # ---------------- V15 重复运行一致性
    content_rows = [{k: v for k, v in p.items() if k != 'timing_s'} for p in payload['pages']]
    recomputed = R.canon_sha256(content_rows)
    pg = {'page_id': '思濟堂方書/page_40', 'book': '思濟堂方書', 'page': 40,
          'image_path': '古籍评测集/医家类/思濟堂方書/page_40.png',
          'reasons': ['annotated']}
    layouts = load(ROOT / 'output/dual_tracks/stage3/frozen_layouts.json')
    book_stats, glob = R.book_reference(layouts)
    a = R.run_page(pg, DEFAULT_CONFIG, book_stats, glob)
    b = R.run_page(pg, DEFAULT_CONFIG, book_stats, glob)
    same = ([r['region_id'] for r in a['regions']] == [r['region_id'] for r in b['regions']]
            and [r['bbox'] for r in a['regions']] == [r['bbox'] for r in b['regions']]
            and a['page_scale']['H'] == b['page_scale']['H'])
    add('V15', 'content hash 与记录一致；同输入两次运行区域/尺度相同',
        recomputed == payload['determinism']['content_sha256'] and same,
        f"记录 {payload['determinism']['content_sha256'][:12]}，重算 {recomputed[:12]}，"
        f"两跑一致={same}")

    # ---------------- V16 单因素对照版本
    has = bool(control.get('groups'))
    ver_ok = (control.get('regions_release') == payload['release']
              and control.get('regions_content_sha256') == payload['determinism']['content_sha256'])
    add('V16', '单因素对照用本版区域重跑并记录版本', has and ver_ok
        and bool(control.get('compatibility_notes')),
        (f"regions_release={control.get('regions_release')}；"
         f"老/新预测 {control['groups']['old']['n_predictions']}/"
         f"{control['groups']['new']['n_predictions']}；"
         f"@0.5 {control['groups']['old']['overall']['0.5']['group_hits']}/"
         f"{control['groups']['new']['overall']['0.5']['group_hits']}")
        if has else '缺对照实验结果')

    summary = {'checks': len(checks),
               'pass': sum(1 for c in checks if c['status'] == 'pass'),
               'fail': sum(1 for c in checks if c['status'] == 'fail'),
               'failed_ids': [c['id'] for c in checks if c['status'] == 'fail']}
    out = {'stage': 'stage4_search_regions_and_scale', 'release': DEFAULT_CONFIG['version'],
           'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
           'config_version': DEFAULT_CONFIG['version'],
           'search_regions': {'path': 'output/dual_tracks/stage4/search_regions.json',
                              'pages': len(payload['pages']),
                              'regions': payload['summary']['regions'],
                              'content_sha256': payload['determinism']['content_sha256']},
           'summary': summary, 'checks': checks}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(f"自检 {summary['checks']} 项：通过 {summary['pass']}、失败 {summary['fail']} "
          f"{summary['failed_ids']}")
    for c in checks:
        print(f"  [{'PASS' if c['status'] == 'pass' else 'FAIL'}] {c['id']} {c['title']} — "
              f"{c['detail'][:110]}")
    return 0 if summary['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
