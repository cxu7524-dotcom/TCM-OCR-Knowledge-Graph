# -*- coding: utf-8 -*-
"""第五阶段自检：把验收要求变成可重跑检查。"""

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

from ocr.vertical import imread_unicode                       # noqa: E402
from dual_track_track import DEFAULT_CONFIG, groups_for_page, page_ink  # noqa: E402
from dual_track_track.api import audit_for_page, windows_for_page  # noqa: E402
from dual_track_track.dedupe import bbox_iou                 # noqa: E402
from dual_track_track.inputs import S5, canon_sha256, sha256_of  # noqa: E402

TZ = timezone(timedelta(hours=8))


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(S5 / 'validation_report.json'))
    args = ap.parse_args(argv)
    payload = load(S5 / 'groups.json')
    metrics = load(S5 / 'metrics.json') if (S5 / 'metrics.json').exists() else {}
    windows = load(S5 / 'windows.json') if (S5 / 'windows.json').exists() else {'pages': {}}
    pages = {p['page_id']: p for p in payload['pages']}
    checks: List[Dict[str, Any]] = []

    def add(cid, title, ok, detail, items=None):
        checks.append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                       'detail': detail, 'items': items or []})

    # V01 输入可追溯
    inp = payload['inputs']
    miss = [k for k, v in inp.items() if isinstance(v, dict) and not v.get('exists')]
    mask_ok = (inp.get('stage4_rule_masks_sha256_recorded')
               == sha256_of(S5.parent / 'stage4' / 'rule_masks.json'))
    add('V01', '输入指纹齐全，第四阶段内容指纹与 rule_masks 哈希对得上',
        not miss and inp.get('stage4_content_prefix_ok') and mask_ok,
        f"stage4 content={str(inp.get('stage4_content_sha256'))[:16]} "
        f"prefix_ok={inp.get('stage4_content_prefix_ok')} 缺文件 {len(miss)}",
        miss)

    # V02 坐标与轨框
    bad = []
    for pid, p in pages.items():
        W, H = p['image_size']
        for g in p.get('groups') or []:
            gb, rb, lb = g.get('group_bbox'), g.get('right_bbox'), g.get('left_bbox')
            if not gb or not rb or not lb:
                bad.append(f'{g["group_id"]}:missing_box')
                continue
            if not (0 <= gb[0] < gb[2] <= W and 0 <= gb[1] < gb[3] <= H):
                bad.append(f'{g["group_id"]}:group_out')
            if rb[0] < lb[0]:
                bad.append(f'{g["group_id"]}:right_left_order')
            if rb[2] > gb[2] + 1 or lb[0] < gb[0] - 1:
                bad.append(f'{g["group_id"]}:track_outside_group')
    add('V02', '组框在页内、轨框在组内、右轨在左轨右侧', not bad,
        f'{sum(len(p.get("groups") or []) for p in pages.values())} 组，异常 {len(bad)}',
        bad[:5])

    # V03 断点枚举与位置
    enum = set(DEFAULT_CONFIG['states']['stop_types'])
    bad = []
    for p in pages.values():
        for g in p.get('groups') or []:
            ev = g.get('stop_events') or []
            if not ev:
                bad.append(f'{g["group_id"]}:no_stop_event')
            for e in ev:
                if e.get('type') not in enum:
                    bad.append(f'{g["group_id"]}:bad_enum={e.get("type")}')
                if e.get('at_y') is None or e.get('decided_at_y') is None:
                    bad.append(f'{g["group_id"]}:no_position')
    add('V03', '每个断点都有合法枚举 + 原图 y + 确认位置', not bad,
        f'断点类型 {payload["summary"]["stop_events"]}；异常 {len(bad)}', bad[:5])

    # V04 重叠窗口去重
    bad = []
    for pid, p in pages.items():
        gs = p.get('groups') or []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                iou = bbox_iou(gs[i]['group_bbox'], gs[j]['group_bbox'])
                if iou > DEFAULT_CONFIG['dedupe']['iou']:
                    bad.append(f'{pid}:{gs[i]["group_id"]}~{gs[j]["group_id"]}:{iou:.2f}')
    add('V04', '重叠回退窗口的同一组只输出一次（无 IoU>0.5 的重复组）', not bad,
        f'去重日志 {sum(len(p.get("dedupe") or []) for p in pages.values())} 条，重复 {len(bad)}',
        bad[:5])

    # V05 尺度来源
    bad = []
    for p in pages.values():
        for r in p.get('regions') or []:
            if r.get('skipped'):
                continue
            if not r.get('scale_status', {}).get('H', {}).get('value'):
                bad.append(f'{r["region_id"]}:no_scale')
        for g in p.get('groups') or []:
            if not g.get('scale_used', {}).get('H'):
                bad.append(f'{g["group_id"]}:no_scale_used')
    skipped = sum(1 for p in pages.values() for r in p.get('regions') or []
                  if r.get('skipped'))
    add('V05', '每组记录所用尺度；跳过的区域有原因', not bad,
        f'跳过区域 {skipped}；异常 {len(bad)}', bad[:5])

    # V06 决定性
    content = [{k: v for k, v in r.items() if k != 'timing_s'} for r in payload['pages']]
    recalculated = canon_sha256(content)
    add('V06', '内容哈希可重算（同输入同配置两次运行一致）',
        recalculated == payload['determinism']['content_sha256'],
        f"{recalculated[:16]} vs {payload['determinism']['content_sha256'][:16]}")

    # V07 评分口径
    m = metrics or {}
    cfg = m.get('config') or {}
    add('V07', '评分使用阶段 3 冻结配置，并与冻结基线同口径对照',
        cfg.get('release') == 'stage3-v1' and bool(m.get('baseline_comparison')),
        f"config={cfg.get('release')} thresholds="
        f"{(cfg.get('runtime') or {}).get('group_iou_thresholds')} "
        f"baseline={((m.get('baseline_comparison') or {}).get('baseline_id'))}")

    # V08 pending / 拒绝有原因
    bad = []
    for p in pages.values():
        for g in p.get('groups') or []:
            if g.get('pending') and not g.get('pending_reasons'):
                bad.append(f'{g["group_id"]}:pending_without_reason')
            if g.get('state') != 'completed':
                bad.append(f'{g["group_id"]}:non_completed_in_output')
    add('V08', '待核组带原因；输出组状态都是 completed', not bad,
        f'待核 {payload["summary"]["pending_groups"]} 组；异常 {len(bad)}', bad[:5])

    # V09 读取入口
    try:
        pid = next(iter(pages))
        gs = groups_for_page(payload, pid)
        ws = windows_for_page(payload, pid)
        au = audit_for_page(payload, pid)
        ok = isinstance(gs, list) and isinstance(ws, dict) and 'coverage' in au
        ok = ok and (pid not in windows.get('pages', {})
                     or isinstance(windows['pages'][pid], dict))
        with_raise = False
        try:
            groups_for_page(payload, 'no/such_page')
        except KeyError:
            with_raise = True
        ok = ok and with_raise
    except Exception as exc:                              # noqa: BLE001
        ok, gs = False, []
    add('V09', '阶段 6 读取入口可用（组/窗口/审计，未知页抛 KeyError）', ok,
        f'示例页 {pid} 组数 {len(gs)}')

    # V10 原图指纹
    bad = []
    for pid, p in pages.items():
        img = imread_unicode(str(ROOT / p['image_path']))
        if img is None:
            bad.append(f'{pid}:unreadable')
            continue
        if list(img.shape[1::-1]) != list(p['image_size']):
            bad.append(f'{pid}:size_mismatch')
    add('V10', '原图可读且尺寸与产物一致', not bad, f'{len(pages)} 页，异常 {len(bad)}',
        bad[:5])

    # V11 p12 用例摘要（只记录事实，不在自检里判优劣）
    p12 = pages.get('丹方彚编/page_12', {})
    g12 = p12.get('groups') or []
    same_col = []
    for i in range(len(g12)):
        for j in range(i + 1, len(g12)):
            a, b = g12[i]['group_bbox'], g12[j]['group_bbox']
            xo = max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2] - a[0],
                                                                        b[2] - b[0]))
            if xo > 0.6:
                same_col.append([g12[i]['group_id'], g12[j]['group_id'], round(xo, 2)])
    add('V11', 'p12 产出组与同栏上下组事实记录', True,
        f'p12 组数 {len(g12)}；同栏成对 {len(same_col)}；'
        f'缺口见 metrics.failure_attribution', same_col)

    # V12 38 组逐条复评
    rp = S5 / 'review_38.json'
    review = load(rp) if rp.exists() else {}
    rows = review.get('rows') or []
    counts = review.get('counts') or {}
    targets = {r['group_id']: r for r in rows}
    p12_targets = ('丹方彚编/p12/DF12-03', '丹方彚编/p12/DF12-11')
    ok12 = (len(rows) == 38
            and (counts.get('stage4_coverage') or {}).get('完全覆盖') == 36
            and (counts.get('stage4_coverage') or {}).get('部分覆盖') == 2
            and all(targets.get(g, {}).get('v4', {}).get('0.5', {}).get('hit')
                    and targets[g]['v4']['0.5'].get('iou', 0) >= 0.5
                    for g in p12_targets)
            and all(Path(ROOT / r['evidence']).exists()
                    for r in rows if r.get('evidence')))
    add('V12', '38 组逐条复评齐全，四阶段覆盖 36 fully + 2 partial，p12 两组命中',
        ok12, f"行 {len(rows)}；覆盖 {counts.get('stage4_coverage')}；"
        f"疑似提前断组 {counts.get('suspected_early_split')}")

    # V13 唯一字形计数 + 候选窗口日志标识
    bad = []
    min_g = int(DEFAULT_CONFIG['glyph']['min_per_side_group'])
    for p in pages.values():
        for g in p.get('groups') or []:
            for side in ('right', 'left'):
                if (g.get('glyph_support', {}).get(side, {}).get('n_unique_confirm') or 0) < min_g:
                    bad.append(f'{g["group_id"]}:{side}:unique_confirm')
    seen_ids = set()
    for pid, regions in (windows.get('pages') or {}).items():
        for rid, entries in regions.items():
            for e in entries:
                wid = e.get('window_id')
                if not wid or wid in seen_ids or 'candidate_seam' not in e:
                    bad.append(f'{pid}:{rid}:{wid}:window_log')
                seen_ids.add(wid)
    add('V13', '确认前分侧唯一字形计数 + 窗口按候选与窗口共同标识', not bad,
        f"组 {sum(len(p.get('groups') or []) for p in pages.values())}；"
        f"日志 {len(seen_ids)} 条；异常 {len(bad)}", bad[:5])

    # V14 停轨坐标 = 末字末端；硬断点三类坐标分开记录
    bad = []
    n_track_stop = 0
    n_hard_stop = 0
    for p in pages.values():
        for g in p.get('groups') or []:
            for side in ('right', 'left'):
                box = g.get(f'{side}_bbox')
                end = (g.get('glyph_support', {}).get(side, {}) or {}).get('last_glyph_end')
                if box and end != box[3]:
                    bad.append(f'{g["group_id"]}:{side}:end_vs_bbox')
            for e in g.get('stop_events') or []:
                ev = e.get('evidence') or {}
                if e['type'] == 'track_ink_end':
                    n_track_stop += 1
                    side = e['side']
                    end = (g.get('glyph_support', {}).get(side, {}) or {}).get('last_glyph_end')
                    if e['at_y'] != end or ev.get('glyph_end_y') != end:
                        bad.append(f'{g["group_id"]}:{side}:track_stop_mismatch')
                    if 'decided_at_y' not in e or 'scan_end_y' not in ev:
                        bad.append(f'{g["group_id"]}:{side}:scan_coord_missing')
                if e['type'] in ('next_title', 'rule_boundary', 'blank_gap'):
                    n_hard_stop += 1
                    if not all(k in ev for k in ('hard_boundary_y', 'scan_trigger_y',
                                                 'glyph_end_y')):
                        bad.append(f'{g["group_id"]}:{e["type"]}:hard_coord_missing')
    add('V14', 'track_ink_end.at_y 与末字末端一致；硬断点三类坐标齐全', not bad,
        f'停轨 {n_track_stop} 条 / 硬断点 {n_hard_stop} 条；异常 {len(bad)}', bad[:5])

    summary = {'checks': len(checks),
               'pass': sum(1 for c in checks if c['status'] == 'pass'),
               'fail': sum(1 for c in checks if c['status'] == 'fail'),
               'failed_ids': [c['id'] for c in checks if c['status'] == 'fail']}
    out = {'stage': 'stage5_dual_track_tracking', 'release': DEFAULT_CONFIG['version'],
           'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
           'groups_file': {'path': 'output/dual_tracks/stage5/groups.json',
                           'sha256': sha256_of(S5 / 'groups.json'),
                           'content_sha256': payload['determinism']['content_sha256']},
           'summary': summary, 'checks': checks}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(f"自检 {summary['checks']} 项：通过 {summary['pass']}、失败 {summary['fail']} "
          f"{summary['failed_ids']}")
    for c in checks:
        print(f"  [{'PASS' if c['status'] == 'pass' else 'FAIL'}] {c['id']} {c['title']} — "
              f"{c['detail'][:100]}")
    return 0 if summary['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
