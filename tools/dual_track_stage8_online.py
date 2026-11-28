# -*- coding: utf-8 -*-
"""从正常 OCR 入口重出 stage8 预测，并逐页对账「离线跑批 vs 在线入口」。

为什么要这个脚本
----------------
v1 的离线预测由 `tools/dual_track_stage8_run.py` 产出，在线结果由 `run_ocr()` 产出，
两条路径各自拼参数。即使两边都「该是 stage8-v1」，也没有任何证据证明它们真的同口径。
v2 起两边都读 `dual_track_fix.settings.effective_options()`，本脚本负责把这件事**验出来**：

  1. 逐页记录在线实际生效的配置（含 fingerprint），与离线 `path_stage8v1.json` 比；
  2. 把在线证据转成与离线同格式的预测文件，交给 `dual_track_stage3_score.py` 打分；
  3. 逐页对账：预测条数、每组两侧框、侧别状态、页级状态。

用法：
  python tools/dual_track_stage8_online.py
输出：
  output/dual_tracks/stage8/predictions/path_online.json
  output/dual_tracks/stage8/online_reconcile.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'output' / 'dual_tracks' / 'stage8'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def _pages() -> List[str]:
    ann = load(S2 / 'annotations.json')
    neg = load(S2 / 'negative_regions.json')
    out: List[str] = []
    for g in ann['groups']:
        if g['page_id'] not in out:
            out.append(g['page_id'])
    for r in neg['regions']:
        if r['page_id'] not in out:
            out.append(r['page_id'])
    return out


def _image_path(page_id: str) -> Path:
    ann = load(S2 / 'annotations.json')
    for g in ann['groups']:
        if g['page_id'] == page_id:
            return ROOT / g['image_path']
    neg = load(S2 / 'negative_regions.json')
    for r in neg['regions']:
        if r['page_id'] == page_id:
            return ROOT / r['image_path']
    raise KeyError(page_id)


def evidence_to_predictions(page_id: str, evidence: List[Dict[str, Any]],
                            source: str = 'ONLINE') -> List[Dict[str, Any]]:
    """在线证据（按侧）→ 冻结格式预测（按组）。"""
    by_group: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for e in evidence:
        gid = e.get('group_id')
        if gid is None:
            continue
        if gid not in by_group:
            by_group[gid] = {'group_id': gid, 'sides': {}, 'group_bbox': e.get('group_bbox'),
                             'region_id': e.get('region_id'),
                             'region_bbox': e.get('region_bbox'),
                             'region_source': e.get('region_source')}
            order.append(gid)
        by_group[gid]['sides'][e.get('side')] = e
    preds: List[Dict[str, Any]] = []
    for gid in order:
        g = by_group[gid]
        right = (g['sides'].get('right') or {})
        left = (g['sides'].get('left') or {})
        rb, lb = right.get('bbox'), left.get('bbox')
        issues = []
        if not rb:
            issues.append('missing_right_lane')
        if not lb:
            issues.append('missing_left_lane')
        gb = g.get('group_bbox')
        if gb is None and rb and lb:
            gb = [min(lb[0], rb[0]), min(lb[1], rb[1]),
                  max(lb[2], rb[2]), max(lb[3], rb[3])]
        if gb is None:
            issues.append('group_bbox_undefined')
        joint_ev = None
        for s in ('right', 'left'):
            basis = (g['sides'].get(s) or {}).get('basis') or {}
            if basis.get('joint'):
                joint_ev = basis['joint']
                break
        preds.append({
            'page_id': page_id,
            'prediction_id': gid,
            'group_bbox': list(gb) if gb else None,
            'right_bbox': list(rb) if rb else None,
            'left_bbox': list(lb) if lb else None,
            'main_bbox': None,
            'parent_bbox': g.get('region_bbox'),
            'slot_bbox': g.get('region_bbox'),
            'region_ref': {'kind': 'slot', 'region_id': g.get('region_id'),
                           'bbox': g.get('region_bbox')},
            'source': source,
            'run_status': 'ok' if not issues else 'degraded',
            'error': ';'.join(issues),
            'evidence': {},
            'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框',
            'side_status': {s: (g['sides'].get(s) or {}).get('side_status')
                            for s in ('right', 'left')},
            'status_reason': {s: (g['sides'].get(s) or {}).get('status_reason')
                              for s in ('right', 'left')},
            'bbox_source': {s: (g['sides'].get(s) or {}).get('bbox_source')
                            for s in ('right', 'left')},
            # 跑批是**复核前**检测框；入口是复核后送复识框。两者都留，逐页对账时
            # 先比 raw，再单列 reviewed 的增量，不把差异伪装成"完全一致"。
            'raw_bbox': {s: (((g['sides'].get(s) or {}).get('basis') or {}).get(
                'boundary_review') or {}).get('before') for s in ('right', 'left')
                if ((g['sides'].get(s) or {}).get('basis') or {}).get('boundary_review')},
            'review_actions': {s: (((g['sides'].get(s) or {}).get('basis') or {}).get(
                'boundary_review') or {}).get('actions') for s in ('right', 'left')
                if ((g['sides'].get(s) or {}).get('basis') or {}).get('boundary_review')},
            # 完整复核记录（before/after/actions/unresolved）随预测落盘，
            # A3/A4 验收与逐页对账都读这一份，不再另跑一次入口。
            'boundary_review': {s: ((g['sides'].get(s) or {}).get('basis') or {}).get(
                'boundary_review') for s in ('right', 'left')
                if ((g['sides'].get(s) or {}).get('basis') or {}).get('boundary_review')},
            'joint_evidence': joint_ev,
            'review_evidence': {s: (g['sides'].get(s) or {}).get('review_evidence')
                                for s in ('right', 'left')
                                if (g['sides'].get(s) or {}).get('review_evidence')},
        })
    return preds


def _key(p: Dict[str, Any]):
    return (tuple(p['group_bbox'] or ()), tuple(p['right_bbox'] or ()),
            tuple(p['left_bbox'] or ()))


def _key_raw(p: Dict[str, Any]):
    rb = (p.get('raw_bbox') or {}).get('right') or p.get('right_bbox')
    lb = (p.get('raw_bbox') or {}).get('left') or p.get('left_bbox')
    # 跑批写的是复核前 group_bbox；入口写的是复核后 union。raw 比较必须同样
    # 用复核前的两条轨现算，否则每一组都会被误报成「raw 不一致」。
    if rb and lb:
        gb = [min(lb[0], rb[0]), min(lb[1], rb[1]),
              max(lb[2], rb[2]), max(lb[3], rb[3])]
    else:
        gb = p.get('group_bbox')
    return (tuple(gb or ()), tuple(rb or ()), tuple(lb or ()))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='stage8v2', help='对账用的离线快照 tag（path_<tag>.json）')
    ap.add_argument('--tag', default='online', help='本次在线预测文件名 tag')
    ap.add_argument('--reconcile-only', action='store_true',
                    help='复用已有 path_<tag>.json，只重算对账（不跑入口）')
    args = ap.parse_args()
    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    os.environ['TCM_OCR_DUAL_TRACK_MODE'] = 'live'
    os.environ['TCM_OCR_DUAL_TRACK_DETECTOR'] = 'stage8'
    # 待核侧要留原图证据：复核裁图随同一次入口落盘，A4 直接引用这里的路径。
    os.environ['TCM_OCR_DUAL_TRACK_CROP_DIR'] = str(OUT / 'acceptance' / 'online_review_crops')

    import app.config as C
    from importlib import reload
    reload(C)
    import ocr.pipeline as PL
    reload(PL)

    # 在线侧的有效配置：与入口同一来源（reload 后的 CONFIG.ocr）；入口若将来透出 plan
    # 则优先用入口自己的记录。两者指纹应当相同。
    from dual_track_fix.settings import effective_options
    eff_live = effective_options({'detector': C.CONFIG.ocr.dual_track_detector})

    pages = _pages()
    preds: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    reuse = OUT / 'predictions' / f'path_{args.tag}.json'
    if args.reconcile_only:
        if not reuse.is_file():
            print('缺少可复用的在线预测：', reuse)
            return 2
        doc0 = load(reuse)
        preds = doc0['predictions']
        runs = doc0.get('page_runs') or []
        pages = [r['page_id'] for r in runs] or pages
        print('复用', reuse, 'preds', len(preds))
    for pid in ([] if args.reconcile_only else pages):
        t0 = time.time()
        img = _image_path(pid)
        try:
            res = PL.run_ocr(str(img), page_id=pid)
        except Exception as exc:                       # 入口本身不允许把整页打挂
            runs.append({'page_id': pid, 'status': 'exception',
                         'error': f'{type(exc).__name__}: {exc}'[:200],
                         'effective': eff_live})
            continue
        ev = list(res.dual_track_evidence or [])
        plan_eff = None
        candidate_audit = []
        for a in (res.dual_track_audit or []):
            if not isinstance(a, dict):
                continue
            if a.get('kind') == 'plan' and a.get('effective'):
                plan_eff = a['effective']
            if a.get('kind') in ('window_superseded', 'insufficient_pair_shape_support',
                                 'rule_attached_pair_recovered', 'group_failed'):
                candidate_audit.append(a)
        p_preds = evidence_to_predictions(pid, ev)
        preds.extend(p_preds)
        n_side_ok = sum(1 for e in ev if e.get('side_status') == 'ok')
        runs.append({'page_id': pid, 'status': getattr(res, 'dual_track_status', None),
                     'effective': plan_eff or eff_live,
                     'effective_source': 'entry_plan' if plan_eff else 'config',
                     'n_evidence': len(ev), 'n_groups': len(p_preds),
                     'n_side_ok': n_side_ok,
                     'candidate_audit': candidate_audit,
                     'n_blocks': len(res.blocks or []),
                     'elapsed_s': round(time.time() - t0, 2)})
        print(f'{pid:22s} {str(getattr(res, "dual_track_status", "")):10s} '
              f'ev={len(ev):3d} groups={len(p_preds):2d} ok={n_side_ok:3d}')

    if not args.reconcile_only:
        doc = {'path': 'ONLINE', 'entry': 'ocr.pipeline.run_ocr(detector=stage8)',
               'predictions': preds, 'page_runs': runs,
               'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
               'timing_s': round(sum(r.get('elapsed_s') or 0 for r in runs), 2)}
        d = OUT / 'predictions'
        d.mkdir(parents=True, exist_ok=True)
        (d / f'path_{args.tag}.json').write_text(
            json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    # ---------------- 逐页对账 ----------------
    off = load(OUT / 'predictions' / f'path_{args.ref}.json')
    off_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for p in off['predictions']:
        off_by_page.setdefault(p['page_id'], []).append(p)
    on_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        on_by_page.setdefault(p['page_id'], []).append(p)

    off_fp = (off.get('effective') or {}).get('fingerprint')
    off_runs = {r['page_id']: r for r in (off.get('page_runs') or [])}
    recon: List[Dict[str, Any]] = []
    for pid in pages:
        a = off_by_page.get(pid, [])
        b = on_by_page.get(pid, [])
        ka = sorted(_key(p) for p in a)
        kb = sorted(_key(p) for p in b)
        kra = sorted(_key(p) for p in a)
        krb = sorted(_key_raw(p) for p in b)
        run = next((r for r in runs if r['page_id'] == pid), {})
        on_fp = (run.get('effective') or {}).get('fingerprint')
        ref_fp = off_fp or (off_runs.get(pid, {}).get('effective') or {}).get('fingerprint')
        status_counts: Dict[str, int] = {}
        n_actions = n_unresolved = 0
        for p in b:
            for s in ('right', 'left'):
                v = (p.get('side_status') or {}).get(s)
                if v:
                    status_counts[v] = status_counts.get(v, 0) + 1
                br = ((p.get('boundary_review') or {}).get(s) or {})
                n_actions += len(br.get('actions') or [])
                n_unresolved += len(br.get('unresolved') or [])
        off_ids = {p['prediction_id'] for p in a}
        on_ids = {p['prediction_id'] for p in b}
        recon.append({
            'page_id': pid,
            'offline_predictions': len(a), 'online_predictions': len(b),
            'offline_groups': len(off_ids),
            'online_groups': len(on_ids),
            'group_ids_identical': off_ids == on_ids,
            'candidate_audit': run.get('candidate_audit', []),
            'both_lane_groups_offline': sum(
                1 for p in a if p.get('right_bbox') and p.get('left_bbox')),
            'both_lane_groups_online': sum(
                1 for p in b if p.get('right_bbox') and p.get('left_bbox')),
            'review_actions': n_actions,
            'unresolved_reviews': n_unresolved,
            'boxes_identical': ka == kb,
            'raw_boxes_identical': kra == krb,
            'reviewed_diff_groups': sum(_key_raw(p) != _key(p) for p in b),
            'reviewed_diff_sides': sum(
                raw != p.get(f'{side}_bbox') for p in b
                for side, raw in (p.get('raw_bbox') or {}).items()),
            'side_status_counts': status_counts,
            'missing_in_online': [list(x) for x in (set(ka) - set(kb))][:5],
            'extra_in_online': [list(x) for x in (set(kb) - set(ka))][:5],
            'status': run.get('status'),
            'n_side_ok': run.get('n_side_ok'),
            'effective_matches_offline': (on_fp == ref_fp) if on_fp and ref_fp else None,
            'effective_fingerprint': on_fp,
            'reference_effective_fingerprint': ref_fp,
        })

    summary = {
        'pages': len(pages),
        'all_boxes_identical': all(r['boxes_identical'] for r in recon),
        'all_raw_boxes_identical': all(r['raw_boxes_identical'] for r in recon),
        'reviewed_diff_pages': [r['page_id'] for r in recon
                                if r['reviewed_diff_groups']],
        'review_actions_total': sum(r['review_actions'] for r in recon),
        'reviewed_diff_groups_total': sum(r['reviewed_diff_groups'] for r in recon),
        'reviewed_diff_sides_total': sum(r['reviewed_diff_sides'] for r in recon),
        'unresolved_reviews_total': sum(r['unresolved_reviews'] for r in recon),
        'all_group_ids_identical': all(r['group_ids_identical'] for r in recon),
        'both_lane_groups': (sum(r['both_lane_groups_offline'] for r in recon),
                             sum(r['both_lane_groups_online'] for r in recon)),
        'page_statuses': sorted({r['status'] for r in recon if r.get('status')}),
        'side_status_counts_total': {
            k: sum(r['side_status_counts'].get(k, 0) for r in recon)
            for k in sorted({k for r in recon for k in r['side_status_counts']})},
        'effective_identical': bool(recon) and all(
            r['effective_matches_offline'] is True for r in recon),
        'offline_ref': f'path_{args.ref}.json',
        'offline_fingerprint': off_fp,
        'online_fingerprints': sorted({r['effective_fingerprint'] for r in recon if r['effective_fingerprint']}),
        'page_mismatch': [r['page_id'] for r in recon if not r['boxes_identical']],
        'offline_total': len(off['predictions']), 'online_total': len(preds),
    }
    (OUT / f'online_reconcile_{args.ref}.json').write_text(
        json.dumps({'summary': summary, 'pages': recon}, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
