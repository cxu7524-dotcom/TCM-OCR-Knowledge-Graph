# -*- coding: utf-8 -*-
"""stage8 从正常 OCR 入口验收修正版（p12 真实页 + 开关一致性 + 侧别隔离）。

跑的是 `ocr.pipeline.run_ocr`，不是评测脚本；检测器由 `OCR_DUAL_TRACK_DETECTOR`
环境变量切到 stage8。检查项与判定都只用检出侧证据 + 原图墨迹，不比对标注。

输出：output/dual_tracks/stage8/acceptance/checklist.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage8' / 'acceptance'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
IMG = ROOT / '古籍评测集' / '医家类' / '丹方彚编' / 'page_12.png'


def ink_mask(gray):
    bg = cv2.GaussianBlur(gray, (31, 31), 0).astype(np.int16)
    return (bg - gray.astype(np.int16)) >= 14


def edge_ink(ink, box) -> Dict[str, float]:
    """框外紧邻一行与框内首/末两行的**列方向墨迹重合度**。

    「笔画被切」= 框外 1px 那一行的墨迹列分布与框内首行高度重合（同一笔被框断）。
    只判「框边有没有墨」会把「框比字松 2px」误当成切笔——实测修正版子框的 y 边界
    来自旧入口的 ±4 余量，本来就不贴字。
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1); x2 = min(ink.shape[1], x2)
    if x2 <= x1:
        return {'top': 0.0, 'bottom': 0.0}

    def frac(row_a, row_b):
        if row_a < 0 or row_b >= ink.shape[0]:
            return 0.0
        a = ink[row_a, x1:x2]
        b = ink[row_b, x1:x2]
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float((a & b).sum() / min(a.sum(), b.sum()))

    # 真正「切断」要同时满足：框外 1px 行与框内首行重合，且框外 3px 行继续与 1px 行重合
    # （说明有一条纵向笔画在框外延续）。只满足第一条的多是相邻大字紧挨框边。
    top = min(frac(y1 - 1, y1), frac(y1 - 3, y1 - 1)) if y1 - 3 >= 0 else 0.0
    bot = min(frac(y2, y2 - 1), frac(y2 + 2, y2)) if y2 + 2 < ink.shape[0] else 0.0
    return {'top': top, 'bottom': bot}


def check_evidence_crops(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐侧检查当前证据引用的裁图，待核侧检查带上下文的原图。"""
    checked, missing = [], []
    for item in evidence:
        path = (item.get('crop_path') if item.get('side_status') == 'ok'
                else (item.get('review_evidence') or {}).get('image_path'))
        file = ROOT / path if path else None
        row = {'evidence_id': item.get('evidence_id'), 'side': item.get('side'),
               'side_status': item.get('side_status'), 'file': path}
        if file is None or not file.is_file() or file.stat().st_size == 0:
            missing.append(row)
        else:
            row['bytes'] = file.stat().st_size
            checked.append(row)
    return {'n_crops': len(checked), 'n_evidence': len(evidence),
            'missing': missing, 'first': checked[:3],
            'pass': bool(evidence) and not missing}


def online_evidence(pred_path: Path) -> List[Dict[str, Any]]:
    """把同一次正常入口的预测还原成按侧证据，供 A3/A4 直接验收。

    不再为验收另跑一次 run_ocr：A3/A4 与 19 页封版复评必须读同一份产物。
    """
    doc = json.loads(pred_path.read_text(encoding='utf-8'))
    out: List[Dict[str, Any]] = []
    for p in doc['predictions']:
        if p.get('page_id') != '丹方彚编/page_12':
            continue
        for side in ('right', 'left'):
            b = p.get(f'{side}_bbox')
            if not b:
                continue
            br = ((p.get('boundary_review') or {}).get(side) or {})
            out.append({
                'evidence_id': f"{p['prediction_id']}:{side}",
                'group_id': p['prediction_id'], 'side': side, 'bbox': list(b),
                'side_status': ((p.get('side_status') or {}).get(side)),
                'status_reason': ((p.get('status_reason') or {}).get(side)),
                'basis': {'boundary_review': br},
                'review_evidence': ((p.get('review_evidence') or {}).get(side)),
                'text': '', 'source': p.get('source'),
            })
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='path_online_v4.json',
                    help='A3/A4 使用的正常入口预测（与封版复评同一份）')
    args = ap.parse_args()
    pred_path = ROOT / 'output' / 'dual_tracks' / 'stage8' / 'predictions' / args.pred

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['TCM_OCR_DUAL_TRACK_DETECTOR'] = 'stage8'
    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    os.environ['TCM_OCR_DUAL_TRACK_MODE'] = 'live'
    crop_dir = OUT / 'crops'
    crop_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TCM_OCR_DUAL_TRACK_CROP_DIR'] = str(crop_dir)

    import app.config as C
    from importlib import reload
    reload(C)
    import ocr.pipeline as PL
    reload(PL)

    res = PL.run_ocr(str(IMG), page_id='丹方彚编/page_12')
    ev = res.dual_track_evidence or []
    # A3/A4 只读封版那一次正常入口产物；res 仅用于 A5–A7 的入口行为检查。
    ev_a = online_evidence(pred_path)
    print(f'A3/A4 复核来源：{args.pred}，轨数 {len(ev_a)}')
    from ocr.vertical import imread_unicode
    _im = imread_unicode(str(IMG))
    gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY) if _im.ndim == 3 else _im
    ink = ink_mask(gray)

    # ---- 1. 同栏上下两组：同一列槽内出现两组纵向不重叠的左右轨
    by_slot: Dict[str, List[Dict[str, Any]]] = {}
    for e in ev:
        if e.get('side') != 'right':
            continue
        rid = (e.get('region_id') or '')
        by_slot.setdefault(rid, []).append(e)
    stacked = []
    for rid, es in by_slot.items():
        boxes = sorted([e for e in es if e.get('bbox')], key=lambda x: x['bbox'][1])
        for a, b in zip(boxes, boxes[1:]):
            if b['bbox'][1] >= a['bbox'][3]:
                stacked.append({'region_id': rid, 'upper': a['bbox'], 'lower': b['bbox']})

    # ---- 2. 左右裁图：两侧裁图都存在且非空
    crop_check = check_evidence_crops(ev)

    H = 21.0                      # 丹方彚编 page_12 实测字高（§4 首尾笔画同用）
    # ---- 3. 标题隔离：核对「本组标题」与「下一标题」，并记录实际检查的框数
    # 两件事必须分开说：
    #   (1) v1 的判据是 `e['main_bbox']`，但证据里**没有这个字段**，n_titles 恒为 0，
    #       也就是说 A3 一直在空转（0 次比对也判 PASS）。v2 改成从检测器取候选框。
    #   (2) 候选框是不是「大字标题」要验：拿它和同列 OCR 行高比。p12 双轨列的字高
    #       一致（见 _a3_slot11_title.png，红框内 5 字与框下同列字同高），
    #       所以本页**没有大字标题**，A3 在本页只能是「不适用」，不能算达标。
    from dual_track_fix.detect import detect as _det
    _page = '丹方彚编/page_12'
    _rr = _det(_im, _page, 21.0, {'band_slots': False})
    cand = sorted({tuple(g['main_bbox']) for g in _rr['groups'] if g.get('main_bbox')},
                  key=lambda b: (b[1], b[0]))
    # 字高不能拿 OCR 块高来比：p12 的块是**整列**（156px），不是单字。
    # 这里直接量原文墨迹：框内墨迹行游程的中位数 = 框内字高；同列框外下半段
    # 的行游程中位数 = 该列正文字高。两者相当就说明框里写的不是大字标题。
    def _char_h(box) -> float:
        x1, y1, x2, y2 = [int(v) for v in box]
        rows = ink[max(0, y1):y2, max(0, x1):x2].sum(axis=1)
        on = rows > 0
        runs, st = [], None
        for i, v in enumerate(on):
            if v and st is None:
                st = i
            elif not v and st is not None:
                runs.append(i - st)
                st = None
        if st is not None:
            runs.append(len(on) - st)
        # 只留「一个字量级」的游程：竖排密排时相邻字会粘连成几十上百像素的长游程，
        # 那种长度反映的是粘连，不是字高。字高上限按 2.2×H 截断。
        runs = [r for r in runs if 4 <= r <= 2.2 * H]
        return float(sorted(runs)[len(runs) // 2]) if runs else 0.0

    ratios, med_h = [], 0.0
    for t in cand:
        h_in = _char_h(t)
        col = next((g['region_bbox'] for g in _rr['groups']
                    if g.get('region_bbox') and g['region_bbox'][0] < t[2]
                    and g['region_bbox'][2] > t[0]), None)
        h_col = 0.0
        if col:
            below = [col[0], t[3], col[2], col[3]]
            h_col = _char_h(below)
        if h_in and h_col:
            ratios.append(round(h_in / h_col, 2))
            med_h = med_h or h_col
    big = [r for r in ratios if r and r >= 1.4]
    title_boxes = [t for t, r in zip(cand, ratios) if r and r >= 1.4] if ratios else []

    title_viol, title_checked, title_pairs = [], 0, []
    for e in ev_a:
        b = e.get('bbox')
        if not b or not title_boxes:
            continue
        own = None
        for t in title_boxes:
            if t[0] < b[2] and t[2] > b[0] and min(b[3], t[3]) - max(b[1], t[1]) > 0:
                if own is None or t[1] > own[1]:
                    own = t
        nxt = next((t for t in title_boxes if own and t[1] >= own[3]
                    and t[0] < b[2] and t[2] > b[0]), None)
        for tb in [t for t in (own, nxt) if t]:
            title_checked += 1
            iy = min(b[3], tb[3]) - max(b[1], tb[1])
            if iy > 0.35 * (tb[3] - tb[1]):
                title_viol.append({'evidence_id': e.get('evidence_id'), 'bbox': b,
                                   'title': list(tb), 'overlap_y': iy})
        title_pairs.append({'evidence_id': e.get('evidence_id'), 'bbox': b,
                            'own_title': list(own) if own else None,
                            'next_title': list(nxt) if nxt else None})

    # 「核对本组及下一标题」点名的那一组：DF12-11（陽起石）附近 y≈655 起的组
    ref = [p for p in title_pairs if p['bbox'][1] >= 640 and p['bbox'][3] > 780
           and p['bbox'][2] > 840]
    pair = {'n_evidence_with_pair': sum(1 for p in title_pairs if p['own_title'] or p['next_title']),
            'n_evidence_with_both': sum(1 for p in title_pairs if p['own_title'] and p['next_title']),
            'df12_11_area': ref[:3],
            'n_candidates': len(cand), 'n_big_titles': len(title_boxes),
            'median_line_h': med_h, 'cand_height_ratio': ratios[:8],
            'n_ev_boxes': sum(1 for e in ev_a if e.get('bbox')),
            'sample_image': 'output/dual_tracks/stage8/_a3_slot11_title.png'}
    # v3：A3 改用**冻结标注的药名框 main_bbox** 做离线验收（tools/
    # dual_track_stage8_title_check.py），不再靠检测器候选猜「大字标题」。
    # 那里对每条轨比对本组药名 + 同列下一药名，记录实比框数与 y 交叠位置。
    tc_path = OUT / 'title_check.json'
    tc = json.loads(tc_path.read_text(encoding='utf-8')) if tc_path.exists() else {}
    a3_offline = {'n_title_boxes': tc.get('n_frozen_title_boxes'),
                  'n_lanes': tc.get('n_lanes'),
                  'n_boxes_compared': tc.get('n_boxes_compared'),
                  'n_intrusion_lanes': tc.get('n_intrusion_lanes'),
                  'prediction_source': tc.get('prediction_source'),
                  'source_matches': tc.get('prediction_source') == args.pred,
                  'pass': bool(tc.get('pass'))
                          and tc.get('prediction_source') == args.pred,
                  'detail_file': tc_path.relative_to(ROOT).as_posix()}
    # 仍保留候选框统计供对照，但判定以冻结框离线验收为准
    a3_applicable = True

    # ---- 4. 首尾笔画：轨框上下边两行必须有墨
    # 只对「合格跨度」的组检查首尾笔画：纵向 ≥2 字高。短于 2 字高的碎片组属候选，
    # 不给合格结论（它们本来就不该出现在合格证据里，列入候选数统计）。
    cut, short, short_items, checked_items = [], 0, [], []
    reviewed_ok, pending_cut = [], []
    for e in ev_a:
        if not e.get('bbox'):
            continue
        rv = (e.get('basis') or {}).get('boundary_review') or {}
        if rv.get('actions'):
            reviewed_ok.append({'evidence_id': e.get('evidence_id'),
                                'side': e.get('side'), 'actions': rv['actions'],
                                'before': rv.get('before'), 'after': rv.get('after')})
        if (e['bbox'][3] - e['bbox'][1]) < 2 * H:
            short += 1
            short_items.append({'evidence_id': e.get('evidence_id'),
                                'group_id': e.get('group_id'), 'side': e.get('side'),
                                'bbox': e['bbox'], 'height': e['bbox'][3] - e['bbox'][1],
                                'verdict': 'short_lane_skipped',
                                'side_status': e.get('side_status'),
                                'status_reason': e.get('status_reason'),
                                'review_actions': rv.get('actions') or [],
                                'original_evidence': (e.get('review_evidence') or {}).get('image_path'),
                                'note': '短于 2 字高，属候选碎片，不给合格结论'})
            continue
        m = edge_ink(ink, e['bbox'])
        item = {'evidence_id': e.get('evidence_id'), 'group_id': e.get('group_id'),
                'side': e.get('side'), 'bbox': e['bbox'], **m,
                'side_status': e.get('side_status'),
                'status_reason': e.get('status_reason'),
                'review_actions': rv.get('actions') or [],
                'original_evidence': (e.get('review_evidence') or {}).get('image_path'),
                'verdict': ('cut' if (m['top'] >= 0.5 or m['bottom'] >= 0.5)
                            else 'ok')}
        checked_items.append(item)
        if item['verdict'] == 'cut':
            cut.append(item)
        if e.get('side_status') == 'pending_validation' and 'cut' in (e.get('status_reason') or ''):
            pending_cut.append(item)
    # 每条触发项落盘，供报告逐条引用（含放大图路径）
    a4 = {'H': H, 'n_checked': len(checked_items), 'n_cut': len(cut),
          'n_pending_cut': len(pending_cut), 'n_reviewed_ok': len(reviewed_ok),
          'skipped_short_lanes': short,
          'cut_items': cut, 'pending_cut_items': pending_cut,
          'reviewed_ok_items': reviewed_ok,
          'short_items': short_items,
          'checked_items': checked_items,
          'zoom_examples': {
              'cut': 'output/dual_tracks/stage8/p12/df12_03_tail_zoom.png',
              'ok': 'output/dual_tracks/stage8/p12/df12_11_tail_zoom.png'}}
    (OUT / 'a4_detail.json').write_text(json.dumps(a4, ensure_ascii=False, indent=1) + '\n',
                                        encoding='utf-8')

    # ---- 5. 并列读数：两侧都有非空文本
    txt = {}
    for e in ev:
        gid = e.get('group_id')
        txt.setdefault(gid, {})[e.get('side')] = (e.get('text') or '').strip()
    paired = {g: t for g, t in txt.items() if t.get('left') and t.get('right')}

    # ---- 6. 开关一致性：关闭后必须 not_run，且正文 OCR 文本不变
    base_text = '\n'.join((b.text or '') for b in res.blocks)
    os.environ['TCM_OCR_DUAL_TRACK'] = '0'
    reload(C)
    reload(PL)
    off = PL.run_ocr(str(IMG), page_id='丹方彚编/page_12')
    os.environ['TCM_OCR_DUAL_TRACK'] = '1'
    reload(C)
    reload(PL)

    # ---- 7. 单侧失败隔离：**真的注入一次异常**，不是只统计现状
    # 选一条证据的左轨框，把它的识别函数替换成必抛异常，再跑一次整页：
    # 要求（a）该侧状态变 recognize_error 且理由可读；（b）同组右轨仍 ok 且有文本；
    #      （c）整页正文文本与基线一致（不因单侧失败而丢正文）。
    import ocr.dual_track as DT
    target = None
    bbox_counts = Counter(tuple(e['bbox']) for e in ev if e.get('bbox'))
    for e in ev:
        if (e.get('side') == 'left' and e.get('side_status') == 'ok' and e.get('text')
                and e.get('group_id') in paired and bbox_counts[tuple(e['bbox'])] == 1):
            target = list(e['bbox'])
            target_gid = e.get('group_id')
            break
    injection: Dict[str, Any] = {'injected': False}
    if target:
        orig = DT._crop_read

        def _patched(image, bbox, zoom, *a, **k):
            if [int(v) for v in bbox][:4] == target:
                raise RuntimeError('injected_side_failure')
            return orig(image, bbox, zoom, *a, **k)

        DT._crop_read = _patched
        try:
            inj = PL.run_ocr(str(IMG), page_id='丹方彚编/page_12')
        finally:
            DT._crop_read = orig
        inj_ev = inj.dual_track_evidence or []
        hit = [e for e in inj_ev if [int(v) for v in (e.get('bbox') or [])][:4] == target]
        mate = [e for e in inj_ev if e.get('group_id') == target_gid and e.get('side') == 'right']
        injection = {
            'injected': True, 'target_bbox': target, 'target_group': target_gid,
            'target_side_status': hit[0].get('side_status') if hit else None,
            'target_reason': hit[0].get('status_reason') if hit else None,
            'mate_side_status': mate[0].get('side_status') if mate else None,
            'mate_text': (mate[0].get('text') or '')[:12] if mate else None,
            'page_status': inj.dual_track_status,
            'n_evidence': len(inj_ev), 'n_evidence_baseline': len(ev),
            'text_identical': ('\n'.join((b.text or '') for b in inj.blocks) == base_text),
            'audit_kinds': sorted({a.get('kind') for a in
                                   (getattr(inj, 'dual_track_audit', None) or [])}),
        }

    checks = [
        {'id': 'A1', 'title': 'p12 同栏上下两组', 'pass': bool(stacked),
         'detail': {'n_stacked': len(stacked), 'examples': stacked[:4]}},
        {'id': 'A2', 'title': '左右裁图落盘', 'pass': crop_check['pass'],
         'detail': crop_check},
        {'id': 'A3', 'title': '标题隔离（冻结药名框离线验收）',
         'pass': bool(a3_offline.get('pass')) if a3_offline else (not title_viol),
         'status': 'checked',
         'detail': {**a3_offline,
                    'n_evidence': len(ev), 'n_boxes_checked_candidates': title_checked,
                    'n_candidate_titles': len(title_boxes),
                    'candidate_pair': pair, 'candidate_examples': title_viol[:3]}},
        {'id': 'A4', 'title': '首尾笔画未被切（合格跨度组）',
         'pass': bool(checked_items) and not cut,
         'detail': {'n_checked': len(checked_items), 'n_cut': len(cut),
                    'skipped_short_lanes': short,
                    'n_short_items': len(short_items),
                    'detail_file': (OUT / 'a4_detail.json').relative_to(ROOT).as_posix(),
                    'cut_items': cut, 'short_items': short_items[:8],
                    'examples': cut[:3]}},
        {'id': 'A5', 'title': '左右并列读数', 'pass': len(paired) >= 1,
         'detail': {'n_paired': len(paired),
                    'examples': [{'group_id': g, **t} for g, t in list(paired.items())[:3]]}},
        {'id': 'A6', 'title': '开关一致性（关闭 = not_run 且正文不变）',
         'pass': off.dual_track_status == 'not_run' and
                 '\n'.join((b.text or '') for b in off.blocks) == base_text,
         'detail': {'off_status': off.dual_track_status,
                    'text_identical': ('\n'.join((b.text or '') for b in off.blocks)
                                       == base_text)}},
        {'id': 'A7', 'title': '单侧失败隔离（实际注入异常）',
         'pass': bool(injection.get('injected')
                      and injection.get('target_side_status') == 'recognize_error'
                      and injection.get('mate_side_status') == 'ok'
                      and bool(injection.get('mate_text'))
                      and injection.get('text_identical')),
         'detail': {**injection,
                    'side_status': {s: sum(1 for e in ev if e.get('side_status') == s)
                                    for s in {e.get('side_status') for e in ev}}}},
    ]
    doc = {'stage': 'stage8_acceptance', 'page_id': '丹方彚编/page_12',
           'detector': 'stage8',
           'dual_track_status': res.dual_track_status,
           'a34_prediction_source': args.pred,
           'n_evidence': len(ev), 'n_groups': len({e.get('group_id') for e in ev}),
           'n_a34_lanes': len(ev_a),
           'checks': checks,
           'passed': sum(1 for c in checks if c['pass']), 'total': len(checks),
           'ocr_text_chars': len(base_text)}
    (OUT / 'checklist.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n',
                                        encoding='utf-8')
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['id']} {c['title']}")
        d = {k: v for k, v in c['detail'].items() if k != 'examples'}
        print('       ', d)
    print(f"\n状态 {res.dual_track_status} 证据 {len(ev)} 组 {doc['n_groups']} "
          f"通过 {doc['passed']}/{doc['total']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
