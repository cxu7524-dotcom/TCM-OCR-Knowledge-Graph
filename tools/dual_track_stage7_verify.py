# -*- coding: utf-8 -*-
"""stage7 自检：入口接入 / 冻结回放一致 / 正文隔离 / 状态往返 / 侧别可追溯。

每一项都读写**真实产物**（run/、replay_compare.json、stage2 标注、store），
不做「凭构造通过」的替代。输出 output/dual_tracks/stage7/validation_report.json。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

TZ = timezone(timedelta(hours=8))
S7 = ROOT / 'output' / 'dual_tracks' / 'stage7'
RUN = S7 / 'run'


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def _iou(a: Any, b: Any) -> float:
    if not a or not b:
        return 0.0
    x = min(a[2], b[2]) - max(a[0], b[0])
    y = min(a[3], b[3]) - max(a[1], b[1])
    if x <= 0 or y <= 0:
        return 0.0
    inter = x * y
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _overlap_ratio(inner: Any, outer: Any) -> float:
    """inner 被 outer 覆盖的面积比例（用于「负例区域内不该有证据」）。"""
    if not inner or not outer:
        return 0.0
    x = min(inner[2], outer[2]) - max(inner[0], outer[0])
    y = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if x <= 0 or y <= 0:
        return 0.0
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (x * y) / area


# ------------------------------------------------------------------ 状态往返

def _state_cases() -> List[Dict[str, Any]]:
    """七类状态样例（对应验收表）；evidence=None 与 [] 是刻意分开的两种空。"""
    return [
        {'id': 'old_record', 'desc': '旧记录缺字段',
         'dict': {'page_id': 'state/legacy', 'blocks': [], 'notes': []},
         'expect': ('not_run', None, False)},
        {'id': 'disabled', 'desc': '关闭功能（未尝试）',
         'status': 'not_run', 'evidence': None, 'run': False,
         'expect': ('not_run', None, False)},
        {'id': 'skipped', 'desc': '明确跳过（示例页/无图）',
         'status': 'skipped', 'evidence': None, 'run': False,
         'expect': ('skipped', None, False)},
        {'id': 'completed_empty', 'desc': '完成，未检出符合条件的组',
         'status': 'completed', 'evidence': [], 'run': True,
         'expect': ('completed', [], True)},
        {'id': 'completed_hit', 'desc': '完成，有结果',
         'status': 'completed', 'run': True,
         'evidence': [{'evidence_id': 's/p#g001:right', 'group_id': 's/p#g001',
                       'page_id': 's/p', 'side': 'right', 'bbox': [10, 10, 30, 90],
                       'text': '陽起石', 'conf': 0.91, 'model': 'm.onnx',
                       'side_status': 'ok'},
                      {'evidence_id': 's/p#g001:left', 'group_id': 's/p#g001',
                       'page_id': 's/p', 'side': 'left', 'bbox': [32, 12, 50, 88],
                       'text': '一味', 'conf': 0.77, 'model': 'm.onnx',
                       'side_status': 'ok'}],
         'expect': ('completed', 'list2', True)},
        {'id': 'partial', 'desc': '部分失败（已成功部分仍可读）',
         'status': 'partial', 'run': True,
         'evidence': [{'evidence_id': 's/p#g002:left', 'group_id': 's/p#g002',
                       'page_id': 's/p', 'side': 'left', 'bbox': [60, 10, 80, 90],
                       'text': '半', 'conf': 0.5, 'model': 'm.onnx',
                       'side_status': 'ok'}],
         'expect': ('partial', 'list1', True)},
        {'id': 'failed', 'desc': '整体失败（保留整页 OCR）',
         'status': 'failed', 'evidence': None, 'run': True,
         'expect': ('failed', None, True)},
    ]


def _build(case: Dict[str, Any]):
    from app.schemas import OCRResult
    if 'dict' in case:
        return OCRResult.from_dict(dict(case['dict']))
    return OCRResult(page_id='state/' + case['id'],
                     dual_track_status=case['status'],
                     dual_track_evidence=case['evidence'],
                     dual_track_run=case.get('run', False))


def _shape(ocr) -> tuple:
    ev = ocr.dual_track_evidence
    if ev is None:
        sig = None
    elif not ev:
        sig = []
    else:
        sig = f'list{len(ev)}'
    return ocr.dual_track_status, sig, ocr.dual_track_run


def check_state_roundtrip() -> Dict[str, Any]:
    """七类状态：构造 → 序列化 → 存储 → 读回 → from_dict，逐项对齐。"""
    from app.config import CONFIG
    from app import store
    from app.schemas import OCRResult

    rows = []
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix='stage7_state_'))
    patched = replace(CONFIG.paths, db_path=tmp / 't.sqlite3', store_dir=tmp)
    with mock.patch.object(store, 'PATHS', patched):
        for case in _state_cases():
            ocr = _build(case)
            want = case['expect']
            got = _shape(ocr)
            json_rt = _shape(OCRResult.from_dict(json.loads(ocr.to_json())))
            store.save_page(ocr, extraction_meta={'mode': 'test'})
            back = store.get_page(ocr.page_id)
            read = OCRResult.from_dict(back['page'])
            got_db = _shape(read)
            problems = read.validate_dual_track()
            good = (got == want and json_rt == want and got_db == want and not problems)
            ok = ok and good
            rows.append({'id': case['id'], 'desc': case['desc'], 'expect': list(want),
                         'built': list(got), 'json_roundtrip': list(json_rt),
                         'after_save_and_load': list(got_db),
                         'validate_problems': problems,
                         'status': 'pass' if good else 'fail'})
        # 另一种存储路径：不带 extraction_meta（只写 pages 列，不写 ocr_payload）
        ocr = _build({'id': 'completed_empty2', 'status': 'completed',
                      'evidence': [], 'run': True})
        store.save_page(ocr)
        read = OCRResult.from_dict(store.get_page(ocr.page_id)['page'])
        col_ok = _shape(read) == ('completed', [], True)
        ok = ok and col_ok
        rows.append({'id': 'completed_empty_column_path', 'desc': '完成无结果（仅列存储）',
                     'expect': ['completed', [], True], 'built': None,
                     'after_save_and_load': list(_shape(read)),
                     'status': 'pass' if col_ok else 'fail'})
    return {'ok': ok, 'rows': rows, 'cases': len(rows)}


# ------------------------------------------------------------------ 单侧失败隔离

def check_side_isolation() -> Dict[str, Any]:
    """注入「右轨识别抛异常」的识别器：左轨证据仍可读，状态降为 partial。"""
    from ocr.dual_track import run_dual_track
    page = '丹方彚编/page_12'
    image = str(ROOT / '古籍评测集/医家类/丹方彚编/page_12.png')
    frozen = _load(ROOT / 'output/dual_tracks/stage6/groups.json')
    rec = next(r for p in frozen['pages'] if p['page_id'] == page
               for r in p['records'] if r['group_id'] == '丹方彚编/page_12#g036')

    def recognizer(_img, bbox, zoom):
        if bbox[0] >= 880:                      # 右轨（x 起点更大）→ 抛异常
            raise RuntimeError('synthetic_side_failure')
        return '左轨读数', 0.5, {'clip': [int(v) for v in bbox], 'rotation': 90,
                                 'zoom': zoom}

    rec_like = dict(rec)
    rec_like['region_bbox'] = [0, 0, 1024, 1024]
    from ocr.dual_track import build_evidence
    from ocr.vertical import imread_unicode
    img = imread_unicode(image)
    built = build_evidence([rec_like], image=img, page_id=page, width=1024, height=1024,
                           recognizer=recognizer)
    ev = built['evidence']
    right = next(e for e in ev if e['side'] == 'right')
    left = next(e for e in ev if e['side'] == 'left')
    ok = (right['side_status'] == 'recognize_error' and right['text'] == ''
          and left['side_status'] == 'ok' and left['text'] == '左轨读数'
          and built['side_counts'].get('recognize_error') == 1)
    return {'ok': ok, 'group': rec['group_id'],
            'right': {'status': right['side_status'], 'text': right['text']},
            'left': {'status': left['side_status'], 'text': left['text']},
            'audit': [a for a in built['audit'] if a['kind'] == 'recognize_error']}


def check_run_isolation() -> Dict[str, Any]:
    """双轨整体抛异常时，整页 OCR 仍返回（块数不变、状态 failed）。"""
    from unittest import mock as _mock
    import ocr.pipeline as pipeline
    image = str(ROOT / '古籍评测集/医家类/思濟堂方書/page_32.png')
    with _mock.patch.object(pipeline, 'run_dual_track',
                            side_effect=RuntimeError('synthetic_dual_failure')):
        ocr = pipeline.run_ocr(image, 'isolation/page_32')
    return {'ok': ocr.dual_track_status == 'failed' and ocr.dual_track_evidence is None
            and ocr.dual_track_run and len(ocr.blocks) > 0
            and any('异常已隔离' in n for n in ocr.notes)
            and not ocr.validate_dual_track(),
            'status': ocr.dual_track_status, 'evidence': ocr.dual_track_evidence,
            'run': ocr.dual_track_run, 'blocks': len(ocr.blocks),
            'problems': ocr.validate_dual_track(),
            'notes': [n for n in ocr.notes if '双轨' in n]}


# ------------------------------------------------- stage7-v2：状态 / 坐标 / run 列

def check_status_semantics() -> Dict[str, Any]:
    """通过正常入口注入两种异常：全部精修失败 → failed+None；单侧复识失败 → partial+list。"""
    import ocr.dual_track as dt
    import ocr.pipeline as pipeline
    import dual_track_refine

    image = str(ROOT / '古籍评测集/医家类/丹方彚编/page_12.png')
    with mock.patch.object(dual_track_refine, 'refine_group',
                           side_effect=RuntimeError('synthetic_refine_failure')):
        a = pipeline.run_ocr(image, 'v2check/all_refine_fail')

    original = dt._crop_read
    state = {'n': 0}

    def flaky(_img, bbox, zoom=1.0, min_conf=0.15):
        state['n'] += 1
        if state['n'] == 1:
            raise RuntimeError('synthetic_side_failure')
        return original(_img, bbox, zoom)

    with mock.patch.object(dt, '_crop_read', flaky):
        b = pipeline.run_ocr(image, 'v2check/side_fail')

    a_audit = a.dual_track_audit or []
    b_ev = b.dual_track_evidence or []
    cases = [
        {'case': '全部精修失败', 'status': a.dual_track_status,
         'evidence': (None if a.dual_track_evidence is None
                      else len(a.dual_track_evidence)),
         'run': a.dual_track_run, 'blocks': len(a.blocks),
         'problems': a.validate_dual_track(),
         'failed_groups': len([x for x in a_audit if x.get('kind') == 'group_failed']),
         'notes': [n for n in a.notes if '双轨' in n],
         'ok': (a.dual_track_status == 'failed' and a.dual_track_evidence is None
                and a.dual_track_run and len(a.blocks) > 0
                and not a.validate_dual_track()
                and any(x.get('kind') == 'group_failed' for x in a_audit))},
        {'case': '单侧复识失败', 'status': b.dual_track_status,
         'evidence': len(b_ev), 'run': b.dual_track_run, 'blocks': len(b.blocks),
         'problems': b.validate_dual_track(),
         'side_status': {k: sum(1 for e in b_ev if e['side_status'] == k)
                         for k in ('ok', 'recognize_error', 'pending_validation',
                                   'invalid_bbox')},
         'error_audit': [x for x in (b.dual_track_audit or [])
                         if x.get('kind') == 'recognize_error'][:3],
         'ok': (b.dual_track_status == 'partial' and b.dual_track_run
                and len(b.blocks) > 0 and not b.validate_dual_track()
                and any(e['side_status'] == 'ok' and e['text'] for e in b_ev)
                and any(e['side_status'] == 'recognize_error' for e in b_ev))},
    ]
    return {'ok': all(c['ok'] for c in cases), 'cases': cases}


def check_bbox_bounds() -> Dict[str, Any]:
    """80 px 宽图像：x2=80 合法、x2=81 拒绝、非数字拒绝；坏侧不裁图，好侧照常读。"""
    from ocr.dual_track import build_evidence
    from tests.test_dual_track_stage7 import _image, _ok_read, _record

    def run(bbox, label, side='right'):
        rec = _record()
        rec['sides'][side]['bbox_after'] = bbox
        calls = []

        def fn(_img, b, zoom):
            calls.append([int(v) for v in b])
            return '好侧读数', 0.6, {'clip': [int(v) for v in b], 'rotation': 90,
                                     'zoom': zoom}

        out = build_evidence([rec], image=_image(80, 120), page_id='v2check/page',
                             width=80, height=120, recognizer=fn)
        bad = next(e for e in out['evidence'] if e['side'] == side)
        good = next(e for e in out['evidence'] if e['side'] != side)
        return {'case': label, 'bbox': bbox,
                'bad_side': {'status': bad['side_status'],
                             'reason': bad['status_reason'], 'bbox': bad['bbox'],
                             'text': bad['text']},
                'good_side': {'status': good['side_status'], 'text': good['text']},
                'recognized_boxes': calls,
                'audit': [a for a in out['audit']
                          if a.get('kind') == 'side_invalid_bbox'][:2]}

    rows = [run([40, 52, 80, 108], 'x2=80 合法'),
            run([40, 52, 81, 108], 'x2=81 越界'),
            run(['a', None, 'c', 'd'], '非数字', side='left')]
    checks = {
        'x2=80': (rows[0]['bad_side']['status'] == 'ok'
                  and rows[0]['bad_side']['bbox'] == [40, 52, 80, 108]),
        'x2=81': (rows[1]['bad_side']['status'] == 'invalid_bbox'
                  and rows[1]['bad_side']['reason'] == 'beyond_width'
                  and rows[1]['good_side']['status'] == 'ok'
                  and rows[1]['recognized_boxes'] == [[18, 52, 30, 108]]),
        'non_numeric': (rows[2]['bad_side']['status'] == 'invalid_bbox'
                        and rows[2]['bad_side']['reason'] == 'not_numeric'
                        and rows[2]['good_side']['status'] == 'ok'
                        and rows[2]['good_side']['text'] == '好侧读数'),
    }
    return {'ok': all(checks.values()), 'checks': checks, 'rows': rows}


def check_run_flag_storage() -> Dict[str, Any]:
    """failed + None + run=true 在完整负载 / JSON 快照 / 仅 SQLite 列三种读回路径下保持一致。"""
    import sqlite3
    from app import store
    from app.config import CONFIG
    from app.schemas import OCRResult

    tmp = Path(tempfile.mkdtemp(prefix='stage7_runflag_v2_'))
    patched = replace(CONFIG.paths, db_path=tmp / 't.sqlite3', store_dir=tmp)

    def make(pid):
        return OCRResult(page_id=pid, image_path='x.png', width=10, height=10,
                         blocks=[], engine='rapidocr', model_loaded=True,
                         is_demo=False, dual_track_status='failed',
                         dual_track_evidence=None, dual_track_run=True,
                         dual_track_audit=[{'kind': 'group_failed',
                                            'group_id': f'{pid}#g1'}])

    def shape(pid):
        r = OCRResult.from_dict(store.get_page(pid)['page'])
        return [r.dual_track_status,
                None if r.dual_track_evidence is None else len(r.dual_track_evidence),
                r.dual_track_run]

    with mock.patch.object(store, 'PATHS', patched):
        store.save_page(make('v2run/payload'), extraction_meta={'mode': 'test'})
        a = shape('v2run/payload')

        store.save_page(make('v2run/snapshot'))
        snap = tmp / 'pages' / 'v2run' / 'snapshot.json'
        snap_run = json.loads(snap.read_text(encoding='utf-8'))['page']['dual_track_run']
        b = shape('v2run/snapshot')

        snap.unlink()
        con = sqlite3.connect(tmp / 't.sqlite3')
        con.execute("UPDATE pages SET ocr_payload='' WHERE page_id='v2run/snapshot'")
        con.commit()
        row = con.execute("SELECT dual_track_status,dual_track_evidence,dual_track_run "
                          "FROM pages WHERE page_id='v2run/snapshot'").fetchone()
        con.close()
        c = shape('v2run/snapshot')

        store.save_page(OCRResult(page_id='v2run/legacy', blocks=[]))
        legacy = shape('v2run/legacy')

    want = ['failed', None, True]
    ok = (a == want and b == want and c == want and snap_run is True
          and legacy == ['not_run', None, False] and not snap.exists())
    return {'ok': ok, 'payload': a, 'snapshot': b, 'snapshot_json_run': snap_run,
            'sqlite_columns_only': c, 'legacy': legacy,
            'sqlite_row': {'status': row[0], 'evidence_text': row[1],
                           'run': int(row[2])},
            'snapshot_removed': not snap.exists()}


# ------------------------------------------------------------------ 其余检查

def check_entry_fields(on: Dict[str, Any]) -> Dict[str, Any]:
    ocr = on['ocr']
    ev = ocr.get('dual_track_evidence')
    st = ocr.get('dual_track_status')
    problems = []
    if st not in ('not_run', 'completed', 'partial', 'failed', 'skipped'):
        problems.append(f'status={st}')
    if not isinstance(ev, list) or not ev:
        problems.append('p12 应有证据列表')
    need = ('group_id', 'region_id', 'side', 'bbox', 'basis', 'text', 'conf',
            'model', 'side_status')
    for e in ev or []:
        miss = [k for k in need if k not in e]
        if miss:
            problems.append(f'{e.get("evidence_id")}:{miss}')
    return {'ok': not problems, 'status': st, 'evidence': len(ev or []),
            'problems': problems[:5]}


def check_p12_sides(on: Dict[str, Any], cl: Dict[str, Any]) -> Dict[str, Any]:
    rows = {r['annotation_id']: r for r in cl['rows']}
    a = rows.get('丹方彚编/p12/DF12-03') or {}
    b = rows.get('丹方彚编/p12/DF12-11') or {}
    checks = []
    # 陽起石：两侧通过校验 → 分别复识
    bn = b.get('sides') or {}
    checks.append({'case': 'DF12-11 陽起石 双侧合格',
                   'ok': (bn.get('right', {}).get('side_status') == 'ok'
                          and bn.get('left', {}).get('side_status') == 'ok'
                          and bool(bn.get('right', {}).get('model'))
                          and bool(bn.get('left', {}).get('model'))
                          and bn.get('right', {}).get('crop_path')
                          and bn.get('left', {}).get('crop_path')),
                   'detail': {k: {'status': v.get('side_status'), 'text': v.get('text'),
                                  'conf': v.get('conf'), 'bbox': v.get('bbox')}
                              for k, v in bn.items()}})
    # 雲母：右轨冲突待核、左轨读数可追溯
    an = a.get('sides') or {}
    checks.append({'case': 'DF12-03 雲母 右轨待核',
                   'ok': (an.get('right', {}).get('side_status') == 'pending_validation'
                          and 'crosses_seam_in_final_rect'
                          in ' '.join(an.get('right', {}).get('pending_reasons') or [])
                          and an.get('left', {}).get('side_status') == 'ok'
                          and bool(an.get('left', {}).get('text'))),
                   'detail': {k: {'status': v.get('side_status'), 'text': v.get('text'),
                                  'reasons': v.get('pending_reasons')}
                              for k, v in an.items()}})
    return {'ok': all(c['ok'] for c in checks), 'checks': checks}


def check_evidence_geometry(on: Dict[str, Any]) -> Dict[str, Any]:
    ocr = on['ocr']
    W, H = ocr['width'], ocr['height']
    bad = []
    for e in ocr.get('dual_track_evidence') or []:
        b = e.get('bbox') or []
        if e['side_status'] == 'ok':
            if not (b and len(b) == 4 and 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H):
                bad.append(f'{e["evidence_id"]}:bbox')
            if not e.get('model'):
                bad.append(f'{e["evidence_id"]}:model')
        if e['evidence_id'] != f'{e["group_id"]}:{e["side"]}':
            bad.append(f'{e["evidence_id"]}:id')
        if e['side'] not in ('right', 'left'):
            bad.append(f'{e["evidence_id"]}:side')
        gb = e.get('group_bbox') or []
        if gb and b and not (gb[0] <= b[0] and gb[2] >= b[2]):
            bad.append(f'{e["evidence_id"]}:outside_group')
    # 右轨必须在左轨右侧（原图坐标下的固定侧别）
    by_group: Dict[str, Dict[str, Any]] = {}
    for e in ocr.get('dual_track_evidence') or []:
        by_group.setdefault(e['group_id'], {})[e['side']] = e
    order_bad = [g for g, v in by_group.items()
                 if v.get('right', {}).get('bbox') and v.get('left', {}).get('bbox')
                 and v['right']['bbox'][0] < v['left']['bbox'][0]]
    return {'ok': not bad and not order_bad, 'issues': bad[:6],
            'right_left_swapped': order_bad[:6],
            'sides': len(ocr.get('dual_track_evidence') or [])}


def _run_page(pid: str, img: str, tag: str) -> Dict[str, Any]:
    """跑一页并存成 run/<slug>_<tag>.json（已存在则复用，避免自检反复推理）。"""
    from ocr.pipeline import run_ocr
    path = RUN / f'{pid.replace("/", "__")}_{tag}.json'
    if path.exists():
        return _load(path)
    ocr = run_ocr(str(ROOT / img), pid)
    payload = {'tag': tag, 'ocr': ocr.to_dict(),
               'summary': {'page_id': pid, 'blocks': len(ocr.blocks),
                           'dual_track_status': ocr.dual_track_status,
                           'evidence': (None if ocr.dual_track_evidence is None
                                        else len(ocr.dual_track_evidence))}}
    RUN.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
                    encoding='utf-8')
    return payload


def check_negative_regions(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """负例区域 × 证据框：按类别分开统计（body_text/title 必须 0，single_track 另记）。"""
    neg = _load(ROOT / 'output' / 'dual_tracks' / 'stage2' / 'negative_regions.json')
    by_page = {p['ocr'].get('page_id'): (p['ocr'].get('dual_track_evidence') or [])
               for p in pages}
    rows = []
    for r in neg['regions']:
        ev = by_page.get(r['page_id'])
        if ev is None:
            continue
        hits = [{'evidence': e['evidence_id'],
                 'ratio': round(_overlap_ratio(e.get('bbox'), r['bbox']), 3)}
                for e in ev if _overlap_ratio(e.get('bbox'), r['bbox']) >= 0.3]
        rows.append({'negative_region_id': r['negative_region_id'],
                     'page_id': r['page_id'], 'category': r['category'],
                     'bbox': r['bbox'], 'hits': hits, 'n_hits': len(hits)})
    body_title = [x for x in rows if x['category'] in ('body_text', 'title')]
    single = [x for x in rows if x['category'] == 'single_track']
    other = [x for x in rows if x['category'] not in ('body_text', 'title', 'single_track')]
    return {
        'rows': rows,
        'body_title_clean': all(x['n_hits'] == 0 for x in body_title),
        'body_title': body_title,
        'single_track': single,
        'single_track_hits': sum(x['n_hits'] for x in single),
        'other': other,
        'other_hits': sum(x['n_hits'] for x in other),
    }


def check_negative_pages() -> Dict[str, Any]:
    """单列正文页走正常入口：completed + 空证据列表（不是 skipped/None）。"""
    out = []
    ok = True
    for pid, img in (('思濟堂方書/page_32', '古籍评测集/医家类/思濟堂方書/page_32.png'),
                     ('思濟堂方書/page_33', '古籍评测集/医家类/思濟堂方書/page_33.png')):
        p = _run_page(pid, img, 'neg')
        ocr = p['ocr']
        good = (ocr.get('dual_track_status') == 'completed'
                and ocr.get('dual_track_evidence') == []
                and len(ocr.get('blocks') or []) > 0)
        ok = ok and good
        out.append({'page_id': pid, 'status': ocr.get('dual_track_status'),
                    'evidence': ocr.get('dual_track_evidence'),
                    'blocks': len(ocr.get('blocks') or []), 'ok': good})
    return {'ok': ok, 'pages': out}


def check_api_surface() -> Dict[str, Any]:
    """现有 OCR 结果接口（GET /api/pages/{id}/ocr）三态返回一致，None 与 [] 不混。"""
    from app.config import CONFIG
    from app import store
    from app.schemas import OCRResult
    tmp = Path(tempfile.mkdtemp(prefix='stage7_api_'))
    patched = replace(CONFIG.paths, db_path=tmp / 't.sqlite3', store_dir=tmp)
    rows = []
    ok = True
    with mock.patch.object(store, 'PATHS', patched):
        import app.main as main_mod
        cases = [('hit', 'completed',
                  [{'evidence_id': 'api/hit#g1:right', 'group_id': 'api/hit#g1',
                    'side': 'right', 'bbox': [1, 2, 3, 4], 'text': '右',
                    'conf': 0.9, 'model': 'm.onnx', 'side_status': 'ok'}]),
                 ('empty', 'completed', []),
                 ('failed', 'failed', None)]
        for tag, st, ev in cases:
            pid = f'api/{tag}'
            store.save_page(OCRResult(page_id=pid, image_path='x.png', width=100,
                                      height=100, blocks=[], engine='rapidocr',
                                      model_loaded=True, is_demo=False,
                                      dual_track_status=st, dual_track_evidence=ev,
                                      dual_track_run=True),
                            extraction_meta={'mode': 'rules'})
            d = main_mod.get_ocr(pid)
            got = d.get('dual_track_evidence')
            shape = None if got is None else ([] if not got else f'list{len(got)}')
            good = (d.get('dual_track_status') == st
                    and shape == (None if ev is None else ([] if not ev else f'list{len(ev)}'))
                    and not d.get('dual_track_problems'))
            ok = ok and good
            rows.append({'case': tag, 'status': d.get('dual_track_status'),
                         'evidence': shape, 'problems': d.get('dual_track_problems'),
                         'ok': good})
    return {'ok': ok, 'rows': rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage7 自检')
    ap.add_argument('--out', default=str(S7 / 'validation_report.json'))
    args = ap.parse_args(argv)
    on = _load(RUN / '丹方彚编__page_12_on.json')
    off_path = RUN / '丹方彚编__page_12_off.json'
    cmp_path = RUN / 'compare_on_off.json'
    cl_path = RUN / 'checklist_丹方彚编__page_12.json'
    replay_path = S7 / 'replay_compare.json'

    checks: List[Dict[str, Any]] = []

    def add(cid: str, title: str, ok: bool, detail: Any) -> None:
        checks.append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                       'detail': detail})

    # S01 入口字段
    r = check_entry_fields(on)
    add('S01', '正常 OCR 入口产出 dual_track_status / dual_track_evidence', r['ok'], r)

    # S02 冻结回放一致
    if replay_path.exists():
        rep = _load(replay_path)
        ok = (rep['first_diff_step'] is None
              and rep['S1_region']['equal'] and rep['S2_pairing']['equal']
              and rep['S3_refine']['equal'])
        add('S02', '在线框选与冻结回放逐步一致', ok,
            {'first_diff_step': rep['first_diff_step'],
             'S1': rep['S1_region']['live_regions'],
             'S2': rep['S2_pairing']['live_groups'],
             'S3': rep['S3_refine']['groups_compared'],
             'stage5_v4': rep['S2_pairing']['stage5_v4_groups'],
             'stage6_v4': rep['S3_refine']['stage6_v4_groups']})
    else:
        add('S02', '在线框选与冻结回放逐步一致', False, '缺 replay_compare.json')

    # S03 / S04 p12 两侧
    cl = _load(cl_path)
    r = check_p12_sides(on, cl)
    add('S03', 'p12 双侧合格组两侧分别复识（陽起石）', r['checks'][0]['ok'],
        r['checks'][0]['detail'])
    add('S04', 'p12 右轨冲突明确待核、合格侧读数可追溯（雲母）', r['checks'][1]['ok'],
        r['checks'][1]['detail'])

    # S05 正文隔离
    if cmp_path.exists():
        cmp = _load(cmp_path)
        ok = (cmp['blocks']['identical'] and cmp['texts']['identical']
              and cmp['bboxes']['identical'] and cmp['reading_order']['identical']
              and cmp['tables']['identical'] and cmp['layout']['identical'])
        add('S05', '开关前后整页块/文本/坐标/阅读顺序/表格/版面一致', ok,
            {'blocks': f"{cmp['blocks']['on']}/{cmp['blocks']['off']}",
             'block_sig': cmp['block_sig'], 'reading_order': cmp['reading_order']['identical'],
             'tables': cmp['tables'], 'layout': cmp['layout'], 'verdict': cmp['verdict']})
    else:
        add('S05', '开关前后整页块/文本/坐标/阅读顺序/表格/版面一致', False,
            '缺 compare_on_off.json')

    # S06 负例：区域级（单列正文 / 标题）+ 页级（单列正文页）
    neg_pages = [on] + [_run_page('思濟堂方書/page_30',
                                  '古籍评测集/医家类/思濟堂方書/page_30.png', 'neg'),
                        _run_page('思濟堂方書/page_44',
                                  '古籍评测集/医家类/思濟堂方書/page_44.png', 'neg')]
    r = check_negative_regions(neg_pages)
    add('S06a', '单列正文 / 标题负例区域内无证据框', r['body_title_clean'],
        {'body_title': r['body_title'], 'other': r['other'],
         'pages': [p['ocr'].get('page_id') for p in neg_pages]})
    r2 = check_negative_pages()
    add('S06b', '单列正文页正常完成且无双轨证据（evidence=[]）', r2['ok'], r2)
    add('S06c', '单条小字列（single_track）负例区域内无证据框', r['single_track_hits'] == 0,
        {'single_track': r['single_track'], 'hits': r['single_track_hits'],
         'note': '此为检出侧（阶段5/6 定位）缺陷，非阶段7 接入缺陷；如实记未通过'})

    # S07 状态往返
    r = check_state_roundtrip()
    add('S07', '七类状态保存读回后 None 与 [] 语义不变', r['ok'],
        {'cases': r['cases'], 'rows': r['rows']})

    # S08 侧别可追溯
    r = check_evidence_geometry(on)
    add('S08', '证据侧别 / 原图框 / 模型可追溯且右轨恒在左轨右侧', r['ok'], r)

    # S09 单侧失败隔离
    r = check_side_isolation()
    add('S09', '单侧复识失败时另一侧证据仍可读', r['ok'], r)

    # S10 双轨整体失败隔离
    r = check_run_isolation()
    add('S10', '双轨整体异常时整页 OCR 结果照常返回', r['ok'], r)

    # S11 现有 OCR 结果接口
    r = check_api_surface()
    add('S11', 'GET /api/pages/{id}/ocr 三态字段一致（None 与 [] 不混）', r['ok'], r)

    # S12 状态按实际结果（stage7-v2）
    r = check_status_semantics()
    add('S12', '全部精修失败→failed+None，单侧复识失败→partial+成功侧证据', r['ok'], r)

    # S13 单侧坐标边界（stage7-v2）
    r = check_bbox_bounds()
    add('S13', '80px 宽图 x2=80 合法 / x2=81 与非数字拒绝，且只废该侧', r['ok'], r)

    # S14 「是否尝试运行」持久化（stage7-v2）
    r = check_run_flag_storage()
    add('S14', 'failed+None+run=true 在负载 / 快照 / 仅列三种路径一致', r['ok'], r)

    passed = sum(1 for c in checks if c['status'] == 'pass')
    report = {'stage': 'stage7_dual_track_ocr_entry',
              'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
              'checks': checks, 'passed': passed, 'total': len(checks),
              'verdict': '全部通过' if passed == len(checks) else f'{len(checks) - passed} 项未通过'}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    for c in checks:
        print(f"{c['id']:5s} {c['status']:4s} {c['title']}")
    print(f"== {passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
