# -*- coding: utf-8 -*-
"""阶段7 回归：OCRResult 双轨字段 / 状态往返 / 按侧判定 / 入口隔离。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402

from app.schemas import OCRResult  # noqa: E402
from ocr.dual_track import build_evidence, run_dual_track  # noqa: E402


def _record(ok_right=True, ok_left=True, gid='synthetic/page#g001') -> dict:
    """按 stage6 refine_group 的返回结构造一条记录（两侧都有轨框）。"""
    def side(bbox, ok, reason=None):
        return {'bbox_before': bbox, 'bbox_after': bbox,
                'edge': {'x1': bbox[0], 'x2': bbox[2], 'notes': []},
                'supplement': {'head': {'accepted': []}, 'tail': {'accepted': []}},
                'accepted_glyphs': [[bbox[0], bbox[1], bbox[0] + 8, bbox[1] + 10]],
                'excluded_glyphs': [], 'flags': [],
                'pending_reasons': [] if ok else [reason or 'crosses_seam_in_final_rect'],
                'horizontal_low_confidence': False, 'supplement_limited': False,
                'final_validation': {'ok': ok, 'reason': None if ok else (reason or
                                                                          'crosses_seam_in_final_rect'),
                                     'rolled_back': [],
                                     'fallback_to_tentative': not ok,
                                     'status': 'accepted' if ok else 'rejected_to_tentative'}}
    return {'group_id': gid, 'page_id': 'synthetic/page', 'region_id': 'synthetic/page#r1',
            'region_source': 'halfpge_fallback', 'low_confidence': True,
            'stage5_pending': False,
            'before': {'right_bbox': [32, 52, 44, 108], 'left_bbox': [18, 52, 30, 108],
                       'group_bbox': [18, 52, 44, 108]},
            'after': {'right_bbox': [32, 52, 44, 108], 'left_bbox': [18, 52, 30, 108],
                      'group_bbox': [18, 52, 44, 108]},
            'sides': {'right': side([32, 52, 44, 108], ok_right),
                      'left': side([18, 52, 30, 108], ok_left)},
            'flags': [], 'pending_reasons': [], 'pending': not (ok_right and ok_left),
            'changed': False, 'confirmed_span': [52, 108],
            'scale_used': {'H': 20.0, 'G': 4.0, 'glyph_pitch': 16.0}}


def _image(w=80, h=120) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def _ok_read(text='陽起石'):
    def fn(_img, bbox, zoom):
        return text, 0.88, {'clip': [int(v) for v in bbox], 'rotation': 90, 'zoom': zoom}
    return fn


class SchemaStateTests(unittest.TestCase):
    def test_default_is_not_run_without_evidence(self):
        r = OCRResult(page_id='p')
        self.assertEqual(r.dual_track_status, 'not_run')
        self.assertIsNone(r.dual_track_evidence)
        self.assertFalse(r.dual_track_run)
        self.assertEqual(r.validate_dual_track(), [])

    def test_old_record_without_fields_reads_as_not_run(self):
        r = OCRResult.from_dict({'page_id': 'p', 'blocks': [], 'notes': ['old']})
        self.assertEqual((r.dual_track_status, r.dual_track_evidence, r.dual_track_run),
                         ('not_run', None, False))

    def test_empty_list_is_not_none(self):
        r = OCRResult.from_dict({'page_id': 'p', 'dual_track_status': 'completed',
                                 'dual_track_evidence': [], 'dual_track_run': True})
        self.assertEqual(r.dual_track_evidence, [])
        rt = OCRResult.from_dict(json.loads(r.to_json()))
        self.assertEqual(rt.dual_track_evidence, [])

    def test_validate_rejects_status_evidence_mismatch(self):
        bad = OCRResult(page_id='p', dual_track_status='completed',
                        dual_track_evidence=None)
        self.assertTrue(any('evidence 必须为列表' in p
                            for p in bad.validate_dual_track()))
        bad2 = OCRResult(page_id='p', dual_track_status='failed',
                         dual_track_evidence=[{'a': 1}])
        self.assertTrue(any('必须为 None' in p for p in bad2.validate_dual_track()))
        bad3 = OCRResult(page_id='p', dual_track_status='bogus')
        self.assertTrue(bad3.validate_dual_track())

    def test_validate_wired_into_ocrresult_validate(self):
        r = OCRResult(page_id='p', dual_track_status='partial', dual_track_evidence=[])
        self.assertEqual([p for p in r.validate() if p.startswith('dual_track')], [])


class StoreRoundTripTests(unittest.TestCase):
    """七类状态：save_page → get_page → from_dict，None 与 [] 不得互相吞并。"""

    CASES = [
        ('old', {'page_id': 'state/old', 'blocks': []}, ('not_run', None, False)),
        ('not_run', {'dual_track_status': 'not_run', 'dual_track_evidence': None,
                     'dual_track_run': False}, ('not_run', None, False)),
        ('skipped', {'dual_track_status': 'skipped', 'dual_track_evidence': None,
                     'dual_track_run': False}, ('skipped', None, False)),
        ('completed_empty', {'dual_track_status': 'completed', 'dual_track_evidence': [],
                             'dual_track_run': True}, ('completed', [], True)),
        ('completed_hit', {'dual_track_status': 'completed', 'dual_track_run': True,
                           'dual_track_evidence': [
                               {'evidence_id': 'g:right', 'group_id': 'g', 'side': 'right',
                                'bbox': [1, 2, 3, 4], 'text': '右', 'conf': 0.9,
                                'model': 'm.onnx', 'side_status': 'ok'}]},
         ('completed', 1, True)),
        ('partial', {'dual_track_status': 'partial', 'dual_track_run': True,
                     'dual_track_evidence': [
                         {'evidence_id': 'g:left', 'group_id': 'g', 'side': 'left',
                          'bbox': [1, 2, 3, 4], 'text': '左', 'conf': 0.5,
                          'model': 'm.onnx', 'side_status': 'ok'}]},
         ('partial', 1, True)),
        ('failed', {'dual_track_status': 'failed', 'dual_track_evidence': None,
                    'dual_track_run': True}, ('failed', None, True)),
    ]

    @staticmethod
    def _shape(r: OCRResult):
        """None（没跑）/ []（跑了没检出）/ 条数（有结果）三态必须分得开。"""
        ev = r.dual_track_evidence
        if ev is None:
            sig = None
        elif not ev:
            sig = []
        else:
            sig = len(ev)
        return r.dual_track_status, sig, r.dual_track_run

    def test_roundtrip_all_states(self):
        from app import store
        from app.config import CONFIG
        tmp = Path(tempfile.mkdtemp(prefix='stage7_store_'))
        patched = replace(CONFIG.paths, db_path=tmp / 't.sqlite3', store_dir=tmp)
        with mock.patch.object(store, 'PATHS', patched):
            for name, fields, want in self.CASES:
                with self.subTest(case=name):
                    r = OCRResult.from_dict(dict(fields, page_id=f'state/{name}'))
                    self.assertEqual(self._shape(r), want)
                    store.save_page(r, extraction_meta={'mode': 'test'})
                    back = OCRResult.from_dict(store.get_page(r.page_id)['page'])
                    self.assertEqual(self._shape(back), want)
                    self.assertEqual(back.validate_dual_track(), [])
                    # 只是列（无 payload）的旧路径也要保留 None / [] 的区别
                    r2 = OCRResult.from_dict(dict(fields, page_id=f'col/{name}'))
                    store.save_page(r2)
                    back2 = OCRResult.from_dict(store.get_page(r2.page_id)['page'])
                    self.assertEqual(self._shape(back2), want)


class EvidenceBuildTests(unittest.TestCase):
    def test_both_sides_ok_are_recognized_with_fixed_side(self):
        out = build_evidence([_record()], image=_image(), page_id='synthetic/page',
                             width=80, height=120, recognizer=_ok_read('甲乙'))
        ev = out['evidence']
        self.assertEqual([e['side'] for e in ev], ['right', 'left'])
        self.assertTrue(all(e['side_status'] == 'ok' for e in ev))
        self.assertEqual([e['text'] for e in ev], ['甲乙', '甲乙'])
        self.assertEqual(ev[0]['bbox'], [32, 52, 44, 108])
        self.assertEqual(ev[1]['bbox'], [18, 52, 30, 108])
        self.assertEqual(out['side_counts'], {'ok': 2})
        self.assertTrue(ev[0]['basis']['lane'] == 'right')

    def test_pending_side_is_not_sent_to_recognition(self):
        calls = []

        def fn(_img, bbox, zoom):
            calls.append(list(bbox))
            return 'x', 0.9, {'clip': list(bbox)}

        out = build_evidence([_record(ok_right=False)], image=_image(),
                             page_id='synthetic/page', width=80, height=120,
                             recognizer=fn)
        right, left = out['evidence']
        self.assertEqual(right['side_status'], 'pending_validation')
        self.assertEqual(right['text'], '')
        self.assertIsNone(right['conf'])
        self.assertEqual(right['bbox_source'], 'tentative')
        self.assertEqual(left['side_status'], 'ok')
        self.assertEqual(calls, [[18, 52, 30, 108]])
        self.assertTrue(any(a['kind'] == 'side_pending' for a in out['audit']))

    def test_invalid_bbox_is_pending_not_recognized(self):
        rec = _record()
        rec['sides']['left']['bbox_after'] = [30, 52, 18, 108]     # 反向外框
        out = build_evidence([rec], image=_image(), page_id='synthetic/page',
                             width=80, height=120, recognizer=_ok_read())
        left = next(e for e in out['evidence'] if e['side'] == 'left')
        self.assertEqual(left['side_status'], 'invalid_bbox')
        self.assertEqual(left['text'], '')
        self.assertTrue(any(a['kind'] == 'side_invalid_bbox' for a in out['audit']))

    def test_one_side_error_keeps_other_side_readable(self):
        def fn(_img, bbox, zoom):
            if bbox[0] >= 30:
                raise RuntimeError('boom')
            return '左读', 0.4, {'clip': list(bbox)}

        out = build_evidence([_record()], image=_image(), page_id='synthetic/page',
                             width=80, height=120, recognizer=fn)
        right = next(e for e in out['evidence'] if e['side'] == 'right')
        left = next(e for e in out['evidence'] if e['side'] == 'left')
        self.assertEqual(right['side_status'], 'recognize_error')
        self.assertEqual(left['side_status'], 'ok')
        self.assertEqual(left['text'], '左读')
        self.assertEqual(out['side_counts']['recognize_error'], 1)

    def test_empty_reading_is_ok_but_audited(self):
        out = build_evidence([_record()], image=_image(), page_id='synthetic/page',
                             width=80, height=120,
                             recognizer=lambda i, b, z: ('', 0.0, {'clip': list(b)}))
        self.assertTrue(all(e['side_status'] == 'ok' for e in out['evidence']))
        self.assertEqual(sum(1 for a in out['audit'] if a['kind'] == 'empty_reading'), 2)

    def test_missing_side_recorded(self):
        rec = _record()
        rec['sides'].pop('left')
        out = build_evidence([rec], image=_image(), page_id='synthetic/page',
                             width=80, height=120, recognizer=_ok_read())
        left = next(e for e in out['evidence'] if e['side'] == 'left')
        self.assertEqual(left['side_status'], 'missing_side')
        self.assertIsNone(left['bbox'])


class EntryTests(unittest.TestCase):
    def test_missing_image_is_skipped_not_failed(self):
        out = run_dual_track(str(ROOT / 'no_such_page.png'), 'x/page_1')
        self.assertEqual(out['status'], 'skipped')
        self.assertIsNone(out['evidence'])
        self.assertFalse(out['attempted'])

    def test_mode_off_is_not_run(self):
        out = run_dual_track('', 'x/page_1', mode='off')
        self.assertEqual(out['status'], 'not_run')
        self.assertIsNone(out['evidence'])

    def test_dual_failure_does_not_break_page_ocr(self):
        import ocr.pipeline as pipeline
        with mock.patch.object(pipeline, 'run_dual_track',
                               side_effect=RuntimeError('synthetic')):
            ocr = pipeline.run_ocr(str(ROOT / '古籍评测集/医家类/思濟堂方書/page_32.png'),
                                   'test/isolation')
        self.assertEqual(ocr.dual_track_status, 'failed')
        self.assertIsNone(ocr.dual_track_evidence)
        self.assertGreater(len(ocr.blocks), 0)
        self.assertEqual(ocr.validate_dual_track(), [])

    def test_stub_page_skips_dual_track(self):
        out = run_dual_track(str(ROOT / '古籍评测集/医家类/思濟堂方書/page_32.png'),
                             'x/page_32', engine='stub', is_demo=True)
        self.assertEqual(out['status'], 'skipped')


class FrozenDualSideTests(unittest.TestCase):
    """冻结的双侧合格案例（p12 陽起石 g036）：裁图 / 文本 / 序列化侧别必须始终对应。"""

    S6 = ROOT / 'output' / 'dual_tracks' / 'stage6' / 'groups.json'
    IMG = ROOT / '古籍评测集/医家类/丹方彚编/page_12.png'

    def setUp(self):
        if not (self.S6.exists() and self.IMG.exists()):
            self.skipTest('缺 stage6 冻结产物或 p12 原图')
        payload = json.loads(self.S6.read_text(encoding='utf-8'))
        page = next(p for p in payload['pages'] if p['page_id'] == '丹方彚编/page_12')
        self.rec = next(r for r in page['records']
                        if r['group_id'] == '丹方彚编/page_12#g036')
        for side in ('right', 'left'):
            self.assertTrue(self.rec['sides'][side]['final_validation']['ok'],
                            f'{side} 应为合格侧（冻结案例前提）')

    def test_side_bbox_text_and_serialization_match(self):
        from ocr.vertical import imread_unicode
        img = imread_unicode(str(self.IMG))
        tmp = Path(tempfile.mkdtemp(prefix='stage7_crop_'))

        def fn(_img, bbox, zoom):
            # 文本里写死侧别与 x1，便于反查「哪一侧用了哪个框」；裁图与真实链路同构
            tag = 'RIGHT' if bbox[0] >= 880 else 'LEFT'
            x0, y0, x1, y1 = (int(v) for v in bbox)
            rot = np.ascontiguousarray(np.rot90(_img[y0:y1, x0:x1]))
            return (f'{tag}@{bbox[0]}', 0.7,
                    {'clip': [x0, y0, x1, y1], 'rotation': 90, 'zoom': zoom,
                     'crop_png': rot})

        rec = dict(self.rec)
        rec['region_bbox'] = [0, 0, 1024, 1024]
        out = build_evidence([rec], image=img, page_id='丹方彚编/page_12',
                             width=1024, height=1024, recognizer=fn, crop_dir=tmp)
        ev = {e['side']: e for e in out['evidence']}
        right_box = self.rec['sides']['right']['bbox_after']
        left_box = self.rec['sides']['left']['bbox_after']
        self.assertEqual(ev['right']['bbox'], [int(v) for v in right_box])
        self.assertEqual(ev['left']['bbox'], [int(v) for v in left_box])
        self.assertTrue(ev['right']['text'].startswith('RIGHT@'))
        self.assertTrue(ev['left']['text'].startswith('LEFT@'))
        self.assertGreater(right_box[0], left_box[0])          # 固定侧别：右轨在右
        # 裁图文件名带侧别，且尺寸=旋转后的 (宽=原框高, 高=原框宽)
        for side, box, tag in (('right', right_box, 'RIGHT'), ('left', left_box, 'LEFT')):
            p = Path(ev[side]['crop_path'])
            self.assertTrue(p.is_file(), p)
            self.assertTrue(p.name.endswith(f'_{side}.png'), p.name)
            im = np.fromfile(str(p), dtype=np.uint8)
            import cv2
            arr = cv2.imdecode(im, cv2.IMREAD_UNCHANGED)
            self.assertEqual(arr.shape[0], box[2] - box[0])   # 旋转后高 = 原框宽
            self.assertEqual(arr.shape[1], box[3] - box[1])   # 旋转后宽 = 原框高
        # 序列化后侧别与框仍一一对应（列表顺序不参与判断）
        back = OCRResult.from_dict(json.loads(json.dumps(
            {'page_id': 'p', 'dual_track_status': 'completed',
             'dual_track_evidence': out['evidence'], 'dual_track_run': True})))
        pairs = {(e['side'], tuple(e['bbox']), e['text']) for e in back.dual_track_evidence}
        self.assertEqual(pairs, {('right', tuple(int(v) for v in right_box),
                                  ev['right']['text']),
                                 ('left', tuple(int(v) for v in left_box),
                                  ev['left']['text'])})


class AuditTests(unittest.TestCase):
    """待核 / 空读数 / 异常 / 跳过原因必须能被读回来（审计不是只在内存里）。"""

    def test_audit_roundtrip_and_validate(self):
        from app import store
        from app.config import CONFIG
        tmp = Path(tempfile.mkdtemp(prefix='stage7_audit_'))
        patched = replace(CONFIG.paths, db_path=tmp / 't.sqlite3', store_dir=tmp)
        audit = [{'kind': 'side_pending', 'group_id': 'g1', 'side': 'right',
                  'reason': 'crosses_seam_in_final_rect'},
                 {'kind': 'empty_reading', 'group_id': 'g1', 'side': 'left'}]
        with mock.patch.object(store, 'PATHS', patched):
            for pid, meta in (('audit/payload', {'mode': 'rules'}), ('audit/column', None)):
                r = OCRResult(page_id=pid, dual_track_status='completed',
                              dual_track_evidence=[{'evidence_id': 'g1:left',
                                                    'group_id': 'g1', 'side': 'left',
                                                    'bbox': [1, 2, 3, 4], 'text': '',
                                                    'side_status': 'ok'}],
                              dual_track_audit=audit, dual_track_run=True)
                store.save_page(r, extraction_meta=meta)
                back = OCRResult.from_dict(store.get_page(pid)['page'])
                self.assertEqual(back.dual_track_audit, audit, pid)
                self.assertEqual(back.validate_dual_track(), [])

    def test_audit_none_for_old_record(self):
        r = OCRResult.from_dict({'page_id': 'p'})
        self.assertIsNone(r.dual_track_audit)
        self.assertEqual(r.validate_dual_track(), [])

    def test_audit_must_be_list_or_none(self):
        r = OCRResult(page_id='p', dual_track_status='completed',
                      dual_track_evidence=[], dual_track_audit={'a': 1})
        self.assertTrue(any('audit 必须' in p for p in r.validate_dual_track()))


class BookReferenceTests(unittest.TestCase):
    """书级列宽参考必须能从图片路径拿到，否则生产 page_id（时间戳）会静默丢掉参考。"""

    def test_book_from_page_id_or_image_path(self):
        from ocr.dual_track import _book_of
        self.assertEqual(_book_of('丹方彚编/page_12', 'x/y.png'), '丹方彚编')
        self.assertEqual(_book_of('P20261125193000-ab12cd',
                                  '古籍评测集/医家类/丹方彚编/page_12.png'), '丹方彚编')
        self.assertEqual(_book_of('', 'a.png'), '')


class StatusByResultTests(unittest.TestCase):
    """stage7-v2：状态必须由实际运行结果决定，不能按「有没有异常」猜。"""

    IMG = ROOT / '古籍评测集/医家类/丹方彚编/page_12.png'

    def test_all_refine_failures_give_failed_none_run_true(self):
        import ocr.pipeline as pipeline
        import dual_track_refine
        with mock.patch.object(dual_track_refine, 'refine_group',
                               side_effect=RuntimeError('synthetic_refine_failure')):
            ocr = pipeline.run_ocr(str(self.IMG), 'v2/all_refine_fail')
        self.assertEqual(ocr.dual_track_status, 'failed')
        self.assertIsNone(ocr.dual_track_evidence)
        self.assertTrue(ocr.dual_track_run)
        self.assertGreater(len(ocr.blocks), 0)                 # 正文照常返回
        self.assertEqual(ocr.validate_dual_track(), [])
        failed = [a for a in (ocr.dual_track_audit or [])
                  if a.get('kind') == 'group_failed']
        self.assertGreater(len(failed), 0)
        self.assertTrue(all('group_id' in a for a in failed))  # 失败定位保留

    def test_single_side_recognize_error_gives_partial_with_success_sides(self):
        import ocr.dual_track as dt
        import ocr.pipeline as pipeline
        original = dt._crop_read
        calls = {'n': 0}

        def flaky(image, bbox, zoom=1.0, min_conf=0.15):
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('synthetic_side_failure')
            return original(image, bbox, zoom)

        with mock.patch.object(dt, '_crop_read', flaky):
            ocr = pipeline.run_ocr(str(self.IMG), 'v2/side_fail')
        self.assertEqual(ocr.dual_track_status, 'partial')
        self.assertTrue(ocr.dual_track_run)
        self.assertGreater(len(ocr.blocks), 0)
        self.assertEqual(ocr.validate_dual_track(), [])
        ev = ocr.dual_track_evidence or []
        self.assertTrue(ev)
        err = [e for e in ev if e['side_status'] == 'recognize_error']
        self.assertEqual(len(err), 1)
        self.assertTrue(any(e['side_status'] == 'ok' and e['text'] for e in ev))
        self.assertTrue(any(a.get('kind') == 'recognize_error'
                            for a in (ocr.dual_track_audit or [])))


class BBoxBoundsTests(unittest.TestCase):
    """单侧坐标：半开区间上界必须严格，坏坐标只废该侧。"""

    def test_x2_equal_width_is_legal(self):
        rec = _record()
        rec['sides']['right']['bbox_after'] = [40, 52, 80, 108]
        out = build_evidence([rec], image=_image(80, 120), page_id='s/p',
                             width=80, height=120, recognizer=_ok_read())
        right = next(e for e in out['evidence'] if e['side'] == 'right')
        self.assertEqual(right['side_status'], 'ok')
        self.assertEqual(right['bbox'], [40, 52, 80, 108])

    def test_x2_beyond_width_is_rejected_and_other_side_still_reads(self):
        rec = _record()
        rec['sides']['right']['bbox_after'] = [40, 52, 81, 108]
        called = []

        def fn(_img, bbox, zoom):
            called.append(list(bbox))
            return '左读', 0.6, {'clip': list(bbox)}

        out = build_evidence([rec], image=_image(80, 120), page_id='s/p',
                             width=80, height=120, recognizer=fn)
        right = next(e for e in out['evidence'] if e['side'] == 'right')
        left = next(e for e in out['evidence'] if e['side'] == 'left')
        self.assertEqual(right['side_status'], 'invalid_bbox')
        self.assertEqual(right['status_reason'], 'beyond_width')
        self.assertEqual(right['text'], '')
        self.assertEqual(left['side_status'], 'ok')
        self.assertEqual(left['text'], '左读')
        self.assertEqual(called, [[18, 52, 30, 108]])        # 坏侧没进裁图
        self.assertTrue(any(a.get('kind') == 'side_invalid_bbox' and a.get('side') == 'right'
                            for a in out['audit']))

    def test_non_numeric_bbox_rejects_only_that_side(self):
        rec = _record()
        rec['sides']['left']['bbox_after'] = ['a', None, 'c', 'd']
        out = build_evidence([rec], image=_image(80, 120), page_id='s/p',
                             width=80, height=120, recognizer=_ok_read('右读'))
        right = next(e for e in out['evidence'] if e['side'] == 'right')
        left = next(e for e in out['evidence'] if e['side'] == 'left')
        self.assertEqual(right['side_status'], 'ok')
        self.assertEqual(right['text'], '右读')
        self.assertEqual(left['side_status'], 'invalid_bbox')
        self.assertEqual(left['status_reason'], 'not_numeric')
        self.assertIsNone(left['bbox'])
        self.assertTrue(any(a.get('kind') == 'side_invalid_bbox'
                            and a.get('reason') == 'not_numeric' for a in out['audit']))


class RunFlagStorageTests(unittest.TestCase):
    """failed + None + run=true 必须在三种存储形态下都原样读回。"""

    def test_failed_run_true_roundtrip_three_paths(self):
        import sqlite3
        from app import store
        from app.config import CONFIG
        tmp = Path(tempfile.mkdtemp(prefix='stage7_runflag_'))
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
            return r.dual_track_status, r.dual_track_evidence, r.dual_track_run

        with mock.patch.object(store, 'PATHS', patched):
            # A 完整负载（ocr_payload）
            store.save_page(make('runflag/payload'), extraction_meta={'mode': 'test'})
            self.assertEqual(shape('runflag/payload'), ('failed', None, True))

            # B 只有 JSON 快照（页列无 extraction_meta → get_page 走快照）
            store.save_page(make('runflag/snapshot'))
            snap = tmp / 'pages' / 'runflag' / 'snapshot.json'
            self.assertTrue(snap.is_file())
            self.assertIs(OCRResult.from_dict(
                json.loads(snap.read_text(encoding='utf-8'))['page']).dual_track_run, True)
            self.assertEqual(shape('runflag/snapshot'), ('failed', None, True))

            # C 只有 SQLite 页面记录：实际删掉快照，并清空 payload
            snap.unlink()
            con = sqlite3.connect(tmp / 't.sqlite3')
            con.execute("UPDATE pages SET ocr_payload='' WHERE page_id='runflag/snapshot'")
            con.commit()
            con.close()
            self.assertFalse(snap.exists())
            self.assertEqual(shape('runflag/snapshot'), ('failed', None, True))

            # 旧记录（缺字段）仍是 not_run / None / run=false
            store.save_page(OCRResult(page_id='runflag/legacy', blocks=[]))
            self.assertEqual(shape('runflag/legacy'), ('not_run', None, False))


if __name__ == '__main__':
    unittest.main()



