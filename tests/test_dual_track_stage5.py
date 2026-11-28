# -*- coding: utf-8 -*-
"""第五阶段测试：合成双轨/单轨、去重、断点枚举、真实产物与读取入口。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np                                   # noqa: E402

from dual_track_track import (DEFAULT_CONFIG, audit_for_page,  # noqa: E402
                              groups_for_page, windows_for_page)
from dual_track_track.dedupe import dedupe_groups                   # noqa: E402
from dual_track_track.detect import Win, _boxes_in_lane, detect_region  # noqa: E402
from dual_track_track.inputs import S5, canon_sha256                # noqa: E402
from dual_track_track.seam import SeamEval                           # noqa: E402

S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def two_track_image(h=420, w=120, gap=2, H=20):
    """两列小字（字高 0.6H、宽 10px），中间 gap 像素缝；H 是正文字高。"""
    ink = np.zeros((h, w), np.uint8)
    gh = int(round(0.6 * H))
    left_x, right_x = 20, 20 + 10 + gap
    for y in range(40, h - 40, gh + 4):
        ink[y:y + gh, left_x:left_x + 10] = 1
        ink[y:y + gh, right_x:right_x + 10] = 1
    return ink


def single_track_image(h=420, w=120, H=20):
    ink = np.zeros((h, w), np.uint8)
    gh = int(round(0.6 * H))
    for y in range(40, h - 40, gh + 4):
        ink[y:y + gh, 20:30] = 1
    return ink


def scale(H=20.0):
    return {'H': H, 'G': 4.0, 'glyph_pitch': H, 'value_origin': 'measured',
            'confidence_status': 'confident', 'dispersion': {'H_iqr': 1.0}}


class SyntheticTests(unittest.TestCase):
    def _run(self, ink, bbox):
        region = {'region_id': 'synthetic#r001', 'bbox': bbox,
                  'column_range_source': 'validated_columns', 'low_confidence': False}
        return detect_region('synthetic/page_1', region, ink, [], scale(),
                             DEFAULT_CONFIG)

    def test_two_tracks_confirmed(self):
        ink = two_track_image()
        groups, audit = self._run(ink, [10, 20, 90, 400])
        self.assertGreaterEqual(len(groups), 1, audit.get('skipped'))
        g = groups[0]
        self.assertTrue(g['right_bbox'] and g['left_bbox'])
        self.assertLess(g['left_bbox'][2], g['right_bbox'][0] + 1)
        self.assertTrue(g['stop_events'])
        self.assertEqual(g['state'], 'completed')
        self.assertIn('small_glyph_height', g['small_scale'])

    def test_single_track_rejected(self):
        ink = single_track_image()
        groups, audit = self._run(ink, [10, 20, 90, 400])
        self.assertEqual(groups, [])
        self.assertTrue(audit.get('skipped') or not audit.get('groups'))

    def test_no_scale_skips_region(self):
        ink = two_track_image()
        region = {'region_id': 'synthetic#r002', 'bbox': [10, 20, 90, 400],
                  'column_range_source': 'halfpage_fallback', 'low_confidence': True}
        groups, audit = detect_region('synthetic/page_1', region, ink, [],
                                      {'H': None}, DEFAULT_CONFIG)
        self.assertEqual(groups, [])
        self.assertEqual(audit['skipped'], 'no_usable_scale')


class ConfirmAndExtensionTests(unittest.TestCase):
    """三项验收回归：唯一字形、严格延伸、候选窗口日志。"""

    def _run(self, ink, bbox=(0, 0, 80, 140), rule_segments=()):
        region = {'region_id': 'synthetic#audit#001', 'bbox': list(bbox),
                  'column_range_source': 'validated_columns', 'low_confidence': False}
        return detect_region('synthetic/page_1', region, ink, rule_segments,
                             {'H': 20.0, 'G': 4.0, 'glyph_pitch': 20.0,
                              'value_origin': 'measured', 'dispersion': {'H_iqr': 1.0}},
                             DEFAULT_CONFIG)

    @staticmethod
    def _block(ink, y, x0, x1, h=8):
        ink[y:y + h, x0:x1] = 1

    def test_one_glyph_per_side_is_not_counted_three_times(self):
        ink = np.zeros((70, 80), np.uint8)
        # 唯一一个字形/侧，却落在 3+ 个重叠窗里（v2 会把每侧计成 3）
        self._block(ink, 18, 20, 30)
        self._block(ink, 18, 32, 42)
        groups, audit = self._run(ink, (0, 0, 80, 70))
        self.assertEqual(groups, [])
        self.assertTrue(any(r.get('rejected') == 'insufficient_unique_glyphs'
                            for r in audit['candidate_runs']), audit['candidate_runs'])

    def test_three_independent_glyphs_each_side_confirms(self):
        ink = np.zeros((70, 80), np.uint8)
        for y in (18, 34, 50):
            self._block(ink, y, 20, 30)
            self._block(ink, y, 32, 42)
        groups, audit = self._run(ink, (0, 0, 80, 70))
        self.assertGreaterEqual(len(groups), 1, audit.get('skipped'))
        g = groups[0]
        self.assertEqual(g['state'], 'completed')
        for side in ('right', 'left'):
            self.assertGreaterEqual(g['glyph_support'][side]['n_unique_confirm'], 3)

    def test_independent_stop_positions_and_rule_break(self):
        ink = np.zeros((140, 80), np.uint8)
        for y in (18, 30, 42, 54, 66, 78, 90):
            self._block(ink, y, 20, 30)          # 左轨一路到 y=98
        for y in (18, 30, 42):
            self._block(ink, y, 32, 42)          # 右轨 y=50 后真实结束
        # 左轨结束后的横栏线（宽 ≥3H、高 ≤0.35H）与孤立污点
        self._block(ink, 110, 8, 72, h=3)
        self._block(ink, 126, 40, 42, h=2)
        groups, audit = self._run(ink)
        self.assertGreaterEqual(len(groups), 1, audit.get('skipped'))
        g = next(x for x in groups if x['state'] == 'completed')
        stops = {(e['type'], e['side']): e['at_y'] for e in g['stop_events']}
        self.assertIn(('track_ink_end', 'right'), stops)
        self.assertTrue(('track_ink_end', 'left') in stops
                        or ('seam_missing', 'pair') in stops, stops)
        self.assertGreater(g['left_bbox'][3], g['right_bbox'][3])
        # 停轨 at_y 必须是末字原图末端，不再是扫描窗末端
        self.assertEqual(stops[('track_ink_end', 'right')], 50)
        self.assertEqual(g['right_bbox'][3], 50)
        self.assertEqual(g['glyph_support']['right']['last_glyph_end'], 50)
        self.assertEqual(g['glyph_support']['left']['last_glyph_end'], 98)
        right_stop = next(e for e in g['stop_events']
                          if e['type'] == 'track_ink_end' and e['side'] == 'right')
        self.assertGreater(right_stop['decided_at_y'], right_stop['at_y'])
        self.assertEqual(right_stop['evidence']['glyph_end_y'], 50)
        self.assertTrue(any(e['type'] in ('rule_boundary', 'blank_gap', 'seam_missing')
                            for e in g['stop_events']))
        # 栏线与污点都没有被并进任何一条轨
        self.assertLess(g['left_bbox'][3], 110)

    def test_cross_seam_glyph_is_not_extended(self):
        ink = np.zeros((100, 80), np.uint8)
        for y in (18, 30, 42, 54, 66):
            self._block(ink, y, 20, 30)
            self._block(ink, y, 32, 42)
        self._block(ink, 60, 24, 40, h=8)        # 横跨中缝 x≈31 的宽字形
        groups, audit = self._run(ink, (0, 0, 80, 100))
        self.assertGreaterEqual(len(groups), 1, audit.get('skipped'))
        g = next(x for x in groups if x['state'] == 'completed')
        # 跨缝字形要么标待核，要么在窗口判据里被拒；无论如何不得进轨框
        self.assertTrue(g['pending'] or any(
            w.get('reason') in ('ambig_ink_crossing', 'track_geometry')
            and w['y'][0] >= 50 for w in audit['windows']))
        for b in g['right_bbox'], g['left_bbox']:
            self.assertFalse(b[0] < 31 < b[2], b)

    def test_hard_boundary_rejects_crossing_full_glyph_box(self):
        # 直接核对辅助判据：字形框 [41,46) 跨断组线 y=44，整框拒绝
        ev = SeamEval(valid=True, seam_x=31,
                      right_boxes=[[32, 41, 42, 46]], left_boxes=[])
        w = Win(y0=35, y1=51, ev=ev)
        lane = {'x0': 32, 'x1': 42, 'cx': 37.0, 'tol': 8.0}
        self.assertEqual(_boxes_in_lane(w, 'right', lane, 31, 20,
                                        hard_y=44), [])

    def test_hard_boundary_integration(self):
        ink = np.zeros((100, 80), np.uint8)
        for y in (6, 18, 30):
            self._block(ink, y, 20, 30)
            self._block(ink, y, 32, 42)
        self._block(ink, 41, 32, 42, h=5)        # 跨 y=44 的末字
        self._block(ink, 44, 58, 78, h=26)       # 标题硬断点：at_y=44
        groups, audit = self._run(ink, (0, 0, 80, 100))
        g = next(x for x in groups if x['state'] == 'completed')
        hard = [e for e in g['stop_events'] if e.get('evidence', {}).get('hard_boundary_y') == 44]
        self.assertTrue(hard, g['stop_events'])
        for e in hard:
            self.assertEqual(e['at_y'], 44)
            self.assertIn('scan_trigger_y', e['evidence'])
            self.assertIn('glyph_end_y', e['evidence'])
        for b in g['right_bbox'], g['left_bbox']:
            self.assertLessEqual(b[3], 44)
        self.assertNotIn([32, 41, 42, 46],
                         [list(b) for b in (g['right_bbox'], g['left_bbox'])])

    def test_window_log_identifies_candidates_at_same_height(self):
        ink = np.zeros((70, 160), np.uint8)
        for x0, x1 in ((20, 30), (32, 42), (100, 110), (112, 122)):
            for y in (18, 34, 50):
                self._block(ink, y, x0, x1)
        groups, audit = self._run(ink, (0, 0, 160, 70))
        self.assertGreaterEqual(len(groups), 2, audit.get('skipped'))
        by_y = {}
        for w in audit['windows']:
            by_y.setdefault(tuple(w['y']), set()).add(w['candidate_seam'])
        shared = [y for y, seams in by_y.items() if len(seams) >= 2]
        self.assertTrue(shared, by_y)
        ids = {w['window_id'] for w in audit['windows']}
        self.assertEqual(len(ids), len(audit['windows']))
        for w in audit['windows']:
            self.assertIn(str(w['candidate_seam']), w['window_id'])
            self.assertTrue('valid' in w and ('reason' in w or w['valid']))


class DedupeTests(unittest.TestCase):
    def _g(self, gid, bbox):
        return {'group_id': gid, 'group_bbox': bbox, 'region_source': 'halfpage_fallback',
                'glyph_support': {'right': {'n': 3}, 'left': {'n': 3}}}

    def test_overlapping_windows_keep_one(self):
        a = self._g('a', [100, 100, 140, 300])
        b = self._g('b', [104, 104, 144, 296])
        kept, log = dedupe_groups([a, b], DEFAULT_CONFIG['dedupe'])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(log), 1)
        self.assertTrue(kept[0].get('merged_candidates'))

    def test_same_column_stacked_kept(self):
        a = self._g('a', [100, 100, 140, 200])
        b = self._g('b', [100, 260, 140, 360])
        kept, log = dedupe_groups([a, b], DEFAULT_CONFIG['dedupe'])
        self.assertEqual(len(kept), 2)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load(S5 / 'groups.json')

    def test_group_fields_present(self):
        keys = {'group_id', 'page_id', 'region_id', 'region_source', 'low_confidence',
                'confirmed_span', 'group_bbox', 'right_bbox', 'left_bbox',
                'glyph_support', 'seam', 'stop_events', 'pending'}
        found = 0
        for p in self.payload['pages']:
            for g in p.get('groups') or []:
                self.assertTrue(keys <= set(g), g['group_id'])
                found += 1
        self.assertGreaterEqual(found, 1)

    def test_stop_enum(self):
        enum = set(DEFAULT_CONFIG['states']['stop_types'])
        for p in self.payload['pages']:
            for g in p.get('groups') or []:
                for e in g['stop_events']:
                    self.assertIn(e['type'], enum)
                    # at_y 是实际断点，decided_at_y 是扫描确认的位置；两者允许差一个窗高
                    self.assertLessEqual(abs(e['at_y'] - e['decided_at_y']), 60)

    def test_p12_has_group_and_valid_boxes(self):
        p = next(x for x in self.payload['pages'] if x['page_id'] == '丹方彚编/page_12')
        self.assertGreaterEqual(len(p['groups']), 1)
        W, H = p['image_size']
        for g in p['groups']:
            b = g['group_bbox']
            self.assertTrue(0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H)

    def test_readers(self):
        pid = '丹方彚编/page_12'
        self.assertIsInstance(groups_for_page(self.payload, pid), list)
        self.assertIsInstance(windows_for_page(self.payload, pid), dict)
        self.assertIn('coverage', audit_for_page(self.payload, pid))
        with self.assertRaises(KeyError):
            groups_for_page(self.payload, 'no/such_page')

    def test_content_hash(self):
        rows = [{k: v for k, v in r.items() if k != 'timing_s'}
                for r in self.payload['pages']]
        self.assertEqual(canon_sha256(rows), self.payload['determinism']['content_sha256'])

    def test_inputs_traceable(self):
        inp = self.payload['inputs']
        self.assertTrue(inp['stage4_content_prefix_ok'])
        self.assertEqual(inp['stage4_release'], 'stage4-v2')
        for k in ('stage4_search_regions', 'stage4_rule_masks', 'stage3_evaluation_config'):
            self.assertTrue(inp[k]['exists'])


if __name__ == '__main__':
    unittest.main()
