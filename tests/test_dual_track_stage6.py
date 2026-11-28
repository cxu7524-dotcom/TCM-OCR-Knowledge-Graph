# -*- coding: utf-8 -*-
"""阶段6 回归：子框外缘由接纳字形决定、补笔上限与硬断点、冻结产物。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402

from dual_track_refine import DEFAULT_CONFIG, refine_group  # noqa: E402
from dual_track_refine.refine import _screen_glyphs, _search_supplement  # noqa: E402
from dual_track_refine.api import S6  # noqa: E402


def _group() -> dict:
    return {
        'group_id': 'synthetic/page#r001#0040', 'page_id': 'synthetic/page',
        'region_id': 'synthetic/page#r001', 'region_source': 'validated_columns',
        'low_confidence': False, 'pending': False, 'internal_seam': 31,
        'confirmed_span': [52, 108],
        'right_bbox': [32, 52, 42, 108], 'left_bbox': [20, 52, 30, 108],
        'group_bbox': [20, 52, 42, 108],
        'scale_used': {'H': 20.0, 'G': 4.0, 'glyph_pitch': 16.0},
        'small_scale': {'small_glyph_height': 8.0, 'small_glyph_width': 10.0,
                        'small_glyph_pitch': 12.0},
        'seam': {'points': [{'y': 52, 'x': 31}, {'y': 108, 'x': 31}]},
        'stop_events': [{'type': 'region_boundary', 'side': 'pair',
                         'at_y': 120, 'decided_at_y': 120, 'evidence': {}}],
    }


class RefineTests(unittest.TestCase):
    def _ink(self, extra=None, clipped=False):
        ink = np.zeros((120, 80), np.uint8)
        for y in (52, 64, 76, 88, 100):
            ink[y:y + 8, 20:30] = 1
            ink[y:y + 8, 32:42] = 1
        if extra:
            for y in extra:
                ink[y:y + 6, 20:29] = 1
                ink[y:y + 6, 32:41] = 1
        if clipped:
            ink[100:108, 0:10] = 1
        return ink

    def test_edges_follow_accepted_glyphs(self):
        rec = refine_group(_group(), [0, 0, 80, 120], self._ink(), [], DEFAULT_CONFIG)
        for side, x1, x2 in (('right', 32, 42), ('left', 20, 30)):
            s = rec['sides'][side]
            self.assertEqual((s['edge']['x1'], s['edge']['x2']), (x1, x2))
            self.assertEqual(s['bbox_after'][0], x1)
            self.assertEqual(s['bbox_after'][2], x2)

    @staticmethod
    def _g(b, side='left'):
        return {'bbox': list(b), 'w': b[2] - b[0], 'h': b[3] - b[1],
                'cx': (b[0] + b[2]) / 2.0, 'cy': (b[1] + b[3]) / 2.0,
                '_side': side}

    def _search(self, pool, head_y=0):
        anchor = self._g([20, 52, 30, 60])
        return _search_supplement(
            anchor, 'head', 'left', pool, set(), [[20, 52, 30, 60]],
            [32, 52, 42, 60], [20, 30], [0, 0, 80, 120], [],
            [{'y': 52, 'x': 31}, {'y': 108, 'x': 31}], 31, [], [], [],
            head_y, 120, 8.0, 9.0, 40.0, DEFAULT_CONFIG)

    def test_boundary_clipped_glyph_not_used_for_edge(self):
        pool = [self._g([0, 45, 10, 51]), self._g([20, 45, 29, 51])]
        _, excluded, _ = self._search(pool)
        self.assertIn('boundary_clipped', [x['reason'] for x in excluded])

    def test_supplement_cap_and_limited_flag(self):
        pool = [self._g([20, y, 29, y + 6]) for y in (15, 25, 35, 45)]
        accepted, _, limited = self._search(pool)
        self.assertLessEqual(len(accepted), 2)
        self.assertTrue(limited)

    def test_neighbor_bound_blocks_head_supplement(self):
        pool = [self._g([20, 45, 29, 51])]
        accepted, excluded, _ = self._search(pool, head_y=48)
        self.assertEqual(accepted, [])
        self.assertTrue(any(x['reason'] == 'above_hard_boundary'
                            for x in excluded))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((S6 / 'groups.json').read_text(encoding='utf-8'))
        cls.metrics = json.loads((S6 / 'metrics.json').read_text(encoding='utf-8'))

    def test_frozen_group_ids(self):
        s5 = json.loads((S6.parent / 'stage5' / 'groups.json').read_text(encoding='utf-8'))
        a = {g['group_id'] for p in s5['pages'] for g in p['groups']}
        b = {r['group_id'] for p in self.payload['pages'] for r in p['records']}
        self.assertEqual(a, b)

    def test_p12_one_to_one_and_title_isolation(self):
        ids = ('丹方彚编/p12/DF12-03', '丹方彚编/p12/DF12-11')
        pairs = {}
        for page in self.metrics['scores']['stage6_refined']['by_page'].values():
            pairs.update({p[1]: p for p in page['0.5']['pairs']})
        recs = {r['group_id']: r for p in self.payload['pages'] for r in p['records']}
        for gid in ids:
            self.assertIn(gid, pairs)
            pred, _, iou = pairs[gid]
            self.assertGreaterEqual(iou, 0.5)
            r = recs[pred]
            self.assertTrue(r['after']['right_bbox'] and r['after']['left_bbox'])
            self.assertEqual(r['region_source'], 'halfpage_fallback')
            self.assertTrue(r['low_confidence'])
        self.assertLessEqual(recs[pairs[ids[0]][0]]['after']['group_bbox'][3], 536)
        self.assertGreaterEqual(recs[pairs[ids[1]][0]]['after']['group_bbox'][1], 655)

    def test_low_confidence_reasons_locatable(self):
        for p in self.payload['pages']:
            for r in p['records']:
                if r['horizontal_low_confidence'] or r['supplement_limited']:
                    self.assertTrue(r['flags'])
                    for side in ('right', 'left'):
                        s = r['sides'][side]
                        self.assertIn('edge', s)
                        self.assertIn('supplement', s)


class BoundaryV2Tests(unittest.TestCase):
    """stage6-v2：栏线邻近/相交、补笔后整框复验。"""

    @staticmethod
    def _g(b):
        return {'bbox': list(b), 'w': b[2] - b[0], 'h': b[3] - b[1],
                'cx': (b[0] + b[2]) / 2.0, 'cy': (b[1] + b[3]) / 2.0}

    def test_rule_proximity_keeps_full_glyph_and_tail(self):
        g = _group()
        g['right_bbox'] = [32, 52, 46, 108]
        g['left_bbox'] = [20, 52, 30, 108]
        g['group_bbox'] = [20, 52, 46, 108]
        g['confirmed_span'] = [52, 108]
        ink = np.zeros((120, 80), np.uint8)
        for y in (52, 64, 76, 88):
            ink[y:y + 8, 32:44] = 1
            ink[y:y + 8, 18:30] = 1
        ink[100:108, 32:46] = 1
        rec = refine_group(g, [0, 0, 80, 120], ink, [[48, 49, 96, 112]],
                           DEFAULT_CONFIG, {'head_y': 0, 'tail_y': 120})
        right = rec['sides']['right']
        self.assertEqual(right['bbox_after'][3], 108)
        self.assertIn([32, 100, 46, 108], right['accepted_glyphs'])
        self.assertTrue(right['rule_proximity_review'])

    def test_rule_intersection_is_pending_and_excluded(self):
        glyph = self._g([32, 100, 46, 108])
        kept, excluded = _screen_glyphs(
            [glyph], 'right', [0, 0, 80, 120], [[44, 45, 100, 108]],
            [{'y': 52, 'x': 31}, {'y': 108, 'x': 31}], 31, 8.0, 10.0,
            DEFAULT_CONFIG, 12.0)
        self.assertEqual(kept, [])
        self.assertIn('ambiguous_rule_attached', [x['reason'] for x in excluded])

    def test_independent_rule_does_not_affect_glyph(self):
        glyph = self._g([32, 100, 46, 108])
        kept, excluded = _screen_glyphs(
            [glyph], 'right', [0, 0, 80, 120], [[60, 61, 100, 108]],
            [{'y': 52, 'x': 31}, {'y': 108, 'x': 31}], 31, 8.0, 10.0,
            DEFAULT_CONFIG, 12.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(excluded, [])

    def _supplement(self, cand, current, seam_pts):
        return _search_supplement(
            self._g(current[0]), 'head', 'right', [self._g(cand)], set(),
            [list(b) for b in current], [20, 60, 29, 68], [32, 46],
            [0, 0, 80, 120], [], seam_pts, 31, [], [], [], 0, 120,
            8.0, 10.0, 30.0, DEFAULT_CONFIG)

    def test_local_ok_but_final_rect_crosses_seam_is_rejected(self):
        accepted, excluded, _ = self._supplement(
            [30, 45, 44, 51], [[30, 60, 42, 68]],
            [{'y': 45, 'x': 28}, {'y': 64, 'x': 31}])
        self.assertEqual(accepted, [])
        self.assertIn('crosses_seam_in_final_rect',
                      [x['reason'] for x in excluded])

    def test_local_and_final_ok_supplement_accepted(self):
        accepted, excluded, _ = self._supplement(
            [32, 45, 42, 51], [[32, 60, 42, 68]],
            [{'y': 45, 'x': 31}, {'y': 64, 'x': 31}])
        self.assertEqual([x['bbox'] for x in accepted], [[32, 45, 42, 51]])
        self.assertEqual(excluded, [])


class FinalRectTests(unittest.TestCase):
    """stage6-v3：首尾累计、折线节点与半开坐标的最终整框复验。"""

    @staticmethod
    def _group(seam_pts, y1=76):
        g = _group()
        g['group_id'] = 'synthetic/page#final#001'
        g['right_bbox'] = [32, 52, 44, y1]
        g['left_bbox'] = [18, 52, 30, y1]
        g['group_bbox'] = [18, 52, 44, y1]
        g['confirmed_span'] = [52, y1]
        g['internal_seam'] = 31
        g['seam'] = {'points': seam_pts}
        g['small_scale'] = {'small_glyph_height': 6.0, 'small_glyph_width': 12.0,
                            'small_glyph_pitch': 12.0}
        g['stop_events'] = [{'type': 'region_boundary', 'side': 'pair',
                             'at_y': 120, 'decided_at_y': 120, 'evidence': {}}]
        return g

    @staticmethod
    def _ink(right_ys, head=None, tail=None):
        ink = np.zeros((120, 80), np.uint8)
        for y in right_ys:
            ink[y:y + 8, 32:44] = 1
            ink[y:y + 8, 18:30] = 1
        if head is not None:
            ink[head:head + 6, 30:44] = 1
        if tail is not None:
            ink[tail:tail + 6, 30:44] = 1
        return ink

    def test_head_and_tail_each_ok_but_merged_crosses_seam(self):
        pts = [{'y': 45, 'x': 28}, {'y': 64, 'x': 31}, {'y': 81, 'x': 28}]
        g = self._group(pts)
        ink = self._ink([60], head=45, tail=75)
        rec = refine_group(g, [0, 0, 80, 120], ink, [], DEFAULT_CONFIG,
                           {'head_y': 0, 'tail_y': 120})
        right = rec['sides']['right']
        self.assertGreaterEqual(right['bbox_after'][0], 32)
        self.assertIn('crosses_seam_in_final_rect',
                      [x['reason'] for x in right['supplement']['head']['excluded']]
                      + [x['reason'] for x in right['final_validation']['rolled_back']])

    def test_seam_turn_in_blank_gap_is_included(self):
        pts = [{'y': 45, 'x': 28}, {'y': 64, 'x': 31}, {'y': 81, 'x': 28}]
        g = self._group(pts)
        ink = self._ink([52, 70], head=45)
        rec = refine_group(g, [0, 0, 80, 120], ink, [], DEFAULT_CONFIG,
                           {'head_y': 0, 'tail_y': 120})
        right = rec['sides']['right']
        self.assertTrue(any(x['bbox'] == [30, 45, 44, 51]
                            for x in right['supplement']['head']['excluded'])
                        or any(x['bbox'] == [30, 45, 44, 51]
                               for x in right['final_validation']['rolled_back']),
                        right['supplement']['head'])

    def test_both_ends_can_be_supplemented_when_safe(self):
        pts = [{'y': 45, 'x': 28}, {'y': 81, 'x': 28}]
        g = self._group(pts)
        ink = self._ink([60], head=45, tail=75)
        rec = refine_group(g, [0, 0, 80, 120], ink, [], DEFAULT_CONFIG,
                           {'head_y': 0, 'tail_y': 120})
        right = rec['sides']['right']
        self.assertEqual(right['bbox_after'][1], 45)
        self.assertEqual(right['bbox_after'][3], 81)
        self.assertEqual(right['bbox_after'][0], 30)
        self.assertTrue(right['final_validation']['ok'])
        self.assertEqual(right['final_validation']['rolled_back'], [])


class FinalStatusTests(unittest.TestCase):
    """stage6-v4：冲突状态回填与浮点中缝节点。"""

    def test_existing_conflict_backfills_failed_status(self):
        g = FinalRectTests._group([
            {'y': 62.4, 'x': 31}, {'y': 62.5, 'x': 33}, {'y': 80, 'x': 31}])
        ink = FinalRectTests._ink([60])
        rec = refine_group(g, [0, 0, 80, 120], ink, [], DEFAULT_CONFIG,
                           {'head_y': 0, 'tail_y': 120})
        right = rec['sides']['right']
        fc = right['final_validation']
        self.assertFalse(fc['ok'])
        self.assertEqual(fc['reason'], 'crosses_seam_in_final_rect')
        self.assertTrue(fc['fallback_to_tentative'])
        self.assertEqual(fc['status'], 'rejected_to_tentative')
        self.assertEqual(right['bbox_after'], right['bbox_before'])
        self.assertTrue(right['horizontal_low_confidence'])
        self.assertTrue(any('existing_boundary_conflict' in x
                            for x in right['pending_reasons']))

    def test_float_seam_node_used_without_truncation(self):
        from dual_track_refine.refine import _whole_rect_ok
        pts = [{'y': 62.4, 'x': 31}, {'y': 62.5, 'x': 33}, {'y': 80, 'x': 31}]
        ok, why = _whole_rect_ok([[32, 52, 44, 76]], 'right', [0, 0, 80, 120],
                                 [], pts, 31, [], None, DEFAULT_CONFIG)
        self.assertFalse(ok)
        self.assertEqual(why, 'crosses_seam_in_final_rect')


if __name__ == '__main__':
    unittest.main()
