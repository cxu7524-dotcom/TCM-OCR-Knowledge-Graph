# -*- coding: utf-8 -*-
"""第四阶段测试：半页隔离、路径判定、尺度墨迹单位、掩膜记录、坐标与可复现。

合成样例用来覆盖真实数据里缺的分支（成组栏线、断线、贴字线、窄裁块、缺尺度、
只在右半页画栏线、一个不合格间隔 + 尾段、单点斜线、0/1 与 0/255 编码）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402

from dual_track_search import (DEFAULT_CONFIG, audit_for_page,  # noqa: E402
                               build_rule_mask, estimate_scale, glyph_bands,
                               group_rule_runs, group_rules, line_eligibility,
                               mask_digest, page_mask_record, plan_page,
                               regions_for_page, rule_lines_for_page, scale_for_page,
                               validate_columns, verify_record)
from dual_track_search.columns import merge_split_columns   # noqa: E402
from dual_track_search.rules import (classify_rule_touching,  # noqa: E402
                                     detect_rule_candidates, merge_broken_rules)

S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def blank_bw(h: int = 400, w: int = 300) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def draw_vline(bw, x: int, y0: int, y1: int, thick: int = 1, value: int = 255):
    bw[y0:y1, x:x + thick] = value


def draw_glyphs(bw, x0: int, x1: int, ys, h: int = 30, value: int = 255):
    for y in ys:
        bw[y:y + h, x0:x1] = value


def vlines_columns(xs, y0=20, y1=380):
    return [{'bbox': [x, y0, x + 6, y1]} for x in xs]


class RulePathTests(unittest.TestCase):
    """路径一：成组栏线（含多 run 与半页隔离）。"""

    def _page(self, xs=(40, 90, 140, 190, 240), h=400, w=300):
        bw = blank_bw(h, w)
        for x in xs:
            draw_vline(bw, x, 20, 380)
        for a, b in zip(xs[:-1], xs[1:]):
            draw_glyphs(bw, a + 10, b - 10, range(60, 360, 60), h=30)
        return bw

    def test_grouped_rules_detected(self):
        bw = self._page()
        cands = detect_rule_candidates(bw, (0, 400), DEFAULT_CONFIG['rules'])
        cands = classify_rule_touching(cands, bw, DEFAULT_CONFIG['rules'])
        self.assertGreaterEqual(len(cands), 5)
        group, reasons = group_rules(cands, bw, (0, 400), DEFAULT_CONFIG['rules'])
        self.assertIsNotNone(group, reasons)
        self.assertGreaterEqual(len(group.columns), 3)
        self.assertAlmostEqual(group.pitch_px, 50.0, delta=2.0)

    def test_too_few_lines_falls_through(self):
        bw = blank_bw()
        draw_vline(bw, 100, 20, 380)
        cands = detect_rule_candidates(bw, (0, 400), DEFAULT_CONFIG['rules'])
        group, reasons = group_rules(cands, bw, (0, 400), DEFAULT_CONFIG['rules'])
        self.assertIsNone(group)
        self.assertTrue(any('below_min' in r['why'] or 'pitch' in r['why'] for r in reasons))

    def test_touching_rule_classified(self):
        bw = blank_bw()
        draw_vline(bw, 100, 20, 380)
        for y in range(20, 380, 8):
            bw[y:y + 6, 101:106] = 255
        cands = detect_rule_candidates(bw, (0, 400), DEFAULT_CONFIG['rules'])
        cands = classify_rule_touching(cands, bw, DEFAULT_CONFIG['rules'])
        line = [c for c in cands if abs(c.x0 - 100) <= 2][0]
        self.assertEqual(line.kind, 'touching')
        self.assertEqual(line.confidence, 'low')
        group_rule_runs(cands, bw, (0, 400), DEFAULT_CONFIG['rules'])   # 设 status/qualified
        d = line.to_dict()
        self.assertEqual(d['status'], 'touching')
        self.assertFalse(d['qualified'])
        self.assertFalse(d['excluded_from_scale'])
        self.assertEqual(d['eligible_segments'], [])

    def test_independent_rule_classified(self):
        bw = blank_bw()
        draw_vline(bw, 100, 20, 380)
        cands = detect_rule_candidates(bw, (0, 400), DEFAULT_CONFIG['rules'])
        cands = classify_rule_touching(cands, bw, DEFAULT_CONFIG['rules'])
        line = [c for c in cands if abs(c.x0 - 100) <= 2][0]
        self.assertEqual(line.kind, 'independent')
        group_rule_runs(cands, bw, (0, 400), DEFAULT_CONFIG['rules'])
        d = line.to_dict()
        self.assertEqual(d['status'], 'confirmed')
        self.assertTrue(d['qualified'])
        self.assertTrue(d['excluded_from_scale'])
        self.assertEqual(len(d['eligible_segments']), 1)

    def test_broken_rule_merged_with_record(self):
        bw = blank_bw()
        draw_vline(bw, 100, 20, 180)
        draw_vline(bw, 100, 190, 380)
        cands = detect_rule_candidates(bw, (0, 400), DEFAULT_CONFIG['rules'])
        self.assertEqual(len(cands), 2)
        merged, log = merge_broken_rules(cands, DEFAULT_CONFIG['rules'])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].kind, 'broken')
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['segments'], 2)
        # 合并后覆盖率按「真实纵向跨度 / 正文区高」算，不能把两段覆盖率相加
        self.assertLessEqual(merged[0].coverage, 1.0)

    def test_bad_gap_splits_runs_not_page(self):
        """一个过大间隔 + 一条孤立尾线：前面两段继续成组，尾线单独记录。"""
        w, h = 700, 620
        bw = blank_bw(h, w)
        xs = [80, 130, 180, 230, 400, 450, 500, 660]
        for x in xs:
            draw_vline(bw, x, 40, 580)
        for a, b in zip(xs[:-1], xs[1:]):
            if b - a < 120:
                draw_glyphs(bw, a + 10, b - 10, range(70, 540, 50), h=28)
        cands = detect_rule_candidates(bw, (40, 580), DEFAULT_CONFIG['rules'])
        cands = classify_rule_touching(cands, bw, DEFAULT_CONFIG['rules'])
        cands, _ = merge_broken_rules(cands, DEFAULT_CONFIG['rules'])
        groups, log = group_rule_runs(cands, bw, (40, 580), DEFAULT_CONFIG['rules'])
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(g.lines) for g in groups], [4, 3])
        self.assertTrue(any(x.get('why') == 'run_single_line' for x in log))

    def test_coords_mapped_back_from_half_crop(self):
        """半页裁块内检测后，x 必须加回 offset（只加一次）。"""
        W, H = 1024, 800
        bw = blank_bw(H, W)
        xs = [600, 650, 700, 750]
        for x in xs:
            draw_vline(bw, x, 60, 740)
        for a, b in zip(xs[:-1], xs[1:]):
            draw_glyphs(bw, a + 12, b - 12, range(90, 700, 52), h=30)
        plan = plan_page('synthetic/offset', 'synthetic', bw, (W, H),
                         [(520, 1024), (0, 504)], [], DEFAULT_CONFIG).to_dict()
        xs_lines = [ln['x0'] for ln in plan['rule_lines']]
        self.assertEqual(sorted(xs_lines), xs)


class HalfIsolationTests(unittest.TestCase):
    """验收缺口 1：只在右半页画栏线，左半页不得复用。"""

    @classmethod
    def setUpClass(cls):
        W, H = 1024, 800
        bw = blank_bw(H, W)
        cls.xs = [600, 650, 700, 750]
        for x in cls.xs:
            draw_vline(bw, x, 60, 740)
        for a, b in zip(cls.xs[:-1], cls.xs[1:]):
            draw_glyphs(bw, a + 12, b - 12, range(90, 700, 52), h=30)
        cls.left_cols = [[100, 60, 140, 740], [160, 60, 200, 740], [220, 60, 260, 740]]
        for c in cls.left_cols:
            draw_glyphs(bw, c[0] + 4, c[2] - 4, range(90, 700, 52), h=30)
        columns = [{'bbox': c} for c in cls.left_cols] + \
                  [{'bbox': [x + 2, 60, x + 46, 740]} for x in cls.xs[:-1]]
        cls.plan = plan_page('synthetic/half', 'synthetic', bw, (W, H),
                             [(520, 1024), (0, 504)], columns, DEFAULT_CONFIG).to_dict()

    def _by_half(self, hid):
        return [r for r in self.plan['regions'] if r['half_id'] == hid]

    def test_left_half_reuses_nothing(self):
        left = self._by_half('left')
        self.assertTrue(left)
        self.assertFalse(any(r['column_range_source'] == 'grouped_rules' for r in left))
        hd = next(h for h in self.plan['halves_detail'] if h['half_id'] == 'left')
        self.assertEqual(hd['rule_candidates'], [])
        self.assertFalse(hd['paths']['grouped_rules']['ok'])

    def test_right_half_has_rules(self):
        right = self._by_half('right')
        legacy = [r for r in right if r['column_range_source'] == 'grouped_rules']
        self.assertGreaterEqual(len(legacy), 1)
        self.assertEqual([r['bbox'][0] for r in legacy], [601, 651, 701])

    def test_every_region_inside_its_half(self):
        for hid, (hx0, hx1) in (('right', (520, 1024)), ('left', (0, 504))):
            for r in self._by_half(hid):
                self.assertGreaterEqual(r['bbox'][0], hx0, r['region_id'])
                self.assertLessEqual(r['bbox'][2], hx1, r['region_id'])
                self.assertEqual(r['half_bounds'], [hx0, hx1])
                self.assertEqual(r['transform']['offset_x'], hx0)

    def test_out_of_scope_is_not_uncovered(self):
        # 中缝带 [504,520) 属范围外，不能算「应处理未覆盖」
        rose = [o for o in self.plan['out_of_scope'] if o['range'] == [504, 520]]
        self.assertEqual(len(rose), 1)
        self.assertIn('中缝', rose[0]['why'])
        self.assertEqual(self.plan['status'], 'completed')


class ColumnPathTests(unittest.TestCase):
    """验收缺口 2：校验必须真正决定路径。"""

    ref = dict(book_stats={'cols': 310, 'median_width': 43.0}, global_stats=None,
               cfg=DEFAULT_CONFIG['columns'])

    def test_body_overlap_downgrades_both(self):
        cols = [{'bbox': [30, 10, 70, 390]}, {'bbox': [45, 10, 85, 390]}]
        r = validate_columns(cols, (0, 1024), (1024, 600), rule_lines=[], **self.ref)
        self.assertEqual([v.decision for v in r['verdicts']], ['fallback', 'fallback'])
        m = r['verdicts'][0].measurements['overlaps'][0]
        self.assertEqual(m['overlap_px'], 25)
        self.assertAlmostEqual(m['ratio_of_narrower'], 0.625, places=3)
        self.assertIn('主体横向重叠', r['verdicts'][0].reasons[-1])

    def test_rule_inside_box_splits(self):
        cols = [{'bbox': [30, 10, 70, 390]}]
        rules = [{'center': 50.0, 'kind': 'independent', 'confidence': 'high',
                  'coverage': 0.9}]
        r = validate_columns(cols, (0, 1024), (1024, 600), rule_lines=rules, **self.ref)
        v = r['verdicts'][0]
        self.assertEqual(v.decision, 'split')
        self.assertEqual([o['bbox'][:3] for o in v.outputs],
                         [[30, 10, 50], [50, 10, 70]])
        self.assertEqual(r['reference']['split_min_width_px'], 15)

    def test_rule_inside_too_narrow_falls_back(self):
        cols = [{'bbox': [30, 10, 60, 390]}]
        rules = [{'center': 33.0, 'kind': 'independent', 'confidence': 'high',
                  'coverage': 0.9}]
        r = validate_columns(cols, (0, 1024), (1024, 600), rule_lines=rules, **self.ref)
        self.assertEqual(r['verdicts'][0].decision, 'fallback')
        self.assertIn('拆分不可靠', r['verdicts'][0].reasons[-1])

    def test_no_reference_goes_low_confidence(self):
        cols = [{'bbox': [300, 10, 530, 390]}]
        r = validate_columns(cols, (0, 1024), (1024, 600), book_stats=None,
                             global_stats=None, cfg=DEFAULT_CONFIG['columns'],
                             rule_lines=[])
        v = r['verdicts'][0]
        self.assertEqual(v.decision, 'fallback')
        self.assertIsNone(r['reference']['value_px'])
        self.assertEqual(r['reference']['source'], 'none')
        self.assertAlmostEqual(v.measurements['page_width_ratio'], 0.2246, places=3)
        self.assertIn('无宽度参考', v.reasons[-1])

    def test_page_median_substitute_is_sourced(self):
        cols = [{'bbox': [100 + k * 50, 10, 140 + k * 50, 390]} for k in range(8)]
        cols.append({'bbox': [520, 10, 750, 390]})
        r = validate_columns(cols, (0, 1024), (1024, 600),
                             book_stats={'cols': 5, 'median_width': 43.0},
                             global_stats={'cols': 1702, 'median_width': 37.0},
                             cfg=DEFAULT_CONFIG['columns'],
                             page_stats={'cols': 9, 'median_width': 40.0},
                             rule_lines=[])
        self.assertEqual(r['reference']['source'], 'same_page_median')
        self.assertEqual(r['reference']['value_px'], 40.0)
        self.assertEqual(r['verdicts'][-1].decision, 'fallback')

    def test_cross_half_box_falls_back(self):
        cols = [{'bbox': [480, 10, 560, 390]}]
        r = validate_columns(cols, (520, 1024), (1024, 600), rule_lines=[], **self.ref)
        v = r['verdicts'][0]
        self.assertEqual(v.decision, 'fallback')
        self.assertIn('越过半页边界', v.reasons[0])

    def test_wide_box_rejected_with_reason(self):
        cols = [{'bbox': [545, 0, 991, 873]}, {'bbox': [100, 0, 140, 800]}]
        r = validate_columns(cols, (0, 1024), (1024, 887),
                             book_stats={'cols': 310, 'median_width': 43.0},
                             global_stats={'cols': 1702, 'median_width': 37.0},
                             cfg=DEFAULT_CONFIG['columns'])
        rej = r['rejected']
        self.assertEqual(len(rej), 1)
        self.assertEqual(rej[0].width, 446)
        self.assertIn('446', rej[0].reasons[0])
        self.assertIn('10.4', rej[0].reasons[0])
        self.assertEqual(r['reference']['source'], 'same_book')

    def test_merge_split_half_column(self):
        cols = [{'bbox': [0, 0, 25, 800]}, {'bbox': [28, 0, 53, 800]}]
        merged, log = merge_split_columns(cols, 43.0, DEFAULT_CONFIG['columns'])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['bbox'][2] - merged[0]['bbox'][0], 53)
        self.assertEqual(len(log), 1)

    def test_does_not_merge_two_real_columns(self):
        cols = [{'bbox': [0, 0, 29, 800]}, {'bbox': [49, 0, 78, 800]}]
        merged, log = merge_split_columns(cols, 29.0, DEFAULT_CONFIG['columns'])
        self.assertEqual(len(merged), 2)
        self.assertEqual(log, [])


class ScaleTests(unittest.TestCase):
    """验收缺口 3：墨迹单位统一 + H/G/pitch 分列。"""

    def _strip(self, h=30, gap=5, n=9, w=40, value=255):
        total = n * h + (n - 1) * gap + 20
        bw = blank_bw(total, w)
        ys = [10 + k * (h + gap) for k in range(n)]
        draw_glyphs(bw, 2, w - 2, ys, h=h, value=value)
        return bw, ys

    def _est(self, bw, w):
        return estimate_scale(bw, [{'bbox': [0, 0, w, bw.shape[0]]}],
                              (0, bw.shape[0]), (w, bw.shape[0]),
                              DEFAULT_CONFIG['scale'], narrow_columns=1)

    def test_H_G_pitch_from_synthetic_bands(self):
        bw, _ = self._strip()
        est = self._est(bw, 40)
        self.assertEqual((est.H, est.G, est.glyph_pitch), (30.0, 5.0, 35.0))
        self.assertEqual(est.confidence_status, 'confident')
        self.assertEqual(est.value_origin, 'measured')
        self.assertEqual(est.small_glyph_status, 'not_estimated')
        self.assertIsNone(est.small_glyph_height)

    def test_binary_and_255_identical(self):
        bw1, _ = self._strip(value=1)
        bw255, _ = self._strip(value=255)
        a, b = self._est(bw1, 40), self._est(bw255, 40)
        self.assertEqual((a.H, a.G, a.glyph_pitch), (b.H, b.G, b.glyph_pitch))
        self.assertEqual(a.sample_count, b.sample_count)
        self.assertEqual([g['y0'] for g in glyph_bands(bw1, 0, 40, 0, bw1.shape[0],
                                                       DEFAULT_CONFIG['scale'])],
                         [g['y0'] for g in glyph_bands(bw255, 0, 40, 0, bw255.shape[0],
                                                       DEFAULT_CONFIG['scale'])])

    def test_single_dot_diagonal_rejected(self):
        bw = blank_bw(300, 40)
        for i in range(250):
            bw[10 + i, 5 + (i % 30)] = 255
        est = self._est(bw, 40)
        self.assertEqual(glyph_bands(bw, 0, 40, 0, 300, DEFAULT_CONFIG['scale']), [])
        self.assertEqual(est.confidence_status, 'failed')
        self.assertIn('no_glyph_band', est.failure_reason)
        self.assertIsNone(est.H)

    def test_rule_band_excluded_from_projection(self):
        bw, _ = self._strip(w=44)
        # 在字带中间放一条独立栏线（1px 竖线），排除前后 H 不应变化
        for y in range(0, bw.shape[0]):
            bw[y, 20] = 255
        base = self._est(bw, 44)
        cands = detect_rule_candidates(bw, (0, bw.shape[0]), DEFAULT_CONFIG['rules'])
        excl = [(c.x0, c.x1) for c in cands]
        self.assertTrue(excl)
        est = estimate_scale(bw, [{'bbox': [0, 0, 44, bw.shape[0]]}], (0, bw.shape[0]),
                             (44, bw.shape[0]), DEFAULT_CONFIG['scale'],
                             exclude_rules=excl, narrow_columns=1)
        self.assertEqual(est.H, base.H)
        self.assertEqual(est.filter_conditions['excluded_rule_columns'], [list(map(int, e))
                                                                          for e in excl])
        bands = glyph_bands(bw, 0, 44, 0, bw.shape[0], DEFAULT_CONFIG['scale'], excl)
        self.assertTrue(bands[0]['excluded_rule_bands'])

    def test_glyph_bands_half_open(self):
        bw, ys = self._strip(n=2)
        bands = glyph_bands(bw, 0, bw.shape[1], 0, bw.shape[0], DEFAULT_CONFIG['scale'])
        self.assertEqual(bands[0]['y0'], ys[0])
        self.assertEqual(bands[0]['y1'], ys[0] + 30)

    def test_narrow_block_still_estimates(self):
        bw, _ = self._strip(n=9, w=32)
        est = self._est(bw, 32)
        self.assertEqual(est.confidence_status, 'confident')
        self.assertEqual(est.source, 'glyph_bands_narrow_block')
        self.assertTrue(est.filter_conditions['narrow_block'])

    def test_insufficient_samples_reports_reason(self):
        bw, _ = self._strip(n=2, w=32)
        est = self._est(bw, 32)
        self.assertEqual(est.confidence_status, 'failed')
        self.assertIn('insufficient', est.failure_reason)

    def test_no_default_values_on_failure(self):
        bw = blank_bw(300, 40)
        d = self._est(bw, 40).to_dict()
        for k in ('H', 'G', 'glyph_pitch'):
            self.assertIsNone(d[k], k)
        self.assertEqual(d['value_origin'], 'none')

    def test_inheritance_is_marked(self):
        page = self._est(self._strip()[0], 40)
        empty = blank_bw(300, 40)
        est = self._est(empty, 40)
        est.inherit_missing_from(page, '区域样本不足')
        d = est.to_dict()
        self.assertEqual(d['value_origin'], 'inherited_from_page')
        self.assertEqual(d['confidence_status'], 'inherited')
        self.assertEqual(sorted(d['inherited_fields']), ['G', 'H', 'glyph_pitch'])
        self.assertEqual(d['H'], page.H)


class MaskRecordTests(unittest.TestCase):
    """掩膜可重建，贴字线单列。"""

    def _page(self):
        return {'page_id': 'synthetic/mask', 'image_size': [200, 120],
                'halves': [[100, 200], [0, 100]],
                'rule_lines': [
                    {'x0': 20, 'x1': 22, 'y0': 10, 'y1': 110, 'kind': 'independent',
                     'status': 'confirmed', 'confidence': 'high', 'touches_text': False,
                     'coverage': 1.0, 'qualified': True, 'ambiguous': False,
                     'half_id': 'left', 'group_id': None, 'segments': []},
                    {'x0': 60, 'x1': 62, 'y0': 10, 'y1': 110, 'kind': 'touching',
                     'status': 'touching', 'confidence': 'low', 'touches_text': True,
                     'coverage': 1.0, 'qualified': False, 'ambiguous': False,
                     'half_id': 'right', 'group_id': None, 'segments': []},
                ]}

    def test_record_and_rebuild_match(self):
        rec = page_mask_record(self._page())
        chk = verify_record(rec)
        self.assertTrue(all(v['match'] for v in chk.values()))
        self.assertEqual(rec['counts']['all'], 2)
        self.assertEqual(rec['counts']['qualified'], 1)
        self.assertEqual(rec['counts']['touching'], 1)
        self.assertEqual(rec['counts']['segments'], 2)
        self.assertEqual(rec['masks']['all']['ink_px'], 2 * 100 + 2 * 100)
        self.assertEqual(rec['masks']['independent']['ink_px'], 2 * 100)

    def test_touching_not_in_independent_scope(self):
        rec = page_mask_record(self._page())
        m_all = build_rule_mask(rec['lines'], rec['image_size'], 'all')
        m_ind = build_rule_mask(rec['lines'], rec['image_size'], 'independent')
        self.assertEqual(int(m_all[:, 60:62].sum()), 200)
        self.assertEqual(int(m_ind[:, 60:62].sum()), 0)
        self.assertNotEqual(mask_digest(m_all), mask_digest(m_ind))
        self.assertIn('uint8 0/1', rec['encoding'])


class BrokenSegmentTests(unittest.TestCase):
    """断线合并保留逐段证据：混合段保歧义、独立段照常合并、掩膜按段置位。"""

    @staticmethod
    def _build(mixed: bool, h=400, w=300):
        bw = blank_bw(h, w)
        draw_vline(bw, 100, 20, 180, thick=3)
        draw_vline(bw, 100, 190, 380, thick=3)
        if mixed:
            for y in range(20, 180, 6):
                bw[y:y + 4, 103:108] = 255
                bw[y:y + 4, 95:100] = 255
        return bw

    def _merged(self, mixed: bool):
        bw = self._build(mixed)
        cands = detect_rule_candidates(bw, (20, 380), DEFAULT_CONFIG['rules'])
        cands = classify_rule_touching(cands, bw, DEFAULT_CONFIG['rules'])
        merged, log = merge_broken_rules(cands, DEFAULT_CONFIG['rules'], body_h=360)
        self.assertEqual(len(merged), 1)
        m = merged[0]
        e = line_eligibility(m, 360, DEFAULT_CONFIG['rules'])
        m.qualified = e['qualified']
        return bw, m, log

    def test_mixed_keeps_segment_evidence_and_ambiguity(self):
        bw, m, log = self._merged(mixed=True)
        self.assertEqual(m.kind, 'broken')
        self.assertEqual(m.status, 'mixed_ambiguous')
        self.assertTrue(m.ambiguous)
        self.assertFalse(m.qualified)
        self.assertEqual([(s['y0'], s['y1']) for s in m.segments], [(20, 181), (191, 380)])
        self.assertEqual([s['touches_text'] for s in m.segments], [True, False])
        self.assertEqual([s['confidence'] for s in m.segments], ['low', 'high'])
        self.assertAlmostEqual(m.eligible_coverage, 189 / 360, places=3)
        self.assertEqual(log[0]['status'], 'mixed_ambiguous')
        self.assertEqual(len(log[0]['segment_detail']), 2)
        self.assertEqual(log[0]['coord_system'], 'original_image')

    def test_mixed_mask_skips_touching_and_gap(self):
        bw, m, _ = self._merged(mixed=True)
        rec = page_mask_record({'page_id': 'synthetic/mixed', 'image_size': [300, 400],
                                'halves': [[0, 300]], 'rule_lines': [m.to_dict()]})
        allm = build_rule_mask(rec['lines'], [300, 400], 'all')
        ind = build_rule_mask(rec['lines'], [300, 400], 'independent')
        self.assertGreater(int(allm[100, 100:103].sum()), 0)      # 贴字段：all 有墨
        self.assertEqual(int(allm[185, 100:103].sum()), 0)        # 断口：不置位
        self.assertEqual(int(ind[100, 100:103].sum()), 0)         # 贴字段：独立掩膜不置位
        self.assertEqual(int(ind[185, 100:103].sum()), 0)         # 断口：不置位
        self.assertEqual(rec['masks']['independent']['ink_px'], 0)  # 整条歧义 → 不用
        self.assertFalse(rec['lines'][0]['excluded_from_scale'])

    def test_both_independent_still_merge_and_mask_covers_segments(self):
        bw, m, log = self._merged(mixed=False)
        self.assertEqual(m.status, 'confirmed')
        self.assertTrue(m.qualified)
        self.assertEqual(m.gap_total_px, 10)
        self.assertEqual(log[0]['gap_px'], 10)
        rec = page_mask_record({'page_id': 'synthetic/both', 'image_size': [300, 400],
                                'halves': [[0, 300]], 'rule_lines': [m.to_dict()]})
        ind = build_rule_mask(rec['lines'], [300, 400], 'independent')
        self.assertEqual(int(ind[100, 100:103].sum()), 3)         # 上段独立 → 置位
        self.assertEqual(int(ind[185, 100:103].sum()), 0)         # 断口仍留空
        self.assertEqual(int(ind[300, 100:103].sum()), 3)         # 下段独立 → 置位
        # 掩膜按段画：ink = 两段面积，不含断口 10px
        self.assertEqual(rec['masks']['independent']['ink_px'],
                         (181 - 20) * 3 + (380 - 191) * 3)

    def test_qualification_is_shared_by_all_four_consumers(self):
        # 混合线：定栏（group_rule_runs）不用、拆分（_inside_rules）不用、排除/掩膜不用
        bw, m, _ = self._merged(mixed=True)
        self.assertFalse(m.qualified)
        groups, log = group_rule_runs([m], bw, (20, 380), DEFAULT_CONFIG['rules'])
        self.assertEqual(groups, [])
        self.assertTrue(any(x.get('why') == 'candidate_count_below_min' for x in log))
        rec = validate_columns([{'bbox': [60, 20, 140, 380]}], (0, 300), (300, 400),
                               book_stats={'cols': 310, 'median_width': 60.0},
                               cfg=DEFAULT_CONFIG['columns'], rule_lines=[m.to_dict()])
        v = rec['verdicts'][0]
        self.assertEqual(v.decision, 'accept')
        self.assertEqual(v.measurements['rules_inside_unqualified'][0]['center'], 101.5)
        self.assertNotIn('rules_inside', v.measurements)


class CoordOffsetTests(unittest.TestCase):
    """非零半页偏移：候选、run 拒绝记录、回退归因三处必须同一坐标系。"""

    @classmethod
    def setUpClass(cls):
        W, H = 1024, 400
        bw = blank_bw(H, W)
        for a, b in [(40, 80), (110, 150), (180, 220)]:
            for y in range(40, 340, 40):
                bw[y:y + 28, a:b] = 255
        bw[40:380, 780:782] = 255
        cols = [{'bbox': [40, 40, 80, 340]}, {'bbox': [110, 40, 150, 340]},
                {'bbox': [180, 40, 220, 340]}]
        cls.plan = plan_page('synthetic/coord', 'synthetic', bw, (W, H),
                             [(400, 1024), (0, 380)], cols, DEFAULT_CONFIG).to_dict()
        cls.right = next(h for h in cls.plan['halves_detail'] if h['half_id'] == 'right')

    def test_audit_coords_are_original_image(self):
        self.assertEqual(self.right['crop_offset_x'], 400)
        self.assertIn('original_image', self.right['coord_system'])
        lines = [l for l in self.plan['rule_lines'] if l['half_id'] == 'right']
        self.assertEqual([l['x0'] for l in lines], [780])
        spans = [r['span'] for r in self.right['rule_runs'] if r.get('span')]
        self.assertEqual(spans, [[781.0, 781.0]])
        for r in self.right['rule_runs']:
            self.assertEqual(r['coord_system'], 'original_image')
            if r.get('x_list'):
                self.assertGreaterEqual(min(r['x_list']), 400)   # 不出现裁块坐标 381

    def test_fallback_cause_points_at_same_line(self):
        windows = (self.right['paths']['halfpage_fallback'] or {}).get('windows') or []
        hit = [w for w in windows if w['bbox'][0] <= 780 <= w['bbox'][2]]
        self.assertTrue(hit)
        for w in hit:
            self.assertEqual(w['cause']['cause'], 'rule_run_rejected')
            self.assertEqual(w['cause']['run_span'], [781.0, 781.0])
            self.assertEqual(w['cause']['coord_system'], 'original_image')

    def test_no_crop_coordinate_leakage(self):
        blob = json.dumps(self.right, ensure_ascii=False)
        self.assertNotIn('"span": [381', blob)
        self.assertNotIn('"x": 380', blob)


class PlanIntegrationTests(unittest.TestCase):
    """整页编排：真实数据上的半页边界、路径、覆盖、坐标、可复现。"""

    @classmethod
    def setUpClass(cls):
        cls.payload = load(S4 / 'search_regions.json')
        cls.pages = {p['page_id']: p for p in cls.payload['pages']}

    def test_p12_rejects_446_and_uses_fallback(self):
        page = self.pages['丹方彚编/page_12']
        right = page['column_validation']['right']
        self.assertEqual(right['accepted'], 0)
        self.assertEqual(len(right['rejected_detail']), 1)
        self.assertEqual(right['rejected_detail'][0]['width'], 446)
        right_regions = [r for r in page['regions'] if r['half_id'] == 'right']
        self.assertTrue(right_regions)
        self.assertTrue(all(r['column_range_source'] == 'halfpage_fallback'
                            for r in right_regions))
        self.assertFalse(any(r['bbox'][2] - r['bbox'][0] > 400 for r in right_regions))

    def test_p12_groups_covered(self):
        page = self.pages['丹方彚编/page_12']
        cov = page['group_coverage']
        self.assertEqual(len(cov), 11)
        self.assertTrue(all(c['status'] == 'fully_covered' for c in cov),
                        [c['status'] for c in cov])

    def test_p40_no_cross_half_region(self):
        page = self.pages['思濟堂方書/page_40']
        halves = {'right': page['halves'][0], 'left': page['halves'][1]}
        for r in page['regions']:
            hx0, hx1 = halves[r['half_id']]
            self.assertGreaterEqual(r['bbox'][0], hx0, r['region_id'])
            self.assertLessEqual(r['bbox'][2], hx1, r['region_id'])
        # v1 的两个对侧错框不再存在
        ids = {r['region_id'] for r in page['regions']}
        self.assertNotIn('思濟堂方書/page_40#right#grouped_rules#001', ids)
        self.assertNotIn('思濟堂方書/page_40#left#grouped_rules#002', ids)

    def test_all_regions_inside_their_half(self):
        for page in self.payload['pages']:
            halves = page['halves']
            for r in page['regions']:
                k = 0 if r['half_id'] in ('right', 'full') else 1
                hx0, hx1 = halves[k] if len(halves) > 1 else (0, page['image_size'][0])
                self.assertTrue(hx0 <= r['bbox'][0] < r['bbox'][2] <= hx1,
                                (r['region_id'], r['bbox'], [hx0, hx1]))
                self.assertEqual(r['half_bounds'], [hx0, hx1])

    def test_coverage_ledger_closes(self):
        for page in self.payload['pages']:
            for h in page['halves_detail']:
                c = h['coverage']
                self.assertEqual(c['should_process_px'],
                                 c['covered_px'] + c['gutter_px'] + c['uncovered_px'],
                                 (page['page_id'], h['half_id']))
                self.assertEqual(c['uncovered'],
                                 [u for u in c['uncovered'] if u['kind'] ==
                                  'uncovered_should_process'])

    def test_out_of_scope_has_reason(self):
        for page in self.payload['pages']:
            for o in page['out_of_scope']:
                self.assertEqual(o['kind'], 'out_of_scope')
                self.assertTrue(o['why'])

    def test_region_ids_unique_and_stable(self):
        for page in self.payload['pages']:
            ids = [r['region_id'] for r in page['regions']]
            self.assertEqual(len(ids), len(set(ids)), page['page_id'])
            for r in page['regions']:
                self.assertIn(page['page_id'], r['region_id'])
                self.assertIn(r['half_id'], r['region_id'])

    def test_scale_fields_separate(self):
        for page in self.payload['pages']:
            ps = page['page_scale']
            for k in ('H', 'G', 'glyph_pitch', 'page_height', 'value_origin'):
                self.assertIn(k, ps)
            if ps['confidence_status'] != 'failed':
                self.assertIsNotNone(ps['H'])
                self.assertEqual(len({ps['H'], ps['G'], ps['glyph_pitch']}), 3)

    def test_region_scale_provenance(self):
        for page in self.payload['pages']:
            for r in page['regions']:
                sc = r['scale']
                self.assertIn(sc['value_origin'],
                              ('measured', 'inherited_from_page', 'partially_inherited', 'none'))
                if sc['value_origin'] in ('inherited_from_page', 'partially_inherited'):
                    self.assertTrue(sc['inherited_fields'], r['region_id'])
                    self.assertEqual(sc['confidence_status'], 'inherited')
                    for k in sc['inherited_fields']:
                        self.assertEqual(sc[k], page['page_scale'][k])

    def test_stage5_reader(self):
        regs = regions_for_page(self.payload, '丹方彚编/page_12')
        self.assertTrue(regs)
        for k in ('page_id', 'half_id', 'region_id', 'bbox', 'column_range_source',
                  'low_confidence', 'scale', 'validation_measurements',
                  'rejection_reasons', 'transform', 'half_bounds'):
            self.assertIn(k, regs[0])
        sc = scale_for_page(self.payload, '丹方彚编/page_12')
        self.assertIn('H', sc)
        rl = rule_lines_for_page(self.payload, '思濟堂方書/page_40')
        self.assertTrue(rl)
        self.assertTrue(all('excluded_from_scale' in x for x in rl))
        self.assertTrue(all('qualified' in x and 'status' in x for x in rl))
        self.assertTrue(all(x['coord_system'] == 'original_image' for x in rl))
        au = audit_for_page(self.payload, '思濟堂方書/page_40')
        self.assertIn('coverage', au)
        with self.assertRaises(KeyError):
            regions_for_page(self.payload, 'no/such_page')

    def test_rule_mask_payload_readable(self):
        masks = load(S4 / 'rule_masks.json')
        self.assertEqual(masks['stage'], 'stage4_rule_masks')
        self.assertEqual(len(masks['pages']), len(self.payload['pages']))
        for rec in masks['pages']:
            self.assertTrue(verify_record(rec)['independent']['match'])
        self.assertEqual(masks['summary']['total'],
                         sum(len(p['rule_lines']) for p in self.payload['pages']))

    def test_path_decisions_present(self):
        p = load(S4 / 'path_decisions.json')
        self.assertTrue(p['summary']['all_pass'])
        self.assertEqual(p['before_after']['E_p40']['v1']['crossing_count'], 2)
        self.assertEqual(p['before_after']['E_p40']['v2']['crossing_count'], 0)


class ReproducibilityTests(unittest.TestCase):
    def test_same_input_same_regions(self):
        from dual_track_stage4_run import book_reference, run_page   # noqa: E402
        layouts = load(S3 / 'frozen_layouts.json')
        book_stats, glob = book_reference(layouts)
        pg = {'page_id': '思濟堂方書/page_33', 'book': '思濟堂方書', 'page': 33,
              'image_path': '古籍评测集/医家类/思濟堂方書/page_33.png',
              'reasons': ['annotated']}
        a = run_page(pg, DEFAULT_CONFIG, book_stats, glob)
        b = run_page(pg, DEFAULT_CONFIG, book_stats, glob)
        self.assertEqual([r['region_id'] for r in a['regions']],
                         [r['region_id'] for r in b['regions']])
        self.assertEqual([r['bbox'] for r in a['regions']],
                         [r['bbox'] for r in b['regions']])
        self.assertEqual(a['page_scale']['H'], b['page_scale']['H'])

    def test_content_hash_recorded(self):
        payload = load(S4 / 'search_regions.json')
        from dual_track_stage4_run import canon_sha256       # noqa: E402
        rows = [{k: v for k, v in p.items() if k != 'timing_s'} for p in payload['pages']]
        self.assertEqual(canon_sha256(rows), payload['determinism']['content_sha256'])


class UpstreamFingerprintTests(unittest.TestCase):
    def test_upstream_recorded(self):
        up = load(S4 / 'search_regions.json')['upstream']
        for k in ('stage2_annotations', 'stage3_baseline_snapshot',
                  'stage3_frozen_layouts', 'stage3_evaluation_config'):
            self.assertTrue(up[k]['exists'], k)
            self.assertEqual(len(up[k]['sha256']), 64, k)

    def test_image_hash_matches_frozen_layout(self):
        payload = load(S4 / 'search_regions.json')
        for p in payload['pages']:
            self.assertTrue(p['image_sha256_matches_frozen_layout'], p['page_id'])

    def test_config_version(self):
        cfg = load(S4 / 'search_regions.json')['config']
        self.assertEqual(cfg['version'], 'stage4-v2')
        for k in ('rules', 'columns', 'windows', 'scale', 'scope'):
            self.assertIn(k, cfg)


if __name__ == '__main__':
    unittest.main()
