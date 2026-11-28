"""第二阶段修正的单元测试：组间冲突、主字吞字、栏线判定、匹配夹具、口径字段。

跑法：
  .venv/Scripts/python.exe -m unittest tests.test_dual_track_stage2 -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D  # noqa: E402
import dual_track_stage2_build as B  # noqa: E402
import dual_track_stage2_data as S  # noqa: E402

OUT = ROOT / 'output/dual_tracks/stage2'


def g(region, page_id='书/page_12', main=None, right=None, left=None):
    return {'region': region, 'page_id': page_id, 'group_id': f'{page_id}/{region}',
            'main_bbox': main or [100, 100, 140, 200],
            'right_bbox': right or [142, 200, 160, 400],
            'left_bbox': left or [120, 200, 140, 400]}


class CrossGroupTests(unittest.TestCase):
    def test_up_down_swallow_is_detected(self):
        # 上方组轨框延伸到下方组标题上，正是 v1 的缺陷形态
        groups = [g('A', right=[142, 200, 160, 818], left=[120, 200, 140, 818]),
                  g('B', main=[120, 540, 160, 668])]
        hits = B.find_cross_group_overlaps(groups)
        self.assertTrue(hits)
        self.assertEqual({h['a'] for h in hits} | {h['b'] for h in hits}, {'A', 'B'})

    def test_edge_touch_is_not_overlap(self):
        groups = [g('A', right=[142, 200, 160, 300], left=[120, 200, 140, 300]),
                  g('B', main=[120, 300, 160, 400], right=[142, 400, 160, 500],
                    left=[120, 400, 140, 500])]
        self.assertEqual(B.find_cross_group_overlaps(groups), [])

    def test_title_intrusion_detected(self):
        groups = [g('UP', right=[142, 200, 160, 600], left=[120, 200, 140, 600]),
                  g('DOWN', main=[120, 540, 160, 668])]
        hits = B.find_title_intrusions(groups)
        # 左右两轨都压到标题 -> 两条记录
        self.assertEqual(len(hits), 2)
        self.assertEqual({h['upper'] for h in hits}, {'UP'})
        self.assertEqual({h['lower'] for h in hits}, {'DOWN'})
        self.assertEqual({h['lane'] for h in hits}, {'right_bbox', 'left_bbox'})

    def test_clean_pair_has_no_intrusion(self):
        groups = [g('UP', right=[142, 200, 160, 536], left=[120, 200, 140, 536]),
                  g('DOWN', main=[120, 540, 160, 668])]
        self.assertEqual(B.find_title_intrusions(groups), [])


class MainSwallowTests(unittest.TestCase):
    def test_one_pixel_contact_is_edge_contact(self):
        grp = g('A', main=[100, 100, 140, 300], right=[142, 299, 160, 400],
                left=[120, 299, 140, 400])
        res = B.find_main_swallow([grp])
        self.assertEqual(res['swallowed'], [])
        self.assertTrue(all(c['verdict'] == 'edge_contact' for c in res['contacts']))

    def test_deep_swallow_fails(self):
        grp = g('A', main=[100, 100, 140, 320], right=[142, 300, 160, 400],
                left=[120, 300, 140, 400])
        res = B.find_main_swallow([grp])
        self.assertTrue(res['swallowed'])


class RuleOverlapTests(unittest.TestCase):
    def test_edge_rule_is_edge_coincidence(self):
        m = B.classify_rule_overlap([637, 306, 661, 557], [637, 204, 640, 820])
        self.assertEqual(m['verdict'], 'edge_coincidence')
        self.assertLess(m['overlap_width_px'], 10)
        self.assertGreater(m['height_coverage'], 0.9)

    def test_wide_rule_overlap_is_contamination(self):
        m = B.classify_rule_overlap([600, 300, 700, 600], [600, 300, 660, 600])
        self.assertEqual(m['verdict'], 'contamination')
        self.assertGreater(m['area_ratio'], 0.25)

    def test_no_overlap_returns_none(self):
        self.assertIsNone(B.classify_rule_overlap([0, 0, 10, 10], [20, 0, 24, 10]))


class MatchingTests(unittest.TestCase):
    def test_case_a_max_cardinality(self):
        r = B.solve_matching([[0.90, 0.80], [0.80, 0.10]], 0.5)
        self.assertEqual(r['cardinality'], 2)
        self.assertEqual(r['pairs'], [[0, 1], [1, 0]])

    def test_case_b_total_iou(self):
        r = B.solve_matching([[0.90, 0.80], [0.85, 0.50]], 0.5)
        self.assertEqual(r['pairs'], [[0, 1], [1, 0]])
        self.assertAlmostEqual(r['total_iou'], 1.65, places=9)

    def test_near_tie_keeps_precision(self):
        r = B.solve_matching([[0.5000001, 0.5000002]], 0.5)
        self.assertEqual(r['pairs'], [[0, 1]])
        self.assertAlmostEqual(r['total_iou'], 0.5000002, places=12)

    def test_rounding_would_have_created_tie(self):
        m = [[0.5004, 0.5006]]
        r = B.solve_matching(m, 0.5)
        self.assertEqual(r['pairs'], [[0, 1]])
        # 若先舍入到 2 位，两值都变 0.50 -> 会错误地按字典序选 G1
        self.assertEqual(round(m[0][0], 2), round(m[0][1], 2))

    def test_threshold_equal_counts(self):
        self.assertEqual(B.solve_matching([[0.50]], 0.5)['cardinality'], 1)
        self.assertEqual(B.solve_matching([[0.4999999999]], 0.5)['cardinality'], 0)

    def test_no_edge(self):
        self.assertEqual(B.solve_matching([[0.2, 0.1]], 0.5)['cardinality'], 0)

    def test_tie_break_lexicographic(self):
        r = B.solve_matching([[0.80, 0.80], [0.80, 0.80]], 0.5)
        self.assertEqual(r['pairs'], [[0, 0], [1, 1]])

    def test_unequal_counts(self):
        r = B.solve_matching([[0.95, 0.60], [0.70, 0.90], [0.80, 0.85]], 0.5)
        self.assertEqual(r['cardinality'], 2)
        self.assertEqual(r['pairs'], [[0, 0], [1, 1]])


class DiagnosticCountTests(unittest.TestCase):
    def test_duplicate_gives_one_tp_one_fp(self):
        c = B._reference_counts([[0.90], [0.85]], 0.5)
        self.assertEqual((c['tp'], c['fp'], c['fn']), (1, 1, 0))

    def test_invalid_pred_kept_as_fp(self):
        c = B._reference_counts([[0.95], None], 0.5)
        self.assertEqual((c['tp'], c['fp'], c['fn']), (1, 1, 0))
        self.assertEqual(c['invalid_predictions'], 1)

    def test_fragment_no_tp(self):
        c = B._reference_counts([[0.40], [0.35], [0.30]], 0.5)
        self.assertEqual((c['tp'], c['fp'], c['fn']), (0, 3, 1))

    def test_lane_category(self):
        self.assertEqual(B._lane_category(
            {'right_iou_fixed': 0.7, 'left_iou_fixed': 0.6,
             'right_pred_vs_gt_left': 0, 'left_pred_vs_gt_right': 0}), 'pairing_success')
        self.assertEqual(B._lane_category(
            {'right_iou_fixed': 0.2, 'left_iou_fixed': 0.2,
             'right_pred_vs_gt_left': 0.7, 'left_pred_vs_gt_right': 0.7}), 'left_right_swap')
        self.assertEqual(B._lane_category(
            {'right_iou_fixed': 0.6, 'left_iou_fixed': 0.3,
             'right_pred_vs_gt_left': 0.4, 'left_pred_vs_gt_right': 0.3}),
            'child_localization_failure')


class RealDataTests(unittest.TestCase):
    """真实产物的口径断言：这些字段变了说明有人把旧语义写回去了。"""

    @classmethod
    def setUpClass(cls):
        cls.ann = json.loads((OUT / 'annotations.json').read_text(encoding='utf-8'))
        cls.neg = json.loads((OUT / 'negative_regions.json').read_text(encoding='utf-8'))
        cls.proto = json.loads((OUT / 'evaluation_protocol.json').read_text(encoding='utf-8'))
        cls.contract = json.loads((OUT / 'cross_stage_contract.json').read_text(encoding='utf-8'))
        cls.splits = json.loads((OUT / 'dataset_splits.json').read_text(encoding='utf-8'))
        cls.report = json.loads((OUT / 'validation_report.json').read_text(encoding='utf-8'))
        cls.cases = json.loads((OUT / 'matching_cases.json').read_text(encoding='utf-8'))

    def test_no_cross_group_overlap_on_real_data(self):
        self.assertEqual(B.find_cross_group_overlaps(self.ann['groups']), [])

    def test_no_title_intrusion_on_real_data(self):
        self.assertEqual(B.find_title_intrusions(self.ann['groups']), [])

    def test_df12_top_lanes_end_before_lower_titles(self):
        gs = {x['region']: x for x in self.ann['groups'] if x['book'] == S.DF_BOOK}
        pairs = [('DF12-03', 'DF12-11'), ('DF12-04', 'DF12-10'), ('DF12-05', 'DF12-09')]
        for up, low in pairs:
            self.assertLessEqual(gs[up]['left_bbox'][3], gs[low]['main_bbox'][1],
                                 f'{up} 左轨压到 {low} 标题')
            self.assertLessEqual(gs[up]['right_bbox'][3], gs[low]['main_bbox'][1],
                                 f'{up} 右轨压到 {low} 标题')

    def test_wide_box_labels_all_unknown(self):
        for p in self.ann['pages']:
            wb = p['labels']['wide_box']
            self.assertEqual(wb['state'], 'unknown', p['page_id'])
            self.assertIn('旧布局', wb['measured_object'])
            self.assertIsNone(wb['layout_snapshot_id'])

    def test_scale_object_semantics(self):
        spec = self.contract['scale_object']['field_spec']
        self.assertIn('正文中位字高', spec['H']['meaning'])
        self.assertIn('留白', spec['G']['meaning'])
        self.assertIn('字心', spec['glyph_pitch']['meaning'])
        self.assertIn('page_height', spec)
        self.assertIn('图像高度', spec['page_height']['meaning'])
        for f in ('small_glyph_height', 'small_glyph_width'):
            self.assertIn(f, spec)
        self.assertNotIn('fallback_default', spec['source']['values'])

    def test_seam_profile_normalized(self):
        cfg = self.contract['seam_profile']['config']
        self.assertIn('0/1', cfg['ink_mask']['representation'])
        self.assertIn('平均', cfg['vertical_projection'])
        self.assertEqual(cfg['epsilon']['unit'], 'ratio')
        self.assertEqual(cfg['min_peak']['unit'], 'ratio')
        self.assertLess(cfg['min_peak']['value'], 1.0)
        self.assertLess(cfg['epsilon']['value'], 1.0)

    def test_coverage_not_computable_flags(self):
        mt = self.proto['coverage_reporting']['metric_table']
        self.assertFalse(mt['recall']['computable'])
        self.assertFalse(mt['precision']['computable'])
        self.assertEqual(self.proto['coverage_reporting']['p12_specific']['whole_page_recall'],
                         'not_computable_current_version')

    def test_run_state_decision_table_covers_combos(self):
        rows = self.proto['run_states']['page_state_decision_table']
        text = json.dumps(rows, ensure_ascii=False)
        for kw in ('[] 空列表', 'skipped', 'failed', 'partial', 'not_run', '功能关闭',
                   '且无组', '无区域'):
            self.assertIn(kw, text)

    def test_matching_cases_all_verified(self):
        self.assertTrue(self.cases['all_expected_verified'])
        ids = {c['case_id'] for c in self.cases['cases']}
        for cid in ('CASE-A', 'CASE-B', 'CASE-G-NEAR-TIE', 'CASE-H-PRECISION-VS-ROUND'):
            self.assertIn(cid, ids)

    def test_validation_all_pass(self):
        self.assertTrue(self.report['summary']['ok'], self.report['summary'])

    def test_independent_claim_marked(self):
        ic = self.splits['independent_claim']
        self.assertEqual(ic['status'], 'exposure_history_unclear')
        self.assertIn('不得声称完全盲测', ic['what_it_supports'])

    def test_no_sample_interval_claim(self):
        ss = self.splits['sample_sufficiency']
        self.assertEqual(ss['candidate_target']['status'], 'candidate_target')
        self.assertNotIn('保证', json.dumps(ss, ensure_ascii=False))



class GlyphReviewTests(unittest.TestCase):
    """字形归属核对：几何合法 ≠ 字形正确。"""

    @classmethod
    def setUpClass(cls):
        cls.ann = json.loads((OUT / 'annotations.json').read_text(encoding='utf-8'))

    def test_every_new_group_has_five_checks(self):
        for g in self.ann['groups']:
            if g['origin'] == 'stage1':
                continue
            gr = g.get('glyph_review') or {}
            for k in ('main_last_glyph', 'left_first_glyph', 'right_first_glyph',
                      'left_last_glyph', 'right_last_glyph', 'review_status',
                      'evidence_paths', 'review_notes'):
                self.assertIn(k, gr, f"{g['group_id']} 缺 {k}")

    def test_first_glyphs_declared_complete(self):
        for g in self.ann['groups']:
            if g['origin'] == 'stage1':
                continue
            gr = g['glyph_review']
            for k in ('main_last_glyph', 'left_first_glyph', 'right_first_glyph'):
                self.assertTrue(gr[k].get('glyph_complete'), f"{g['group_id']}.{k}")

    def test_tails_not_cut(self):
        for g in self.ann['groups']:
            if g['origin'] == 'stage1':
                continue
            gr = g['glyph_review']
            for k in ('left_last_glyph', 'right_last_glyph'):
                self.assertFalse(gr[k]['cut'], f"{g['group_id']}.{k} 末笔被切")

    def test_df12_03_main_stops_at_last_glyph(self):
        g = [x for x in self.ann['groups'] if x['region'] == 'DF12-03'][0]
        self.assertEqual(g['main_bbox'][3], 275)
        self.assertGreaterEqual(g['left_bbox'][1], g['main_bbox'][3])
        self.assertGreaterEqual(g['right_bbox'][1], g['main_bbox'][3])
        self.assertEqual(g['glyph_review']['main_last_glyph']['text'], '母')

    def test_geometry_capability_declared(self):
        rep = json.loads((OUT / 'validation_report.json').read_text(encoding='utf-8'))
        v16 = [c for c in rep['checks'] if c['id'] == 'V16'][0]
        self.assertEqual(v16['capability'], 'geometry_only')
        self.assertIn('不能证明', v16['detail'])


class ScaleSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads((OUT / 'cross_stage_contract.json').read_text(encoding='utf-8'))

    def test_fields_are_separate(self):
        spec = self.c['scale_object']['field_spec']
        self.assertIn('正文中位字高', spec['H']['meaning'])
        self.assertIn('留白', spec['G']['meaning'])
        self.assertIn('字心距离', spec['glyph_pitch']['meaning'])
        for f in ('small_glyph_height', 'small_glyph_width', 'small_glyph_gap',
                  'small_glyph_pitch', 'page_height'):
            self.assertIn(f, spec)

    def test_synthetic_example(self):
        ex = self.c['scale_object']['synthetic_example']
        self.assertEqual(ex['expected']['H'], 30)
        self.assertEqual(ex['expected']['G'], 5)
        self.assertEqual(ex['expected']['glyph_pitch'], 35)

    def test_no_undefined_local_scale(self):
        txt = json.dumps(self.c, ensure_ascii=False)
        self.assertNotIn('G_local', txt)
        self.assertNotIn('scale_for_band 用局部', txt)

    def test_seam_uses_named_fields(self):
        cfg = self.c['seam_profile']['config']
        self.assertIn('glyph_pitch', cfg['smoothing']['window_px'])
        self.assertIn('small_glyph_width', cfg['left_right_search_band']['scale_for_band'])
        self.assertIn('vertical_stop_condition', cfg)
        self.assertIn('window_height_and_step', cfg)


class PageStateTests(unittest.TestCase):
    def test_all_fixtures(self):
        fx = B.page_state_fixtures()
        self.assertTrue(fx['all_ok'])
        bad = [c['id'] for c in fx['cases'] if not c['ok']]
        self.assertEqual(bad, [])

    def test_group_count_invariance(self):
        fx = B.page_state_fixtures()
        self.assertTrue(fx['group_count_invariance']['ok'])
        states = {s['page_state'] for s in fx['group_count_invariance']['samples']}
        self.assertEqual(len(states), 1)

    def test_completed_with_zero_groups_is_completed(self):
        r = B.aggregate_page_state([{'region_id': 'r', 'in_scope': True, 'status': 'completed'}])
        self.assertEqual(r['page_state'], 'completed')
        self.assertEqual(r['dual_track_evidence'], [])

    def test_out_of_scope_skip_does_not_degrade(self):
        r = B.aggregate_page_state([
            {'region_id': 'a', 'in_scope': True, 'status': 'completed'},
            {'region_id': 'b', 'in_scope': False, 'status': 'skipped', 'skip_reason': 'out_of_scope'}])
        self.assertEqual(r['page_state'], 'completed')

    def test_incomplete_skip_degrades(self):
        r = B.aggregate_page_state([
            {'region_id': 'a', 'in_scope': True, 'status': 'completed'},
            {'region_id': 'b', 'in_scope': True, 'status': 'skipped', 'skip_reason': 'incomplete'}])
        self.assertEqual(r['page_state'], 'partial')

    def test_region_division_failure_is_failed(self):
        r = B.aggregate_page_state([], region_division_failed=True)
        self.assertEqual(r['page_state'], 'failed')
        self.assertIsNone(r['dual_track_evidence'])

    def test_protocol_includes_fixtures(self):
        proto = json.loads((OUT / 'evaluation_protocol.json').read_text(encoding='utf-8'))
        self.assertIn('fixtures', proto['run_states'])
        self.assertTrue(proto['run_states']['fixtures']['all_ok'])


if __name__ == '__main__':
    unittest.main()
