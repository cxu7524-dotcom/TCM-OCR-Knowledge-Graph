# -*- coding: utf-8 -*-
"""双轨评测器测试：几何解析、匹配三层目标、轨级分类、诊断、区域归属、评分。

重点是三件容易错的事：
  1. 「近似分数不算并列」——0.5000000001 与 0.5000000002 必须选后者；
  2. 「区域归属在匹配之前」——被 ignore 的预测不得占用标注；
  3. 「穷尽区未匹配就是误检」——boundary_conflict 是诊断，不是豁免。
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_eval import (  # noqa: E402
    BboxError, RegionTable, bbox_area, classify_invalid, classify_lane_pairing,
    duplicate_clusters, endpoint_sensitivity, iou, match_groups, negative_coverage_ratio,
    parse_bbox, score_predictions, solve_exhaustive, solve_min_cost_max_flow, union_bbox,
)
from dual_track_eval.matching import _best_exact_search  # noqa: E402
from dual_track_eval.regions import BASE_STATUSES, PROVISIONAL  # noqa: E402

OUT2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'


def gt(gid, box, page='p', certainty='确认', endpoint='pending'):
    return {'group_id': gid, 'page_id': page, 'group_bbox': box,
            'right_bbox': [box[2] - 5, box[1], box[2], box[3]],
            'left_bbox': [box[0], box[1], box[0] + 5, box[3]],
            'certainty': certainty,
            'endpoint_interpretation': {'status': endpoint}}


def pred(pid, box, page='p'):
    return {'prediction_id': pid, 'page_id': page, 'group_bbox': box,
            'right_bbox': [box[2] - 5, box[1], box[2], box[3]],
            'left_bbox': [box[0], box[1], box[0] + 5, box[3]]}


class GeometryTests(unittest.TestCase):
    def test_parse_accepts_ints_and_integral_float(self):
        self.assertEqual(parse_bbox([0, 0, 10, 20]), (0, 0, 10, 20))
        self.assertEqual(parse_bbox((1.0, 2.0, 3.0, 4.0)), (1, 2, 3, 4))
        self.assertEqual(parse_bbox(['1', ' 2 ', '3', '4']), (1, 2, 3, 4))

    def test_parse_rejects(self):
        cases = {
            'non_finite': [0, 0, float('inf'), 10],
            'bool_element': [True, 0, 5, 5],
            'non_integer_float': [0.5, 0, 5, 5],
            'wrong_length': [1, 2, 3],
            'not_sequence': None,
            'unparsable_string': ['a', 1, 2, 3],
        }
        for code, value in cases.items():
            with self.assertRaises(BboxError) as cm:
                parse_bbox(value)
            self.assertEqual(cm.exception.code, code, msg=repr(value))
        with self.assertRaises(BboxError) as cm:
            parse_bbox([0, 0, float('-inf'), 1])
        self.assertEqual(cm.exception.code, 'non_finite')

    def test_iou_half_open(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)
        self.assertEqual(iou([0, 0, 0, 10], [0, 0, 10, 10]), 0.0)

    def test_union_bbox(self):
        self.assertEqual(union_bbox([[5, 1, 6, 2], [1, 3, 2, 4]]), (1, 1, 6, 4))
        self.assertIsNone(union_bbox([]))

    def test_union_first_ratio_capped_at_one(self):
        self.assertAlmostEqual(negative_coverage_ratio([0, 0, 10, 10],
                                                      [[0, 0, 10, 6], [0, 4, 10, 10]]), 1.0)

    def test_union_ratio_partial(self):
        self.assertAlmostEqual(negative_coverage_ratio([0, 0, 10, 10], [[0, 0, 5, 10]]), 0.5)


class MatchingFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((OUT2 / 'matching_cases.json').read_text(encoding='utf-8'))

    def test_all_stage2_fixtures(self):
        for c in self.cases['cases']:
            got = match_groups(c['iou_matrix'], c['threshold'])
            self.assertEqual(got['cardinality'], c['expected']['cardinality'], c['case_id'])
            self.assertEqual(got['pairs'], c['expected']['pairs'], c['case_id'])
            exp_total = c['expected'].get('total_iou')
            if exp_total is not None:
                self.assertAlmostEqual(got['total_iou'], exp_total, places=6, msg=c['case_id'])

    def test_case_a_prefers_global_optimum(self):
        got = match_groups([[0.90, 0.80], [0.80, 0.10]], 0.5)
        self.assertEqual(got['pairs'], [[0, 1], [1, 0]])
        self.assertAlmostEqual(got['total_iou'], 1.60, places=6)

    def test_case_g_near_tie_uses_second_edge(self):
        got = match_groups([[0.5000001, 0.5000002]], 0.5)
        self.assertEqual(got['pairs'], [[0, 1]])

    def test_case_f_threshold_boundary(self):
        self.assertEqual(match_groups([[0.50]], 0.5)['cardinality'], 1)
        self.assertEqual(match_groups([[0.4999999999]], 0.5)['cardinality'], 0)

    def test_lane_fixtures(self):
        for c in self.cases['lane_pairing_cases']:
            got = classify_lane_pairing(
                {'right_iou_fixed': c['right_iou_fixed'], 'left_iou_fixed': c['left_iou_fixed'],
                 'right_pred_vs_gt_left_iou': c['right_pred_vs_gt_left'],
                 'left_pred_vs_gt_right_iou': c['left_pred_vs_gt_right']})
            self.assertEqual(got['category'], c['expected'], c['case_id'])


class NearTiePrecisionTests(unittest.TestCase):
    """本轮缺陷 1：TOL 把不同总分当并列，选中低分解但报高总分。"""

    def test_gap_of_1e_10_not_a_tie(self):
        m = [[0.5000000001, 0.5000000002]]
        r = match_groups(m, 0.5)
        self.assertEqual(r['pairs'], [[0, 1]])
        self.assertEqual(r['total_iou'], 0.5000000002)
        self.assertEqual(r['tie_break_status'], 'exact')

    def test_returned_total_equals_pairs_sum(self):
        cases = [
            [[0.5000000001, 0.5000000002]],
            [[0.91, 0.9000000001], [0.9000000002, 0.9]],
            [[0.7, 0.700000000001], [0.700000000002, 0.7]],
        ]
        for m in cases:
            r = match_groups(m, 0.5)
            recomputed = sum(m[i][j] for i, j in r['pairs'])
            self.assertEqual(r['total_iou'], recomputed, msg=str(m))
            self.assertAlmostEqual(r['total_iou'], recomputed, places=15)

    def test_cumulative_near_ties(self):
        # 两条边各自只差 1e-12，累加后仍应选总分更高的一组
        m = [[0.6, 0.600000000001], [0.600000000001, 0.6]]
        r = match_groups(m, 0.5)
        self.assertAlmostEqual(r['total_iou'], 1.2 + 2e-12, places=15)
        recomputed = sum(m[i][j] for i, j in r['pairs'])
        self.assertEqual(r['total_iou'], recomputed)

    def test_exact_tie_uses_lexicographic(self):
        r = match_groups([[0.9, 0.9], [0.9, 0.9]], 0.5)
        self.assertEqual(r['pairs'], [[0, 0], [1, 1]])
        self.assertEqual(r['total_iou'], 1.8)

    def test_seed_improvement_is_reported(self):
        # seed 由最小费用流给出；精确搜索若发现更高分，要标记出来
        exact = _best_exact_search([[0.5, 0.6]], 0.5, 1, 0.5)
        self.assertEqual(exact['status'], 'exact')
        self.assertAlmostEqual(exact['best_total'], 0.6, places=12)
        self.assertTrue(exact['improved_over_seed'])


class ScaleLimitTests(unittest.TestCase):
    def test_scale_limit_is_declared_not_silent(self):
        # 21 × 30 = 630 > 600 → 兜底搜索不执行，但必须显式说明
        n, m = 21, 30
        matrix = [[0.6] * m for _ in range(n)]
        r = match_groups(matrix, 0.5)
        self.assertEqual(r['tie_break_status'], 'skipped_scale_limit')
        self.assertFalse(r['tie_break_scope']['applied'])
        self.assertIn('不得声称第三目标已满足', r['tie_break_scope']['consequence'])
        # 基数与总 IoU 仍然正确
        self.assertEqual(r['cardinality'], min(n, m))
        self.assertAlmostEqual(r['total_iou'], 0.6 * min(n, m), places=9)

    def test_scale_limit_k(self):
        n, m = 5, 14      # n*m=70 ≤ 600，但 k=5 > 12? 不超；换成 k 超限的规模
        matrix = [[0.9] * m for _ in range(n)]
        r = match_groups(matrix, 0.5)
        self.assertEqual(r['tie_break_status'], 'exact')
        self.assertEqual(r['cardinality'], 5)


class MatchingImplementationTests(unittest.TestCase):
    def test_empty_matches(self):
        r = match_groups([], 0.5)
        self.assertEqual(r['cardinality'], 0)
        self.assertEqual(r['pairs'], [])
        self.assertEqual(r['tie_break_status'], 'no_match')

    def test_no_valid_edge(self):
        self.assertEqual(match_groups([[0.1, 0.2]], 0.5)['cardinality'], 0)

    def test_unequal_counts(self):
        self.assertEqual(match_groups([[0.9, 0.8], [0.7, 0.6], [0.65, 0.55]], 0.5)['cardinality'], 2)

    def test_max_cardinality_beats_greedy(self):
        r = match_groups([[0.99, 0.0], [0.80, 0.85]], 0.5)
        self.assertEqual(r['cardinality'], 2)
        self.assertEqual(r['pairs'], [[0, 0], [1, 1]])

    def test_stable_tie_break(self):
        a = match_groups([[0.9, 0.9], [0.9, 0.9]], 0.5)
        b = match_groups([[0.9, 0.9], [0.9, 0.9]], 0.5)
        self.assertEqual(a['pairs'], b['pairs'])

    def test_random_small_matrices_agree_with_exhaustive(self):
        rnd = random.Random(20261124)
        for trial in range(60):
            n, m = rnd.randint(1, 4), rnd.randint(1, 4)
            matrix = [[round(rnd.choice([0.0, 0.25, 0.5, 0.49, 0.51, 0.6, 0.75, 0.8, 0.9,
                                         rnd.random()]), 7) for _ in range(m)] for _ in range(n)]
            for thr in (0.5, 0.75):
                fast = match_groups(matrix, thr)
                ref = solve_exhaustive(matrix, thr)
                ctx = f'trial={trial} thr={thr} matrix={matrix}'
                self.assertEqual(fast['cardinality'], ref['cardinality'], ctx)
                self.assertAlmostEqual(fast['total_iou'], ref['total_iou'], places=9, msg=ctx)
                self.assertEqual(fast['pairs'], ref['pairs'], ctx)

    def test_mcmf_and_exact_agree_on_cardinality(self):
        m = [[0.5, 0.9], [0.9, 0.5]]
        self.assertEqual(solve_min_cost_max_flow(m, 0.5)['cardinality'],
                         match_groups(m, 0.5)['cardinality'])


class ThresholdIndependenceTests(unittest.TestCase):
    """本轮缺陷 5：0.75 的匹配集合不必是 0.5 的子集。"""

    MATRIX = [[0.65, 0.74, 0.76],
              [0.76, 0.95, 0.00],
              [0.00, 0.76, 0.74],
              [0.65, 0.00, 0.85]]

    def test_reference_counterexample(self):
        lo = match_groups(self.MATRIX, 0.5)
        hi = match_groups(self.MATRIX, 0.75)
        self.assertEqual(lo['pairs'], [[0, 0], [1, 1], [3, 2]])
        self.assertAlmostEqual(lo['total_iou'], 2.45, places=9)
        self.assertEqual(hi['pairs'], [[1, 0], [2, 1], [3, 2]])
        self.assertAlmostEqual(hi['total_iou'], 2.37, places=9)

    def test_high_threshold_may_use_other_predictions(self):
        lo_preds = {i for i, _ in match_groups(self.MATRIX, 0.5)['pairs']}
        hi_preds = {i for i, _ in match_groups(self.MATRIX, 0.75)['pairs']}
        self.assertFalse(hi_preds <= lo_preds)
        self.assertEqual(hi_preds - lo_preds, {2})

    def test_each_edge_meets_its_threshold(self):
        for thr in (0.5, 0.75):
            for i, j in match_groups(self.MATRIX, thr)['pairs']:
                self.assertGreaterEqual(self.MATRIX[i][j], thr, (thr, i, j))

    def test_nodes_used_once(self):
        for thr in (0.5, 0.75):
            pairs = match_groups(self.MATRIX, thr)['pairs']
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))

    def test_high_threshold_cardinality_not_greater(self):
        self.assertLessEqual(match_groups(self.MATRIX, 0.75)['cardinality'],
                             match_groups(self.MATRIX, 0.5)['cardinality'])

    def test_total_matches_pairs(self):
        for thr in (0.5, 0.75):
            r = match_groups(self.MATRIX, thr)
            self.assertAlmostEqual(r['total_iou'],
                                   sum(self.MATRIX[i][j] for i, j in r['pairs']), places=12)


class DiagnosticTests(unittest.TestCase):
    def test_duplicate_marks_extra_without_flipping_tp(self):
        dups = duplicate_clusters([[0.9, 0.0], [0.85, 0.0]], 0.5, [[0, 0]])
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]['extra_preds'], [1])
        self.assertEqual(dups[0]['matched_pred'], 0)

    def test_extra_pred_already_matched_elsewhere_is_flagged(self):
        dups = duplicate_clusters([[0.9, 0.0], [0.85, 0.88]], 0.5, [[0, 0], [1, 1]])
        self.assertEqual(dups[0]['extra_preds_already_matched_elsewhere'], [1])

    def test_merge_detects_one_pred_two_gts(self):
        from dual_track_eval.diagnostics import merge_clusters
        out = merge_clusters([[0, 0, 20, 60]], [[0, 0, 20, 25], [0, 30, 20, 60]])
        self.assertEqual(len(out), 1)

    def test_fragment_detects_split_coverage(self):
        from dual_track_eval.diagnostics import fragment_clusters
        out = fragment_clusters([[0.1], [0.1]], [[0, 0, 20, 15], [0, 15, 20, 30]],
                                [[0, 0, 20, 30]], [])
        self.assertEqual(len(out), 1)


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.neg = [{'negative_region_id': 'N1', 'page_id': 'p', 'bbox': [0, 0, 10, 100],
                     'category': 'body_text', 'status': 'confirmed'},
                    {'negative_region_id': 'N2', 'page_id': 'p', 'bbox': [12, 0, 14, 100],
                     'category': 'rule', 'status': 'unknown'}]
        self.amb = [{'ambiguous_region_id': 'A1', 'page_id': 'p', 'bbox': [50, 0, 60, 100]}]
        self.ign = [{'ignore_region_id': 'I1', 'page_id': 'p', 'bbox': [90, 0, 100, 100]}]

    def test_ignore_wins_in_precheck(self):
        t = RegionTable('p', self.neg, self.amb, self.ign)
        self.assertEqual(t.precheck([92, 10, 98, 90])['excluded_by'], 'ignore')

    def test_ambiguous_not_excluded_at_precheck(self):
        t = RegionTable('p', self.neg, self.amb, self.ign)
        pre = t.precheck([52, 10, 58, 90])
        self.assertIsNone(pre['excluded_by'])
        self.assertTrue(pre['in_ambiguous'])

    def test_positive_beats_ambiguous_mask(self):
        t = RegionTable('p', self.neg, self.amb, self.ign)
        r = t.attribute([52, 10, 58, 90], matched=True)
        self.assertEqual(r['base_status'], 'TP')

    def test_confirmed_negative_fp(self):
        t = RegionTable('p', self.neg)
        r = t.attribute([0, 0, 10, 100])
        self.assertEqual(r['base_status'], 'FP')
        self.assertEqual(r['review_category'], 'confirmed_false_positive')

    def test_unknown_negative_not_used(self):
        t = RegionTable('p', self.neg)
        r = t.attribute([12, 0, 14, 100])
        self.assertEqual(r['negative_region_ids'], [])
        self.assertEqual(r['base_status'], 'PENDING_CANDIDATE')

    def test_thresholds_are_provisional(self):
        self.assertEqual(PROVISIONAL['status'], 'pending_calibration')


class RegionBeforeMatchTests(unittest.TestCase):
    """本轮缺陷 2：ignore 预测不得占用标注（tp=1 且 ignored=1 是错的）。"""

    GTS = [gt('G1', [0, 0, 10, 20])]
    PREDS = [pred('P1', [0, 0, 20, 20])]
    IGN = [{'ignore_region_id': 'I1', 'page_id': 'p', 'bbox': [10, 0, 20, 20]}]

    def test_user_counterexample(self):
        t = RegionTable('p', negatives=[], ambiguous=[], ignores=self.IGN)
        # 标注与 ignore 相交面积为 0，预测中心落 ignore，预测与标注 IoU=0.5
        from dual_track_eval.geometry import intersect_area, iou
        self.assertEqual(intersect_area([0, 0, 10, 20], [10, 0, 20, 20]), 0)
        self.assertAlmostEqual(iou([0, 0, 20, 20], [0, 0, 10, 20]), 0.5)
        res = score_predictions(self.PREDS, self.GTS, {'p': t})
        x = res['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['tp'], 0)
        self.assertEqual(x['ignored'], 1)
        self.assertEqual(x['fp'], 0)
        self.assertEqual(x['group_hits'], 0)

    def test_ignored_not_tp_and_baseline_status_unique(self):
        t = RegionTable('p', negatives=[], ambiguous=[], ignores=self.IGN)
        res = score_predictions(self.PREDS, self.GTS, {'p': t})
        row = res['pages'][0]['thresholds']['0.5']['predictions'][0]
        self.assertEqual(row['base_status'], 'IGNORED')
        self.assertFalse(row['counts_as_fp'])
        self.assertIn(row['base_status'], BASE_STATUSES)

    def test_status_partition_covers_all_predictions(self):
        t = RegionTable('p', negatives=[], ambiguous=[], ignores=self.IGN)
        preds = self.PREDS + [pred('P2', [200, 0, 220, 20]),
                              {'prediction_id': 'BAD', 'page_id': 'p',
                               'group_bbox': [0, 0, float('inf'), 1]}]
        res = score_predictions(preds, self.GTS, {'p': t})
        page = res['pages'][0]
        t5 = page['thresholds']['0.5']
        self.assertTrue(t5['status_partition_ok'])
        self.assertEqual(t5['status_sum'], page['counts']['valid_predictions'])

    def test_ambiguous_pending_when_no_positive(self):
        t = RegionTable('p', negatives=[], ambiguous=[{'ambiguous_region_id': 'A1',
                                                       'page_id': 'p', 'bbox': [200, 0, 220, 20]}])
        res = score_predictions([pred('P1', [200, 0, 220, 20])], self.GTS, {'p': t})
        x = res['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['pending_ambiguous'], 1)
        self.assertEqual(x['tp'], 0)
        self.assertEqual(x['fp'], 0)


class CountingTests(unittest.TestCase):
    """本轮缺陷 3：穷尽区未匹配必须计 FP，诊断不能豁免。"""

    def test_exhaustive_boundary_conflict_still_fp(self):
        t = RegionTable('p',
                        negatives=[{'negative_region_id': 'N1', 'page_id': 'p',
                                    'bbox': [0, 0, 1, 20], 'category': 'rule',
                                    'status': 'confirmed'}],
                        coverage='exhaustive_page')
        res = score_predictions([pred('P1', [0, 0, 20, 20])], [], {'p': t})
        x = res['pages'][0]['thresholds']['0.5']
        row = x['predictions'][0]
        self.assertEqual(row['review_category'], 'boundary_conflict')
        self.assertEqual(row['base_status'], 'FP')
        self.assertTrue(row['counts_as_fp'])
        self.assertEqual(x['fp'], 1)
        self.assertEqual(x['boundary_conflict_fp'], 1)

    def test_exhaustive_unmatched_without_negative_is_fp(self):
        t = RegionTable('p', coverage='exhaustive_page')
        x = score_predictions([pred('P1', [0, 0, 20, 20])], [], {'p': t})['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['fp'], 1)
        self.assertEqual(x['predictions'][0]['review_category'], 'unmatched_uncovered')

    def test_partial_unmatched_is_candidate(self):
        t = RegionTable('p', coverage='partial_positions')
        x = score_predictions([pred('P1', [0, 0, 20, 20])], [], {'p': t})['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['fp'], 0)
        self.assertEqual(x['new_candidates'], 1)

    def test_negative_only_unmatched_is_candidate(self):
        t = RegionTable('p', negatives=[{'negative_region_id': 'N1', 'page_id': 'p',
                                         'bbox': [0, 0, 1, 20], 'category': 'rule',
                                         'status': 'confirmed'}],
                        coverage='negative_only')
        x = score_predictions([pred('P1', [0, 0, 20, 20])], [], {'p': t})['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['fp'], 0)
        self.assertEqual(x['new_candidates'], 1)

    def test_invalid_prediction_classified_and_counted(self):
        t = RegionTable('p')
        bad = [{'prediction_id': 'B1', 'page_id': 'p', 'group_bbox': None,
                'right_bbox': [1, 2, 3, 4], 'left_bbox': None},
               {'prediction_id': 'B2', 'page_id': 'p', 'group_bbox': [0, 0, 10, 10],
                'right_bbox': [5, 0, 10, 10], 'left_bbox': [0, 0, 5, 10]}]
        res = score_predictions(bad, [], {'p': t}, page_sizes={'p': [8, 8]})
        page = res['pages'][0]
        self.assertEqual(page['counts']['invalid_predictions'], 1)
        issues = set(page['invalid'][0]['issues'])
        self.assertIn('invalid_group_bbox', issues)
        self.assertIn('missing_or_invalid_left_lane', issues)
        self.assertTrue(res['invalid']['count'] >= 1)
        self.assertIn('out_of_image', res['summary']['overall']['0.5']['invalid_by_issue'])
        # 越界框有组框 → 属 degraded，仍参与匹配但被标记
        self.assertEqual(res['invalid']['degraded'], 1)

    def test_invalid_unlocatable_policy(self):
        v, inv = __import__('dual_track_eval').normalize_predictions(
            [{'prediction_id': 'X', 'page_id': 'p', 'group_bbox': None,
              'right_bbox': None, 'left_bbox': None}])
        self.assertEqual(len(v), 0)
        self.assertEqual(len(inv), 1)
        self.assertIn('invalid_group_bbox', inv[0]['_issues'])


class ConfigDrivenTests(unittest.TestCase):
    """本轮缺陷 4：评分必须由配置驱动，改参数要能改变结果。"""

    GTS = [gt('G1', [0, 0, 20, 100])]
    PREDS = [pred('P1', [0, 0, 19, 100])]     # IoU ≈ 0.95/1.0 → 在 0.75 下也可能命中

    def test_group_threshold_changes_result(self):
        t = {'p': RegionTable('p')}
        loose = score_predictions(self.PREDS, self.GTS, t,
                                  {'group_iou_thresholds': [0.5]})
        strict = score_predictions(self.PREDS, self.GTS, t,
                                   {'group_iou_thresholds': [0.99]})
        self.assertEqual(loose['summary']['overall']['0.5']['group_hits'], 1)
        self.assertEqual(strict['summary']['overall']['0.99']['group_hits'], 0)

    def test_lane_threshold_changes_category(self):
        t = {'p': RegionTable('p')}
        # gt: right 10-20 / left 0-8；pred: right 10-18 / left 2-8
        # 固定侧别 IoU 分别 0.8 / 0.75 → 阈值 0.5 判 success，阈值 0.85 判 failure
        g = [{'group_id': 'G1', 'page_id': 'p', 'group_bbox': [0, 0, 20, 100],
              'right_bbox': [10, 0, 20, 100], 'left_bbox': [0, 0, 8, 100],
              'certainty': '确认', 'endpoint_interpretation': {'status': 'pending'}}]
        p = [{'prediction_id': 'P1', 'page_id': 'p', 'group_bbox': [2, 0, 18, 100],
              'right_bbox': [10, 0, 18, 100], 'left_bbox': [2, 0, 8, 100]}]
        loose = score_predictions(p, g, t,
                                  {'group_iou_thresholds': [0.5],
                                   'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5}})
        strict = score_predictions(p, g, t,
                                   {'group_iou_thresholds': [0.5],
                                    'lane_child_thresholds': {'right_iou': 0.85, 'left_iou': 0.85}})
        self.assertEqual(loose['pages'][0]['thresholds']['0.5']['lane']['counts']
                         .get('pairing_success'), 1)
        self.assertEqual(strict['pages'][0]['thresholds']['0.5']['lane']['counts']
                         .get('child_localization_failure'), 1)

    def test_region_threshold_changes_category(self):
        neg = [{'negative_region_id': 'N1', 'page_id': 'p', 'bbox': [0, 0, 10, 100],
                'category': 'body_text', 'status': 'confirmed'}]
        p = [pred('P1', [0, 0, 20, 100])]     # ratio 0.5
        t_lo = {'p': RegionTable('p', neg, thresholds={'suspected_fp_min': 0.9})}
        t_hi = {'p': RegionTable('p', neg, thresholds={'suspected_fp_min': 0.5})}
        lo = score_predictions(p, [], t_lo, {'group_iou_thresholds': [0.5]})
        hi = score_predictions(p, [], t_hi, {'group_iou_thresholds': [0.5]})
        self.assertEqual(lo['pages'][0]['thresholds']['0.5']['predictions'][0]['base_status'],
                         'PENDING_CANDIDATE')
        self.assertEqual(hi['pages'][0]['thresholds']['0.5']['predictions'][0]['base_status'],
                         'FP')

    def test_effective_config_echoed(self):
        r = score_predictions(self.PREDS, self.GTS, {'p': RegionTable('p')},
                              {'group_iou_thresholds': [0.6]})
        self.assertEqual(r['config']['group_iou_thresholds'], [0.6])


class ScorerTests(unittest.TestCase):
    def test_perfect_match_and_lane_success(self):
        r = score_predictions([pred('P1', [0, 0, 20, 100])], [gt('G1', [0, 0, 20, 100])],
                              {'p': RegionTable('p')})
        t = r['pages'][0]['thresholds']['0.5']
        self.assertEqual(t['tp'], 1)
        self.assertEqual(t['lane']['counts'].get('pairing_success'), 1)

    def test_lane_swap_detected(self):
        p = [{'prediction_id': 'P1', 'page_id': 'p', 'group_bbox': [0, 0, 20, 100],
              'right_bbox': [0, 0, 5, 100], 'left_bbox': [15, 0, 20, 100]}]
        t = score_predictions(p, [gt('G1', [0, 0, 20, 100])], {'p': RegionTable('p')})
        self.assertEqual(t['pages'][0]['thresholds']['0.5']['lane']['counts']
                         .get('left_right_swap'), 1)

    def test_ambiguous_lane_excluded_from_denominator(self):
        r = score_predictions([pred('P1', [0, 0, 20, 100])],
                              [gt('G1', [0, 0, 20, 100], certainty='ambiguous_lane_extent')],
                              {'p': RegionTable('p')})
        t = r['pages'][0]['thresholds']['0.5']
        self.assertEqual(t['group_hits'], 1)
        self.assertEqual(t['lane']['denominator'], 0)
        self.assertEqual(len(t['lane']['excluded']), 1)

    def test_duplicate_extra_counts_fp(self):
        r = score_predictions([pred('P1', [0, 0, 20, 100]), pred('P2', [1, 0, 21, 100])],
                              [gt('G1', [0, 0, 20, 100])], {'p': RegionTable('p')})
        t = r['pages'][0]['thresholds']['0.5']
        self.assertEqual(t['tp'], 1)
        self.assertEqual(t['duplicate_extra_fp'], 1)
        self.assertEqual(t['fp'], 1)
        self.assertEqual(t['duplicate_without_flipping_tp'], [])

    def test_missing_lane_bbox_falls_to_child_failure(self):
        p = [{'prediction_id': 'P1', 'page_id': 'p', 'group_bbox': [0, 0, 20, 100],
              'right_bbox': None, 'left_bbox': [0, 0, 5, 100]}]
        r = score_predictions(p, [gt('G1', [0, 0, 20, 100])], {'p': RegionTable('p')})
        self.assertEqual(r['pages'][0]['thresholds']['0.5']['lane']['counts']
                         .get('child_localization_failure'), 1)

    def test_endpoint_sensitivity_reports_both(self):
        gts = [gt('G1', [0, 0, 20, 100]), gt('G2', [0, 200, 20, 300])]
        preds = [pred('P1', [0, 0, 21, 101]), pred('P2', [0, 200, 21, 301])]
        r = endpoint_sensitivity(preds, gts, {'p': RegionTable('p')})
        self.assertEqual(r['delta']['pending_groups'], 2)
        d = r['delta']['by_threshold']['0.5']
        self.assertEqual(d['group_hits']['nominal'], 2)
        self.assertGreater(d['total_iou']['inclusive'], d['total_iou']['nominal'])
        self.assertEqual(len(r['delta']['changed_pairs']), 4)
        self.assertIn('未决', r['delta']['interpretation'])

    def test_recall_precision_declared_not_computable(self):
        r = score_predictions([], [gt('G1', [0, 0, 20, 100])], {'p': RegionTable('p')})
        self.assertEqual(r['summary']['overall']['0.5']['recall'], 'not_computable')
        self.assertEqual(r['summary']['overall']['0.5']['precision'], 'not_computable')


class RealAnnotationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ann = json.loads((OUT2 / 'annotations.json').read_text(encoding='utf-8'))

    def test_self_prediction_hits_everything(self):
        gts = self.ann['groups']
        preds = [{'prediction_id': f"self:{g['group_id']}", 'page_id': g['page_id'],
                  'group_bbox': g['group_bbox'], 'right_bbox': g['right_bbox'],
                  'left_bbox': g['left_bbox'], 'source': 'annotation_copy'}
                 for g in gts]
        pages = sorted({g['page_id'] for g in gts})
        sizes = {p['page_id']: p.get('image_size') for p in self.ann['pages']}
        r = score_predictions(preds, gts, {}, pages=pages, page_sizes=sizes)
        for tk in ('0.5', '0.75'):
            o = r['summary']['overall'][tk]
            self.assertEqual(o['group_hits'], len(gts), tk)
            self.assertEqual(o['fp'], 0, tk)
            self.assertEqual(o['fn'], 0, tk)
            self.assertEqual(o['hit_rate_on_known'], 1.0, tk)
            self.assertTrue(o['status_partition_ok'], tk)


class FallbackPrecisionTests(unittest.TestCase):
    """本轮缺陷 1：M−IoU 在 M 较大时把小差异抹平，回退路径选错边。"""

    def test_1x601_nextafter_picks_second(self):
        import math
        a = 0.5
        b = math.nextafter(0.5, 1.0)
        row = [a, b] + [0.0] * 599
        self.assertEqual(sum(1 for v in row if v >= 0.5), 2)
        r = match_groups([row], 0.5)
        self.assertEqual(r['pairs'], [[0, 1]])
        self.assertEqual(r['total_iou'], b)
        self.assertEqual(r['tie_break_status'], 'skipped_scale_limit')
        # 规模限制只影响第三目标，基数与配对必须仍然正确
        self.assertEqual(r['cardinality'], 1)
        self.assertFalse(r['tie_break_scope']['applied'])

    def test_cost_does_not_collapse(self):
        """直接验证代价不再相等：这是上面缺陷的根因。"""
        import math
        a, b = 0.5, math.nextafter(0.5, 1.0)
        self.assertNotEqual(-a, -b)
        M = 3.0
        self.assertEqual(M - a, M - b)      # 旧写法确实抹平了

    def test_fallback_matches_reference_on_mid_size(self):
        """中等规模（超过字典序限制、但可穷举参考）与参考解一致。"""
        rnd = random.Random(7)
        n, m = 2, 25     # 50 个节点：不受限；用来确认构造没问题
        matrix = [[round(rnd.choice([0.0, 0.5, 0.5000000000000001, 0.9]), 16)
                   for _ in range(m)] for _ in range(n)]
        fast = match_groups(matrix, 0.5)
        ref = solve_exhaustive(matrix, 0.5)
        self.assertEqual(fast['cardinality'], ref['cardinality'])
        self.assertAlmostEqual(fast['total_iou'], ref['total_iou'], places=12)

    def test_scale_limit_only_affects_tie_break(self):
        """超限时仍要求：基数最大、总分最大，只是不保证字典序。"""
        import math
        rows = [[0.5, math.nextafter(0.5, 1.0)] + [0.0] * 598 for _ in range(2)]
        r = match_groups(rows, 0.5)
        self.assertEqual(r['cardinality'], 2)
        self.assertEqual(r['tie_break_status'], 'skipped_scale_limit')
        self.assertEqual(r['total_iou'], sum(rows[i][j] for i, j in r['pairs']))

    def test_custom_limits_are_honored(self):
        r = match_groups([[0.6, 0.7]], 0.5, limits={'max_nodes': 1})
        self.assertEqual(r['tie_break_status'], 'skipped_scale_limit')
        self.assertEqual(r['pairs'], [[0, 1]])       # 配对仍取最优
        big = match_groups([[0.6, 0.7]], 0.5, limits={'max_nodes': 100})
        self.assertEqual(big['tie_break_status'], 'exact')


class LaneConfigTests(unittest.TestCase):
    """本轮缺陷 2 的一半：excluded_certainty 由配置驱动，空名单有意义。"""

    def _run(self, exclude, certainty='ambiguous_lane_extent'):
        g = [gt('G1', [0, 0, 20, 100], certainty=certainty)]
        p = [pred('P1', [0, 0, 20, 100])]
        cfg = {'group_iou_thresholds': [0.5],
               'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5}}
        if exclude is not None:
            cfg['lane_eligible_exclude_certainty'] = exclude
        return score_predictions(p, g, {'p': RegionTable('p')}, cfg)['pages'][0]['thresholds']['0.5']

    def test_default_excludes_ambiguous(self):
        x = self._run(None)
        self.assertEqual(x['lane']['denominator'], 0)
        self.assertEqual(len(x['lane']['excluded']), 1)
        self.assertEqual(x['group_hits'], 1)

    def test_empty_list_means_no_exclusion(self):
        x = self._run([])
        self.assertEqual(x['lane']['denominator'], 1)
        self.assertEqual(len(x['lane']['excluded']), 0)
        self.assertEqual(x['lane']['counts'].get('pairing_success'), 1)

    def test_custom_certainty_in_list(self):
        x = self._run(['确认'], certainty='确认')
        self.assertEqual(x['lane']['denominator'], 0)
        self.assertEqual(x['group_hits'], 1)


class ExhaustiveRegionTests(unittest.TestCase):
    """本轮缺陷 3：局部穷尽区域必须用真实 bbox，区域外不能按穷尽算。"""

    REGIONS = [{'region_id': 'R1', 'page_id': 'p', 'bbox': [0, 0, 100, 200],
                'coverage': 'exhaustive_region'}]

    def _table(self, regions=None, coverage='exhaustive_region'):
        return RegionTable('p', coverage=coverage,
                           evaluation_regions=self.REGIONS if regions is None else regions)

    def test_unmatched_inside_region_is_fp(self):
        t = self._table()
        x = score_predictions([pred('P1', [10, 10, 40, 100])], [], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['predictions'][0]['base_status'], 'FP')
        self.assertEqual(x['predictions'][0]['evaluation_region_id'], 'R1')
        self.assertEqual(x['fp'], 1)

    def test_unmatched_outside_region_is_not_fp(self):
        t = self._table()
        x = score_predictions([pred('P1', [300, 10, 330, 100])], [], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['predictions'][0]['base_status'], 'PENDING_CANDIDATE')
        self.assertEqual(x['predictions'][0]['review_category'], 'unmatched_outside_region')
        self.assertEqual(x['fp'], 0)
        self.assertEqual(x['new_candidates'], 1)

    def test_positive_match_wins_inside_region(self):
        t = self._table()
        x = score_predictions([pred('P1', [10, 10, 40, 100])],
                              [gt('G1', [10, 10, 40, 100])], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['tp'], 1)
        self.assertEqual(x['fp'], 0)

    def test_ignore_and_ambiguous_keep_priority(self):
        t = RegionTable('p',
                        ambiguous=[{'ambiguous_region_id': 'A1', 'page_id': 'p',
                                    'bbox': [10, 10, 40, 100]}],
                        ignores=[{'ignore_region_id': 'I1', 'page_id': 'p',
                                  'bbox': [50, 10, 90, 100]}],
                        coverage='exhaustive_region', evaluation_regions=self.REGIONS)
        x = score_predictions([pred('IGN', [50, 10, 90, 100]),
                               pred('AMB', [10, 10, 40, 100])], [], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        cats = {r['prediction_id']: r['base_status'] for r in x['predictions']}
        self.assertEqual(cats['IGN'], 'IGNORED')
        self.assertEqual(cats['AMB'], 'PENDING_AMBIGUOUS')
        self.assertEqual(x['fp'], 0)

    def test_missing_region_bbox_is_data_error(self):
        t = RegionTable('p', coverage='exhaustive_region')
        self.assertIsNotNone(t.data_error)
        r = score_predictions([pred('P1', [0, 0, 10, 10])], [], {'p': t})
        self.assertEqual(r['pages'][0]['status'], 'data_error')
        self.assertEqual(r['summary']['overall']['0.5']['status'], 'not_computable_partial')
        self.assertEqual(r['data_errors'][0]['page_id'], 'p')

    def test_invalid_region_bbox_reported(self):
        t = RegionTable('p', coverage='exhaustive_region',
                        evaluation_regions=[{'region_id': 'BAD', 'page_id': 'p',
                                             'bbox': [10, 10, 10, 20]}])
        self.assertIn('BAD', t.data_error)


class PartialPolicyConsistencyTests(unittest.TestCase):
    """partial_policy 与「确定负例」「重复预测」必须同一口径。"""

    def test_partial_matched_duplicate_extra_is_fp_not_candidate(self):
        r = score_predictions([pred('P1', [0, 0, 20, 100]), pred('P2', [1, 0, 21, 100])],
                              [gt('G1', [0, 0, 20, 100])], {'p': RegionTable('p')})
        x = r['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['duplicate_extra_fp'], 1)
        states = {row['prediction_id']: row['base_status'] for row in x['predictions']}
        self.assertEqual(states['P1'], 'TP')
        self.assertEqual(states['P2'], 'FP')

    def test_partial_confirmed_negative_beats_candidate(self):
        t = RegionTable('p', negatives=[{'negative_region_id': 'N1', 'page_id': 'p',
                                         'bbox': [0, 0, 20, 100], 'category': 'body_text',
                                         'status': 'confirmed'}])
        x = score_predictions([pred('P1', [0, 0, 20, 100])], [], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['predictions'][0]['base_status'], 'FP')
        self.assertEqual(x['predictions'][0]['review_category'], 'confirmed_false_positive')

    def test_partial_small_edge_overlap_stays_candidate(self):
        t = RegionTable('p', negatives=[{'negative_region_id': 'N1', 'page_id': 'p',
                                         'bbox': [0, 0, 2, 100], 'category': 'rule',
                                         'status': 'confirmed'}])
        x = score_predictions([pred('P1', [0, 0, 20, 100])], [], {'p': t}
                              )['pages'][0]['thresholds']['0.5']
        self.assertEqual(x['predictions'][0]['base_status'], 'PENDING_CANDIDATE')
        self.assertGreater(x['predictions'][0]['negative_coverage_ratio'], 0)


if __name__ == '__main__':
    unittest.main()
