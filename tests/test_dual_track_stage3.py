# -*- coding: utf-8 -*-
"""第三阶段产物测试：冻结快照、布局标签、配置/目标、预测契约、评分可复现。

这些测试的作用是「防止报告与产物脱节」：报告里写的数字必须能在产物里查到，
产物里的字段必须能被程序读出并重算。
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_eval import RegionTable, endpoint_sensitivity, score_predictions  # noqa: E402

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def exists(p: Path) -> bool:
    return p.exists()


class ScaleErratumTests(unittest.TestCase):
    """勘误：1.5 * (35 - 5) = 45，不是 30。"""

    def test_addendum_records_45(self):
        a = load(S3 / 'protocol_addendum.json')
        e = [x for x in a['errata'] if x['id'] == 'E1-SCALE-EXAMPLE'][0]
        self.assertIn('45', e['correct'])
        self.assertIn('30px', e['wrong'])
        self.assertEqual(e['field_values_unchanged'],
                         {'H': 30, 'G': 5, 'glyph_pitch': 35})

    def test_math(self):
        H = 30
        gap = 5
        pitch = H + gap
        self.assertEqual(pitch, 35)
        self.assertEqual(1.5 * (pitch - gap), 45)
        self.assertNotEqual(1.5 * (pitch - gap), 30)

    def test_stage2_contract_file_untouched(self):
        a = load(S3 / 'protocol_addendum.json')
        self.assertFalse(a['upstream']['stage2_evaluation_protocol']['modified'])
        self.assertEqual(len(a['upstream']['stage2_evaluation_protocol']['sha256']), 64)

    def test_ambiguities_have_minimal_example(self):
        a = load(S3 / 'protocol_addendum.json')
        self.assertGreaterEqual(len(a['protocol_ambiguities']), 5)
        for x in a['protocol_ambiguities']:
            for k in ('id', 'protocol_ref', 'question', 'minimal_example',
                      'rule_adopted', 'why', 'scope'):
                self.assertIn(k, x, x['id'])
                self.assertTrue(str(x[k]).strip(), f"{x['id']}.{k} 为空")


class LayoutLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lay = load(S3 / 'frozen_layouts.json')
        cls.lab = load(S3 / 'frozen_layout_labels.json')
        cls.ann = load(S2 / 'annotations.json')

    def test_all_pages_frozen(self):
        self.assertEqual(self.lay['status_counts'].get('ok'), len(self.lay['pages']))
        self.assertGreaterEqual(len(self.lay['pages']), 108)

    def test_labels_cover_annotated_pages(self):
        need = {p['page_id'] for p in self.ann['pages']}
        have = {r['page_id'] for r in self.lab['pages']}
        self.assertTrue(need <= have, need - have)

    def test_states_in_enum(self):
        for r in self.lab['pages']:
            self.assertIn(r['state'], ('normal', 'abnormal', 'unknown'))

    def test_unknown_has_reason(self):
        for r in self.lab['pages']:
            if r['state'] == 'unknown':
                self.assertTrue(r['reason'], r['page_id'])

    def test_abnormal_lists_outlier_columns(self):
        for r in self.lab['pages']:
            if r['state'] == 'abnormal':
                self.assertTrue(r['measured']['outlier_cols'], r['page_id'])
                for c in r['measured']['outlier_cols']:
                    self.assertGreater(c['width'], r['measured']['threshold_px'])

    def test_measured_object_is_layout_column(self):
        self.assertIn('旧布局', self.lab['rule']['measured_object'])
        self.assertIn('column', self.lab['rule']['measured_object'])

    def test_p12_wide_box_is_recorded(self):
        r = [x for x in self.lab['pages'] if x['page_id'] == '丹方彚编/page_12'][0]
        self.assertEqual(r['state'], 'abnormal')
        cols = r['measured']['outlier_cols']
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0]['width'], 446)
        self.assertEqual(cols[0]['bbox'], [545, 0, 991, 873])
        self.assertGreater(cols[0]['ratio_to_reference'], 10)

    def test_no_annotation_group_width_used(self):
        # 规则里不得出现「人工 group_bbox 宽度」作为测量对象
        blob = json.dumps(self.lab['rule'], ensure_ascii=False)
        self.assertIn('旧布局', blob)
        self.assertNotIn('group_bbox 宽度作为测量', blob)


class ConfigTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load(S3 / 'evaluation_config.json')
        cls.tg = load(S3 / 'evaluation_targets.json')
        cls.snap = load(S3 / 'baseline_snapshot.json')

    def test_config_links_snapshot_and_layout(self):
        self.assertEqual(self.cfg['baseline_snapshot_id'], self.snap['snapshot_id'])
        self.assertTrue(self.cfg['layout_snapshot_id'].startswith('layout-'))

    def test_config_declares_not_applicable(self):
        na = self.cfg['not_applicable']
        for k in ('seam_threshold', 'scale_estimation', 'gap_fill_params'):
            self.assertIn(k, na)

    def test_region_thresholds_pending(self):
        r = self.cfg['regions']['thresholds']
        self.assertEqual(r['status'], 'pending_calibration')
        self.assertIn('basis', r)

    def test_targets_have_denominators(self):
        for m in self.tg['metrics']:
            has_ratio = 'target' in m and 'baseline' in m
            has_typed = m.get('denominator_type') in ('dynamic', 'negative_regions')
            self.assertTrue(has_ratio or has_typed, m['id'])
            self.assertTrue(m['basis'])

    def test_targets_not_lower_than_baseline_except_guard(self):
        for m in self.tg['metrics']:
            if m['id'] in ('T5', 'T6'):
                continue
            self.assertGreaterEqual(m['target_value'], m['baseline_value'], m['id'])

    def test_fixed_cases_use_real_group_ids(self):
        lst = self.tg['fixed_cases']['list']
        self.assertGreaterEqual(len(lst), 5)
        ids = {x['group_id'] for x in lst}
        self.assertIn('丹方彚编/p12/DF12-03', ids)
        # 旧版的错字「彚編」不得再出现
        self.assertFalse(any('彚編' in i for i in ids), ids)
        for c in lst:
            for k in ('exists', 'unique', 'page_consistent', 'eligibility_present'):
                self.assertTrue(c['checks'][k], f"{c['group_id']}.{k}")

    def test_regression_threshold_is_single_rule(self):
        t3 = [m for m in self.tg['metrics'] if m['id'] == 'T3'][0]
        self.assertIn('pass_rule', t3)
        self.assertIn('warning_rule', t3)
        self.assertIn('不改变', t3['warning_rule'])
        self.assertEqual(t3['target'], '19/20')

    def test_lane_target_uses_dynamic_denominator(self):
        t5 = [m for m in self.tg['metrics'] if m['id'] == 'T5'][0]
        self.assertEqual(t5['denominator_type'], 'dynamic')
        self.assertIn('不得固定', t5['denominator_rule'])
        self.assertAlmostEqual(t5['target_ratio'], 0.90)

    def test_negative_target_uses_region_denominator(self):
        t6 = [m for m in self.tg['metrics'] if m['id'] == 'T6'][0]
        self.assertEqual(t6['denominator_type'], 'negative_regions')
        self.assertEqual(t6['status'], 'conditional')
        self.assertIn('负例区域', t6['denominator'])
        self.assertIn('px²', t6['denominator'])
        self.assertNotEqual(t6['baseline'], '0/1')

    def test_not_computable_declared(self):
        for k in ('recall', 'precision', 'false_group_count'):
            self.assertIn('not_computable', self.tg['not_computable'][k])


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = load(S3 / 'baseline_snapshot.json')

    def test_code_hashes_present(self):
        self.assertEqual(self.s['code']['missing'], [])
        for f, h in self.s['code']['files'].items():
            self.assertEqual(len(h), 64, f)

    def test_model_fingerprint(self):
        self.assertEqual(len(self.s['model']['sha256']), 64)
        self.assertTrue(self.s['model']['name'].endswith('.onnx'))

    def test_inputs_fingerprinted(self):
        pages = self.s['inputs']['pages']
        self.assertGreaterEqual(len(pages), 19)
        for p in pages:
            self.assertEqual(len(p['sha256']), 64)

    def test_credentials_redacted(self):
        blob = json.dumps(self.s['config'], ensure_ascii=False).lower()
        for bad in ('api_key": "', 'secret": "', 'password": "'):
            self.assertNotIn(bad, blob)

    def test_primary_declared(self):
        self.assertEqual(self.s['entries']['primary']['role'], 'primary_baseline')
        self.assertIn('propose_page', self.s['entries']['primary']['entry'])
        self.assertGreaterEqual(len(self.s['entries']['auxiliary']), 3)

    def test_per_page_has_three_paths(self):
        pp = self.s['per_page_runs']
        self.assertGreaterEqual(len(pp), 15)
        some = [v for v in pp.values() if 'C' in v]
        self.assertTrue(some)


class PredictionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {}
        for code, name in (('B', 'path_b.json'), ('C', 'path_c.json'),
                           ('X', 'path_x_components.json')):
            p = S3 / 'predictions' / name
            if exists(p):
                cls.paths[code] = load(p)

    def test_all_paths_available(self):
        self.assertEqual(sorted(self.paths), ['B', 'C', 'X'])

    def test_group_bbox_is_lane_union(self):
        from dual_track_eval.geometry import union_bbox
        for code, payload in self.paths.items():
            for p in payload['predictions']:
                if not p.get('right_bbox') or not p.get('left_bbox'):
                    continue
                self.assertEqual(p['group_bbox'],
                                 list(union_bbox([p['right_bbox'], p['left_bbox']])), code)

    def test_every_page_has_run_record(self):
        for code, payload in self.paths.items():
            pids = {p['page_id'] for p in payload['predictions']}
            runs = {r['page_id'] for r in payload['page_runs']}
            self.assertTrue(pids <= runs, (code, pids - runs))

    def test_failed_predictions_carry_error(self):
        for code, payload in self.paths.items():
            for p in payload['predictions']:
                if p.get('run_status') == 'failed':
                    self.assertTrue(p.get('error'))

    def test_recognition_records_have_rotation(self):
        r = load(S3 / 'recognition' / 'path_a.json')
        self.assertEqual(r['preprocessing']['rotation'], 'rot90_ccw（竖排转横排）')
        rec = [x for x in r['records'] if x.get('lanes')]
        self.assertTrue(rec)
        lane = next(iter(rec[0]['lanes'].values()))
        self.assertIn('rotation', lane)
        self.assertIn('scale', lane)


class ScoringReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ann = load(S2 / 'annotations.json')
        cls.neg = load(S2 / 'negative_regions.json')
        cls.store = load(S3 / 'metrics' / 'c.json')
        cls.cfg = {'group_iou_thresholds': [0.5, 0.75],
                   'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5}}

    def _tables(self):
        cov = {p['page_id']: p.get('annotation_coverage', 'partial_positions')
               for p in self.ann['pages']}
        neg_pages = {r['page_id'] for r in self.neg['regions']}
        out = {}
        for pid in set(cov) | neg_pages:
            out[pid] = RegionTable(pid, self.neg['regions'], [], [],
                                   coverage=cov.get(pid) or 'negative_only')
        return out

    def test_rescore_matches_stored(self):
        preds = load(S3 / 'predictions' / 'path_c.json')['predictions']
        pages = sorted({p['page_id'] for p in preds} |
                       {g['page_id'] for g in self.ann['groups']})
        r = score_predictions(preds, self.ann['groups'], self._tables(), self.cfg, pages=pages)
        for tk in ('0.5', '0.75'):
            a = r['summary']['overall'][tk]
            b = self.store['scores']['overall'][tk]
            for k in ('group_hits', 'annotations_total', 'tp', 'fp', 'fn',
                      'new_candidates', 'lane_denominator'):
                self.assertEqual(a[k], b[k], f'{tk}.{k}')
            self.assertEqual(a['lane_counts'], b['lane_counts'])
            self.assertAlmostEqual(a['hit_rate_on_known'], b['hit_rate_on_known'], places=9)

    def test_baseline_numbers(self):
        o = self.store['scores']['overall']
        self.assertEqual(o['0.5']['group_hits'], 24)
        self.assertEqual(o['0.75']['group_hits'], 11)
        self.assertEqual(o['0.5']['annotations_total'], 38)
        self.assertEqual(o['0.5']['lane_denominator'], 23)

    def test_endpoint_sensitivity_recorded(self):
        s = self.store['endpoint_sensitivity']['delta']
        self.assertEqual(s['pending_groups'], 23)
        self.assertEqual(s['body_report_uses'], 'nominal')
        for tk in ('0.5', '0.75'):
            self.assertIn(tk, s['by_threshold'])

    def test_no_fp_in_partial_pages(self):
        """部分标注区不计 FP：所有 FP 必须来自负例区或穷尽区。"""
        for page in self.store['pages']:
            for r in page['thresholds']['0.5']['predictions']:
                if r['label'] == 'FP':
                    self.assertIn(r['review_category'],
                                  ('confirmed_false_positive', 'suspected_false_positive',
                                   'unmatched_uncovered'), page['page_id'])

    def test_metrics_have_page_level_and_limits(self):
        self.assertGreaterEqual(len(self.store['pages']), 19)
        self.assertIn('not_computable', self.store['metric_limits']['recall'])

    def test_failure_attribution_complete(self):
        fa = self.store['failure_attribution']
        runs = load(S3 / 'predictions' / 'path_c.json')['page_runs']
        self.assertEqual(len(fa['per_page']), len(runs))
        for r in fa['per_page']:
            self.assertIn(r['stage'], ('input', 'region_localization', 'grouping',
                                       'box_boundary', 'no_gt_page', 'none'))


class ReportConsistencyTests(unittest.TestCase):
    def test_report_exists_and_mentions_numbers(self):
        p = S3 / '阶段报告.md'
        self.assertTrue(p.exists())
        t = p.read_text(encoding='utf-8')
        for token in ('24/38', '11/38', 'baseline-', 'layout-'):
            self.assertIn(token, t)


class ScoreEntryChainTests(unittest.TestCase):
    """入口级回归：配置解析 → build_tables → 评分 → 归因，全链路。"""

    @classmethod
    def setUpClass(cls):
        import dual_track_stage3_score as SC
        cls.SC = SC
        cls.ann = load(S2 / 'annotations.json')
        cls.neg = load(S2 / 'negative_regions.json')
        cls.base_doc = load(S3 / 'evaluation_config.json')
        cls.preds_c = load(S3 / 'predictions' / 'path_c.json')['predictions']

    # -------------------------------------------------- 1. 空排除名单
    def test_empty_excluded_certainty_survives_to_runtime(self):
        res = self.SC.resolve_config(S3 / 'evaluation_config.json',
                                     ['lane_scoring.excluded_certainty=[]'])
        rt = self.SC.runtime_config(res['doc'])
        self.assertEqual(rt['lane_eligible_exclude_certainty'], [])
        self.assertEqual(res['overrides'][0]['value'], [])
        self.assertIn('空名单', res['overrides'][0]['note'])

    def test_absent_field_uses_default(self):
        doc = copy.deepcopy(self.base_doc)
        doc['lane_scoring'].pop('excluded_certainty', None)
        rt = self.SC.runtime_config(doc)
        self.assertEqual(rt['lane_eligible_exclude_certainty'],
                         ['ambiguous_lane_split', 'ambiguous_lane_extent'])

    def test_empty_list_changes_lane_denominator_on_real_c(self):
        pages = sorted({p['page_id'] for p in self.preds_c})
        sizes = self.SC.page_sizes_map(self.ann)
        tabs = self.SC.build_tables(self.ann, self.neg, pages)
        default = self.SC.runtime_config(self.base_doc)
        r_def = score_predictions(self.preds_c, self.ann['groups'], tabs, default, pages=pages,
                                  page_sizes=sizes)
        empty = self.SC.resolve_config(S3 / 'evaluation_config.json',
                                       ['lane_scoring.excluded_certainty=[]'])
        r_empty = score_predictions(self.preds_c, self.ann['groups'], tabs,
                                    self.SC.runtime_config(empty['doc']), pages=pages,
                                    page_sizes=sizes)
        a = r_def['summary']['overall']['0.5']
        b = r_empty['summary']['overall']['0.5']
        self.assertEqual(a['group_hits'], 24)
        self.assertEqual(b['group_hits'], 24)
        self.assertEqual(a['lane_denominator'], 23)
        self.assertEqual(b['lane_denominator'], 24)
        self.assertEqual(a['lane_excluded'], 1)
        self.assertEqual(b['lane_excluded'], 0)

    # -------------------------------------------------- 2. 局部穷尽区域
    def test_evaluation_regions_through_build_tables(self):
        import copy as _copy
        ann = _copy.deepcopy(self.ann)
        target = '思濟堂方書/page_33'
        for p in ann['pages']:
            if p['page_id'] == target:
                p['annotation_coverage'] = 'exhaustive_region'
                p['evaluation_regions'] = [{'region_id': 'R-P33-RIGHT',
                                            'bbox': [800, 0, 1000, 800],
                                            'coverage': 'exhaustive_region'}]
        pages = [target]
        tabs = self.SC.build_tables(ann, self.neg, pages)
        cfg = self.SC.runtime_config(self.base_doc)
        r = score_predictions([p for p in self.preds_c if p['page_id'] == target],
                              ann['groups'], tabs, cfg, pages=pages,
                              page_sizes=self.SC.page_sizes_map(ann))
        page = r['pages'][0]
        t = page['thresholds']['0.5']
        rows = {x['prediction_id'].split('#')[-1]: x for x in t['predictions']}
        # 区域内：#00 命中、#01 未匹配 → FP
        self.assertEqual(rows['00']['base_status'], 'TP')
        self.assertEqual(rows['01']['base_status'], 'FP')
        self.assertEqual(rows['01']['evaluation_region_id'], 'R-P33-RIGHT')
        # 区域外：#02 / #03 按部分覆盖 → 候选，不计 FP
        for k in ('02', '03'):
            self.assertEqual(rows[k]['base_status'], 'PENDING_CANDIDATE')
            self.assertEqual(rows[k]['annotation_coverage'], 'partial_positions')
        self.assertEqual(t['fp'], 1)
        self.assertEqual(t['new_candidates'], 2)

    def test_page_id_conflict_is_data_error(self):
        from dual_track_eval.regions import RegionTable
        t = RegionTable('p', coverage='exhaustive_region',
                        evaluation_regions=[{'region_id': 'R1', 'page_id': 'OTHER',
                                             'bbox': [0, 0, 10, 10],
                                             'coverage': 'exhaustive_region'}])
        self.assertIsNotNone(t.data_error)
        self.assertIn('page_id', t.data_error)
        self.assertIn('R1', t.data_error)

    def test_missing_page_id_inherits_parent(self):
        from dual_track_eval.regions import RegionTable
        t = RegionTable('p', coverage='exhaustive_region',
                        evaluation_regions=[{'region_id': 'R1', 'bbox': [0, 0, 10, 10],
                                             'coverage': 'exhaustive_region'}])
        self.assertIsNone(t.data_error)
        self.assertEqual(t.evaluation_regions[0]['page_id'], 'p')

    # -------------------------------------------------- 3. data_error 进报告
    def test_data_error_page_reaches_failure_attribution(self):
        from dual_track_eval import RegionTable
        tabs = {'good': RegionTable('good'),
                'bad': RegionTable('bad', coverage='exhaustive_region')}
        preds = [{'prediction_id': 'P1', 'page_id': 'good', 'group_bbox': [0, 0, 20, 100],
                  'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100]},
                 {'prediction_id': 'P2', 'page_id': 'bad', 'group_bbox': [0, 0, 20, 100],
                  'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100]}]
        gts = [{'group_id': 'GG', 'page_id': 'good', 'group_bbox': [0, 0, 20, 100],
                'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100],
                'certainty': '确认'}]
        res = score_predictions(preds, gts, tabs, pages=['good', 'bad'])
        payload = {'page_runs': [
            {'page_id': 'good', 'status': 'completed', 'n_predictions': 1, 'n_columns_detected': 5},
            {'page_id': 'bad', 'status': 'completed', 'n_predictions': 1, 'n_columns_detected': 5}]}
        fa = self.SC.failure_attribution(payload, res)
        self.assertEqual(fa['data_error_pages'], ['bad'])
        bad_row = [r for r in fa['per_page'] if r['page_id'] == 'bad'][0]
        self.assertEqual(bad_row['stage'], 'data_error')
        self.assertIn('exhaustive_region', bad_row['reason'])
        # 正常页仍计分
        self.assertEqual(res['summary']['overall']['0.5']['group_hits'], 1)
        self.assertEqual(res['summary']['overall']['0.5']['status'], 'not_computable_partial')
        self.assertEqual(res['summary']['overall']['0.5']['data_error_page_count'], 1)
        self.assertIn('不得当成全量结果',
                      res['summary']['overall']['0.5']['scope_note'])


class FullEntryMixedInputTests(unittest.TestCase):
    """调用完整评分入口（score_file）处理「正常页 + 缺区域坐标页」的混合输入。"""

    @classmethod
    def setUpClass(cls):
        import dual_track_stage3_score as SC
        cls.SC = SC
        cls.tmp = ROOT / 'output' / 'dual_tracks' / 'stage3' / '_test_tmp'
        cls.tmp.mkdir(parents=True, exist_ok=True)
        cls.payload = {
            'path': 'synthetic_mixed', 'entry': 'test',
            'predictions': [
                {'prediction_id': 'P1', 'page_id': 'good', 'group_bbox': [0, 0, 20, 100],
                 'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100], 'run_status': 'ok'},
                {'prediction_id': 'P2', 'page_id': 'bad', 'group_bbox': [0, 0, 20, 100],
                 'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100], 'run_status': 'ok'},
            ],
            'page_runs': [
                {'page_id': 'good', 'status': 'completed', 'n_predictions': 1,
                 'n_columns_detected': 5},
                {'page_id': 'bad', 'status': 'completed', 'n_predictions': 1,
                 'n_columns_detected': 4},
            ],
        }
        cls.pred_file = cls.tmp / 'pred.json'
        cls.pred_file.write_text(json.dumps(cls.payload, ensure_ascii=False), encoding='utf-8')
        cls.ann = {'release': 'synthetic',
                   'pages': [{'page_id': 'good', 'annotation_coverage': 'partial_positions',
                              'image_size': [1024, 800]},
                             {'page_id': 'bad', 'annotation_coverage': 'exhaustive_region',
                              'image_size': [1024, 800]}],
                   'groups': [{'group_id': 'GG', 'page_id': 'good',
                               'group_bbox': [0, 0, 20, 100],
                               'right_bbox': [15, 0, 20, 100],
                               'left_bbox': [0, 0, 5, 100],
                               'certainty': '确认',
                               'endpoint_interpretation': {'status': 'pending'}}]}
        rt = SC.runtime_config(load(S3 / 'evaluation_config.json'))
        cls.result = SC.score_file(cls.pred_file, cls.ann, {'regions': []}, {}, rt,
                                   {'version': 'test', 'overrides': [],
                                    'source': {'file': 'inline', 'sha256': '0' * 64},
                                    'runtime': rt})
        cls.fa = SC.failure_attribution(cls.payload, cls.result)

    @classmethod
    def tearDownClass(cls):
        for f in cls.tmp.glob('*'):
            f.unlink()
        cls.tmp.rmdir()

    def test_report_builds_for_both_thresholds(self):
        for tk in ('0.5', '0.75'):
            o = self.result['scores']['overall'][tk]
            self.assertEqual(o['group_hits'], 1, tk)
            self.assertEqual(o['annotations_total'], 1, tk)
            self.assertEqual(o['status'], 'not_computable_partial', tk)

    def test_normal_page_still_scored(self):
        self.assertEqual(self.result['scores']['overall']['0.5']['tp'], 1)
        good = [p for p in self.result['pages'] if p['page_id'] == 'good'][0]
        self.assertEqual(good['thresholds']['0.5']['group_hits'], 1)
        self.assertEqual(good['thresholds']['0.75']['group_hits'], 1)

    def test_error_page_has_reason(self):
        self.assertEqual(len(self.result['data_errors']), 1)
        err = self.result['data_errors'][0]
        self.assertEqual(err['page_id'], 'bad')
        self.assertIn('exhaustive_region', err['error'])
        self.assertEqual(self.fa['data_error_pages'], ['bad'])
        row = [r for r in self.fa['per_page'] if r['page_id'] == 'bad'][0]
        self.assertEqual(row['stage'], 'data_error')
        self.assertTrue(row['reason'])

    def test_main_scope_declared(self):
        s = self.result['scores']['scope']
        self.assertEqual(s['pages_evaluated'], 2)
        self.assertEqual(s['pages_scorable'], 1)
        self.assertEqual(s['pages_excluded'], 1)
        self.assertEqual(s['excluded_pages'], ['bad'])

    def test_endpoint_uses_same_scorable_set(self):
        d = self.result['endpoint_sensitivity']['delta']
        self.assertEqual(d['scope']['scorable_pages'], ['good'])
        self.assertEqual(d['scope']['excluded_pages'], ['bad'])
        for tk in ('0.5', '0.75'):
            self.assertEqual(d['by_threshold'][tk]['pages'], 1, tk)
            self.assertEqual(d['by_threshold'][tk]['group_hits']['nominal'], 1, tk)
        # nominal 与 inclusive 的总 IoU 都只在 good 上算
        for tk in ('0.5', '0.75'):
            self.assertAlmostEqual(d['by_threshold'][tk]['total_iou']['nominal'], 1.0, places=6)

    def test_relative_path_accepted(self):
        import os
        rel = os.path.relpath(self.pred_file, ROOT)
        rt = self.SC.runtime_config(load(S3 / 'evaluation_config.json'))
        r = self.SC.score_file(Path(rel), self.ann, {'regions': []}, {}, rt,
                               {'version': 'test', 'overrides': [],
                                'source': {'file': 'inline', 'sha256': '0' * 64},
                                'runtime': rt})
        self.assertEqual(r['scores']['overall']['0.5']['group_hits'], 1)


if __name__ == '__main__':
    unittest.main()
