"""双轨小字页第一阶段适配逻辑的单元测试。

覆盖：尺寸不一致、框越界、重复编号、CSV/JSON 不一致、端点约定待核、
缺省字段不算错误、group_bbox 由左右轨并集生成、重复运行核心结果一致、必填字段齐全。

跑法：
  .venv/Scripts/python.exe -m unittest tests.test_dual_track_stage1 -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D  # noqa: E402
import check_dual_track_dataset as C  # noqa: E402
import normalize_dual_track_dataset as N  # noqa: E402

BOOK = '测试書'
PAGE = 10
IMAGE_PATH = D.rel(D.DATASET / BOOK / D.page_file_name(PAGE))
REAL_SIZE = [100, 96]

REQUIRED_FIELDS = (
    'page_id', 'book', 'page', 'image_path', 'image_sha256', 'image_size', 'group_id',
    'original_region_id', 'main_bbox', 'right_bbox', 'left_bbox', 'group_bbox',
    'annotation_source', 'boundary_precision', 'certainty', 'review_status',
    'coordinate_convention', 'original_coordinate_convention', 'evaluation_eligibility',
    'source_reference',
)


def make_coord(**over):
    rec = {
        'book': BOOK, 'page': PAGE, 'region': 'T10-01',
        'main_text': '甲', 'right_text': '乙', 'left_text': '丙',
        'main_bbox': [10, 10, 30, 50],
        'right_bbox': [22, 50, 32, 90],
        'left_bbox': [8, 50, 20, 75],
        'source': '本轮测量（栏内墨迹外接框，6倍叠框核对）',
        'column': 1, 'image_size': [100, 100],
        'coordinate_system': '原扫描图像素，左上角原点，x右y下',
    }
    rec.update(over)
    return rec


def make_manifest(**over):
    rec = {
        'book': BOOK, 'page': f'page_{PAGE}.png', 'status': '收录', 'confidence': '确认',
        'main': '甲', 'right': '乙', 'left': '丙', 'note': '测试页',
        'source': f'古籍评测集/医家类/{BOOK}/page_{PAGE}.png',
        'sha256': '0' * 64,
        'dataset_path': f'古籍评测集/双轨小字页/{BOOK}/page_{PAGE}.png',
    }
    rec.update(over)
    return rec


def overrides(size=REAL_SIZE):
    return {
        IMAGE_PATH: {'path': IMAGE_PATH, 'exists': True, 'readable': True,
                     'image_size': list(size), 'sha256': '0' * 64},
        D.rel(D.BOOKS / BOOK / D.page_file_name(PAGE)): {
            'path': D.rel(D.BOOKS / BOOK / D.page_file_name(PAGE)), 'exists': True,
            'readable': True, 'image_size': list(size), 'sha256': '0' * 64},
    }


def coord_to_csv_row(rec):
    """与 load_coords_csv 的输出同形态（bbox 已解析为 list，column 为 int/None）。"""
    return {
        'book': rec['book'], 'page': rec['page'], 'region': rec['region'],
        'main_text': rec['main_text'], 'right_text': rec['right_text'], 'left_text': rec['left_text'],
        'main_bbox': list(rec['main_bbox']), 'right_bbox': list(rec['right_bbox']),
        'left_bbox': list(rec['left_bbox']), 'source': rec['source'],
        'column': rec.get('column'),
        'page_raw': str(rec['page']), 'column_raw': '' if rec.get('column') is None else str(rec['column']),
    }


def synthetic_ctx(coords=None, manifest=None, csv_rows=None, size=REAL_SIZE):
    coords = coords if coords is not None else [make_coord()]
    manifest = manifest if manifest is not None else [make_manifest()]
    csv_rows = csv_rows if csv_rows is not None else [coord_to_csv_row(c) for c in coords]
    return C.Context(manifest=manifest, manifest_csv=[], coords=coords,
                     coords_csv=csv_rows, image_overrides=overrides(size))


class GeometryHelpersTests(unittest.TestCase):
    def test_as_bbox_rejects_bad_shapes(self):
        self.assertIsNone(D.as_bbox([1, 2, 3]))
        self.assertIsNone(D.as_bbox(None))
        self.assertIsNone(D.as_bbox(['a', 2, 3, 4]))
        self.assertEqual(D.as_bbox([1, 2, 3, 4]), (1, 2, 3, 4))

    def test_union_bbox_is_outer_rect(self):
        self.assertEqual(D.union_bbox([[22, 50, 32, 90], [8, 50, 20, 75]]), [8, 50, 32, 90])

    def test_intersect_area_touching_is_zero(self):
        self.assertEqual(D.intersect_area([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertEqual(D.intersect_area([0, 0, 10, 10], [9, 0, 20, 10]), 10)


class ImageSizeTests(unittest.TestCase):
    def test_size_mismatch_is_flagged(self):
        ctx = synthetic_ctx()
        res = C.check_image_size(ctx)
        self.assertEqual(res['status'], 'fail')
        self.assertEqual(len(res['mismatched']), 1)

    def test_size_match_is_pass(self):
        ctx = synthetic_ctx(coords=[make_coord(image_size=list(REAL_SIZE))])
        self.assertEqual(C.check_image_size(ctx)['status'], 'pass')

    def test_normalized_uses_real_size_and_records_correction(self):
        payload = N.build(synthetic_ctx())
        group = payload['groups'][0]
        self.assertEqual(group['image_size'], REAL_SIZE)
        self.assertEqual(group['original_image_size'], [100, 100])
        self.assertFalse(group['image_size_matches_original'])
        sizes = [c for c in payload['corrections'] if c['field'] == 'image_size']
        self.assertEqual(len(sizes), 1)
        self.assertEqual(sizes[0]['old'], [100, 100])
        self.assertEqual(sizes[0]['new'], REAL_SIZE)
        self.assertTrue(sizes[0]['basis'])


class BboxTests(unittest.TestCase):
    def test_out_of_image_box_fails(self):
        ctx = synthetic_ctx(coords=[make_coord(main_bbox=[10, 10, 30, 120])])
        res = C.check_bbox_geometry(ctx)
        self.assertEqual(res['status'], 'fail')
        self.assertIn('越出原图', ' '.join(res['bad']))

    def test_zero_area_box_fails(self):
        ctx = synthetic_ctx(coords=[make_coord(left_bbox=[8, 50, 8, 75])])
        res = C.check_bbox_geometry(ctx)
        self.assertEqual(res['status'], 'fail')

    def test_invalid_number_fails(self):
        ctx = synthetic_ctx(coords=[make_coord(right_bbox=[22, 50, None, 90])])
        res = C.check_bbox_shape(ctx)
        self.assertEqual(res['status'], 'fail')

    def test_lane_order_reversed_fails(self):
        ctx = synthetic_ctx(coords=[make_coord(right_bbox=[8, 50, 20, 75], left_bbox=[22, 50, 32, 90])])
        self.assertEqual(C.check_lane_order(ctx)['status'], 'fail')

    def test_group_bbox_is_union_of_lanes_not_main(self):
        g = N.build(synthetic_ctx())['groups'][0]
        self.assertEqual(g['group_bbox'], [8, 50, 32, 90])
        self.assertNotIn(g['main_bbox'], g['group_bbox'])


class IdTests(unittest.TestCase):
    def test_duplicate_region_fails(self):
        ctx = synthetic_ctx(coords=[make_coord(), make_coord()])
        res = C.check_unique_ids(ctx)
        self.assertEqual(res['status'], 'fail')
        self.assertEqual(len(res['duplicates']), 1)

    def test_same_region_on_other_page_is_unique(self):
        ctx = synthetic_ctx(coords=[make_coord(), make_coord(page=11)])
        self.assertEqual(C.check_unique_ids(ctx)['status'], 'pass')


class JsonCsvTests(unittest.TestCase):
    def test_bbox_difference_detected(self):
        coord = make_coord()
        row = coord_to_csv_row(coord)
        row['main_bbox'] = json.dumps([10, 10, 31, 50])
        ctx = synthetic_ctx(coords=[coord], csv_rows=[row])
        res = C.check_json_csv(ctx)
        self.assertEqual(res['status'], 'fail')
        self.assertTrue(any('main_bbox' in d for d in res['differences']))

    def test_missing_row_detected(self):
        ctx = synthetic_ctx(coords=[make_coord(), make_coord(region='T10-02')],
                            csv_rows=[coord_to_csv_row(make_coord())])
        res = C.check_json_csv(ctx)
        self.assertEqual(res['status'], 'fail')

    def test_column_null_vs_empty_is_not_error(self):
        ctx = synthetic_ctx(coords=[make_coord(column=None)])
        res = C.check_json_csv(ctx)
        self.assertEqual(res['status'], 'pass')
        self.assertEqual(N.build(ctx)['groups'][0]['column_state'], 'default_absent')


class ConventionTests(unittest.TestCase):
    def test_measured_source_is_half_open(self):
        res = C.check_coordinate_convention(synthetic_ctx())
        self.assertEqual(res['status'], 'pass')
        self.assertEqual(res['confirmed_half_open'], ['测试書/p10/T10-01'])

    def test_manual_source_is_pending(self):
        coord = make_coord(source='output/small_columns_step1_holdout/annotations.json（人工标注）')
        res = C.check_coordinate_convention(synthetic_ctx(coords=[coord]))
        self.assertEqual(res['status'], 'pending')
        self.assertEqual(res['pending'], ['测试書/p10/T10-01'])

    def test_pending_convention_limits_eligibility(self):
        coord = make_coord(source='docs/双列小字_步骤3有限覆盖准入与新书页.md 人工近似框')
        g = N.build(synthetic_ctx(coords=[coord]))['groups'][0]
        self.assertEqual(g['coordinate_convention'], D.CONV_PENDING)
        self.assertEqual(g['original_coordinate_convention'], 'unknown')
        self.assertIn('glyph_exact_boundary', g['evaluation_eligibility']['not_usable_for'])
        exact = g['evaluation_eligibility']['exact_boundary_evaluation']
        self.assertFalse(exact['allowed'])
        self.assertIn('approximate_box', exact['blockers'])
        self.assertIn('endpoint_unresolved', exact['blockers'])
        self.assertEqual(exact['requires'], ['protocol_tolerance'])

    def test_half_open_group_is_usable(self):
        g = N.build(synthetic_ctx())['groups'][0]
        self.assertEqual(g['evaluation_eligibility']['status'], 'usable')
        exact = g['evaluation_eligibility']['exact_boundary_evaluation']
        self.assertFalse(exact['allowed'])
        # 端点已确认，但三类来源都没做字形边界标定，精确边界仍不可直接使用
        self.assertEqual(exact['blockers'], ['no_boundary_calibration'])

    def test_endpoint_interpretation_is_separate_from_error(self):
        coord = make_coord(source='output/small_columns_step1_holdout/annotations.json（人工标注）')
        g = N.build(synthetic_ctx(coords=[coord]))['groups'][0]
        ep = g['endpoint_interpretation']
        self.assertEqual(ep['status'], 'pending')
        self.assertEqual(ep['edge_shift_px'], 1)
        self.assertEqual(sorted(ep['candidates']),
                         sorted(['half_open:[x1,y1,x2,y2)', 'inclusive:[x1,y1,x2,y2]']))
        # 端点解释差异不能变成标注误差
        self.assertEqual(g['glyph_boundary_error']['status'], 'unknown_uncalibrated')
        self.assertIsNone(g['glyph_boundary_error']['value_px'])

    def test_no_uncalibrated_pixel_error_is_invented(self):
        for src in ('本轮测量（栏内墨迹外接框，6倍叠框核对）',
                    'output/small_columns_step1_holdout/annotations.json（人工标注）',
                    'docs/双列小字_步骤3有限覆盖准入与新书页.md 人工近似框'):
            g = N.build(synthetic_ctx(coords=[make_coord(source=src)]))['groups'][0]
            self.assertIsNone(g['glyph_boundary_error']['value_px'])
            self.assertEqual(g['glyph_boundary_error']['calibration'], 'none')
            self.assertNotIn('expected_error_px', json.dumps(g))
            self.assertNotIn('expected_error_px', json.dumps(D.source_profile(
                D.classify_source(src))))

    def test_protocol_tolerance_is_pending_not_a_number(self):
        g = N.build(synthetic_ctx())['groups'][0]
        tol = g['evaluation_tolerance_px']
        self.assertEqual(tol['status'], 'pending_protocol')
        self.assertIsNone(tol['value_px'])

    def test_glyph_boundary_check_marks_uninspected(self):
        g = N.build(synthetic_ctx())['groups'][0]
        self.assertEqual(g['glyph_boundary_check']['status'], 'not_inspected')
        self.assertTrue(g['glyph_boundary_check']['evidence_image'])

    def test_unclassifiable_source_raises(self):
        with self.assertRaises(ValueError):
            D.classify_source('某个没见过的来源')


class BboxParsingTests(unittest.TestCase):
    """异常数字必须给出明确的非法结果，且永不抛异常。"""

    def test_non_finite_is_invalid(self):
        for bad in (float('inf'), float('-inf'), float('nan')):
            bbox, code, detail = D.parse_bbox([0, 0, bad, 10])
            self.assertIsNone(bbox)
            self.assertEqual(code, D.BBOX_NON_FINITE)
            self.assertIn('第 2 个元素', detail)
            self.assertIsNone(D.as_bbox([0, 0, bad, 10]))

    def test_bool_is_rejected(self):
        for bad in (True, False):
            bbox, code, _ = D.parse_bbox([bad, 0, 5, 5])
            self.assertIsNone(bbox)
            self.assertEqual(code, D.BBOX_BOOL)

    def test_non_integral_float_is_rejected(self):
        bbox, code, _ = D.parse_bbox([0.5, 0, 5, 5])
        self.assertIsNone(bbox)
        self.assertEqual(code, D.BBOX_NON_INTEGRAL)

    def test_numeric_string_convention(self):
        # 约定：只接受 int(s, 10) 能完整解析的十进制整数字符串
        self.assertEqual(D.as_bbox(['899', '236', ' 933 ', '+316']), (899, 236, 933, 316))
        for bad in ('899.0', '1e3', '0x10', '', 'abc', ' '):
            bbox, code, _ = D.parse_bbox([bad, 0, 5, 5])
            self.assertIsNone(bbox, f'{bad!r} 应判非法')
            self.assertEqual(code, D.BBOX_NOT_NUMERIC)

    def test_integral_float_is_normalized(self):
        self.assertEqual(D.as_bbox([1.0, 2.0, 3.0, 4.0]), (1, 2, 3, 4))
        self.assertEqual(D.parse_bbox([1.0, 2.0, 3.0, 4.0])[1], D.BBOX_OK)

    def test_legal_integers_unchanged(self):
        self.assertEqual(D.as_bbox([899, 236, 933, 316]), (899, 236, 933, 316))

    def test_shape_errors(self):
        self.assertEqual(D.parse_bbox(None)[1], D.BBOX_EMPTY)
        self.assertEqual(D.parse_bbox([1, 2, 3])[1], D.BBOX_LENGTH)
        self.assertEqual(D.parse_bbox('1,2,3,4')[1], D.BBOX_NOT_SEQUENCE)
        self.assertEqual(D.parse_bbox([object(), 0, 1, 1])[1], D.BBOX_NOT_NUMERIC)

    def test_never_raises_on_nasty_input(self):
        nasty = [[0, 0, float('inf'), 10], [0, 0, float('nan'), 10], [True, 0, 1, 1],
                 None, [], 'x', 3, {'a': 1}, [1, 2], [None] * 4,
                 [0, 0, 10 ** 40, 10], [b'1', 0, 1, 1]]
        for value in nasty:
            try:
                D.parse_bbox(value)
                D.as_bbox(value)
            except Exception as exc:  # noqa: BLE001 - 这里就是要证明不抛
                self.fail(f'parse_bbox/as_bbox 对 {value!r} 抛了 {type(exc).__name__}: {exc}')


class InvalidRecordRobustnessTests(unittest.TestCase):
    """一条记录坐标非法时，要出具体明细，且不影响其他记录。"""

    def _mixed_ctx(self):
        return synthetic_ctx(coords=[
            make_coord(region='T10-01'),
            make_coord(region='T10-02', right_bbox=[0, 0, float('inf'), 10]),
            make_coord(region='T10-03'),
        ])

    def test_bad_record_yields_detail_and_others_still_checked(self):
        ctx = self._mixed_ctx()
        shape = C.check_bbox_shape(ctx)
        self.assertEqual(shape['status'], 'fail')
        self.assertEqual(len(shape['invalid']), 1)
        item = shape['invalid'][0]
        self.assertEqual(item['group_id'], '测试書/p10/T10-02')
        self.assertEqual(item['field'], 'right_bbox')
        self.assertEqual(item['code'], D.BBOX_NON_FINITE)
        self.assertIn('inf', str(item['value']))

        geo = C.check_bbox_geometry(ctx)
        self.assertEqual(geo['status'], 'fail')
        self.assertEqual(len(geo['skipped_invalid']), 1)
        self.assertEqual(len(geo['rows']), 8)          # 3 组 × 3 框 - 1 个非法
        self.assertTrue(all(r['in_image'] for r in geo['rows']))

    def test_downstream_checks_do_not_crash(self):
        ctx = self._mixed_ctx()
        self.assertEqual(C.check_lane_order(ctx)['status'], 'fail')
        self.assertGreaterEqual(len(C.check_lane_order(ctx)['skipped_invalid']), 0)
        self.assertIn(C.check_overlaps(ctx)['status'], ('pass', 'pending', 'fail'))
        report = C.run_checks(ctx)
        self.assertEqual(len(report['checks']), len(C.CHECKS))
        json.dumps(report)  # 报告本身必须可序列化（inf 已转文本）

    def test_report_is_json_serializable_with_inf(self):
        ctx = synthetic_ctx(coords=[make_coord(main_bbox=[0, 0, float('-inf'), 10])])
        report = C.run_checks(ctx)
        text = json.dumps(report, ensure_ascii=False)
        self.assertIn('inf', text)


class ManifestConsistencyTests(unittest.TestCase):
    """清单 JSON/CSV 人为差异必须被检出（内存构造，不动原始文件）。"""

    @classmethod
    def setUpClass(cls):
        cls.base_json = [dict(r) for r in D.load_manifest()]
        cls.base_csv = [dict(r) for r in D.load_manifest_csv()]

    def _ctx(self, json_rows=None, csv_rows=None):
        return C.Context(manifest=json_rows if json_rows is not None else [dict(r) for r in self.base_json],
                         manifest_csv=csv_rows if csv_rows is not None else [dict(r) for r in self.base_csv],
                         coords=[], coords_csv=[], image_overrides={})

    def test_real_files_pass(self):
        res = C.check_manifest_json_csv(self._ctx())
        self.assertEqual(res['status'], 'pass', res['differences'])
        self.assertEqual(res['differences'], [])
        self.assertEqual(res['counts']['json'], 23)
        self.assertEqual(res['counts']['csv'], 23)

    def test_sha256_change_detected(self):
        rows = [dict(r) for r in self.base_csv]
        rows[0]['sha256'] = 'f' * 64
        res = C.check_manifest_json_csv(self._ctx(csv_rows=rows))
        self.assertEqual(res['status'], 'fail')
        hit = [d for d in res['differences'] if d['field'] == 'sha256']
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]['kind'], 'value_mismatch')
        self.assertEqual(hit[0]['a'], self.base_json[0]['sha256'])
        self.assertEqual(hit[0]['b'], 'f' * 64)
        self.assertIn('page_33', hit[0]['key'])

    def test_deleted_row_detected(self):
        rows = [dict(r) for r in self.base_csv][1:]
        res = C.check_manifest_json_csv(self._ctx(csv_rows=rows))
        self.assertEqual(res['status'], 'fail')
        kinds = {d['kind'] for d in res['differences']}
        self.assertIn('count_mismatch', kinds)
        self.assertIn('missing_in_csv', kinds)
        missing = [d for d in res['differences'] if d['kind'] == 'missing_in_csv']
        self.assertEqual(len(missing), 1)
        self.assertIn('page_33', missing[0]['key'])

    def test_duplicate_row_detected(self):
        rows = [dict(r) for r in self.base_csv]
        rows.append(dict(rows[0]))
        res = C.check_manifest_json_csv(self._ctx(csv_rows=rows))
        self.assertEqual(res['status'], 'fail')
        dup = [d for d in res['differences'] if d['kind'] == 'duplicate']
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0]['a'], 2)
        self.assertIn('page_33', dup[0]['key'])

    def test_status_change_detected(self):
        rows = [dict(r) for r in self.base_csv]
        rows[0]['status'] = '排除'
        res = C.check_manifest_json_csv(self._ctx(csv_rows=rows))
        self.assertEqual(res['status'], 'fail')
        hit = [d for d in res['differences'] if d['field'] == 'status']
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]['b'], '排除')
        self.assertEqual(hit[0]['a'], '收录')

    def test_dataset_path_change_detected(self):
        rows = [dict(r) for r in self.base_csv]
        rows[1]['dataset_path'] = '古籍评测集/双轨小字页/思濟堂方書/page_99.png'
        res = C.check_manifest_json_csv(self._ctx(csv_rows=rows))
        self.assertEqual(res['status'], 'fail')
        hit = [d for d in res['differences'] if d['field'] == 'dataset_path']
        self.assertEqual(len(hit), 1)
        self.assertIn('page_99', hit[0]['b'])

    def test_absent_dataset_path_matches_empty_cell(self):
        # 排除页：JSON 无该键、CSV 为空串，属同一「无」，不算差异
        res = C.check_manifest_json_csv(self._ctx())
        self.assertFalse([d for d in res['differences'] if d['field'] == 'dataset_path'])

    def test_missing_in_json_detected(self):
        rows = [dict(r) for r in self.base_json][1:]
        res = C.check_manifest_json_csv(self._ctx(json_rows=rows))
        self.assertEqual(res['status'], 'fail')
        kinds = {d['kind'] for d in res['differences']}
        self.assertIn('missing_in_json', kinds)

    def test_temp_file_roundtrip(self):
        """走真实文件：写到临时目录再读回，确认差异仍被检出，且原始清单不被触碰。"""
        import csv as _csv
        import hashlib
        import tempfile
        before = hashlib.sha256(D.MANIFEST_CSV.read_bytes()).hexdigest()
        rows = [dict(r) for r in self.base_csv]
        rows[2]['note'] = rows[2]['note'] + '（人为测试串）'
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '清单_测试.csv'
            with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            loaded = D.read_csv_rows(path)
            ctx = C.Context(manifest=[dict(r) for r in self.base_json], manifest_csv=loaded,
                            coords=[], coords_csv=[], image_overrides={})
            res = C.check_manifest_json_csv(ctx)
        self.assertEqual(res['status'], 'fail')
        hit = [d for d in res['differences'] if d['field'] == 'note']
        self.assertEqual(len(hit), 1)
        self.assertIn('人为测试串', hit[0]['b'])
        self.assertEqual(hashlib.sha256(D.MANIFEST_CSV.read_bytes()).hexdigest(), before)


class SchemaTests(unittest.TestCase):
    def test_group_has_required_fields(self):
        g = N.build(synthetic_ctx())['groups'][0]
        missing = [f for f in REQUIRED_FIELDS if f not in g]
        self.assertEqual(missing, [])

    def test_page_record_keeps_pending_and_excluded(self):
        manifest = [
            make_manifest(),
            make_manifest(page='page_11.png', status='待核', confidence='待核', note='边界待核',
                          main='丁', right='戊', left='己',
                          dataset_path='古籍评测集/双轨小字页/待核/测试書_page_11.png'),
            make_manifest(page='page_12.png', status='排除', confidence='—', note='序',
                          main='', right='', left='', dataset_path=None),
        ]
        payload = N.build(synthetic_ctx(manifest=manifest))
        pages = {p['page']: p for p in payload['pages']}
        self.assertEqual(pages[11]['main_bbox'], None)
        self.assertEqual(pages[11]['group_ids'], [])
        self.assertEqual(pages[11]['box_state'], 'absent_pending_review')
        self.assertTrue(pages[11]['reason'])
        self.assertEqual(pages[12]['box_state'], 'not_applicable_excluded')
        self.assertEqual(payload['counts']['pages_by_status'],
                         {'待核': 1, '排除': 1, '收录': 1})

    def test_coverage_marked_partial(self):
        payload = N.build(synthetic_ctx())
        self.assertEqual(payload['annotation_coverage'], D.COVERAGE_PARTIAL)
        self.assertEqual(payload['groups'][0]['annotation_coverage'], D.COVERAGE_PARTIAL)


class DeterminismTests(unittest.TestCase):
    def test_two_runs_produce_identical_payloads(self):
        a = json.dumps(N.build(synthetic_ctx())['groups'], ensure_ascii=False, sort_keys=True)
        b = json.dumps(N.build(synthetic_ctx())['groups'], ensure_ascii=False, sort_keys=True)
        self.assertEqual(a, b)

    def test_output_files_byte_identical(self):
        ctx = synthetic_ctx()
        with tempfile.TemporaryDirectory() as td:
            p1 = N.write_outputs(N.build(ctx), Path(td) / 'a')
            p2 = N.write_outputs(N.build(ctx), Path(td) / 'b')
            for k in p1:
                self.assertEqual(Path(p1[k]).read_bytes(), Path(p2[k]).read_bytes(), f'{k} 两次运行不一致')

    def test_no_timestamp_in_normalized_file(self):
        with tempfile.TemporaryDirectory() as td:
            paths = N.write_outputs(N.build(synthetic_ctx()), Path(td))
            text = Path(paths['normalized_annotations']).read_text(encoding='utf-8')
            self.assertNotIn('generated_at', text)


class RealDatasetTests(unittest.TestCase):
    """真实数据集上的固定事实：这些数字变了说明来源被动过，需要人工确认。"""

    @classmethod
    def setUpClass(cls):
        cls.ctx = C.Context()
        cls.payload = N.build(cls.ctx)
        cls.report = C.run_checks(cls.ctx)

    def test_counts(self):
        c = self.payload['counts']
        self.assertEqual(c['groups_total'], 26)
        self.assertEqual(c['pages_total'], 23)
        self.assertEqual(c['pages_by_status'], {'待核': 2, '排除': 8, '收录': 13})
        self.assertEqual(c['groups_by_source'], {'人工近似框': 2, '历史人工标注': 21, '墨迹测量': 3})

    def test_check_tally(self):
        s = self.report['summary']
        self.assertEqual(s['checks'], 14)
        self.assertEqual(s['failed_ids'], ['C02', 'C13'])
        self.assertEqual(s['pending_ids'], ['C09', 'C12'])

    def test_manifest_consistency_on_real_files(self):
        res = [c for c in self.report['checks'] if c['id'] == 'C14'][0]
        self.assertEqual(res['status'], 'pass')
        self.assertEqual(res['differences'], [])

    def test_precision_semantics_on_real_data(self):
        sem = self.payload['precision_semantics']
        self.assertIsNone(sem['glyph_boundary_error']['magnitude'])
        self.assertIsNone(sem['evaluation_tolerance_px']['magnitude'])
        counts = self.payload['counts']
        self.assertEqual(counts['groups_by_endpoint_status'], {'pending': 23, 'resolved': 3})
        self.assertEqual(counts['groups_by_glyph_check'],
                         {'inspected_ok': 10, 'inspected_uncertain': 1, 'not_inspected': 15})
        for g in self.payload['groups']:
            self.assertIsNone(g['glyph_boundary_error']['value_px'])
            self.assertEqual(g['evaluation_tolerance_px']['status'], 'pending_protocol')

    def test_consistency_between_normalized_and_source(self):
        cor = {'entries': self.payload['corrections']}
        ann = {'groups': self.payload['groups']}
        res = N.verify_consistency(ann, cor)
        self.assertTrue(res['ok'])
        self.assertEqual(res['checked_groups'], 26)

    def test_every_group_traces_back_to_source(self):
        for g in self.payload['groups']:
            self.assertTrue(g['source_reference']['coordinates_json'])
            self.assertTrue(g['image_sha256'])
            self.assertIn('双轨坐标', g['source_reference']['coordinates_json'])
            self.assertEqual(len(g['image_size']), 2)


if __name__ == '__main__':
    unittest.main()
