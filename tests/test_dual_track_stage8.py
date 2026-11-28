# -*- coding: utf-8 -*-
"""阶段8 回归：列槽周期 / 子框裁剪与墨迹收紧 / 放宽补丁可还原 / 默认配置不改变生产行为。

设计原则：只用合成图与内存对象，不触碰真实页与模型，
保证本文件可以在无 OCR 模型的机器上跑通。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402

from dual_track_fix import relax  # noqa: E402
from dual_track_fix.detect import DEFAULT, _clip, _ink_cols, _tighten  # noqa: E402
from dual_track_fix.slots import estimate_pitch  # noqa: E402


def _page(h=200, w=320, ink=0, bg=255) -> np.ndarray:
    """白底灰度图（ink 为笔画灰度，越小越黑）。"""
    return np.full((h, w), bg, dtype=np.uint8)


class PitchTests(unittest.TestCase):
    """列槽周期 = 投影自相关的第一个主峰，且落在 [1.3H, 3.2H]。"""

    def test_periodic_stripes_recover_pitch(self):
        img = _page()
        for x in range(20, 320, 60):
            img[20:180, x:x + 8] = 0
        proj = (255 - img).astype(np.float64).sum(axis=0) / 255.0
        pitch = estimate_pitch(proj, H=20)
        self.assertIsNotNone(pitch)
        self.assertTrue(55 <= pitch <= 65, f'周期应≈60，实际 {pitch}')

    def test_flat_page_has_no_pitch(self):
        proj = np.zeros(320)
        self.assertIsNone(estimate_pitch(proj, H=20))

    def test_too_short_projection_returns_none(self):
        # 长度不足 3.2H 时无法在半页内找峰，必须返回 None 而不是报错
        self.assertIsNone(estimate_pitch(np.ones(40), H=20))


class InkColumnTests(unittest.TestCase):
    """_ink_cols 取墨迹 x 外接，且必须排除贯穿框高的竖线。"""

    def _gray(self):
        g = _page(h=100, w=120)
        g[10:90, 45:48] = 0          # 贯穿框高的竖线（栏线/版框）
        g[30:70, 60:71] = 0          # 字形墨迹
        return g

    def test_vertical_line_is_excluded(self):
        r = _ink_cols(self._gray(), 0, 10, 120, 90)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r[0], 50, f'竖线 x=45..47 未被排除：{r}')
        self.assertLessEqual(r[1], 80)

    def test_box_too_small_returns_none(self):
        self.assertIsNone(_ink_cols(self._gray(), 0, 10, 2, 12))


class ClipTests(unittest.TestCase):
    """子框裁到列槽，但不能把框裁没（剩余宽度 < 6px 时退回原框）。"""

    def test_clip_to_slot(self):
        self.assertEqual(_clip([10, 20, 300, 400], [50, 0, 200, 500]),
                         [50, 20, 200, 400])

    def test_clip_keeps_box_inside_slot(self):
        self.assertEqual(_clip([60, 20, 180, 400], [50, 0, 200, 500]),
                         [60, 20, 180, 400])

    def test_slot_barely_overlaps_falls_back(self):
        self.assertEqual(_clip([60, 20, 180, 400], [178, 0, 200, 500]),
                         [60, 20, 180, 400])


class TightenTests(unittest.TestCase):
    """输出子框按墨迹 x 收紧；收紧过多（< min_keep）时放弃而不是交碎框。"""

    def _gray(self):
        g = _page(h=100, w=120)
        g[30:70, 60:71] = 0
        return g

    def test_tighten_to_ink(self):
        out = _tighten(self._gray(), [50, 20, 100, 80], min_keep=0.10)
        self.assertIsNotNone(out)
        self.assertEqual(out[1], 20)
        self.assertEqual(out[3], 80, 'y 方向不得改变')
        self.assertTrue(58 <= out[0] <= 62, out)
        self.assertTrue(69 <= out[2] <= 73, out)

    def test_too_narrow_gives_none(self):
        # 墨迹仅 10px，占 50px 框宽的 0.2 < 0.45 → 放弃
        self.assertIsNone(_tighten(self._gray(), [50, 20, 100, 80], min_keep=0.45))


class SpanLimitTests(unittest.TestCase):
    """v2 起跨度上限是**调用参数**，不是 monkeypatch：换算要稳，旧入口要被拦住。"""

    def test_none_means_historical_150(self):
        self.assertEqual(relax.span_limit_for(873, None), 150)
        self.assertEqual(relax.DEFAULT_SPAN_LIMIT, 150)

    def test_ratio_times_column_height(self):
        # p12 右半页列高 871、比例 0.9 → 784
        self.assertEqual(relax.span_limit_for(871, 0.9), 784)
        # 短列不得低于 150（思濟堂短注列高 100 时仍是 150）
        self.assertEqual(relax.span_limit_for(100, 0.9), 150)

    def test_bad_input_falls_back(self):
        self.assertEqual(relax.span_limit_for('oops', 0.9), 150)
        self.assertEqual(relax.span_limit_for(None, 0.9), 150)

    def test_install_is_gone_and_refuses(self):
        # 残留调用必须炸，而不是「装上了但没生效」
        with self.assertRaises(RuntimeError):
            relax.install()

    def test_propose_in_column_keeps_legacy_default(self):
        import inspect
        import ocr.small_columns as sc
        sig = inspect.signature(sc.propose_in_column)
        self.assertIsNone(sig.parameters['span_ratio'].default,
                          'span_ratio 默认必须是 None（= 150px 历史行为）')
        self.assertEqual(sc.SPAN_LIMIT, 150)


class DefaultConfigTests(unittest.TestCase):
    """生产默认必须保持 stage5；stage8 的每个开关都要能被显式关掉。"""

    def test_stage8_switches_default_to_the_frozen_choice(self):
        self.assertTrue(DEFAULT['slots'])
        self.assertTrue(DEFAULT['clip'])
        self.assertTrue(DEFAULT['tighten'])
        self.assertTrue(DEFAULT['use_limits'])
        self.assertTrue(DEFAULT['relax_span'] is False or DEFAULT['relax_span'] is True)
        self.assertFalse(DEFAULT['dedupe_slot'], '同槽去重是负结果，必须默认关闭')

    def test_production_detector_defaults_to_stage5(self):
        import os
        if 'TCM_OCR_DUAL_TRACK_DETECTOR' in os.environ:
            self.skipTest('环境变量已覆盖默认值')
        from app.config import CONFIG
        self.assertEqual(getattr(CONFIG.ocr, 'dual_track_detector', 'stage5'), 'stage5')


class BandSlotTests(unittest.TestCase):
    """N-10 分栏相位（负结果）：开关默认关，机制本身要能被单测钉住。

    这一组不是「功能测试」，是防腐测试——既然把代码留在了库里（供下一轮在
    split_columns 层复用），就必须保证：默认不动、相位搜索落在合法位置、
    两栏相位差足够大时才判 band_split。
    """

    def test_band_slots_default_off(self):
        from dual_track_fix.settings import FROZEN, effective_options
        self.assertFalse(FROZEN['band_slots'])
        eff = effective_options()
        self.assertFalse(eff['options']['band_slots'])

    def test_fingerprint_changes_when_switched_on(self):
        from dual_track_fix.settings import effective_options
        a = effective_options()
        b = effective_options({'band_slots': True})
        self.assertNotEqual(a['fingerprint'], b['fingerprint'])

    def test_phase_search_stays_inside_column(self):
        import numpy as np
        from dual_track_fix.slots import best_phase_cuts
        # 合成投影：周期 40、相位 12（墨迹集中在相位处），求出的切点必须避开墨
        proj = np.zeros(400)
        for k in range(0, 400, 40):
            proj[k + 12:k + 20] = 30
        best = best_phase_cuts(proj, 100, 300, 40, 0.45)
        self.assertIsNotNone(best)
        for c in best['cuts']:
            self.assertTrue(100 < c < 300)
            self.assertLess(proj[c - 1:c + 2].sum(), 30 * 3 * 2, '切点落在墨迹上')

    def test_phase_search_rejects_too_narrow_slots(self):
        import numpy as np
        from dual_track_fix.slots import best_phase_cuts
        # 宽 45、周期 40 且要求最小槽宽 0.45×周期=18：两段宽 22/23 之外切不开
        proj = np.ones(400) * 5
        best = best_phase_cuts(proj, 100, 145, 40, 0.45)
        self.assertTrue(best is None or len(best['cuts']) <= 1)


class JointAndValidationTests(unittest.TestCase):
    """v3：跨槽联合取样 + 校验状态实际化 + 首尾吸附的防腐测试。"""

    def test_joint_rejects_when_main_does_not_span_boundary(self):
        from dual_track_fix.joint import joint_propose
        gray = np.full((120, 200), 200, dtype=np.uint8)
        # 药名带不跨边界 → 不触发
        r = joint_propose(gray, [20, 0, 100, 120], [100, 0, 180, 120],
                          [30, 10, 90, 40], 100, 21.0, 118)
        self.assertIsNone(r)

    def test_joint_rejects_when_no_seam_valley_at_boundary(self):
        from dual_track_fix.joint import joint_propose
        gray = np.full((200, 200), 200, dtype=np.uint8)
        # 药名跨边界 100，但两侧无墨迹 → 中缝不在边界 → 不触发
        r = joint_propose(gray, [20, 0, 100, 200], [100, 0, 180, 200],
                          [60, 20, 140, 50], 100, 21.0, 190)
        self.assertIsNone(r)

    def test_joint_emits_full_lanes_on_synthetic_dual_track(self):
        from dual_track_fix.joint import joint_propose
        gray = np.full((300, 240), 230, dtype=np.uint8)
        # 药名带跨边界 x=120（y 20-40），下方两轨紧邻中缝：[88,119]/[121,152]
        # 缝只有 2px（与真实 p12 一致），谷值才会落在边界 120
        gray[20:40, 80:160] = 40
        for yy in range(60, 250, 8):
            gray[yy:yy + 5, 88:119] = 30
            gray[yy:yy + 5, 121:152] = 30
        r = joint_propose(gray, [60, 0, 120, 300], [120, 0, 180, 300],
                          [80, 20, 160, 40], 120, 21.0, 270)
        self.assertIsNotNone(r)
        self.assertLessEqual(abs(r['seam_x'] - 120), 1)   # 缝在 [119,121)，谷可取任一端
        self.assertLess(r['left_bbox'][2], r['seam_x'])
        self.assertGreaterEqual(r['right_bbox'][0], r['seam_x'])
        self.assertEqual(r['evidence']['method'], 'joint_sample_across_slot_boundary')

    def test_validate_lane_reports_conflict_not_ok(self):
        from ocr.dual_track import _stage8_validate_lane
        ink = np.zeros((200, 100), dtype=np.uint8)
        ink[30:150, 20:40] = 1
        # 栏线与轨框实质相交 → 证据冲突，必须 ok=False
        fv = _stage8_validate_lane([20, 30, 40, 150], ink,
                                   [[10, 60, 50, 70]], {}, 21.0)
        self.assertFalse(fv['ok'])
        self.assertIn('rule_conflict', fv['conflict_reasons'])
        self.assertEqual(fv['status'], 'rejected_to_tentative')

    def test_head_that_cannot_be_completed_is_pending(self):
        from ocr.dual_track import _stage8_record

        ink = np.zeros((200, 100), dtype=np.uint8)
        ink[26:40, 20:40] = 1
        box = [20, 32, 40, 80]
        group = {'group_id': 'g', 'page_id': 'p', 'group_bbox': box, 'left_bbox': box}
        rec = _stage8_record(group, [0, 0, 100, 200], ink)
        fv = rec['sides']['left']['final_validation']
        self.assertFalse(fv['ok'])
        self.assertIn('cut_head', fv['review_reasons'])
        self.assertEqual(fv['status'], 'pending_cut_review')
        self.assertEqual(rec['sides']['left']['bbox_after'], box)

    def test_stage8_record_sets_real_validation_status(self):
        from ocr.dual_track import _stage8_record
        ink = np.zeros((200, 100), dtype=np.uint8)
        ink[40:100, 20:40] = 1
        g = {'group_id': 'g', 'page_id': 'p', 'group_bbox': [20, 40, 60, 100],
             'left_bbox': [20, 40, 40, 100], 'right_bbox': [40, 40, 60, 100],
             'scale_used': {'H': 21}}
        rec = _stage8_record(g, [0, 0, 100, 200], ink, [[10, 50, 50, 60]], {})
        self.assertEqual(rec['refine'], 'validated_stage8_v4')
        self.assertTrue(rec['sides']['left']['final_validation']['conflict_reasons'])

    def test_record_keeps_group_when_review_reason_is_none(self):
        """待核截笔且校验器 reason=None 时不得抛错丢组（温疫页 38→36 的根因）。"""
        import ocr.dual_track as DT
        ink = np.zeros((200, 100), dtype=np.uint8)
        ink[40:100, 20:40] = 1
        g = {'group_id': 'g', 'page_id': 'p', 'group_bbox': [20, 40, 60, 100],
             'left_bbox': [20, 40, 40, 100], 'right_bbox': [40, 40, 60, 100],
             'scale_used': {'H': 21}}
        def _fv(*a, **k):
            # 每次返回新字典；复用同一个 dict 会让左/右两侧串改同一份 reason。
            return {'ok': False, 'reason': None, 'status': 'pending_cut_review',
                    'review_reasons': [], 'conflict_reasons': []}

        def _review(*a, **k):
            return ([20, 40, 40, 100],
                    {'before': [20, 40, 40, 100], 'after': [20, 40, 40, 100],
                     'actions': [], 'unresolved': ['cut_head']})

        with mock.patch.object(DT, '_stage8_review_lane', side_effect=_review), \
                mock.patch.object(DT, '_stage8_validate_lane', side_effect=_fv):
            rec = DT._stage8_record(g, [0, 0, 100, 200], ink, [], {})
        fv = rec['sides']['left']['final_validation']
        self.assertFalse(fv['ok'])
        self.assertEqual(fv['reason'], 'cut_head')
        self.assertTrue(rec['low_confidence'])


class Stage8BoundaryReviewTests(unittest.TestCase):
    def test_acceptance_checks_current_side_crop_paths(self):
        import tempfile
        from pathlib import Path
        from tools.dual_track_stage8_accept import check_evidence_crops

        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'old.png'
            existing.write_bytes(b'old image')
            evidence = [
                {'evidence_id': 'g:right', 'side': 'right', 'side_status': 'ok',
                 'crop_path': str(existing)},
                {'evidence_id': 'g:left', 'side': 'left', 'side_status': 'pending_validation',
                 'review_evidence': {'image_path': str(Path(tmp) / 'missing.png')}},
            ]
            result = check_evidence_crops(evidence)
            self.assertFalse(result['pass'])
            self.assertEqual(result['n_crops'], 1)
            self.assertEqual(result['missing'][0]['evidence_id'], 'g:left')

    def test_head_recovers_complete_owned_glyph_beyond_short_margin(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[22:37, 24:34] = 1
        ink[45:58, 24:34] = 1
        ink[66:79, 24:34] = 1
        lane = self._right_lane(ink, [20, 29, 40, 90],
                                main_bbox=[15, 2, 42, 19],
                                evidence={'valley_x': 20})
        self.assertEqual(lane['bbox_after'], [20, 22, 40, 90])
        self.assertTrue(lane['final_validation']['ok'])

    def test_head_recovery_does_not_pull_in_title_ink(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[22:37, 24:34] = 1
        ink[45:58, 24:34] = 1
        ink[66:79, 24:34] = 1
        lane = self._right_lane(ink, [20, 29, 40, 90],
                                main_bbox=[15, 2, 42, 27],
                                evidence={'valley_x': 20})
        self.assertEqual(lane['bbox_after'], [20, 29, 40, 90])
        self.assertIn('cut_head', lane['boundary_review']['unresolved'])

    def test_small_connected_edge_stroke_requires_review(self):
        from ocr.dual_track import _stage8_record

        ink = np.zeros((120, 70), np.uint8)
        ink[28:32, 25] = 1
        ink[45:55, 24:32] = 1
        box = [20, 30, 40, 85]
        group = {'group_id': 'g', 'page_id': 'p', 'group_bbox': box,
                 'right_bbox': box, 'scale_used': {'H': 21}}
        lane = _stage8_record(group, [0, 0, 70, 120], ink)['sides']['right']
        self.assertFalse(lane['final_validation']['ok'])
        self.assertIn('cut_head', lane['boundary_review']['unresolved'])

    def test_frame_is_detected_from_a_rule_spanning_neighbouring_columns(self):
        from dual_track_fix.detect import column_frame_top

        gray = np.full((200, 220), 235, np.uint8)
        gray[183:186, 78:101] = 30
        slot = [70, 0, 115, 200]
        self.assertIsNone(column_frame_top(gray, slot, 21))
        gray[188:190, 30:180] = 30
        self.assertEqual(column_frame_top(gray, slot, 21), 188)

    def test_rule_review_crop_preserves_pixels_outside_mask(self):
        from dual_track_fix.ruling import masked_crop, ruling_mask
        from ocr.dual_track import _stage8_ink_components

        ink = np.zeros((140, 110), np.uint8)
        ink[15:121, 24:26] = 1
        ink[120:122, 6:100] = 1
        ink[30:53, 34:57] = 1
        image = np.repeat((235 - 200 * ink)[..., None], 3, axis=2)
        original = image.copy()
        mask = ruling_mask(ink, 21, *_stage8_ink_components(ink))
        clean, pixels = masked_crop(image, mask)
        self.assertTrue(mask[100, 24])
        self.assertTrue(mask[120, 49])
        self.assertFalse(mask[40, 38])
        np.testing.assert_array_equal(clean[~mask], image[~mask])
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(pixels > 0, mask)

    def test_confirmation_requires_ink_owned_on_both_sides(self):
        from ocr.dual_track import _stage8_ink_components, _stage8_pair_support

        ink = np.zeros((110, 90), np.uint8)
        ink[28:42, 27:40] = 1
        ink[50:65, 26:41] = 1
        ink[12:90, 51:53] = 1
        group = {'group_bbox': [22, 25, 70, 70],
                 'left_bbox': [22, 25, 45, 70], 'right_bbox': [46, 25, 70, 70],
                 'evidence': {'core_y': [22, 73], 'valley_x': 46}}
        self.assertEqual(_stage8_pair_support(group, *_stage8_ink_components(ink), 21),
                         {'left': 407, 'right': 0})
        ink[31:46, 55:68] = 1
        self.assertEqual(_stage8_pair_support(group, *_stage8_ink_components(ink), 21),
                         {'left': 407, 'right': 195})

    def test_rule_attached_glyph_can_certify_pair_without_changing_source(self):
        from ocr.dual_track import (_stage8_ink_components, _stage8_pair_support,
                                    _stage8_support_without_rules)

        ink = np.zeros((110, 90), np.uint8)
        ink[10:100, 24:26] = 1
        ink[32:47, 25:40] = 1
        ink[55:66, 53:68] = 1
        original = ink.copy()
        group = {'group_bbox': [23, 27, 70, 80],
                 'left_bbox': [23, 27, 45, 80], 'right_bbox': [46, 27, 70, 80],
                 'evidence': {'core_y': [27, 80], 'valley_x': 46}}
        self.assertEqual(_stage8_pair_support(group, *_stage8_ink_components(ink), 21)['left'], 0)
        self.assertEqual(_stage8_pair_support(group, *_stage8_support_without_rules(ink, 21),
                                              21), {'left': 210, 'right': 165})
        np.testing.assert_array_equal(ink, original)

    def test_only_independent_outer_rule_is_excluded(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[30:41, 24:32] = 1
        ink[67:78, 25:33] = 1
        ink[25:112, 38:40] = 1
        box = [20, 30, 41, 80]
        lane = self._right_lane(ink, box)
        self.assertEqual(lane['bbox_after'], [20, 30, 38, 80])
        self.assertEqual(lane['boundary_review']['actions'][0]['kind'],
                         'outer_rule_exclusion')

        ink[45:62, 32:43] = 1   # 实字与栏线粘连，不能裁去这部分
        lane = self._right_lane(ink, box)
        self.assertEqual(lane['bbox_after'], box)
        self.assertIn('cut_tail', lane['boundary_review']['unresolved'])

    def test_short_title_window_is_superseded_without_dropping_disjoint_notes(self):
        from dual_track_fix.detect import _superseded_windows

        def group(gid, head, tail, core, title, seam=40):
            return {'group_id': gid, 'region_id': 'slot', 'group_bbox': [20, head, 60, tail],
                    'main_bbox': [20, title[0], 60, title[1]],
                    'evidence': {'core_y': core, 'valley_x': seam}}

        short = group('false_split', 70, 108, [62, 99], [32, 68])
        confirmed = group('confirmed', 115, 240, [117, 234], [76, 111])
        separated = group('earlier_note', 24, 58, [23, 56], [7, 20])
        kept, audit = _superseded_windows([short, confirmed, separated], 21)
        self.assertEqual([g['group_id'] for g in kept], ['confirmed', 'earlier_note'])
        self.assertEqual(audit[0]['group_id'], 'false_split')
        self.assertEqual(audit[0]['superseded_by'], 'confirmed')

    def _right_lane(self, ink, box, **group_fields):
        from ocr.dual_track import _stage8_record

        group = {
            'group_id': 'g', 'page_id': 'p', 'group_bbox': box,
            'right_bbox': box, 'scale_used': {'H': 21}, **group_fields,
        }
        return _stage8_record(group, [0, 0, 70, 120], ink)['sides']['right']

    def test_leading_thin_stroke_inside_one_glyph_is_preserved(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[20:22, 24:30] = 1
        for offset in range(3):
            ink[22 + offset, 24 + offset] = 1
        ink[25:33, 26:32] = 1
        ink[50:57, 24:31] = 1

        box = [20, 20, 40, 90]
        lane = self._right_lane(ink, box)
        self.assertEqual(lane['bbox_after'], box)
        self.assertEqual(lane['boundary_review']['actions'], [])
        self.assertTrue(lane['final_validation']['ok'])

    def test_verified_title_is_removed_at_complete_component_boundary(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[5:24, 20:50] = 1
        ink[28:36, 34:47] = 1
        ink[48:58, 34:47] = 1

        lane = self._right_lane(ink, [32, 20, 50, 85], main_bbox=[15, 5, 55, 21])
        self.assertEqual(lane['bbox_after'], [32, 24, 50, 85])
        self.assertEqual(lane['boundary_review']['actions'][0]['kind'], 'title_trim')
        self.assertTrue(lane['final_validation']['ok'])

    def test_connected_title_stroke_can_be_narrower_than_title_band(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[12:36, 24:38] = 1
        ink[40:52, 34:47] = 1
        box = [31, 35, 50, 87]
        lane = self._right_lane(ink, box, main_bbox=[10, 12, 60, 35],
                                evidence={'valley_x': 31})
        self.assertEqual(lane['bbox_after'][1], 36)
        self.assertEqual(lane['boundary_review']['actions'][0]['kind'], 'title_trim')

    def test_tail_only_extends_if_complete_stroke_fits(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[32:42, 24:32] = 1
        ink[68:80, 24:32] = 1

        safe = self._right_lane(ink, [20, 30, 40, 78])
        self.assertEqual(safe['bbox_after'][3], 80)
        self.assertTrue(safe['final_validation']['ok'])

        clipped = self._right_lane(ink, [20, 30, 40, 72])
        self.assertEqual(clipped['bbox_after'][3], 72)
        self.assertFalse(clipped['final_validation']['ok'])
        self.assertIn('cut_tail', clipped['boundary_review']['unresolved'])

    def test_printed_frame_bounds_longer_tail_completion(self):
        ink = np.zeros((150, 150), np.uint8)
        ink[94:112, 62:77] = 1
        ink[114:117, 15:130] = 1
        box = [56, 45, 81, 105]

        lane = self._right_lane(ink, box)
        self.assertEqual(lane['bbox_after'], [56, 45, 81, 112])
        self.assertEqual(lane['boundary_review']['actions'][0]['kind'],
                         'tail_complete_before_frame')
        self.assertTrue(lane['final_validation']['ok'])

    def test_printed_frame_touching_tail_stroke_stays_pending(self):
        ink = np.zeros((150, 150), np.uint8)
        ink[94:115, 62:77] = 1
        ink[114:117, 15:130] = 1
        box = [56, 45, 81, 105]

        lane = self._right_lane(ink, box)
        self.assertEqual(lane['bbox_after'], box)
        self.assertIn('cut_tail', lane['boundary_review']['unresolved'])

    def test_tail_completion_can_expand_inside_its_own_seam(self):
        ink = np.zeros((150, 150), np.uint8)
        ink[94:112, 55:70] = 1
        ink[114:117, 15:130] = 1
        box = [56, 45, 81, 105]

        lane = self._right_lane(ink, box, evidence={'valley_x': 54})
        self.assertEqual(lane['bbox_after'], [55, 45, 81, 112])
        blocked = self._right_lane(ink, box, evidence={'valley_x': 56})
        self.assertEqual(blocked['bbox_after'], box)
        self.assertIn('cut_tail', blocked['boundary_review']['unresolved'])

    def test_following_confirmed_title_stops_only_the_overrunning_lane(self):
        from ocr.dual_track import _stage8_record

        ink = np.zeros((250, 100), np.uint8)
        for y in (31, 55, 79, 103):
            ink[y:y + 13, 51:65] = 1
            if y < 103:
                ink[y:y + 13, 23:39] = 1
        ink[144:168, 31:63] = 1
        for y in (177, 201, 225):
            ink[y:y + 11, 23:39] = 1
            ink[y:y + 11, 51:65] = 1
        upper = {'group_id': 'upper', 'page_id': 'p', 'region_id': 'r',
                 'group_bbox': [21, 30, 67, 238], 'left_bbox': [21, 30, 43, 95],
                 'right_bbox': [48, 30, 67, 238], 'scale_used': {'H': 21}}
        lower = {'group_id': 'lower', 'region_id': 'r', 'main_bbox': [20, 144, 69, 169],
                 'group_bbox': [21, 175, 67, 237],
                 'left_bbox': [21, 175, 43, 237], 'right_bbox': [48, 175, 67, 237],
                 'evidence': {'windows': 12}}
        record = _stage8_record(upper, [20, 0, 70, 250], ink, following=[lower])
        self.assertEqual(record['after']['right_bbox'], [48, 30, 67, 116])
        self.assertEqual(record['after']['left_bbox'], upper['left_bbox'])
        self.assertEqual(record['sides']['right']['boundary_review']['actions'][0]['kind'],
                         'following_title_stop')


class AutoJointAndReconcileTests(unittest.TestCase):
    """v4：原图自动触发跨槽候选；复核前框按复核前两条轨现算。"""

    def _synthetic_page(self):
        """标题跨槽边界 x=120，标题带内窄缝，正文两轨分落两槽。"""
        gray = np.full((280, 240), 230, dtype=np.uint8)
        gray[20:50, 80:117] = 30          # 标题左半
        gray[20:50, 124:160] = 30         # 标题右半（缝 117..124 = 7px）
        # 标题左半在槽边界处结束（m[2]=120），与 p12 的单槽 detector main 一致。
        for y in range(56, 250, 10):
            gray[y:y + 6, 88:119] = 25   # 左轨紧邻中缝
            gray[y:y + 6, 122:152] = 25  # 右轨紧邻中缝，谷值才能落在 120
        m = [80, 20, 120, 50]
        groups = [
            {'group_id': 'a', 'region_id': 'a', 'main_bbox': m,
             'group_bbox': [88, 56, 119, 250]},
            {'group_id': 'b', 'region_id': 'b', 'main_bbox': m,
             'group_bbox': [122, 56, 152, 250]},
        ]
        regions = [{'region_id': 'a', 'bbox': [60, 0, 120, 280]},
                   {'region_id': 'b', 'bbox': [120, 0, 180, 280]}]
        return gray, groups, regions

    def test_auto_joint_fires_from_image_only(self):
        from dual_track_fix.auto_joint import auto_joint_candidates
        gray, groups, regions = self._synthetic_page()
        out = auto_joint_candidates(gray, groups, regions, 20.0, 260)
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual(c['trigger']['boundary_x'], 120)
        self.assertEqual(c['trigger']['title_gap_x'], [117, 124])
        self.assertEqual(c['trigger']['section_y'][0], 56)
        self.assertEqual(c['evidence']['method'], 'joint_sample_auto_from_image')
        self.assertLess(abs(c['seam_x'] - 120), 2)

    def test_auto_joint_rejects_blank_page(self):
        from dual_track_fix.auto_joint import auto_joint_candidates
        gray = np.full((280, 240), 230, dtype=np.uint8)
        out = auto_joint_candidates(gray, [], [], 20.0, 260)
        self.assertEqual(out, [])

    def test_raw_key_uses_pre_review_lanes(self):
        from dual_track_stage8_online import _key_raw
        p = {'group_bbox': [10, 10, 50, 80], 'right_bbox': [30, 12, 50, 80],
             'left_bbox': [10, 10, 30, 78],
             'raw_bbox': {'right': [30, 10, 50, 79], 'left': [10, 11, 30, 80]}}
        self.assertEqual(_key_raw(p), ((10, 10, 50, 80), (30, 10, 50, 79),
                                       (10, 11, 30, 80)))

    def test_glyph_snap_respects_main_box_floor(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.dual_track_stage8_joint import snap_lane_y
        gray = np.full((300, 200), 230, dtype=np.uint8)
        for yy in range(80, 200, 8):
            gray[yy:yy + 4, 30:50] = 30
        out = snap_lane_y(gray, [30, 50, 50, 190], y_min=75, y_max=220)
        self.assertGreaterEqual(out[1], 75)   # 不得早于药名下沿
        self.assertLessEqual(out[3], 220)


if __name__ == '__main__':
    unittest.main()
