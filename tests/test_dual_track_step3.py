# -*- coding: utf-8 -*-
"""第三步回归测试：共线断续外侧栏线排除、底版框前空白行回退、可配置补笔上限、掩膜像素核验与触字待核。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.dual_track import _stage8_review_lane  # noqa: E402
from tools.dual_track_stage8_ink_review import _mask_touches_glyph, _verify_crop  # noqa: E402


class Step3BoundaryAndMaskTests(unittest.TestCase):
    def test_collinear_broken_outer_rule_is_excluded_safely(self):
        ink = np.zeros((160, 80), np.uint8)
        for y in range(25, 135, 22):
            ink[y:y + 12, 22:34] = 1
        # 外侧 x=41..43 存在共线断续纵栏线（每段高 14px，未达 2.4H=48px，但纵向总覆盖 70px）
        for y in range(15, 145, 24):
            ink[y:y + 14, 41:43] = 1
        out, rev = _stage8_review_lane([20, 20, 44, 140], ink, {}, 20.0, side='right')
        self.assertLessEqual(out[2], 41)
        self.assertEqual(rev['actions'][0]['kind'], 'outer_rule_exclusion')

    def test_frame_top_pullback_stops_at_blank_row_before_frame(self):
        ink = np.zeros((160, 120), np.uint8)
        ink[30:44, 42:54] = 1
        ink[112:126, 42:54] = 1   # 末字结束于 y=126，y=128..129 为干净空白行
        ink[130:134, 10:110] = 1  # 横贯多栏底版框位于 y=130..134
        out, rev = _stage8_review_lane([40, 25, 58, 132], ink, {}, 20.0, side='right')
        self.assertLessEqual(out[3], 130)
        self.assertTrue(any(a['kind'] == 'frame_top_pullback' for a in rev['actions']))

    def test_configurable_completion_limit_bounds_stroke_extension(self):
        ink = np.zeros((120, 60), np.uint8)
        ink[30:42, 20:32] = 1
        ink[70:86, 20:32] = 1   # 越过 y2=80 共 6px
        # 默认上限 4px -> 不补并标 cut_tail
        out_default, rev_default = _stage8_review_lane([18, 25, 36, 80], ink, {}, 20.0)
        self.assertEqual(out_default[3], 80)
        self.assertIn('cut_tail', rev_default['unresolved'])
        # 显式放宽可配置上限至 6px -> 完整补至 86
        out_cfg, rev_cfg = _stage8_review_lane([18, 25, 36, 80], ink, {}, 20.0, max_complete_px=6)
        self.assertEqual(out_cfg[3], 86)
        self.assertEqual(rev_cfg['unresolved'], [])

    def test_mask_pixel_verification_and_failure_reasons(self):
        orig = np.full((60, 50, 3), 210, np.uint8)
        orig[45:48, 5:45] = 30  # 横栏线
        mask = np.zeros((60, 50), bool)
        mask[45:48, 5:45] = True
        cleared = orig.copy()
        cleared[mask] = 210

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            raw_p = td_p / 'raw.png'
            msk_p = td_p / 'mask.png'
            clr_p = td_p / 'clr.png'
            cv2.imwrite(str(raw_p), orig[10:55, 5:45])
            cv2.imwrite(str(msk_p), (mask[10:55, 5:45].astype(np.uint8) * 255))
            cv2.imwrite(str(clr_p), cleared[10:55, 5:45])

            ev = {
                'image_path': str(raw_p),
                'rule_mask_review': {
                    'crop_bbox': [5, 10, 45, 55],
                    'marked_pixels': int(mask[10:55, 5:45].sum()),
                    'mask_path': str(msk_p),
                    'masked_crop_path': str(clr_p),
                },
            }
            st, m_out = _verify_crop(orig, ev)
            self.assertEqual(st, 'verified')
            self.assertIsNotNone(m_out)

            # 篡改掩膜外 1 个像素 -> source_pixels_mismatch
            bad_clr = cleared[10:55, 5:45].copy()
            bad_clr[0, 0] = [0, 0, 0]
            cv2.imwrite(str(clr_p), bad_clr)
            st_bad, _ = _verify_crop(orig, ev)
            self.assertEqual(st_bad, 'source_pixels_mismatch')

            # 损坏文件 -> decode_failed
            clr_p.write_bytes(b'not_a_png')
            st_dec, _ = _verify_crop(orig, ev)
            self.assertEqual(st_dec, 'decode_failed')

            # 缺失文件 -> missing_attachment
            clr_p.unlink()
            st_mis, _ = _verify_crop(orig, ev)
            self.assertEqual(st_mis, 'missing_attachment')

    def test_mask_touching_real_glyph_is_flagged_not_auto_explained(self):
        ink = np.zeros((100, 60), bool)
        ink[65:80, 20:34] = True   # 真实末字连通域（宽 14px，底至 y=79 位于框内）
        ink[80:83, 5:55] = True    # 紧贴底版框（y=80..82）
        ruled = np.zeros((100, 60), bool)
        ruled[80:83, 5:55] = True
        self.assertTrue(_mask_touches_glyph(ink, ruled, [18, 20, 40, 80], 80, H=20.0))

    def test_outside_box_frame_residue_does_not_falsely_flag_glyph_touch(self):
        ink = np.zeros((100, 60), bool)
        ink[30:50, 20:34] = True   # 框内字形远离底边 y=80
        ink[81:83, 5:55] = True    # 框外横版框（已掩膜）
        ink[83:84, 18:35] = True   # 框外未掩膜版框下沿残段（位于 y>=83，不在框内）
        ruled = np.zeros((100, 60), bool)
        ruled[81:83, 5:55] = True
        self.assertFalse(_mask_touches_glyph(ink, ruled, [18, 20, 40, 80], 80, H=20.0))

    def test_title_trim_ignores_outer_vertical_rule_in_blank_gap(self):
        # 复现药名上部跨中缝构件 + 下部构件 + 外侧细纵栏线贯穿空白带时，
        # 右轨仍应将起点推入标题与首字之间的空白带，同时保留首字完整笔画。
        ink = np.zeros((180, 90), np.uint8)
        ink[21:36, 20:37] = 1     # 药名上部跨中缝构件（跨 seam_x=30，y=[21,36)）
        ink[29:48, 35:46] = 1     # 药名右半下部构件（y=[29,48)）
        # y=[48,53) 5 行空白，但外侧 x=[55,58) 有细纵栏线贯穿 [41,62)
        ink[41:62, 55:58] = 1
        ink[53:76, 35:48] = 1     # 首字「如」（y=[53,76)）
        ink[85:108, 35:48] = 1    # 后续小字
        out, rev = _stage8_review_lane(
            [30, 30, 60, 140], ink, {}, 22.0, seam_x=30.0, side='right', core_y=[30, 130],
        )
        self.assertGreaterEqual(out[1], 48)
        self.assertLessEqual(out[1], 52)
        self.assertTrue(any(a['kind'] == 'title_trim' for a in rev['actions']))

    def test_collinear_dashed_vertical_rule_mask_and_long_stroke_safety(self):
        from dual_track_fix.ruling import ruling_mask
        from ocr.dual_track import _stage8_ink_components

        ink = np.zeros((420, 120), np.uint8)
        # x=[90,92) 共线断续纵栏线：10 段高 12px、间距 20px，总跨度 300px
        for y in range(40, 360, 32):
            ink[y:y + 12, 90:92] = 1
        # x=[40,42) 单个字符的长竖笔（高 32px，非整页共线断续栏线）
        ink[120:152, 40:42] = 1
        labels, stats = _stage8_ink_components(ink)
        mask = ruling_mask(ink, 22.0, labels, stats)
        self.assertGreater(int(mask[40:360, 90:92].sum()), 200)
        self.assertEqual(int(mask[120:152, 40:42].sum()), 0)

    def test_fixed_sample_precondition_guard_and_v2_retract_basis(self):
        from tools.dual_track_step6_pipeline import (
            _extract_v2_retract_basis,
            validate_fixed_sample_preconditions,
        )

        bad_sha = validate_fixed_sample_preconditions(
            image_sha256='0' * 64,
            expected_image_sha256='1' * 64,
            actual_bbox=[660, 650, 690, 823],
            expected_y1_range=[648, 653],
        )
        self.assertFalse(bad_sha['ok'])
        self.assertEqual(bad_sha['status'], 'pending_precondition_mismatch')

        bad_box = validate_fixed_sample_preconditions(
            image_sha256='a' * 64,
            expected_image_sha256='a' * 64,
            actual_bbox=[660, 630, 690, 823],
            expected_y1_range=[648, 653],
        )
        self.assertFalse(bad_box['ok'])
        self.assertEqual(bad_box['status'], 'pending_precondition_mismatch')

        basis = _extract_v2_retract_basis({
            'basis': '旧冻结误标',
            'review': {'conclusion': '标签有误', 'retract_reason': '实测双轨连续成立'},
        })
        self.assertIn('实测双轨连续成立', basis)
        self.assertNotIn('旧冻结误标', basis)

    def test_disjoint_core_y_following_title_stops_upper_group_above_title(self):
        from ocr.dual_track import _stage8_following_title_stop, _stage8_ink_components

        ink = np.zeros((320, 80), np.uint8)
        # 上组双轨小字 y=[40,145)
        for y in range(42, 140, 22):
            ink[y:y + 14, 15:27] = 1
            ink[y:y + 14, 34:46] = 1
        # y=[145,150) 空白带，y=[150,180) 居中大字药名（跨 seam_x=30）
        ink[150:180, 20:42] = 1
        # 下组双轨小字 y=[188,295)
        for y in range(188, 290, 22):
            ink[y:y + 14, 15:27] = 1
            ink[y:y + 14, 34:46] = 1
        labels, stats = _stage8_ink_components(ink)
        upper = {
            'group_id': 'p#slot05#00', 'region_id': 'p#slot05',
            'group_bbox': [12, 40, 50, 300],
            'right_bbox': [30, 40, 50, 300], 'left_bbox': [12, 40, 30, 285],
            'evidence': {'core_y': [40, 148], 'valley_x': 30, 'windows': 30},
        }
        lower = {
            'group_id': 'p#slot05#01', 'region_id': 'p#slot05',
            'group_bbox': [12, 185, 50, 300],
            'right_bbox': [30, 185, 50, 300], 'left_bbox': [12, 185, 30, 285],
            'main_bbox': [12, 120, 50, 185],
            'evidence': {'core_y': [185, 295], 'valley_x': 30, 'windows': 25},
        }
        stop = _stage8_following_title_stop(upper, upper['right_bbox'], [lower], labels, stats, 20.0)
        self.assertIsNotNone(stop)
        self.assertEqual(stop['reason'], 'disjoint_core_y_title_break')
        self.assertGreaterEqual(stop['to'], 144)
        self.assertLessEqual(stop['to'], 150)

    def test_trailing_title_char_trimmed_only_with_upper_title_and_centered_char(self):
        ink = np.zeros((240, 80), np.uint8)
        # 上方大字药名首字 y=[40,68) 跨中缝 x=30
        ink[40:68, 20:42] = 1
        # 紧邻 y1=68 下方 3px 空白后的居中续字 y=[71,95) 跨中缝，x=[18,43) 宽 25<=1.5*20
        ink[71:93, 18:33] = 1
        ink[76:95, 34:43] = 1
        # y=[95,100) 5 行空白，下方为真实双轨小字 y=[100,200)
        for y in range(100, 195, 22):
            ink[y:y + 15, 13:26] = 1
            ink[y:y + 15, 34:48] = 1
        out, rev = _stage8_review_lane(
            [30, 68, 52, 200], ink, {}, 20.0,
            title=[12, 30, 52, 68], seam_x=30.0, side='right', group_x=[12, 52],
        )
        self.assertGreaterEqual(out[1], 95)
        self.assertLessEqual(out[1], 100)
        self.assertTrue(any(a['kind'] == 'trailing_title_char_trim' for a in rev['actions']))

        # 反例：无上方药名且首行是双轨并列宽字（总宽 35 > 1.5*20）时绝不裁剪
        ink_neg = np.zeros((240, 80), np.uint8)
        ink_neg[71:93, 13:26] = 1
        ink_neg[71:93, 29:48] = 1
        for y in range(100, 195, 22):
            ink_neg[y:y + 15, 13:26] = 1
            ink_neg[y:y + 15, 34:48] = 1
        out_neg, _ = _stage8_review_lane(
            [30, 68, 52, 200], ink_neg, {}, 20.0,
            title=[12, 30, 52, 68], seam_x=30.0, side='right', group_x=[12, 52],
        )
        self.assertEqual(out_neg[1], 68)


if __name__ == '__main__':
    unittest.main()
