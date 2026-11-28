# -*- coding: utf-8 -*-
"""第二步回归测试（5 项基础 + 3 项补充验收反例）：
1. 同槽嵌套重复去重
2. 同栏独立两组保留
3. 单侧只有栏线或污点不成立
4. 跨槽冲突但双侧有独立墨迹支持保留（函数级）
5. 真实药名尾笔靠近首字安全修剪（正例）
6. 确认区段内完整首字跨中缝保护（y=30→54 反例 + 归属不清保留待核）
7. 去重保留组未通过双侧校验时不得吞掉合格候选，且审计保留 ID 必在输出中（slot09#01→slot09#00 反例）
8. 联合组外接框跨中间空隙时完整决策保留独立候选，真实重复单槽候选正常去重
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_track_fix.detect import _superseded_windows, _uncovered_side_ink  # noqa: E402
from ocr.dual_track import _stage8_pair_support, _stage8_record, run_dual_track  # noqa: E402


class Step2GroupingAndBoundaryTests(unittest.TestCase):
    def test_nested_same_slot_duplicate_is_superseded_without_title_break(self):
        gray = np.full((220, 70), 240, np.uint8)
        for y in range(20, 200, 18):
            gray[y:y + 12, 12:28] = 40
            gray[y:y + 12, 36:52] = 40
        H = 20.0
        outer = {
            'group_id': 'p#slot00#01', 'page_id': 'p', 'region_id': 'p#slot00',
            'region_source': 'slot_split',
            'group_bbox': [10, 18, 54, 200],
            'left_bbox': [10, 18, 32, 200],
            'right_bbox': [32, 18, 54, 200],
            'main_bbox': [10, 2, 54, 18],
            'evidence': {'valley_x': 32, 'core_y': [20, 160]},
        }
        nested = {
            'group_id': 'p#slot00#02', 'page_id': 'p', 'region_id': 'p#slot00',
            'region_source': 'slot_split',
            'group_bbox': [10, 90, 54, 200],
            'left_bbox': [10, 90, 32, 200],
            'right_bbox': [32, 90, 54, 200],
            'main_bbox': [10, 60, 54, 88],
            'evidence': {'valley_x': 32, 'core_y': [92, 180]},
        }
        kept, audit = _superseded_windows([outer, nested], H, gray=gray)
        self.assertEqual([g['group_id'] for g in kept], ['p#slot00#01'])
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]['reason'], 'same_slot_nested_pair')

    def test_stacked_groups_with_title_break_are_both_kept(self):
        gray = np.full((240, 70), 240, np.uint8)
        for y in range(20, 95, 18):
            gray[y:y + 12, 12:28] = 40
            gray[y:y + 12, 36:52] = 40
        gray[102:122, 22:44] = 30
        for y in range(126, 220, 18):
            gray[y:y + 12, 12:28] = 40
            gray[y:y + 12, 36:52] = 40
        H = 20.0
        upper = {
            'group_id': 'p#slot03#01', 'page_id': 'p', 'region_id': 'p#slot03',
            'region_source': 'slot_split',
            'group_bbox': [10, 18, 54, 220],
            'left_bbox': [10, 18, 32, 220],
            'right_bbox': [32, 18, 54, 220],
            'main_bbox': [10, 2, 54, 18],
            'evidence': {'valley_x': 32, 'core_y': [20, 90]},
        }
        lower = {
            'group_id': 'p#slot03#05', 'page_id': 'p', 'region_id': 'p#slot03',
            'region_source': 'slot_split',
            'group_bbox': [10, 125, 54, 220],
            'left_bbox': [10, 125, 32, 220],
            'right_bbox': [32, 125, 54, 220],
            'main_bbox': [10, 100, 54, 124],
            'evidence': {'valley_x': 32, 'core_y': [126, 210]},
        }
        kept, audit = _superseded_windows([upper, lower], H, gray=gray)
        self.assertEqual([g['group_id'] for g in kept], ['p#slot03#01', 'p#slot03#05'])
        self.assertEqual(audit, [])

    def test_one_sided_rule_or_smudge_lacks_pair_support(self):
        ink = np.zeros((120, 70), np.uint8)
        ink[20:32, 12:24] = 1
        ink[42:54, 12:24] = 1
        ink[10:110, 52:54] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        group = {'group_bbox': [10, 18, 55, 90],
                 'left_bbox': [10, 18, 30, 90], 'right_bbox': [30, 18, 55, 90]}
        sup = _stage8_pair_support(group, labels, stats, 20.0)
        self.assertLess(sup['right'], 10)
        self.assertGreater(sup['left'], 50)

    def test_cross_slot_joint_keeps_independent_dual_track_note(self):
        ink = np.zeros((160, 120), np.uint8)
        ink[20:100, 12:24] = 1
        ink[20:100, 32:44] = 1
        s = {'left_bbox': [10, 18, 28, 130], 'right_bbox': [30, 18, 48, 130]}
        jgb = [42, 80, 90, 150]
        uncov_l = _uncovered_side_ink(ink, s['left_bbox'], [jgb])
        uncov_r = _uncovered_side_ink(ink, s['right_bbox'], [jgb])
        self.assertGreaterEqual(min(uncov_l, uncov_r), 10)

    def test_title_tail_near_first_glyph_trims_only_title(self):
        ink = np.zeros((140, 80), np.uint8)
        ink[15:39, 32:46] = 1   # 真实药名尾笔：从 y=15（轨顶上方 15px）跨入 y1=30 至 y=39
        ink[42:56, 42:55] = 1
        ink[42:56, 20:33] = 1
        ink[62:76, 42:55] = 1
        ink[62:76, 20:33] = 1
        group = {
            'group_id': 'p#slot08#00', 'page_id': 'p', 'region_id': 'p#slot08',
            'group_bbox': [18, 30, 58, 120],
            'right_bbox': [38, 30, 58, 120],
            'left_bbox': [18, 30, 38, 120],
            'main_bbox': [18, 5, 58, 38],
            'evidence': {'valley_x': 38, 'core_y': [40, 110]},
            'scale_used': {'H': 20.0},
        }
        rec = _stage8_record(group, [18, 0, 58, 140], ink=ink)
        self.assertEqual(rec['after']['right_bbox'][1], 39)
        self.assertEqual(rec['after']['left_bbox'][1], 39)
        self.assertTrue(rec['sides']['right']['final_validation']['ok'])
        self.assertTrue(rec['sides']['left']['final_validation']['ok'])

    def test_confirmed_first_glyph_crossing_seam_is_not_trimmed_30_to_54(self):
        # 复现 y=30→54 反例：确认区段 core_y=[30,110] 内首字 y=30..54 跨中缝，上方药名在 y=5..22 已结束
        ink = np.zeros((150, 80), np.uint8)
        ink[5:22, 24:52] = 1    # 上方独立药名，已在 y=22 结束
        ink[30:54, 32:46] = 1   # 确认区段内 y=30 起笔的首字（跨过中缝 x=38），绝不能被裁到 54！
        ink[60:74, 42:55] = 1
        ink[60:74, 20:33] = 1
        group = {
            'group_id': 'p#first_glyph', 'page_id': 'p', 'region_id': 'p#slot01',
            'group_bbox': [18, 30, 58, 125],
            'right_bbox': [38, 30, 58, 125],
            'left_bbox': [18, 30, 38, 125],
            'main_bbox': [18, 5, 58, 22],
            'evidence': {'valley_x': 38, 'core_y': [30, 110]},
            'scale_used': {'H': 20.0},
        }
        rec = _stage8_record(group, [18, 0, 58, 150], ink=ink)
        self.assertEqual(rec['after']['right_bbox'][1], 30)
        self.assertEqual(rec['after']['left_bbox'][1], 30)
        self.assertEqual(rec['sides']['right']['boundary_review']['actions'], [])

        # 归属不清反例：无药名支持而笔画从 y=28 微跨 y1=30（core_y=[28,110]）→ 保原框 30 并标待核
        ink2 = np.zeros((150, 80), np.uint8)
        ink2[28:52, 32:46] = 1
        ink2[60:74, 42:55] = 1
        ink2[60:74, 20:33] = 1
        group2 = dict(group, main_bbox=[18, 2, 58, 14],
                      evidence={'valley_x': 38, 'core_y': [28, 110]})
        rec2 = _stage8_record(group2, [18, 0, 58, 150], ink=ink2)
        self.assertEqual(rec2['after']['right_bbox'][1], 30)
        self.assertFalse(rec2['sides']['right']['final_validation']['ok'])

    def test_invalid_keeper_does_not_swallow_valid_nested_candidate(self):
        # 复现 slot09#01 -> slot09#00：外层长框 slot09#00 右侧无字形（未过双侧校验），
        # 内层候选 slot09#01 双侧有真实字形，必须保留 slot09#01 且审计中无悬空保留 ID。
        img = np.full((260, 120, 3), 240, np.uint8)
        # 在 y=110..210 构造真实双轨小字（左 x=24..36，右 x=44..56）
        for y in range(112, 205, 18):
            img[y:y + 12, 24:36] = 30
            img[y:y + 12, 44:56] = 30
        # 在 y=30..100 仅左侧 x=24..36 有单列字形（slot09#00 若将中缝定在 x=60 则右侧为空）
        for y in range(32, 95, 18):
            img[y:y + 12, 24:36] = 30

        fake_d8 = {
            'regions': [{'region_id': 't#slot09', 'bbox': [20, 0, 80, 250], 'source': 'slot_split'}],
            'slot_audit': [],
            'all_proposals': [
                {
                    # 伪长框：右轨 x=60..78 全空白，双侧校验必败
                    'group_id': 't#slot09#00', 'page_id': 't', 'region_id': 't#slot09',
                    'region_source': 'slot_split',
                    'group_bbox': [20, 28, 78, 220],
                    'left_bbox': [20, 28, 60, 220],
                    'right_bbox': [60, 28, 78, 220],
                    'main_bbox': [20, 5, 78, 25],
                    'evidence': {'valley_x': 40, 'core_y': [30, 210]},
                },
                {
                    # 真实双轨短候选：左右均有小字
                    'group_id': 't#slot09#01', 'page_id': 't', 'region_id': 't#slot09',
                    'region_source': 'slot_split',
                    'group_bbox': [20, 110, 60, 215],
                    'left_bbox': [20, 110, 40, 215],
                    'right_bbox': [40, 110, 60, 215],
                    'main_bbox': [20, 85, 60, 108],
                    'evidence': {'valley_x': 40, 'core_y': [112, 205]},
                },
            ],
            'groups': [],
            'dropped_duplicates': [],
        }
        with tempfile.TemporaryDirectory() as td:
            img_p = Path(td) / 'syn.png'
            cv2.imwrite(str(img_p), img)
            with patch('dual_track_fix.detect.detect', return_value=fake_d8):
                res = run_dual_track(
                    str(img_p), 't', recognizer=lambda im, b, z: ('测试', 0.99, {}),
                    cfg={'detector': 'stage8'},
                )
        out_ids = {e['group_id'] for e in (res['evidence'] or [])}
        self.assertIn('t#slot09#01', out_ids)
        self.assertNotIn('t#slot09#00', out_ids)
        for a in res['audit']:
            if a.get('kind') in ('window_superseded', 'cross_slot_joint_superseded'):
                self.assertIn(a['superseded_by'], out_ids)

    def test_cross_slot_joint_outer_box_gap_preserves_independent_candidate(self):
        # 完整决策回归：联合组左右轨之间存在空隙（外接框跨越空隙），空隙内的独立双轨组必须保留；
        # 而被联合组实际轨框覆盖一侧的重复单槽候选必须被去重。
        img = np.full((240, 160, 3), 240, np.uint8)
        for y in range(45, 195, 18):
            img[y:y + 12, 12:24] = 30    # 联合组左轨
            img[y:y + 12, 112:124] = 30  # 联合组右轨
            img[y:y + 12, 44:56] = 30    # 空隙内独立组左轨
            img[y:y + 12, 64:76] = 30    # 空隙内独立组右轨
            img[y:y + 12, 134:146] = 30  # 邻栏单列墨迹

        fake_d8 = {
            'regions': [
                {'region_id': 't#slot01', 'bbox': [10, 0, 150, 230], 'source': 'slot_split'},
                {'region_id': 't#slot02', 'bbox': [40, 0, 80, 230], 'source': 'slot_split'},
                {'region_id': 't#slot00', 'bbox': [106, 0, 142, 230], 'source': 'slot_split'},
            ],
            'slot_audit': [],
            'all_proposals': [
                {
                    'group_id': 't#slot01#joint00', 'page_id': 't', 'region_id': 't#slot01',
                    'region_source': 'joint_auto',
                    'group_bbox': [10, 40, 126, 200],
                    'left_bbox': [10, 40, 28, 200],
                    'right_bbox': [108, 40, 126, 200],
                    'main_bbox': [10, 10, 126, 38],
                    'evidence': {'valley_x': 68},
                },
                {
                    # 落在联合组外接框 [10,126] 中间空隙 [40,80] 内的真实独立双轨组 → 必须保留
                    'group_id': 't#slot02#indep', 'page_id': 't', 'region_id': 't#slot02',
                    'region_source': 'slot_split',
                    'group_bbox': [40, 42, 80, 198],
                    'left_bbox': [40, 42, 60, 198],
                    'right_bbox': [60, 42, 80, 198],
                    'main_bbox': [40, 12, 80, 38],
                    'evidence': {'valley_x': 60, 'core_y': [45, 195]},
                },
                {
                    # 左轨完全压在联合组右轨 [108,126] 上、右轨蹭到邻栏单列的错位单槽候选 → 必须去重
                    'group_id': 't#slot00#dup', 'page_id': 't', 'region_id': 't#slot00',
                    'region_source': 'slot_split',
                    'group_bbox': [108, 42, 142, 198],
                    'left_bbox': [108, 42, 126, 198],
                    'right_bbox': [126, 42, 142, 198],
                    'main_bbox': [108, 12, 142, 38],
                    'evidence': {'valley_x': 126, 'core_y': [45, 195]},
                },
            ],
            'groups': [],
            'dropped_duplicates': [],
        }
        with tempfile.TemporaryDirectory() as td:
            img_p = Path(td) / 'syn_joint.png'
            cv2.imwrite(str(img_p), img)
            with patch('dual_track_fix.detect.detect', return_value=fake_d8):
                res = run_dual_track(
                    str(img_p), 't', recognizer=lambda im, b, z: ('测试', 0.99, {}),
                    cfg={'detector': 'stage8'},
                )
        out_ids = {e['group_id'] for e in (res['evidence'] or [])}
        self.assertIn('t#slot01#joint00', out_ids)
        self.assertIn('t#slot02#indep', out_ids)
        self.assertNotIn('t#slot00#dup', out_ids)


if __name__ == '__main__':
    unittest.main()
