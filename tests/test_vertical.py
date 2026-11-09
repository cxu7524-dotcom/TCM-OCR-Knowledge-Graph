# -*- coding: utf-8 -*-
"""
tests/test_vertical.py —— 竖排逐列检测的单元测试（标准库 unittest）

覆盖验收要求的四项：
  1. 旋转后 bbox 反变换回原图坐标；
  2. 列排序（从右向左）；
  3. 空列 / 窄列 / 矮列过滤；
  4. 超长列切块。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ocr import vertical  # noqa: E402

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False


def make_page(cols, h=300, w=400):
    """白底 + 一列列小横条模拟竖排文字（实心大块会让自适应二值化失效）。

    cols = [(x, y0, y1, width), ...]
    """
    img = np.full((h, w, 3), 255, np.uint8)
    for x, y0, y1, cw in cols:
        for y in range(y0, max(y0 + 1, y1 - 15), 30):
            cv2.rectangle(img, (x, y), (x + cw, y + 16), (0, 0, 0), -1)
    return img


class TestRotateBBoxBack(unittest.TestCase):
    """逆时针 90°：原图 (x,y) → 旋转图 (y, W-1-x)；反变换 (x',y') → (W-1-y', x')。"""

    def test_point_mapping(self):
        # 原图 W=100, H=200
        back = vertical.rotate_bbox_back([20, 89, 20, 89], (200, 100), rotate=90)
        self.assertEqual(back, [10, 20, 10, 20])

    def test_bbox_mapping(self):
        # 旋转图中的一个矩形，反变换后应落在原图内且宽高互换
        rot_bbox = [10, 20, 40, 60]          # x1,y1,x2,y2
        back = vertical.rotate_bbox_back(rot_bbox, (200, 100), rotate=90)
        self.assertEqual(back, [39, 10, 79, 40])
        self.assertLessEqual(back[2], 100)
        self.assertLessEqual(back[3], 200)

    def test_clockwise(self):
        # 顺时针 90°：原图 (x,y) → (H-1-y, x)，反变换 (x',y') → (y', H-1-x')
        back = vertical.rotate_bbox_back([5, 5, 5, 5], (200, 100), rotate=270)
        self.assertEqual(back, [5, 194, 5, 194])


@unittest.skipUnless(_CV2_OK, "需要 opencv")
class TestSplitColumns(unittest.TestCase):
    def test_right_to_left_order(self):
        img = make_page([(40, 40, 260, 40), (150, 40, 260, 40), (260, 40, 260, 40)])
        cols = vertical.split_columns(img)
        self.assertGreaterEqual(len(cols), 3)
        centers = [(c["bbox"][0] + c["bbox"][2]) / 2 for c in cols]
        self.assertEqual(centers, sorted(centers, reverse=True))   # 从右向左
        self.assertEqual([c["col_index"] for c in cols], list(range(len(cols))))

    def test_narrow_column_filtered(self):
        # 5px 宽的列 < col_min_w(12) 应被过滤
        img = make_page([(40, 40, 260, 40), (150, 40, 260, 5), (260, 40, 260, 40)])
        cols = vertical.split_columns(img)
        widths = [c["bbox"][2] - c["bbox"][0] for c in cols]
        self.assertTrue(all(w >= 12 for w in widths), widths)

    def test_short_column_filtered(self):
        # 高度 30px < 页面 12% 的列应被过滤
        img = make_page([(40, 40, 260, 40), (150, 10, 40, 40), (260, 40, 260, 40)])
        cols = vertical.split_columns(img)
        heights = [c["bbox"][3] - c["bbox"][1] for c in cols]
        self.assertTrue(all(h >= 300 * vertical.MIN_COL_RATIO for h in heights), heights)

    def test_empty_image(self):
        self.assertEqual(vertical.split_columns(np.zeros((0, 0, 3), np.uint8)), [])
        self.assertFalse(vertical.is_vertical_page(np.zeros((0, 0, 3), np.uint8)))


class TestCutPoints(unittest.TestCase):
    def test_short_column_single_piece(self):
        proj = np.ones(200)
        self.assertEqual(vertical._cut_points(proj, 220), [0, 200])

    def test_long_column_split(self):
        proj = np.ones(1000)
        pts = vertical._cut_points(proj, 220)
        self.assertGreater(len(pts), 3)
        self.assertEqual(pts[0], 0)
        self.assertEqual(pts[-1], 1000)
        # 每块不超过 max_h 的 1.5 倍（切点受字间隙约束会偏移），且不出现过短块
        for a, b in zip(pts[:-1], pts[1:]):
            self.assertLessEqual(b - a, int(220 * 1.5))
            self.assertGreaterEqual(b - a, vertical.MIN_PIECE_H)

    def test_cut_prefers_low_projection(self):
        # 在 y=210 附近人为造一个低谷，切点应落在低谷附近
        proj = np.ones(600) * 10
        proj[205:215] = 0.1
        pts = vertical._cut_points(proj, 220)
        self.assertTrue(any(abs(p - 210) <= 8 for p in pts[1:-1]), pts)


@unittest.skipUnless(_CV2_OK, "需要 opencv")
class TestSplitSpread(unittest.TestCase):
    """对开双页：书脊检测 + 右半页先读 + 单页不硬切。"""

    def _spread(self, gutter=10, h=300, w=400):
        """两半各 3 列，中间一条贯穿全页的暗带（装订阴影型中缝）。"""
        img = make_page([(30, 40, 260, 25), (70, 40, 260, 25), (110, 40, 260, 25),
                         (250, 40, 260, 25), (290, 40, 260, 25), (330, 40, 260, 25)],
                        h=h, w=w)
        gx = w // 2
        cv2.rectangle(img, (gx - gutter // 2, 0), (gx + gutter // 2, h), (20, 20, 20), -1)
        return img

    def test_detect_spread_right_half_first(self):
        halves = vertical.split_spread(self._spread())
        self.assertEqual(len(halves), 2, halves)
        self.assertGreater(halves[0][0], halves[1][1])      # 右半页的 x0 > 左半页的 x1
        self.assertEqual(halves[1][0], 0)
        self.assertEqual(halves[0][1], 400)

    def test_portrait_single_page_not_split(self):
        # 高/宽 = 300/180 > 1.2 → 竖长单页，必须整页返回，不许硬切
        img = make_page([(20, 40, 260, 22), (50, 40, 260, 22), (80, 40, 260, 22),
                         (110, 40, 260, 22), (140, 40, 260, 22)], h=300, w=180)
        self.assertEqual(vertical.split_spread(img), [(0, 180)])

    def test_no_gutter_returns_whole_page(self):
        # 横宽但没有书脊信号 → 仍整页（宁可不切）
        img = make_page([(20, 40, 260, 25), (50, 40, 260, 25), (80, 40, 260, 25),
                         (110, 40, 260, 25), (140, 40, 260, 25), (170, 40, 260, 25)],
                        h=300, w=400)
        self.assertEqual(vertical.split_spread(img), [(0, 400)])

    def test_spine_line_allows_illustration_half(self):
        """阶段七 P2：中缝竖线证据强时，右半页只有 1 列（插图/目录）也要切。"""
        img = make_page([(30, 40, 260, 25), (70, 40, 260, 25), (110, 40, 260, 25),
                         (300, 40, 260, 25)], h=300, w=400)
        cv2.rectangle(img, (198, 0), (201, 299), (20, 20, 20), -1)  # 贯穿全页的中缝竖线
        halves = vertical.split_spread(img)
        self.assertEqual(len(halves), 2, halves)
        self.assertGreater(halves[0][0], halves[1][1])

    def test_empty_image(self):
        self.assertEqual(vertical.split_spread(np.zeros((0, 0, 3), np.uint8)), [])


@unittest.skipUnless(_CV2_OK, "需要 opencv")
class TestWideColumnSplit(unittest.TestCase):
    """跨列框抑制：宽度 > 1.8× 中位列宽的段要按内部次峰再切一刀。"""

    def test_wide_column_split_into_two(self):
        # 直接构造「两列被并进同一个框」的情形：66px 宽，中间 6px 暗谷，中位列宽 30
        h, w = 300, 300
        img = np.full((h, w, 3), 255, np.uint8)
        clean = np.zeros((h, w), np.uint8)
        for x0 in (100, 136):
            for y in range(40, 260, 30):
                cv2.rectangle(img, (x0, y), (x0 + 29, y + 16), (0, 0, 0), -1)
            clean[40:260, x0:x0 + 30] = 255
        col = {"bbox": [100, 40, 166, 260], "crop": img[40:260, 100:166].copy()}
        pieces = vertical._split_wide_column(img, clean, col, med_w=30)
        self.assertEqual(len(pieces), 2, pieces)
        self.assertTrue(all(p.get("wide_split") for p in pieces))
        # 切点应落在 6px 暗谷内（130~136），不必精确到像素
        self.assertTrue(130 <= pieces[1]["bbox"][0] <= 136, pieces)

    def test_normal_column_not_split(self):
        h, w = 300, 300
        img = np.full((h, w, 3), 255, np.uint8)
        clean = np.zeros((h, w), np.uint8)
        clean[40:260, 100:130] = 255
        col = {"bbox": [100, 40, 130, 260], "crop": img[40:260, 100:130].copy()}
        self.assertEqual(len(vertical._split_wide_column(img, clean, col, med_w=30)), 1)

    def test_cross_column_rate_counts_spanning_boxes(self):
        class B:                       # 极简 Block 替身，只需要 bbox / line_id / text
            def __init__(self, bbox, text="x"):
                self.bbox, self.line_id, self.text = bbox, "L", text

        cols = [{"bbox": [0, 0, 50, 300]}, {"bbox": [60, 0, 110, 300]}]
        blocks = [B([0, 0, 50, 300]), B([10, 0, 100, 300])]     # 第二个横跨两列
        r = vertical.cross_column_rate(blocks, cols)
        self.assertEqual(r["total"], 2)
        self.assertEqual(r["cross"], 1)
        self.assertEqual(r["rate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
