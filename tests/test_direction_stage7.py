# -*- coding: utf-8 -*-
"""
tests/test_direction_stage7.py —— 阶段七 P0/P1 单测

覆盖：
  1. `TCM_OCR_DIR_RULE=old/new` 可回退，旧判据行为与阶段六一致；
  2. det 碎片化（宽扁框占多数）的竖排页能被新判据救回；
  3. 真横排页不误判；
  4. 稳定列聚类不会被「中心漂移」把相邻列串成一列；
  5. 返回字段保留阶段六全部键，追加新信号明细。
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                      # noqa: E402
from app.schemas import Block, source_hash_of      # noqa: E402
from ocr import layout                             # noqa: E402


def blk(text, bbox, line_id, conf=0.9):
    return Block(page_id="T", bbox=list(bbox), line_id=line_id, quote=text,
                 source_hash=source_hash_of("t", line_id, text), text=text,
                 text_norm=text, kind="paragraph", conf=conf, col_id=0,
                 block_ref=f"T:{line_id}")


class _ConfigSwitch:
    """临时把 layout.CONFIG 换成指定 rule 的副本（frozen dataclass 不可直接赋值）。"""

    def __init__(self, rule: str):
        self.rule = rule

    def __enter__(self):
        self.origin = layout.CONFIG
        ocr = dataclasses.replace(self.origin.ocr, dir_rule=self.rule)
        layout.CONFIG = SimpleNamespace(ocr=ocr)
        return self

    def __exit__(self, *args):
        layout.CONFIG = self.origin


def fragmented_vertical():
    """模拟 det 把竖列切成水平碎片：80 个宽扁框，10 列等距，每列 8 段。"""
    out = []
    for c in range(10):
        x = 900 - c * 90
        for r in range(8):
            y = 80 + r * 95
            out.append(blk(f"字{c}{r}", [x, y, x + 60, y + 40], f"B{c:02d}{r}"))
    return out


class TestDirectionRuleSwitch(unittest.TestCase):
    def test_old_and_new_on_fragmented_page(self):
        bs = fragmented_vertical()
        with _ConfigSwitch("old"):
            d_old = layout.detect_direction_detail(bs)
        with _ConfigSwitch("new"):
            d_new = layout.detect_direction_detail(bs)
        # 碎片页竖长条框占比为 0，旧判据判横排；新判据用「密集规则列」救回
        self.assertEqual(d_old["direction"], "horizontal-ltr")
        self.assertEqual(d_new["direction"], "vertical-rtl")
        self.assertTrue(d_new["dense_regular"])
        self.assertEqual(d_new["rule"], "new")

    def test_horizontal_rows_unchanged(self):
        bs = [blk("桂枝三兩", [100, 100 + i * 60, 300, 130 + i * 60], f"L{i:02d}")
              for i in range(3)]
        for rule in ("old", "new"):
            with _ConfigSwitch(rule):
                self.assertEqual(layout.detect_direction(bs), "horizontal-ltr")

    def test_tall_columns_still_vertical(self):
        bs = [blk("甘草味甘平無毒", [760 - i * 130, 190, 800 - i * 130, 560], f"L{i:02d}")
              for i in range(5)]
        for rule in ("old", "new"):
            with _ConfigSwitch(rule):
                self.assertEqual(layout.detect_direction(bs), "vertical-rtl")

    def test_return_fields_keep_stage6_keys(self):
        bs = fragmented_vertical()
        with _ConfigSwitch("new"):
            d = layout.detect_direction_detail(bs)
        for key in ("direction", "vertical_ratio", "vertical_boxes", "total_boxes",
                    "columns", "headers", "thresholds", "reason"):
            self.assertIn(key, d)
        for key in ("rule", "aspect_median", "col_gap_cv", "text_len_cv",
                    "img_columns", "img_col_aspect_median", "img_vertical", "dense_regular"):
            self.assertIn(key, d)

    def test_empty_blocks(self):
        with _ConfigSwitch("new"):
            self.assertEqual(layout.detect_direction([]), "horizontal-ltr")


class TestStableColumns(unittest.TestCase):
    def test_no_chaining(self):
        # 11 个框，x 中心每隔 45px 一个；阈值按中位框宽算，固定锚点不应把全部串成一列
        bs = [blk("字", [900 - i * 45, 100, 940 - i * 45, 500], f"C{i:02d}")
              for i in range(11)]
        stable = layout._stable_columns(bs)
        self.assertGreaterEqual(len(stable), 5, len(stable))
        self.assertEqual([c["col_id"] for c in stable], list(range(len(stable))))
        self.assertEqual([c["center"] for c in stable],
                         sorted([c["center"] for c in stable], reverse=True))

    def test_same_shape_as_project_columns(self):
        bs = fragmented_vertical()
        for c in layout._stable_columns(bs):
            for key in ("col_id", "center", "members", "x1", "x2", "aspect_median"):
                self.assertIn(key, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
