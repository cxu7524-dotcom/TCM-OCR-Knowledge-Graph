# -*- coding: utf-8 -*-
"""
tests/test_extract_determinism.py —— 抽取层可复现性（阶段五 P0 回归护栏）

背景：阶段四报出「合成集字段 F1 从 0.6471 掉到 0.5882」，排查发现与 OCR 无关
（`TCM_OCR_VERTICAL_SPLIT=0/1` 两份 ocr_predictions.jsonl 的 sha256 完全相同），
真正原因是 `extract/rules.py` 用 `max(hits, key=len)` 在**等长命中**时由 set 迭代顺序决定，
而 Python 字符串哈希逐进程随机（PYTHONHASHSEED），同一份 OCR 输出会抽出不同实体。
这些用例锁死「同一输入 → 同一输出」。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.schemas import Block              # noqa: E402
from extract import rules                  # noqa: E402


class TestPickMain(unittest.TestCase):
    def test_equal_length_picks_first_occurrence(self):
        """等长命中取「文本中最先出现」者——与 set 迭代顺序无关。"""
        text = "久服轻身延年温中下气"
        self.assertEqual(rules._pick_main(text, {"温中下气", "轻身延年"}), "轻身延年")
        self.assertEqual(rules._pick_main(text, ["温中下气", "轻身延年"]), "轻身延年")
        self.assertEqual(rules._pick_main(text, ("轻身延年", "温中下气")), "轻身延年")

    def test_longest_wins(self):
        self.assertEqual(rules._pick_main("味甘平無毒", {"味甘", "味甘平"}), "味甘平")

    def test_no_hit(self):
        self.assertEqual(rules._pick_main("無關文字", {"甘草"}), "")

    def test_earliest_position(self):
        self.assertEqual(rules._pick_main("甲乙丙丁", {"丙丁", "甲乙"}), "甲乙")


class TestExtractRepeatable(unittest.TestCase):
    def _blocks(self):
        # 抽取层拿到的 text_norm 已由 ocr/normalize 转成简体，这里直接用简体文本
        rows = [
            ("R001", "甘草味甘平主五藏六府寒热邪气坚筋骨长肌肉"),
            ("R002", "久服轻身延年温中下气"),
            ("R003", "无毒忌海藻菘菜"),
            ("R004", "桂枝汤桂枝三两芍药三两生姜三两"),
        ]
        return [Block(page_id="P-TEST", bbox=[0, i * 40, 300, i * 40 + 30],
                      line_id=line, quote=text, source_hash="h" + line,
                      text=text, text_norm=text, kind="paragraph", conf=0.9,
                      col_id=0, reading_order=i, block_ref=f"P-TEST:{line}")
                for i, (line, text) in enumerate(rows)]

    def test_same_input_same_output(self):
        a = [e.to_dict() for e in rules.extract_entities(self._blocks())]
        b = [e.to_dict() for e in rules.extract_entities(self._blocks())]
        self.assertEqual(a, b)

    def test_effect_is_single_and_stable(self):
        """同一行只出一个功效，且是「最长 + 最先出现」那个。"""
        ents = [e for e in rules.extract_entities(self._blocks()) if e.type == "功效"]
        names = [e.name for e in ents if e.line_id == "R002"]
        self.assertEqual(names, ["轻身延年"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
