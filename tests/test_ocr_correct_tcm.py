# -*- coding: utf-8 -*-
"""中医古籍纠错词表的生效与防误伤测试。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

from app.config import CONFIG
from app.schemas import Block
from ocr.correct import correct_blocks, correct_text


class TestTCMCorrections(unittest.TestCase):
    def test_corrections_and_complete_trace(self):
        cases = [
            ("邪氯伏於募原", "邪氣伏於膜原", [("氯", "氣"), ("募原", "膜原")]),
            ("邪自傅裹出於表裹", "邪自傳裏出於表裏", [("傅裹", "傳裏"), ("表裹", "表裏")]),
            ("泉人皆感襟氣與裸氣", "衆人皆感雜氣與雜氣", [("泉人", "衆人"), ("襟氣", "雜氣"), ("裸氣", "雜氣")]),
            ("寒熟潮熟皆由益因管衛失和", "寒熱潮熱皆由蓋因營衛失和", [("寒熟", "寒熱"), ("潮熟", "潮熱"), ("管衛", "營衛"), ("益因", "蓋因")]),
            ("用白木煎服", "用白术煎服", [("白木", "白术")]),
        ]
        for source, expected, replacements in cases:
            with self.subTest(source=source):
                result, trace = correct_text(source)
                self.assertEqual(result, expected)
                self.assertEqual(trace, [
                    {"rule": "字词", "before": before, "after": after, "confidence": "中"}
                    for before, after in replacements
                ])

    def test_other_v3_rules(self):
        pairs = {
            "傳裹": "傳裏", "入裹": "入裏", "似裹": "似裏", "裹氣": "裏氣", "裹證": "裏證",
            "傅裏": "傳裏", "傅胃": "傳胃", "傅表": "傳表", "傅染": "傳染", "傅陽": "傳陽",
            "襟證": "雜證", "發熟": "發熱", "內熟": "內熱", "内熟": "内熱", "胃熟": "胃熱",
            "清熟": "清熱", "管衞": "營衞", "益當": "蓋當", "益非": "蓋非", "益先": "蓋先",
            "益不": "蓋不", "大頭痛": "大頭瘟",
        }
        for before, after in pairs.items():
            with self.subTest(before=before):
                self.assertEqual(correct_text(before), (after, [
                    {"rule": "字词", "before": before, "after": after, "confidence": "中"}
                ]))

    def test_unchanged_text_has_no_trace(self):
        for text in ("補益其氣", "溫疫論卷之二", "白水煎服", "味甘平", "熟地黃"):
            with self.subTest(text=text):
                self.assertEqual(correct_text(text), (text, []))

    def test_disabled_blocks_are_untouched(self):
        block = Block(page_id="TEST-001", bbox=[0, 0, 10, 10], line_id="L01",
                      quote="邪氯伏於募原", source_hash="test", text="邪氯伏於募原",
                      text_norm="邪氯伏於募原", kind="paragraph", conf=0.9,
                      col_id=0, block_ref="TEST-001:L01")
        blocks = [block]
        disabled_config = replace(CONFIG, ocr=replace(CONFIG.ocr, correct_enabled=False))
        with patch("ocr.correct.CONFIG", disabled_config):
            result, trace = correct_blocks(blocks)
        self.assertIs(result, blocks)
        self.assertIs(result[0], block)
        self.assertEqual(block.text, "邪氯伏於募原")
        self.assertEqual(block.quote, "邪氯伏於募原")
        self.assertEqual(trace, [])


if __name__ == "__main__":
    unittest.main()
