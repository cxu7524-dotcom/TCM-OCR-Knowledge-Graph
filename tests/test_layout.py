# -*- coding: utf-8 -*-
"""
tests/test_layout.py —— 版面 / 归一化 / 指标口径的单元测试（标准库 unittest）

跑法（项目根目录）：
    python -m unittest discover -s tests -v
无需 pytest、无需 GPU、无需联网。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.schemas import Block, source_hash_of          # noqa: E402
from eval.metrics import (                            # noqa: E402
    compute_cer, compute_field_f1, compute_refusal, levenshtein,
)
from ocr import layout, normalize, table               # noqa: E402


def block(text: str, bbox, line_id: str, col_id: int = 0, conf: float = 0.9) -> Block:
    return Block(
        page_id="TEST-001", bbox=list(bbox), line_id=line_id, quote=text,
        source_hash=source_hash_of("test", line_id, text), text=text,
        text_norm=text, kind="paragraph", conf=conf, col_id=col_id,
        block_ref=f"TEST-001:{line_id}",
    )


class TestDirection(unittest.TestCase):
    def test_vertical_rtl(self):
        # 5 列竖长条 + 1 横排页眉 → 竖排从右向左
        bs = [block("本草綱目 卷之一", [70, 60, 830, 96], "L00")]
        for i in range(5):
            x = 760 - i * 130
            bs.append(block("甘草味甘平無毒", [x, 190, x + 40, 560], f"L{i+1:02d}", i))
        d = layout.detect_direction_detail(bs)
        self.assertEqual(d["direction"], "vertical-rtl")
        self.assertGreaterEqual(d["vertical_ratio"], 0.30)
        self.assertGreaterEqual(d["columns"], 3)
        self.assertIn("竖长条框", d["reason"])

    def test_horizontal_ltr(self):
        # 3 行横排文字 → 横排
        bs = [block("桂枝三兩", [100, 100 + i * 60, 300, 130 + i * 60], f"L{i:02d}")
              for i in range(3)]
        self.assertEqual(layout.detect_direction(bs), "horizontal-ltr")

    def test_empty(self):
        self.assertEqual(layout.detect_direction([]), "horizontal-ltr")


class TestColumns(unittest.TestCase):
    def test_right_to_left_order(self):
        bs = [block("甲", [100, 200, 140, 500], "L01"),
              block("乙", [360, 200, 400, 500], "L02"),
              block("丙", [620, 200, 660, 500], "L03")]
        cols = layout.project_columns(bs)
        self.assertEqual(len(cols), 3)
        self.assertGreater(cols[0]["center"], cols[1]["center"])
        self.assertGreater(cols[1]["center"], cols[2]["center"])
        self.assertEqual([c["col_id"] for c in cols], [0, 1, 2])

    def test_merge_fragments(self):
        # 同列被切碎的两段应合并成完整原文
        a = block("主治五", [630, 193, 667, 327], "R01", 0, 0.72)
        b = block("藏六府寒熱邪氣", [630, 307, 667, 610], "R02", 0, 0.77)
        merged = layout.merge_fragments([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "主治五藏六府寒熱邪氣")
        self.assertEqual(merged[0].bbox, [630, 193, 667, 610])
        self.assertAlmostEqual(merged[0].conf, 0.72)

    def test_header_is_first(self):
        header = block("本草綱目 卷之一", [70, 60, 830, 96], "L00")
        col = block("甘草味甘平無毒", [760, 190, 800, 560], "L01")
        items, lay = layout.analyze_layout([header, col], "TEST-001", merge=False)
        self.assertEqual([b.text for b in items][0], "本草綱目 卷之一")
        self.assertEqual(items[0].col_id, -1)


class TestNormalize(unittest.TestCase):
    def test_multi_char_first(self):
        self.assertEqual(normalize.normalize_text("五藏六府寒熱邪氣"), "五脏六腑寒热邪气")

    def test_original_unchanged(self):
        b = block("甘草味甘平無毒", [0, 0, 10, 10], "L01")
        normalize.normalize_blocks([b])
        self.assertEqual(b.text, "甘草味甘平無毒")   # 原文不动
        self.assertEqual(b.text_norm, "甘草味甘平无毒")

    def test_lexicon_loaded(self):
        raw = normalize.load_lexicon()
        self.assertFalse(raw.get("降级", False), "字表文件应已就绪")
        self.assertGreater(normalize.mapping_size(), 10)


class TestTable(unittest.TestCase):
    def test_row_column_clustering(self):
        bs = []
        for r in range(3):
            for c in range(3):
                x, y = 100 + c * 240, 210 + r * 92
                bs.append(block(f"r{r}c{c}", [x, y, x + 60, y + 35],
                                f"L{r*3+c}", c))
        rows = table.cluster_rows(bs)
        self.assertEqual(len(rows), 3)
        self.assertEqual([len(r) for r in rows], [3, 3, 3])
        cols = table.cluster_columns(bs)
        self.assertEqual(len(cols), 3)
        self.assertLess(cols[0], cols[-1])  # 横排表格列序为左→右（x 升序）

    def test_vertical_not_table(self):
        bs = [block("甘草味甘平無毒", [760 - i * 130, 190, 800 - i * 130, 560],
                    f"L{i:02d}") for i in range(5)]
        self.assertEqual(table.restore_tables(bs, "TEST-001"), [])


class TestMetrics(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein("abc", "ab"), 1)
        self.assertEqual(levenshtein("abc", "axc"), 1)

    def test_cer_excludes_demo(self):
        gt = [{"page_id": "P1", "bbox": [0, 0, 1, 1], "line_id": "L1",
               "quote": "甘草", "source_hash": "x"}]
        pred = [{"page_id": "P1", "bbox": [0, 0, 1, 1], "line_id": "R1",
                 "quote": "甘草", "source_hash": "x", "is_demo": True}]
        r = compute_cer(gt, pred)
        self.assertEqual(r["status"], "待测")
        self.assertEqual(r["excluded"]["pred"], 1)

    def test_field_f1_perfect(self):
        base = {"page_id": "P1", "bbox": [0, 0, 1, 1], "line_id": "L1",
                "quote": "甘草", "source_hash": "x"}
        rec = {**base, "type": "药材", "name": "甘草", "value": "甘草"}
        r = compute_field_f1([rec], [dict(rec)])
        self.assertEqual(r["status"], "实测")
        self.assertEqual(r["value"], 1.0)

    def test_refusal_missing_counts_as_wrong(self):
        gold = [{"question": "黄芪用量？", "expected_refused": True}]
        r = compute_refusal(gold, [])
        self.assertEqual(r["value"], 0.0)   # 未跑批计入分母，不美化
        self.assertEqual(r["missing"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
