# -*- coding: utf-8 -*-
"""
tests/test_table_display.py —— 步骤二「方剂表还原」的护栏

锁死的四件事：
  1. 表格的行列位置与空白单元格原样保留，且 cell_refs 与 rows 同形；
  2. 每个可定位单元格的 ref 能对上文本块，块文本与格内文本逐格一致（出处可复核）；
  3. 旧记录（无 cell_refs / 无 tables 字段）分别走「逐格核对补齐」与
     「缺少表格字段」两条路，绝不拿文本相似度硬凑出处；
  4. GET /api/pages/{id}/ocr 的返回带 tables / tables_status，且区分
     「本页无表」与「历史记录缺表格字段」。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.schemas import Block, TableStruct               # noqa: E402
from ocr import table as table_mod                        # noqa: E402

PID = "P-TBL"


def _cell(text, x, y, i, w=40, h=40):
    return Block(page_id=PID, bbox=[x, y, x + w, y + h],
                 line_id=f"R{i:03d}", quote=text, source_hash=f"h{i}",
                 text=text, text_norm=text, kind="table_cell", conf=0.9,
                 col_id=0, reading_order=i, block_ref=f"{PID}:R{i:03d}")


def _grid_block():
    """5 行 × 2 列，最后一行缺左格（模拟漏检），用于验证空白位保留。"""
    rows = [("當歸", "三錢"), ("川芎", "二錢"), ("白芍", "三錢"),
            ("熟地", "四錢"), ("甘草", "")]
    blocks, i = [], 1
    for r, (left, right) in enumerate(rows):
        y = 100 + r * 100
        if left:
            blocks.append(_cell(left, 100, y, i)); i += 1
        if right:
            blocks.append(_cell(right, 300, y, i)); i += 1
    return blocks, rows


class TestRestoreTablesProvenance(unittest.TestCase):
    def setUp(self):
        self.blocks, self.expect_rows = _grid_block()
        self.tables = table_mod.restore_tables(self.blocks, PID)

    def test_single_table_with_expected_shape(self):
        self.assertEqual(len(self.tables), 1)
        t = self.tables[0]
        self.assertEqual(len(t.rows), 5)
        self.assertTrue(all(len(r) == 2 for r in t.rows), t.rows)
        self.assertEqual(t.table_id, f"{PID}-T01")

    def test_blank_cell_is_kept(self):
        """漏检格必须留空位，不能被挤掉或补成上一格内容。"""
        self.assertEqual(self.tables[0].rows[4], ["甘草", ""])

    def test_cell_refs_shape_matches_rows(self):
        t = self.tables[0]
        self.assertEqual(len(t.cell_refs), len(t.rows))
        for refs, row in zip(t.cell_refs, t.rows):
            self.assertEqual(len(refs), len(row))

    def test_refs_resolve_to_blocks_with_same_text(self):
        """每个非空 ref 都能对上同文本的块；空位必须是 ""。"""
        t = self.tables[0]
        by_ref = {b.block_ref: b for b in self.blocks}
        for r, row in enumerate(t.rows):
            for c, text in enumerate(row):
                ref = t.cell_refs[r][c]
                if not text:
                    self.assertEqual(ref, "", f"空格却有出处：({r},{c})")
                    continue
                self.assertIn(ref, by_ref, f"出处对不上块：({r},{c}) {ref}")
                self.assertEqual(by_ref[ref].text, text)


class TestAlignCellRefs(unittest.TestCase):
    """旧记录只有 rows + 扁平 cells：只在可逐格核验时补齐出处。"""

    def setUp(self):
        self.blocks = [_cell("當歸", 100, 100, 1), _cell("三錢", 300, 100, 2),
                       _cell("川芎", 100, 200, 3), _cell("二錢", 300, 200, 4)]
        self.rows = [["當歸", "三錢"], ["川芎", "二錢"]]
        self.cells = [b.block_ref for b in self.blocks]

    def test_aligned_when_counts_and_texts_match(self):
        refs = table_mod.align_cell_refs(self.rows, self.cells, self.blocks)
        self.assertEqual(refs, [["P-TBL:R001", "P-TBL:R002"],
                                ["P-TBL:R003", "P-TBL:R004"]])

    def test_refused_when_count_differs(self):
        self.assertEqual(table_mod.align_cell_refs(self.rows, self.cells[:3], self.blocks), [])

    def test_refused_when_text_differs(self):
        """顺序错位导致文本对不上时必须拒绝，不能按位置硬塞。"""
        swapped = [self.cells[1], self.cells[0], self.cells[2], self.cells[3]]
        self.assertEqual(table_mod.align_cell_refs(self.rows, swapped, self.blocks), [])

    def test_refused_when_duplicate_refs(self):
        dup = [self.cells[0], self.cells[0], self.cells[2], self.cells[3]]
        self.assertEqual(table_mod.align_cell_refs(self.rows, dup, self.blocks), [])

    def test_line_id_only_refs_are_accepted(self):
        """ref 只有 line_id（不带 page_id:）时也能核验对齐。"""
        refs = table_mod.align_cell_refs(self.rows, [b.line_id for b in self.blocks], self.blocks)
        self.assertEqual(refs[1][0], "R003")


class TestTablesPayload(unittest.TestCase):
    def _payload(self, page, blocks=None):
        from app.main import _tables_payload
        return _tables_payload(page, blocks or [])

    def test_missing_field_is_reported_not_faked(self):
        """快照里根本没有 tables 字段 → 明确报缺字段，而非「本页无表」。"""
        res = self._payload({"page_id": PID})
        self.assertEqual(res["tables_status"], "no_table_field")
        self.assertEqual(res["tables"], [])
        self.assertIn("历史记录缺少表格字段", res["tables_note"])

    def test_empty_list_means_no_table(self):
        res = self._payload({"page_id": PID, "tables": []})
        self.assertEqual(res["tables_status"], "none")
        self.assertIn("未识别到方剂表", res["tables_note"])

    def test_cell_refs_passthrough(self):
        t = TableStruct(table_id="T1", page_id=PID, rows=[["當歸", "三錢"], ["", ""]],
                        cells=["P-TBL:R001", "P-TBL:R002"],
                        cell_refs=[["P-TBL:R001", "P-TBL:R002"], ["", ""]])
        res = self._payload({"page_id": PID, "tables": [t.to_dict()]})
        self.assertEqual(res["tables_status"], "ok")
        tb = res["tables"][0]
        self.assertEqual(tb["refs_source"], "ocr")
        self.assertEqual(tb["stats"], {"rows": 2, "cols": 2,
                                       "filled_cells": 2, "located_cells": 2})

    def test_legacy_table_derives_refs(self):
        """旧快照（无 cell_refs）但 rows / cells 可逐格核对 → 补齐并标记 derived。"""
        blocks, rows = _grid_block()
        page = {"page_id": PID, "tables": [{
            "table_id": "T-old", "page_id": PID,
            "rows": [list(r) for r in [["當歸", "三錢"], ["川芎", "二錢"]]],
            "cells": ["P-TBL:R001", "P-TBL:R002", "P-TBL:R003", "P-TBL:R004"],
        }]}
        res = self._payload(page, [b.to_dict() for b in blocks])
        tb = res["tables"][0]
        self.assertEqual(tb["refs_source"], "derived")
        self.assertEqual(tb["cell_refs"][1], ["P-TBL:R003", "P-TBL:R004"])
        self.assertIn("逐格核对", res["tables_note"])

    def test_legacy_table_without_cells_marks_missing(self):
        blocks, _rows = _grid_block()
        page = {"page_id": PID, "tables": [{"table_id": "T-old", "page_id": PID,
                                            "rows": [["當歸", "三錢"]], "cells": []}]}
        res = self._payload(page, [b.to_dict() for b in blocks])
        tb = res["tables"][0]
        self.assertEqual(tb["refs_source"], "missing")
        self.assertEqual(tb["cell_refs"], [["", ""]])
        self.assertIn("无法定位", res["tables_note"])

    def test_ragged_rows_are_padded(self):
        page = {"page_id": PID, "tables": [{"rows": [["甲"], ["乙", "丙"]], "cells": []}]}
        tb = self._payload(page)["tables"][0]
        self.assertEqual(tb["rows"], [["甲", ""], ["乙", "丙"]])
        self.assertEqual(tb["stats"]["cols"], 2)


class TestOcrEndpointCarriesTables(unittest.TestCase):
    def test_endpoint_shape(self):
        from app import main
        blocks, _rows = _grid_block()
        snap = {"page": {"page_id": PID, "tables": [t.to_dict() for t in
                                                    table_mod.restore_tables(blocks, PID)]},
                "entities": [], "triples": [], "is_demo": False}
        with (mock.patch.object(main.store, "get_page", return_value=snap),
              mock.patch.object(main.store, "get_blocks", return_value=[])):
            res = main.get_ocr(PID)
        self.assertEqual(res["tables_status"], "ok")
        self.assertEqual(res["tables"][0]["stats"]["rows"], 5)
        self.assertEqual(len(res["tables"][0]["cell_refs"]), 5)
        self.assertIn("blocks", res)

    def test_legacy_snapshot_without_tables_key(self):
        from app import main
        snap = {"page": {"page_id": PID, "blocks": []}, "entities": [],
                "triples": [], "is_demo": False}
        with (mock.patch.object(main.store, "get_page", return_value=snap),
              mock.patch.object(main.store, "get_blocks", return_value=[])):
            res = main.get_ocr(PID)
        self.assertEqual(res["tables_status"], "no_table_field")
        self.assertEqual(res["tables"], [])


class TestSchemaBackwardCompatibility(unittest.TestCase):
    def test_from_dict_without_cell_refs(self):
        t = TableStruct.from_dict({"table_id": "T", "rows": [["甘草"]]})
        self.assertEqual(t.cell_refs, [])

    def test_roundtrip_keeps_cell_refs(self):
        t = TableStruct(rows=[["甘草"]], cell_refs=[["P:R001"]])
        self.assertEqual(TableStruct.from_dict(t.to_dict()).cell_refs, [["P:R001"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
