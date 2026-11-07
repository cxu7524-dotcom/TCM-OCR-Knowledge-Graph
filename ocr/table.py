# -*- coding: utf-8 -*-
"""
ocr/table.py —— 方剂表 / 剂量结构还原

古籍中的方剂常以「药名 + 剂量 + 炮制法」连写（如「桂枝三两 芍药三两 甘草二两炙」），
或排成表格。本模块把 `kind=dose|table_cell` 的文本块还原为结构化行列。

本阶段为**规则桩**：按分隔符切分 + 剂量正则识别，不加载表格结构识别模型。

TODO（下一阶段）：
  - 接入表格线检测与单元格还原，支持跨页表格、无框表、双行小字夹注；
  - 剂量单位换算（两 / 钱 / 分 / 枚 / 升）与炮制法（炙 / 炮 / 酒洗）拆列；
  - 表格结构还原准确率指标所需的 GT 对齐逻辑（见 eval/metrics.py）。
"""

from __future__ import annotations

import re
from typing import Iterable, List

from app.schemas import Block, TableStruct, source_hash_of

# 剂量单元：数字（含中文数字）+ 量词
DOSE_RE = re.compile(r"([一-龥]{0,2}\d+|[一二三四五六七八九十百]+)\s*(两|钱|分|枚|升|合|斤|片|勺|盏)")
SPLIT_RE = re.compile(r"[\s,，;；、]+")


def parse_dose_cell(text: str) -> List[List[str]]:
    """把「桂枝三两 芍药三两」这类连写拆成 [药名, 剂量, 备注] 行。

    输入：一行文本；输出：若干行，每行最多 3 列。解析不出时返回原文单行。
    """
    if not text:
        return []
    rows: List[List[str]] = []
    tokens = [t for t in SPLIT_RE.split(text.strip()) if t]
    buf = ""
    for tok in tokens:
        buf = (buf + tok).strip()
        m = DOSE_RE.search(buf)
        if m:
            dose = m.group(0)
            name = buf[:m.start()].strip()
            rest = buf[m.end():].strip()
            rows.append([name or "（未识别药名）", dose, rest])
            buf = ""
    if buf:
        rows.append([buf, "", ""])
    return rows or [[text.strip(), "", ""]]


def restore_tables(blocks: Iterable[Block], page_id: str = "") -> List[TableStruct]:
    """表格结构还原入口（桩）。

    输入：OCR Block 列表（只看 kind in {table_cell, dose}）
    输出：TableStruct 列表，含 rows / cells(block_ref) / source_hash / is_demo
    异常：内部兜底，永远返回列表，不抛异常。
    """
    out: List[TableStruct] = []
    try:
        candidates = [b for b in blocks
                      if isinstance(b, Block) and b.kind in ("table_cell", "dose")]
        for i, b in enumerate(candidates, start=1):
            rows = parse_dose_cell(b.text)
            out.append(TableStruct(
                table_id=f"{page_id or 'page'}-T{i:02d}",
                page_id=page_id,
                rows=rows,
                cells=[b.block_ref or b.line_id],
                source_hash=source_hash_of(page_id, b.block_ref, b.text),
                is_demo=True,
            ))
    except Exception:
        return []
    return out


def table_cell_count(tables: Iterable[TableStruct]) -> int:
    """还原出的单元格数量，供结构统计展示（不是评测指标）。"""
    return sum(len(t.rows) for t in tables)
