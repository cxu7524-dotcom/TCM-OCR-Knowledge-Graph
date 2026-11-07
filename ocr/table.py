# -*- coding: utf-8 -*-
"""
ocr/table.py —— 方剂表 / 剂量结构还原（阶段二：基于 bbox 行列聚类）

输入是 **OCR 真实检测框**，不使用 GT 坐标：
  1. 剔除页眉 / 页脚（宽高比 >= 4 的横排长条）；
  2. 按 bbox 的 y 中心聚类成行（自适应阈值 = 中位框高 × 0.8）；
  3. 行内按 x 升序，得到 rows 二维文本数组；
  4. 行列数达标才判定为表格，否则返回空（竖排正文不会被误判）；
  5. 每个单元格保留 quote / bbox / block_ref，药味 / 剂量 / 单位 / 炮制由列位置解析。

「可用水平」判定口径（与 eval/metrics.py 一致）：
  行列结构正确率与单元格内容正确率**同时 ≥ 0.80** 才算可用；
  结构 = 各行长度序列，内容 = 归一化后的单元格文本序列。

TODO（阶段三）：无框表、跨页表、合并单元格、双行小字夹注的结构还原。
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from app.schemas import Block, TableStruct, norm_for_match, source_hash_of

HEADER_ASPECT = 4.0
ROW_GAP_RATIO = 0.8
DOSE_RE = re.compile(r"([一二三四五六七八九十百\d]+)\s*(两|兩|钱|錢|分|枚|升|合|斤|片|勺|盏|盞)")
SPLIT_RE = re.compile(r"[\s,，;；、]+")


def parse_dose_cell(text: str) -> List[List[str]]:
    """阶段一兼容接口：把「桂枝三两 芍药三两」连写拆成 [药名, 剂量, 备注] 行。"""
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


def _box(b: Block) -> Tuple[float, float, float, float]:
    try:
        return tuple(float(v) for v in b.bbox[:4])  # type: ignore[return-value]
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0, 0.0)


def _is_header(b: Block) -> bool:
    x1, y1, x2, y2 = _box(b)
    return (x2 - x1) / max(1.0, y2 - y1) >= HEADER_ASPECT


def cluster_rows(blocks: Sequence[Block],
                 gap_ratio: float = ROW_GAP_RATIO) -> List[List[Block]]:
    """按 y 中心聚类成行，行内按 x 升序。"""
    items = [b for b in blocks if isinstance(b, Block) and not _is_header(b)]
    if not items:
        return []
    heights = [max(1.0, _box(b)[3] - _box(b)[1]) for b in items]
    threshold = max(12.0, statistics.median(heights) * gap_ratio)
    ordered = sorted(items, key=lambda b: (_box(b)[1] + _box(b)[3]) / 2)
    rows: List[List[Block]] = []
    cur: List[Block] = [ordered[0]]
    cur_y = (_box(ordered[0])[1] + _box(ordered[0])[3]) / 2
    for b in ordered[1:]:
        cy = (_box(b)[1] + _box(b)[3]) / 2
        if cy - cur_y <= threshold:
            cur.append(b)
            cur_y = sum((_box(x)[1] + _box(x)[3]) / 2 for x in cur) / len(cur)
        else:
            rows.append(sorted(cur, key=lambda x: _box(x)[0]))
            cur, cur_y = [b], cy
    rows.append(sorted(cur, key=lambda x: _box(x)[0]))
    return rows


def cluster_columns(blocks: Sequence[Block],
                    gap_ratio: float = 0.8) -> List[float]:
    """全局 x 列聚类，返回**从左到右（x 升序）**的列中心。

    方剂表为横排表格，列序与 GT 一致为左→右；竖排正文的列序由 layout.py 另行处理。
    """
    items = [b for b in blocks if isinstance(b, Block) and not _is_header(b)]
    if not items:
        return []
    widths = [max(1.0, _box(b)[2] - _box(b)[0]) for b in items]
    threshold = max(12.0, statistics.median(widths) * gap_ratio)
    centers = sorted((_box(b)[0] + _box(b)[2]) / 2 for b in items)
    groups: List[List[float]] = [[centers[0]]]
    for c in centers[1:]:
        if c - (sum(groups[-1]) / len(groups[-1])) <= threshold:
            groups[-1].append(c)
        else:
            groups.append([c])
    # 横排表格：从左到右（x 小 → 大），与 GT 的 rows 列序对齐
    return [sum(g) / len(g) for g in groups]


def split_dose(text: str) -> Dict[str, str]:
    """把「三兩 / 十二枚」拆成 数值 / 单位；非剂量返回空。"""
    m = DOSE_RE.search(text or "")
    if not m:
        return {"value": "", "unit": "", "raw": text or ""}
    return {"value": m.group(1), "unit": m.group(2), "raw": text}


def restore_tables(blocks: Iterable[Block], page_id: str = "") -> List[TableStruct]:
    """表格还原入口（真实 bbox 聚类）。异常时返回空列表，不抛异常。"""
    try:
        items = [b for b in blocks if isinstance(b, Block)]
        rows = cluster_rows(items)
        if len(rows) < 2:
            return []
        widths = [len(r) for r in rows]
        if max(widths) < 2 or sum(widths) / len(widths) < 1.8:
            return []  # 竖排正文（每行多为 1 块）不判为表格

        # 列聚类 + 缺失单元格补空位：让行列结构不被个别漏检打散
        flat = [b for row in rows for b in row]
        col_centers = cluster_columns(flat)
        widths = [max(1.0, _box(b)[2] - _box(b)[0]) for b in flat]
        tolerance = statistics.median(widths) * 1.5
        matrix: List[List[str]] = []
        for row in rows:
            line = [""] * len(col_centers)
            for b in row:
                cx = (_box(b)[0] + _box(b)[2]) / 2
                j = min(range(len(col_centers)),
                        key=lambda k: abs(col_centers[k] - cx)) if col_centers else 0
                if col_centers and abs(col_centers[j] - cx) <= tolerance and not line[j]:
                    line[j] = b.text
                elif col_centers and not line[j]:
                    line[j] = b.text
            matrix.append(line or [b.text for b in row])
        cells = [b.block_ref or b.line_id for row in rows for b in row]
        quote = "".join("".join(r) for r in matrix)
        return [TableStruct(
            table_id=f"{page_id}-T01", page_id=page_id,
            rows=matrix, cells=cells,
            source_hash=source_hash_of("table", page_id, quote),
            is_demo=all(getattr(b, "is_demo", False) for b in items) if items else True,
        )]
    except Exception:
        return []


def structured_cells(table: TableStruct) -> List[Dict[str, Any]]:
    """给前端 / 报告用的结构化单元格：药味 / 剂量值 / 单位 / 炮制（按列位置）。"""
    out: List[Dict[str, Any]] = []
    for r, row in enumerate(table.rows):
        for c, text in enumerate(row):
            dose = split_dose(text)
            out.append({
                "row": r, "col": c, "text": text,
                "field": ["药味", "剂量", "炮制"][c] if c < 3 else f"列{c+1}",
                "dose_value": dose["value"], "dose_unit": dose["unit"],
                "norm": norm_for_match(text),
                "page_id": table.page_id, "table_id": table.table_id,
                "source_hash": table.source_hash,
            })
    return out


def table_cell_count(tables: Iterable[TableStruct]) -> int:
    return sum(len(t.rows) for t in tables)
