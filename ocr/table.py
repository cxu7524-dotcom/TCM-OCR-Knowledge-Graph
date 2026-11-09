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

# 炮制字闭集纠错：木刻 / 低分辨率下与常见字混淆。
# 依据：实测 p0009 炮制格「擘」被识别为「擎」（conf 0.34），同列「切/炙/去皮」均正确。
# 只对长度 ≤3 的短单元格生效，避免误伤药名与主治；每次纠错写 trace 可回溯。
PROCESS_FIX = {"擎": "擘", "璧": "擘", "擘": "擘"}


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


def fill_missing_cells(image_path: str, blocks: Sequence[Block], page_id: str
                       ) -> Tuple[List[Block], List[Dict[str, Any]]]:
    """表格缺失单元格补识别（不依赖任何 GT 坐标）。

    做法：用**已检测到的单元格**聚类出网格（行带 / 列带），对没有检测框的网格位置
    crop 出图像区域、跳过检测直接走识别模型（`recognize_crop`），识别成功则补一个 Block。
    只在判定为表格页时生效（平均每行块数 ≥1.8 且行列数均 ≥2）。
    """
    try:
        return _fill_missing_cells(image_path, blocks, page_id)
    except Exception:
        # 补格失败只影响表格环节，绝不降级整页 OCR 结果
        return list(blocks), []


def _recognize_cell(im, box, pads: Tuple[int, ...] = (0, 4, 8),
                    scales: Tuple[int, ...] = (2, 3, 4)) -> Tuple[str, float]:
    """单格识别：裁切边界对结果极其敏感，按「原边界 → 外扩 4px → 外扩 8px」依次尝试。

    实测依据（p0009 炮制格）：crop (580,692,644,730) 能识别出字（conf 0.34），
    边界改为 (578,690,646,732) 则直接返回空 —— 差 2px 就从出字变空。
    因此固定边界不可靠，改为多边界 × 多缩放取置信度最高的非空结果。
    """
    try:
        from PIL import Image
        import numpy as np
        from .recognize import recognize_crop
    except ImportError:
        return "", 0.0
    best_text, best_conf = "", 0.0
    for p in pads:
        b = (max(0, int(box[0]) - p), max(0, int(box[1]) - p),
             min(im.width, int(box[2]) + p), min(im.height, int(box[3]) + p))
        if b[2] - b[0] < 8 or b[3] - b[1] < 8:
            continue
        crop = im.crop(b)
        for s in scales:
            try:
                arr = np.array(crop.resize((crop.width * s, crop.height * s), Image.LANCZOS))
            except (ValueError, OSError):
                continue
            text, conf = recognize_crop(arr)
            if text and conf > best_conf:
                best_text, best_conf = text, conf
        if best_text:          # 该边界已出字，不再外扩，避免把邻格字卷进来
            break
    return best_text, best_conf


def _fix_process(text: str, ci: int, trace: List[Dict[str, Any]], page_id: str,
                 ri: int) -> str:
    """炮制列字形纠错：只对第 3 列（ci==2）且长度 ≤3 的短格生效。"""
    if ci != 2 or not text or len(text) > 3:
        return text
    fixed = PROCESS_FIX.get(text)
    if fixed and fixed != text:
        trace.append({"page_id": page_id, "row": ri, "col": ci,
                      "before": text, "after": fixed, "rule": "炮制字形纠错（闭集）"})
        return fixed
    return text


def _fill_missing_cells(image_path: str, blocks: Sequence[Block], page_id: str
                        ) -> Tuple[List[Block], List[Dict[str, Any]]]:
    items = [b for b in blocks if isinstance(b, Block)]
    rows = cluster_rows(items)
    if len(rows) < 2:
        return list(blocks), []
    flat = [b for r in rows for b in r]
    cols = cluster_columns(flat)
    if len(cols) < 2 or sum(len(r) for r in rows) / len(rows) < 1.8:
        return list(blocks), []
    try:
        from PIL import Image
        import numpy as np
        from .recognize import recognize_crop
    except ImportError:
        return list(blocks), []

    try:
        im = Image.open(image_path).convert("RGB")
    except (OSError, ValueError):
        return list(blocks), []

    widths = [max(1.0, _box(b)[2] - _box(b)[0]) for b in flat]
    half = statistics.median(widths) / 2
    row_bands = [(min(_box(b)[1] for b in r), max(_box(b)[3] for b in r)) for r in rows]
    col_bands = [(c - half, c + half) for c in cols]
    added: List[Block] = []
    trace: List[Dict[str, Any]] = []

    # 每列文本长度中位数：用于识别「只识别出半个词」的低质量单元格
    col_med_len: Dict[int, float] = {}
    for ci, (x1, x2) in enumerate(col_bands):
        lens = [len(b.text) for b in flat
                if x1 - 4 <= (_box(b)[0] + _box(b)[2]) / 2 <= x2 + 4]
        col_med_len[ci] = statistics.median(lens) if lens else 0

    for ri, (y1, y2) in enumerate(row_bands):
        for ci, (x1, x2) in enumerate(col_bands):
            occupant = None
            for b in flat:
                bx1, by1, bx2, by2 = _box(b)
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4:
                    occupant = b
                    break
            if occupant is not None:
                # 已占用但文本明显偏短（如「芍藥」只识别出「藥」）→ 整格二次识别
                med = col_med_len.get(ci, 0)
                if med >= 2 and len(occupant.text) < med * 0.6:
                    box = (max(0, int(x1)), max(0, int(y1)),
                           min(im.width, int(x2)), min(im.height, int(y2)))
                    if box[2] - box[0] >= 10 and box[3] - box[1] >= 10:
                        text, conf = _recognize_cell(im, box)
                        text = _fix_process(text, ci, trace, page_id, ri)
                        if text and len(text) > len(occupant.text):
                            trace.append({"page_id": page_id, "row": ri, "col": ci,
                                          "before": occupant.text, "after": text,
                                          "conf": conf, "rule": "短文本整格二次识别"})
                            occupant.text = text
                            occupant.quote = text
                            occupant.source_hash = source_hash_of(
                                "ocr-refill", page_id, occupant.line_id, text)
                continue
            box = (max(0, int(x1)), max(0, int(y1)),
                   min(im.width, int(x2)), min(im.height, int(y2)))
            if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                continue
            crop = im.crop(box)
            text, conf = _recognize_cell(im, box)
            text = _fix_process(text, ci, trace, page_id, ri)
            if not text:
                continue
            line_id = f"C{len(added) + 1:03d}"
            added.append(Block(
                page_id=page_id, bbox=[box[0], box[1], box[2], box[3]],
                line_id=line_id, quote=text,
                source_hash=source_hash_of("ocr-fill", page_id, line_id, text),
                text=text, text_norm=text, kind="table_cell", conf=conf,
                col_id=ci, reading_order=0, block_ref=f"{page_id}:{line_id}",
            ))
            trace.append({"page_id": page_id, "row": ri, "col": ci,
                          "bbox": list(box), "text": text, "conf": conf,
                          "rule": "表格补格（无检测框，跳过检测直接识别）"})
    return list(blocks) + added, trace


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
