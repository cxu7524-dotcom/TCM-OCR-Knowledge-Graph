# -*- coding: utf-8 -*-
"""配伍段逐味抽取（通用入口，与具体页面无关）。

输入只有 OCR 块：由文本和阅读顺序确定「方题 → 配伍段 → 制法/服法边界」，
在段内按顺序逐味产出「组成」关系与待核记录。页面名、块编号、区域坐标、金标
都不进入本模块——它们只在评测阶段（tools/p51_step5c_contract.py）读取。

要点：
  - 标题可能出现在块内：配伍从标题**末尾**开始读同一块的剩余文字；
  - 截断发生在**字符位置**：遇制法/服法或下一处明确标题即止；
  - 同页多方按各自标题与范围归属，不合并；
  - 标题或边界存疑时写入待核，不猜。

药名口径沿用账本 ID 14：药名词表 + 繁简/异体等价（羗→羌、殼→壳），
同位置优先完整长词，逐味互不重叠，允许同块多味、共用剂量、无独立剂量。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.schemas import Block, Triple, source_hash_of
from extract.formula_segments import _Stream, _herb_hits, _starts
from extract.schema import HERBS
from ocr.normalize import normalize_text

EQUIV = {"殼": "壳", "羗": "羌"}
_TRANS = str.maketrans(EQUIV)

# 通用制法/服法边界（比 formula_segments._STOP 多认无剂量的「水搗碎温服」「共爲細末」等）
PREP = re.compile(
    r"(?:"
    # 引药边界：与 extract/formula_segments.py::_STOP 同义（不加引 / 引加），
    # 语义是「基本配伍到此结束，后面是引药与服法」。必须排在剂型/煎法之前，
    # 否则「引加生薑二錢…水煎熱服」会一路读到「水煎」才截断，把引药并进基本组成。
    r"不加引|引加"
    r"|(?:水|酒|米湯|米汤|薑湯|姜汤|童便|井水|甘瀾水|甘澜水|東流水|东流水)[^钱錢]{0,10}(?:搗|捣|研|磨|煎|煮|熬|調|调|服)"
    r"|(?:搗|捣)碎|(?:研|磨)(?:[为為爲])?末|(?:[为為爲])末"
    r"|每服|空心服|食[前后]服|(?:温|溫|熱|热|冷|凉|涼)服|煎服|(?:水|酒)?煎[^钱錢]{0,4}服"
    r"|(?:右|上)[为為爲]|(?:右|上)(?:件|味)|共[为為爲捣搗]|共.{0,1}(?:細|细)末"
    r")")

CANON: Dict[str, str] = {}
for _h in HERBS:
    CANON.setdefault(normalize_text(_h).translate(_TRANS), _h)


def _norm(text: str) -> str:
    return normalize_text(text).translate(_TRANS)


def _prep_boundary(stream: _Stream, begin: int, limit: int) -> Dict[str, Any] | None:
    """在 [begin, limit) 内找最早的制法/服法位置（字符级）。"""
    for part in stream.parts:
        a, b = max(begin, part["start"]), min(limit, part["end"])
        if a >= b:
            continue
        m = PREP.search(stream.norm, a, b)
        if m:
            return {"pos": m.start(), "cue": m.group(0)[:12],
                    "line_id": part["block"].line_id, "block_ref": part["block"].block_ref}
    return None


def _scan_herbs(stream: _Stream, begin: int, end: int) -> List[Dict[str, Any]]:
    """段内逐味扫描：互不重叠，同位置优先长词，返回原文偏移。"""
    out: List[Dict[str, Any]] = []
    for part in stream.parts:
        a, b = max(begin, part["start"]), min(end, part["end"])
        if a >= b:
            continue
        norm = stream.norm[a:b]
        align = part["alignment"][a - part["start"]:b - part["start"]]
        i = 0
        while i < len(norm):
            picked = None
            for L in (4, 3, 2):
                if i + L > len(norm):
                    continue
                key = norm[i:i + L].translate(_TRANS)
                if key in CANON:
                    picked = (L, CANON[key])
                    break
            if picked is None:
                i += 1
                continue
            L, herb = picked
            ra, rb = align[i][0], align[i + L - 1][1]
            block = part["block"]
            out.append({"herb": herb, "surface": block.text[ra:rb],
                        "line_id": block.line_id, "block_ref": block.block_ref or block.line_id,
                        "bbox": list(block.bbox), "start": ra, "end": rb,
                        "quote": block.text, "source_hash": block.source_hash})
            i += L
    return out


def extract_composition(blocks: Iterable[Block]) -> Dict[str, Any]:
    """OCR 块 → {segments, relations, pending}。不读页面名/金标。"""
    items = [b for b in blocks if isinstance(b, Block)]
    if not items:
        return {"page_id": "", "segments": [], "relations": [], "pending": []}
    stream = _Stream(items)
    starts = [s for s in _starts(stream) if s["name_norm"]]
    segments: List[Dict[str, Any]] = []
    relations: List[Triple] = []
    pending: List[Dict[str, Any]] = []

    for idx, s in enumerate(starts):
        begin = s["end"]
        limit = starts[idx + 1]["start"] if idx + 1 < len(starts) else len(stream.norm)
        prep = _prep_boundary(stream, begin, limit)
        end = min(limit, prep["pos"]) if prep else limit
        name_src = stream.source(s["start"], s["end"])
        title = name_src["text"]
        herbs = _scan_herbs(stream, begin, end) if end > begin else []
        seg_blocks = sorted({h["line_id"] for h in herbs})
        segments.append({"index": idx, "title": title, "title_source": name_src,
                         "begin": begin, "end": end, "text": stream.norm[begin:end],
                         "boundary": prep, "herb_count": len(herbs),
                         "herb_blocks": seg_blocks,
                         "next_title": starts[idx + 1]["name_norm"] if idx + 1 < len(starts) else None})
        if not herbs:
            pending.append({"reason": "段内未识别到药名", "title": title,
                            "range": [begin, end], "text": stream.norm[begin:end][:60]})
        if prep is None and end == len(stream.norm):
            pending.append({"reason": "未见制法边界（段落到页尾）", "title": title,
                            "range": [begin, end]})
        if prep and not herbs:
            pending.append({"reason": "制法出现在配伍之前", "title": title,
                            "cue": prep["cue"], "range": [begin, prep["pos"]]})

        for h in herbs:
            block = next(b for b in items if b.line_id == h["line_id"])
            triple_hash = block.source_hash or source_hash_of("ocr", block.page_id,
                                                              block.line_id, block.text)
            subject_ev = [{"block_ref": name_src["sources"][0]["block_ref"],
                           "line_id": name_src["sources"][0]["line_id"],
                           "quote": name_src["sources"][0]["text"],
                           "start": name_src["sources"][0]["block_raw_range"][0],
                           "end": name_src["sources"][0]["block_raw_range"][1],
                           "surface": title,
                           "bbox": name_src["sources"][0]["bbox"]}]
            # 药名证据：quote 用原字形（如「羗活」），与 [start,end) 一一对应
            rel_ev = [{"block_ref": h["block_ref"], "line_id": h["line_id"], "bbox": h["bbox"],
                       "quote": h["surface"], "start": h["start"], "end": h["end"],
                       "surface": h["surface"], "source_hash": triple_hash,
                       "mapping_precision": "exact"}]
            relations.append(Triple(
                page_id=block.page_id, bbox=list(h["bbox"]), line_id=h["line_id"],
                quote=h["quote"], source_hash=triple_hash,
                head=title, relation="组成", tail=h["herb"],
                evidence_refs=[h["block_ref"]], subject_rule="segment_scan",
                is_demo=False, subject_evidence=subject_ev, relation_evidence=rel_ev))

    return {"page_id": stream.page_id, "segments": segments,
            "relations": relations, "pending": pending}
