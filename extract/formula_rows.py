"""从步骤 2 的单页方剂候选范围中抽取逐味药量与独立原文出处。"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

from app.schemas import Block
from ocr.normalize import normalize_text
from .formula_segments import _Stream, _herb_hits, extract_formula_candidates
from .rules import _DOSE_TAIL_R, _resolve_herb
from .schema import DOSE_RE

# 古籍原书写法未完全覆盖在现有词表中；不改变全局术语表。
_ALIASES = tuple(normalize_text(t) for t in (
    '芍藥', '正白苓', '吳茱萸', '良薑', '殭蠶', '正嫩烏藥', '真鴉片', '石草霜', '西牛黃', '麝香',
))
_DOSE = re.compile(r'(?:各\s*)?(?:等分|少[許许]|少量|自\s*' + DOSE_RE + r'(?:可(?:以)?加至)' + DOSE_RE + r'|' + DOSE_RE + r')')
_AMBIGUOUS = re.compile(r'[?？□�〔〕\[\]◯〇]|不[明清]|模糊')
_AUTO_CONFIDENCE = 0.90
# 不以宽泛的“共為”或单独“水”作为完整结束证据。
_AUTO_END = re.compile(r'(?:共为细末|共捣碎|右[一二三四五六七八九十\d]+味|(?:水|酒)(?:[一二三四五六七八九十\d]+[钟鍾盏盞升合碗杯])?煎|煎服)')
_ALLOWED_GAP = re.compile(r'^[\s，。、；:：()（）]*(?:(?:酒炒|土炒|炮|炒|炙|製|制|去皮|生用|打碎)[\s，。、；:：()（）]*)*$')


def _herbs(text: str) -> list[dict[str, Any]]:
    raw = [(p, p + len(term), '词典') for p, term in _herb_hits(text)]
    for term in _ALIASES:
        for m in re.finditer(re.escape(term), text):
            raw.append((m.start(), m.end(), '原书写法'))
    # 复用既有药量规则的词典外候选，但明确标待核，不悄然当作确认药名。
    dose_ranges = [(m.start(), m.end()) for m in _DOSE.finditer(text)]
    for m in _DOSE_TAIL_R.finditer(text):
        term, kind = _resolve_herb(m.group(1))
        if kind == 'rules_dose_ctx' and term:
            start = m.start(1) + m.group(1).rfind(term)
            if any(a <= start < b for a, b in dose_ranges):
                continue
            raw.append((start, start + len(term), '药量上下文待核'))
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[2]))
    selected = []
    for a, b, kind in raw:
        if selected and a < selected[-1]['end']:
            continue
        selected.append({'start': a, 'end': b, 'method': kind})
    return selected


def _intersects(start: int, end: int, intervals: Sequence[Sequence[int]]) -> bool:
    return any(start < b and a < end for a, b in intervals)


def _uncertain(source: dict, low_refs: set[str]) -> bool:
    return any(s['block_ref'] in low_refs or s['mapping_precision'] in ('block', 'segment')
               for s in source['sources'])


def _auto_absence_proof(stream: _Stream, blocks: Sequence[Block], candidate: dict,
                        herbs: Sequence[dict], uncertain_ranges: Sequence[Sequence[int]],
                        low_confidence: float) -> bool:
    """仅对完整、高置信、无剂量残字的整段作保守的机器缺载判断。"""
    if (candidate.get('end_reason') != '制法服法'
            or candidate.get('continuation') == '可能续页' or len(herbs) < 2):
        return False
    a, b = candidate['composition']['normalized_range']
    text = stream.norm[a:b]
    marker = _AUTO_END.match(stream.norm, b)
    if not marker or _DOSE.search(text) or _AMBIGUOUS.search(text):
        return False
    if _intersects(a, marker.end(), uncertain_ranges):
        return False
    cursor = 0
    for herb in herbs:
        if herb['method'] == '药量上下文待核' or not _ALLOWED_GAP.fullmatch(text[cursor:herb['start']]):
            return False
        cursor = herb['end']
    if not _ALLOWED_GAP.fullmatch(text[cursor:]):
        return False
    # 首个组方块必须从块首开始，避免把被截去的上文误认为完整配伍。
    parts = [p for p in stream.parts if p['start'] <= a < p['end']]
    if not parts or a != parts[0]['start']:
        return False
    cue = candidate.get('start_cue') or {}
    cue_range = cue.get('normalized_range')
    if not cue_range or not (0 <= cue_range[0] < cue_range[1] <= a):
        return False
    if _intersects(cue_range[0], marker.end(), uncertain_ranges):
        return False
    relevant = [p for p in stream.parts if p['start'] < marker.end() and p['end'] > cue_range[0]]
    if not relevant:
        return False
    refs = [p['block'].block_ref or p['block'].line_id for p in relevant]
    if any(not x for x in refs) or len(set(refs)) != len(refs):
        return False
    orders = [p['block'].reading_order for p in relevant]
    if any(not isinstance(n, int) or n <= 0 for n in orders):
        return False
    if any(right != left + 1 for left, right in zip(orders, orders[1:])):
        return False
    for p in relevant:
        block = p['block']
        if (not math.isfinite(block.conf) or not max(_AUTO_CONFIDENCE, low_confidence) <= block.conf <= 1.0
                or len(block.bbox) != 4
                or block.bbox[0] < 0 or block.bbox[1] < 0
                or block.bbox[0] >= block.bbox[2] or block.bbox[1] >= block.bbox[3]
                or normalize_text(block.text) != stream.norm[p['start']:p['end']]):
            return False
        # 变长对齐不能支持字符级完整性证明。
        if any(level in ('segment', 'block') for _, _, level in p['alignment']):
            return False
    # _Stream 跳过表格块；区间内有跳过块时不将缺失解释为无剂量。
    if any(isinstance(x, Block) and orders[0] <= x.reading_order <= orders[-1]
           and (x.kind == 'table_cell' or (x.block_ref or x.line_id) not in refs) for x in blocks):
        return False
    return True


def extract_formula_rows(blocks: Iterable[Block], candidates: Sequence[dict] | None = None, *,
                         verified_complete: bool = False,
                         uncertain_ranges: Sequence[Sequence[int]] = (),
                         low_confidence: float = 0.7) -> dict[str, Any]:
    """将候选转为方剂/药味行；verified_complete 只用于人工核实的完整转写。

    uncertain_ranges 使用步骤 2 返回的全页归一化半开区间。默认 OCR 路径
    对无法确认缺载的剂量标待核；符合严格的机器证据或人工核实条件时标原文未载。
    低于阈值的 OCR 块、来源映射粗糙或显式不确定区间的字段标待核。
    """
    blocks = list(blocks)
    stream = _Stream(blocks)
    if candidates is None:
        candidates = extract_formula_candidates(blocks)['candidates']
    low_refs = {b.block_ref or b.line_id for b in blocks
                if isinstance(b, Block) and b.conf < low_confidence
                and not (verified_complete and b.conf == 0)}
    groups = []
    for candidate in candidates:
        if candidate.get('page_id') != stream.page_id:
            raise ValueError('候选与 OCR 块不属于同一页')
        comp = candidate['composition']
        a, b = comp['normalized_range']
        if not (0 <= a <= b <= len(stream.norm)) or stream.norm[a:b] != comp['text_norm']:
            raise ValueError('候选配伍范围与当前 OCR 文本不匹配')
        text = stream.norm[a:b]
        herbs = _herbs(text)
        auto_absence = _auto_absence_proof(stream, blocks, candidate, herbs, uncertain_ranges, low_confidence)
        if auto_absence:
            marker = _AUTO_END.match(stream.norm, b)
            absence_evidence = {'start_cue': candidate['start_cue'],
                                'terminator': stream.source(b, marker.end()),
                                'min_confidence': max(_AUTO_CONFIDENCE, low_confidence)}
        else:
            absence_evidence = None
        rows = []
        for h in herbs:
            source = stream.source(a + h['start'], a + h['end'])
            unsure = (h['method'] == '药量上下文待核' or _uncertain(source, low_refs)
                      or _intersects(a + h['start'], a + h['end'], uncertain_ranges))
            rows.append({'herb': {'value': '待核' if unsure else source['text'],
                                  'status': '待核' if unsure else '已识别', 'source': source},
                         'dose': None, 'herb_method': h['method']})
        unassigned = []
        for idx, h in enumerate(herbs):
            unassigned.append(idx)
            boundary = herbs[idx + 1]['start'] if idx + 1 < len(herbs) else len(text)
            segment = text[h['end']:boundary]
            match = _DOSE.search(segment)
            if not match:
                continue
            ds, de = h['end'] + match.start(), h['end'] + match.end()
            src = stream.source(a + ds, a + de)
            before = segment[:match.start()]
            matched = match.group(0)
            shared = matched.startswith('各') or matched == '等分'
            assigned = list(unassigned) if shared else [idx]
            # 未注明“各”时不能把前面的无剂量药味随意归给最后药味。
            if shared:
                unassigned.clear()
            else:
                unassigned.clear()
            after = segment[match.end():]
            is_unsure = (not _ALLOWED_GAP.fullmatch(before) or not _ALLOWED_GAP.fullmatch(after)
                         or _AMBIGUOUS.search(segment)
                         or _uncertain(src, low_refs) or _intersects(a + ds, a + de, uncertain_ranges))
            raw_dose = src['text'].strip()
            shown = re.sub(r'^各\s*', '', raw_dose) if matched.startswith('各') else raw_dose
            for target in assigned:
                rows[target]['dose'] = {'value': '待核' if is_unsure else shown,
                                        'status': '待核' if is_unsure else '已识别',
                                        'source': src, 'shared': shared}
        for idx, row in enumerate(rows):
            if row['dose'] is not None:
                continue
            h = herbs[idx]
            next_start = herbs[idx + 1]['start'] if idx + 1 < len(herbs) else len(text)
            trailing = text[h['end']:next_start]
            manual_absence = (verified_complete and candidate.get('end_reason') == '制法服法'
                                 and row['herb']['status'] == '已识别'
                                 and bool(_ALLOWED_GAP.fullmatch(trailing))
                                 and not _AMBIGUOUS.search(trailing)
                                 and not _intersects(a + h['start'], a + next_start, uncertain_ranges)
                                 and not _uncertain(row['herb']['source'], low_refs))
            missing_confirmed = manual_absence or auto_absence
            row['dose'] = {'value': '原文未载' if missing_confirmed else '待核',
                           'status': '原文未载' if missing_confirmed else '待核',
                           'source': None, 'shared': False,
                           'absence_basis': ('人工核实' if manual_absence else '自动结构判定' if auto_absence else None),
                           'absence_evidence': absence_evidence if auto_absence and not manual_absence else None,
                           'evidence_context': comp}
        name = candidate['name']
        if name is None:
            name_field = {'value': '无名方', 'status': '无名方', 'source': candidate['start_cue']}
        else:
            na, nb = name['normalized_range']
            name_uncertain = _uncertain(name, low_refs) or _intersects(na, nb, uncertain_ranges)
            name_field = {'value': '待核' if name_uncertain else name['text'],
                          'status': '待核' if name_uncertain else '已识别', 'source': name}
        groups.append({'page_id': stream.page_id, 'name_status': candidate['name_status'],
                       'name_field': name_field,
                       'name': candidate['name'], 'start_cue': candidate['start_cue'],
                       'continuation': candidate['continuation'], 'composition': comp,
                       'context': candidate['passage'], 'rows': rows})
    return {'page_id': stream.page_id, 'formulas': groups}
