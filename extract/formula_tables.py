"""正文方剂结果转成兼容原有表格接口的可追溯网格。"""
from __future__ import annotations

from typing import Any
from app.schemas import Block, source_hash_of
from ocr.normalize import normalize_text


def _source_refs(field: dict | None) -> list[str]:
    if not field:
        return []
    source = field.get('source') or field.get('evidence_context') or {}
    return list(dict.fromkeys(s.get('block_ref', '') for s in source.get('sources', [])
                              if s.get('block_ref')))


def _same_as_printed(formula: dict, tables: list[dict], blocks: list[Block]) -> bool:
    """仅当每行药、量分别完整对应同一原书单元格来源和文本时判重。"""
    by_ref = {b.block_ref or b.line_id: b for b in blocks}
    expected = []
    for row in formula.get('rows', []):
        fields = [row['herb'], row['dose']]
        evidence = []
        for field in fields:
            if field.get('status') != '已识别':
                return False
            src = (field.get('source') or {}).get('sources') or []
            if len(src) != 1:
                return False
            s = src[0]
            block = by_ref.get(s.get('block_ref'))
            if (block is None or s.get('block_raw_range') != [0, len(block.text)]
                    or normalize_text(field.get('value', '')) != normalize_text(block.text)):
                return False
            evidence.append((s['block_ref'], normalize_text(block.text)))
        expected.append(tuple(evidence))
    if not expected:
        return False
    for table in tables:
        rows, refs = table.get('rows') or [], table.get('cell_refs') or []
        actual = set()
        for row, line in zip(rows, refs):
            for i in range(min(len(row), len(line)) - 1):
                actual.add(((line[i], normalize_text(str(row[i] or ''))),
                            (line[i + 1], normalize_text(str(row[i + 1] or '')))))
        if all(pair in actual for pair in expected):
            return True
    return False


def text_formula_tables(extraction: dict, printed_tables: list[dict], blocks: list[Block],
                        is_demo: bool) -> tuple[list[dict], int]:
    out = []
    suppressed = 0
    for formula in extraction.get('formulas', []):
        if not formula.get('rows'):
            continue
        if _same_as_printed(formula, printed_tables, blocks):
            suppressed += 1
            continue
        cue = formula.get('start_cue') or {}
        comp = formula.get('composition') or {}
        identity = [(x.get('block_ref'), x.get('block_raw_range')) for x in cue.get('sources', [])]
        identity += [(x.get('block_ref'), x.get('block_raw_range')) for x in comp.get('sources', [])]
        page_id = formula.get('page_id', '')
        table_id = f"{page_id}-F{source_hash_of(page_id, identity)}"
        rows = [['方名', '藥名', '劑量']]
        multi = [[[], [], []]]
        statuses = [['标题', '标题', '标题']]
        scopes = [['无', '无', '无']]
        details = [[None, None, None]]
        name = formula['name_field']
        for line in formula['rows']:
            fields = [name, line['herb'], line['dose']]
            rows.append([field.get('value', '') for field in fields])
            multi.append([[] if field.get('status') == '无名方' else _source_refs(field)
                          for field in fields])
            statuses.append([field.get('status', '') for field in fields])
            scopes.append(['无' if field.get('status') == '无名方' else
                           '原文字段' if field.get('source') else
                           '配伍上下文' if field.get('evidence_context') else '无'
                           for field in fields])
            details.append(fields)
        refs = [[cell[0] if cell else '' for cell in line] for line in multi]
        filled = sum(bool(str(c or '').strip()) for row in rows for c in row)
        located = sum(bool(c) for row in multi for c in row)
        out.append({'table_id': table_id, 'page_id': page_id, 'source_type': '正文抽取',
                    'rows': rows, 'cell_refs': refs, 'cell_refs_multi': multi,
                    'cell_statuses': statuses, 'cell_ref_scopes': scopes, 'field_details': details,
                    'continuation': formula.get('continuation', ''), 'start_cue': cue,
                    'source_hash': source_hash_of(page_id, identity), 'is_demo': bool(is_demo),
                    'refs_source': '正文出处',
                    'stats': {'rows': len(rows), 'cols': 3, 'filled_cells': filled,
                              'located_cells': located}})
    return out, suppressed
