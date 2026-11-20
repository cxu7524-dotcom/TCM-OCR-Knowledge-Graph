"""正文方剂候选切段；不改变现有实体、关系或规整表管线。

所有区间为半开区间。匹配用 text_norm，展示及定位用 Block.text。
归一化与原文长度不同时，变更区间保守映射到原文对应片段；插入字符
无法独立回指时映射整块并标记 block 精度。调用方须保持单页输入。
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import statistics
from typing import Any, Iterable

from app.schemas import Block
from ocr.normalize import normalize_text
from .rules import _HERB_IDX, _formula_hits, _pair_herb_dose, _scan
from .schema import DISEASES, DOSE_RE

# 现有药材的常见原书写法；保持候选检测局部，不更改通用实体词表。
_EXTRA_HERBS = tuple(normalize_text(x) for x in ('白礬', '牙皂角', '朴硝', '連翹', '山梔子'))
_STOP = re.compile(r'不加引|引加|共[捣搗搞]碎|共[为為]|(?:水|酒)[一二三四五六七八九十\d]+[钟鍾盏盞升合碗杯]|[右上]([一二三四五六七八九十\d]+味)|(?:水|酒)[一二三四五六七八九十\d]*[钟鍾盏盞升合碗杯]*[煎煮]|(?:水|酒)煎|煎服|煎[温溫热熱]|每服|研[为為]|[为為]末')
_ANON = re.compile(r'又一方|又方')
_DISEASE_HEADINGS = tuple(sorted(DISEASES | {normalize_text('心痛'), normalize_text('寒痛')}, key=lambda x: (-len(x), x)))
_DOSE_AFTER_HERB = re.compile(r'(?:各)?' + DOSE_RE)
_MENTION_PREFIX = re.compile(r'(?:即|用|服|如|名|宜|加|与|以|依|方见|方見)$')
_CONDITIONAL = re.compile(r'^(?:即|卽|用|服|如|名|宜|加|与|以|依|方见|方見)')
_PREPARATION = re.compile(r'^(?:不加引|共|水|酒|每服|煎服|研末)')
_DERIVATIVE = re.compile(r'^\s*(?:即|卽).{0,20}(?:加|去|减|減)')


def _alignment(raw: str, norm: str) -> list[tuple[int, int, str]]:
    """归一化每字符对应块内原文区间；不确定时保守返回整个原文块。"""
    result: list[tuple[int, int, str] | None] = [None] * len(norm)
    for tag, a, b, c, d in SequenceMatcher(None, raw, norm, autojunk=False).get_opcodes():
        if tag == 'equal':
            for off in range(d - c):
                result[c + off] = (a + off, a + off + 1, 'exact')
        elif tag == 'replace' and b - a == d - c:
            for off in range(d - c):
                result[c + off] = (a + off, a + off + 1, 'aligned')
        elif tag == 'replace' and b > a:
            for off in range(c, d):
                result[off] = (a, b, 'segment')
        elif tag == 'insert':
            for off in range(c, d):
                result[off] = (0, len(raw), 'block')
    return [v if v is not None else (0, len(raw), 'block') for v in result]


def _herb_hits(text: str) -> list[tuple[int, str]]:
    hits = _scan(text, _HERB_IDX)
    for term in _EXTRA_HERBS:
        for match in re.finditer(re.escape(term), text):
            if not any(match.start() < p + len(t) and p < match.end() for p, t in hits):
                hits.append((match.start(), term))
    # 大、小及常见产地字可修饰药名；仅在段首或明确剂量之后延伸。
    extended = []
    for pos, term in hits:
        if pos and text[pos - 1] in '大小淮川杭广' and (pos == 1 or
                re.search(r'(?:两|钱|分|厘|枚|片|升|合|斤|勺|盏|钟|盅|铢|半)[\s，、]*$', text[:pos-1])):
            pos -= 1
            term = text[pos] + term
        extended.append((pos, term))
    return sorted(extended, key=lambda v: (v[0], -len(v[1])))


def _explicit_single_dose(text: str, hits: list[tuple[int, str]]) -> bool:
    """单味成方须药名后紧接剂量；不从煎服或条件加减反推。"""
    if len(hits) != 1 or hits[0][0] != 0:
        return False
    dose = _DOSE_AFTER_HERB.match(text[len(hits[0][1]):])
    if not dose:
        return False
    tail = text[len(hits[0][1]) + dose.end():]
    return not re.match(r'^(?:性|味|有|乃|主治|药性|药理|用于|用於|不可)', tail)


def _disease_heading_evidence(text: str) -> bool:
    """病证标题之后紧接药量串；只接受明确剂量或等分等正证据。"""
    text = text.lstrip(' \t\n，、：:')
    if not text or _CONDITIONAL.match(text) or re.match(r'^(?:者|之|论|論|加|减|減)', text):
        return False
    end = _STOP.search(text)
    chunk = text[:end.start()] if end else text
    if not chunk or len(chunk) > 120:
        return False
    hits = _herb_hits(chunk)
    if not hits:
        pairs = _pair_herb_dose(chunk)
        if pairs and chunk.startswith(pairs[0][0]):
            hits = [(0, pairs[0][0])]
    if not hits or hits[0][0] != 0:
        return False
    return (_explicit_single_dose(chunk, hits) or
            (len(hits) >= 2 and (re.search(DOSE_RE, chunk) is not None or '等分' in chunk)))


class _Stream:
    def __init__(self, blocks: Iterable[Block]):
        items = [b for b in blocks if isinstance(b, Block) and b.text
                 and b.kind not in ('table_cell', 'small_column_evidence')]
        pages = {b.page_id for b in items}
        if len(pages) > 1:
            raise ValueError('正文候选只接受单页 blocks')
        self.page_id = next(iter(pages), '')
        self.parts: list[dict[str, Any]] = []
        self.norm = ''
        self.raw = ''
        for b in sorted(items, key=lambda b: (b.reading_order, b.line_id, b.block_ref)):
            norm = b.text_norm or normalize_text(b.text)
            start, raw_start = len(self.norm), len(self.raw)
            self.norm += norm
            self.raw += b.text
            self.parts.append({'block': b, 'start': start, 'end': len(self.norm),
                               'raw_start': raw_start, 'alignment': _alignment(b.text, norm)})

    def source(self, start: int, end: int) -> dict[str, Any]:
        sources = []
        for part in self.parts:
            a, b = max(start, part['start']), min(end, part['end'])
            if a >= b:
                continue
            block = part['block']
            maps = part['alignment'][a - part['start']:b - part['start']]
            raw_a, raw_b = min(x[0] for x in maps), max(x[1] for x in maps)
            levels = {x[2] for x in maps}
            precision = next((x for x in ('block', 'segment', 'aligned', 'exact') if x in levels), 'block')
            sources.append({'block_ref': block.block_ref or block.line_id,
                            'line_id': block.line_id, 'bbox': list(block.bbox),
                            'normalized_range': [a, b], 'block_raw_range': [raw_a, raw_b],
                            'raw_range': [part['raw_start'] + raw_a, part['raw_start'] + raw_b],
                            'text': block.text[raw_a:raw_b], 'mapping_precision': precision})
        return {'normalized_range': [start, end], 'text_norm': self.norm[start:end],
                'text': ''.join(x['text'] for x in sources), 'sources': sources}


def _starts(stream: _Stream) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    body_tops = [part['block'].bbox[1] for part in stream.parts
                 if len(part['block'].bbox) == 4 and len(part['block'].text) >= 8]
    baseline = statistics.median(body_tops) if body_tops else None
    # 只在块首（或显式 title）识别题名，避免在主治、解释和加减中将提及误当新方。
    for part in stream.parts:
        b, t = part['block'], stream.norm[part['start']:part['end']]
        lead = len(t) - len(t.lstrip(' \t\n，。:：'))
        stripped = t[lead:]
        if not stripped or _CONDITIONAL.match(stripped) or _PREPARATION.match(stripped):
            continue
        for pos, name in _formula_hits(stripped):
            if pos != 0:
                continue
            tail = stripped[len(name):]
            # 题名独立成块，或标题与主治/配伍同块；正文长句开头的提及不当题名。
            top_heading = (baseline is not None and len(b.bbox) == 4
                           and b.bbox[1] <= baseline - 10 and len(tail) <= 120
                           and _herb_hits(tail) and _herb_hits(tail)[0][0] == 0
                           and (len(_herb_hits(tail)) >= 2
                                or _explicit_single_dose(tail, _herb_hits(tail))))
            if (len(stripped) <= len(name) + 2 or b.kind == 'title' or top_heading
                    or tail.startswith(('治', '主治', '主', '方用'))):
                starts.append({'start': part['start'] + lead, 'end': part['start'] + lead + len(name),
                               'name_norm': name, 'name_status': '有名方'})
            break
    # 少数字题名可能被 OCR 在相邻块中切开；只接受跨两块的短标题，
    # 并要求第二块在方名之后仅余极少标点，避免拼接普通长句制造方名。
    for left, right in zip(stream.parts, stream.parts[1:]):
        a = left['start']
        t1 = stream.norm[a:left['end']].strip()
        t2 = stream.norm[right['start']:right['end']].strip()
        if not t1 or len(t1) > 5 or not t2 or len(t2) > 5 or _CONDITIONAL.match(t1) or _PREPARATION.match(t1):
            continue
        combined = t1 + t2
        for pos, name in _formula_hits(combined):
            if pos == 0 and len(t1) < len(name) and len(combined) <= len(name) + 2:
                if stream.norm[a:left['end']] == t1 and stream.norm[right['start']:right['end']] == t2:
                    starts.append({'start': a, 'end': a + len(name), 'name_norm': name,
                                   'name_status': '有名方'})
                break
    # 病证标题仅在块首且紧跟明确药量串时可作为无名方起点；
    # 标题本身不是方名。允许正文分在后续相邻块中。
    for part in stream.parts:
        t = stream.norm[part['start']:part['end']]
        if _CONDITIONAL.match(t):
            continue
        for heading in _DISEASE_HEADINGS:
            if not t.startswith(heading):
                continue
            a, b = part['start'], part['start'] + len(heading)
            evidence = stream.norm[b:min(len(stream.norm), b + 120)]
            if _disease_heading_evidence(evidence):
                starts.append({'start': a, 'end': b, 'name_norm': '',
                               'name_status': '无名方', 'start_kind': '病证标题与药量串'})
            break
    for m in _ANON.finditer(stream.norm):
        if _MENTION_PREFIX.search(stream.norm[max(0, m.start() - 2):m.start()]):
            continue
        starts.append({'start': m.start(), 'end': m.end(), 'name_norm': '', 'name_status': '无名方'})
    starts.sort(key=lambda v: (v['start'], v['end']))
    return [v for i, v in enumerate(starts) if not i or v['start'] != starts[i-1]['start']]


def _composition(stream: _Stream, begin: int, end: int, *, allow_single: bool = False) -> tuple[int, int, bool] | None:
    # 在块边界上寻找药味密集的组方区域，允许药味和剂量分散在相邻竖列。
    for part in stream.parts:
        a, b = max(begin, part['start']), min(end, part['end'])
        if a >= b:
            continue
        # 在说明性文字或条件加减里出现的药名不能触发配伍。
        text = stream.norm[a:b]
        if _CONDITIONAL.match(text.lstrip()):
            continue
        first = _herb_hits(text)
        if not first and allow_single:
            trimmed = text.lstrip(' \t\n，、：:')
            pairs = _pair_herb_dose(trimmed)
            if pairs and trimmed.startswith(pairs[0][0]):
                first = [(len(text) - len(trimmed), pairs[0][0])]
        if not first:
            continue
        start = a + first[0][0]
        # 若配伍起点之前已进入制服法，后面的药量不能倒退解释为配伍。
        if _STOP.search(stream.norm, begin, start):
            return None
        stop_match = _STOP.search(stream.norm, start, end)
        stop = stop_match.start() if stop_match else end
        chunk = stream.norm[start:stop]
        hits = _herb_hits(chunk)
        if not hits and allow_single:
            pairs = _pair_herb_dose(chunk)
            if pairs and chunk.startswith(pairs[0][0]):
                hits = [(0, pairs[0][0])]
        if len(hits) < 2 and not (allow_single and not stream.norm[a:start].strip(' \t\n，、：:') and _explicit_single_dose(chunk, hits)):
            continue
        # 紧随剂量、炮制或其他药名，避免论述段偶然出现的两个药名触发。
        if len(chunk) > 200 or re.search(r'所以|之为|之為|药理|此方乃', chunk):
            continue
        return start, stop, bool(stop_match)
    return None


def extract_formula_candidates(blocks: Iterable[Block]) -> dict[str, Any]:
    """返回候选与页首续方提示；输入应为同一页且已有 reading_order 的 OCR 块。

    为保守起见只接受显式起方、后接可辨组方的候选；没有题名的页首
    连续药味作为提示，不推断方名。输出不修改或写入 OCR / 图谱数据。
    """
    stream = _Stream(blocks)
    starts = _starts(stream)
    candidates = []
    hints = []
    for n, item in enumerate(starts):
        end = starts[n+1]['start'] if n+1 < len(starts) else len(stream.norm)
        remainder = stream.norm[item['end']:end]
        if _DERIVATIVE.match(remainder):
            continue
        comp = _composition(stream, item['end'], end, allow_single=True)
        if not comp:
            if (item['name_status'] == '有名方' and end == len(stream.norm)
                    and not _STOP.search(remainder)):
                hints.append({'status': '可能续页', 'reason': '页尾有方名但尚未看到组方',
                              'passage': stream.source(item['start'], end)})
            continue
        a, b, ended = comp
        # 引用「即某方加……」不是独立配伍，即使其后列出多个药味也不成候选。
        if _DERIVATIVE.match(stream.norm[item['end']:a]):
            continue
        name = stream.source(item['start'], item['end'])
        candidates.append({'page_id': stream.page_id, 'name_status': item['name_status'],
                           'name': name if item['name_status'] == '有名方' else None,
                           'start_cue': name, 'start_kind': item.get('start_kind', '方名' if item['name_status'] == '有名方' else '又一方'), 'composition': stream.source(a, b),
                           'passage': stream.source(item['start'], b),
                           'continuation': '可能续页' if not ended and end == len(stream.norm) else '未见续页迹象',
                           'end_reason': '制法服法' if ended else ('下一方' if end < len(stream.norm) else '页尾')})
    limit = starts[0]['start'] if starts else len(stream.norm)
    if limit:
        head = _composition(stream, 0, limit)
        if head and (head[0] <= (stream.parts[0]['end'] if stream.parts else 0)
                     or (stream.parts and len(stream.norm[:stream.parts[0]['end']]) <= 12
                         and head[0] <= (stream.parts[1]['end'] if len(stream.parts) > 1 else 0))):
            a, b, _ = head
            hints.append({'status': '可能续页', 'reason': '页首有配伍文字但没有本页起方',
                          'passage': stream.source(a, b)})
    return {'page_id': stream.page_id, 'candidates': candidates, 'continuation_hints': hints}
