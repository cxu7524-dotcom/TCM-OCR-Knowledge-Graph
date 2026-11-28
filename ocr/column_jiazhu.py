from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import PROJECT_ROOT
from app.schemas import Block, source_hash_of
from .normalize import normalize_text
from .recognize import CropCTCResult, map_chars_to_y_spans, recognize_crop, recognize_crop_with_ctc
from .column_segments import segment_column

_CUES = tuple('錢分兩厘枚片斤升合寸半各酒炒製炙去洗研拌炮焙煨薑蜜醋鹽土麩皮心核尖詳見按注亦可') + ('童便', '生用', '炙用', '炒用', '熟用', '另研', '少許', '等分', '為末', '水煎')


def _norm(text: str) -> str:
    return normalize_text(text or '')


@lru_cache(maxsize=1)
def load_tcm_lexicon() -> frozenset[str]:
    terms = set(_CUES)
    try:
        data = json.loads((PROJECT_ROOT / 'data' / 'lexicon' / '中医术语表.json').read_text(encoding='utf-8'))
        for key, values in data.items():
            if isinstance(values, list) and key not in {'来源'}:
                terms.update(v for v in values if isinstance(v, str) and v.strip())
    except (OSError, ValueError, TypeError):
        pass
    return frozenset(_norm(term) for term in terms if term)


def lexicon_coverage(text: str) -> int:
    normalized = _norm(text)
    return sum(1 for term in load_tcm_lexicon() if term in normalized)


def has_lexicon_anomaly(text: str) -> bool:
    return lexicon_coverage(text) == 0


def has_dosage_or_jiazhu_cue(text: str) -> bool:
    normalized = _norm(text)
    return any(_norm(cue) in normalized for cue in _CUES)



@lru_cache(maxsize=1)
def load_tcm_herbs() -> frozenset[str]:
    try:
        data = json.loads((PROJECT_ROOT / 'data' / 'lexicon' / '中医术语表.json').read_text(encoding='utf-8'))
        values = data.get('药材', [])
        return frozenset(name for v in values if isinstance(v, str)
                         if len(name := _norm(v.strip())) >= 2)
    except (OSError, ValueError, TypeError, AttributeError):
        return frozenset()


def has_herb_entity(text: str) -> bool:
    normalized = _norm(text).replace('礬', '矾')
    return any(herb in normalized for herb in load_tcm_herbs())


def count_distinct_herbs(text: str) -> int:
    normalized = _norm(text).replace('礬', '矾')
    return sum(herb in normalized for herb in load_tcm_herbs())


_STRICT_CUES = re.compile(_norm(r'(?:[一二三四五六七八九十百千兩半各每][錢分兩厘枚片斤升合寸]|(?:錢|兩|分)半|酒洗|酒炒|酒製|土炒|薑製|去皮|去心|去核|去尖|另研|生用|炙用|炒用|拌炒|等分|少許)'))


def count_strict_dosage_cues(text: str) -> int:
    return len(_STRICT_CUES.findall(_norm(text)))

def _cjk_count(text: str) -> int:
    return sum(1 for c in text if 0x3400 <= ord(c) <= 0x4dbf or 0x4e00 <= ord(c) <= 0x9fff or 0x20000 <= ord(c) <= 0x3134f or 0xf900 <= ord(c) <= 0xfaff)



def _is_pure_cjk(text: str) -> bool:
    s = text.strip()
    return bool(s) and _cjk_count(s) == len(s)

def _clean_small_text(text: str) -> bool:
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return False
    return sum(1 for c in text if c.isascii() and not c.isspace()) / max(1, len(text)) <= .15


@dataclass
class GateDecision:
    trigger: bool
    reasons: List[str]
    ctc: Optional[CropCTCResult]
    density_ratio: float
    max_y_span_ratio: float
    lexicon_hits: int


def evaluate_column_gate(block: Block, crop_bgr: np.ndarray, ctc_res: Optional[CropCTCResult] = None, *, ctc_rec_fn=None) -> GateDecision:
    h, w = crop_bgr.shape[:2]
    if block.kind != 'paragraph' or w < 16 or h < 1.8 * w:
        return GateDecision(False, [], ctc_res, 0.0, 0.0, 0)
    if ctc_res is None:
        fn = ctc_rec_fn or recognize_crop_with_ctc
        ctc_res = fn(np.ascontiguousarray(np.rot90(crop_bgr)), min_conf=0.0)
    ctc = ctc_res or CropCTCResult()
    cap = max(1.0, h / max(w * .90, 1.0))
    text = block.text or ctc.text or ''
    n = _cjk_count(text)
    density = n / cap
    spans = map_chars_to_y_spans(ctc.chars, 0, h)
    max_span = max((end - start for _, _, start, end in spans), default=0.0) / max(w, 1.0)
    over = n > 1.48 * cap
    under = (h >= 2.5 * w and n < .55 * cap) or max_span >= 1.85
    hits = lexicon_coverage(text)
    if block.conf >= .88 and ctc.conf_mean >= .84 and ctc.conf_min >= .35 and not over:
        return GateDecision(False, [], ctc, density, max_span, hits)
    if block.conf >= .84 and ctc.conf_mean >= .82 and ctc.conf_var < .055 and not over and not under:
        return GateDecision(False, [], ctc, density, max_span, hits)
    reasons = []
    if block.conf < .78 or ctc.conf_mean < .78:
        reasons.append('low_conf')
    if ctc.conf_var >= .055 and ctc.conf_min < .45:
        reasons.append('high_var_valley')
    if ctc.conf_min < .35 and (block.conf < .85 or ctc.conf_var >= .045):
        reasons.append('deep_valley')
    if over:
        reasons.append('density_over')
    if under:
        reasons.append('density_under')
    if not hits and (ctc.conf_min < .50 or density < .70 or density > 1.30):
        reasons.append('lexicon_anomaly')
    return GateDecision(bool(reasons), reasons, ctc, density, max_span, hits)


def prepare_track_crop(crop: np.ndarray, *, is_small: bool, scale: float = 2.0, thin_mode: str = 'mild') -> np.ndarray:
    if crop.size == 0:
        raise ValueError('empty crop')
    if thin_mode not in {'none', 'mild', 'erode'}:
        raise ValueError('unknown thin mode')
    work = np.ascontiguousarray(crop)
    if is_small:
        scale = max(1.0, min(3.0, float(scale)))
        if scale > 1:
            work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if thin_mode == 'mild':
            dilated = cv2.dilate(work, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
            work = cv2.addWeighted(work, .55, dilated, .45, 0)
        elif thin_mode == 'erode':
            work = cv2.dilate(work, np.ones((2, 2), np.uint8), iterations=1)
        pad = max(6, int(round(min(work.shape[:2]) * .18)))
    else:
        pad = max(4, int(round(work.shape[1] * .10)))
    edge = np.concatenate((work[0].reshape(-1, *work.shape[2:]), work[-1].reshape(-1, *work.shape[2:]), work[:, 0].reshape(-1, *work.shape[2:]), work[:, -1].reshape(-1, *work.shape[2:])), axis=0)
    median = np.median(edge, axis=0)
    color = float(median) if work.ndim == 2 else tuple(float(x) for x in median)
    work = cv2.copyMakeBorder(work, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)
    if work.shape[0] > work.shape[1]:
        work = np.rot90(work)
    return np.ascontiguousarray(work)


def reassemble_column_block(image: np.ndarray, block: Block, *, scale: float = 2.0, thin_mode: str = 'mild', rec_fn=None, ctc_rec_fn=None, seg_fn=None, strict_single_track: bool = False, cursive_strict: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[GateDecision]]:
    ih, iw = image.shape[:2]
    bx1, by1, bx2, by2 = (int(v) for v in block.bbox)
    bx1, bx2 = max(0, bx1), min(iw, bx2)
    by1, by2 = max(0, by1), min(ih, by2)
    if bx2 <= bx1 or by2 <= by1:
        return None, None
    crop = image[by1:by2, bx1:bx2]
    gate = evaluate_column_gate(block, crop, ctc_rec_fn=ctc_rec_fn)
    if not gate.trigger:
        return None, gate
    analysis = (seg_fn or segment_column)(crop, keep_bw=True)
    segments = sorted(analysis.segments, key=lambda s: (s.y1, s.y2))
    if not any(s.seg_type in {'dual_track', 'single_track'} for s in segments):
        return None, gate
    if (not any(s.seg_type == 'dual_track' for s in segments)
            and float(block.conf or 0) >= .76 and 'density_under' not in gate.reasons
            and gate.ctc is not None and gate.ctc.conf_min >= .45):
        return None, gate
    h, w = crop.shape[:2]
    wi = max(1, analysis.inner_x2 - analysis.inner_x1)
    spans = map_chars_to_y_spans(gate.ctc.chars, 0, h) if gate.ctc else []
    entries = []
    fn = rec_fn or recognize_crop

    def read(kind, alignment, box, group=None):
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        prepared = prepare_track_crop(crop[y1:y2, x1:x2], is_small=kind != 'large', scale=scale, thin_mode=thin_mode)
        text, conf = fn(prepared)
        text = str(text or '').strip()
        conf = float(conf or 0)
        if not math.isfinite(conf):
            conf = 0.0
        return {'seg_type': kind, 'alignment': alignment, 'bbox': [bx1+x1, by1+y1, bx1+x2, by1+y2], 'text': text, 'conf': conf, 'order': 0, 'group_id': group}

    limit = round(.16 * wi)
    windows = []
    for i, seg in enumerate(segments):
        raw_gap = max(0, seg.y1 - (segments[i-1].y2 if i else 0))
        above = raw_gap // 2
        below = max(0, (segments[i+1].y1 if i+1 < len(segments) else h) - seg.y2) // 2
        if seg.seg_type == 'single_track':
            above = max(0, int(raw_gap * .75) - 1) if i else raw_gap
        up_limit = round(.38 * wi) if seg.seg_type == 'single_track' else limit
        windows.append((max(0, seg.y1-min(up_limit, above)), min(h, seg.y2+min(limit, below))))

    kept = {}
    large_cache = {}

    def read_run(start, end):
        key = (start, end)
        if key not in large_cache:
            above = max(0, segments[start].y1 - (segments[start-1].y2 if start else 0))
            below = max(0, (segments[end+1].y1 if end+1 < len(segments) else h) - segments[end].y2)
            box = (analysis.inner_x1, max(0, segments[start].y1-min(limit, above // 2)),
                   analysis.inner_x2, min(h, segments[end].y2+min(limit, below // 2)))
            alignment = segments[start].alignment if start == end and segments[start].seg_type == 'large' else 'center'
            large_cache[key] = read('large', alignment, box)
        return large_cache[key]

    for i, seg in enumerate(segments):
        y1, y2 = windows[i]
        if seg.seg_type == 'dual_track':
            group = f'{block.line_id or block.block_ref or "column"}:dual:{i}'
            split = seg.split_x if seg.split_x is not None else (analysis.inner_x1 + analysis.inner_x2)//2
            tracks = []
            for kind, alignment, box in [('dual_right', 'right', seg.right_box), ('dual_left', 'left', seg.left_box)]:
                if box is None:
                    tracks.append(None)
                    continue
                x1, _, x2, _ = box
                x1 = max(split if alignment == 'right' else analysis.inner_x1, x1-limit)
                x2 = min(analysis.inner_x2 if alignment == 'right' else split, x2+limit)
                tracks.append(read(kind, alignment, (x1, y1, x2, y2), group))
            usable = [t for t in tracks if t and _cjk_count(t['text']) and t['conf'] >= .30]
            overlap = [prob for _, prob, a, b in spans if b > seg.y1 and a < seg.y2]
            cue = any(has_dosage_or_jiazhu_cue(t['text']) for t in usable)
            max_side_len = max((_cjk_count(t['text']) for t in usable), default=0)
            overlap_mean = sum(overlap) / len(overlap) if overlap else 0.0
            overlap_min = min(overlap) if overlap else 0.0
            reject = any(t and not _clean_small_text(t['text']) for t in tracks)
            if not cue and max_side_len < 3:
                reject = True
            if overlap and overlap_mean >= .72 and overlap_min >= .50:
                reject |= not (len(usable) == 2 and min(t['conf'] for t in usable) >= .72
                               and (cue or max_side_len >= 3))
            if cursive_strict:
                reject |= not (usable and min(t['conf'] for t in usable) >= .85
                               and any(count_strict_dosage_cues(t['text']) for t in usable))
            if usable and not reject:
                kept[i] = usable
        elif seg.seg_type == 'single_track':
            item = read('single_track', seg.alignment, (max(analysis.inner_x1, seg.x1-limit), y1, min(analysis.inner_x2, seg.x2+limit), y2))
            if not item or not _clean_small_text(item['text']) or not _cjk_count(item['text']) or item['conf'] < .30:
                continue
            if cursive_strict:
                valid = seg.alignment == 'right' and count_strict_dosage_cues(item['text']) >= 1 and item['conf'] >= .85
            elif strict_single_track:
                valid = seg.alignment == 'right' and has_dosage_or_jiazhu_cue(item['text']) and item['conf'] >= .85
            else:
                valid = has_dosage_or_jiazhu_cue(item['text'])
                if not valid and seg.alignment == 'right' and item['conf'] >= .75 and _is_pure_cjk(item['text']):
                    start = max(kept, default=-1) + 1
                    if start < i:
                        previous = read_run(start, i-1)
                        valid = bool(previous and has_herb_entity(previous['text']))
            if valid:
                kept[i] = [item]

    if not kept:
        return None, gate
    i = 0
    while i < len(segments):
        if i in kept:
            entries.extend(kept[i])
            i += 1
        else:
            start = i
            while i + 1 < len(segments) and i + 1 not in kept:
                i += 1
            item = read_run(start, i)
            if item:
                entries.append(item)
            i += 1
    if not any(e['seg_type'] != 'large' and _cjk_count(e['text']) for e in entries):
        return None, gate
    for i, entry in enumerate(entries):
        entry['order'] = i
    text = ''.join(e['text'] for e in entries)
    count = _cjk_count(text)
    old_count = _cjk_count(block.text)
    ratio = count / max(1, len(text))
    old_ratio = old_count / max(1, len(block.text))
    weight = sum(len(e['text']) for e in entries)
    conf = sum(len(e['text'])*e['conf'] for e in entries) / max(1, weight)
    old_conf = float(block.conf or 0)
    positive = (count > old_count and conf >= min(.58, old_conf-.12) and (conf >= .55 or lexicon_coverage(text) > 0 or has_dosage_or_jiazhu_cue(text) or old_conf < .70)) or (old_conf < .72 and conf >= old_conf+.08 and count >= old_count)
    has_small_cue = any(has_dosage_or_jiazhu_cue(e['text']) for e in entries if e['seg_type'] != 'large')
    only_single = not any(e['seg_type'] in {'dual_right', 'dual_left'} for e in entries)
    if ((old_conf >= .70 and (conf < old_conf - .02 or (not has_small_cue and conf <= old_conf)))
            or (strict_single_track and only_single and conf < old_conf + .05)
            or (cursive_strict and conf < old_conf + .08)) or count < 1 or ratio < min(.70, max(.50, old_ratio)) or conf < .45 or not positive:
        return None, gate
    trace = {'line_id': block.line_id, 'before': block.text, 'after': text, 'old_conf': old_conf, 'new_conf': round(conf, 4), 'gate_reasons': list(gate.reasons), 'segment_types': [e['seg_type'] for e in entries]}
    return {'text': text, 'conf': round(conf, 4), 'segments': entries, 'trace': trace}, gate



@dataclass
class PagePrior:
    pblocks: List[Block]
    page_mean_conf: float
    page_median_conf: float
    high_conf_ratio: float
    width_cv: float
    herb_count: int
    strict_dosage_count: int
    low_quality_cursive: bool
    has_text_prior: bool


def evaluate_page_prior(blocks: List[Block]) -> PagePrior:
    pblocks = [b for b in blocks if b.kind == 'paragraph'
               and b.bbox[2] - b.bbox[0] >= 16
               and b.bbox[3] - b.bbox[1] >= 1.8 * (b.bbox[2] - b.bbox[0])]
    widths = [b.bbox[2] - b.bbox[0] for b in pblocks]
    confs = [float(b.conf) for b in pblocks if float(b.conf or 0) > 0]
    mean = float(np.mean(confs)) if confs else 0.0
    median = float(np.median(confs)) if confs else 0.0
    high = sum(c >= .78 for c in confs) / len(confs) if confs else 0.0
    cv = float(np.std(widths) / np.mean(widths)) if widths else 0.0
    herbs = count_distinct_herbs(' '.join(b.text or '' for b in pblocks))
    dosage = sum(count_strict_dosage_cues(b.text or '') for b in pblocks)
    regular = dosage >= 3 and cv < .30 and high >= .45
    low = len(confs) >= 6 and not regular and ((mean < .65 and high < .45) or
          (min(mean, median) < .68 and high < .45 and cv >= .30))
    return PagePrior(pblocks, mean, median, high, cv, herbs, dosage, low,
                     herbs >= 3 or dosage >= 2)

def apply_column_seg_jiazhu(image: np.ndarray, blocks: List[Block], *, scale: float = 2.0, thin_mode: str = 'none', rec_fn=None, ctc_rec_fn=None, seg_fn=None) -> Tuple[List[Block], List[Dict[str, Any]]]:
    traces = []
    prior = evaluate_page_prior(blocks)
    if prior.low_quality_cursive and prior.strict_dosage_count == 0:
        return blocks, traces
    widths = [b.bbox[2] - b.bbox[0] for b in prior.pblocks]
    med_w = float(np.median(widths)) if widths else None
    candidates = []
    for block in blocks:
        bw = block.bbox[2] - block.bbox[0]
        if med_w is not None and (bw > 2.05 * med_w or (bw > 1.85 * med_w and _cjk_count(block.text) >= 24)):
            continue
        candidates.append(block)

    strict = len(prior.pblocks) >= 4 and not prior.has_text_prior
    cursive = prior.low_quality_cursive

    def run(block, strict_mode):
        kwargs = dict(scale=scale, thin_mode=thin_mode, rec_fn=rec_fn, ctc_rec_fn=ctc_rec_fn, seg_fn=seg_fn)
        if strict_mode:
            kwargs['strict_single_track'] = True
        if cursive:
            kwargs['cursive_strict'] = True
        result, _ = reassemble_column_block(image, block, **kwargs)
        if result is not None:
            block.text = block.quote = result['text']
            block.conf = result['conf']
            block.segments = result['segments']
            block.source_hash = source_hash_of(block.page_id, block.line_id, block.bbox, block.text)
            traces.append(result['trace'])
        return result

    unchanged = []
    confirmed_dual = False
    for block in candidates:
        result = run(block, strict)
        if result is None:
            unchanged.append(block)
        elif any(e['seg_type'] in {'dual_right', 'dual_left'} and e['conf'] >= .72 for e in result['segments']):
            confirmed_dual = True
    if strict and confirmed_dual:
        for block in unchanged:
            run(block, False)
    return blocks, traces
