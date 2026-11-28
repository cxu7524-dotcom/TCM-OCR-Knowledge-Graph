# -*- coding: utf-8 -*-
"""Offline, candidate-constrained CTC reranking for classical TCM texts."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ocr.normalize import mapping

_ROOT = Path(__file__).resolve().parents[1]
_LOG = logging.getLogger(__name__)
_GROUPS = ("为爲為", "热熱", "证證", "导導", "执執", "盖蓋", "众衆眾",
           "里裏裡", "传傳", "杂雜", "气氣", "谵譫", "误誤", "变變",
           "发發髮", "与與", "无無", "经經", "药藥薬", "实實", "医醫",
           "来來", "烦煩", "即卽", "缓緩", "现現", "况況", "疡瘍",
           "疮瘡", "瘿癭", "痹痺", "润潤", "补補", "复復複", "温溫",
           "时時", "门門", "脏臟", "脉脈", "术朮", "历歷曆", "归歸",
           "独獨", "当當", "阳陽", "阴陰", "虚虛", "伤傷", "湿濕",
           "风風", "饮飲", "呕嘔", "泻瀉", "肿腫", "络絡", "卫衛",
           "营營", "肾腎", "胆膽", "肠腸", "脑腦", "节節", "剂劑",
           "汤湯", "症癥", "体體", "头頭", "岁歲", "参參", "恶惡",
           "惊驚", "数數", "断斷", "旧舊", "显顯", "晕暈", "会會",
           "条條", "极極", "枣棗", "树樹", "样樣", "楼樓", "乐樂",
           "汇匯", "汉漢", "冲沖", "泽澤", "洁潔", "浅淺", "浆漿",
           "浊濁", "济濟", "浓濃", "涂塗", "涨漲", "涩澀", "渗滲",
           "溃潰", "滞滯", "满滿", "潜潛", "灭滅", "灵靈", "灾災",
           "炉爐", "烂爛", "烧燒", "疗療", "瘫癱", "癫癲", "疯瘋",
           "痉痙", "痒癢", "疟瘧", "痈癰", "癣癬", "痨癆", "痫癇",
           "痪瘓")


@lru_cache(maxsize=1)
def _canonical_map():
    result = {k: v for k, v in mapping().items() if len(k) == len(v) == 1}
    for group in _GROUPS:
        result.update({ch: group[0] for ch in group})
    return result


def _canon(ch: str) -> str:
    table = _canonical_map()
    seen = set()
    while ch in table and ch not in seen and table[ch] != ch:
        seen.add(ch)
        ch = table[ch]
    return ch


def _han(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def _clean(text: str) -> str:
    return "".join(_canon(ch) for ch in text if _han(ch))


class ClassicalTCMLM:
    def __init__(self, corpus_path=None, lexicon_path=None, exclude_books=("温疫",)):
        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()
        self.lexicon_terms = set()
        corpus_path = Path(corpus_path) if corpus_path is not None else _ROOT / "data/annotation/official/guji_raw.jsonl"
        lexicon_path = Path(lexicon_path) if lexicon_path is not None else _ROOT / "data/lexicon/中医术语表.json"
        # The held-out book is always excluded, even when additional exclusions are supplied.
        exclusions = ("温疫", *exclude_books)
        try:
            with corpus_path.open(encoding="utf-8-sig") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict) or any(term in str(record.get(field, "")) for term in exclusions for field in ("book", "name")):
                            continue
                        text = record.get("ocr", "")
                        if isinstance(text, str):
                            self._add(_clean(text), 1)
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass
        try:
            data = json.loads(lexicon_path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                for values in data.values():
                    if isinstance(values, list):
                        for value in values:
                            if isinstance(value, str):
                                term = _clean(value)
                                if len(term) >= 2 and term not in self.lexicon_terms:
                                    self.lexicon_terms.add(term)
                                    self._add(term, 12)
        except (OSError, ValueError, TypeError):
            pass

    def _add(self, text: str, weight: int):
        for ch in text:
            self.unigram[ch] += weight
        for i in range(len(text) - 1):
            self.bigram[tuple(text[i:i + 2])] += weight
        for i in range(len(text) - 2):
            self.trigram[tuple(text[i:i + 3])] += weight

    @classmethod
    @lru_cache(maxsize=1)
    def from_default_corpus(cls):
        return cls()

    def context_score(self, left: str, cand: str, right: str) -> Tuple[float, int]:
        left, cand, right = map(_canon, (left, cand, right))
        vocab = max(1000, len(self.unigram))
        score = 0.0
        if left:
            score += math.log((self.bigram[left, cand] + 0.5) / (self.unigram[left] + 0.5 * vocab))
        if right:
            score += math.log((self.bigram[cand, right] + 0.5) / (self.unigram[cand] + 0.5 * vocab))
        lex_hit = int(any(term in self.lexicon_terms for term in
                          (left + cand if left else "", cand + right if right else "", left + cand + right if left and right else "")))
        score += 2.5 * lex_hit
        if left and right and self.trigram[left, cand, right] >= 1:
            score += 1.5
        hits = (self.bigram[left, cand] if left else 0) + (self.bigram[cand, right] if right else 0) + 10 * lex_hit
        return score, hits


def rerank_sequence_from_probs(probs: np.ndarray, chars: Sequence[str], lm: ClassicalTCMLM, *,
                               max_top1_prob: float = 0.82, min_cand_prob: float = 0.03,
                               min_prob_ratio: float = 0.08, alpha: float = 0.65,
                               min_gain: float = 1.8, min_bg_hits: int = 2,
                               min_bg_ratio: float = 3.0) -> Tuple[str, List[Dict[str, Any]]]:
    indices = np.asarray(probs).argmax(axis=1)
    align = [(int(idx), t) for t, idx in enumerate(indices)
             if idx != 0 and (t == 0 or idx != indices[t - 1])]
    out = [chars[idx] for idx, _ in align]
    traces = []
    for i, (idx, t) in enumerate(align):
        ch1, p1 = chars[idx], float(probs[t, idx])
        if not _han(ch1) or p1 >= max_top1_prob:
            continue
        left = out[i - 1] if i > 0 and _han(out[i - 1]) else ""
        right = out[i + 1] if i + 1 < len(out) and _han(out[i + 1]) else ""
        if not left and not right:
            continue
        lm1, bg1 = lm.context_score(left, ch1, right)
        s1 = math.log(max(p1, 1e-9)) + alpha * lm1
        best = None
        candidates = sorted((j for j in range(1, min(len(chars), probs.shape[1])) if _han(chars[j])),
                            key=lambda j: probs[t, j], reverse=True)[:5]
        for j in candidates:
            ch, pc = chars[j], float(probs[t, j])
            if pc < min_cand_prob or pc / max(p1, 1e-9) < min_prob_ratio or _canon(ch) == _canon(ch1):
                continue
            lmc, bgc = lm.context_score(left, ch, right)
            score = math.log(max(pc, 1e-9)) + alpha * lmc
            if score - s1 >= min_gain and bgc >= min_bg_hits and bgc >= min_bg_ratio * max(1, bg1):
                if best is None or score > best[0]:
                    best = (score, ch, pc, bgc)
        if best is not None:
            score, ch, pc, bgc = best
            out[i] = ch
            traces.append({"pos": i, "before": ch1, "after": ch, "top1_p": round(p1, 4),
                           "cand_p": round(pc, 4), "bg_before": bg1, "bg_after": bgc,
                           "gain": round(score - s1, 3)})
    return "".join(out), traces


@contextmanager
def ctc_rerank_context(engine, lm=None):
    rec = getattr(engine, "text_rec", None) or getattr(engine, "text_recognizer", None)
    if rec is None:
        raise AttributeError("RapidOCR recognizer unavailable")
    orig = rec.postprocess_op
    if lm is None:
        lm = ClassicalTCMLM.from_default_corpus()

    def wrapped(preds, *args, **kwargs):
        result = orig(preds, *args, **kwargs)
        try:
            updated = []
            for k, row in enumerate(result):
                text, _ = rerank_sequence_from_probs(preds[k], orig.character, lm)
                updated.append((text, *row[1:]) if isinstance(row, tuple) else [text, *row[1:]])
            return tuple(updated) if isinstance(result, tuple) else updated
        except Exception:
            _LOG.exception("CTC reranking failed; retaining original recognition")
            return result

    rec.postprocess_op = wrapped
    try:
        yield
    finally:
        rec.postprocess_op = orig
