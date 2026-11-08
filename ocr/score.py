# -*- coding: utf-8 -*-
"""
ocr/score.py —— 两路 OCR 候选的**无监督**质量分（阶段五 P2）

背景：阶段四用 `Σ(字数×置信度)` 择优，天然偏向「字多的整页」——逐列一旦漏一列就输，
34 页验证集里逐列全输（P1 等于没生效）。本模块改用四项可解释、不读 GT 的信号：

  1. 合法字符率（权重最高）：识别字符落在「当前识别模型字符表 ∪ 异体字表」的比例。
     木刻页误识别的典型产物是符号 / 拉丁字母 / 生造字，这条最敏感。
  2. 平均置信度：模型自身输出，反映切图质量（列被切坏时整列 conf 会塌）。
  3. 列数合理性：古籍一页 8–30 列；逐列路径切出 2 列或 60 列都说明切列失败。
  4. 重复率惩罚：同一字符连续 ≥4 次是「乱识别」的典型特征（如「一一一一」）。

全部信号都不读 GT、不看 CER，可复现；明细写入 OCRResult.notes，便于人工复核。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Set

from app.config import CONFIG

_LEXICON_CACHE: Optional[Set[str]] = None


def _lexicon_chars() -> Set[str]:
    """异体字表里的全部字（繁简两栏都算合法字形）。"""
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    chars: Set[str] = set()
    p = CONFIG.paths.data_dir / "lexicon" / "异体字表.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        for group in ("繁简", "异体", "通假"):
            for k, v in (data.get(group) or {}).items():
                chars.update(str(k))
                chars.update(str(v))
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        pass
    _LEXICON_CACHE = chars
    return chars


def legal_charset() -> Set[str]:
    """合法字符集：识别模型字符表（字典内嵌 ONNX metadata）+ 异体字表。"""
    chars: Set[str] = set(_lexicon_chars())
    try:
        from .recognize import model_charset
        chars |= model_charset()
    except Exception:
        pass
    return chars


def legal_char_rate(blocks: Iterable[Any], charset: Optional[Set[str]] = None) -> Dict[str, Any]:
    """识别字符中落在合法字符集里的比例。"""
    cs = charset if charset is not None else legal_charset()
    text = "".join((getattr(b, "text", "") or "") for b in blocks)
    text = "".join(ch for ch in text if not ch.isspace())
    if not text or not cs:
        return {"rate": None, "total": len(text), "illegal": 0}
    bad = sum(1 for ch in text if ch not in cs)
    return {"rate": round(1.0 - bad / len(text), 4), "total": len(text), "illegal": bad}


def mean_conf(blocks: Iterable[Any]) -> float:
    confs = [float(getattr(b, "conf", 0) or 0) for b in blocks]
    return round(sum(confs) / len(confs), 4) if confs else 0.0


def column_score(n_cols: int, lo: int = 8, hi: int = 30) -> float:
    """列数合理性：落在 [lo, hi] 得满分，偏离按距离线性扣分，最低 0。"""
    if n_cols <= 0:
        return 0.0
    if lo <= n_cols <= hi:
        return 1.0
    span = max(1, lo - 1) if n_cols < lo else max(1, hi)
    return max(0.0, 1.0 - abs(n_cols - (lo if n_cols < lo else hi)) / float(span))


def repeat_penalty(text: str) -> Dict[str, Any]:
    """同一字符连续 ≥4 次扣分：惩罚值 = 违规连续段数 / 段落数（上限 1）。"""
    text = "".join(ch for ch in (text or "") if not ch.isspace())
    if not text:
        return {"penalty": 0.0, "runs": 0}
    runs = 0
    cur, n = text[0], 1
    for ch in text[1:]:
        if ch == cur:
            n += 1
        else:
            if n >= 4:
                runs += 1
            cur, n = ch, 1
    if n >= 4:
        runs += 1
    return {"penalty": round(min(1.0, runs / 3.0), 4), "runs": runs}


WEIGHTS = {"legal": 0.50, "conf": 0.20, "cols": 0.15, "repeat": 0.15}


def composite(blocks, n_cols: int, charset: Optional[Set[str]] = None) -> Dict[str, Any]:
    """复合无监督分（0–1）与明细。分数越高越可信。"""
    blocks = list(blocks)
    legal = legal_char_rate(blocks, charset)
    conf = mean_conf(blocks)
    cols = column_score(n_cols)
    rep = repeat_penalty("".join((getattr(b, "text", "") or "") for b in blocks))
    legal_v = legal["rate"] if legal["rate"] is not None else 0.0
    score = (WEIGHTS["legal"] * legal_v + WEIGHTS["conf"] * conf
             + WEIGHTS["cols"] * cols + WEIGHTS["repeat"] * (1.0 - rep["penalty"]))
    return {
        "score": round(score, 4),
        "legal": legal_v, "illegal": legal["illegal"], "chars": legal["total"],
        "conf": conf, "cols": n_cols, "col_score": cols,
        "repeat_runs": rep["runs"], "repeat_penalty": rep["penalty"],
    }


def fmt(detail: Dict[str, Any]) -> str:
    """一行可读明细，写进 notes。"""
    return (f"score={detail['score']:.4f}（合法字符率 {detail['legal']:.3f}"
            f"[非法 {detail['illegal']}/{detail['chars']}]，"
            f"平均置信 {detail['conf']:.3f}，列数 {detail['cols']}"
            f"[合理性 {detail['col_score']:.2f}]，连续重复段 {detail['repeat_runs']}）")
