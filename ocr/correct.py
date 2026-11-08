# -*- coding: utf-8 -*-
"""
ocr/correct.py —— 字形纠错层（可解释、可回滚）

设计原则（对齐阶段三提示词 P1.3）：
  1. 只做「上下文可判定」的纠错：固定书名、页眉正则、数字后的剂量单位、药名近形字、术语表整词；
  2. **不做全局繁简回写**（「温」→「溫」这类不纠），避免在真实混合简繁文本上引入错误；
  3. 不补漏字、不猜整句 —— 漏识别交给检测/切分改进；
  4. 每条纠错写入 trace（before / after / rule / confidence），可整体关闭（TCM_OCR_CORRECT=0）；
  5. 纠错后同步 `quote = text`，保证证据仍能回「系统输出的原文」自洽命中。

规则表：`data/lexicon/ocr_confusions.json`
诊断依据：`eval/ocr_error_analysis.py`（只用于确认规则方向，不逐字硬编码测试页）

TODO（阶段四）：引入字形相似度模型（结构码/部件）替代手工编辑距离，并把纠错纳入评测单列。
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Tuple

from app.config import CONFIG
from app.schemas import Block

CONFUSIONS_PATH = CONFIG.paths.data_dir / "lexicon" / "ocr_confusions.json"
_DOSE_DIGITS = "0-9一二三四五六七八九十百"


def load_rules() -> Dict[str, Any]:
    try:
        return json.loads(CONFUSIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _apply_fixed(text: str, rules: Dict[str, str], trace: List[Dict[str, Any]]) -> str:
    for src, dst in (rules or {}).items():
        if src and src in text:
            text = text.replace(src, dst)
            trace.append({"rule": "固定词", "before": src, "after": dst, "confidence": "高"})
    return text


def _apply_header(text: str, rules: List[Dict[str, Any]], trace: List[Dict[str, Any]]) -> str:
    for r in rules or []:
        pat, rep = r.get("pattern"), r.get("replace")
        if not pat or rep is None:
            continue
        new, n = re.subn(pat, rep, text)
        if n:
            trace.append({"rule": "页眉", "before": text, "after": new,
                          "pattern": pat, "count": n, "confidence": "高"})
            text = new
    return text


def _apply_dose(text: str, units: Dict[str, str], trace: List[Dict[str, Any]]) -> str:
    """数字 + 误识单位 → 正字单位（只在数字后生效，避免误伤正文里的「雨」）。"""
    for bad, good in (units or {}).items():
        pat = re.compile(f"([{_DOSE_DIGITS}])\\s*{re.escape(bad)}")
        new, n = pat.subn(lambda m: m.group(1) + good, text)
        if n:
            trace.append({"rule": "剂量单位", "before": f"{bad}→{good}", "count": n,
                          "confidence": "高"})
            text = new
    return text


def _apply_words(text: str, words: Dict[str, str], trace: List[Dict[str, Any]]) -> str:
    for bad, good in (words or {}).items():
        if bad and bad in text:
            text = text.replace(bad, good)
            trace.append({"rule": "字词", "before": bad, "after": good, "confidence": "中"})
    return text


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _at_word_boundary(text: str, i: int, L: int) -> bool:
    """窗口前后必须是边界（串首尾 / 空格 / 标点），避免在词中间替换造成误纠。"""
    left_ok = i == 0 or not _is_cjk(text[i - 1])
    right_ok = i + L >= len(text) or not _is_cjk(text[i + L])
    return left_ok and right_ok


def _apply_herbs(text: str, variants: List[str], trace: List[Dict[str, Any]]) -> str:
    """药名近形字：**独立词边界** + 首字相同 + 编辑距离 1 + 候选在繁体药名表内才纠。

    边界约束是必须的：早期版本允许词内替换，会把「味甘平」误纠成「味甘草」，
    属于典型的过度纠错（已在 P1 修复并记录）。
    """
    if not text or not variants:
        return text
    i = 0
    while i < len(text):
        replaced = False
        for L in (3, 2):
            win = text[i:i + L]
            if len(win) < L or win in variants or not _at_word_boundary(text, i, L):
                continue
            for cand in variants:
                if len(cand) != L or cand[0] != win[0]:
                    continue
                if difflib.SequenceMatcher(None, win, cand).ratio() >= 0.5 and \
                        _edit_distance(win, cand) == 1:
                    text = text[:i] + cand + text[i + L:]
                    trace.append({"rule": "药名", "before": win, "after": cand,
                                  "confidence": "中"})
                    i += L
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            i += 1
    return text


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def correct_text(text: str, rules: Dict[str, Any] | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    """对一段文本做确定性纠错，返回（纠错后文本, trace）。不抛异常。"""
    if not text:
        return text, []
    rules = rules if rules is not None else load_rules()
    trace: List[Dict[str, Any]] = []
    try:
        out = _apply_fixed(text, rules.get("固定词", {}), trace)
        out = _apply_header(out, rules.get("页眉规则", []), trace)
        out = _apply_dose(out, rules.get("剂量单位", {}), trace)
        out = _apply_words(out, rules.get("字词纠正", {}), trace)
        out = _apply_herbs(out, rules.get("药名变体", []), trace)
        return out, trace
    except Exception:
        return text, []


def correct_blocks(blocks: List[Block]) -> Tuple[List[Block], List[Dict[str, Any]]]:
    """对一批 Block 纠错：同步 text / quote，保留 source_hash（代表来源），记录 trace。

    返回（同一批对象, 全部纠错记录）。关闭开关时原样返回。
    """
    if not CONFIG.ocr.correct_enabled:
        return blocks, []
    rules = load_rules()
    all_trace: List[Dict[str, Any]] = []
    for b in blocks:
        if not isinstance(b, Block):
            continue
        new_text, trace = correct_text(b.text, rules)
        if trace and new_text != b.text:
            all_trace.append({
                "page_id": b.page_id, "line_id": b.line_id,
                "before": b.text, "after": new_text,
                "rules": trace,
                "note": "纠错只改系统输出，不改 GT；可用 TCM_OCR_CORRECT=0 关闭",
            })
            b.text = new_text
            b.quote = new_text          # 证据回「系统输出原文」自洽命中
    return blocks, all_trace


def write_trace(records: List[Dict[str, Any]]) -> str:
    """把纠错记录合并写盘：同 page_id 覆盖、其他页保留（可反复跑、不膨胀）。"""
    path = CONFIG.paths.reports_dir / "ocr_corrections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = {str(r.get("page_id", "")) for r in records}
    kept: List[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("page_id", "")) not in pages:
                kept.append(json.dumps(obj, ensure_ascii=False))
    kept.extend(json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return str(path)


def describe() -> Dict[str, Any]:
    rules = load_rules()
    return {
        "enabled": CONFIG.ocr.correct_enabled,
        "version": rules.get("版本", "unknown"),
        "rules": {
            "固定词": len(rules.get("固定词", {})),
            "页眉规则": len(rules.get("页眉规则", [])),
            "剂量单位": len(rules.get("剂量单位", {})),
            "药名变体": len(rules.get("药名变体", [])),
            "字词纠正": len(rules.get("字词纠正", {})),
        },
        "policy": "只做上下文可判定的纠错；不做全局繁简回写；不补漏字",
    }
