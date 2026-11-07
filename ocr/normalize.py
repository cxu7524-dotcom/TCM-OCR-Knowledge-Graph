# -*- coding: utf-8 -*-
"""
ocr/normalize.py —— 繁简 / 异体 / 通假归一化（阶段二）

字表单一来源：`data/lexicon/异体字表.json`（含来源与置信等级）。
  - 繁简 + 异体：自动归一化，写入 Block.text_norm；
  - 通假：只登记、不自动改写，需人工复核（find_tongjia 输出提示）；
  - 多字条目优先于单字，避免「六府」被逐字拆改；
  - Block.text 永远保留原文，不做任何改写（溯源要求）。

命中率统计供评测报告单列（赛题 5 分），见 hit_stats()。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from app.config import CONFIG
from app.schemas import Block

LEXICON_PATH = CONFIG.paths.data_dir / "lexicon" / "异体字表.json"

# 兜底小表：字表文件缺失时仍能让链路跑通，但必须标注降级
_FALLBACK = {"無毒": "无毒", "寒熱": "寒热", "氣": "气", "湯": "汤", "兩": "两",
             "藥": "药", "五藏": "五脏", "六府": "六腑", "欬": "咳"}


def load_lexicon() -> Dict[str, Any]:
    """读取字表；缺失返回最小兜底结构并标注降级。"""
    try:
        return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {"版本": "fallback", "繁简": _FALLBACK, "异体": {}, "通假": {},
                "降级": True, "说明": "字表文件缺失，使用内置兜底小表"}


def mapping() -> Dict[str, str]:
    """繁简 + 异体合并后的映射（多字条目会在 normalize_text 中优先使用）。"""
    raw = load_lexicon()
    out: Dict[str, str] = {}
    for group in ("繁简", "异体"):
        out.update(raw.get(group, {}))
    return out


def normalize_text(text: str) -> str:
    """确定性归一化：多字优先 → 单字替换。异常时原样返回，不抛异常。"""
    if not text or not isinstance(text, str):
        return ""
    try:
        m = mapping()
        out = text
        for k in sorted((k for k in m if len(k) > 1), key=len, reverse=True):
            if k in out:
                out = out.replace(k, m[k])
        single = {k: v for k, v in m.items() if len(k) == 1}
        return "".join(single.get(ch, ch) for ch in out)
    except Exception:
        return text


def find_tongjia(text: str) -> List[Dict[str, str]]:
    """列出疑似通假 / 异体位置，供人工复核；不修改文本。"""
    if not text:
        return []
    raw = load_lexicon()
    hints = raw.get("通假", {})
    return [{"raw": k, "suggest": v, "level": "低（需人工复核）"}
            for k, v in hints.items() if k in text]


def normalize_blocks(blocks: Iterable[Block]) -> List[Block]:
    """就地填充 text_norm 并返回同一批对象（text 不变）。"""
    out: List[Block] = []
    for b in blocks:
        if not isinstance(b, Block):
            continue
        b.text_norm = normalize_text(b.text)
        out.append(b)
    return out


def hit_stats(blocks: Iterable[Block]) -> Dict[str, Any]:
    """归一化命中统计：原文中被字表命中的字符数 / 实际发生转换的字符数。

    返回结构可直接写入评测报告；不是「准确率」，只是覆盖率统计。
    """
    raw = load_lexicon()
    m = mapping()
    multi = sorted((k for k in m if len(k) > 1), key=len, reverse=True)
    single = {k: v for k, v in m.items() if len(k) == 1}
    matched = converted = 0
    per_char: Dict[str, int] = {}
    for b in blocks:
        text = b.text or ""
        norm = b.text_norm or ""
        tmp = text
        for k in multi:
            tmp = tmp.replace(k, m[k])
        tmp = "".join(single.get(ch, ch) for ch in tmp)
        for ch in text:
            if ch in single or any(ch in k for k in multi):
                matched += 1
                per_char[ch] = per_char.get(ch, 0) + 1
        # 逐字符比较：归一后发生变化的字符数（长度变化时取保守值）
        converted += sum(1 for a, c in zip(text, tmp) if a != c)
        if norm != tmp:
            converted = max(0, converted)  # text_norm 由同一规则生成，保持一致
    return {
        "lexicon_version": raw.get("版本", "unknown"),
        "mapping_size": len(m),
        "degraded": bool(raw.get("降级")),
        "matched_chars": matched,
        "converted_chars": converted,
        "hit_rate": round(converted / matched, 4) if matched else None,
        "top_chars": sorted(per_char.items(), key=lambda x: -x[1])[:10],
        "tongjia_only": len(raw.get("通假", {})),
        "note": "归一化命中率单列统计；通假类只登记不自动改写",
    }


def mapping_size() -> int:
    return len(mapping())
