# -*- coding: utf-8 -*-
"""
ocr/normalize.py —— 繁简 / 异体字 / 通假字归一化

职责：把 OCR 文本转成 `Block.text_norm`，供检索与实体匹配使用；
      `Block.text` 永远保留原文，不做任何改写（溯源要求）。

本阶段只给一张**演示用小表**，不是全量字表；命中不了的原样返回，绝不抛异常。

TODO（下一阶段）：
  - 引入古籍异体字表 / 通假字表（按年代、版本分层，需标注来源与置信度）；
  - 归一化必须记录映射日志到 trace，避免「原文 → 归一文本」不可解释；
  - 区分「异体字直接合并」与「通假字需人工确认」两类，后者不得自动改写。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from app.schemas import Block

# 演示用小表：繁→简 / 常见异体 / 中医古籍常见通假（仅示例，非全量）
SIMPLE_MAP: Dict[str, str] = {
    "草藥": "草药", "藥": "药", "湯": "汤", "劑": "剂", "兩": "两",
    "無毒": "无毒", "無": "无", "歸經": "归经", "歸": "归", "經": "经",
    "氣": "气", "府": "腑", "藏": "脏", "府寒": "腑寒",
    "輕身": "轻身", "輕": "轻", "堅": "坚", "筋骨": "筋骨",
    "脈": "脉", "證": "证", "癥": "症", "溼": "湿", "溼氣": "湿气",
    "薑": "姜", "棗": "枣", "芍藥": "芍药", "桂枝": "桂枝",
    "海藻": "海藻", "菘菜": "菘菜", "茯苓": "茯苓", "當歸": "当归",
    "補": "补", "血": "血", "溫": "温", "熱": "热", "寒": "寒",
}

# 通假 / 异体提示词：只登记、不自动改写（避免把原文改错）
TONGJIA_HINTS: Dict[str, str] = {
    "五藏": "五脏", "六府": "六腑", "欬逆": "咳逆", "利小便": "利小便",
}

# 多字符映射要优先于单字符，避免被拆开误替换
_MULTI = sorted((k for k in SIMPLE_MAP if len(k) > 1), key=len, reverse=True)
_SINGLE = {k: v for k, v in SIMPLE_MAP.items() if len(k) == 1}


def normalize_text(text: str) -> str:
    """繁简 / 异体归一化。输入 None 或任何异常都返回空串，不抛异常。"""
    if not text or not isinstance(text, str):
        return ""
    out = text
    try:
        for k in _MULTI:
            if k in out:
                out = out.replace(k, SIMPLE_MAP[k])
        out = "".join(_SINGLE.get(ch, ch) for ch in out)
        out = re.sub(r"\s+", " ", out).strip()
    except Exception:
        return text
    return out


def find_tongjia(text: str) -> List[Dict[str, str]]:
    """列出疑似通假 / 异体位置，供人工复核；不修改文本。"""
    if not text:
        return []
    return [{"raw": k, "suggest": v} for k, v in TONGJIA_HINTS.items() if k in text]


def normalize_blocks(blocks: Iterable[Block]) -> List[Block]:
    """就地填充 text_norm 并返回同一批对象（原文 text 不变）。"""
    result: List[Block] = []
    for b in blocks:
        if not isinstance(b, Block):
            continue
        b.text_norm = normalize_text(b.text)
        result.append(b)
    return result


def mapping_size() -> int:
    """当前小表规模，供 /api/health 展示；不得当评测结论。"""
    return len(SIMPLE_MAP)
