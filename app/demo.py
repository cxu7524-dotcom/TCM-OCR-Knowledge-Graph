# -*- coding: utf-8 -*-
"""
app/demo.py —— 示例数据统一装载器（演示态的唯一入口）

所有示例数据桩输出都从这里取，保证：
  1. 界面、接口、报告里的演示标注文案一致；
  2. 示例数据本身带完整溯源字段，不会绕过 app/schemas.py 的契约；
  3. 即使 data/demo/ 下的文件被误删，也能返回内置最小示例，服务不崩、不抛异常。

严格约束：示例数据**不得进入评测计分**；eval/metrics.py 会显式排除 is_demo=True 的记录。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import CONFIG
from .schemas import Block, OCRResult, source_hash_of

DEMO_PAGE_FILE = CONFIG.paths.demo_dir / "demo_page.json"
DEMO_QUESTION_FILE = CONFIG.paths.demo_dir / "demo_questions.json"

DEMO_NOTE = "示例数据 · 非命题单位样本"
DEMO_LONG_NOTE = (
    "本页为合成示意数据，非真实古籍扫描页；识别结果、实体与三元组只用于打通端到端链路，"
    "不得作为评测或诊疗依据。真实古籍页请用「上传图片」走本地 OCR 推理。"
)

# 兜底最小示例：文件缺失时也不会让链路断掉
_FALLBACK_BLOCKS = [
    {"line_id": "L1", "col_id": 0, "text": "甘草 味甘平 无毒",
     "kind": "paragraph", "bbox": [500, 80, 560, 520]},
    {"line_id": "L2", "col_id": 1, "text": "主治五藏六府寒热邪气 坚筋骨 长肌肉",
     "kind": "paragraph", "bbox": [410, 80, 470, 760]},
]


def _read_json(path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def load_demo_ocr(page_id: str = "DEMO-001", image_path: str = "") -> OCRResult:
    """加载示例 OCR 页；可把示例块重绑到指定 page_id（上传演示时用）。

    返回结构完整的 OCRResult，且 engine="stub" / model_loaded=False / is_demo=True。
    """
    raw = _read_json(DEMO_PAGE_FILE)
    blocks_raw = raw.get("blocks") or _FALLBACK_BLOCKS
    blocks: List[Block] = []
    for i, item in enumerate(blocks_raw, start=1):
        d = dict(item)
        d["page_id"] = page_id
        d.setdefault("line_id", f"L{i}")
        d.setdefault("bbox", [0, 0, 0, 0])
        d["block_ref"] = f"{page_id}:{d['line_id']}"
        d["source_hash"] = source_hash_of(page_id, d["line_id"], d.get("text", ""))
        b = Block.from_dict(d)
        b.quote = b.text                       # 溯源：文本块自身即原文
        blocks.append(b)

    ocr = OCRResult(
        page_id=page_id,
        image_path=image_path or "data/demo/demo_page.json",
        width=int(raw.get("width", 720)),
        height=int(raw.get("height", 1000)),
        blocks=blocks,
        engine="stub",
        model_loaded=False,
        is_demo=True,
        source_hash=source_hash_of("demo", page_id),
        notes=[DEMO_NOTE, DEMO_LONG_NOTE],
        created_at="",
    )
    return ocr


def load_demo_questions() -> List[Dict[str, Any]]:
    """示例问题清单：含「应能回答」与「应拒答」两类，供前端一键演示。"""
    data = _read_json(DEMO_QUESTION_FILE)
    items = data.get("questions")
    if isinstance(items, list) and items:
        return items
    return [
        {"q": "甘草的性味是什么？", "expect": "answer"},
        {"q": "桂枝汤由哪些药材组成？", "expect": "answer"},
        {"q": "甘草有什么禁忌？", "expect": "answer"},
        {"q": "这个方子能治好我的失眠吗？该吃多少？", "expect": "refuse"},
        {"q": "本品对晚期肝癌的治愈率是多少？", "expect": "refuse"},
    ]


def demo_flags() -> Dict[str, Any]:
    """给接口与前端的演示态标记。"""
    return {
        "is_demo": True,
        "badge": CONFIG.demo_badge,
        "note": DEMO_LONG_NOTE,
        "model_loaded": False,
        "disclaimer": CONFIG.disclaimer,
    }
