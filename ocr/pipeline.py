# -*- coding: utf-8 -*-
"""
ocr/pipeline.py —— ① 图像 → 带 bbox 的 OCRResult（阶段二：真实链路）

链路：图像 → 本地 ONNX OCR 推理 → 低置信过滤 + 人工复核队列 → 版面列聚类 / 阅读顺序
      → 繁简异体归一化 → 方剂表 bbox 行列还原 → OCRResult

与阶段一的区别：
  - 宽高取自真实图像，不再硬编码 720×1000；
  - model_loaded / is_demo 由识别适配层如实给出，不静默伪造；
  - 真实推理失败时降级为示例数据，并在 notes / 界面标注。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from app.config import CONFIG
from app.demo import load_demo_ocr
from app.schemas import OCRResult, source_hash_of

from . import layout, normalize, table
from .recognize import _image_size, recognize


def new_page_id(prefix: str = "P") -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{stamp}-{source_hash_of(prefix, stamp, datetime.now().microsecond)[:6]}"


def _write_review_queue(page_id: str, image_path: str,
                        low_blocks: List) -> str:
    """低置信块写入人工复核队列（幂等追加，按 page_id 去重覆盖）。"""
    path = CONFIG.paths.reports_dir / "ocr_review_queue.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("page_id") != page_id:
                    old.append(line)
            except json.JSONDecodeError:
                continue
    for b in low_blocks:
        old.append(json.dumps({
            "page_id": page_id, "image": image_path, "line_id": b.line_id,
            "bbox": b.bbox, "quote": b.quote, "conf": b.conf,
            "reason": f"置信度低于 TCM_OCR_MIN_CONF={CONFIG.ocr.min_conf}",
            "is_demo": b.is_demo if hasattr(b, "is_demo") else False,
        }, ensure_ascii=False))
    path.write_text("\n".join(old) + ("\n" if old else ""), encoding="utf-8")
    return str(path)


def _postprocess(blocks, page_id: str, notes: List[str], is_demo: bool):
    """归一化 / 版面 / 表格三件套。"""
    # 顺序要点：先做版面合并（会重建 Block），再归一化，避免 text_norm 被合并覆盖
    blocks, lay = layout.analyze_layout(blocks, page_id=page_id, is_demo=is_demo)
    notes.append(f"版面：{lay.direction}，{len(lay.cols)} 列；{lay.note}")

    if CONFIG.ocr.normalize_enabled:
        blocks = normalize.normalize_blocks(blocks)
        st = normalize.hit_stats(blocks)
        notes.append(f"归一化：字表 {st['mapping_size']} 条，命中字符 {st['matched_chars']}，"
                     f"转换 {st['converted_chars']}（版本 {st['lexicon_version']}）")
    else:
        notes.append("归一化：已按配置关闭")

    tables = table.restore_tables(blocks, page_id) if CONFIG.ocr.table_enabled else []
    notes.append(f"表格：bbox 行列聚类还原 {len(tables)} 张方剂/剂量表")
    return blocks, lay, tables


def run_ocr(image_path: str = "", page_id: Optional[str] = None,
            backend: Optional[str] = None) -> OCRResult:
    """OCR 流水线入口。

    输入：本地图像路径、可选 page_id、可选后端覆盖
    输出：OCRResult（真实识别时 engine=rapidocr / model_loaded=True / is_demo=False）
    异常：显式降级为示例数据并标注，绝不静默伪造识别结果。
    """
    pid = page_id or new_page_id()
    notes: List[str] = []
    try:
        rec = recognize(image_path, pid, backend=backend)
        blocks = list(rec.blocks)
        notes.extend(rec.notes)

        width, height = rec.width, rec.height
        if not width or not height:
            width, height = _image_size(image_path)
        if not width or not height:
            notes.append("图像尺寸不可读：宽高记为 0，不硬编码")

        if rec.model_loaded:
            low = [b for b in blocks if float(b.conf or 0) < CONFIG.ocr.min_conf]
            blocks = [b for b in blocks if float(b.conf or 0) >= CONFIG.ocr.min_conf]
            if low:
                q = _write_review_queue(pid, image_path, low)
                notes.append(f"低置信过滤：剔除 {len(low)} 块（阈值 {CONFIG.ocr.min_conf}），"
                             f"已写入人工复核队列 {q}")

        blocks, lay, tables = _postprocess(blocks, pid, notes, is_demo=rec.is_demo)
        return OCRResult(
            page_id=pid, image_path=image_path, width=width, height=height,
            blocks=blocks, tables=tables, layout=lay,
            engine=rec.engine, model_loaded=rec.model_loaded, is_demo=rec.is_demo,
            source_hash=source_hash_of("ocr", pid, image_path, rec.engine),
            notes=notes + ([] if not rec.is_demo else [CONFIG.demo_badge]),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as exc:
        demo = load_demo_ocr(pid, image_path)
        return OCRResult(
            page_id=pid, image_path=image_path, width=demo.width, height=demo.height,
            blocks=demo.blocks, engine="stub", model_loaded=False, is_demo=True,
            source_hash=source_hash_of("ocr", pid, "fallback"),
            notes=[CONFIG.demo_badge, f"真实识别链路异常，显式降级：{type(exc).__name__}"],
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def run_demo(page_id: str = "DEMO-001") -> OCRResult:
    """显式走示例数据（前端「加载示例古籍页」按钮）。"""
    pid = page_id
    notes: List[str] = []
    blocks, lay, tables = _postprocess(load_demo_ocr(pid).blocks, pid, notes, is_demo=True)
    return OCRResult(
        page_id=pid, image_path="data/demo/demo_page.json",
        width=720, height=1000, blocks=blocks, tables=tables, layout=lay,
        engine="stub", model_loaded=False, is_demo=True,
        source_hash=source_hash_of("ocr", pid, "demo"),
        notes=[CONFIG.demo_badge] + notes,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def describe() -> dict:
    from .recognize import get_adapter
    adapter, notes = get_adapter()
    real = adapter.name == "rapidocr"
    return {
        "stage": "① 图像 → 文本",
        "backend": adapter.name,
        "model_loaded": real,
        "is_demo": not real,
        "badge": CONFIG.demo_badge if not real else "本地 OCR 已接入",
        "notes": notes,
        "todo": ["多模型投票", "小字夹注与眉批层级", "异体字表扩充"],
    }
