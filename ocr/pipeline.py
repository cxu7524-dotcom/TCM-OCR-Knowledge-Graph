# -*- coding: utf-8 -*-
"""
ocr/pipeline.py —— ① 图像 → 带 bbox 的 OCRResult（端到端流水线）

链路：图像 → 识别适配层 → 文本归一化 → 版面分析（竖排列 / 阅读顺序）→ 表格还原 → OCRResult
每个 Block 都带 page_id / bbox / line_id / quote / source_hash，全链路不丢字段。

本阶段是**可运行的空跑流水线**：识别结果来自 data/demo/ 示例数据，
返回值中 engine="stub"、model_loaded=False、is_demo=True，并在 notes 里写明。

TODO（下一阶段）：接入真实 OCR 后，把 recognize.py 的 infer() 与版面检测结果接到这里，
     并补上低置信块复核、跨列续句、页眉页脚剔除。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from app.config import CONFIG
from app.demo import load_demo_ocr
from app.schemas import OCRResult, source_hash_of

from . import layout, normalize, table
from .recognize import recognize


def new_page_id(prefix: str = "P") -> str:
    """生成页 ID：时间戳 + 来源指纹，不依赖数据库自增。"""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{stamp}-{source_hash_of(prefix, stamp, datetime.now().microsecond)[:6]}"


def _postprocess(blocks, page_id: str, notes: List[str]):
    """归一化 / 版面 / 表格三件套，统一在这里串，避免各路由各写一遍。"""
    if CONFIG.ocr.normalize_enabled:
        blocks = normalize.normalize_blocks(blocks)
        notes.append(f"归一化：演示小表 {normalize.mapping_size()} 条（非全量异体字表）")
    else:
        notes.append("归一化：已按配置关闭")

    lay = layout.analyze_layout(blocks, page_id=page_id) if CONFIG.ocr.vertical_layout else None
    if lay is None:
        from app.schemas import LayoutResult
        lay = LayoutResult(page_id=page_id, direction="horizontal-ltr",
                           engine="stub", is_demo=True, note="竖排分析已按配置关闭")
    notes.append(f"版面：{lay.direction}，识别 {len(lay.cols)} 列（规则桩，模型未接入）")

    tables = table.restore_tables(blocks, page_id) if CONFIG.ocr.table_enabled else []
    notes.append(f"表格：还原 {len(tables)} 张方剂/剂量结构（规则桩，模型未接入）")
    return blocks, lay, tables


def run_ocr(image_path: str = "", page_id: Optional[str] = None,
            backend: Optional[str] = None) -> OCRResult:
    """OCR 流水线入口。

    输入：图像路径（可为空，表示走示例页）、可选 page_id、可选后端覆盖
    输出：OCRResult（结构完整，含 blocks / layout / tables / engine / notes）
    异常：内部兜底，始终返回可序列化的 OCRResult，不抛异常。
    """
    pid = page_id or new_page_id()
    notes: List[str] = []
    try:
        rec = recognize(image_path, pid, backend=backend)
        blocks = rec.blocks
        notes.extend(rec.notes)
        # 置信度过滤只在真实模型加载后启用；桩数据 conf=0 属预期，不做过滤
        if rec.model_loaded:
            blocks = [b for b in blocks if b.conf >= CONFIG.ocr.min_conf]
            notes.append(f"已按 min_conf={CONFIG.ocr.min_conf} 过滤低置信块")
        blocks, lay, tables = _postprocess(blocks, pid, notes)

        return OCRResult(
            page_id=pid,
            image_path=image_path or "data/demo/demo_page.json",
            width=720, height=1000,
            blocks=blocks,
            tables=tables,
            layout=lay,
            engine=rec.engine,
            model_loaded=rec.model_loaded,
            is_demo=rec.is_demo,
            source_hash=source_hash_of("ocr", pid, image_path, rec.engine),
            notes=notes + [CONFIG.demo_badge],
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as exc:  # 桩不得抛异常
        demo = load_demo_ocr(pid, image_path)
        return OCRResult(
            page_id=pid, image_path=image_path, width=720, height=1000,
            blocks=demo.blocks, engine="stub", model_loaded=False, is_demo=True,
            source_hash=source_hash_of("ocr", pid, "fallback"),
            notes=[CONFIG.demo_badge, f"流水线降级：{type(exc).__name__}"],
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def run_demo(page_id: str = "DEMO-001") -> OCRResult:
    """显式走示例数据的便捷入口（前端「加载示例古籍页」按钮）。"""
    pid = page_id
    notes: List[str] = []
    blocks, lay, tables = _postprocess(load_demo_ocr(pid).blocks, pid, notes)
    return OCRResult(
        page_id=pid,
        image_path="data/demo/demo_page.json",
        width=720, height=1000,
        blocks=blocks, tables=tables, layout=lay,
        engine="stub", model_loaded=False, is_demo=True,
        source_hash=source_hash_of("ocr", pid, "demo"),
        notes=[CONFIG.demo_badge] + notes,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def describe() -> dict:
    """给 /api/health 用的模块说明（无任何实测数字）。"""
    return {
        "stage": "① 图像 → 文本",
        "backend": CONFIG.ocr.backend,
        "model_loaded": False,
        "is_demo": True,
        "badge": CONFIG.demo_badge,
        "todo": ["接入本地 OCR 推理并映射 bbox", "竖排版面检测与列切分", "异体字全量表"],
    }
