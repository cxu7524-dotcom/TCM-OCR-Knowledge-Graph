# -*- coding: utf-8 -*-
"""
ocr/recognize.py —— OCR 推理适配层（可插拔）

目标：让上层 pipeline 只认一个接口，模型后端可替换、可降级、可不装。
后端：
  rapidocr    默认（rapidocr-onnxruntime，CPU 可跑，可选依赖）
  paddleocr   可选替代实现
  stub        不加载任何模型，返回带完整结构的示例数据

本阶段所有真实后端都只做**可用性探测 + TODO 占位**，不猜第三方 API、不做假识别：
  包已安装但推理代码未接入时，明确降级为桩并在 notes 里写明，绝不静默伪造识别结果。

TODO（下一阶段）：
  - 按官方文档实现 RapidOcrAdapter.infer()：图像 → (text, conf, box) 列表；
  - 竖排单行识别、小字夹注、印章与批注的过滤策略；
  - 置信度阈值与低置信块的人工复核队列（阈值在 app/config.py）。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import List, Tuple

from app.config import CONFIG
from app.demo import load_demo_ocr
from app.schemas import Block


@dataclass
class RecognizeResult:
    """识别适配层产出（不是最终 OCRResult，后者由 pipeline 补齐版面与表格）。"""

    blocks: List[Block] = field(default_factory=list)
    engine: str = "stub"
    model_loaded: bool = False
    is_demo: bool = True
    notes: List[str] = field(default_factory=list)


class BaseAdapter:
    """适配器接口契约：available() 与 recognize()。子类不得抛异常。"""

    name = "base"

    def available(self) -> bool:
        return False

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:  # pragma: no cover
        raise NotImplementedError

    def todo(self) -> str:
        return ""


class StubOcrAdapter(BaseAdapter):
    """示例数据适配器：零依赖、结构完整、显式标注。"""

    name = "stub"

    def available(self) -> bool:
        return True

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:
        demo = load_demo_ocr(page_id=page_id, image_path=image_path)
        return RecognizeResult(
            blocks=demo.blocks, engine="stub", model_loaded=False, is_demo=True,
            notes=list(demo.notes) + [f"输入图像：{image_path or '（示例页）'}；未执行任何 OCR 推理"],
        )

    def todo(self) -> str:
        return "桩数据仅用于打通链路；真实识别请接入 rapidocr 后端。"


class RapidOcrAdapter(BaseAdapter):
    """默认 OCR 后端。本阶段只探测依赖，不做推理。"""

    name = "rapidocr"
    MODULE = "rapidocr_onnxruntime"

    def available(self) -> bool:
        try:
            return importlib.util.find_spec(self.MODULE) is not None
        except (ImportError, ValueError):
            return False

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:
        # TODO: 按 rapidocr-onnxruntime 官方 API 实现 infer() 并映射到 Block
        #       映射要求：box → bbox，text → text，score → conf，
        #                 且必须补齐 page_id / line_id / quote / source_hash。
        return StubOcrAdapter().recognize(image_path, page_id)

    def todo(self) -> str:
        return "未实现 infer()；包可用时也降级为桩，避免伪造识别结果。"


class PaddleOcrAdapter(RapidOcrAdapter):
    """可选 OCR 后端。"""

    name = "paddleocr"
    MODULE = "paddleocr"

    def todo(self) -> str:
        return "未实现 infer()；可选后端，接入前请勿用于演示。"


_ADAPTERS = {c.name: c for c in (RapidOcrAdapter, PaddleOcrAdapter, StubOcrAdapter)}


def get_adapter(backend: str | None = None) -> Tuple[BaseAdapter, List[str]]:
    """选择后端并返回（适配器，说明）。依赖缺失时按 auto_fallback 降级。"""
    want = (backend or CONFIG.ocr.backend or "stub").lower()
    notes: List[str] = []
    cls = _ADAPTERS.get(want, StubOcrAdapter)
    adapter: BaseAdapter = cls()
    if adapter.name != "stub" and not adapter.available():
        notes.append(f"后端 {adapter.name} 依赖未安装（可选依赖，见 requirements.txt）")
        if CONFIG.ocr.auto_fallback:
            notes.append("已按 auto_fallback 降级为 stub：示例数据 / 模型未接入")
            adapter = StubOcrAdapter()
    elif adapter.name != "stub":
        notes.append(f"后端 {adapter.name} 依赖已安装，但推理调用未实现；本阶段仍输出示例数据")
        adapter = StubOcrAdapter()
    notes.append(f"适配器 TODO：{adapter.todo()}" if adapter.todo() else "")
    return adapter, [n for n in notes if n]


def recognize(image_path: str, page_id: str, backend: str | None = None) -> RecognizeResult:
    """识别入口。任何情况下返回结构完整的 RecognizeResult，不抛异常。"""
    try:
        adapter, notes = get_adapter(backend)
        result = adapter.recognize(image_path, page_id)
        result.notes.extend(notes)
        return result
    except Exception as exc:  # 桩不得抛异常
        fallback = StubOcrAdapter().recognize(image_path, page_id)
        fallback.notes.append(f"识别降级：{type(exc).__name__}")
        return fallback
