# -*- coding: utf-8 -*-
"""
ocr/recognize.py —— OCR 推理适配层（阶段二：本地模型真接入）

后端：
  rapidocr  默认。rapidocr-onnxruntime（包内自带 ONNX 权重，纯 CPU、全程离线）。
  paddleocr 可选，本阶段未实现推理，选中时如实降级并标注。
  stub      不加载模型，返回 data/demo 示例数据（仅在依赖缺失 / 显式指定时）。

**API 依据（已阅读本机已安装包源码，非凭记忆）**：
  `rapidocr_onnxruntime/__init__.py` → `from .rapid_ocr_api import RapidOCR`
  `rapid_ocr_api.py: RapidOCR.__call__(img_content: str|np.ndarray|bytes|Path, **kwargs)`
    返回 `(result, elapse_list)` 或 `(None, None)`；
    `result` 形如 `[[box_4x2_list, text, score_str], ...]`，box 为四点坐标，score 为字符串。
  `get_crop_img_list()` 对高宽比 ≥1.5 的检测框自动 `np.rot90`，因此竖排列可直接识别。
  `config.yaml`：use_cuda=false，模型路径 models/*.onnx（随包分发，无公网下载）。

TODO（阶段三）：多模型投票、小字夹注专用识别、GPU 之外的加速与批处理。
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from app.config import CONFIG
from app.demo import load_demo_ocr
from app.schemas import Block, source_hash_of

_ENGINES: dict = {}
_ENGINE_LOCK = threading.Lock()


def _engine(rec_model: str | None = None):
    """懒加载 + 按识别模型缓存，避免重复初始化 ONNX Runtime。

    `rec_model` 为空时用包内简体 ch_PP-OCRv3_rec；指定本地 ONNX 路径时加载该模型。
    繁体模型（chinese_cht_PP-OCRv3_rec_mobile.onnx）的字典内嵌在 ONNX metadata
    的 `character` 字段里，`session.get_character_list()` 可直接读取，无需额外 keys 文件。
    """
    from pathlib import Path as _P
    raw = (rec_model if rec_model is not None else CONFIG.ocr.rec_model) or ""
    raw = raw.strip()
    if raw and not _P(raw).is_absolute():
        raw = str(_P(__file__).resolve().parents[1] / raw)
    key = raw
    if key not in _ENGINES:
        with _ENGINE_LOCK:
            if key not in _ENGINES:
                from rapidocr_onnxruntime import RapidOCR
                kwargs = {"rec_model_path": key} if key else {}
                _ENGINES[key] = RapidOCR(**kwargs)
    return _ENGINES[key]


def model_charset() -> set:
    """当前识别模型的字符表（供 ocr/score.py 算「合法字符率」）。

    rapidocr 的 `TextRecognizer` 把字典放在 `postprocess_op.character`
    （首项是 'blank' 占位，末尾含空格），这里剔除占位符后返回 set。
    取不到（未装依赖 / 未加载模型）时返回空集，调用方按「无信号」处理。
    """
    try:
        engine = _engine()
        rec = getattr(engine, "text_recognizer", None)
        op = getattr(rec, "postprocess_op", None)
        chars = getattr(op, "character", None)
        if not chars:
            return set()
        return {c for c in chars if c and c not in {"blank", " "}}
    except Exception:
        return set()


@dataclass
class RecognizeResult:
    """识别适配层产出（pipeline 再补版面 / 归一化 / 表格）。"""

    blocks: List[Block] = field(default_factory=list)
    engine: str = "stub"
    model_loaded: bool = False
    is_demo: bool = True
    width: int = 0
    height: int = 0
    notes: List[str] = field(default_factory=list)


class BaseAdapter:
    name = "base"

    def available(self) -> bool:
        return False

    def infer(self, image_path: str) -> List[Tuple[Any, str, float]]:
        """图像 → [(box, text, conf)]；box 为四点坐标列表。"""
        return []

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:  # pragma: no cover
        raise NotImplementedError

    def todo(self) -> str:
        return ""


class StubOcrAdapter(BaseAdapter):
    """示例数据适配器：零依赖、结构完整、显式标注 is_demo=True。"""

    name = "stub"

    def available(self) -> bool:
        return True

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:
        demo = load_demo_ocr(page_id=page_id, image_path=image_path)
        return RecognizeResult(
            blocks=demo.blocks, engine="stub", model_loaded=False, is_demo=True,
            width=demo.width, height=demo.height,
            notes=list(demo.notes) + [f"输入图像：{image_path or '（示例页）'}；未执行任何 OCR 推理"],
        )

    def todo(self) -> str:
        return "桩数据仅用于降级演示；真实识别请接入 rapidocr 后端。"


class RapidOcrAdapter(BaseAdapter):
    """本地 CPU OCR：rapidocr-onnxruntime。"""

    name = "rapidocr"
    MODULE = "rapidocr_onnxruntime"

    def available(self) -> bool:
        try:
            return importlib.util.find_spec(self.MODULE) is not None
        except (ImportError, ValueError):
            return False

    def infer(self, image_path: str) -> List[Tuple[Any, str, float]]:
        """调用本地 ONNX 模型，返回 [(box, text, conf)]。异常时不伪造，返回空列表。"""
        try:
            result, _ = _engine()(
                image_path,
                box_thresh=CONFIG.ocr.det_box_thresh,
                text_score=CONFIG.ocr.det_text_score,
            )
        except Exception:
            return []
        if not result:
            return []
        out: List[Tuple[Any, str, float]] = []
        for item in result:
            try:
                box, text, score = item[0], str(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if text.strip():
                out.append((box, text.strip(), round(score, 4)))
        return out

    def recognize(self, image_path: str, page_id: str) -> RecognizeResult:
        """图像 → Block 列表，补齐溯源五元组，model_loaded=True / is_demo=False。"""
        raw = self.infer(image_path)
        width, height = _image_size(image_path)
        notes = [
            "推理后端：rapidocr-onnxruntime（本机 ONNX Runtime，CPU，模型随包分发，无公网调用）",
            f"检测文本块 {len(raw)} 个",
        ]
        if not raw:
            notes.append("模型未返回任何文本块：本页不产出识别结果（不以示例文本冒充）")

        blocks: List[Block] = []
        for i, (box, text, conf) in enumerate(raw, start=1):
            bbox = _box_to_bbox(box)
            line_id = f"R{i:03d}"
            blocks.append(Block(
                page_id=page_id, bbox=bbox, line_id=line_id, quote=text,
                source_hash=source_hash_of("ocr", page_id, line_id, text),
                text=text, text_norm=text, kind="paragraph", conf=conf,
                col_id=0, reading_order=i, block_ref=f"{page_id}:{line_id}",
            ))
        return RecognizeResult(
            blocks=blocks, engine="rapidocr", model_loaded=True, is_demo=False,
            width=width, height=height, notes=notes,
        )

    def todo(self) -> str:
        return "已接入真实推理；阶段三可补多模型投票与小字夹注识别。"


class PaddleOcrAdapter(RapidOcrAdapter):
    """可选后端：本阶段未实现推理，选中时如实降级。"""

    name = "paddleocr"
    MODULE = "paddleocr"

    def infer(self, image_path: str) -> List[Tuple[Any, str, float]]:
        return []

    def todo(self) -> str:
        return "未实现 infer()；可选后端，接入前请勿用于评测。"


def recognize_crop(image_array: Any, min_conf: float = 0.15) -> Tuple[str, float]:
    """对单张裁剪图直接走识别模型（跳过检测），用于表格缺失单元格补识别。

    输入：numpy 数组（PIL Image 转来即可）
    输出：(text, conf)；识别为空或低于阈值返回 ("", 0.0)，不抛异常。
    注意：该路径**不使用任何 GT 坐标**，网格位置由检测框聚类推断。
    """
    try:
        engine = _engine()
        res, _ = engine.text_recognizer([image_array])
        if res:
            text, score = res[0]
            text, score = str(text).strip(), float(score)
            if text and score >= min_conf:
                return text, round(score, 4)
    except Exception:
        return "", 0.0
    return "", 0.0


def _box_to_bbox(box: Any) -> List[int]:
    """四点框 → [x1,y1,x2,y2] 外接矩形（坐标可直接回原图叠加）。"""
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    except (TypeError, ValueError, IndexError):
        return [0, 0, 0, 0]


def _image_size(image_path: str) -> Tuple[int, int]:
    """读取图像真实宽高；不可读时返回 (0, 0)，不硬编码。"""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


_ADAPTERS = {c.name: c for c in (RapidOcrAdapter, PaddleOcrAdapter, StubOcrAdapter)}


def get_adapter(backend: str | None = None) -> Tuple[BaseAdapter, List[str]]:
    """选择后端。阶段二修复：包已安装时不再强制降级为 stub。"""
    want = (backend or CONFIG.ocr.backend or "rapidocr").lower()
    notes: List[str] = []
    adapter: BaseAdapter = _ADAPTERS.get(want, StubOcrAdapter)()

    if adapter.name == "paddleocr":
        notes.append("paddleocr 推理未实现，按约定降级为 stub（不伪造识别结果）")
        adapter = StubOcrAdapter()
    elif adapter.name != "stub" and not adapter.available():
        notes.append(f"后端 {adapter.name} 依赖未安装（可选依赖，见 requirements.txt）")
        if CONFIG.ocr.auto_fallback:
            notes.append("已按 auto_fallback 降级为 stub：示例数据 / 模型未接入")
            adapter = StubOcrAdapter()
    elif adapter.name == "rapidocr":
        notes.append("本地 OCR 依赖可用，执行真实推理（engine=rapidocr, model_loaded=True）")
    return adapter, notes


def recognize(image_path: str, page_id: str, backend: str | None = None) -> RecognizeResult:
    """识别入口。任何情况下返回结构完整的 RecognizeResult，不抛异常。"""
    try:
        adapter, notes = get_adapter(backend)
        result = adapter.recognize(image_path, page_id)
        result.notes.extend(notes)
        return result
    except Exception as exc:  # 真实推理失败 → 显式降级，绝不静默伪造
        fallback = StubOcrAdapter().recognize(image_path, page_id)
        fallback.notes.append(f"真实推理失败，降级为示例数据：{type(exc).__name__}")
        fallback.is_demo = True
        return fallback
