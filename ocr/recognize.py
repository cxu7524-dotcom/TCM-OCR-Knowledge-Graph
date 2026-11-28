# -*- coding: utf-8 -*-
"""
ocr/recognize.py —— OCR 推理适配层（本地模型真接入）

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
import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

from app.config import CONFIG
from app.demo import load_demo_ocr
from app.schemas import Block, source_hash_of

_ENGINES: dict = {}
_ENGINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.RLock()
_LOG = logging.getLogger(__name__)


def resolve_rec_model() -> tuple:
    """返回 (实际可用的模型绝对路径, 显示名, 说明)。

    三级回退在 app/config.py 里完成（env → 自动探测 v5 → 包内 v3），这里只做
    「文件到底存不存在」的最终确认：配置指向的文件若已被删除，必须显式回退到
    包内 v3，而不是让 RapidOCR 在初始化时抛异常。
    """
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    raw = (CONFIG.ocr.rec_model or "").strip()
    if raw:
        path = _P(raw) if _P(raw).is_absolute() else root / raw
        if path.is_file():
            return str(path), path.name, f"来源={CONFIG.ocr.rec_model_source}"
    return "", "ch_PP-OCRv3_rec（包内简体）", "回退：未找到本地 v5 权重"


def rec_model_name() -> str:
    """当前实际生效的识别模型名（供 /api/health 与前端配置条显示）。"""
    return resolve_rec_model()[1]


def _resolve_rec_keys(model_path: str) -> str:
    """查找识别模型的外部字典，环境变量优先于同名伴随文件。"""
    from pathlib import Path as _P
    if not model_path:
        return ""
    raw = os.environ.get("TCM_OCR_REC_KEYS", "").strip()
    if raw:
        path = _P(raw)
        if not path.is_absolute():
            path = _P(__file__).resolve().parents[1] / path
        if path.is_file():
            return str(path.resolve())
    companion = _P(model_path).with_suffix(".keys.txt")
    return str(companion.resolve()) if companion.is_file() else ""


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
    if raw and not _P(raw).is_file():
        raw = ""          # 配置指向的权重不存在 → 显式回退包内 v3，不抛异常
    model_path = raw
    keys_path = _resolve_rec_keys(model_path)
    key = (model_path, keys_path)
    if key not in _ENGINES:
        with _ENGINE_LOCK:
            if key not in _ENGINES:
                from rapidocr_onnxruntime import RapidOCR
                kwargs = {"rec_model_path": model_path} if model_path else {}
                if keys_path:
                    kwargs["rec_keys_path"] = keys_path
                _ENGINES[key] = RapidOCR(**kwargs)
    return _ENGINES[key]


def _text_recognizer(engine: Any):
    """Return the recognizer exposed by current or older RapidOCR releases."""
    for name in ("text_rec", "text_recognizer"):
        rec = getattr(engine, name, None)
        if rec is not None:
            return rec
    raise AttributeError("RapidOCR exposes neither text_rec nor text_recognizer")



@contextmanager
def _rerank_inference(engine):
    # All calls using a shared engine participate, including calls with reranking off.
    with _INFERENCE_LOCK:
        rec = getattr(engine, "text_rec", None) or getattr(engine, "text_recognizer", None)
        if rec is None or getattr(rec, "postprocess_op", None) is None:
            yield
            return
        context = None
        if CONFIG.ocr.ctc_rerank:
            try:
                from ocr.ctc_rerank import ctc_rerank_context
                context = ctc_rerank_context(engine)
                context.__enter__()
            except Exception:
                _LOG.exception("Unable to enable CTC reranking; using original recognition")
                context = None
        try:
            yield
        finally:
            if context is not None:
                context.__exit__(None, None, None)


def model_charset() -> set:
    """当前识别模型的字符表（供 ocr/score.py 算「合法字符率」）。

    rapidocr 的 `TextRecognizer` 把字典放在 `postprocess_op.character`
    （首项是 'blank' 占位，末尾含空格），这里剔除占位符后返回 set。
    取不到（未装依赖 / 未加载模型）时返回空集，调用方按「无信号」处理。
    """
    try:
        engine = _engine()
        rec = _text_recognizer(engine)
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


@dataclass
class CharCTC:
    char: str
    prob: float
    timestep: int
    total_steps: int
    content_steps: float
    rel_pos: float
    rel_end: float


@dataclass
class CropCTCResult:
    text: str = ""
    conf: float = 0.0
    chars: List[CharCTC] = field(default_factory=list)
    conf_mean: float = 0.0
    conf_var: float = 0.0
    conf_min: float = 0.0
    total_steps: int = 0
    content_steps: float = 0.0


def compute_content_steps(img_shape: Tuple[int, ...], rec_image_shape: Sequence[int],
                          total_steps: int) -> float:
    """Estimate the portion of CTC steps occupied by the unpadded image."""
    try:
        steps = max(0, int(total_steps))
        if not steps:
            return 0.0
        h, w = float(img_shape[0]), float(img_shape[1])
        img_h, img_w = float(rec_image_shape[1]), float(rec_image_shape[2])
        if not all(math.isfinite(v) and v > 0 for v in (h, w, img_h, img_w)):
            return 0.0
        ratio = w / h
        padded = int(img_h * max(img_w / img_h, ratio))
        resized = min(padded, int(math.ceil(img_h * ratio)))
        if padded <= 0:
            return 0.0
        return max(1.0, min(float(steps), steps * resized / float(padded)))
    except (TypeError, ValueError, IndexError, OverflowError, ZeroDivisionError):
        return 0.0


def extract_ctc_chars(probs: Any, charset: Sequence[str], *,
                      content_steps: float | None = None,
                      text_override: str | None = None,
                      conf_override: float | None = None) -> CropCTCResult:
    """Greedily collapse a (time, character) CTC probability matrix."""
    import numpy as np

    try:
        matrix = np.asarray(probs)
        if matrix.ndim != 2:
            return CropCTCResult()
        total, columns = matrix.shape
        if not total:
            return CropCTCResult()
        content = float(content_steps) if content_steps is not None else float(total)
        if not math.isfinite(content):
            content = float(total)
        content = max(1.0, min(float(total), content))
        selected = []
        if columns:
            indices = np.argmax(matrix, axis=1)
            for t, raw_idx in enumerate(indices):
                idx = int(raw_idx)
                if idx and (t == 0 or idx != int(indices[t - 1])) and idx < len(charset):
                    selected.append((t, idx))
        override = text_override.strip() if text_override is not None else ""
        replacements = override if override and len(override) == len(selected) else None
        chars = []
        for i, (t, idx) in enumerate(selected):
            char = replacements[i] if replacements is not None else charset[idx]
            if replacements is not None:
                try:
                    replacement_idx = charset.index(char)
                    if replacement_idx < columns:
                        idx = replacement_idx
                except (ValueError, AttributeError):
                    pass
            prob = float(matrix[t, idx])
            prob = max(0.0, min(1.0, prob)) if math.isfinite(prob) else 0.0
            start = max(0.0, min(1.0, t / content))
            end_t = selected[i + 1][0] if i + 1 < len(selected) else t + content / max(1, len(selected))
            end = max(start, min(1.0, end_t / content))
            chars.append(CharCTC(char, prob, t, total, content, start, end))
        values = [c.prob for c in chars]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((p - mean) ** 2 for p in values) / len(values) if values else 0.0
        text = override or "".join(c.char for c in chars)
        return CropCTCResult(text, float(conf_override) if conf_override is not None else mean,
                             chars, mean, variance, min(values) if values else 0.0,
                             total, content)
    except (TypeError, ValueError, IndexError, OverflowError):
        return CropCTCResult()


def map_chars_to_y_spans(chars: Sequence[CharCTC], y1: float, y2: float) -> List[Tuple[str, float, float, float]]:
    """Map relative reading positions into the supplied vertical interval."""
    if y2 < y1:
        y1, y2 = y2, y1
    result = []
    for char in chars:
        start = max(0.0, min(1.0, char.rel_pos))
        end = max(start, min(1.0, char.rel_end))
        result.append((char.char, char.prob, y1 + (y2 - y1) * start,
                       y1 + (y2 - y1) * end))
    return result


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

    def infer(self, image_input: Any) -> List[Tuple[Any, str, float]]:
        """调用本地 ONNX 模型，返回 [(box, text, conf)]。异常时不伪造，返回空列表。"""
        try:
            engine = _engine()
            with _rerank_inference(engine):
                result, _ = engine(
                    image_input,
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
        _path, _name, _src = resolve_rec_model()
        notes = [
            "推理后端：rapidocr-onnxruntime（本机 ONNX Runtime，CPU，模型随包分发，无公网调用）",
            f"识别模型：{_name}（{_src}）",
            f"检测文本块 {len(raw)} 个",
        ]
        if not _path:
            notes.append("未找到本地 v5 权重，已回退包内简体 ch_PP-OCRv3_rec：CER 会明显变差；"
                         "如需最佳效果请把 ch_PP-OCRv5_rec_mobile.onnx 放到 模型权重/ 或用 "
                         "TCM_OCR_REC_MODEL 指定路径")
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
        with _rerank_inference(engine):
            res, _ = _text_recognizer(engine)([image_array])
        if res:
            text, score = res[0]
            text, score = str(text).strip(), float(score)
            if text and score >= min_conf:
                return text, round(score, 4)
    except Exception as exc:
        _LOG.warning("Crop OCR failed (%s): %s", type(exc).__name__, exc)
        return "", 0.0
    return "", 0.0


def recognize_crop_with_ctc(image_array: Any, min_conf: float = 0.0, *,
                            rec_model: str | None = None) -> CropCTCResult:
    """Recognize one crop and optionally expose its per-character CTC signals."""
    try:
        engine = _engine(rec_model)
        rec = _text_recognizer(engine)
        with _INFERENCE_LOCK:
            orig_op = rec.postprocess_op
            charset = list(getattr(orig_op, "character", None) or [])
            rec_shape = getattr(rec, "rec_image_shape", (3, 48, 320))
            captured = []
            with _rerank_inference(engine):
                current_op = rec.postprocess_op

                def capture(preds, *args, **kwargs):
                    captured.append(preds[0].copy())
                    return current_op(preds, *args, **kwargs)

                rec.postprocess_op = capture
                try:
                    res, _ = rec([image_array])
                finally:
                    rec.postprocess_op = current_op
        if not res or not captured:
            return CropCTCResult()
        text, score = str(res[0][0]).strip(), float(res[0][1])
        if not text or not math.isfinite(score) or score < min_conf:
            return CropCTCResult()
        probs = captured[0]
        content = compute_content_steps(image_array.shape, rec_shape, probs.shape[0])
        return extract_ctc_chars(probs, charset, content_steps=content,
                                 text_override=text, conf_override=score)
    except Exception as exc:
        _LOG.warning("Crop CTC OCR failed (%s): %s", type(exc).__name__, exc)
        return CropCTCResult()


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
    """选择后端。包已安装时不再强制降级为 stub。"""
    want = (backend or CONFIG.ocr.backend or "rapidocr").lower()
    notes: List[str] = []
    adapter: BaseAdapter = _ADAPTERS.get(want, StubOcrAdapter)()

    if adapter.name == "paddleocr":
        notes.append("paddleocr 推理未实现，按约定降级为 stub（不伪造识别结果）")
        adapter = StubOcrAdapter()
    elif adapter.name != "stub" and not adapter.available():
        notes.append(f"后端 {adapter.name} 依赖未安装（可选依赖，见 requirements.txt）")
        if CONFIG.ocr.auto_fallback:
            notes.append("已按 auto_fallback 降级为 stub：本次不产出识别结果，只回放示例数据")
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
