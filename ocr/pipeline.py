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

from . import correct, layout, normalize, score, table
from .recognize import RecognizeResult, _image_size, recognize


def _text_score(blocks) -> float:
    """旧口径：Σ(字数 × 置信度)。保留仅为日志对照，**不再用于择优**。

    该分数天然偏向字多的整页（逐列漏一列就输），阶段四 34 页验证集逐列全输即由此而来。
    """
    return sum(len(b.text or "") * float(b.conf or 0) for b in blocks)


def _recognize_page(image_path: str, page_id: str, backend: Optional[str]) -> RecognizeResult:
    """竖排页跑两路候选（逐列 / 整页），按**复合无监督分**择优；横排页只跑整页。

    择优信号见 `ocr/score.py`（合法字符率 / 平均置信度 / 列数合理性 / 重复率惩罚），
    不再用 Σ(字数×conf)——那个口径把逐列路径系统性压死。
    """
    base = recognize(image_path, page_id, backend)
    pick = (CONFIG.ocr.pick or "auto").strip().lower()
    if not CONFIG.ocr.vertical_split or not image_path or pick == "page":
        return base
    try:
        from .vertical import imread_unicode, is_vertical_page, ocr_page, page_layout
        img = imread_unicode(image_path)
        if img is None or not is_vertical_page(img):
            return base
        blocks, meta = ocr_page(image_path, page_id)
        if not blocks:
            base.notes.append("竖排逐列未识别出内容，采用整页结果")
            return base
        if pick == "vertical":
            return RecognizeResult(
                blocks=blocks, engine="rapidocr-vertical", model_loaded=True,
                is_demo=False, width=meta["width"], height=meta["height"],
                notes=meta["notes"] + ["择优策略：强制逐列（TCM_OCR_PICK=vertical，A/B 对照用）"],
            )
        # 两路的「列数合理性」用同一份列结构评估，避免对整页路径不公平
        n_cols_page = len(page_layout(image_path)["cols"]) or 1
        charset = score.legal_charset()
        # 置信度低于阈值的块在 run_ocr 里会被剔除，这里先剔一遍，保证两路候选可比
        def _kept(bs):
            return [b for b in bs if float(b.conf or 0) >= CONFIG.ocr.min_conf]
        v = score.composite(_kept(blocks), int(meta.get("columns") or 0), charset)
        p = score.composite(_kept(base.blocks), n_cols_page, charset)
        base.notes.append(f"择优（整页）：{score.fmt(p)}")
        meta["notes"].append(f"择优（逐列）：{score.fmt(v)}")
        margin = float(CONFIG.ocr.pick_margin or 1.0)
        if v["score"] > p["score"] * margin:
            return RecognizeResult(
                blocks=blocks, engine="rapidocr-vertical", model_loaded=True,
                is_demo=False, width=meta["width"], height=meta["height"],
                notes=meta["notes"] + [f"择优结果：逐列胜（{v['score']:.4f} > "
                                       f"整页 {p['score']:.4f} × {margin:.2f}）"],
            )
        base.notes.append(f"择优结果：整页胜（逐列 {v['score']:.4f} ≤ "
                          f"整页 {p['score']:.4f} × {margin:.2f}），采用整页结果")
        return base
    except Exception as exc:
        base.notes.append(f"竖排逐列降级为整页识别：{type(exc).__name__}")
        return base


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


def _postprocess(blocks, page_id: str, notes: List[str], is_demo: bool,
                 image_path: str = ""):
    """版面 → 表格补格 → 字形纠错 → 归一化 → 表格还原。"""
    # 顺序要点：
    #  ① 版面合并会重建 Block，必须最先做；
    #  ② 表格补格会新增 Block，必须在纠错之前，让补出来的文本也走同一套纠错规则；
    #  ③ 归一化放最后，避免 text_norm 被后续合并覆盖。
    blocks, lay = layout.analyze_layout(blocks, page_id=page_id, is_demo=is_demo)
    notes.append(f"版面：{lay.direction}，{len(lay.cols)} 列；{lay.note}")

    filled = []
    if CONFIG.ocr.table_enabled:
        try:
            blocks, filled = table.fill_missing_cells(image_path, blocks, page_id)
        except Exception as exc:
            notes.append(f"表格补格降级（不影响识别结果）：{type(exc).__name__}")

    if CONFIG.ocr.correct_enabled:
        blocks, corrections = correct.correct_blocks(blocks)
        if corrections:
            path = correct.write_trace(corrections)
            rules_hit: dict = {}
            for c in corrections:
                for r in c["rules"]:
                    rules_hit[r["rule"]] = rules_hit.get(r["rule"], 0) + 1
            notes.append(f"字形纠错：{len(corrections)} 块，规则命中 {rules_hit}，trace {path}")
        else:
            notes.append("字形纠错：本页无命中（规则表见 data/lexicon/ocr_confusions.json）")
    else:
        notes.append("字形纠错：已按配置关闭（TCM_OCR_CORRECT=0）")

    if CONFIG.ocr.normalize_enabled:
        blocks = normalize.normalize_blocks(blocks)
        st = normalize.hit_stats(blocks)
        notes.append(f"归一化：字表 {st['mapping_size']} 条，命中字符 {st['matched_chars']}，"
                     f"转换 {st['converted_chars']}（版本 {st['lexicon_version']}）")
    else:
        notes.append("归一化：已按配置关闭")

    if filled:
        sample = filled[0]
        got = sample.get("after") or sample.get("text") or ""
        notes.append(f"表格补格：{len(filled)} 处二次识别"
                     f"（例：{got}，rule={sample.get('rule', '')}）")

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
        rec = _recognize_page(image_path, pid, backend)
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

        blocks, lay, tables = _postprocess(blocks, pid, notes, is_demo=rec.is_demo,
                                           image_path=image_path)
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
