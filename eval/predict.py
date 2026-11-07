# -*- coding: utf-8 -*-
"""
eval/predict.py —— 真实预测与问答跑批（供 eval/run_all.py 调用）

数据流（全部基于真实 OCR 输出，不读取 GT 文本作为预测）：
  data/samples/synthetic/pages.jsonl
        ↓ run_ocr（本地 ONNX 模型）
  data/reports/ocr_predictions.jsonl          OCR 文本块（含 bbox / quote / conf）
        ↓ extract/rules.py                     ↓ ocr/table.py
  extraction_predictions.jsonl              table_predictions.jsonl
        ↓ qa.answer（证据强约束 + 拒答）
  data/reports/qa_log.jsonl                  问答跑批日志

幂等：四个文件齐备且非空时默认跳过；`--refresh` / force=True 强制重跑。
降级：影像或 OCR 不可用时如实登记，CER 由评测脚本自动标记「待测」。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                                    # noqa: E402
from app.schemas import Block                                    # noqa: E402
from eval.dataset import GT_DIR, load_jsonl                       # noqa: E402
from ocr.pipeline import run_ocr                                  # noqa: E402

REPORT_DIR = CONFIG.paths.reports_dir
PAGE_SPEC = CONFIG.paths.samples_dir / "synthetic" / "pages.jsonl"
FILES = {
    "ocr": REPORT_DIR / "ocr_predictions.jsonl",
    "entities": REPORT_DIR / "extraction_predictions.jsonl",
    "tables": REPORT_DIR / "table_predictions.jsonl",
    "qa_log": REPORT_DIR / "qa_log.jsonl",
}


def _jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _exists_all() -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in FILES.values())


def environment_info() -> Dict[str, Any]:
    """如実登记环境、已装依赖与降级情况。"""
    pkgs = ["rapidocr_onnxruntime", "onnxruntime", "PIL", "numpy", "cv2",
            "paddleocr", "fastapi", "uvicorn"]
    state = {}
    for name in pkgs:
        try:
            state[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            state[name] = False
    images = list((CONFIG.paths.samples_dir / "synthetic" / "pages").glob("*.png"))
    return {
        "Python": sys.version.split()[0],
        "OCR 后端": "rapidocr-onnxruntime（本机 ONNX Runtime，CPU，模型随包分发，无公网调用）",
        "LLM 后端": CONFIG.llm.backend + "（本阶段不部署 / 不调用）",
        "向量通道": CONFIG.vector.backend + "（占位，未启用）",
        "已装依赖": [k for k, v in state.items() if v],
        "未装依赖": [k for k, v in state.items() if not v],
        "合成影像": f"Pillow 本地渲染 {len(images)} 页（不进仓库，可复现）",
        "低置信阈值": CONFIG.ocr.min_conf,
        "检测阈值": f"box_thresh={CONFIG.ocr.det_box_thresh}, "
                    f"text_score={CONFIG.ocr.det_text_score}",
        "公网 / 密钥": "无（未读取任何 API Key，未访问任何公网推理服务）",
    }


def _block_row(b: Block) -> Dict[str, Any]:
    d = b.to_dict()
    d.update({"is_demo": False, "is_synthetic": True,
              "engine": "rapidocr", "quote": b.text})
    return d


def run_all_predictions() -> Dict[str, Any]:
    """完整跑一遍：OCR → 抽取 / 表格 → 问答，写四个预测文件。"""
    from extract.rules import extract_all
    from qa.generate import answer as qa_answer

    specs = load_jsonl(PAGE_SPEC)
    if not specs:
        return {"error": "页面规格为空，请先运行 python tools/make_synthetic_pages.py"}

    ocr_rows: List[Dict[str, Any]] = []
    ent_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    all_blocks: List[Block] = []
    engines = set()
    degraded = 0

    for spec in specs:
        pid = str(spec.get("page_id", ""))
        image = _ROOT / str(spec.get("image", ""))
        ocr = run_ocr(str(image), pid)
        engines.add(ocr.engine)
        if ocr.is_demo or not ocr.model_loaded:
            degraded += 1
        for b in ocr.blocks:
            ocr_rows.append(_block_row(b))
            all_blocks.append(b)
        entities, _ = extract_all(ocr.blocks)
        for e in entities:
            d = e.to_dict()
            d.update({"is_demo": False, "is_synthetic": True})
            ent_rows.append(d)
        for t in ocr.tables:
            table_rows.append({
                "page_id": t.page_id, "table_id": t.table_id, "rows": t.rows,
                "cells": t.cells, "source_hash": t.source_hash,
                "is_demo": False, "is_synthetic": True,
            })

    _jsonl(FILES["ocr"], ocr_rows)
    _jsonl(FILES["entities"], ent_rows)
    _jsonl(FILES["tables"], table_rows)

    # 问答跑批：有依据问题 + 无依据问题，统一在全库真实 OCR 文本块上检索
    qa_rows: List[Dict[str, Any]] = []
    gold = load_jsonl(GT_DIR / "qa_log_gold.jsonl")
    no_ev = load_jsonl(GT_DIR / "qa_no_evidence.jsonl")
    for item in gold + no_ev:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        ans = qa_answer(q, all_blocks, is_demo=False)
        qa_rows.append({
            "question": q,
            "page_id": item.get("page_id", ""),
            "category": item.get("category", "有依据问题"),
            "expected_refused": bool(item.get("expected_refused")),
            "refused": ans.refused,
            "refuse_reason": ans.refuse_reason,
            "text": ans.text,
            "evidences": [e.to_dict() for e in ans.evidences],
            "trace": ans.trace,
            "backend": ans.backend,
            "is_demo": False,
            "is_synthetic": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    _jsonl(FILES["qa_log"], qa_rows)

    return {
        "predictions_generated": True,
        "pages": len(specs),
        "ocr_blocks": len(ocr_rows),
        "entities": len(ent_rows),
        "tables": len(table_rows),
        "qa_cases": len(qa_rows),
        "engines": sorted(engines),
        "degraded_pages": degraded,
        "environment": environment_info(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def ensure_predictions(force: bool = False) -> Dict[str, Any]:
    """幂等入口：文件齐备则复用，否则（或 force）重跑。"""
    if _exists_all() and not force:
        return {"predictions_generated": False,
                "reason": "预测文件齐备，复用（--refresh 可强制重跑）",
                "environment": environment_info()}
    try:
        meta = run_all_predictions()
        meta["environment"] = environment_info()
        meta.setdefault("reason", "")
        return meta
    except Exception as exc:
        return {"predictions_generated": False,
                "reason": f"预测生成失败：{type(exc).__name__}: {exc}",
                "environment": environment_info()}


def main() -> None:
    meta = ensure_predictions(force="--refresh" in sys.argv)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
