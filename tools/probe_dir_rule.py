# -*- coding: utf-8 -*-
"""
tools/probe_dir_rule.py —— 阶段七 P0：59 页方向判定明细（old/new 各跑一次）

只调用 `ocr.recognize` + `ocr.layout.detect_direction_detail`，**不读 GT、不算 CER**，
输出逐页 direction 与新增无监督信号明细，供报告复核与 A/B 对照。

用法（环境变量决定 rule，CONFIG 是进程内单例，需分两次进程跑）：
    TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=old \
      TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx python tools/probe_dir_rule.py
    TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=new \
      TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx python tools/probe_dir_rule.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                 # noqa: E402
from ocr import layout                        # noqa: E402
from ocr.recognize import recognize           # noqa: E402
from tools.eval_guji import load_dir_pages    # noqa: E402


def main() -> int:
    pages = load_dir_pages(_ROOT / "古籍评测集" / "医家类")
    out = []
    for i, p in enumerate(pages, start=1):
        rec = recognize(p["image"], f"DIR-{i:03d}", None)
        d = layout.detect_direction_detail(rec.blocks, image_path=p["image"])
        out.append({
            "book": p["book"], "page_no": p["page_no"], "image": p["image"],
            "direction": d["direction"], "rule": d["rule"],
            "vertical_ratio": d["vertical_ratio"], "columns": d["columns"],
            "aspect_median": d["aspect_median"], "col_gap_cv": d["col_gap_cv"],
            "text_len_cv": d["text_len_cv"], "box_count": d["box_count"],
            "img_columns": d["img_columns"],
            "img_col_aspect_median": d["img_col_aspect_median"],
            "img_vertical": d["img_vertical"], "dense_regular": d["dense_regular"],
            "reason": d["reason"],
        })
    rule = (CONFIG.ocr.dir_rule or "old").strip().lower()
    dst = _ROOT / "data" / "reports" / f"dir_stage7_{rule}.json"
    dst.write_text(json.dumps({"rule": rule, "pages": out},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rule={rule}  {dict(Counter(r['direction'] for r in out))} → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
