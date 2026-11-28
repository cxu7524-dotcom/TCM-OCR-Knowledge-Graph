# -*- coding: utf-8 -*-
"""第 4 步：候选池构建（开发页现跑 / 第 51 页由冻结产物装配）。

统一池格式，供 rules 引擎使用：
  blocks:      [{line_id, block_ref, bbox, text, n_hanzi, est_char_h}]
  windows:     {window_id: {line_id, x1,y1,x2,y2, height, mult}}
  candidates:  [{model, window_id, rank, text, seq_logprob, n_content_steps, alignment, line_id}]

用法：
  # 开发页：现跑整页 OCR（关闭纠错）+ 4H/2H 窗 + 两模型 CTC 前 5
  python tools/p51_step4_pool.py build --image <页面图> --page-id DEV-P32 --out <pool.json>

  # 第 51 页：由第 1-3 步冻结产物装配，不重新推理
  python tools/p51_step4_pool.py frozen --out <pool.json>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr import pipeline                      # noqa: E402
from ocr.recognize import _engine, _text_recognizer  # noqa: E402
from ocr.vertical import imread_unicode       # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_win", ROOT / "tools/p51_local_windows.py")
_win = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_win)
_spec2 = importlib.util.spec_from_file_location("p51_ctc", ROOT / "tools/p51_step3_ctc_candidates.py")
_ctc = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_ctc)

MODELS = _ctc.MODELS
P51_OCR = ROOT / "output/p51_baseline_20261122_171947_953/ocr.json"
P51_WIN4 = ROOT / "output/p51_step2_20261122_183000/windows.json"
P51_CAND4 = ROOT / "output/p51_step3b_20261122_185500/ctc_candidates.jsonl"
P51_WIN2 = ROOT / "output/p51_step3c_20261122_195000/short_windows.json"
P51_CAND2 = ROOT / "output/p51_step3c_20261122_195000/short_candidates.jsonl"


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hz(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def run_page_ocr(image: Path, page_id: str, tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = replace(pipeline.CONFIG,
                  paths=replace(pipeline.CONFIG.paths, reports_dir=tmp),
                  ocr=replace(pipeline.CONFIG.ocr, correct_enabled=False))
    with patch.object(pipeline, "CONFIG", cfg), patch.object(pipeline.correct, "CONFIG", cfg):
        res = pipeline.run_ocr(str(image), page_id=page_id)
    if res.is_demo or not res.model_loaded or not res.blocks:
        raise SystemExit("未获得真实 OCR：%s" % image)
    return {"engine": res.engine, "width": res.width, "height": res.height,
            "blocks": [b.to_dict() for b in res.blocks]}


def infer_candidates(img: np.ndarray, windows: list, page_id: str) -> list:
    """对每个窗口 × 两模型做一次前向，取 CTC 前 5 与贪心。"""
    engines, chars = {}, {}
    for key, rel, _d in MODELS:
        rec = _text_recognizer(_engine(str(ROOT / rel)))
        engines[key] = rec
        chars[key] = list(getattr(rec.postprocess_op, "character", []))
    cands = []
    for w in windows:
        crop = img[w["y1"]:w["y2"], w["x1"]:w["x2"]]
        rot = np.ascontiguousarray(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE))
        for key in engines:
            rec = engines[key]
            with _ctc.PredSpy(rec) as spy:
                res, _t = rec([rot], return_word_box=True)
            preds = spy.preds[0]
            T = preds.shape[0]
            cs = _ctc.content_steps(rot.shape, tuple(getattr(rec, "rec_image_shape", (3, 48, 320))), T)
            for r, (sc, text, align) in enumerate(
                    _ctc.ctc_beam(preds, chars[key], _ctc.BEAM_WIDTH, _ctc.TOPK_PER_STEP,
                                  _ctc.MAX_PREFIX)[:_ctc.TOP_N], start=1):
                if not text.strip():
                    continue
                cands.append({"model": key, "window_id": w["window_id"], "rank": r,
                              "text": text, "seq_logprob": round(sc, 6),
                              "n_content_steps": round(cs, 3),
                              "alignment": [[c, t] for c, t in align],
                              "line_id": w["line_id"]})
    return cands


def build(args):
    img = imread_unicode(str(args.image))
    if img is None:
        raise SystemExit("图像读取失败：%s" % args.image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ocr = run_page_ocr(args.image, args.page_id, args.out.parent / "_ocr_tmp")
    blocks = []
    for b in ocr["blocks"]:
        n = hz(b["text"])
        x1, y1, x2, y2 = b["bbox"]
        if n == 0:
            continue
        blocks.append({"line_id": b["line_id"], "block_ref": b["block_ref"], "bbox": b["bbox"],
                       "text": b["text"], "n_hanzi": n, "est_char_h": max(1, int(round((y2 - y1) / n)))})
    windows = []
    for mult, tag in ((4, "D4"), (2, "D2")):
        for b in blocks:
            for w in _win.windows_for(b, mult):
                w = dict(w)
                w["window_id"] = "%s:%s:%s" % (tag, b["line_id"], w["window_id"].split(":")[-1])
                w["mult"] = mult
                windows.append(w)
    cands = infer_candidates(img, windows, args.page_id)
    pool = {"page_id": args.page_id, "image": str(args.image.resolve()),
            "image_sha256": sha256_of(args.image), "image_size": [img.shape[1], img.shape[0]],
            "ocr_engine": ocr["engine"], "blocks": blocks,
            "windows": {w["window_id"]: {"line_id": w["line_id"], "x1": w["x1"], "y1": w["y1"],
                                         "x2": w["x2"], "y2": w["y2"],
                                         "height": w["height"], "mult": w["mult"]} for w in windows},
            "candidates": cands,
            "source": "本页现跑：整页 OCR(纠错关) + 4H/2H 窗 + 两模型 CTC 前 5"}
    args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(json.dumps({"page": args.page_id, "blocks": len(blocks), "windows": len(windows),
                      "candidates": len(cands), "out": str(args.out)}, ensure_ascii=False))


def frozen(args):
    ocr = json.loads(P51_OCR.read_text(encoding="utf8"))
    w4 = json.loads(P51_WIN4.read_text(encoding="utf8"))
    w2 = json.loads(P51_WIN2.read_text(encoding="utf8"))
    blocks = []
    for b in ocr["blocks"]:
        n = hz(b["text"])
        if n == 0:
            continue
        x1, y1, x2, y2 = b["bbox"]
        blocks.append({"line_id": b["line_id"], "block_ref": b["block_ref"], "bbox": b["bbox"],
                       "text": b["text"], "n_hanzi": n, "est_char_h": max(1, int(round((y2 - y1) / n)))})
    windows = {}
    for w in w4["windows"]:
        if w["group"] != "B":
            continue
        windows[w["window_id"]] = {"line_id": w["line_id"], "x1": w["x1"], "y1": w["y1"],
                                   "x2": w["x2"], "y2": w["y2"], "height": w["height"], "mult": 4}
    for w in w2["windows"]:
        windows[w["window_id"]] = {"line_id": w["line_id"], "x1": w["x1"], "y1": w["y1"],
                                   "x2": w["x2"], "y2": w["y2"], "height": w["height"], "mult": 2}
    cands = []
    for path, src in ((P51_CAND4, "4H"), (P51_CAND2, "2H")):
        for line in path.open(encoding="utf8"):
            c = json.loads(line)
            cands.append({"model": c["model"], "window_id": c["window_id"], "rank": c["rank"],
                          "text": c["text"], "seq_logprob": c["seq_logprob"],
                          "n_content_steps": c.get("n_content_steps") or c["n_steps"],
                          "alignment": c["alignment"], "line_id": c["line_id"], "src": src})
    pool = {"page_id": "RETEST-P51", "image": str((ROOT / "古籍评测集/医家类/思濟堂方書/page_51.png").resolve()),
            "image_sha256": sha256_of(ROOT / "古籍评测集/医家类/思濟堂方書/page_51.png"),
            "image_size": [ocr["width"], ocr["height"]], "ocr_engine": ocr["engine"],
            "blocks": blocks, "windows": windows, "candidates": cands,
            "source": "冻结装配：第1步 ocr.json + 第2步 4H 窗 + 第3步b 4H候选 + 第3步c 2H候选",
            "artifacts": {"ocr": str(P51_OCR), "win4": str(P51_WIN4), "cand4": str(P51_CAND4),
                          "win2": str(P51_WIN2), "cand2": str(P51_CAND2)}}
    args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(json.dumps({"blocks": len(blocks), "windows": len(windows), "candidates": len(cands),
                      "out": str(args.out)}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--image", type=Path, required=True)
    b.add_argument("--page-id", required=True)
    b.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("frozen")
    f.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (build if args.cmd == "build" else frozen)(args)


if __name__ == "__main__":
    main()
