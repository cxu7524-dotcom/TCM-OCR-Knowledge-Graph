# -*- coding: utf-8 -*-
"""第 51 页固定窗口同框识别：v5 与繁体 v3 在同一批原像素窗口上的候选召回（第三步）。

只做候选召回，不做自动采纳：
  - 输入：第 2 步 windows.json（B 组 414 窗）+ crops/*.png 原像素裁图
  - 两套本地模型：ch_PP-OCRv5_rec_mobile.onnx / chinese_cht_PP-OCRv3_rec_mobile.onnx
  - 同一旋转（90° 逆时针，与 ocr/vertical.py、CONFIG.ocr.rotate 一致）、同一外部预处理
    （裁图原样读取，不做缩放/二值化；各自模型自带内部预处理与字典）
  - 原始输出先落盘，再从磁盘回读后加载人工标注做位置核对（避免标注泄漏进推理）

A 组候选 = 第一步原始 OCR + 固定窗口 v5 输出
B 组候选 = A 组 + 固定窗口 v3 输出（唯一实验变量）

用法：
  python tools/p51_step3_window_rec.py --windows <windows.json> --crops <crops目录> \
      --gold <gold_names.json> --baseline <第一步ocr.json> --out <新目录>
  python tools/p51_step3_window_rec.py --compare <目录A> <目录B>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.normalize import normalize_text      # noqa: E402
from ocr.recognize import _engine, _text_recognizer  # noqa: E402
from ocr.vertical import imread_unicode       # noqa: E402

EXPECTED_TABLE_HASH = "df31b25047b94b12eeb449f5298301864101d87c672c4ca796600d7f8eacfa2b"
MODELS = [
    ("v5", "模型权重/ch_PP-OCRv5_rec_mobile.onnx", "项目现用简体 v5 识别器"),
    ("v3_cht", "模型权重/chinese_cht_PP-OCRv3_rec_mobile.onnx", "繁体 v3 识别器"),
]
# 项目既有等价口径（ocr/normalize.py 的字表不含这两组，单独补齐，保留原始字形记录）
EQUIV = {"殼": "壳", "羗": "羌"}
TARGETS = ["柴胡", "川芎", "茯苓"]
TOL_OVERLAP = 0.5          # 字形区间与人工字形框的最低重叠比
COL_PAD = 2                # 末字列宽估计的兜底


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def eq_char(c: str) -> str:
    n = normalize_text(c)
    if len(n) != 1:
        n = c
    return EQUIV.get(n, n)


def eq_text(s: str) -> str:
    return "".join(eq_char(c) for c in s)


def load_windows(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf8"))
    h = data.get("table_hash")
    if h != EXPECTED_TABLE_HASH:
        raise SystemExit("窗口表指纹不符：%s" % h)
    bw = [w for w in data["windows"] if w["group"] == "B"]
    aw = [w for w in data["windows"] if w["group"] == "A"]
    if len(bw) != 414 or len(aw) != 20:
        raise SystemExit("窗口数与第二步不符：B=%d A=%d" % (len(bw), len(aw)))
    return bw


def infer_all(windows: list, crop_dir: Path) -> dict:
    """逐窗口 × 逐模型单张推理（batch=1，保证可重复）。"""
    engines = {}
    meta = {}
    for key, rel, desc in MODELS:
        p = ROOT / rel
        if not p.is_file():
            raise SystemExit("缺少模型权重：%s" % p)
        engines[key] = _text_recognizer(_engine(str(p)))
        meta[key] = {"key": key, "path": str(p), "sha256": sha256_of(p),
                     "bytes": p.stat().st_size, "desc": desc}
    rows = []
    for w in windows:
        crop_path = crop_dir / ("%s.png" % w["window_id"].replace(":", "_"))
        img = imread_unicode(str(crop_path))
        if img is None:
            for key, _r, _d in MODELS:
                rows.append({"window_id": w["window_id"], "line_id": w["line_id"],
                             "block_ref": w["block_ref"], "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]],
                             "model": key, "crop": str(crop_path), "crop_sha256": None,
                             "text": "", "score": None, "n_cols": None, "cols": None,
                             "char_probs": None, "error": "裁图读取失败", "empty": True})
            continue
        crop_sha = sha256_of(crop_path)
        rot = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        for key, _rel, _desc in MODELS:
            rec = {"window_id": w["window_id"], "line_id": w["line_id"],
                   "block_ref": w["block_ref"], "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]],
                   "model": key, "crop": str(crop_path), "crop_sha256": crop_sha,
                   "rotated_shape": list(rot.shape), "text": "", "score": None,
                   "n_cols": None, "words": None, "cols": None, "state": None,
                   "char_probs": None, "error": None, "empty": True}
            try:
                res, _t = engines[key]([rot], return_word_box=True)
                if not res:
                    rec["error"] = "模型返回空结果"
                else:
                    text, score, wb = res[0]
                    rec.update({"text": str(text), "score": round(float(score), 6)})
                    if wb and len(wb) >= 5:
                        rec.update({"n_cols": float(wb[0]), "words": wb[1], "cols": wb[2],
                                    "state": wb[3], "char_probs": [round(float(x), 6) for x in wb[4]]})
                    rec["empty"] = not rec["text"].strip()
            except Exception as exc:                     # 空输出与异常都逐条留痕
                rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                rec["empty"] = True
            rows.append(rec)
    return {"models": meta, "windows": rows}


def flatten_cols(rec: dict):
    """把 word_col_list 摊平成逐字列位置；长度与 text 不一致则返回 None。"""
    cols = rec.get("cols")
    if not cols:
        return None
    flat = [int(c) for group in cols for c in group]
    return flat if len(flat) == len(rec["text"]) else None


def span_y(w: dict, rec: dict, cols: list, i: int, n: int) -> tuple:
    """第 i..i+n-1 字 → 原图 y 区间（解码列位置／总列数 × 窗高）。窗高即旋转前高度。"""
    H = w["y2"] - w["y1"]
    nc = float(rec.get("n_cols") or 1.0)
    y1 = w["y1"] + cols[i] / nc * H
    j = i + n - 1
    if j + 1 < len(cols):
        wc = max(1, cols[j + 1] - cols[j])
    else:
        wc = COL_PAD
    y2 = w["y1"] + (cols[j] + wc) / nc * H
    return max(w["y1"], y1), min(w["y2"], y2)


def overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    ov = max(0.0, min(a2, b2) - max(a1, b1))
    denom = min(a2 - a1, b2 - b1)
    return ov / denom if denom > 0 else 0.0


def occurrences(rec: dict, w: dict, target: str, gold_boxes: list) -> list:
    """在一条原始输出里找目标药名的连续字串，并给出位置依据。"""
    text = rec["text"] or ""
    if not text:
        return []
    t = eq_text(text)
    tgt = eq_text(target)
    cols = flatten_cols(rec)
    out = []
    start = 0
    while True:
        i = t.find(tgt, start)
        if i < 0:
            break
        start = i + 1
        item = {"window_id": rec["window_id"], "model": rec["model"], "line_id": rec["line_id"],
                "block_ref": rec["block_ref"], "text": text, "start": i, "end": i + len(tgt),
                "slice": text[i:i + len(tgt)], "score": rec["score"]}
        if cols is not None:
            ya, yb = span_y(w, rec, cols, i, len(tgt))
            item["y_span"] = [round(ya, 1), round(yb, 1)]
            item["position_basis"] = "decoder_char_position"
            ratios = []
            for g in gold_boxes:
                if g["line_id"] != w["line_id"]:
                    continue
                gb = g["box"]
                ratios.append((g["name"], round(overlap_ratio(ya, yb, gb[1], gb[3]), 3)))
            ratios.sort(key=lambda x: -x[1])
            item["overlap"] = ratios
            best = ratios[0] if ratios else ("", 0.0)
            item["best_match"] = best[0]
            item["overlap_ratio"] = best[1]
            if best[0] != target or best[1] < TOL_OVERLAP:
                item["position_ok"] = False
                item["position_note"] = "位置存疑：最佳匹配=%s(%.2f)" % (best[0], best[1])
            elif len(ratios) > 1 and ratios[1][1] >= best[1] - 0.05:
                item["position_ok"] = False
                item["position_note"] = "位置存疑：与 %s 重叠接近" % ratios[1][0]
            else:
                item["position_ok"] = True
                item["position_note"] = "位置成立（与人工字形框重叠 %.2f）" % best[1]
        else:
            item["position_basis"] = "none"
            item["position_ok"] = False
            item["position_note"] = "解码器未给出字符位置，无法映射回原图"
        out.append(item)
    return out


def baseline_occurrences(baseline: dict, target: str, gold: dict) -> list:
    """第一步原始 OCR：整列文本，块级位置。

    块级位置必须与人工字形框在原图上相交才算成立——否则只是「正文别处出现过」，
    按规范不计入召回（例：R013 正文「川芎生地」、R012「用人参茯苓助正」）。
    """
    tgt = eq_text(target)
    gx1, gy1, gx2, gy2 = gold["box"]
    out = []
    for b in baseline["blocks"]:
        t = b.get("text") or ""
        idx = eq_text(t).find(tgt)
        if idx < 0:
            continue
        bx1, by1, bx2, by2 = b["bbox"]
        ox = min(bx2, gx2) - max(bx1, gx1)
        oy = min(by2, gy2) - max(by1, gy1)
        ok = ox > 0 and oy > 0
        out.append({"window_id": "A:%s" % b["line_id"], "model": "baseline_ocr",
                    "line_id": b["line_id"], "block_ref": b["block_ref"], "text": t,
                    "start": idx, "end": idx + len(tgt), "slice": t[idx:idx + len(tgt)],
                    "score": b.get("conf"), "bbox": b["bbox"],
                    "position_basis": "block_level",
                    "position_ok": bool(ok),
                    "position_note": ("块级位置成立：块 bbox 与人工字形框在原图相交"
                                      if ok else
                                      "位置不符：该读法出现在正文别的列（%s），非方剂位置，不计入"
                                      % b["line_id"])})
    return out


def competitors(recs: list, target: str, gold_box: list, wmap: dict) -> list:
    """收集与目标同首字、位置落在该药名附近的 2 字竞争读法（如 茯苓/茯茶/茯芬）。"""
    out = []
    for rec in recs:
        w = wmap.get(rec["window_id"])
        if w is None or rec["line_id"] != gold_box["line_id"] or not rec["text"]:
            continue
        cols = flatten_cols(rec)
        if cols is None:
            continue
        text = rec["text"]
        for i in range(len(text) - 1):
            pair = text[i:i + 2]
            if eq_char(pair[0]) != eq_char(target[0]) or eq_text(pair) == eq_text(target):
                continue
            ya, yb = span_y(w, rec, cols, i, 2)
            r = overlap_ratio(ya, yb, gold_box["box"][1], gold_box["box"][3])
            if r >= 0.3:
                out.append({"reading": pair, "model": rec["model"], "window_id": rec["window_id"],
                            "y_span": [round(ya, 1), round(yb, 1)], "overlap": round(r, 3),
                            "score": rec["score"]})
    # 去重（同读法同窗口同模型只留一条）
    seen, uniq = set(), []
    for x in out:
        k = (x["reading"], x["window_id"], x["model"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        a = json.loads((args.compare[0] / "score.json").read_text(encoding="utf8"))
        b = json.loads((args.compare[1] / "score.json").read_text(encoding="utf8"))
        same = (a["A"]["recall"] == b["A"]["recall"] and a["B"]["recall"] == b["B"]["recall"]
                and a["sources"] == b["sources"])
        print(json.dumps({"A": [a["A"]["recall"], b["A"]["recall"]],
                          "B": [a["B"]["recall"], b["B"]["recall"]],
                          "sources_identical": a["sources"] == b["sources"],
                          "raw_sha_identical": a.get("raw_sha256") == b.get("raw_sha256"),
                          "identical": same}, ensure_ascii=False))
        return 0 if same else 1

    if not (args.windows and args.crops and args.gold and args.baseline and args.out):
        raise SystemExit("需要 --windows/--crops/--gold/--baseline/--out")
    args.out.mkdir(parents=True, exist_ok=False)

    windows = load_windows(args.windows)
    missing = [w for w in windows
               if not (args.crops / ("%s.png" % w["window_id"].replace(":", "_"))).is_file()]
    if missing:
        raise SystemExit("缺少裁图 %d 张，例如 %s" % (len(missing), missing[0]["window_id"]))

    # ---- 推理阶段：只读窗口与原始 OCR；原始输出先落盘 ----
    raw = infer_all(windows, args.crops)
    raw.update({"image": str((ROOT / "古籍评测集/医家类/思濟堂方書/page_51.png")),
                "image_sha256": sha256_of(ROOT / "古籍评测集/医家类/思濟堂方書/page_51.png"),
                "windows_json": str(args.windows.resolve()),
                "windows_table_hash": EXPECTED_TABLE_HASH,
                "n_windows": len(windows),
                "rotation": "cv2.ROTATE_90_COUNTERCLOCKWISE（与 CONFIG.ocr.rotate=90 一致）",
                "preprocess": "裁图原样读取（BGR），不缩放/不二值化；模型各自内部预处理与字典",
                "batch": 1, "return_word_box": True,
                "note": "score 为模型对该条输出的整体均值分，不是某字置信度；char_probs 为解码序列逐位概率，未校准"})
    raw_path = args.out / "raw_outputs.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    raw_sha = sha256_of(raw_path)
    print(json.dumps({"raw": str(raw_path), "raw_sha256": raw_sha,
                      "inferences": len(raw["windows"])}, ensure_ascii=False))

    # ---- 核对阶段：从磁盘回读原始输出，再加载人工标注 ----
    raw = json.loads(raw_path.read_text(encoding="utf8"))
    gold = json.loads(args.gold.read_text(encoding="utf8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf8"))
    gold_drugs = gold["drugs"]
    wmap = {w["window_id"]: w for w in windows}

    by_model = {"v5": [], "v3_cht": []}
    for rec in raw["windows"]:
        by_model.setdefault(rec["model"], []).append(rec)

    per_drug = {}
    for g in gold_drugs:
        name = g["name"]
        hits = {"baseline_ocr": baseline_occurrences(baseline, name, g),
                "v5": [], "v3_cht": []}
        for key in ("v5", "v3_cht"):
            for rec in by_model[key]:
                w = wmap[rec["window_id"]]
                hits[key].extend(occurrences(rec, w, name, gold_drugs))
        per_drug[name] = {
            "line_id": g["line_id"], "box": g["box"],
            "baseline_ok": any(h["position_ok"] for h in hits["baseline_ocr"]),
            "v5_ok": any(h["position_ok"] for h in hits["v5"]),
            "v3_ok": any(h["position_ok"] for h in hits["v3_cht"]),
            "hits": {k: v for k, v in hits.items()},
            "competitors": (competitors(by_model["v5"], name, g, wmap)
                            + competitors(by_model["v3_cht"], name, g, wmap)),
        }

    # A = 原始 OCR ∪ v5；B = A ∪ v3
    for name, d in per_drug.items():
        d["A_ok"] = d["baseline_ok"] or d["v5_ok"]
        d["B_ok"] = d["A_ok"] or d["v3_ok"]
        # 未召回时区分「位置可定位但读错」与「根本没有该读法」
        pos_comp = [c for c in d["competitors"] if c["overlap"] >= TOL_OVERLAP]
        if d["B_ok"]:
            d["status"] = "recovered"
        elif pos_comp:
            d["status"] = "read_but_wrong"
        else:
            d["status"] = "not_read"
        d["top_readings"] = sorted(pos_comp, key=lambda c: -c["overlap"])[:6]

    def count(field, names):
        return sum(1 for n in names if per_drug[n][field])

    a_recall = count("A_ok", TARGETS)
    b_recall = count("B_ok", TARGETS)
    sources = {n: {"A_ok": per_drug[n]["A_ok"], "B_ok": per_drug[n]["B_ok"],
                   "baseline": per_drug[n]["baseline_ok"], "v5": per_drug[n]["v5_ok"],
                   "v3": per_drug[n]["v3_ok"]} for n in TARGETS}
    kept8 = [n for n in per_drug if n not in TARGETS]
    kept8_ok = [n for n in kept8 if per_drug[n]["B_ok"]]
    kept8_by_baseline = [n for n in kept8 if per_drug[n]["baseline_ok"]]

    score = {
        "metric": "第51页三处误读的正确候选召回数", "denominator": 3,
        "targets": TARGETS,
        "A": {"recall": a_recall, "total": 3, "composition": "第一步原始OCR + 固定窗口v5"},
        "B": {"recall": b_recall, "total": 3, "composition": "A + 固定窗口繁体v3"},
        "delta": b_recall - a_recall,
        "v5_alone": [n for n in TARGETS if per_drug[n]["v5_ok"]],
        "v3_alone": [n for n in TARGETS if per_drug[n]["v3_ok"]],
        "baseline_alone": [n for n in TARGETS if per_drug[n]["baseline_ok"]],
        "status": {n: {"status": per_drug[n]["status"],
                       "v5_readings": [c["reading"] for c in per_drug[n]["top_readings"]
                                       if c["model"] == "v5"],
                       "v3_readings": [c["reading"] for c in per_drug[n]["top_readings"]
                                       if c["model"] == "v3_cht"],
                       "confirmed_hits": [
                           {"model": h["model"], "window_id": h["window_id"], "slice": h["slice"],
                            "y_span": h.get("y_span"), "overlap": h.get("overlap_ratio"),
                            "score": h["score"]}
                           for k in ("baseline_ocr", "v5", "v3_cht")
                           for h in per_drug[n]["hits"][k] if h["position_ok"]],
                       "note": per_drug[n]["hits"]["baseline_ocr"] and
                               [h["position_note"] for h in per_drug[n]["hits"]["baseline_ocr"]
                                if not h["position_ok"]] or []} for n in TARGETS},
        "sources": sources,
        "kept_8": {"total": len(kept8), "ok": len(kept8_ok), "names_ok": kept8_ok,
                   "by_baseline_only": [n for n in kept8_by_baseline
                                        if not per_drug[n]["v5_ok"] and not per_drug[n]["v3_ok"]]},
        "models": raw["models"], "raw_sha256": raw_sha,
        "windows_table_hash": EXPECTED_TABLE_HASH,
        "n_windows": len(windows),
        "empty_or_error": sum(1 for r in raw["windows"] if r["empty"] or r["error"]),
    }
    (args.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (args.out / "candidates.json").write_text(json.dumps(per_drug, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf8")
    print(json.dumps({"A": a_recall, "B": b_recall, "delta": b_recall - a_recall,
                      "v5": score["v5_alone"], "v3": score["v3_alone"],
                      "baseline": score["baseline_alone"],
                      "kept8": score["kept_8"]["ok"], "empty_or_error": score["empty_or_error"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
