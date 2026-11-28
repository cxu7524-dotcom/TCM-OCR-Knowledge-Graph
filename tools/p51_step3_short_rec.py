# -*- coding: utf-8 -*-
"""第 51 页 2 字高短窗识别与候选评分（第 3 步短窗补测，第二步）。

唯一变量：窗口高度方案（4H → 2H）。其余全部沿用 ID 10：
权重、字表、90° 逆时针、外部预处理、batch=1、beam=25、每步 top-8 类别+空白、
max_prefix=30、每窗每模型前 5 条去重文本、位置计分口径。

A 组 = ID 10 冻结候选（ctc_candidates.jsonl）
B 组 = A 组 + 本轮 489 张短窗的候选

候选落盘后才读人工标注。

用法：
  python tools/p51_step3_short_rec.py --short-windows <short_windows.json> --crops <目录> \
      --baseline-cands <ID10 ctc_candidates.jsonl> --baseline-score <ID10 score.json> \
      --gold <gold_names.json> --out <新目录>
  python tools/p51_step3_short_rec.py --compare <目录A> <目录B>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.recognize import _engine, _text_recognizer  # noqa: E402
from ocr.vertical import imread_unicode             # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_ctc", ROOT / "tools/p51_step3_ctc_candidates.py")
_ctc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ctc)

MODELS = _ctc.MODELS
TARGETS = _ctc.TARGETS
TOL = _ctc.TOL_OVERLAP
eq_text = _ctc.eq_text
ctc_beam = _ctc.ctc_beam
content_steps = _ctc.content_steps
greedy_alignment = _ctc.greedy_alignment
span_from_alignment = _ctc.span_from_alignment
cmpr = _ctc._s3.overlap_ratio


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def position_ok(w: dict, y1: float, y2: float, target: str, gold: list):
    """沿用 ID 10 口径：同列字形框里取重叠最高者，必须是本味且 ≥0.5；与亚军接近则记存疑。"""
    ratios = []
    for g in gold:
        if g["line_id"] != w["line_id"]:
            continue
        ratios.append((g["name"], cmpr(y1, y2, g["box"][1], g["box"][3])))
    ratios.sort(key=lambda x: -x[1])
    if not ratios:
        return False, "同列无人工字形框", []
    best = ratios[0]
    ok = best[0] == target and best[1] >= TOL
    if ok and len(ratios) > 1 and ratios[1][1] >= best[1] - 0.05:
        return False, "位置存疑：与 %s 重叠接近" % ratios[1][0], ratios
    return ok, ("位置成立（重叠 %.2f）" % best[1] if ok
                else "位置不符：最佳匹配 %s(%.2f)" % best), ratios


def score_set(cands: list, wmap: dict, gold: list) -> dict:
    """对一组候选按同一口径算逐味命中。"""
    per = {}
    for g in gold:
        name = g["name"]
        tgt = eq_text(name)
        hits = []
        for c in cands:
            if tgt not in eq_text(c["text"]):
                continue
            w = wmap.get(c["window_id"])
            if w is None:
                continue
            T = c.get("n_content_steps") or c["n_steps"]
            t = eq_text(c["text"])
            i = t.find(tgt)
            while i >= 0:
                if len(c["alignment"]) != len(c["text"]):
                    ok, note, y_span = False, "对齐长度与文本不一致，无法回映", None
                else:
                    ya, yb = span_from_alignment(w, c["alignment"], i, len(tgt), T)
                    ok, note, _r = position_ok(w, ya, yb, name, gold)
                    y_span = [round(ya, 1), round(yb, 1)]
                hits.append({"window_id": c["window_id"], "model": c["model"],
                             "rank": c["rank"], "text": c["text"],
                             "slice": c["text"][i:i + len(tgt)],
                             "y_span": y_span, "seq_logprob": c.get("seq_logprob"),
                             "ok": bool(ok), "note": note})
                i = t.find(tgt, i + 1)
        okh = [h for h in hits if h["ok"]]
        per[name] = {"line_id": g["line_id"], "box": g["box"], "ok": bool(okh),
                     "n_hits_ok": len(okh),
                     # 不截断，否则短窗命中会被 4H 窗命中挤掉，导致来源统计失真
                     "hits_ok": okh[:200],
                     "hits_bad": [h for h in hits if not h["ok"]][:6]}
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-windows", type=Path)
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--baseline-cands", type=Path)
    ap.add_argument("--baseline-score", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        ha = sha256_of(args.compare[0] / "short_candidates.jsonl")
        hb = sha256_of(args.compare[1] / "short_candidates.jsonl")
        na = sum(1 for _ in (args.compare[0] / "short_candidates.jsonl").open(encoding="utf8"))
        nb = sum(1 for _ in (args.compare[1] / "short_candidates.jsonl").open(encoding="utf8"))
        sa = json.loads((args.compare[0] / "score.json").read_text(encoding="utf8"))
        sb = json.loads((args.compare[1] / "score.json").read_text(encoding="utf8"))
        same = (sa["A"]["recall"] == sb["A"]["recall"] and sa["B"]["recall"] == sb["B"]["recall"]
                and sa["sources"] == sb["sources"])
        print(json.dumps({"A": [sa["A"]["recall"], sb["A"]["recall"]],
                          "B": [sa["B"]["recall"], sb["B"]["recall"]],
                          "candidates": [na, nb], "cand_sha_identical": ha == hb,
                          "sources_identical": sa["sources"] == sb["sources"],
                          "identical": ha == hb and same}, ensure_ascii=False))
        return 0 if (ha == hb and same) else 1

    if not (args.short_windows and args.crops and args.baseline_cands
            and args.baseline_score and args.gold and args.out):
        raise SystemExit("参数不全")
    args.out.mkdir(parents=True, exist_ok=True)   # 可与短窗产物同目录（文件名不重叠）

    sw = json.loads(args.short_windows.read_text(encoding="utf8"))
    windows = sw["windows"]
    wmap = {w["window_id"]: w for w in windows}
    base_score = json.loads(args.baseline_score.read_text(encoding="utf8"))

    engines, meta = {}, {}
    for key, rel, desc in MODELS:
        p = ROOT / rel
        engines[key] = _text_recognizer(_engine(str(p)))
        meta[key] = {"key": key, "path": str(p), "bytes": p.stat().st_size,
                     "sha256": sha256_of(p), "desc": desc,
                     "charset_size": len(getattr(engines[key].postprocess_op, "character", []))}

    rows, cands, dump = [], [], {}
    for w in windows:
        crop = Path(w["crop_path"])
        img = imread_unicode(str(crop))
        if img is None:
            raise SystemExit("裁图读取失败：%s" % crop)
        rot = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        for key, _rel, _desc in MODELS:
            rec = engines[key]
            item = {"window_id": w["window_id"], "model": key, "text": "", "score": None,
                    "error": None, "empty": True, "crop_sha256": sha256_of(crop),
                    "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]]}
            try:
                with _ctc.PredSpy(rec) as spy:
                    res, _t = rec([rot], return_word_box=True)
                preds = spy.preds[0]
                T, C = preds.shape
                cs = content_steps(rot.shape, tuple(getattr(rec, "rec_image_shape", (3, 48, 320))), T)
                text0, score0 = (res[0][0], float(res[0][1])) if res else ("", 0.0)
                chars = list(getattr(rec.postprocess_op, "character", []))
                gtext, galign = greedy_alignment(preds, chars)
                ranked = ctc_beam(preds, chars, _ctc.BEAM_WIDTH, _ctc.TOPK_PER_STEP,
                                  _ctc.MAX_PREFIX)
                item.update({"text": text0, "score": round(score0, 6), "empty": not text0.strip(),
                             "n_steps": int(T), "n_classes": int(C),
                             "content_steps": round(cs, 3),
                             "greedy_text_from_preds": gtext,
                             "greedy_matches_interface": gtext == text0})
                rows.append(item)
                for r, (sc, ctext, align) in enumerate(ranked[:_ctc.TOP_N], start=1):
                    cands.append({"model": key, "window_id": w["window_id"], "rank": r,
                                  "text": ctext, "seq_logprob": round(sc, 6),
                                  "seq_prob": float(np.exp(sc)),
                                  "n_steps": int(T), "n_content_steps": round(cs, 3),
                                  "alignment": [[c, t] for c, t in align],
                                  "greedy_text": text0,
                                  "avg_score_interface": round(score0, 6),
                                  "line_id": w["line_id"], "block_ref": w["block_ref"],
                                  "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]],
                                  "window_group": "S", "window_height": w["height"]})
                if w["window_id"] in ("S:R019:W10", "S:R019:W11", "S:R019:W12",
                                      "S:R019:W13", "S:R019:W14", "S:R019:W15",
                                      "S:R020:W10", "S:R020:W11", "S:R020:W12"):
                    dump["%s|%s" % (w["window_id"], key)] = preds
            except Exception as exc:
                item["error"] = "%s: %s" % (type(exc).__name__, exc)
                rows.append(item)

    cfg = {"reuse": "tools/p51_step3_ctc_candidates.py 的推理与解码逻辑（PredSpy/ctc_beam/content_steps）",
           "beam_width": _ctc.BEAM_WIDTH, "topk_per_step": _ctc.TOPK_PER_STEP,
           "max_prefix": _ctc.MAX_PREFIX, "top_n": _ctc.TOP_N, "sort_rule": _ctc.SORT_RULE,
           "only_variable_vs_id10": "窗口高度 4H -> 2H",
           "short_windows": str(args.short_windows.resolve()),
           "short_table_hash": sw["table_hash"], "n_windows": len(windows),
           "baseline_cands": str(args.baseline_cands.resolve()),
           "baseline_cands_sha256": sha256_of(args.baseline_cands),
           "baseline_score": str(args.baseline_score.resolve()),
           "baseline_score_sha256": sha256_of(args.baseline_score),
           "baseline_ledger_id": 10, "models": meta}
    (args.out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf8")
    with (args.out / "short_candidates.jsonl").open("w", encoding="utf8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (args.out / "short_raw.json").write_text(json.dumps(
        {"rows": rows, "empty_or_error": sum(1 for r in rows if r["empty"] or r["error"]),
         "greedy_mismatch": sum(1 for r in rows if not r.get("greedy_matches_interface", False))},
        ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    if dump:
        np.savez_compressed(args.out / "preds_short_targets.npz", **dump)
    cand_sha = sha256_of(args.out / "short_candidates.jsonl")
    print(json.dumps({"windows": len(windows), "inferences": len(rows),
                      "candidates": len(cands), "empty_or_error": sum(1 for r in rows if r["empty"] or r["error"]),
                      "cand_sha256": cand_sha[:16], "preds_dumped": len(dump)}, ensure_ascii=False))

    # ---- 评分：候选已落盘 ----
    gold = json.loads(args.gold.read_text(encoding="utf8"))["drugs"]
    base_cands = [json.loads(l) for l in args.baseline_cands.open(encoding="utf8")]
    # A 组窗口表沿用第二步（4H 窗），B 组额外要有短窗
    base_wd = json.loads((ROOT / "output/p51_step2_20261122_183000/windows.json").read_text(encoding="utf8"))
    all_wmap = {w["window_id"]: w for w in base_wd["windows"]}
    all_wmap.update(wmap)

    A = score_set(base_cands, all_wmap, gold)
    B = score_set(base_cands + cands, all_wmap, gold)
    a_recall = sum(1 for n in TARGETS if A[n]["ok"])
    b_recall = sum(1 for n in TARGETS if B[n]["ok"])
    short_only = [n for n in TARGETS if B[n]["ok"] and not A[n]["ok"]]
    sources = {}
    for n in TARGETS:
        sources[n] = {"A_ok": A[n]["ok"], "B_ok": B[n]["ok"],
                      "from_short": [h for h in B[n]["hits_ok"]
                                     if h["window_id"].startswith("S:")][:4],
                      "inherited": [h for h in A[n]["hits_ok"]][:3]}
    kept8 = [n for n in B if n not in TARGETS]
    kept8_ok = [n for n in kept8 if B[n]["ok"]]

    score = {"metric": "第51页三处误读的正确候选召回数", "denominator": 3,
             "stage": "第3步短窗补测（4H -> 2H）",
             "control": {"ledger_id": 10, "baseline_A": base_score["A"]["recall"],
                         "baseline_B": base_score["B"]["recall"],
                         "baseline_cands_sha256": cfg["baseline_cands_sha256"],
                         "replayed_A_from_frozen": a_recall,
                         "replay_matches_baseline": a_recall == base_score["B"]["recall"]},
             "A": {"recall": a_recall, "total": 3, "composition": "ID10 冻结候选"},
             "B": {"recall": b_recall, "total": 3, "composition": "A + 489 张 2H 短窗候选"},
             "delta": b_recall - a_recall,
             "short_only": short_only,
             "sources": sources,
             "kept_8": {"total": len(kept8), "ok": len(kept8_ok), "names_ok": kept8_ok},
             "n_short_windows": len(windows), "n_short_candidates": len(cands),
             "short_candidates_sha256": cand_sha,
             "config": cfg}
    (args.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf8")
    (args.out / "per_drug.json").write_text(json.dumps({"A": A, "B": B}, ensure_ascii=False,
                                                       indent=2) + "\n", encoding="utf8")
    print(json.dumps({"A": a_recall, "B": b_recall, "short_only": short_only,
                      "kept8": len(kept8_ok),
                      "replay_matches_id10": a_recall == base_score["B"]["recall"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
