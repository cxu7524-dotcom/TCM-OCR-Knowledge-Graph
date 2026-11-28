# -*- coding: utf-8 -*-
"""第 51 页固定窗口：CTC 多候选解码补测（第 3 步补测）。

唯一改动：在模型 ONNX 前向输出进入 CTC 解码**之前**截获逐时间步分布，
用同一份分布做前缀束搜索（保留 CTC 空白与重复字规则），取前 5 条去重文本候选。
权重、字表、90° 逆时针旋转、外部预处理、batch=1 全部沿用 tools/p51_step3_window_rec.py。

A 组 = 账本 ID 9 的候选集合（原始 OCR + v5 原解码 + 繁体 v3 原解码）
B 组 = A 组 + 同一批窗口的 CTC 前 5 条候选（本轮唯一变量）

候选生成阶段只用模型输出与模型自带字表；人工药名/方剂组成/药名词表只在落盘后的评分阶段使用。

用法：
  python tools/p51_step3_ctc_candidates.py --windows <windows.json> --crops <目录> \
      --baseline-raw <ID9 raw_outputs.json> --baseline-ocr <第一步 ocr.json> \
      --gold <gold_names.json> --out <新目录>
  python tools/p51_step3_ctc_candidates.py --compare <目录A> <目录B>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.recognize import _engine, _text_recognizer  # noqa: E402
from ocr.vertical import imread_unicode             # noqa: E402

# 复用第 3 步的口径与等价规则，避免两套实现漂移
_spec = importlib.util.spec_from_file_location("p51_step3", ROOT / "tools/p51_step3_window_rec.py")
_s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s3)

EXPECTED_TABLE_HASH = _s3.EXPECTED_TABLE_HASH
MODELS = _s3.MODELS
TARGETS = _s3.TARGETS
TOL_OVERLAP = _s3.TOL_OVERLAP
eq_text = _s3.eq_text

# 候选生成参数（运行前固定，写入 run_config.json）
BEAM_WIDTH = 25        # 束宽
TOPK_PER_STEP = 8      # 逐时间步类别剪枝：每步只保留概率最高的 8 个非空白类别 + 空白
MAX_PREFIX = 30        # 前缀最大长度（防长尾）
TOP_N = 5              # 最终去重候选数上限
SORT_RULE = "按 (-序列分数, 文本字典序) 稳定排序；同文本保留分数最高的一条"


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class PredSpy:
    """把 TextRecognizer 交给 postprocess 的 preds 截获下来，不改变解码行为。"""

    def __init__(self, rec):
        self.rec = rec
        self.orig = rec.postprocess_op
        self.preds = None

    def __enter__(self):
        spy = self

        class Wrap:
            def __call__(self, preds, return_word_box=False, **kw):
                spy.preds = np.array(preds, copy=True)
                return spy.orig(preds, return_word_box, **kw)

        self.rec.postprocess_op = Wrap()
        return self

    def __exit__(self, *exc):
        self.rec.postprocess_op = self.orig
        return False


def ctc_beam(probs: np.ndarray, chars: list, beam: int, topk: int, max_len: int) -> list:
    """CTC 前缀束搜索。probs: (T, C)，已含空白类别 0。返回 [(log_score, text, alignment)]。

    alignment 为该前缀最优对齐路线的 (字符下标, 时间步) 列表，用于回映原图。
    """
    T, C = probs.shape
    lp = np.log(np.clip(probs, 1e-12, None))
    # 每步候选类别（含空白），稳定排序：概率降序，同概率按类别号升序
    step_idx = []
    for t in range(T):
        idx = np.argpartition(-probs[t], min(topk + 0, C - 1))[:topk]
        idx = sorted(set([0] + [int(i) for i in idx]),
                     key=lambda c: (-float(probs[t, c]), c))
        step_idx.append(idx)

    # state: prefix -> {"pb": 空白结尾概率, "pnb": 非空白结尾, "ab": 对齐, "anb": 对齐, "mb","mnb": 各自最大贡献}
    beams = {(): {"pb": 1.0, "pnb": 0.0, "ab": (), "anb": (), "mb": 1.0, "mnb": 0.0}}
    for t in range(T):
        new = {}

        def add(prefix, blank, val, align):
            s = new.get(prefix)
            if s is None:
                s = {"pb": 0.0, "pnb": 0.0, "ab": (), "anb": (), "mb": 0.0, "mnb": 0.0}
                new[prefix] = s
            if blank:
                s["pb"] += val
                if val > s["mb"]:
                    s["mb"], s["ab"] = val, align
            else:
                s["pnb"] += val
                if val > s["mnb"]:
                    s["mnb"], s["anb"] = val, align

        for prefix, s in beams.items():
            tot = s["pb"] + s["pnb"]
            if tot <= 0:
                continue
            # 1) 走空白：前缀不变
            best_align = s["ab"] if s["pb"] >= s["pnb"] else s["anb"]
            add(prefix, True, tot * probs[t, 0], best_align)
            # 2) 输出非空白字符
            if len(prefix) >= max_len:
                continue
            for c in step_idx[t]:
                if c == 0:
                    continue
                pc = probs[t, c]
                if pc <= 0:
                    continue
                if prefix and prefix[-1] == c:
                    # 同字重复必须隔着空白：只能从「空白结尾」状态延伸
                    if s["pb"] > 0:
                        add(prefix + (c,), False, s["pb"] * pc, s["ab"] + (t,))
                else:
                    add(prefix + (c,), False, tot * pc, best_align + (t,))
        # 剪枝：按总概率保留 beam 条（稳定排序）
        ranked = sorted(new.items(), key=lambda kv: (-(kv[1]["pb"] + kv[1]["pnb"]), kv[0]))[:beam]
        beams = dict(ranked)

    out = []
    for prefix, s in beams.items():
        if not prefix:
            continue
        score = s["pb"] + s["pnb"]
        align = s["anb"] if s["pnb"] >= s["pb"] else s["ab"]
        text = "".join(chars[c] for c in prefix)
        out.append((float(np.log(max(score, 1e-12))), text, list(zip([int(c) for c in prefix],
                                                                   [int(t) for t in align]))))
    # 同文本去重，保留分数最高；稳定排序
    best = {}
    for sc, text, align in out:
        if text not in best or sc > best[text][0]:
            best[text] = (sc, align)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(sc, text, align) for text, (sc, align) in ranked]


def greedy_alignment(preds: np.ndarray, chars: list) -> tuple:
    """从同一份分布做贪心解码，并给出逐字时间步（与原接口文本应一致）。"""
    idx = preds.argmax(axis=1)
    prev, text, align = -1, [], []
    for t, c in enumerate(idx):
        c = int(c)
        if c != 0 and c != prev:
            text.append(chars[c])
            align.append((c, int(t)))
        prev = c
    return "".join(text), align


def content_steps(rot_shape: tuple, rec_shape: tuple, T: int) -> float:
    """时间步 → 像素的正确分母。

    识别器把裁图缩放到 rec_img_shape 高，再按 max(6.67, 宽高比) 补零到固定宽度；
    ONNX 输出的 T 个时间步覆盖的是**补零后**的宽度，内容只占前 resized/padded 份。
    直接用 T 做分母会把位置整体压向窗口顶部（实测偏差约 46px）。
    """
    img_h, img_w = rec_shape[1], rec_shape[2]
    h, w = rot_shape[0], rot_shape[1]
    ratio = w / float(h)
    max_ratio = max(img_w / float(img_h), ratio)
    padded = int(img_h * max_ratio)
    resized = min(padded, int(math.ceil(img_h * ratio)))
    return max(1.0, T * resized / float(padded))


def span_from_alignment(w: dict, alignment: list, i: int, n: int, T: int) -> tuple:
    """候选内第 i..i+n-1 字 → 原图 y 区间（时间步 / 总步数 × 窗高）。"""
    H = w["y2"] - w["y1"]
    t0 = alignment[i][1]
    last = i + n - 1
    if last + 1 < len(alignment):
        step = max(1, alignment[last + 1][1] - alignment[last][1])
    elif i > 0:
        step = max(1, alignment[i][1] - alignment[i - 1][1])
    else:
        step = 2
    y1 = w["y1"] + t0 / T * H
    y2 = w["y1"] + (alignment[last][1] + step) / T * H
    return max(w["y1"], y1), min(w["y2"], y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--baseline-raw", type=Path)
    ap.add_argument("--baseline-ocr", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        na = sum(1 for _ in (args.compare[0] / "ctc_candidates.jsonl").open(encoding="utf8"))
        nb = sum(1 for _ in (args.compare[1] / "ctc_candidates.jsonl").open(encoding="utf8"))
        ha = sha256_of(args.compare[0] / "ctc_candidates.jsonl")
        hb = sha256_of(args.compare[1] / "ctc_candidates.jsonl")
        sa = json.loads((args.compare[0] / "score.json").read_text(encoding="utf8"))
        sb = json.loads((args.compare[1] / "score.json").read_text(encoding="utf8"))
        same_score = (sa["A"]["recall"] == sb["A"]["recall"] and sa["B"]["recall"] == sb["B"]["recall"]
                      and sa["per_target"] == sb["per_target"])
        print(json.dumps({"A": [sa["A"]["recall"], sb["A"]["recall"]],
                          "B": [sa["B"]["recall"], sb["B"]["recall"]],
                          "candidates": [na, nb], "candidates_sha_identical": ha == hb,
                          "score_identical": same_score,
                          "identical": ha == hb and same_score}, ensure_ascii=False))
        return 0 if (ha == hb and same_score) else 1

    if not (args.windows and args.crops and args.baseline_raw and args.baseline_ocr
            and args.gold and args.out):
        raise SystemExit("需要 --windows/--crops/--baseline-raw/--baseline-ocr/--gold/--out")
    args.out.mkdir(parents=True, exist_ok=False)

    wd = json.loads(args.windows.read_text(encoding="utf8"))
    if wd.get("table_hash") != EXPECTED_TABLE_HASH:
        raise SystemExit("窗口几何指纹不符：%s" % wd.get("table_hash"))
    windows = [w for w in wd["windows"] if w["group"] == "B"]
    if len(windows) != 414:
        raise SystemExit("B 组窗口数不是 414：%d" % len(windows))
    wmap = {w["window_id"]: w for w in windows}

    base_raw = json.loads(args.baseline_raw.read_text(encoding="utf8"))
    base_sha = sha256_of(args.baseline_raw)
    base_map = {(r["window_id"], r["model"]): r for r in base_raw["windows"]}

    engines, meta = {}, {}
    for key, rel, desc in MODELS:
        p = ROOT / rel
        if not p.is_file():
            raise SystemExit("缺少权重：%s" % p)
        engines[key] = _text_recognizer(_engine(str(p)))
        meta[key] = {"key": key, "path": str(p), "bytes": p.stat().st_size,
                     "sha256": sha256_of(p), "desc": desc,
                     "charset_size": len(getattr(engines[key].postprocess_op, "character", []))}

    # ---- 输出阶段：截获分布 + 原解码复现（同一遍前向）----
    rows, cands, mismatch, preds_dump = [], [], [], {}
    for w in windows:
        crop = args.crops / ("%s.png" % w["window_id"].replace(":", "_"))
        img = imread_unicode(str(crop))
        if img is None:
            raise SystemExit("裁图读取失败：%s" % crop)
        rot = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        for key, _rel, _desc in MODELS:
            rec = engines[key]
            with PredSpy(rec) as spy:
                res, _t = rec([rot], return_word_box=True)
            preds = spy.preds[0]                      # (T, C)
            T, C = preds.shape
            cs = content_steps(rot.shape, tuple(getattr(rec, "rec_image_shape", (3, 48, 320))), T)
            text0, score0 = (res[0][0], float(res[0][1])) if res else ("", 0.0)
            b = base_map.get((w["window_id"], key))
            same = bool(b) and (b["text"], round(float(b["score"]), 6)) == (text0, round(score0, 6))
            if not same:
                mismatch.append({"window_id": w["window_id"], "model": key,
                                 "baseline": (b["text"] if b else None,
                                              b["score"] if b else None),
                                 "now": (text0, round(score0, 6))})
            chars = list(getattr(rec.postprocess_op, "character", []))
            gtext, galign = greedy_alignment(preds, chars)
            rows.append({"window_id": w["window_id"], "model": key,
                         "text": text0, "score": round(score0, 6),
                         "greedy_text_from_preds": gtext,
                         "greedy_matches_interface": gtext == text0,
                         "greedy_alignment": [[c, t] for c, t in galign],
                         "n_steps": int(T), "n_classes": int(C),
                         "content_steps": round(cs, 3),
                         "matches_id9": same,
                         "crop": str(crop), "crop_sha256": sha256_of(crop)})
            ranked = ctc_beam(preds, chars, BEAM_WIDTH, TOPK_PER_STEP, MAX_PREFIX)
            for r, (sc, ctext, align) in enumerate(ranked[:TOP_N], start=1):
                cands.append({"model": key, "window_id": w["window_id"], "rank": r,
                              "text": ctext, "seq_logprob": round(sc, 6),
                              "seq_prob": float(np.exp(sc)),
                              "n_steps": int(T), "n_content_steps": round(cs, 3),
                              "alignment": [[c, t] for c, t in align],
                              "greedy_text": text0,
                              "avg_score_interface": round(score0, 6),
                              "line_id": w["line_id"], "block_ref": w["block_ref"],
                              "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]]})
            # 三处目标窗口的分布留档，便于复查
            if w["window_id"] in ("B:R020:W12", "B:R020:W13", "B:R019:W08",
                                  "B:R019:W09", "B:R019:W02"):
                preds_dump["%s|%s" % (w["window_id"], key)] = preds

    cfg = {"beam_width": BEAM_WIDTH, "topk_per_step": TOPK_PER_STEP,
           "max_prefix": MAX_PREFIX, "top_n": TOP_N, "sort_rule": SORT_RULE,
           "ctc_rules": "空白类别 0；同字重复须隔空白才另起一字",
           "stage": "只有在 ONNX 前向后、CTC 解码前截获的逐时间步分布上做候选生成",
           "score_semantics": "seq_logprob=整条候选的CTC序列对数概率；"
                              "avg_score_interface=原接口对贪心解码的均值分，两者含义不同，不混用",
           "windows_json": str(args.windows.resolve()),
           "windows_table_hash": wd["table_hash"],
           "baseline_raw": str(args.baseline_raw.resolve()),
           "baseline_raw_sha256": base_sha,
           "n_windows": len(windows), "models": meta}
    (args.out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf8")
    with (args.out / "ctc_candidates.jsonl").open("w", encoding="utf8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (args.out / "greedy_replay.json").write_text(json.dumps(
        {"rows": rows, "mismatch": mismatch,
         "identical_to_id9": len(mismatch) == 0}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf8")
    np.savez_compressed(args.out / "preds_target_windows.npz", **preds_dump)
    cand_sha = sha256_of(args.out / "ctc_candidates.jsonl")
    print(json.dumps({"greedy_rows": len(rows), "mismatch_vs_id9": len(mismatch),
                      "candidates": len(cands), "candidates_sha256": cand_sha[:16],
                      "preds_dumped": len(preds_dump)}, ensure_ascii=False))

    # ---- 评分阶段：候选已落盘，才读人工标注 ----
    gold = json.loads(args.gold.read_text(encoding="utf8"))
    gold_drugs = gold["drugs"]
    baseline = json.loads(args.baseline_ocr.read_text(encoding="utf8"))

    def greedy_hits(target):
        """A 组：贪心解码的命中，用同一份分布的时间步回映位置（与 CTC 候选同口径）。"""
        out = []
        tgt = eq_text(target)
        for r in rows:
            if not r["text"]:
                continue
            w = wmap[r["window_id"]]
            t = eq_text(r["text"])
            align = r["greedy_alignment"]
            i = 0
            while True:
                k = t.find(tgt, i)
                if k < 0:
                    break
                i = k + 1
                if len(align) != len(r["text"]):
                    ok, note, y_span = False, "贪心对齐长度与文本不一致，无法回映", None
                else:
                    ya, yb = span_from_alignment(w, align, k, len(tgt), r["content_steps"])
                    ok, note, _b = position_ok(w, ya, yb, target, gold_drugs)
                    y_span = [round(ya, 1), round(yb, 1)]
                out.append({"source": "greedy_" + r["model"], "model": r["model"],
                            "window_id": r["window_id"], "text": r["text"],
                            "slice": r["text"][k:k + len(tgt)], "y_span": y_span,
                            "ok": ok, "note": note})
        return out

    def position_ok(w, y1, y2, target, gold_drugs):
        ratios = []
        for g in gold_drugs:
            if g["line_id"] != w["line_id"]:
                continue
            gb = g["box"]
            ratios.append((g["name"], _s3.overlap_ratio(y1, y2, gb[1], gb[3])))
        ratios.sort(key=lambda x: -x[1])
        if not ratios:
            return False, "同列无人工字形框", None
        best = ratios[0]
        ok = best[0] == target and best[1] >= TOL_OVERLAP
        if ok and len(ratios) > 1 and ratios[1][1] >= best[1] - 0.05:
            return False, "位置存疑：与 %s 重叠接近" % ratios[1][0], best
        return ok, ("位置成立（重叠 %.2f）" % best[1] if ok
                    else "位置不符：最佳匹配 %s(%.2f)" % best), best

    per_target = {}
    for g in gold_drugs:
        name = g["name"]
        # A 组：原始 OCR + 两模型贪心（= 账本 ID 9 的候选集合）
        a_hits = []
        for b in baseline["blocks"]:
            t = eq_text(b.get("text") or "")
            k = t.find(eq_text(name))
            if k >= 0:
                bx1, by1, bx2, by2 = b["bbox"]
                gx1, gy1, gx2, gy2 = g["box"]
                ok = (min(bx2, gx2) - max(bx1, gx1) > 0) and (min(by2, gy2) - max(by1, gy1) > 0)
                a_hits.append({"source": "baseline_ocr", "line_id": b["line_id"], "ok": ok,
                               "note": "块级位置与原图字形框相交" if ok else "正文别列，不计入"})
        a_hits.extend(greedy_hits(name))
        a_ok = any(h["ok"] for h in a_hits)

        # B 组：A + CTC 前 5 条候选
        b_hits = []
        tgt = eq_text(name)
        for c in cands:
            t = eq_text(c["text"])
            if tgt not in t:
                continue
            w = wmap[c["window_id"]]
            T = c.get("n_content_steps") or c["n_steps"]
            i = t.find(tgt)
            while i >= 0:
                if len(c["alignment"]) == len(c["text"]):
                    ya, yb = span_from_alignment(w, c["alignment"], i, len(tgt), T)
                    ok, note, best = position_ok(w, ya, yb, name, gold_drugs)
                    y_span = [round(ya, 1), round(yb, 1)]
                else:
                    ok, note, best = False, "对齐长度与文本不一致，无法回映", None
                    y_span = None
                b_hits.append({"source": "ctc_" + c["model"], "model": c["model"],
                               "window_id": c["window_id"], "rank": c["rank"],
                               "text": c["text"], "slice": c["text"][i:i + len(tgt)],
                               "seq_logprob": c["seq_logprob"], "y_span": y_span,
                               "ok": ok, "note": note})
                i = t.find(tgt, i + 1)
        b_ok = a_ok or any(h["ok"] for h in b_hits)
        per_target[name] = {"line_id": g["line_id"], "box": g["box"], "status": "A" if a_ok else ("B" if b_ok else "miss"),
                            "A_ok": a_ok, "B_ok": b_ok,
                            "greedy_runs": [x["text"] for x in rows
                                            if x["window_id"] in {h.get("window_id") for h in a_hits}],
                            "A_hits": a_hits,
                            "ctc_hits_ok": [h for h in b_hits if h["ok"]],
                            "ctc_hits_bad": [h for h in b_hits if not h["ok"]][:5]}

    a_recall = sum(1 for n in TARGETS if per_target[n]["A_ok"])
    b_recall = sum(1 for n in TARGETS if per_target[n]["B_ok"])
    kept8 = [n for n in per_target if n not in TARGETS]
    kept8_ok = [n for n in kept8 if per_target[n]["B_ok"]]

    score = {"metric": "第51页三处误读的正确候选召回数", "denominator": 3,
             "stage": "第3步补测：CTC 多候选解码",
             "control": {"ledger_id": 9, "raw_sha256": base_sha,
                         "note": "A 组即账本 ID 9 的候选集合（原始OCR+两模型原解码）"},
             "A": {"recall": a_recall, "total": 3},
             "B": {"recall": b_recall, "total": 3},
             "delta": b_recall - a_recall,
             "per_target": per_target,
             "kept_8": {"total": len(kept8), "ok": len(kept8_ok), "names_ok": kept8_ok},
             "greedy_replay_identical_to_id9": len(mismatch) == 0,
             "mismatch": mismatch[:10],
             "candidates_sha256": cand_sha,
             "config": cfg,
             "models": meta}
    (args.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf8")
    print(json.dumps({"A": a_recall, "B": b_recall, "kept8": len(kept8_ok),
                      "per_target": {k: v["status"] for k, v in per_target.items()
                                     if k in TARGETS}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
