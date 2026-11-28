# -*- coding: utf-8 -*-
"""第 3 步补测的诊断：某味药在候选/字符层面的最高名次（不改固定参数，只做定位）。

回答两个问题：
  1) 正确药名在更深束（depth=50）下能否出现？最高第几名？——判断是「未进前 5」还是「根本没有」。
  2) 疑字处（如「茯茶」的第二个字）在逐时间步分布里排第几？——判断该继续查解码还是查图像输入。

用法：
  python tools/p51_step3_ctc_diagnose.py --windows <windows.json> --crops <目录> \
      --gold <gold_names.json> --target 茯苓 --out-dir <第三步补测输出目录>
"""
from __future__ import annotations

import argparse
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
eq_text = _ctc.eq_text
DEPTH = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    wd = json.loads(args.windows.read_text(encoding="utf8"))
    windows = [w for w in wd["windows"] if w["group"] == "B"]
    gold = json.loads(args.gold.read_text(encoding="utf8"))
    g = next(x for x in gold["drugs"] if x["name"] == args.target)
    line = g["line_id"]
    gbox = g["box"]
    tgt = eq_text(args.target)
    correct_last = args.target[-1]

    engines = {k: _text_recognizer(_engine(str(ROOT / rel))) for k, rel, _d in MODELS}
    out = {"target": args.target, "line_id": line, "gold_box": gbox, "depth": DEPTH,
           "windows_in_line": [], "best_rank": {}, "char_rank": {}}

    for w in [x for x in windows if x["line_id"] == line]:
        crop = args.crops / ("%s.png" % w["window_id"].replace(":", "_"))
        img = imread_unicode(str(crop))
        rot = np.ascontiguousarray(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        row = {"window_id": w["window_id"], "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]],
               "models": {}}
        for key, _rel, _d in MODELS:
            rec = engines[key]
            with _ctc.PredSpy(rec) as spy:
                res, _t = rec([rot], return_word_box=True)
            preds = spy.preds[0]          # (T, C)：shape[0] 才是时间步
            T = preds.shape[0]
            cs = _ctc.content_steps(rot.shape, tuple(getattr(rec, "rec_image_shape", (3, 48, 320))), T)
            chars = list(getattr(rec.postprocess_op, "character", []))
            gtext, galign = _ctc.greedy_alignment(preds, chars)
            ranked = _ctc.ctc_beam(preds, chars, _ctc.BEAM_WIDTH, _ctc.TOPK_PER_STEP, _ctc.MAX_PREFIX)
            # 深束：直接放宽 beam 与候选数
            deep = _ctc.ctc_beam(preds, chars, DEPTH, 12, _ctc.MAX_PREFIX)
            hits = []
            for r, (sc, text, align) in enumerate(deep, start=1):
                t = eq_text(text)
                k = t.find(tgt)
                while k >= 0:
                    if len(align) == len(text):
                        ya, yb = _ctc.span_from_alignment(w, align, k, len(tgt), cs)
                        ov = _ctc._s3.overlap_ratio(ya, yb, gbox[1], gbox[3])
                    else:
                        ya, yb, ov = 0, 0, 0.0
                    if ov >= 0.5:
                        hits.append({"rank": r, "text": text, "overlap": round(ov, 3),
                                     "y_span": [round(ya, 1), round(yb, 1)],
                                     "seq_logprob": round(sc, 4)})
                    k = t.find(tgt, k + 1)
            # 疑字处逐字符排名：贪心对齐里覆盖字形框下半部分的那个时间步
            char_rank = None
            if len(galign) == len(gtext):
                spans = []
                for i in range(len(gtext)):
                    ya, yb = _ctc.span_from_alignment(w, galign, i, 1, cs)
                    spans.append((i, ya, yb, _ctc._s3.overlap_ratio(ya, yb, gbox[1], gbox[3])))
                cands_i = [s for s in spans if s[3] >= 0.3]
                if cands_i:
                    i, _ya, _yb, _ov = max(cands_i, key=lambda s: s[1])  # 取最靠下的那个字
                    t_step = galign[i][1]
                    probs = preds[t_step]
                    order = np.argsort(-probs)[:8]
                    rank = None
                    for r, c in enumerate(order, start=1):
                        if chars[int(c)] == correct_last:
                            rank = r
                    char_rank = {"greedy_char": gtext[i], "timestep": int(t_step),
                                 "n_steps": int(T),
                                 "top5": [{"char": chars[int(c)], "prob": round(float(probs[c]), 5)}
                                          for c in order[:5]],
                                 "correct_char": correct_last,
                                 "correct_char_rank": rank,
                                 "correct_char_prob": round(
                                     float(probs[chars.index(correct_last)])
                                     if correct_last in chars else 0.0, 5)}
            row["models"][key] = {"greedy": gtext, "n_candidates_top5": len(ranked),
                                  "deep_hits_at_position": hits[:5],
                                  "best_rank_in_line": min([h["rank"] for h in hits], default=None),
                                  "char_rank": char_rank}
            cur = out["best_rank"].get(key)
            br = row["models"][key]["best_rank_in_line"]
            if br is not None and (cur is None or br < cur):
                out["best_rank"][key] = br
            cr = out["char_rank"].get(key)
            if char_rank and (cr is None or (char_rank["correct_char_rank"] or 99) <
                              (cr["correct_char_rank"] or 99)):
                out["char_rank"][key] = {**char_rank, "window_id": w["window_id"]}
        out["windows_in_line"].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / ("diagnose_%s.json" % args.target)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"target": args.target, "best_rank": out["best_rank"],
                      "char_rank": {k: (v["greedy_char"], v["correct_char_rank"],
                                        v["correct_char_prob"], v["window_id"])
                                    for k, v in out["char_rank"].items()},
                      "path": str(p)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
