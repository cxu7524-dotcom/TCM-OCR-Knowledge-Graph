# -*- coding: utf-8 -*-
"""第 4 步：药名校正的自动采纳规则（冻结后跑第 51 页）。

只读：原始 OCR 文本、窗口坐标、识别候选（含解码路径与内容步数）、通用药名词表。
人工标注、正确药名、目标区间只在 eval 子命令里读取，不参与决策。

规则（开发页冻结，见 frozen_rules.json）：
  R0 原词已是合法药名 → 保留原文
  R1 双模型 + 跨窗高（4H 与 2H 各有支持）→ 自动采纳
  R2 同模型跨窗高复现的非首选候选，名次 ≤ RANK_MAX 且同窗分差 ≤ GAP_MAX → 自动采纳
  R3 仅单窗/单模型/低名次支持，且两模型首选一致指向另一读法 → 待核
  R4 区间含剂量字（一二三…錢分兩半升合）或跨药名/剂量边界 → 待核
  R5 两条修订区间重叠 → 全部待核

用法：
  python tools/p51_step4_adopt.py apply --pool <pool.json> --ocr <原始 ocr.json> --out <目录>
  python tools/p51_step4_adopt.py eval --pool <pool.json> --decisions <decisions.jsonl> \
      --gold <gold_names.json> --out <报告.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import source_hash_of          # noqa: E402
from extract.schema import HERBS                # noqa: E402
from ocr.normalize import normalize_text        # noqa: E402

MAX_SPAN = 4
EQUIV = {"殼": "壳", "羗": "羌"}
DOSE_CHARS = set("一二三四五六七八九十百錢分兩半升合枚片箇个隻只")
RANK_MAX = 3        # R2 允许的最差名次
GAP_MAX = 0.35      # R2 允许的同窗分差（首选 logprob - 本候选 logprob，nat）；开发页正确非首选样本见 0.006/0.201
MIN_OVERLAP = 0.5   # 候选回映区间与 OCR 字位的最低重叠比


def eq_char(c: str) -> str:
    n = normalize_text(c)
    if len(n) != 1:
        n = c
    return EQUIV.get(n, n)


def eq_text(s: str) -> str:
    return "".join(eq_char(c) for c in s)


HERB_NORM = {}
for h in HERBS:
    HERB_NORM.setdefault(eq_text(h), h)
BY_LEN = {}
for h, orig in HERB_NORM.items():
    BY_LEN.setdefault(len(h), []).append(h)


def hamming1(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(1 for x, y in zip(a, b) if x != y) == 1


def char_ranges(gray, b: dict) -> tuple:
    """块内逐字 y 区间。

    优先用墨迹行投影切出的字形带（相邻字之间有空白间隙，R019 这种组间大间距也能分开）；
    带数与文本字数不符时退化为「首带起点 → 末带终点」的线性插值，并标记方法。
    返回 ([(y1,y2)...], 方法)。
    """
    x1, y1, x2, y2 = b["bbox"]
    n = len(b["text"])
    roi = gray[y1:y2, x1:x2] < 150
    prof = roi.sum(axis=1)
    rows = prof > 0
    bands, st = [], None
    for i, v in enumerate(rows):
        if v and st is None:
            st = i
        if not v and st is not None:
            bands.append((st, i))
            st = None
    if st is not None:
        bands.append((st, len(rows)))
    # 合并极小间隙、丢弃噪点
    merged = []
    for a, bb in bands:
        if merged and a - merged[-1][1] <= 1:      # 字间空隙常只有 2px，阈值必须收紧
            merged[-1] = (merged[-1][0], bb)
        else:
            merged.append((a, bb))
    merged = [(a, bb) for a, bb in merged if bb - a >= 3]
    if len(merged) == n:
        return [(y1 + a, y1 + bb) for a, bb in merged], "glyph_bands"
    if not merged:
        h = (y2 - y1) / float(n)
        return [(y1 + i * h, y1 + (i + 1) * h) for i in range(n)], "uniform"
    # 带数与字数不符：按顺序做 DP 对齐（允许一带有两字、也允许跳过噪点带）
    return _dp_map(n, merged, y1, y2)


def _dp_map(n: int, bands: list, y1: int, y2: int) -> tuple:
    """把 n 个字按顺序对到 k 条带上，最小化与均匀先验的偏差。"""
    k = len(bands)
    h = (y2 - y1) / float(n)
    pc = [y1 + (i + 0.5) * h for i in range(n)]          # 均匀先验中心
    bc = [(a + bb) / 2.0 + y1 for a, bb in bands]
    INF = float("inf")
    cost = [[INF] * (k + 1) for _ in range(n + 1)]
    back = [[None] * (k + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for j in range(1, k + 1):                            # 跳过带
        if cost[0][j - 1] < INF:
            cost[0][j] = cost[0][j - 1] + 40.0
            back[0][j] = (0, j - 1)
    for i in range(n):
        for j in range(1, k + 1):
            # 一字对一带
            if cost[i][j - 1] < INF:
                c = cost[i][j - 1] + abs(pc[i] - bc[j - 1])
                if c < cost[i + 1][j]:
                    cost[i + 1][j] = c
                    back[i + 1][j] = (i, j - 1)
            # 两字对一带
            if i + 1 < n and cost[i][j - 1] < INF:
                c = cost[i][j - 1] + abs(pc[i] - bc[j - 1]) + abs(pc[i + 1] - bc[j - 1]) + 20.0
                if c < cost[i + 1][j]:
                    cost[i + 1][j] = c
                    back[i + 1][j] = (i, j - 1, "pair")
    i, j = n, k
    assign = [None] * n
    while i > 0 and j > 0:
        b = back[i][j]
        if b is None:
            j -= 1
            continue
        if len(b) == 3:
            assign[i - 1] = j - 1
            assign[i - 2] = j - 1
            i, j = b[0], b[1]
        else:
            assign[i - 1] = j - 1
            i, j = b
    for t in range(n):
        if assign[t] is None:
            assign[t] = min(k - 1, int(t * k / n))
    boxes = [(y1 + bands[assign[t]][0], y1 + bands[assign[t]][1]) for t in range(n)]
    return boxes, "dp_bands"


def overlap(a1, a2, b1, b2) -> float:
    ov = max(0.0, min(a2, b2) - max(a1, b1))
    d = min(a2 - a1, b2 - b1)
    return ov / d if d > 0 else 0.0


def span_y(w: dict, cand: dict, j: int, n: int) -> tuple:
    """候选第 j..j+n-1 字 → 原图 y 区间（解码时间步 / 内容步数 × 窗高）。"""
    H = w["y2"] - w["y1"]
    cs = float(cand.get("n_content_steps") or 1.0)
    al = cand["alignment"]
    t0 = al[j][1]
    last = j + n - 1
    if last + 1 < len(al):
        step = max(1, al[last + 1][1] - al[last][1])
    elif j > 0:
        step = max(1, al[j][1] - al[j - 1][1])
    else:
        step = 2
    y1 = w["y1"] + t0 / cs * H
    y2 = w["y1"] + (al[last][1] + step) / cs * H
    return max(w["y1"], y1), min(w["y2"], y2)


def build_index(pool: dict) -> dict:
    """(line_id, 规范化子串) → [(cand_idx, 起始位置)]，长度 2..MAX_SPAN。"""
    idx = {}
    for ci, c in enumerate(pool["candidates"]):
        t = eq_text(c["text"])
        for L in range(2, MAX_SPAN + 1):
            for j in range(0, len(t) - L + 1):
                idx.setdefault((c["line_id"], t[j:j + L]), []).append((ci, j))
    return idx


def first_choice_at(pool: dict, windows: dict, block: dict, y1: float, y2: float) -> dict:
    """在该字位，各模型在同一窗口里的首选读法（只看 rank==1 且窗口覆盖该字位）。"""
    out = {}
    for c in pool["candidates"]:
        if c["rank"] != 1 or c["line_id"] != block["line_id"]:
            continue
        w = windows[c["window_id"]]
        if w["y1"] > y1 or w["y2"] < y2:
            continue
        t = eq_text(c["text"])
        # 首选文本里与字位重叠的那一段
        best = None
        for j in range(len(t)):
            a, b = span_y(w, c, j, 1)
            r = overlap(a, b, y1, y2)
            if r >= MIN_OVERLAP and (best is None or r > best[0]):
                best = (r, t[j])
        if best:
            out.setdefault(c["model"], set()).add(best[1])
    return out


def apply_rules(pool: dict) -> dict:
    from ocr.vertical import imread_unicode
    gray = imread_unicode(pool["image"])
    if gray is not None and gray.ndim == 3:
        gray = np.mean(gray[:, :, :3], axis=2).astype(np.uint8)
    windows = pool["windows"]
    idx = build_index(pool)
    blocks = pool["blocks"]
    decisions, adopted = [], []
    pos_method = {}

    for b in blocks:
        text = b["text"]
        L_text = len(text)
        boxes, method = char_ranges(gray, b)
        pos_method[b["line_id"]] = method
        for i in range(L_text - 1):
            for L in range(2, MAX_SPAN + 1):
                if i + L > L_text:
                    break
                token = text[i:i + L]
                if not all("\u4e00" <= ch <= "\u9fff" for ch in token):
                    continue
                ntok = eq_text(token)
                if ntok in HERB_NORM:
                    continue                      # R0：已是合法药名
                cands = [h for h in BY_LEN.get(L, []) if hamming1(ntok, h)]
                if not cands:
                    continue
                y1 = boxes[i][0]
                y2 = boxes[i + L - 1][1]
                for h in cands:
                    support = []
                    for ci, j in idx.get((b["line_id"], h), []):
                        c = pool["candidates"][ci]
                        w = windows[c["window_id"]]
                        a, bb = span_y(w, c, j, L)
                        r = overlap(a, bb, y1, y2)
                        if r < MIN_OVERLAP:
                            continue
                        # 上下文一致：候选里 h 前后字与原 OCR 相邻字相同
                        ctx = 0
                        ct = eq_text(c["text"])
                        if j > 0 and i > 0 and ct[j - 1] == eq_char(text[i - 1]):
                            ctx += 1
                        if j + L < len(ct) and i + L < L_text and ct[j + L] == eq_char(text[i + L]):
                            ctx += 1
                        support.append({"model": c["model"], "window_id": c["window_id"],
                                        "mult": w["mult"], "rank": c["rank"],
                                        "text": c["text"], "seq_logprob": c["seq_logprob"],
                                        "y_span": [round(a, 1), round(bb, 1)],
                                        "overlap": round(r, 3), "ctx_match": ctx})
                    if not support:
                        continue
                    # 同窗分差：同一模型同一窗口里首选与该候选的 logprob 差
                    gaps = []
                    for s in support:
                        rank1 = [c for c in pool["candidates"]
                                 if c["window_id"] == s["window_id"] and c["model"] == s["model"]
                                 and c["rank"] == 1]
                        if rank1:
                            gaps.append(round(rank1[0]["seq_logprob"] - s["seq_logprob"], 3))
                    models = {s["model"] for s in support}
                    mults = {s["mult"] for s in support}
                    cross_model = len(models) >= 2
                    cross_height = len(mults) >= 2
                    best_rank = min(s["rank"] for s in support)
                    best_gap = min(gaps) if gaps else None
                    has_dose = any(ch in DOSE_CHARS for ch in token)
                    fc = first_choice_at(pool, windows, b, y1, y2)
                    first_agree_other = (len(fc) >= 2 and
                                         all(len(v) == 1 and list(v)[0] != eq_char(h[0]) for v in fc.values()))

                    reasons = []
                    if cross_model and cross_height:
                        status, rule = "自动采纳", "R1"
                        reasons.append("双模型+跨窗高")
                    elif len(models) == 1 and cross_height and best_rank <= RANK_MAX and \
                            best_gap is not None and best_gap <= GAP_MAX:
                        status, rule = "自动采纳", "R2"
                        reasons.append("同模型跨窗高复现，名次%d，分差%.2f" % (best_rank, best_gap))
                    elif first_agree_other and (len(models) == 1 or not cross_height):
                        status, rule = "待核", "R3"
                        reasons.append("两模型首选一致指向另一读法，本读法仅%s支持" %
                                       ("单窗" if not cross_height else "单模型"))
                    else:
                        status, rule = "待核", "R3"
                        reasons.append("证据不足：模型数%d 窗高数%d 名次%d 分差%s" %
                                       (len(models), len(mults), best_rank, best_gap))
                    if has_dose:
                        status, rule = "待核", "R4"
                        reasons.append("区间含剂量字")
                    # R6：邻字保护（两侧）。拟替换区间若吞掉一个原文字，而图像候选把该字
                    # 读在药名紧邻一侧，说明原文是「药名+该字」漏了中间一字，不是错字替换。
                    guard = []
                    Lh = len(eq_text(h))
                    for sup in support:
                        ct = eq_text(sup["text"])
                        jj = ct.find(eq_text(h))
                        while jj >= 0:
                            if jj > 0 and ct[jj - 1] == eq_char(token[0]):
                                guard.append({"side": "left", "char": token[0],
                                              "window": sup["window_id"], "model": sup["model"],
                                              "candidate": sup["text"]})
                            if jj + Lh < len(ct) and ct[jj + Lh] == eq_char(token[-1]):
                                guard.append({"side": "right", "char": token[-1],
                                              "window": sup["window_id"], "model": sup["model"],
                                              "candidate": sup["text"]})
                            jj = ct.find(eq_text(h), jj + 1)
                    if guard:
                        g = guard[0]
                        status, rule = "待核", "R6"
                        reasons.append("邻字保护(%s「%s」，%s/%s)：候选「%s」把该字读在药名%s侧，"
                                       "原文疑为漏字而非错字" %
                                       ("左侧" if g["side"] == "left" else "右侧", g["char"],
                                        g["model"], g["window"],
                                        g["candidate"],
                                        "前" if g["side"] == "left" else "后"))

                    decisions.append({
                        "block_ref": b["block_ref"], "line_id": b["line_id"],
                        "span": [i, i + L], "original": token, "proposal": h,
                        "bbox": [b["bbox"][0], round(y1, 1), b["bbox"][2], round(y2, 1)],
                        "status": status, "rule": rule, "reasons": reasons,
                        "best_rank": best_rank, "best_gap": best_gap,
                        "models": sorted(models), "mults": sorted(mults),
                        "n_support": len(support), "support": support[:6],
                        "guard": guard, "pos_method": method})
    # R5：区间重叠 → 全部待核
    acc = [d for d in decisions if d["status"] == "自动采纳"]
    for d in acc:
        for e in acc:
            if d is e or d["block_ref"] != e["block_ref"]:
                continue
            if d["span"][0] < e["span"][1] and e["span"][0] < d["span"][1]:
                d["status"] = "待核"
                d["rule"] = "R5"
                d["reasons"].append("与另一条修订区间重叠")
                break
    for d in decisions:
        if d["status"] == "自动采纳":
            adopted.append(d)
    return {"decisions": decisions, "adopted": adopted, "pos_method": pos_method}


def apply_to_ocr(ocr: dict, adopted: list) -> dict:
    out = json.loads(json.dumps(ocr))
    by_ref = {}
    for d in adopted:
        by_ref.setdefault(d["block_ref"], []).append(d)
    changed = []
    for b in out["blocks"]:
        ds = by_ref.get(b["block_ref"])
        if not ds:
            continue
        text = b["text"]
        for d in sorted(ds, key=lambda x: -x["span"][0]):
            s, e = d["span"]
            text = text[:s] + d["proposal_original"] + text[e:]
        b["original_text"] = b["text"]
        b["text"] = text
        b["text_norm"] = normalize_text(text)
        b["quote"] = text
        b["source_hash"] = source_hash_of("ocr", out.get("page_id", ""), b["line_id"], text)
        b["corrections"] = [{"span": d["span"], "from": d["original"], "to": d["proposal"],
                             "rule": d["rule"], "window": d["support"][0]["window_id"]}
                            for d in ds]
        changed.append(b["block_ref"])
    out["corrections_applied"] = len(adopted)
    out["changed_blocks"] = sorted(set(changed))
    out["note"] = "第4步自动采纳后的 OCR 文本；text_norm/quote/source_hash 已同步，未采纳项保持原文"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("apply", "eval"))
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--ocr", type=Path)
    ap.add_argument("--decisions", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--annot", type=Path, help="开发页标注（eval 用）")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf8"))
    if args.cmd == "apply":
        res = apply_rules(pool)
        # proposal 原字形（保留证据里的原始字形）
        for d in res["decisions"]:
            d["proposal_original"] = d["proposal"]
        ocr = json.loads(args.ocr.read_text(encoding="utf8"))
        corrected = apply_to_ocr(ocr, res["adopted"])
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "decisions.jsonl").open("w", encoding="utf8") as f:
            for d in res["decisions"]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        (args.out / "corrected_ocr.json").write_text(
            json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        rules = {"rank_max": RANK_MAX, "gap_max": GAP_MAX, "min_overlap": MIN_OVERLAP,
                 "max_span": MAX_SPAN, "dose_chars": "".join(sorted(DOSE_CHARS)),
                 "herb_lexicon": "extract/schema.py HERBS (%d 味)" % len(HERBS),
                 "normalize": "ocr/normalize.py + 等价 {殼:壳, 羗:羌}",
                 "rules": {"R0": "原词已是合法药名 → 保留原文",
                           "R1": "双模型 + 跨窗高 → 自动采纳",
                           "R2": "同模型跨窗高、名次<=%d、分差<=%.1f → 自动采纳" % (RANK_MAX, GAP_MAX),
                           "R3": "单窗/单模型/低名次且两模型首选指向他读 → 待核",
                           "R4": "区间含剂量字 → 待核",
                           "R5": "区间重叠 → 待核",
                           "R6": "候选在药名前重复出现原文字（插入/删除迹象）→ 待核"}}
        (args.out / "frozen_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf8")
        summary = {"decisions": len(res["decisions"]), "auto_adopt": len(res["adopted"]),
                   "pos_method": res["pos_method"],
                   "pending": sum(1 for d in res["decisions"] if d["status"] == "待核"),
                   "kept": sum(1 for d in res["decisions"] if d["status"] == "保留原文"),
                   "changed_blocks": corrected["changed_blocks"]}
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    # ---- eval ----
    decisions = [json.loads(l) for l in args.decisions.open(encoding="utf8")]
    corrected = json.loads((args.out / "corrected_ocr.json").read_text(encoding="utf8"))
    if args.gold:                       # 第 51 页：三处目标
        gold = json.loads(args.gold.read_text(encoding="utf8"))["drugs"]
        txt = {b["line_id"]: b["text"] for b in corrected["blocks"]}
        orig = {b["line_id"]: b["original_text"] if "original_text" in b else b["text"]
                for b in corrected["blocks"]}
        rows = []
        for g in gold:
            name = g["name"]
            lid = g["line_id"]
            before = eq_text(orig.get(lid, ""))
            after = eq_text(txt.get(lid, ""))
            rows.append({"name": name, "line_id": lid,
                         "in_original": eq_text(name) in before,
                         "in_corrected": eq_text(name) in after,
                         "auto_changed": orig.get(lid) != txt.get(lid)})
        print(json.dumps({"rows": rows}, ensure_ascii=False, indent=1))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
