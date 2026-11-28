# -*- coding: utf-8 -*-
"""第 51 页药名局部窗口裁图（第二步，离线几何核对，不跑识别）。

输入：原图 + 第一步已验收 ocr.json
输出：A 组（OCR 块原始矩形）/ B 组（固定滑窗）窗口表、裁图、人工字形框、预览图、评分

固定公式（与回执一致，改公式等于换尺子，必须换目录重跑）：
  估计字高 H = round(块高 / 原始 text 中汉字数)
  B 组窗口高 = min(4H, 块高)；步长 = max(1, round(H/2))；宽 = 块左右边界
  从块顶起步，末尾追加对齐块底的窗口，坐标去重

人工字形框来自逐字人工核对（y 区间读自放大标尺图，x 区间由该 y 带内墨迹投影确定），
只供 score() 读取，不参与窗口生成，避免循环论证。

用法：
  python tools/p51_local_windows.py --image <原图> --ocr <ocr.json> --out <新目录>
  python tools/p51_local_windows.py --compare <目录A> <目录B>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.vertical import imread_unicode  # noqa: E402  复用项目原图读取（Windows 中文路径兜底）

INK_TH = 150          # 二值阈值：与本页既有竖排链路一致
INNER_MARGIN = 2      # 字形框外扩像素
NEIGHBOR_MIN_W = 4    # 与邻列块的重叠宽度达到此值才判为「可疑邻列」
NEIGHBOR_MIN_INK = 20 # 可疑重叠区内墨迹像素下限

# ---- 人工标注：逐字 y 区间（原图 1024x817 坐标，读自 output/p51_step2_explore/ruler_*.png）----
MANUAL_CHARS = {
    "R021": [("人", 247, 270), ("參", 275, 303), ("一", 306, 318), ("錢", 335, 357),
             ("羌", 413, 441), ("活", 443, 468), ("三", 488, 512), ("錢", 519, 545),
             ("獨", 581, 608), ("活", 610, 635)],
    "R020": [("桔", 248, 273), ("梗", 276, 301), ("枳", 416, 440), ("壳", 442, 468),
             ("各", 472, 494), ("一", 503, 514), ("錢", 527, 550), ("半", 556, 578),
             ("柴", 583, 607), ("胡", 611, 638)],
    "R019": [("前", 249, 274), ("胡", 278, 304), ("川", 415, 441), ("芎", 443, 470),
             ("茯", 582, 608), ("苓", 611, 638), ("各", 642, 664), ("一", 672, 682),
             ("錢", 695, 722)],
    "R018": [("甘", 249, 272), ("草", 276, 304), ("五", 308, 327), ("分", 337, 355),
             ("生", 416, 441), ("薑", 442, 469), ("三", 478, 498), ("錢", 501, 525)],
}

# 本方 11 味：人工药名 -> (块, 该块内首字序号, 别名备注)
DRUGS = [
    ("人参", "R021", 0, ""),
    ("羌活", "R021", 4, "等价口径：羗活＝羌活"),
    ("独活", "R021", 8, ""),
    ("桔梗", "R020", 0, ""),
    ("枳壳", "R020", 2, "等价口径：枳壳＝枳殼"),
    ("柴胡", "R020", 8, ""),
    ("前胡", "R019", 0, ""),
    ("川芎", "R019", 2, ""),
    ("茯苓", "R019", 4, ""),
    ("甘草", "R018", 0, ""),
    ("生姜", "R018", 4, ""),
]


def hz_count(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(image: Path, ocr_json: Path):
    img = imread_unicode(str(image))
    if img is None:
        raise SystemExit("原图读取失败：%s" % image)
    if img.ndim == 3:
        gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = img
    data = json.loads(ocr_json.read_text(encoding="utf8"))
    return img, gray, data


def x_extent(gray: np.ndarray, bx1: int, bx2: int, y1: int, y2: int) -> tuple:
    """在给定 y 带内按列投影取墨迹左右边界，夹到块内。"""
    band = gray[y1:y2, bx1:bx2] < INK_TH
    cols = band.sum(axis=0)
    keep = np.where(cols >= 2)[0]
    if keep.size == 0:
        return bx1 + 3, bx2 - 3
    x1 = max(bx1, bx1 + int(keep[0]) - INNER_MARGIN)
    x2 = min(bx2, bx1 + int(keep[-1]) + 1 + INNER_MARGIN)
    return x1, x2


def gold_boxes(gray: np.ndarray, blocks: dict) -> list:
    out = []
    for name, lid, idx, note in DRUGS:
        b = blocks[lid]
        bx1, _, bx2, _ = b["bbox"]
        chars = []
        for ch, cy1, cy2 in MANUAL_CHARS[lid][idx:idx + 2]:
            cx1, cx2 = x_extent(gray, bx1, bx2, cy1, cy2)
            chars.append({"char": ch, "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2})
        out.append({
            "name": name, "line_id": lid, "block_ref": b["block_ref"], "note": note,
            "chars": chars,
            "box": [min(c["x1"] for c in chars), min(c["y1"] for c in chars),
                    max(c["x2"] for c in chars), max(c["y2"] for c in chars)],
            "ocr_text": b["text"],
        })
    return out


def block_geo(blocks: list, gray: np.ndarray) -> tuple:
    """逐块估计字高；返回可用块表 + 跳过表。"""
    H, W = gray.shape[:2]
    usable, skipped = [], []
    for b in blocks:
        bx1, by1, bx2, by2 = b["bbox"]
        n = hz_count(b["text"])
        if not (0 <= bx1 < bx2 <= W and 0 <= by1 < by2 <= H):
            skipped.append({"line_id": b["line_id"], "reason": "坐标越界或退化", "bbox": b["bbox"]})
            continue
        if n == 0:
            skipped.append({"line_id": b["line_id"], "reason": "text 中无汉字", "bbox": b["bbox"]})
            continue
        bh = by2 - by1
        Hc = max(1, int(round(bh / n)))
        usable.append({"line_id": b["line_id"], "block_ref": b["block_ref"], "bbox": b["bbox"],
                       "text": b["text"], "n_hanzi": n, "block_h": bh, "est_char_h": Hc,
                       "formula": "%d / %d = %.2f -> %d" % (bh, n, bh / n, Hc)})
    return usable, skipped


def windows_for(blk: dict, mult: int = 4) -> list:
    """固定滑窗。mult 为窗口高度相对估计字高的倍数：第二步用 4，短窗补测用 2。"""
    x1, y1, x2, y2 = blk["bbox"]
    Hc = blk["est_char_h"]
    bw, bh = x2 - x1, y2 - y1
    h = min(mult * Hc, bh)
    step = max(1, int(round(Hc / 2)))
    ys = []
    y = y1
    while y + h <= y2:
        ys.append(y)
        y += step
    ys.append(y2 - h)  # 末尾对齐块底
    ys = sorted(set(ys))
    out = []
    for i, sy in enumerate(ys):
        out.append({"window_id": "B:%s:W%02d" % (blk["line_id"], i + 1), "group": "B",
                    "line_id": blk["line_id"], "block_ref": blk["block_ref"],
                    "x1": x1, "y1": sy, "x2": x2, "y2": sy + h,
                    "est_char_h": Hc, "start_y": sy, "height": h, "step": step,
                    "formula": blk.get("formula", ""), "mult": mult})
    return out


def contains(w: dict, box: list) -> bool:
    return (w["x1"] <= box[0] and w["y1"] <= box[1]
            and w["x2"] >= box[2] and w["y2"] >= box[3])


def neighbor_hits(w: dict, blk: dict, blocks: list, gray: np.ndarray) -> list:
    """窗内是否混入邻列（x 方向相邻的另一列）字迹。同列上下文不算。"""
    hits = []
    for o in blocks:
        if o["line_id"] == blk["line_id"]:
            continue
        ox1, oy1, ox2, oy2 = o["bbox"]
        same_col = not (ox1 >= blk["bbox"][2] - 2 or ox2 <= blk["bbox"][0] + 2)
        if same_col:      # 同列（x 基本重合）→ 属同列上下文，允许
            continue
        ix1, iy1 = max(w["x1"], ox1), max(w["y1"], oy1)
        ix2, iy2 = min(w["x2"], ox2), min(w["y2"], oy2)
        if ix2 - ix1 < NEIGHBOR_MIN_W or iy2 - iy1 < 4:
            continue
        ink = int((gray[iy1:iy2, ix1:ix2] < INK_TH).sum())
        if ink >= NEIGHBOR_MIN_INK:
            hits.append({"line_id": o["line_id"], "overlap_w": ix2 - ix1, "ink": ink})
    return hits


def evaluate(windows: list, blks: dict, gold: list, blocks: list, gray: np.ndarray) -> dict:
    per, ok = [], 0
    for g in gold:
        blk = blks[g["line_id"]]
        cands = [w for w in windows if w["line_id"] == g["line_id"] and contains(w, g["box"])]
        chosen, reasons = None, []
        if not cands:
            reasons.append("该组无窗口包含完整字形框")
        else:
            for w in cands:
                rs = []
                if w["height"] > 4 * blk["est_char_h"]:
                    rs.append("窗口高 %d > 4×字高 %d" % (w["height"], 4 * blk["est_char_h"]))
                nh = neighbor_hits(w, blk, blocks, gray)
                if nh:
                    rs.append("混入邻列字迹：" + ",".join("%s(ink=%d)" % (h["line_id"], h["ink"]) for h in nh))
                if not rs:
                    chosen, reasons = w, []
                    break
                reasons.append("%s:%s" % (w["window_id"], "；".join(rs)))
        if chosen:
            ok += 1
        per.append({"name": g["name"], "line_id": g["line_id"], "block_ref": g["block_ref"],
                    "box": g["box"], "chars": g["chars"], "qualified": bool(chosen),
                    "window_id": chosen["window_id"] if chosen else None,
                    "window_bbox": [chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"]] if chosen else None,
                    "n_candidates": len(cands),
                    "reason": "合格" if chosen else "；".join(reasons[:3])})
    return {"coverage": ok, "total": len(gold), "per_drug": per}


def draw(img: np.ndarray, a_wins: list, b_wins: list, gold: list, path: Path, scale: int = 1):
    import cv2
    canvas = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for w in a_wins:
        cv2.rectangle(canvas, (w["x1"], w["y1"]), (w["x2"] - 1, w["y2"] - 1), (255, 80, 0), 1)
    for w in b_wins:
        if w["line_id"] in MANUAL_CHARS:
            cv2.rectangle(canvas, (w["x1"], w["y1"]), (w["x2"] - 1, w["y2"] - 1), (0, 170, 0), 1)
    for g in gold:
        b = g["box"]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2] - 1, b[3] - 1), (0, 0, 255), 2)
    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imencode(".png", canvas)[1].tofile(str(path))


def crop(img: np.ndarray, w: dict, path: Path):
    import cv2
    patch = img[w["y1"]:w["y2"], w["x1"]:w["x2"]]
    cv2.imencode(".png", patch)[1].tofile(str(path))


def drug_crop(img: np.ndarray, w: dict, g: dict, path: Path, pad: int = 6, scale: int = 5):
    """药名放大图：窗口外留 pad，红框为人工字形框，便于目视首尾笔画。"""
    import cv2
    x1, y1 = max(0, w["x1"] - pad), max(0, w["y1"] - pad)
    x2 = min(img.shape[1], w["x2"] + pad)
    y2 = min(img.shape[0], w["y2"] + pad)
    patch = img[y1:y2, x1:x2]
    canvas = patch.copy() if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    for c in g["chars"]:
        cv2.rectangle(canvas, (c["x1"] - x1, c["y1"] - y1), (c["x2"] - x1 - 1, c["y2"] - y1 - 1), (0, 0, 255), 1)
    cv2.rectangle(canvas, (w["x1"] - x1, w["y1"] - y1), (w["x2"] - x1 - 1, w["y2"] - y1 - 1), (0, 170, 0), 1)
    big = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imencode(".png", big)[1].tofile(str(path))


def table_hash(rows: list) -> str:
    key = ";".join("%s|%s|%d|%d|%d|%d|%d" % (r["window_id"], r["line_id"], r["x1"], r["y1"],
                                             r["x2"], r["y2"], r["est_char_h"]) for r in rows)
    return hashlib.sha256(key.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--ocr", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        a = json.loads((args.compare[0] / "windows.json").read_text(encoding="utf8"))
        b = json.loads((args.compare[1] / "windows.json").read_text(encoding="utf8"))
        ha, hb = table_hash(a["windows"]), table_hash(b["windows"])
        same = ha == hb and len(a["windows"]) == len(b["windows"])
        print(json.dumps({"run_a": table_hash(a["windows"]), "run_b": hb,
                          "windows": [len(a["windows"]), len(b["windows"])],
                          "identical": same}, ensure_ascii=False))
        return 0 if same else 1

    if not (args.image and args.ocr and args.out):
        raise SystemExit("需要 --image/--ocr/--out")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "crops").mkdir()
    img, gray, data = load(args.image, args.ocr)
    blocks = data["blocks"]
    blks = {b["line_id"]: b for b in blocks}

    usable, skipped = block_geo(blocks, gray)
    a_wins = [{"window_id": "A:%s" % b["line_id"], "group": "A", "line_id": b["line_id"],
               "block_ref": b["block_ref"], "x1": b["bbox"][0], "y1": b["bbox"][1],
               "x2": b["bbox"][2], "y2": b["bbox"][3], "est_char_h": b["est_char_h"],
               "start_y": b["bbox"][1], "height": b["block_h"],
               "step": 0, "formula": b["formula"]} for b in usable]
    b_wins = []
    for b in usable:
        b_wins.extend(windows_for(b))

    gold = gold_boxes(gray, blks)
    (args.out / "gold_names.json").write_text(json.dumps(
        {"source": "人工逐字核对（y 区间读自 ruler 图，x 区间由该 y 带墨迹投影确定）",
         "image_sha256": sha256_of(args.image), "ocr_json": str(args.ocr.resolve()),
         "ocr_json_sha256": sha256_of(args.ocr), "drugs": gold}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")

    rows = []
    for w in a_wins + b_wins:
        p = args.out / "crops" / ("%s.png" % w["window_id"].replace(":", "_"))
        crop(img, w, p)
        rows.append({**w, "crop_path": str(p)})
    with (args.out / "windows.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["window_id", "group", "line_id", "block_ref", "x1", "y1",
                                           "x2", "y2", "est_char_h", "start_y", "height", "step",
                                           "formula", "crop_path"])
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "windows.json").write_text(json.dumps(
        {"params": {"est_char_h": "round(block_h / n_hanzi)", "window_h": "min(4*H, block_h)",
                    "step": "max(1, round(H/2))", "width": "block x bounds",
                    "tail": "append bottom-aligned window", "ink_th": INK_TH,
                    "neighbor_min_w": NEIGHBOR_MIN_W, "neighbor_min_ink": NEIGHBOR_MIN_INK},
         "table_hash": table_hash(rows), "blocks": usable, "skipped": skipped,
         "windows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    geos = {b["line_id"]: b for b in usable}
    a_eval = evaluate(a_wins, geos, gold, blocks, gray)
    b_eval = evaluate(b_wins, geos, gold, blocks, gray)

    # 药名裁图（用 B 组合格窗口）
    drug_paths = []
    for g, item in zip(gold, b_eval["per_drug"]):
        w = next((x for x in b_wins if x["window_id"] == item["window_id"]), None)
        if w is None:
            continue
        p = args.out / "crops" / ("drug_%s_%s.png" % (g["line_id"], g["name"]))
        drug_crop(img, w, g, p)
        drug_paths.append({"name": g["name"], "line_id": g["line_id"], "path": str(p)})

    draw(img, a_wins, b_wins, gold, args.out / "preview_all.png")
    draw(img, a_wins, b_wins, gold, args.out / "preview_all_2x.png", scale=2)

    score = {"metric": "第51页药名合格局部窗口覆盖数", "denominator": 11,
             "A": a_eval, "B": b_eval,
             "delta": b_eval["coverage"] - a_eval["coverage"],
             "diagnostics": {
                 "A_containment_only": sum(1 for g in gold if contains(
                     next(w for w in a_wins if w["line_id"] == g["line_id"]), g["box"])),
                 "note": "A_containment_only 只测「字形框是否在块矩形内」，不看高度上限与邻列，用于说明 A 组失分原因"}}
    (args.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    print(json.dumps({"out": str(args.out.resolve()), "windows": len(rows),
                      "blocks_skipped": len(skipped), "A": a_eval["coverage"], "B": b_eval["coverage"],
                      "A_containment_only": score["diagnostics"]["A_containment_only"],
                      "table_hash": table_hash(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
