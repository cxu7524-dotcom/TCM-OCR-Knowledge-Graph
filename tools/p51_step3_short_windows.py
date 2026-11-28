# -*- coding: utf-8 -*-
"""第 51 页 2 字高短窗生成（第 3 步短窗补测，第一步：只裁图，不推理）。

与第二步的唯一差别是窗口高度方案：
  第二步 B 组：h = min(4H, 块高)
  本轮 S 组：  h = min(2H, 块高)
其余完全沿用：宽 = 块左右边界；步长 = max(1, round(H/2))；从块顶起步；
末尾补一个对齐块底的窗口；重复坐标去重；H 直接取第二步 windows.json 的冻结值。

只读原图 + 第二步窗口/字高参数；人工标注不参与。

用法：
  python tools/p51_step3_short_windows.py --image <原图> --windows <第二步 windows.json> \
      --out <新目录>
  python tools/p51_step3_short_windows.py --compare <目录A> <目录B>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.vertical import imread_unicode  # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_win", ROOT / "tools/p51_local_windows.py")
_win = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_win)

EXPECTED_TABLE_HASH = _win.EXPECTED_TABLE_HASH if hasattr(_win, "EXPECTED_TABLE_HASH") else None
BASE_TABLE_HASH = "df31b25047b94b12eeb449f5298301864101d87c672c4ca796600d7f8eacfa2b"
H_MULT = 2          # 本轮唯一变量：窗口高度 = 2 × 估计字高


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table_hash(rows: list) -> str:
    key = ";".join("%s|%s|%d|%d|%d|%d|%d" % (r["window_id"], r["line_id"], r["x1"], r["y1"],
                                             r["x2"], r["y2"], r["est_char_h"]) for r in rows)
    return hashlib.sha256(key.encode()).hexdigest()


def short_windows(blocks: list) -> list:
    """直接调用第二步的 windows_for（mult=2），只把 ID 前缀与组名改成本轮标识。"""
    out = []
    for b in blocks:
        for w in _win.windows_for(b, H_MULT):
            w = dict(w)
            w["window_id"] = "S:" + w["window_id"][2:]   # B:R019:W01 -> S:R019:W01
            w["group"] = "S"
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        a = json.loads((args.compare[0] / "short_windows.json").read_text(encoding="utf8"))
        b = json.loads((args.compare[1] / "short_windows.json").read_text(encoding="utf8"))
        same = (a["table_hash"] == b["table_hash"] and len(a["windows"]) == len(b["windows"]))
        print(json.dumps({"hash": [a["table_hash"][:16], b["table_hash"][:16]],
                          "windows": [len(a["windows"]), len(b["windows"])],
                          "identical": same}, ensure_ascii=False))
        return 0 if same else 1

    if not (args.image and args.windows and args.out):
        raise SystemExit("需要 --image/--windows/--out")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "crops").mkdir()

    img = imread_unicode(str(args.image))
    if img is None:
        raise SystemExit("原图读取失败")
    IH, IW = img.shape[:2]
    wd = json.loads(args.windows.read_text(encoding="utf8"))
    if wd.get("table_hash") != BASE_TABLE_HASH:
        raise SystemExit("第二步窗口指纹不符：%s" % wd.get("table_hash"))
    blocks = wd["blocks"]

    rows = short_windows(blocks)
    rows = [r for r in rows
            if 0 <= r["x1"] < r["x2"] <= IW and 0 <= r["y1"] < r["y2"] <= IH]
    for r in rows:
        p = args.out / "crops" / ("%s.png" % r["window_id"].replace(":", "_"))
        patch = img[r["y1"]:r["y2"], r["x1"]:r["x2"]]
        cv2.imencode(".png", patch)[1].tofile(str(p))
        r["crop_path"] = str(p)

    with (args.out / "short_windows.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["window_id", "group", "line_id", "block_ref", "x1", "y1",
                                           "x2", "y2", "est_char_h", "start_y", "height", "step",
                                           "formula", "crop_path"])
        wr.writeheader()
        wr.writerows(rows)

    meta = {"params": {"height": "min(2*H, block_h)", "width": "block x bounds",
                       "step": "max(1, round(H/2))", "tail": "append bottom-aligned window",
                       "H_source": "第二步 windows.json 冻结的 est_char_h",
                       "only_variable_vs_step2": "H_MULT 4 -> 2"},
            "image": str(args.image.resolve()), "image_sha256": sha256_of(args.image),
            "image_size": [IW, IH],
            "base_windows_json": str(args.windows.resolve()),
            "base_table_hash": wd["table_hash"],
            "n_blocks": len({r["line_id"] for r in rows}), "n_windows": len(rows),
            "table_hash": table_hash(rows), "windows": rows}
    (args.out / "short_windows.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf8")

    # 预览：短窗绿框 + 第二步 4 字高窗蓝框（对照）
    canvas = img.copy()
    for w in wd["windows"]:
        if w["group"] == "B":
            cv2.rectangle(canvas, (w["x1"], w["y1"]), (w["x2"] - 1, w["y2"] - 1), (255, 120, 0), 1)
    for r in rows:
        cv2.rectangle(canvas, (r["x1"], r["y1"]), (r["x2"] - 1, r["y2"] - 1), (0, 170, 0), 1)
    cv2.imencode(".png", canvas)[1].tofile(str(args.out / "preview_short_windows.png"))

    from collections import Counter
    per = Counter(r["line_id"] for r in rows)
    print(json.dumps({"out": str(args.out.resolve()), "windows": len(rows),
                      "blocks": meta["n_blocks"], "table_hash": table_hash(rows),
                      "per_block_targets": {k: per.get(k) for k in ("R018", "R019", "R020", "R021")}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
