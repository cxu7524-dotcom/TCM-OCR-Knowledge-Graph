# -*- coding: utf-8 -*-
"""
tools/eval_guji.py —— 真实古籍页评测（修前 / 修后对照）

输入：
  古籍评测集/*.png                     60 张真实木刻扫描页（外部评测集，非命题单位样本）
  古籍评测集/label.csv                  GT（`OCR` 列，机器转写，非人工校对）
  data/reports/guji_match.jsonl        图片 → 书目条目的反查结果（含 best / status）

切分（防泄漏，按书切分优先）：
  调参页 = 天文問答（TUNING_BOOKS）
  验证页 = 其余四本书（VALIDATION），只在最终验证时跑

指标（与 docs/真实古籍页诊断.md 同口径）：
  CER      = 编辑距离 / GT 字数（预测与 GT 均只去标点空白，**不做繁简归一化**）
  覆盖率   = 预测字符中能在 GT 里找到的比例（bag-of-chars 交集 / 预测字数）
  列数准确率 = 切出列数 / 人工数列数，误差 ≤ ±2 列算对（抽样页，见 MANUAL_COLS）
  跨列框率 = 一个框横跨 ≥2 列的框数 / 总框数
  **块数已停用**（阶段五起）：按列整列识别时块数≈列数，块数多少与质量无单调关系。

用法：
  python tools/eval_guji.py --vertical 0 --out data/reports/guji_eval_v0.json
  python tools/eval_guji.py --vertical 1 --out data/reports/guji_eval_v1.json
  python tools/eval_guji.py --vertical 1 --split validation --limit 5   # 快速抽验
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EVAL_DIR = _ROOT / "古籍评测集"
MATCH_JSONL = _ROOT / "data" / "reports" / "guji_match.jsonl"
GT_CACHE = _ROOT / "data" / "reports" / "guji_gt.json"
MANUAL_COLS = _ROOT / "data" / "annotation" / "guji_manual_cols.json"
TUNING_BOOKS = {"天文問答"}
VALIDATION_BOOKS = {"兵鏡", "杜甫-杜工部集", "兵法集要", "晏子春秋"}


def load_manual_cols() -> Dict[str, int]:
    """人工数列数（抽样页）。文件不存在时返回空 dict，列数准确率记「待测」。"""
    try:
        data = json.loads(MANUAL_COLS.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in (data.get("columns") or {}).items()}
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        return {}


def norm(s: str) -> str:
    """只去标点与空白，**不做繁简归一化**（否则测不出字形错误）。"""
    drop = set(" \t\r\n\u3000，。、；：？！“”‘’（）《》〈〉「」『』【】…—·,.;:?!()[]<>\"'`~～")
    return "".join(ch for ch in (s or "") if ch not in drop)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> Optional[float]:
    g, p = norm(gold), norm(pred)
    if not g:
        return None
    return round(levenshtein(p, g) / len(g), 4)


def coverage(pred: str, gold: str) -> Optional[float]:
    """预测字符中能在 GT 里找到的比例（多重集交集 / 预测字数）。"""
    p, g = norm(pred), norm(gold)
    if not p:
        return None
    pg, gg = Counter(p), Counter(g)
    hit = sum(min(v, gg.get(k, 0)) for k, v in pg.items())
    return round(hit / len(p), 4)


def book_of(best: str) -> str:
    for b in ("杜甫-杜工部集", "兵法集要", "晏子春秋", "天文問答", "兵鏡"):
        if best.startswith(b):
            return b
    return best[:4]


def load_gt() -> Dict[str, str]:
    if GT_CACHE.exists():
        try:
            return json.loads(GT_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    csv.field_size_limit(10 ** 9)
    out: Dict[str, str] = {}
    with (EVAL_DIR / "label.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            if name and name not in out:
                out[name] = row.get("OCR") or ""
    GT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GT_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def load_pages(split: str) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in MATCH_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    pages = []
    for r in rows:
        if r.get("status") != "matched" or not r.get("best"):
            continue
        b = book_of(r["best"])
        if split == "tuning" and b not in TUNING_BOOKS:
            continue
        if split == "validation" and b not in VALIDATION_BOOKS:
            continue
        pages.append({"image": r["image"], "gt_name": r["best"], "book": b,
                      "page_no": r.get("page_no")})
    pages.sort(key=lambda x: (x["book"], x["image"]))
    return pages


def load_dir_pages(dir_path: Path) -> List[Dict[str, Any]]:
    """按目录评测（医家类小规模验证用）：GT 取 label.csv 的 OCR 列，按文件名匹配。"""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    pages: List[Dict[str, Any]] = []
    if not dir_path.is_dir():
        return pages
    for p in sorted(dir_path.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        pages.append({"image": str(p.resolve()), "gt_name": p.name,
                      "book": p.name.split("page")[0], "page_no": None})
    return pages


def median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return round(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="真实古籍页 OCR 评测")
    ap.add_argument("--vertical", default=None, choices=["0", "1"],
                    help="竖排逐列开关（不传则沿用环境变量/配置）")
    ap.add_argument("--pick", default=None, choices=["auto", "vertical", "page"],
                    help="两路取舍策略：auto=复合分择优，vertical/page=强制（A/B 用）")
    ap.add_argument("--correct", default="0", choices=["0", "1"], help="字形纠错层开关")
    ap.add_argument("--split", default="all", choices=["all", "tuning", "validation"])
    ap.add_argument("--dir", default="", help="按目录评测（如 古籍评测集/医家类），GT 按文件名取 label.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.vertical is not None:
        os.environ["TCM_OCR_VERTICAL_SPLIT"] = args.vertical
    if args.pick:
        os.environ["TCM_OCR_PICK"] = args.pick
    os.environ["TCM_OCR_CORRECT"] = args.correct

    from ocr.pipeline import run_ocr          # 环境变量设置后再导入，保证 CONFIG 正确
    from ocr.vertical import cross_column_rate, page_layout

    gt = load_gt()
    manual = load_manual_cols()
    pages = load_dir_pages(Path(args.dir)) if args.dir else load_pages(args.split)
    if args.limit:
        pages = pages[:args.limit]

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, p in enumerate(pages, start=1):
        img = Path(p["image"])
        if not img.is_absolute():
            img = EVAL_DIR / img
        gold = gt.get(p["gt_name"], "")
        if not img.exists() or not gold:
            results.append({**p, "status": "skip", "reason": "缺图或缺 GT"})
            continue
        try:
            ocr = run_ocr(str(img), f"GUJI-{i:03d}")
            text = "".join(b.text for b in sorted(ocr.blocks, key=lambda x: x.reading_order))
            lay = page_layout(str(img))
            cols = lay["cols"]
            cross = cross_column_rate(ocr.blocks, cols)
            mcols = (manual.get(p["image"]) or manual.get(p["gt_name"])
                     or manual.get(Path(p["image"]).name))
            cols_ok = None if not mcols else (abs(len(cols) - mcols) <= 2)
            results.append({
                **p, "status": "ok", "engine": ocr.engine,
                "blocks": len(ocr.blocks), "chars": len(norm(text)),
                "gt_chars": len(norm(gold)),
                "cer": cer(text, gold), "coverage": coverage(text, gold),
                "columns": len(ocr.layout.cols), "cols_detected": len(cols),
                "cols_manual": mcols, "cols_ok": cols_ok,
                "spread": len(lay["halves"]) > 1, "halves": lay["halves"],
                "cross_boxes": cross["cross"], "cross_total": cross["total"],
                "cross_rate": cross["rate"],
                "text_head": text[:80],
            })
        except Exception as exc:
            results.append({**p, "status": "error", "reason": type(exc).__name__})
        if i % 5 == 0:
            print(f"  {i}/{len(pages)} …", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    cers = [r["cer"] for r in ok if r.get("cer") is not None]
    covs = [r["coverage"] for r in ok if r.get("coverage") is not None]
    sampled = [r for r in ok if r.get("cols_manual")]
    cross_total = sum(r["cross_total"] for r in ok)
    cross_boxes = sum(r["cross_boxes"] for r in ok)
    summary = {
        "tag": args.tag or f"vertical={args.vertical}",
        "split": args.split,
        "pages": len(pages), "ok": len(ok),
        "cer_median": median(cers), "cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
        "cer_best": min(cers) if cers else None, "cer_worst": max(cers) if cers else None,
        "coverage_median": median(covs),
        "cols_accuracy": (round(sum(1 for r in sampled if r["cols_ok"]) / len(sampled), 4)
                          if sampled else None),
        "cols_sampled": len(sampled),
        "cols_ok_pages": [r["image"] for r in sampled if r["cols_ok"]],
        "cols_bad_pages": [(r["image"], r["cols_detected"], r["cols_manual"])
                           for r in sampled if not r["cols_ok"]],
        "cross_box_rate": round(cross_boxes / cross_total, 4) if cross_total else None,
        "cross_boxes": cross_boxes, "cross_total": cross_total,
        "engines": dict(Counter(r["engine"] for r in ok)),
        "spread_pages": sum(1 for r in ok if r.get("spread")),
        "chars_median": median([r["chars"] for r in ok]),
        "gt_chars_median": median([r["gt_chars"] for r in ok]),
        "books": dict(Counter(r["book"] for r in ok)),
        "cost_s": round(time.time() - t0, 1),
        "config": {"vertical_split": os.environ.get("TCM_OCR_VERTICAL_SPLIT"),
                   "correct": os.environ.get("TCM_OCR_CORRECT"),
                   "pick": os.environ.get("TCM_OCR_PICK", "auto"),
                   "rec_model": os.environ.get("TCM_OCR_REC_MODEL", "")},
        "notice": "外部古籍评测集，非命题单位样本；GT 为机器转写（label.csv），非人工校对",
    }
    out = Path(args.out) if args.out else _ROOT / "data" / "reports" / "guji_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "pages": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
