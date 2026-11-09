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
OFFICIAL_GT = _ROOT / "data" / "annotation" / "official" / "ocr_transcription.jsonl"
MANUAL_COLS = _ROOT / "data" / "annotation" / "guji_manual_cols.json"
TUNING_BOOKS = {"天文問答"}
VALIDATION_BOOKS = {"兵鏡", "杜甫-杜工部集", "兵法集要", "晏子春秋"}


def load_manual_cols() -> Dict[str, int]:
    """人工数列数（抽样页）。文件不存在时返回空 dict，列数准确率记「待测」。

    阶段七 P4 修掉一个测量 bug：键若只是裸文件名 `page_10.png`，在「医家类/<书名>/page_N.png」
    子目录布局下会和通用古籍集的同名页**串页**——阶段六的 2/6 就是这么来的。
    因此只接受带书名的键（或带 / 的相对路径），裸文件名一律忽略。
    """
    import re as _re
    try:
        data = json.loads(MANUAL_COLS.read_text(encoding="utf-8"))
        out = {}
        for k, v in (data.get("columns") or {}).items():
            if "/" in k or not _re.fullmatch(r"page_\d+\.(?:png|jpg)", str(k), _re.I):
                out[k] = int(v)
        return out
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


def cer_sorted(pred: str, gold: str) -> Optional[float]:
    """**顺序无关**的 CER（诊断量，不参与任何判定）。

    把两侧字符各自排序后再算编辑距离：若 GT 的字符顺序与页面阅读顺序不一致，
    严格 CER 会很高而本值很低。差值即「顺序问题」的贡献，用来区分
    「认错了」与「认对了但排错序」。
    """
    g, p = norm(gold), norm(pred)
    if not g:
        return None
    return round(levenshtein("".join(sorted(p)), "".join(sorted(g))) / len(g), 4)


def match_stats(pred: str, gold: str) -> Dict[str, Any]:
    """覆盖率 + 插入率（阶段六新增，只用于解释 CER，不替代主指标）。

    覆盖率 = 预测字符中能在 GT 里找到的字数 / 预测字数 —— 衡量「认错的字」
    插入率 = (预测字数 - 匹配字数) / GT 字数 —— 衡量「多认出来的字」
             （页眉 / 版心 / 背面透印 / 朱印 / 手写序，GT 只转写正文时必然出现）
    """
    p, g = norm(pred), norm(gold)
    out = {"pred_chars": len(p), "gt_chars": len(g), "matched": 0,
           "coverage": None, "insert_rate": None}
    if not p or not g:
        return out
    pg, gg = Counter(p), Counter(g)
    hit = sum(min(v, gg.get(k, 0)) for k, v in pg.items())
    out["matched"] = hit
    out["coverage"] = round(hit / len(p), 4)
    out["insert_rate"] = round((len(p) - hit) / len(g), 4)
    return out


def coverage(pred: str, gold: str) -> Optional[float]:
    """预测字符中能在 GT 里找到的比例（多重集交集 / 预测字数）。"""
    return match_stats(pred, gold)["coverage"]


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
    """按目录评测：优先用 `import_guji_eval.py --build-gt` 产出的正式 GT 清单。

    阶段六起支持三种布局：
      ① `古籍评测集/医家类/<书名>/page_N.png`（子目录，书名在目录名里）
      ② `古籍评测集/医家类/书名page_N.png`（扁平）
      ③ 正式 GT（`data/annotation/official/ocr_transcription.jsonl`，含 image_path）
    有正式 GT 时以它为准（页数、书名、页码、原文都来自 GT，避免图片目录里混入无标注页）。
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    pages: List[Dict[str, Any]] = []
    if OFFICIAL_GT.exists():
        rows = [json.loads(l) for l in OFFICIAL_GT.read_text(encoding="utf-8").splitlines() if l.strip()]
        root = dir_path.resolve()
        for r in rows:
            ip = Path(str(r.get("image_path") or ""))
            if not ip.is_file():
                continue
            try:
                ip.resolve().relative_to(root)
            except ValueError:
                continue                      # GT 里的图不在本次评测目录下
            pages.append({"image": str(ip.resolve()), "gt_name": r.get("source_name", ip.name),
                          "book": r.get("book", ""), "page_no": r.get("page_no"),
                          "page_id": r.get("page_id", ""), "gold": r.get("text", "")})
        if pages:
            return sorted(pages, key=lambda x: (str(x["book"]), x.get("page_no") or 0))

    if not dir_path.is_dir():
        return pages
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            pages.append({"image": str(p.resolve()), "gt_name": p.name,
                          "book": p.name.split("page")[0], "page_no": None})
        elif p.is_dir():                       # 子目录布局：书名取目录名
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in exts:
                    pages.append({"image": str(q.resolve()),
                                  "gt_name": f"{p.name}{q.name}",
                                  "book": p.name, "page_no": None})
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
        gold = p.get("gold") or gt.get(p["gt_name"], "")
        if not img.exists() or not gold:
            results.append({**p, "status": "skip", "reason": "缺图或缺 GT"})
            continue
        try:
            ocr = run_ocr(str(img), f"GUJI-{i:03d}")
            text = "".join(b.text for b in sorted(ocr.blocks, key=lambda x: x.reading_order))
            lay = page_layout(str(img))
            cols = lay["cols"]
            cross = cross_column_rate(ocr.blocks, cols)
            ms = match_stats(text, gold)
            mcols = (manual.get(p["image"]) or manual.get(p["gt_name"])
                     or manual.get(Path(p["image"]).name))
            cols_ok = None if not mcols else (abs(len(cols) - mcols) <= 2)
            results.append({
                **p, "status": "ok", "engine": ocr.engine,
                "blocks": len(ocr.blocks), "chars": ms["pred_chars"],
                "direction": ocr.layout.direction,
                "gt_chars": ms["gt_chars"], "matched": ms["matched"],
                "cer": cer(text, gold), "coverage": ms["coverage"],
                "insert_rate": ms["insert_rate"],
                "cer_sorted": cer_sorted(text, gold),
                "columns": len(ocr.layout.cols), "cols_detected": len(cols),
                "cols_manual": mcols, "cols_ok": cols_ok,
                "spread": len(lay["halves"]) > 1, "halves": lay["halves"],
                "cross_boxes": cross["cross"], "cross_total": cross["total"],
                "cross_rate": cross["rate"],
                "text_head": text[:80],
            })
        except Exception as exc:
            results.append({**p, "status": "error", "reason": type(exc).__name__})
        if i % 10 == 0:
            print(f"  {i}/{len(pages)} …", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    cers = [r["cer"] for r in ok if r.get("cer") is not None]
    covs = [r["coverage"] for r in ok if r.get("coverage") is not None]
    inss = [r["insert_rate"] for r in ok if r.get("insert_rate") is not None]
    sampled = [r for r in ok if r.get("cols_manual")]
    cross_total = sum(r["cross_total"] for r in ok)
    cross_boxes = sum(r["cross_boxes"] for r in ok)

    by_book: Dict[str, Any] = {}
    for b in sorted({r["book"] for r in ok}):
        rows_b = [r for r in ok if r["book"] == b]
        cb = [r["cer"] for r in rows_b if r.get("cer") is not None]
        vb = [r["coverage"] for r in rows_b if r.get("coverage") is not None]
        ib = [r["insert_rate"] for r in rows_b if r.get("insert_rate") is not None]
        by_book[b] = {
            "pages": len(rows_b),
            "cer_median": median(cb),
            "cer_mean": round(sum(cb) / len(cb), 4) if cb else None,
            "coverage_median": median(vb),
            "insert_rate_median": median(ib),
            "gt_chars_median": median([r["gt_chars"] for r in rows_b]),
            "spread_pages": sum(1 for r in rows_b if r.get("spread")),
        }
    summary = {
        "tag": args.tag or f"vertical={args.vertical}",
        "split": args.dir or args.split,
        "pages": len(pages), "ok": len(ok),
        "cer_median": median(cers), "cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
        "cer_best": min(cers) if cers else None, "cer_worst": max(cers) if cers else None,
        "coverage_median": median(covs),
        "insert_rate_median": median(inss),
        "insert_rate_mean": round(sum(inss) / len(inss), 4) if inss else None,
        "cer_sorted_median": median([r["cer_sorted"] for r in ok
                                     if r.get("cer_sorted") is not None]),
        "by_book": by_book,
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
