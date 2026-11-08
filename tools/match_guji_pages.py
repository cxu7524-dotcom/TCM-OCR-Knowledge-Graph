# -*- coding: utf-8 -*-
"""
tools/match_guji_pages.py —— 用真实古籍页跑 OCR，并反查它对应哪条外部标注

要解决的问题：
  用户下载的 60 张真实古籍页被浏览器改名成 `page_11 (2).png`，书名丢了，
  无法反查对应 `label.csv` 的哪一条。本脚本用「识别文本 vs 标注原文」的字面
  重合度做反查，顺便给出**真实古籍页的真实识别水平**（示例合成页那套 8.9% 不可信）。

匹配口径：
  1. 候选集：label.csv 中同页码的条目（页码相同才比，否则 2584 条全比太慢且易错配）
  2. 主分数：字符 bigram Jaccard（对乱序、漏字比编辑距离稳）
  3. 次分数：difflib 序列相似度（看整段连贯性）
  4. 阈值：Jaccard ≥ 0.30 且与次优差距 ≥ 0.05 才认，否则标 ambiguous

输出：
  data/reports/guji_match.jsonl   逐图匹配结果（含识别文本，便于人工复核）
  控制台摘要：匹配成功数、真实 CER、识别字数分布

用法：
    python tools/match_guji_pages.py --limit 3      # 先试 3 张看效果
    python tools/match_guji_pages.py                # 全部 60 张
    python tools/match_guji_pages.py --dir 古籍评测集/医家类
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "古籍评测集" / "label.csv"
OUT_PATH = ROOT / "data" / "reports" / "guji_match.jsonl"

PAGE_RE = re.compile(r"page_(\d+)", re.I)


def norm(s: str) -> str:
    """比对归一化：只去空白与常见标点，不改字（与 app.schemas.norm_for_match 同口径）。"""
    from app.schemas import norm_for_match
    return norm_for_match(s or "")


def bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def load_candidates() -> Dict[int, List[Dict[str, str]]]:
    """按页码索引 label.csv：{23: [条目, ...]}。"""
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    idx: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        name = (r.get("name") or "").strip()
        ocr = (r.get("OCR") or r.get("ocr") or "").strip()
        m = PAGE_RE.search(name)
        if not m or not ocr:
            continue
        idx.setdefault(int(m.group(1)), []).append({
            "name": name,
            "type": (r.get("type") or "").strip(),
            "ocr": ocr,
        })
    return idx


def ocr_page(img: Path, adapter) -> Tuple[str, int, float]:
    """真实识别一页，返回（拼接文本, 块数, 耗时秒）。失败返回空串。"""
    t0 = time.time()
    try:
        res = adapter.recognize(str(img), "MATCH")
        text = "".join(b.text for b in res.blocks if b.text)
        return text, len(res.blocks), time.time() - t0
    except Exception as exc:
        print(f"  [警告] {img.name} 识别异常：{type(exc).__name__}", file=sys.stderr)
        return "", 0, time.time() - t0


def cer(pred: str, gold: str) -> Optional[float]:
    """字符错误率（归一化后）：编辑距离 / GT 字数。"""
    from eval.metrics import levenshtein
    p, g = norm(pred), norm(gold)
    if not g:
        return None
    return round(levenshtein(p, g) / len(g), 4)


def contain(pred: str, gold: str) -> float:
    """识别覆盖率：pred 里有多少字能在 gold 中找到（difflib 匹配块 / pred 长度）。

    真实古籍页常出现「只认出半页」的情况，整页 Jaccard 会被漏读稀释；
    contain 衡量「认出来的那部分对不对」，更贴合匹配需求。
    """
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    sm = difflib.SequenceMatcher(None, p, g, autojunk=False)
    return round(sum(b.size for b in sm.get_matching_blocks()) / len(p), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="真实古籍页 OCR + 反查标注")
    ap.add_argument("--dir", default="古籍评测集", help="图片目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    ap.add_argument("--threshold", type=float, default=0.10, help="Jaccard 最低线（真实页漏读严重，默认 0.10）")
    args = ap.parse_args()

    img_dir = ROOT / args.dir
    imgs = sorted(p for p in img_dir.glob("*.png"))
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        print(f"目录里没有 png：{img_dir}", file=sys.stderr)
        return 2

    print(f"待识别 {len(imgs)} 张；加载本地 OCR 模型……")
    from ocr.recognize import RapidOcrAdapter
    adapter = RapidOcrAdapter()
    if not adapter.available():
        print("OCR 依赖不可用，无法继续（不伪造结果）", file=sys.stderr)
        return 3

    cand = load_candidates()
    print(f"候选标注 {sum(len(v) for v in cand.values())} 条，覆盖 {len(cand)} 个页码\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    t_all = time.time()

    for i, img in enumerate(imgs, start=1):
        m = PAGE_RE.search(img.name)
        page_no = int(m.group(1)) if m else 0
        text, nblk, cost = ocr_page(img, adapter)
        pool = cand.get(page_no, [])
        scored = []
        for c in pool:
            j = jaccard(text, c["ocr"])
            r = difflib.SequenceMatcher(None, norm(text)[:1500], norm(c["ocr"])[:1500]).ratio()
            scored.append((j, r, c))
        scored.sort(key=lambda x: -x[0])
        best = scored[0] if scored else (0.0, 0.0, None)
        second = scored[1][0] if len(scored) > 1 else 0.0
        gap = best[0] - second
        # 真实古籍页漏读严重，绝对 Jaccard 天然偏低；改用「相对优势」判据：
        # top1 达到最低线，且要么拉开差距，要么领先次优 2 倍以上。
        ok = bool(best[2]) and best[0] >= args.threshold and (
            gap >= 0.05 or (second > 0 and best[0] >= 2 * second))
        status = "matched" if ok else ("ambiguous" if best[2] and best[0] >= args.threshold else "no_match")
        cov = contain(text, best[2]["ocr"]) if best[2] else 0.0

        rec = {
            "image": img.name,
            "page_no": page_no,
            "ocr_chars": len(text),
            "ocr_blocks": nblk,
            "cost_s": round(cost, 2),
            "candidates": len(pool),
            "best": best[2]["name"] if best[2] else None,
            "best_type": best[2]["type"] if best[2] else None,
            "jaccard": round(best[0], 4),
            "seq_ratio": round(best[1], 4),
            "gap": round(gap, 4),
            "contain": cov,
            "status": status,
            "cer": cer(text, best[2]["ocr"]) if (ok and best[2]) else None,
            "ocr_text": text,
        }
        rows.append(rec)
        print("[%2d/%d] %-22s 块%3d %5.1fs 候选%2d  最佳=%-22s J=%.3f 覆盖=%.2f %s"
              % (i, len(imgs), img.name, nblk, cost, len(pool),
                 (best[2]["name"] if best[2] else "-")[:22], best[0], cov, status))

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    matched = [r for r in rows if r["status"] == "matched"]
    cers = [r["cer"] for r in matched if r["cer"] is not None]
    print(f"\n总耗时 {time.time() - t_all:.1f}s → {OUT_PATH}")
    print(f"匹配成功 {len(matched)}/{len(rows)}，ambiguous "
          f"{sum(1 for r in rows if r['status'] == 'ambiguous')}，"
          f"no_match {sum(1 for r in rows if r['status'] == 'no_match')}")
    if cers:
        cers_sorted = sorted(cers)
        print("匹配页真实 CER：中位 %.4f / 最好 %.4f / 最差 %.4f（n=%d）"
              % (cers_sorted[len(cers_sorted) // 2], cers_sorted[0], cers_sorted[-1], len(cers)))
    covs = sorted(r["contain"] for r in rows if r["contain"])
    if covs:
        print("识别覆盖率（认出的字有多少落在 GT 里）：中位 %.3f / 最好 %.3f / 最差 %.3f"
              % (covs[len(covs) // 2], covs[-1], covs[0]))
        low = sum(1 for c in covs if c < 0.5)
        print(f"覆盖率 < 0.5 的页：{low}/{len(covs)}（漏读或错字严重）")
    print("提示：这批图多为天文算法类，仅用于验证真实页识别水平；医家类请另行下载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
