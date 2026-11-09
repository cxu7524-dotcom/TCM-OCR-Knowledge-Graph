# -*- coding: utf-8 -*-
"""
tools/attr_guji_extra.py —— 「多余文字归因」导出（阶段六 C.4）

用途：GT 只转写**正文**，模型却会读出页眉、版心、背面透印、朱印、手写序，
这些在 CER 里全算「插入错误」。本工具把预测里**对不上 GT 的连续片段**导出来，
供人工归类，回答「CER 偏高有多少来自 GT 口径、多少来自模型能力」。

做法：
  1. 用与评测完全相同的配置跑 OCR（不读 GT 做任何决策）；
  2. 对 norm(pred) 与 norm(GT) 做编辑距离回溯，标出 pred 侧「插入」的字符；
  3. 把插入字符按位置聚成片段（间隔 ≤ GAP 视为同一片段），附上下文；
  4. 输出 JSON + 控制台表格，人工归类后写进报告诊断章节。

注意：**归因结果只用于解释，不回写任何指标。**

用法：
    python tools/attr_guji_extra.py --pages 10 --out data/reports/guji_extra_attr.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.eval_guji import norm, OFFICIAL_GT, load_gt, load_dir_pages  # noqa: E402

GAP = 2          # 插入字符间隔 ≤ 2 字算同一片段


def edit_backtrace(pred: str, gold: str) -> List[bool]:
    """返回 pred 每个字符是否「插入」（未与 GT 对齐）。"""
    n, m = len(pred), len(gold)
    # dp[i][j] = pred[:i] 与 gold[:j] 的编辑距离
    prev = list(range(m + 1))
    rows: List[List[int]] = [prev]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        pi = pred[i - 1]
        for j in range(1, m + 1):
            cost = 0 if pi == gold[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        rows.append(cur)
        prev = cur
    flags = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if pred[i - 1] == gold[j - 1] and rows[i][j] == rows[i - 1][j - 1]:
            i, j = i - 1, j - 1
            continue
        if rows[i][j] == rows[i - 1][j] + 1:          # pred 侧插入
            flags[i - 1] = True
            i -= 1
        elif rows[i][j] == rows[i][j - 1] + 1:        # GT 侧缺失（漏读）
            j -= 1
        else:                                          # 替换
            i, j = i - 1, j - 1
    while i > 0:
        flags[i - 1] = True
        i -= 1
    return flags


def runs_of(pred: str, flags: List[bool], gap: int = GAP,
            gold: str = "") -> List[Dict[str, Any]]:
    """把插入字符聚成片段，附 6 字上下文与「该片段字符是否在 GT 多重集中」比例。

    `in_gt` 用来区分两类多余文字：
      - 高（>0.6）：GT 里有这些字，说明是**同一段文字被重复读出**（背面透印 / 目录重读）；
      - 低：GT 里根本没有这些字，属于**非正文内容**（页眉 / 版心 / 朱印 / 手写序）或真多认。
    """
    from collections import Counter as _C
    idx = [i for i, f in enumerate(flags) if f]
    if not idx:
        return []
    gg = _C(gold)
    groups: List[List[int]] = [[idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][-1] <= gap + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        s, e = g[0], g[-1] + 1
        seg = pred[s:e]
        if gold:
            hit = sum(min(v, gg.get(k, 0)) for k, v in _C(seg).items())
            in_gt = round(hit / max(1, len(seg)), 3)
        else:
            in_gt = None
        out.append({
            "text": seg,
            "len": e - s,
            "at": s,
            "in_gt": in_gt,
            "left": pred[max(0, s - 6):s],
            "right": pred[e:e + 6],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="多余文字归因导出")
    ap.add_argument("--pages", type=int, default=10, help="抽检页数（按书名+页码排序取前 N）")
    ap.add_argument("--per-book", type=int, default=0,
                    help="每本书取 N 页（分层抽样，优先于 --pages）")
    ap.add_argument("--out", default="data/reports/guji_extra_attr.json")
    ap.add_argument("--rec-model", default=os.environ.get("TCM_OCR_REC_MODEL", ""))
    args = ap.parse_args()
    if args.rec_model:
        os.environ["TCM_OCR_REC_MODEL"] = args.rec_model

    from ocr.pipeline import run_ocr

    if not OFFICIAL_GT.exists():
        print("先跑 python tools/import_guji_eval.py --build-gt --images 古籍评测集/医家类",
              file=sys.stderr)
        return 2

    all_pages = load_dir_pages(_ROOT / "古籍评测集" / "医家类")
    if args.per_book:
        picked: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        for p in all_pages:                     # 已按 (书名, 页码) 排序
            b = p["book"]
            if seen.get(b, 0) < args.per_book:
                picked.append(p)
                seen[b] = seen.get(b, 0) + 1
        pages = picked
    else:
        pages = all_pages[: args.pages]
    gt = load_gt()
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(pages, start=1):
        gold = p.get("gold") or gt.get(p["gt_name"], "")
        ocr = run_ocr(p["image"], f"ATTR-{i:03d}")
        pred = norm("".join(b.text for b in sorted(ocr.blocks, key=lambda x: x.reading_order)))
        flags = edit_backtrace(pred, norm(gold))
        rs = runs_of(pred, flags, gold=norm(gold))
        extra = sum(r["len"] for r in rs)
        rows.append({
            "book": p["book"], "page_no": p["page_no"], "image": p["image"],
            "gt_chars": len(norm(gold)), "pred_chars": len(pred),
            "extra_chars": extra,
            "extra_ratio": round(extra / max(1, len(norm(gold))), 4),
            "runs": rs,
        })
        print(f"{p['book']}page_{p['page_no']}: 插入 {extra} 字 / GT {len(norm(gold))} 字 "
              f"（{extra / max(1, len(norm(gold))):.2f}），片段 {len(rs)} 个")
        for r in sorted(rs, key=lambda x: -x["len"])[:5]:
            print(f"    [{r['len']:3d}] in_gt={r['in_gt']} {r['text'][:30]!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "notice": "归因结果只用于解释 CER，不回写任何指标；GT 为机器转写且只覆盖正文",
        "pages": len(rows),
        "total_extra": sum(r["extra_chars"] for r in rows),
        "total_gt": sum(r["gt_chars"] for r in rows),
        "items": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n导出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
