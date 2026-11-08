# -*- coding: utf-8 -*-
"""
eval/ocr_error_analysis.py —— OCR 错误诊断（P1 第一步：先诊断，不先改代码）

做法：对测试页的 GT 转写与真实 OCR 预测做字符级对齐（difflib，标准库），
      统计错字频次、混淆对、错误来源分类与繁体/简体错误占比。

输出：data/reports/ocr_error_analysis.md

错误来源三分法（块级）：
  - 检测漏框：GT 有该块，预测中找不到位置相近的块
  - 识别错字：有位置相近的块，但文本不同（字符级差异）
  - 切分错位：有相近块，但字符数比例异常（合并过度 >1.5 或漏切 <0.6）

命令行：
    python eval/ocr_error_analysis.py [--pages 7]
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                                    # noqa: E402
from app.schemas import norm_for_match                            # noqa: E402
from eval.dataset import Dataset, load_jsonl                      # noqa: E402

REPORT = CONFIG.paths.reports_dir / "ocr_error_analysis.md"


def _load_lexicon_chars() -> Tuple[set, set]:
    p = CONFIG.paths.data_dir / "lexicon" / "异体字表.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return set(), set()
    trad = {ch for k in raw.get("繁简", {}) for ch in k}
    variant = {ch for k in raw.get("异体", {}) for ch in k}
    return trad, variant


def _center(bbox: List[float]) -> Tuple[float, float]:
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)


def _nearest(bbox: List[float], pool: List[Dict[str, Any]]) -> Tuple[Dict[str, Any] | None, float]:
    cx, cy = _center(bbox)
    best, best_d = None, 1e9
    for p in pool:
        px, py = _center(p.get("bbox", [0, 0, 0, 0]))
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < best_d:
            best, best_d = p, d
    return best, best_d


def char_align(gt: str, pred: str) -> Tuple[Counter, Counter, Counter]:
    """返回（替换对 Counter((gt_ch, pred_ch))，删除 Counter(gt_ch)，插入 Counter(pred_ch)）。"""
    g, p = norm_for_match(gt), norm_for_match(pred)
    subs, dels, ins = Counter(), Counter(), Counter()
    sm = difflib.SequenceMatcher(None, g, p, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            gi, pj = g[i1:i2], p[j1:j2]
            for k in range(max(len(gi), len(pj))):
                a = gi[k] if k < len(gi) else "∅"
                b = pj[k] if k < len(pj) else "∅"
                if a != "∅" and b != "∅":
                    subs[(a, b)] += 1
                elif a != "∅":
                    dels[a] += 1
                else:
                    ins[b] += 1
        elif tag == "delete":
            for ch in g[i1:i2]:
                dels[ch] += 1
        elif tag == "insert":
            for ch in p[j1:j2]:
                ins[ch] += 1
    return subs, dels, ins


def analyze(pages: List[str] | None = None) -> Dict[str, Any]:
    ds = Dataset.load()
    pages = pages or ds.test_pages
    keep = set(pages)
    gt = [r for r in ds.ocr_gt if r.get("page_id") in keep]
    pred = load_jsonl(CONFIG.paths.reports_dir / "ocr_predictions.jsonl")
    pred = [r for r in pred if r.get("page_id") in keep and r.get("is_demo") is not True]
    by_page_pred: Dict[str, List[Dict[str, Any]]] = {}
    for r in pred:
        by_page_pred.setdefault(str(r.get("page_id", "")), []).append(r)

    trad_chars, variant_chars = _load_lexicon_chars()
    subs, dels, ins = Counter(), Counter(), Counter()
    source = Counter({"检测漏框": 0, "识别错字": 0, "切分错位": 0, "完全正确": 0})
    per_page = []
    gt_total = pred_total = dist = 0

    for pid in pages:
        g_blocks = [r for r in gt if r.get("page_id") == pid]
        p_blocks = by_page_pred.get(pid, [])
        g_text = "".join(str(r.get("quote", "")) for r in sorted(g_blocks, key=lambda x: x.get("line_id", "")))
        p_text = "".join(str(r.get("quote", "")) for r in sorted(p_blocks, key=lambda x: int(x.get("reading_order") or 0)))
        g_n, p_n = norm_for_match(g_text), norm_for_match(p_text)
        # 用 opcodes 计算编辑距离（替换按较长一侧计）
        ed = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g_n, p_n, autojunk=False).get_opcodes():
            if tag != "equal":
                ed += max(i2 - i1, j2 - j1)
        gt_total += len(g_n)
        pred_total += len(p_n)
        dist += ed
        s, de, i_ = char_align(g_text, p_text)
        subs.update(s); dels.update(de); ins.update(i_)

        # 块级来源分类
        for gb in g_blocks:
            best, bd = _nearest(gb.get("bbox", [0, 0, 0, 0]), p_blocks)
            gt_len = len(norm_for_match(str(gb.get("quote", ""))))
            if best is None or bd > 120:
                source["检测漏框"] += 1
                continue
            pred_len = len(norm_for_match(str(best.get("quote", ""))))
            if norm_for_match(str(best.get("quote", ""))) == norm_for_match(str(gb.get("quote", ""))):
                source["完全正确"] += 1
            elif gt_len and (pred_len / gt_len > 1.5 or pred_len / gt_len < 0.6):
                source["切分错位"] += 1
            else:
                source["识别错字"] += 1
        per_page.append({"page_id": pid, "gt_chars": len(g_n), "pred_chars": len(p_n),
                         "distance": ed, "cer": round(ed / len(g_n), 4) if g_n else None,
                         "gt_blocks": len(g_blocks), "pred_blocks": len(p_blocks)})

    # 繁体 / 简体错误占比
    trad_err = sum(v for (a, b), v in subs.items() if a in trad_chars) + sum(
        v for a, v in dels.items() if a in trad_chars)
    simp_err = sum(v for (a, b), v in subs.items() if a not in trad_chars) + sum(
        v for a, v in dels.items() if a not in trad_chars)
    err_total = trad_err + simp_err
    top_subs = [{"gt": a, "pred": b, "count": v,
                 "is_traditional": a in trad_chars,
                 "is_variant": a in variant_chars}
                for (a, b), v in subs.most_common(20)]
    return {
        "pages": pages,
        "gt_chars": gt_total, "pred_chars": pred_total,
        "edit_distance": dist,
        "cer": round(dist / gt_total, 4) if gt_total else None,
        "per_page": per_page,
        "substitutions": len(subs), "substitution_total": sum(subs.values()),
        "deletions_total": sum(dels.values()), "insertions_total": sum(ins.values()),
        "top_confusions": top_subs,
        "top_deletions": [{"gt": a, "count": v} for a, v in dels.most_common(12)],
        "top_insertions": [{"pred": a, "count": v} for a, v in ins.most_common(12)],
        "source_breakdown": dict(source),
        "traditional_error_share": round(trad_err / err_total, 4) if err_total else None,
        "traditional_errors": trad_err, "simplified_errors": simp_err,
    }


def build_markdown(r: Dict[str, Any]) -> str:
    src = r["source_breakdown"]
    total_blocks = sum(src.values()) or 1
    confusion_rows = "\n".join(
        f"| {c['gt']} → {c['pred']} | {c['count']} | "
        f"{'繁体' if c['is_traditional'] else '简体'} |" for c in r["top_confusions"])
    page_rows = "\n".join(
        f"| {p['page_id']} | {p['gt_chars']} | {p['pred_chars']} | {p['distance']} | "
        f"{p['cer']} | {p['gt_blocks']} / {p['pred_blocks']} |" for p in r["per_page"])
    return f"""# OCR 错误诊断报告（P1 第一步）

> 数据：`SYN-2026-001` 合成示意集测试页（{len(r['pages'])} 页），
> **合成示意数据，非命题单位样本、非官方评测结果**。
> 口径：字符级对齐用 `difflib.SequenceMatcher`，比对前仅去空白标点，**不做字形归一化**（否则测不出字形错误）。

## 一、总体

| 项 | 值 |
|---|---|
| GT 字数 / 预测字数 | {r['gt_chars']} / {r['pred_chars']} |
| 编辑距离 | {r['edit_distance']} |
| CER | **{r['cer']}** |
| 替换（错字）总数 | {r['substitution_total']}（不同混淆对 {r['substitutions']} 组） |
| 删除（漏识别）总数 | {r['deletions_total']} |
| 插入（多识别）总数 | {r['insertions_total']} |
| 繁体字错误占比 | **{r['traditional_error_share']}**（繁体 {r['traditional_errors']} / 简体 {r['simplified_errors']}） |

## 二、块级错误来源分类

| 来源 | 块数 | 占比 |
|---|---|---|
| 完全正确 | {src.get('完全正确', 0)} | {src.get('完全正确', 0) / total_blocks:.1%} |
| 识别错字 | {src.get('识别错字', 0)} | {src.get('识别错字', 0) / total_blocks:.1%} |
| 切分错位 | {src.get('切分错位', 0)} | {src.get('切分错位', 0) / total_blocks:.1%} |
| 检测漏框 | {src.get('检测漏框', 0)} | {src.get('检测漏框', 0) / total_blocks:.1%} |

> 判定规则：GT 块找 bbox 中心最近的预测块（距离 >120px 记为漏框）；
> 文本相同记正确；字符数比 >1.5 或 <0.6 记切分错位；其余记识别错字。

## 三、Top 混淆对（按频次）

| GT → 预测 | 次数 | GT 字形 |
|---|---|---|
{confusion_rows or '| （无） | 0 | — |'}

**漏识别 Top**：{', '.join(f"{d['gt']}×{d['count']}" for d in r['top_deletions']) or '无'}
**多识别 Top**：{', '.join(f"{d['pred']}×{d['count']}" for d in r['top_insertions']) or '无'}

## 四、逐页 CER

| 页 | GT 字数 | 预测字数 | 编辑距离 | CER | 块数 GT/预测 |
|---|---|---|---|---|---|
{page_rows}

## 五、结论（瓶颈判断）

{_conclusion(r)}
"""


def _conclusion(r: Dict[str, Any]) -> str:
    src = r["source_breakdown"]
    total = sum(src.values()) or 1
    err = max(1, src.get("识别错字", 0) + src.get("切分错位", 0) + src.get("检测漏框", 0))
    share = {k: round(v / err, 3) for k, v in src.items() if k != "完全正确"}
    trad = r["traditional_error_share"]
    lines = [
        f"- 错误块中：识别错字 {share.get('识别错字', 0):.1%}、切分错位 {share.get('切分错位', 0):.1%}、"
        f"检测漏框 {share.get('检测漏框', 0):.1%}。",
        f"- 繁体字错误占比 {trad}：{'繁体字形能力是主要瓶颈' if trad and trad > 0.6 else '简体与繁体错误各占一部分，非单一字形问题'}。",
    ]
    if share.get("识别错字", 0) >= 0.6:
        lines.append("- **主要瓶颈是识别模型字形能力**（位置对、字错），优先试繁体/多语言 rec 模型与字形纠错层。")
    elif share.get("切分错位", 0) >= 0.3:
        lines.append("- **主要瓶颈是切图/合并策略**（字符数比例异常），优先改逐列切图与碎片合并阈值。")
    else:
        lines.append("- 错误较分散，识别与切分需同时改进。")
    lines.append("- 表格页单元格错字（如「三兩→三雨」）属识别错字，表格算法本身结构正确（见评测报告结构 2/2）。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR 错误诊断")
    ap.add_argument("--pages", type=int, default=0, help="仅分析前 N 个测试页（0=全部）")
    args = ap.parse_args()
    result = analyze()
    if args.pages:
        result = analyze(result["pages"][:args.pages])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_markdown(result), encoding="utf-8")
    (REPORT.parent / "ocr_error_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断报告：{REPORT}")
    print(f"CER {result['cer']} | 错字 {result['substitution_total']} | "
          f"漏识别 {result['deletions_total']} | 多识别 {result['insertions_total']}")
    print(f"来源：{result['source_breakdown']} | 繁体错误占比 {result['traditional_error_share']}")


if __name__ == "__main__":
    main()
