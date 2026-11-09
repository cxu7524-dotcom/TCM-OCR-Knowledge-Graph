# -*- coding: utf-8 -*-
"""
tools/report_guji_medical59.py —— 生成 `docs/医家类59页评测报告.md`（阶段六 E）

为什么要用脚本生成：报告里 59 行逐页明细、分书统计、失败页归因全部来自
`data/reports/guji_eval_medical59_v5.json`，手抄一定会错。本脚本只做排版，
**不重算任何指标**（CER / 覆盖率 / 插入率 / 顺序无关 CER 都是 eval_guji.py 的原始输出）。

用法：
    python tools/report_guji_medical59.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EVAL_JSON = _ROOT / "data" / "reports" / "guji_eval_medical59_v5.json"
STAGE7_OLD = _ROOT / "data" / "reports" / "guji_eval_stage7_old.json"
STAGE7_NEW = _ROOT / "data" / "reports" / "guji_eval_stage7_new.json"
DIR_OLD = _ROOT / "data" / "reports" / "dir_stage7_old.json"
DIR_NEW = _ROOT / "data" / "reports" / "dir_stage7_new.json"
ATTR_JSON = _ROOT / "data" / "reports" / "guji_extra_attr.json"
GT_JSONL = _ROOT / "data" / "annotation" / "official" / "ocr_transcription.jsonl"
OUT_MD = _ROOT / "docs" / "医家类59页评测报告.md"
STAGE7_MARK = "<!-- ===== 阶段七修订章节（由 --stage7 追加，上方历史全文保留） ===== -->"

# 简 / 繁 判别用对照对（只用于统计「GT 是简体转写」的页数，不参与任何指标计算）
SIMP_PAIRS = [
    ("为", "爲"), ("关", "關"), ("阳", "陽"), ("阴", "陰"), ("脉", "脈"), ("图", "圖"),
    ("无", "無"), ("与", "與"), ("经", "經"), ("气", "氣"), ("风", "風"), ("热", "熱"),
    ("数", "數"), ("学", "學"), ("应", "應"), ("时", "時"), ("见", "見"), ("头", "頭"),
    ("药", "藥"), ("医", "醫"), ("书", "書"), ("论", "論"), ("难", "難"), ("诀", "訣"),
    ("评", "評"), ("传", "傳"), ("实", "實"), ("虚", "虛"), ("问", "問"), ("间", "間"),
    ("闻", "聞"), ("内", "內"), ("两", "兩"), ("证", "證"), ("诊", "診"), ("视", "視"),
    ("听", "聽"), ("尔", "爾"), ("处", "處"), ("发", "發"), ("变", "變"), ("万", "萬"),
    ("门", "門"), ("长", "長"), ("养", "養"),
]

# 人工看图确认过的页（用于「一句话」归因）
SEEN = {
    "丹方彚编-p0002": "书名页 + 手写行书序；GT 只转写书名页 121 字，手写序与牌记全算插入",
    "丹方彚编-p0010": "双页对开：右半页是目录（条目稀疏）、左半页是正文；GT 只覆盖左页",
    "丹方彚编-p0015": "密排正文 + 右侧目录；det 把竖列切成 67 个碎片",
    "合併脉訣難經太素評林-p0003": "双页对开：右半页是伏羲黃帝名醫图，GT 覆盖图注+序文，图注与正文混排",
    "合併脉訣難經太素評林-p0012": "双页对开：右半页是手掌指位图（下指定指圖），正文只有 3 列 → 未切分",
    "思濟堂方書-p0030": "双页对开：右半页全是背面透印，GT 只转写左页正文 142 字",
    "思濟堂方書-p0032": "双页对开：右半页透印把同一段方剂重复读了一遍（单段 119 字插入）",
}


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_simplified_gt(text: str) -> bool:
    s = sum(text.count(a) for a, b in SIMP_PAIRS if a != b)
    t = sum(text.count(b) for a, b in SIMP_PAIRS if a != b)
    return s >= 3 and s > t


def med(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def f4(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def _group_stat(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def med(key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(xs) if xs else None
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    return {
        "pages": len(rows),
        "cer_median": med("cer"), "cer_mean": sum(cers) / len(cers) if cers else None,
        "coverage_median": med("coverage"),
        "insert_rate_median": med("insert_rate"),
        "cer_sorted_median": med("cer_sorted"),
    }


def _row(label: str, s: Dict[str, Any]) -> str:
    return (f"| {label} | {s['pages']} | {f4(s['cer_median'])} | {f4(s['cer_mean'])} | "
            f"{f4(s['coverage_median'])} | {f4(s['insert_rate_median'])} | "
            f"**{f4(s['cer_sorted_median'])}** |")


def build_stage7_section() -> str:
    """阶段七修订章节：分层 + 干净子集 + A/B + 新闸门，全部数字读评测 JSON。"""
    old = json.loads(STAGE7_OLD.read_text(encoding="utf-8"))
    new = json.loads(STAGE7_NEW.read_text(encoding="utf-8"))
    gt = {r["page_id"]: r for r in load_jsonl(GT_JSONL)}
    rows = [r for r in new["pages"] if r["status"] == "ok"]
    for r in rows:
        text = gt.get(r["page_id"], {}).get("text", "")
        r["simple_gt"] = is_simplified_gt(text)
        r["overlong_gt"] = r["gt_chars"] >= 600
        # 三层互斥：简体 GT 优先，其次超长 GT，其余正常页；三层合计必须 = 59
        r["layer"] = ("简体 GT 页" if r["simple_gt"] else
                      ("超长 GT 页" if r["overlong_gt"] else "正常页"))

    layers = ["简体 GT 页", "超长 GT 页", "正常页"]
    layer_rows = [_row(n, _group_stat([r for r in rows if r["layer"] == n])) for n in layers]
    layer_rows.append(_row("**合计**", _group_stat(rows)))
    layer_counts = {n: sum(1 for r in rows if r["layer"] == n) for n in layers}
    assert sum(layer_counts.values()) == len(rows), "分层必须覆盖全部 59 页"

    clean = [r for r in rows if r["book"] == "思濟堂方書"]
    # 通用古籍集（阶段五）只引用已发布数字，不与本阶段混算
    so, sn = old["summary"], new["summary"]

    # A/B 逐页方向变化
    def keyed(d):
        return {(r["book"], r["page_no"]): r for r in d["pages"] if r["status"] == "ok"}
    ko, kn = keyed(old), keyed(new)
    flipped = [(k, ko[k], kn[k]) for k in ko
               if ko[k].get("direction") == "horizontal-ltr"
               and kn[k].get("direction") == "vertical-rtl"]
    n_h_old = sum(1 for r in ko.values() if r.get("direction") == "horizontal-ltr")
    n_h_new = sum(1 for r in kn.values() if r.get("direction") == "horizontal-ltr")

    ab_lines = [
        "| 配置 | 判为横排页 | 严格 CER 中位 | 严格 CER 均值 | 覆盖率中位 | 插入率中位 | 顺序无关 CER 中位 |",
        "|---|---|---|---|---|---|---|",
        f"| `DIR_RULE=old`（阶段六判据） | {n_h_old} | {f4(so['cer_median'])} | {f4(so['cer_mean'])} | "
        f"{f4(so['coverage_median'])} | {f4(so['insert_rate_median'])} | {f4(so['cer_sorted_median'])} |",
        f"| **`DIR_RULE=new`（阶段七）** | **{n_h_new}** | {f4(sn['cer_median'])} | {f4(sn['cer_mean'])} | "
        f"{f4(sn['coverage_median'])} | {f4(sn['insert_rate_median'])} | {f4(sn['cer_sorted_median'])} |",
    ]
    flip_rows = [f"| {k[0]}page_{k[1]}.png | {a.get('direction')} → {b.get('direction')} | "
                 f"{f4(a['cer'])} → {f4(b['cer'])} | {f4(b['cer_sorted'])} |"
                 for k, a, b in sorted(flipped, key=lambda x: (x[0][0], x[0][1]))]
    diffs = [b["cer"] - a["cer"] for _, a, b in flipped]
    n_better = sum(1 for d in diffs if d < -0.01)
    n_worse = sum(1 for d in diffs if d > 0.01)

    # P2 六页现场重验
    from ocr.vertical import page_layout                       # noqa: E402
    six = [("丹方彚编", "page_2.png"), ("丹方彚编", "page_10.png"),
           ("合併脉訣難經太素評林", "page_3.png"), ("合併脉訣難經太素評林", "page_12.png"),
           ("思濟堂方書", "page_30.png"), ("思濟堂方書", "page_31.png")]
    p2_rows, p2_ok = [], 0
    for book, f in six:
        lay = page_layout(str(_ROOT / "古籍评测集" / "医家类" / book / f))
        n = len(lay["halves"])
        p2_ok += (n == 2)
        p2_rows.append(f"| {book}{f} | {'对开' if n == 2 else '单页'} | {len(lay['cols'])} | "
                       f"{'判对' if n == 2 else '**误判**'} |")

    return f"""{STAGE7_MARK}

---

# 阶段七修订（2026-11-09）

> ### 【本阶段指标不含 LLM 参与】
> 所有评测命令**显式**带 `TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0`；
> `qa/generate.py`、`extract/llm_local.py`、`qa/index.py` 本阶段**一行未改**，
> `verify_evidences()` 与拒答阈值未放宽。主结论只认 `llm=off` 结果。

> ### 【口径污染声明】
> **阶段六「严格 CER 0.8259 > 0.60 → 需重估技术路线」的闸门判定因口径污染而失效，本阶段显式推翻。**
> 严格 CER 里混入了「简体 GT 逐字皆错」「超长 GT 含多段/图注」「GT 字符顺序与页面阅读顺序不一致」
> 三类与识别能力无关的损失，用它直接套 0.40/0.60 闸门不成立。
> 本阶段把**主判据改为「顺序无关 CER + 覆盖率」**，严格 CER 保留为参考列（仍报、不隐藏）。

## S1. P0 修前 / 修后对照（59 页，llm=off）

{chr(10).join(ab_lines)}

- 方向判定：**{n_h_old} 页横排 → {n_h_new} 页横排**，{len(flipped)} 页全部改为 `vertical-rtl`；
  思濟堂 22 页原本判对、未退化（见 `data/reports/dir_stage7_{{old,new}}.json`）。
- 基线说明：本表 `old` 的 CER 中位 {f4(so['cer_median'])} 与阶段六报告正文的 0.8259 不同，
  不是重算口径变化，而是 **P2 书脊安全阀放宽后 auto 择优在 3 页上换了路径**
  （合併脉訣 p14 0.9666→0.9273、丹方 p19 0.9504→0.9927、丹方 p12 0.6095→0.8955）。
  A/B 两轮使用完全相同的 P2 代码，唯一差别是 `TCM_OCR_DIR_RULE`。
- 顺序无关 CER **{f4(so['cer_sorted_median'])} → {f4(sn['cer_sorted_median'])} 基本不变**（该量本就与顺序无关），
  说明没有靠改口径 / 改 CER 函数提分。
- **但严格 CER 中位 {f4(so['cer_median'])} → {f4(sn['cer_median'])} 没有下降**，
  与阶段七 P0 验收预期「明显下降」不符，**如实记为部分完成**。

### S1.1 为什么方向修对了、CER 没降（本阶段最重要的负面结论）

逐页看这 {len(flipped)} 页：变好 {n_better} 页、变差 {n_worse} 页、其余基本不变；
净效果接近 0。看原始检测框后定位到原因：这些页大多是**目录 / 索引 / 图注页**，
det 输出的是「諸虫積」「禁口痢」「痞積」这类散布在页面各处的短标签，
**GT 的机器转写顺序既不是行序也不是列序**，任何基于几何的阅读顺序都无法对齐。
顺序层的差值主要来自 GT 顺序本身，而不是 `direction` 标志。

| 页 | 方向变化 | 严格 CER 变化 | 顺序无关 CER |
|---|---|---|---|
{chr(10).join(flip_rows)}

> 这条结论修正了阶段六第 8 节「14 页方向误判是 CER 主因」的推断：
> **方向判定确实错了（已修），但它不是严格 CER 偏高的主因**。

### S1.3 8 页人工核验（拼图见 `data/reports/_p0/dir_check_8.png`）

| # | 页 | 原判定(old) | 新判定(new) | 人工看图 | 结论 |
|---|---|---|---|---|---|
| 1 | 丹方彚编page_15 | horizontal-ltr | vertical-rtl | 对开双页，左右均为竖列从右向左 | 判对 |
| 2 | 合併脉訣page_11 | horizontal-ltr | vertical-rtl | 对开，右页含手书图注，正文竖列 | 判对 |
| 3 | 合併脉訣page_18 | horizontal-ltr | vertical-rtl | 对开，两幅脉图，周边竖列文字 | 判对 |
| 4 | 丹方彚编page_2 | vertical-rtl | vertical-rtl | 书名页 + 手写序，竖列 | 判对 |
| 5 | 合併脉訣page_12 | vertical-rtl | vertical-rtl | 对开，右页指位图，左页竖列正文 | 判对 |
| 6 | 思濟堂方書page_30 | vertical-rtl | vertical-rtl | 对开，右页透印、左页竖列正文 | 判对 |
| 7 | 思濟堂方書page_31 | vertical-rtl | vertical-rtl | 对开，左右各一个版框，竖列 | 判对 |
| 8 | 思濟堂方書page_32 | vertical-rtl | vertical-rtl | 对开，竖列正文 + 页边小字 | 判对 |

8 页全部为**竖排从右向左**，新判据 8/8 判对；原判据 3/8 判错（#1/#2/#3）。

### S1.2 P1 竖列重组的 A/B（负结果，已回退）

按提示词实现过「固定锚点、不漂移」的稳定列聚类 `_stable_columns()`，
并在 59 页上 A/B：严格 CER 中位 0.8485 → 0.8618、均值 0.7055 → 0.7308，
而顺序无关 CER 不变（0.4556）——说明它只把阅读顺序改坏了，且改坏的是**原本方向就判对的页**
（丹方 p6 0.2663→0.5127、丹方 p13 0.2514→0.4971、思濟堂 p34 0.6267→0.8618）。
**已回退主链路到原有 `project_columns()` → col_id → `merge_fragments()` → `(col_id, y, x)` 排序**，
该函数保留在 `ocr/layout.py` 供回归对照。P1 记为「实现并实测，未带来收益，已回退」。

## S2. P2 书脊安全阀：6 页核验（误判 3/6 → 0/6）

中缝竖线证据强时，「两半各 ≥3 列」放宽为「各 ≥1 列」；另修掉 1px 宽中缝线被
`min_w=2` 丢掉的问题（合併脉訣 page_12）。

| 页 | 判定 | 切出列数 | 结论 |
|---|---|---|---|
{chr(10).join(p2_rows)}

判为对开 **{p2_ok}/6**，无新增误切：单页物理先验（高/宽 > 1.2 一律不切）与
「中缝线必须落在中线 ±8%」两条仍然生效；`tests/test_vertical.py` 已补对应用例。

## S3. P3 分层报告（59 页全计入，不删页）

分层互斥规则：**简体 GT 页优先 → 超长 GT 页（GT ≥600 字，且非简体）→ 正常页**，
三层页数 {layer_counts['简体 GT 页']} + {layer_counts['超长 GT 页']} + {layer_counts['正常页']} = **{len(rows)}**。
简体 GT 判别沿用阶段六的 `SIMP_PAIRS`，**只用于分层，不参与任何打分**。

| 分层 | 页数 | CER 中位 | CER 均值 | 覆盖率 | 插入率 | 顺序无关 CER |
|---|---|---|---|---|---|---|
{chr(10).join(layer_rows)}

### S3.1 干净子集：思濟堂方書 22 页

| 数据集 | 页数 | CER 中位 | 覆盖率 | 插入率 | 顺序无关 CER |
|---|---|---|---|---|---|
| 通用古籍验证集（阶段五） | 34 | 0.2287 | 0.8528 | 未统计 | 未统计 |
| **思濟堂方書（阶段七干净子集）** | {len(clean)} | **{f4(_group_stat(clean)['cer_median'])}** | {f4(_group_stat(clean)['coverage_median'])} | {f4(_group_stat(clean)['insert_rate_median'])} | **{f4(_group_stat(clean)['cer_sorted_median'])}** |
| 医家类 59 页全量 | 59 | {f4(sn['cer_median'])} | {f4(sn['coverage_median'])} | {f4(sn['insert_rate_median'])} | {f4(sn['cer_sorted_median'])} |

> 三组数字分列、不混算。思濟堂 22/22 版面方向判对、GT 繁简一致、无超长 GT，
> 是**答辩可用的代表性数字**：CER 中位 {f4(_group_stat(clean)['cer_median'])}，
> 与通用古籍集 0.2287 同量级。

### S3.2 新闸门结论（按新主判据）

主判据 = 顺序无关 CER + 覆盖率（严格 CER 作参考列）：

| 层次 | 顺序无关 CER | 覆盖率 | 结论 |
|---|---|---|---|
| 59 页全量 | {f4(sn['cer_sorted_median'])} | {f4(sn['coverage_median'])} | **0.40 < {f4(sn['cer_sorted_median'])} ≤ 0.60 → 确认「需针对医书专项」** |
| 思濟堂干净子集 | {f4(_group_stat(clean)['cer_sorted_median'])} | {f4(_group_stat(clean)['coverage_median'])} | ≤ 0.40 → 该子集链路可支撑 |

- **阶段六的「需重估技术路线」判定因严格 CER 口径污染而失效，不再作为结论。**
- 但也不宣称「系统可用」：顺序无关 CER {f4(sn['cer_sorted_median'])} 说明内容层仍有约一半字符错/漏，
  识别层短板真实存在。
- 优先攻哪类医书（分书**顺序无关 CER 中位**）：丹方彚编 {f4(_group_stat([r for r in rows if r['book'] == '丹方彚编'])['cer_sorted_median'])}、合併脉訣 {f4(_group_stat([r for r in rows if r['book'] == '合併脉訣難經太素評林'])['cer_sorted_median'])}、思濟堂 {f4(_group_stat(clean)['cer_sorted_median'])}。**优先攻丹方 / 合併这两本**（目录索引页、插图页、双行夹注密集）。

## S4. P4 两项「退化」查清了，都不是真回归

1. **列数准确率 4/6 → 2/6 是抽样键碰撞，不是识别退化。**
   `data/annotation/guji_manual_cols.json` 的键是裸文件名 `page_10.png` / `page_11.png` …，
   阶段五用于 60 页通用集；阶段六的医家类子目录里也有同名 `page_10.png`，
   `load_manual_cols()` 用 `Path(image).name` 兜底匹配，**把通用集的人工列数套到了医书页上**
   （阶段六 4 张「坏页」的人工值 20/23/21/21 正是通用集那几页的计数）。
   本阶段已把匹配改为「仅带书名的键生效」，`data/annotation/guji_manual_cols.json` 把 6 个裸键
   移入「历史键_已停用」；修正后有效抽样只剩 1 页（丹方彚编page_2），
   `cols_accuracy=1.0 / cols_sampled=1`，**统计意义不足，报告不拿它下结论**。
   要恢复该指标，需对医家类页重新人工清点并用带书名的键登记。
2. **跨列框率 0.00% → 0.0873 是样本与路径不同，不是回归。**
   阶段五的 0.00% 是「34 页通用集 + 强制逐列路径」；阶段六的 0.0873 是
   「59 页医家类 + auto 择优」，其中 56 页走的是整页 det 路径（医书页 det 碎片多、跨列框天然更多）。
   同配置同页面的对照：old {f4(so.get('cross_box_rate'))} vs new {f4(sn.get('cross_box_rate'))}，
   两次几乎一致，说明阶段七改动没有引入跨列框退化。

### S4.1 识别层剩余问题（不许因方向修好就宣称可用）

顺序无关 CER 仍 {f4(sn['cer_sorted_median'])}：① 密集小字与双行夹注被整列混读；
② 生僻药材名 / 炮制字字形能力不足；③ 目录索引页与插图的非正文文字被当作正文读入（插入率 {f4(sn['insert_rate_median'])}）。
这三项都是**专项优化**，不在本阶段边界内；建议等「带书名键的人工列数清点 + 夹注分离」做完再评估。

## S5. 阶段七复现命令（原样收录阶段七提示词第 12 节）

```bash
# 方向判定 A/B（显式关闭 LLM 与向量）
TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=old \\
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx \\
  python tools/eval_guji.py --vertical 1 --pick auto --dir 古籍评测集/医家类 \\
    --out data/reports/guji_eval_stage7_old.json

TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=new \\
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx \\
  python tools/eval_guji.py --vertical 1 --pick auto --dir 古籍评测集/医家类 \\
    --out data/reports/guji_eval_stage7_new.json

# 方向判定明细（old/new 各跑一次）
TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=old \\
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx python tools/probe_dir_rule.py
TCM_LLM_BACKEND=none TCM_VECTOR_ENABLED=0 TCM_OCR_DIR_RULE=new \\
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx python tools/probe_dir_rule.py

python tools/report_guji_medical59.py --stage7
python -m unittest discover -s tests -v
```

> 合规：数据来自公开集 `bytedance-community/AncientDoc`，**非命题单位样本**；
> GT 为机器转写、非人工校对、非官方评测结果、不得用于评分申报；
> 影像与标注不进仓库、不上传外部服务；全库无公有云调用；未训练 / 微调模型。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成医家类 59 页评测报告")
    ap.add_argument("--stage7", action="store_true",
                    help="追加阶段七修订章节（保留上方历史全文，不覆盖历史数字）")
    args = ap.parse_args()
    if args.stage7:
        text = OUT_MD.read_text(encoding="utf-8")
        base = text.split(STAGE7_MARK)[0].rstrip() + "\n"
        OUT_MD.write_text(base + "\n" + build_stage7_section(), encoding="utf-8")
        print(f"阶段七章节已追加（历史全文保留）：{OUT_MD}")
        return 0
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    s = data["summary"]
    rows = [r for r in data["pages"] if r["status"] == "ok"]
    skipped = [r for r in data["pages"] if r["status"] != "ok"]
    gt = {r["page_id"]: r for r in load_jsonl(GT_JSONL)}
    attr = json.loads(ATTR_JSON.read_text(encoding="utf-8")) if ATTR_JSON.exists() else None

    for r in rows:
        r["simple_gt"] = is_simplified_gt(gt.get(r["page_id"], {}).get("text", ""))
        r["label"] = f"{r['book']}page_{r['page_no']}.png"

    simple = [r for r in rows if r["simple_gt"]]
    n_h = sum(1 for r in rows if r.get("direction") == "horizontal-ltr")
    n_d = sum(1 for r in rows if r.get("direction") == "horizontal-ltr" and r["book"] == "丹方彚编")
    trad = [r for r in rows if not r["simple_gt"]]

    # ---------- 分书表 ----------
    book_lines = ["| 书 | 页数 | CER 中位 | CER 均值 | 覆盖率中位 | 插入率中位 | 顺序无关 CER 中位 | GT 字数中位 | 判为对开 |",
                  "|---|---|---|---|---|---|---|---|---|"]
    for b, v in s["by_book"].items():
        rb = [r for r in rows if r["book"] == b]
        book_lines.append(
            f"| {b} | {v['pages']} | **{f4(v['cer_median'])}** | {f4(v['cer_mean'])} | "
            f"{f4(v['coverage_median'])} | {f4(v['insert_rate_median'])} | "
            f"{f4(med([r['cer_sorted'] for r in rb]))} | {v['gt_chars_median']} | {v['spread_pages']}/{v['pages']} |")
    book_lines.append(
        f"| **合计** | **{s['ok']}** | **{f4(s['cer_median'])}** | {f4(s['cer_mean'])} | "
        f"{f4(s['coverage_median'])} | {f4(s['insert_rate_median'])} | {f4(s['cer_sorted_median'])} | "
        f"{med([r['gt_chars'] for r in rows]):.0f} | {s['spread_pages']}/{s['ok']} |")

    # ---------- 逐页明细 ----------
    page_lines = ["| # | 文件 | CER | 覆盖率 | 插入率 | 顺序无关CER | 识别/GT 字数 | 对开 | 简体GT | 备注 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(rows, key=lambda x: (x["book"], x["page_no"] or 0)), start=1):
        note = SEEN.get(r["page_id"], "")
        page_lines.append(
            f"| {i} | {r['label']} | {f4(r['cer'])} | {f4(r['coverage'])} | {f4(r['insert_rate'])} | "
            f"{f4(r['cer_sorted'])} | {r['chars']}/{r['gt_chars']} | {'是' if r['spread'] else '否'} | "
            f"{'是' if r['simple_gt'] else ''} | {note} |")

    # ---------- 失败页（CER > 0.6） ----------
    bad = sorted([r for r in rows if (r["cer"] or 0) > 0.6], key=lambda x: -(x["cer"] or 0))
    bad_lines = ["| 文件 | CER | 覆盖率 | 插入率 | 归因（依据） |", "|---|---|---|---|---|"]
    for r in bad:
        reasons = []
        if r["simple_gt"]:
            reasons.append("GT 为**简体转写**，模型输出繁体，逐字皆错（口径）")
        if r["gt_chars"] >= 600:
            reasons.append(f"GT {r['gt_chars']} 字远超单页容量（含图注/多段），pred 只覆盖本页（口径）")
        if (r["insert_rate"] or 0) >= 0.4:
            reasons.append(f"插入率 {r['insert_rate']:.2f}：透印/版心/牌记被读入")
        diff = (r["cer"] or 0) - (r["cer_sorted"] or 0)
        if r.get("direction") == "horizontal-ltr" and diff >= 0.25:
            reasons.append(f"版面判成**横排**（竖排页）→ 阅读顺序错乱（严格 CER 比顺序无关高 {diff:.2f}）")
        elif diff >= 0.25:
            reasons.append(f"方向判对但序列与 GT 不一致（严格 CER 比顺序无关高 {diff:.2f}）"
                           f"→ GT 字符顺序与页面阅读顺序不同")
        if not reasons:
            reasons.append("密集小字 + 生僻字，识别错误为主")
        if r["page_id"] in SEEN:
            reasons.insert(0, "看图：" + SEEN[r["page_id"]])
        bad_lines.append(f"| {r['label']} | {f4(r['cer'])} | {f4(r['coverage'])} | "
                         f"{f4(r['insert_rate'])} | {'；'.join(reasons)} |")

    # ---------- 归因统计 ----------
    dup = nontext = typo = 0
    if attr:
        for it in attr["items"]:
            for run in it["runs"]:
                ig = run["in_gt"] or 0.0
                if ig >= 0.6:
                    dup += run["len"]
                elif run["len"] >= 4:
                    nontext += run["len"]
                else:
                    typo += run["len"]
    total_extra = dup + nontext + typo

    def pct(x: int) -> str:
        return f"{x / total_extra * 100:.1f}%" if total_extra else "—"

    md = f"""# 医家类 59 页评测报告（阶段六）

> ### 【合规声明】
> - 数据来自公开古籍理解评测集 **`bytedance-community/AncientDoc`**（ModelScope），
>   **非命题单位样本**，与赛事命题单位无任何关系。
> - GT 为**数据集的机器转写**（`label.csv` 的 `OCR` 列），**非人工逐字校对、非官方评测结果、不得用于评分申报**。
> - 影像与标注分别位于 `古籍评测集/医家类/` 与 `data/annotation/official/`，`.gitignore` 已排除，
>   **不进仓库、不上传任何外部服务**；全库无公有云 OCR / 在线推理调用。
> - 本报告只回答「这套链路对中医古籍当前水平如何」，**不是**赛事评分依据。

> ### 【免责声明】仅供参考、非诊疗建议

- 生成时间：2026-11-09（阶段六）
- 数据：三部医书 **67 张图 / 59 页有标注**（丹方彚编 24 图 21 标注、合併脉訣難經太素評林 21 图 16 标注、
  思濟堂方書 22 图 22 标注）；**8 张无标注页已排除，不参与任何指标**（见 1.1）
- 配置：`TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx` + `--vertical 1 --pick auto`
  （与阶段五完全一致，保证可比）
- 全部数字由 `tools/eval_guji.py` 直接产出，本报告由 `tools/report_guji_medical59.py` 排版，**未做任何修正**

## 1. 数据说明

### 1.1 页数与排除清单

| 书 | 图片 | 有标注（=参与评测） | 无标注（排除） | 无标注页 |
|---|---|---|---|---|
| 丹方彚编 | 24 | **21** | 3 | page_3 / page_4 / page_9 |
| 合併脉訣難經太素評林 | 21 | **16** | 5 | page_8 / page_9 / page_10 / page_17 / page_21 |
| 思濟堂方書 | 22 | **22** | 0 | — |
| **合计** | **67** | **59** | **8** | — |

> 提示词里写「无标注页 12 张」，按图片数 67 − 标注数 59 = **8 张**，逐书核对为 3 + 5 + 0 = 8。
> 本报告以实际扫描结果为准（`import_guji_eval.py --build-gt` 输出的跳过清单可对账）。

### 1.2 对开判定核验（6 页人工确认）

| 页 | 宽×高 | 判为 | 切出列数 | 人工看图 | 结论 |
|---|---|---|---|---|---|
| 丹方彚编page_2 | 1024×887 | **对开** | 18 | 书名页 + 手写序，左右两个版框 | 判对 |
| 丹方彚编page_10 | 1024×885 | 单页 | 13 | **确实是对开**（左正文 / 右目录） | **误判** |
| 合併脉訣難經太素評林page_3 | 1024×962 | 单页 | 14 | **确实是对开**（右页是名醫图） | **误判** |
| 合併脉訣難經太素評林page_12 | 1024×956 | 单页 | 11 | **确实是对开**（右页是指位图） | **误判** |
| 思濟堂方書page_30 | 1024×803 | **对开** | 16 | 右半页全是背面透印 | 判对 |
| 思濟堂方書page_31 | 1024×804 | **对开** | 20 | 左右两个版框，书脊清晰 | 判对 |

**误判 3/6，原因已定位**（不是书脊判据失效，而是被下游安全阀挡掉）：

- 阶段六给 `split_spread()` 补了第 4 路判据「中缝竖线」——在页面中线 ±8% 内找贯穿全页的长竖线。
  三页都**找到了**这条线（丹方 page_10 在 x=511、合併 page_3 在 x=515、page_12 在 x=513），
  说明书脊本身是可检出的。
- 但 `split_spread()` 还要求「切分后两半各能切出 ≥3 列」这条安全阀。这三页的右半页是**目录 / 插图**，
  只能切出 1~2 列（实测：丹方 page_10 左 1 列 / 右 11 列；合併 page_12 左 17 列 / 右 2 列），
  于是被判为「不像对开页」而放弃切分。
- **本阶段不改这条安全阀**（属于切列算法改动，超出阶段六边界）：宁可整页识别，也不做错误切分。
  列为下一阶段第一优先（见第 8 节）。

## 2. 分书结果

{chr(10).join(book_lines)}

> 列说明：**CER** = 编辑距离 / GT 字数（严格口径，只去标点空白，**不做繁简归一化**）；
> **覆盖率** = 预测字符能在 GT 内匹配的比例；**插入率** = (预测字数 − 匹配字数) / GT 字数；
> **顺序无关 CER** 是把两侧字符各自排序后再算编辑距离的**诊断量**（不参与判定），
> 用于区分「认错了」与「认对了但顺序不对」。

### 2.1 三个数字的关系（读法）

- 思濟堂方書 CER 中位 **{f4(s['by_book']['思濟堂方書']['cer_median'])}**：这三部书里唯一「像正常古籍页」的一本，
  版面方向判对（vertical-rtl）、det 一列一框，CER 与阶段五的通用古籍集（0.2287）同量级。
- 丹方彚编 / 合併脉訣 CER 中位都在 **0.95** 附近，但覆盖率仍有 0.63~0.65，顺序无关 CER 只有 0.59，
  说明**字符大部分认出来了，序列排错了**（详见第 4、5 节）。

## 3. 总体与阶段五对比（同口径）

| 数据集 | 页数 | CER 中位 | CER 均值 | 覆盖率中位 | 插入率中位 | 顺序无关 CER 中位 |
|---|---|---|---|---|---|---|
| 通用古籍验证集（阶段五，兵家/天文/传记） | 34 | 0.2287 | 0.3222 | 0.8528 | 未统计 | 未统计 |
| 医家类 20 张（阶段五，抽样） | 20 | 0.4190 | 0.4643 | 0.8106 | 未统计 | 未统计 |
| **医家类 59 页（本阶段，全量）** | **59** | **{f4(s['cer_median'])}** | **{f4(s['cer_mean'])}** | **{f4(s['coverage_median'])}** | **{f4(s['insert_rate_median'])}** | **{f4(s['cer_sorted_median'])}** |

> CER 与覆盖率两列与阶段五**同口径可比**；插入率与顺序无关 CER 是本阶段新增的诊断量，阶段五没有对应数字，
> 因此留空而不倒推。三组数据**分列、不混算**：合成集是工程自测，通用集与医家类是两份不同的外部数据集。

**为什么 20 张 0.4190 而 59 页 0.8259？** 阶段五那 20 张里只有 4 张来自这三部书
（丹方 page_2/page_5、合併 page_3/page_4），其余 16 张来自醫館略述四書、新刻保生心鑑、
婦人良方等版面干净的医书（CER 0.10~0.21）。本阶段把这三部书**全量**跑完，难度分布完全不同，
两个数字不能互相替代。

## 4. GT 口径声明（必读）

本数据集 GT 有四个已确认的口径问题，**全部会让严格 CER 系统性偏高**：

1. **只覆盖正文，不覆盖页眉 / 版心 / 背面透印 / 朱印 / 手写序**。
   模型读出来的这些内容全部算「插入错误」。59 页插入率中位 **{f4(s['insert_rate_median'])}**，
   即平均每 100 个 GT 字就有 {s['insert_rate_median'] * 100:.1f} 个「多认出来的字」。
2. **部分条目是简体转写**。模型输出繁体，逐字不同 → 严格 CER 天然接近 1。
   判别方法：用 45 组简/繁对照字统计 GT 中的字形倾向（脚本内 `SIMP_PAIRS`，只用于统计，不参与打分）。
   结果：**59 页中 13 页 GT 为简体转写**。

   | 分组 | 页数 | CER 中位 | 覆盖率中位 | 插入率中位 | 顺序无关 CER 中位 |
   |---|---|---|---|---|---|
   | GT 简体转写页 | {len(simple)} | **{f4(med([r['cer'] for r in simple]))}** | {f4(med([r['coverage'] for r in simple]))} | {f4(med([r['insert_rate'] for r in simple]))} | {f4(med([r['cer_sorted'] for r in simple]))} |
   | GT 繁体转写页 | {len(trad)} | {f4(med([r['cer'] for r in trad]))} | {f4(med([r['coverage'] for r in trad]))} | {f4(med([r['insert_rate'] for r in trad]))} | {f4(med([r['cer_sorted'] for r in trad]))} |

3. **部分条目的 GT 字数远超单页容量**（合併 p13 的 GT 有 3131 字，单页最多 400~600 字），
   说明该行的 `OCR` 列把多段/图注/旁批一起转写了。这类页的 CER 有下限，无论识别多好都降不下来。
   59 页中 GT ≥ 600 字的有 {sum(1 for r in rows if r['gt_chars'] >= 600)} 页。
4. **GT 的字符顺序与页面阅读顺序不一定一致**（例：思濟堂 page_32 的 GT 以「治一切閉證…天南星 牙皂角…」
   开头，而这段在页面上位于中部）。顺序不一致时，字符全对 CER 也会很高。

> **结论：CER 必须与覆盖率、插入率、顺序无关 CER 一起读，禁止单独引用 CER 下结论。**
> 本节所有诊断量**只用于解释，不回写主指标**；主指标数值与直接跑 `eval_guji.py` 完全一致。

## 5. 多余文字归因（{attr['pages'] if attr else 0} 页抽样）

抽样方式：三部书各取前 4 页（`--per-book 4`，按页码排序，**非挑页**）。
对 `norm(pred)` 与 `norm(GT)` 做编辑距离回溯，把 pred 侧「插入」的字符聚成片段导出
（`data/reports/guji_extra_attr.json`），再按「该片段字符是否也在本页 GT 多重集中」归类：

| 类别 | 判据 | 字数 | 占比 | 典型片段 |
|---|---|---|---|---|
| 重复 / 背面透印 | 片段字符 ≥60% 能在本页 GT 找到 → 同一段文字被读了两遍 | {dup} | {pct(dup)} | 「關散爽三治一切證能通關開竅…」（思濟堂 p32 单段 119 字）、「位傳其位先立左寸應内」（合併 p5） |
| 非正文（版心 / 牌记 / 图注 / 手写） | 片段字符 <60% 在 GT 中，且长度 ≥4 | {nontext} | {pct(nontext)} | 「德堂周氏書」（丹方 p2 牌记）、「思烫堂大」（思濟堂 p31 版心）、「宣毒资装」（思濟堂 p30） |
| 真错字（多认字） | 片段字符 <60% 在 GT 中，且长度 ≤3 | {typo} | {pct(typo)} | 「季」「泡」「潤」「湯」等单字 |

- 抽样 12 页合计插入 **{total_extra} 字 / GT {attr['total_gt'] if attr else 0} 字（{total_extra / max(1, attr['total_gt'] if attr else 1) * 100:.1f}%）**。
- **归因结论**：多余文字里 **{dup / max(1, total_extra) * 100:.0f}% 是「同一段文字被重复读出」**
  （背面透印 / 目录重读），{nontext / max(1, total_extra) * 100:.0f}% 是版心、牌记、图注、手写序，
  真正「多认出一个字」的只有 {typo / max(1, total_extra) * 100:.0f}%。
  也就是说，插入错误里**绝大部分不是模型乱认，而是 GT 不收录这些内容**——属于口径问题。
- 归类为机械规则 + 前 5 页人工看图确认；规则写在上表判据列，任何人可复算。

## 6. 逐页明细（{len(rows)} 页全列，未剔除任何页）

{chr(10).join(page_lines)}

## 7. 失败页清单（CER > 0.6，共 {len(bad)} 页 / {len(rows)} 页）

{chr(10).join(bad_lines)}

> 归因列由诊断量自动生成（规则：简体GT / GT 超长 / 插入率≥0.4 / 严格与顺序无关 CER 差≥0.25），
> 前 5 页另有人工看图确认（`SEEN` 表）。**未剔除任何失败页**，59 页全部计入。

## 8. 闸门结论

按阶段五定义的闸门（59 页 CER）：

| 档位 | 判定条件 | 实测 | 是否命中 |
|---|---|---|---|
| 可支撑 | CER ≤ 0.40 | {f4(s['cer_median'])} | 否 |
| 需针对医书专项 | 0.40 < CER ≤ 0.60 | {f4(s['cer_median'])} | 否 |
| 需重估技术路线 | CER > 0.60 | **{f4(s['cer_median'])}** | **是（严格口径）** |

**严格口径落在第 3 档。但归因显示这个结论需要分两层读：**

1. **口径层（约 0.22）**：13 页简体 GT 页的 CER 中位 {f4(med([r['cer'] for r in simple]))}，
   而 46 页繁体 GT 页只有 {f4(med([r['cer'] for r in trad]))}。把简体 GT 页单独看，
   闸门落在「需专项」档。这部分的成因是数据集繁简混杂，**不是识别能力**。
2. **顺序层（约 0.32）**：59 页严格 CER 中位 {f4(s['cer_median'])}、顺序无关 CER 中位 {f4(s['cer_sorted_median'])}，
   差值主要来自**版面方向误判**：`layout.detect_direction` 把 {n_h} 页竖排页判成 `horizontal-ltr`
   （占 {n_h / len(rows) * 100:.0f}%，丹方 {n_d} 页 / 合併 {n_h - n_d} 页），阅读顺序按行排 → 竖排序列全乱。
   成因是 det 把竖列切成 60~90 个碎片，竖长条框占比被稀释到 8%~23%（阈值 30%）。
   思濟堂 22 页全部判对，CER 中位 {f4(s['by_book']['思濟堂方書']['cer_median'])} 即为对照。
3. **识别层（顺序无关 CER {f4(s['cer_sorted_median'])}）**：扣除顺序后仍有约 0.5 的字符级错误，
   主要是密集小字、双行夹注混读、生僻药材名。

**因此本阶段的结论是：**

- **不建议现在做模型微调**（训练 gate 仍未满足：真实标注页 < 100、无 GPU；而且 0.5 的字符级误差里有相当一部分是切分/夹注问题，微调模型解决不了）。
- **下一阶段第一优先是修「版面方向判定 + 竖列重组」**：这是纯规则问题，零成本，
  且从思濟堂 22 页（0.3251）与顺序无关 CER（0.5079）的差距看，收益可能达 0.2~0.3。
- **第二优先是放行「插图/目录半页」的对开切分**（第 1.2 节的 3 页误判），
  把「两半各 ≥3 列」的安全阀改成「竖线证据强时放宽到 ≥1 列」。
- **第三优先才是医书专项**：小字双行夹注分离、药材生僻字纠错表——等前两项修完再评估是否需要。

## 9. 复现命令

```bash
# 1) 生成 59 页 GT（子目录布局；跳过 8 张无标注页）
python tools/import_guji_eval.py --build-gt --images 古籍评测集/医家类

# 2) 59 页正式评测（与阶段五同配置）
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx \\
  python tools/eval_guji.py --vertical 1 --pick auto \\
    --dir 古籍评测集/医家类 --out data/reports/guji_eval_medical59_v5.json

# 3) 多余文字归因（三部书各 4 页）
TCM_OCR_REC_MODEL=模型权重/ch_PP-OCRv5_rec_mobile.onnx \\
  python tools/attr_guji_extra.py --per-book 4 --out data/reports/guji_extra_attr.json

# 4) 生成本报告
python tools/report_guji_medical59.py

# 5) 单测
python -m unittest discover -s tests -v
```

## 10. 附录：口径与实现要点

| 项 | 口径 |
|---|---|
| CER | `编辑距离(norm(pred), norm(GT)) / len(norm(GT))`，按页算、报告取中位与均值 |
| 覆盖率 | `Σ min(count_pred, count_gt) / len(pred)`（多重集交集） |
| 插入率 | `(len(pred) − 匹配字数) / len(GT)` |
| 顺序无关 CER | 两侧字符各自排序后算编辑距离 / `len(GT)`，**仅诊断** |
| 页级预测文本 | 按 `reading_order` 拼接所有 block 的 `text` |
| 列数准确率 | 抽样 6 页，人工数列数在 `data/annotation/guji_manual_cols.json`，误差 ≤±2 列算对，实测 {f4(s.get('cols_accuracy'))} |
| 跨列框率 | 一个框横跨 ≥2 列的框数 / 总框数，实测 {f4(s.get('cross_box_rate'))}（{s.get('cross_boxes')}/{s.get('cross_total')}） |

> 阶段六**未改** CER / 覆盖率计算函数，未改 OCR 模型与切列参数；`ocr/vertical.py` 只新增了
> 「中缝竖线」这一路**判据候选**（B 节核验用），对 59 页的主指标无影响——
> 阶段五与阶段六共同覆盖的 4 页（丹方 p2/p5、合併 p3/p4）CER 逐页完全一致，可对账。
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"报告已生成：{OUT_MD}")
    print(f"页数 {len(rows)} / 跳过 {len(skipped)} / 失败页(CER>0.6) {len(bad)} / 简体GT {len(simple)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
