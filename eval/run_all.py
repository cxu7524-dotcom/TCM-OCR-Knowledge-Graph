# -*- coding: utf-8 -*-
"""
eval/run_all.py —— 一键重跑正式评测（阶段二）

跑法（项目根目录）：
    python eval/run_all.py                  # 缺预测则自动生成，然后算分
    python eval/run_all.py --refresh       # 强制重跑 OCR / 抽取 / 问答，再算分
    python eval/run_all.py --no-predict    # 只读已有预测文件
    python eval/run_all.py --all-pages     # 全量页计分（默认只用测试页）

产出（幂等覆盖）：
    data/reports/正式评测报告.md      正式报告（五项实测 + 阈值判定 + 溯源）
    data/reports/评测明细.json        机器可读明细（每个数字的分子分母与输入）
    data/reports/评测报告骨架.md      阶段一自检章节（示例数据、非指标，保留）

严禁：编造数字、把 data/demo 示例数据或合成集结果冒充命题单位样本 / 官方评测结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                                    # noqa: E402
from app.demo import load_demo_ocr                                # noqa: E402
from eval.audit import audit_qa_logs, self_check                  # noqa: E402
from eval.dataset import Dataset, load_jsonl                     # noqa: E402
from eval.metrics import (                                       # noqa: E402
    METRIC_SPECS, compute_all, report_metrics,
)

REPORT_DIR = CONFIG.paths.reports_dir
FORMAL_REPORT = REPORT_DIR / "正式评测报告.md"
DETAIL_JSON = REPORT_DIR / "评测明细.json"
SKELETON_REPORT = REPORT_DIR / "评测报告骨架.md"

PREDICTION_FILES = {
    "ocr": "ocr_predictions.jsonl",
    "entities": "extraction_predictions.jsonl",
    "tables": "table_predictions.jsonl",
    "qa_log": "qa_log.jsonl",
}
SELF_CHECK_QUERIES = [
    "甘草的性味是什么？", "甘草有什么禁忌？",
    "黄芪在本书中用量几何？", "本品对晚期肝癌的治愈率是多少？",
]

SYNTHETIC_NOTICE = "合成示意数据，用于工程自测，非命题单位样本、非官方评测结果"


def _load_predictions() -> dict:
    return {k: load_jsonl(REPORT_DIR / name) for k, name in PREDICTION_FILES.items()}


def _ensure_predictions(refresh: bool, no_predict: bool) -> dict:
    """按需触发 OCR / 抽取 / 问答跑批；失败时返回可读原因，不中断报告生成。"""
    meta: dict = {"predictions_generated": False, "reason": ""}
    if no_predict:
        meta["reason"] = "--no-predict：只读已有预测文件"
        return meta
    try:
        from eval.predict import ensure_predictions
        meta = ensure_predictions(force=refresh)
    except Exception as exc:
        meta = {"predictions_generated": False,
                "reason": f"预测生成不可用：{type(exc).__name__}: {exc}"}
    return meta


def _blocks_by_page(pred_ocr: list) -> dict:
    """从预测 JSONL 还原 Block 对象，供 quote 回原文校验。"""
    from app.schemas import Block
    out: dict = {}
    for r in pred_ocr:
        if r.get("is_demo") is True:
            continue
        try:
            b = Block.from_dict(r)
            b.quote = r.get("quote", b.text)
            out.setdefault(b.page_id, []).append(b)
        except Exception:
            continue
    return out


def _fmt(value) -> str:
    if value is None:
        return "待测"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_rows(results: dict) -> str:
    lines = ["| 指标 | 实测值 | 参考阈值 | 达标情况 | 状态 | 输入文件 |",
             "|---|---|---|---|---|---|"]
    for spec in METRIC_SPECS:
        key = spec["key"]
        r = results.get(key, {})
        th = spec["threshold"]
        lines.append(
            f"| {spec['name']} | {_fmt(r.get('value'))} | "
            f"{'无' if th is None else f'≥ {th:.2f}'} | {r.get('judge', '待测')} | "
            f"{r.get('status', '待测')} | {'；'.join(spec['needs'])} |")
    return "\n".join(lines)


def _detail_lines(results: dict) -> str:
    lines = []
    for spec in METRIC_SPECS:
        r = results.get(spec["key"], {})
        lines.append(f"- **{spec['name']}**：{r.get('detail') or r.get('reason') or '无有效样本'}")
        if r.get("excluded"):
            lines.append(f"  - 排除的示例 / 无效记录：{r['excluded']}")
    return "\n".join(lines)


def _self_check_section() -> str:
    """阶段一自检：示例数据，明确非指标。"""
    try:
        blocks = load_demo_ocr("DEMO-001").blocks
        result = self_check(blocks, SELF_CHECK_QUERIES)
    except Exception as exc:
        return f"链路自检未执行：{type(exc).__name__}（不影响正式指标）"
    lines = [
        "> 本节全部来自 `data/demo/` **示例数据 / 模型未接入**，仅用于验证四步链路与拒答逻辑，",
        "> **不是评测结果、不计入上方任何指标**。",
        "",
        "| 问题 | 期望 | 实际 | 证据数 | quote 全部回原文命中 |",
        "|---|---|---|---|---|",
    ]
    for q, c in zip(SELF_CHECK_QUERIES, result["cases"]):
        expect = "拒答" if ("黄芪" in q or "治愈率" in q) else "作答"
        actual = "拒答" if c["refused"] else "作答"
        ok = "—" if not c["evidence_count"] else ("是" if c["quote_hit_all"] else "否")
        lines.append(f"| {q} | {expect} | {actual}（{c['refuse_reason'] or '—'}） | "
                     f"{c['evidence_count']} | {ok} |")
    return "\n".join(lines)


def _skeleton_report(metrics: dict) -> str:
    """保留阶段一骨架报告（口径 + 自检），明确标注示例数据非指标。"""
    return f"""# 评测报告骨架（阶段一遗留 · 仅口径与自检）

> **本文件不是正式评测结果**。正式结果见 `正式评测报告.md`。
> 本节所有内容基于 `data/demo/` 示例数据，**非指标、不计分**。
> 仅供参考、非诊疗建议。

- 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 阶段：{metrics["phase"]}
- 政策：{metrics["policy"]}

## 一、五项指标口径（正式值见正式报告）

| 指标 | 口径 | 缺失条件 |
|---|---|---|
""" + "\n".join(
        f"| {s['name']} | {s['formula']} | {'；'.join(s['needs'])} |"
        for s in METRIC_SPECS) + f"""

## 二、链路自检明细（示例数据，非指标）

{_self_check_section()}

## 三、数据合规

{SYNTHETIC_NOTICE}。命题单位脱敏样本仅限赛事研发使用，不得外传。
真实扫描页与人工标注不进仓库、不上传任何外部服务。
"""


def build_formal_report(ds: Dataset, results: dict, predict_meta: dict,
                        pred_counts: dict, test_only: bool, audit: dict) -> str:
    check = ds.check()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env = predict_meta.get("environment", {})
    variant = results.get("variant_hit", {})
    trad = results.get("cer", {}).get("traditional_subset", {})
    varsub = results.get("cer", {}).get("variant_subset", {})

    env_lines = "\n".join(f"- {k}：{v}" for k, v in env.items()) or "- 未登记（预测未生成）"
    return f"""# 赛题二 · 正式评测报告（阶段二首轮实测）

> ### 【免责声明】仅供参考、非诊疗建议
> ### 【数据声明】{SYNTHETIC_NOTICE}
> 命题单位脱敏样本：**未到位**；本报告所有数值均来自本地合成示意页，
> **不是官方评测结果、不得用于评分申报**。合成集与命题样本结果分列、不混算。

- 生成时间：{now}
- GT 版本哈希：`{ds.version}`（六类 GT 文件内容 sha256 前 16 位，见 `eval/dataset.py`）
- GT 目录：`{ds.gt_dir}`
- 数据来源：`SYN-2026-001` 合成示意集（`tools/make_synthetic_pages.py` 固定种子生成）
- GT 性质：**机器生成 GT，人工抽检 N 页**（抽检比例与结果见 `data/annotation/标注指南.md`）
- 页面切分：训练 {check["train_pages"]} 页 / 测试 {check["test_pages"]} 页（按 page_id 哈希，种子 2026）
- 计分范围：**{"仅测试页" if test_only else "全部页面"}**（测试页 {len(results.get("test_pages", []))} 个）
- 预测文件条数：{json.dumps(pred_counts, ensure_ascii=False)}
- 重跑命令：`python eval/run_all.py`（幂等）；`--refresh` 强制重跑 OCR / 抽取 / 问答

## 一、五项指标实测值与达标判定

{_metric_rows(results)}

> 阈值依据 `docs/赛题二原文.md` 第 1 节：字段 F1 ≥ 0.80、可溯源率 ≥ 0.90、
> 无依据正确拒答率 ≥ 0.85；方剂表「可用水平」本工程口径为
> 行列结构正确率与单元格内容正确率**均 ≥ 0.80**（否则标未达标）。
> CER 无参考阈值，只报实测值。**未达标项如实写出，不做美化。**

### 逐项计算明细（分子 / 分母可复核）

{_detail_lines(results)}

## 二、繁体 / 异体字单列统计（赛题 5 分，不混入总体 CER）

| 项 | 页数 | 值 | 说明 |
|---|---|---|---|
| 繁体页 CER | {len(trad.get("pages", []))} | {_fmt(trad.get("cer"))} | {trad.get("note", "词表未就绪")} |
| 含异体字页 CER | {len(varsub.get("pages", []))} | {_fmt(varsub.get("cer"))} | {varsub.get("note", "词表未就绪")} |
| 异体字归一化命中率 | — | {_fmt(variant.get("value"))} | {variant.get("note") or variant.get("reason", "")}（命中 {variant.get("hit", 0)} / {variant.get("total", 0)}） |

## 三、环境与降级（如实登记）

{env_lines}

- 预测生成：{predict_meta.get("reason") or ("已执行" if predict_meta.get("predictions_generated") else "复用已有预测文件")}
- 若 OCR 依赖不可用，CER 必须写「待测」，**不得用示例文本冒充识别结果**。

## 四、数字溯源

| 数字 | 输入文件 | GT 版本 |
|---|---|---|
| CER / 繁体 / 异体字 | `data/annotation/ocr_transcription.jsonl` + `data/reports/ocr_predictions.jsonl` | `{ds.version}` |
| 字段 F1 | `data/annotation/entities.jsonl` + `data/reports/extraction_predictions.jsonl` | `{ds.version}` |
| 表格结构 | `data/annotation/tables.jsonl` + `data/reports/table_predictions.jsonl` | `{ds.version}` |
| 可溯源率 | `data/reports/qa_log.jsonl` + `data/reports/ocr_predictions.jsonl`（原文块） | `{ds.version}` |
| 拒答率 | `data/annotation/qa_no_evidence.jsonl` + `data/reports/qa_log.jsonl` | `{ds.version}` |

- GT 校验：{json.dumps(check, ensure_ascii=False)}
- `is_demo=True` 记录由 `eval/dataset.py:clean_records()` 与五个 `compute_*` 双重排除。

## 五、反向溯源审计（真实问答日志，与可溯源率交叉核对）

| 项 | 值 |
|---|---|
| 参与审计的证据总数 | {audit.get("total", 0)} |
| 完全可溯（traced） | {audit.get("traced", 0)} |
| 部分可溯（partial） | {audit.get("partial", 0)} |
| 未溯（miss） | {audit.get("miss", 0)} |
| 无摘抄（no_quote） | {audit.get("no_quote", 0)} |
| 逐片段完全可溯比 | {_fmt(audit.get("audit_ratio"))} |
| 整串 quote 命中比 | {_fmt(audit.get("whole_hit_rate"))} |

> {audit.get("cross_check", "")}
> 可溯源率主指标采用**整串 quote 回原文命中**口径（与 `qa/generate.py:verify_evidences()` 一致），
> 本节逐片段口径用于发现「部分可溯」的证据，二者差异如实列出，不互相替代。
> {audit.get("note", "")}

## 六、链路自检（示例数据，非指标）

{_self_check_section()}

## 七、结论与未达标项

""" + "\n".join(
        f"- {s['name']}：{_fmt(results.get(s['key'], {}).get('value'))} "
        f"（{results.get(s['key'], {}).get('judge', '待测')}）"
        for s in METRIC_SPECS) + f"""

- 命题单位样本：**未到位 / 待测**，本报告不含任何命题样本数字。
- 免责声明：{CONFIG.disclaimer}；问答仅返回原文摘录，未做推断。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="赛题二 · 一键正式评测")
    ap.add_argument("--refresh", action="store_true", help="强制重跑 OCR / 抽取 / 问答")
    ap.add_argument("--no-predict", action="store_true", help="只读已有预测文件")
    ap.add_argument("--all-pages", action="store_true", help="全量页计分（默认仅测试页）")
    args = ap.parse_args()

    CONFIG.paths.ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predict_meta = _ensure_predictions(refresh=args.refresh, no_predict=args.no_predict)
    ds = Dataset.load()
    predictions = _load_predictions()
    pred_counts = {k: len(v) for k, v in predictions.items()}
    blocks_by_page = _blocks_by_page(predictions.get("ocr", []))
    results = compute_all(ds, predictions, blocks_by_page, test_only=not args.all_pages)
    audit = audit_qa_logs(predictions.get("qa_log", []), blocks_by_page)

    FORMAL_REPORT.write_text(
        build_formal_report(ds, results, predict_meta, pred_counts,
                            test_only=not args.all_pages, audit=audit), encoding="utf-8")
    detail = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notice": SYNTHETIC_NOTICE,
        "disclaimer": CONFIG.disclaimer,
        "gt_version": ds.version,
        "dataset": ds.check(),
        "predict_meta": predict_meta,
        "prediction_counts": pred_counts,
        "results": results,
        "trace_audit": audit,
        "reproduce": "python eval/run_all.py [--refresh]",
    }
    DETAIL_JSON.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    SKELETON_REPORT.write_text(_skeleton_report(report_metrics()), encoding="utf-8")

    print(f"正式评测报告：{FORMAL_REPORT}")
    print(f"评测明细：{DETAIL_JSON}")
    print(f"GT 版本：{ds.version} | 测试页 {len(results.get('test_pages', []))} 个")
    for key, label in (("cer", "CER"), ("field_f1", "字段F1"),
                       ("table_structure", "表格结构"), ("traceability", "可溯源率"),
                       ("refusal", "拒答率")):
        r = results.get(key, {})
        print(f"  {label}: {_fmt(r.get('value'))} （{r.get('judge', '待测')}）")
    print(f"数据声明：{SYNTHETIC_NOTICE}；命题单位样本未到位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
