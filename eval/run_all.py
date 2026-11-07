# -*- coding: utf-8 -*-
"""
eval/run_all.py —— 一键重跑评测骨架并产出报告

跑法（项目根目录，无需 FastAPI、无需数据库服务、无需密钥）：
    python eval/run_all.py

幂等：每次覆盖写 data/reports/评测报告骨架.md，可反复执行。
约束：五项指标一律「待测」，不填任何未经实测的数字；
      示例数据只出现在「链路自检明细」里，并明确排除在指标之外。

TODO（下一阶段）：GT 就绪后接 eval/metrics.py 的 compute_*，
     输出分页面 / 分类型明细与置信区间，并把报告升级为正式评测报告。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# 允许直接 `python eval/run_all.py`（此时 sys.path[0] 是 eval/）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                                   # noqa: E402
from app.demo import load_demo_ocr, load_demo_questions           # noqa: E402
from app.store import init_db, stats                              # noqa: E402
from eval.audit import self_check                                 # noqa: E402
from eval.metrics import missing_requirements, report_metrics     # noqa: E402

REPORT_PATH = CONFIG.paths.reports_dir / "评测报告骨架.md"
SELF_CHECK_QUERIES = [
    "甘草的性味是什么？",       # 应可答
    "甘草有什么禁忌？",         # 应可答
    "黄芪在本书中用量几何？",   # 应拒答：原文未检索到依据
    "本品对晚期肝癌的治愈率是多少？",  # 应拒答：越界
]


def _metric_table(metrics: dict) -> str:
    lines = ["| 指标 | 口径 | 状态 | 数值 | 缺失条件 |",
             "|---|---|---|---|---|"]
    for m in metrics["metrics"]:
        lines.append(
            f"| {m['name']} | {m['formula']} | **{m['status']}** | "
            f"{'—' if m['value'] is None else m['value']} | {'；'.join(m['needs'])} |")
    return "\n".join(lines)


def _missing_list() -> str:
    return "\n".join(
        f"- **{m['name']}**：缺 {'；'.join(m['missing'])}"
        for m in missing_requirements())


def _self_check_section() -> str:
    """链路自检：仅证明四步能跑通与拒答生效，不产出任何指标。"""
    try:
        blocks = load_demo_ocr("DEMO-001").blocks
        result = self_check(blocks, SELF_CHECK_QUERIES)
    except Exception as exc:
        return f"链路自检未执行：{type(exc).__name__}（不影响指标「待测」结论）"

    lines = [
        "> 以下全部来自 `data/demo/` **示例数据 / 模型未接入**，"
        "仅用于验证链路与拒答逻辑，**不是评测结果、不计入上方任何指标**。",
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


def build_report() -> str:
    """生成 Markdown 报告文本（幂等，不含任何实测数字）。"""
    metrics = report_metrics()
    init_db()
    db = stats()
    demo_q = load_demo_questions()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# 赛题二 · 评测报告骨架（框架搭建阶段）

> ### 【免责声明】{CONFIG.disclaimer}
> 本报告为**框架阶段产物**，五项指标全部为「待测」。
> 示例数据 / 模型未接入，任何数值均不得作为评测结论或诊疗依据。

- 生成时间：{now}
- 阶段：{metrics["phase"]}
- 政策：{metrics["policy"]}
- 标注 GT 是否就绪：{"是" if metrics["ground_truth_available"] else "否（data/annotation/ 仅有字段规范）"}
- 本地库统计（运行产物，非指标）：页面 {db.get("pages", 0)} / 文本块 {db.get("blocks", 0)} / 实体 {db.get("entities", 0)} / 三元组 {db.get("triples", 0)}
- 示例问题数：{len(demo_q)}（仅供演示）

## 一、五项指标口径与状态

{_metric_table(metrics)}

**结论：五项均为「待测」。** 本阶段不训练模型、不采集标注、不做任何实测，
禁止用示例数据或占位比例冒充 CER / F1 / 可溯源率 / 拒答率。

## 二、要跑出真实指标，还缺什么

{_missing_list()}

另需补齐：`eval/metrics.py` 中 compute_* 实现、GT 与预测的统一对齐工具、
人工复核裁决记录、以及命题单位样本的脱敏与使用审批记录。

## 三、链路自检明细（示例数据，非指标）

{_self_check_section()}

自检目的仅有两个：① 四步链路能从上传走到问答；② 无依据 / 越界问题会拒答
并说明「原文未检索到依据」。**示例数据的通过率不是可溯源率，也不是拒答率。**

## 四、数据合规

- 命题单位脱敏样本仅限赛事研发使用，不得外传；真实扫描页与人工标注不进仓库、
  不上传任何外部服务（见 `.gitignore` 与 `data/samples/README.md`）。
- `data/demo/` 为合成示意数据，不含真实古籍影像与患者信息。

## 五、重跑与下一步

重跑：`python eval/run_all.py`（覆盖本报告，幂等）。
下一步优先级：① 本地 OCR 接入 → ② 古籍样本与人工标注 → ③ 本地 LLM 与证据强约束
→ ④ 五项指标实测 → ⑤ 报告 / PPT / 演示视频。
"""


def main() -> int:
    CONFIG.paths.ensure_dirs()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"评测报告骨架已生成：{REPORT_PATH}")
    print("五项指标：CER / 字段F1 / 表格结构还原 / 可溯源率 / 拒答率 → 全部「待测」")
    print(f"免责声明：{CONFIG.disclaimer}；示例数据 / 模型未接入，未产出任何实测数字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
