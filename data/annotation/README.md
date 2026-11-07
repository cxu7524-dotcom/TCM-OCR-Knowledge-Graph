# 人工标注 Ground Truth 目录（占位）

> 当前**没有任何标注数据**，因此五项指标一律「待测」。
> 标注文件（`.json` / `.jsonl`）已列入 `.gitignore`，不进仓库；本说明与字段规范可进仓库。

## 文件规范

| 文件 | 用途 | 对应指标 |
|---|---|---|
| `ocr_transcription.jsonl` | 逐页逐行人工转写 | CER 字符错误率 |
| `entities.jsonl` | 实体与字段标注 | 字段级 P / R / F1 |
| `relations.jsonl` | 三元组标注 | 字段级 P / R / F1（关系线） |
| `tables.jsonl` | 方剂表 / 剂量表结构与单元格 | 表格结构还原准确率 |
| `qa_log_gold.jsonl` | 真实问答日志与期望答案 | 可溯源率 |
| `qa_no_evidence.jsonl` | 人工构造的无依据问题集（期望 `refused=true`） | 拒答率 |

## 统一字段（与 `app/schemas.py` 对齐）

每条标注必须携带溯源五元组，否则视为无效标注：

```json
{
  "page_id": "S2026-001__p0001",
  "bbox": [x1, y1, x2, y2],
  "line_id": "L03",
  "quote": "原文摘抄，须与 Block.text 一字不差",
  "source_hash": "来源指纹（16 位）",
  "type": "药材",
  "name": "甘草",
  "value": "甘草",
  "annotator": "标注人代号",
  "version": "v1"
}
```

- `bbox` 原点在图像左上角，单位像素，格式 `[x1,y1,x2,y2]`，需满足 `x2>=x1, y2>=y1`。
- `quote` **不得改写**，繁简 / 异体差异请记在 `text_norm` 或备注里，避免审计口径漂移。
- `entities.jsonl` 的对齐键为 `(page_id, type, name, value)`，与 `extract/evaluate.py` 的 `align_key()` 完全一致。

## 标注流程

1. 以 `data/samples/` 的脱敏扫描页为唯一来源，逐页转写（OCR 结果只作参考，不直接采信）。
2. 实体 / 关系由两名标注人独立标注，不一致项交由第三人裁决并记录裁决理由。
3. 表格结构须同时标注行列坐标与单元格文本；结构或内容任一不符即判错。
4. 无依据问题集须覆盖「原文完全未出现」「需个人诊疗建议」「疗效 / 剂量类越界」三类。
5. 每轮标注产出版本号与变更日志，评测报告必须注明所用 GT 版本。

## 与评测的关系

- `extract/evaluate.py` 会拒绝 `is_demo=True` 的记录；`data/demo/` 的示例数据**不得**写入本目录。
- `eval/metrics.py:has_ground_truth()` 只检查本目录是否存在 `.json` / `.jsonl`，存在后才会由下一阶段的 `compute_*` 计算指标。
- 本阶段 `eval/run_all.py` 无论本目录是否有文件，五项指标仍输出「待测」（口径实现尚未接入）。

## 当前状态

- [x] 目录与字段规范
- [ ] 标注指南与一致性检验（Kappa / 抽检比例）
- [ ] 首批 GT 与双人复核记录
- [ ] 指标实测与正式评测报告
