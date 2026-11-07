# 中医古籍 OCR 与知识图谱智能问答原型

面向中医古籍与中药资源的 **OCR 识别 → 结构化抽取 → 知识图谱 → 可溯源问答** 四步链路原型。

> ### 【免责声明】仅供参考、非诊疗建议
> 本系统为赛事研发原型，所有内容不构成诊断、用药或治疗建议，不得用于临床决策。
> 当前阶段 **模型未接入、指标未实测**，界面与报告中的一切内容均为 `data/demo/` 示例数据。

---

## 1. 当前阶段：只搭框架

本阶段交付 **可运行、可演示、可复现的工程骨架与接口契约**，不训练模型、不追指标、不编造实测数字。

| 已打通 | 尚未做 |
|---|---|
| 四步端到端空跑链路（上传 → 识别与结构化 → 图谱 → 问答） | OCR 模型选型实测与精度优化 |
| 统一数据契约与全链路溯源字段 | 异体字全量字表、真实规模图谱 |
| SQLite + JSON 幂等入库 | 本地 LLM 部署与调参 |
| 零依赖前端四步演示台 + 力导向图 | 古籍样本采集与人工标注 |
| 评测脚本骨架与五项指标口径 | 任何指标数字（一律「待测」） |

**所有桩输出都在界面、接口响应和评测报告里显式标注「示例数据 / 模型未接入」。**

## 2. 快速开始

环境：Python 3.10+；**不需要数据库服务、不需要密钥、不需要 GPU、不联网即可推理**。

```bash
pip install -r requirements.txt
python app/main.py                 # 或双击 启动操作台.bat
```

- 页面地址：<http://127.0.0.1:8000/>
- 接口文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>
- 换端口：`python app/main.py --port 8100`（或设 `TCM_PORT=8100`）
- 演示数据：`data/demo/`（点击第一步的「加载示例古籍页」即可走通全流程）
- 评测重跑：`python eval/run_all.py`（报告输出到 `data/reports/`）

零依赖自检（不装 FastAPI 也能验契约与评测骨架）：

```bash
python -m app.store --init
python eval/run_all.py
```

## 3. 四步链路与接口

| 步骤 | 前端 | 后端接口 | 负责模块 |
|---|---|---|---|
| ① 上传古籍页 | 步骤一 | `POST /api/pages`（`multipart` 上传，或表单 `demo=1` 走示例数据） | `ocr/` + `app/store.py` |
| ② 显示 OCR 与结构化结果 | 步骤二 | `GET /api/pages/{id}/ocr` | `ocr/pipeline.py` + `extract/rules.py` |
| ③ 图谱可视化 | 步骤三 | `GET /api/graph` | `graph/build.py` + `graph/export.py` |
| ④ 可溯源问答 | 步骤四 | `POST /api/qa` | `qa/retrieve.py` → `qa/generate.py` |

辅助接口：`GET /api/health`（配置快照）、`GET /api/pages`（页面清单）、`GET /api/meta`（演示模式说明）、`GET /uploads/{name}`（已上传页预览）。

问答行为约束：**没有原文证据必须拒答**，返回「原文未检索到依据」并带 `refused=true`，禁止模型自由发挥；生成前逐条校验 `Evidence.quote` 能在 `Block.text` 中命中，命中失败即转拒答。

## 4. 目录结构

```
README.md / requirements.txt / 启动操作台.bat
├── app/            main.py（FastAPI 入口）、config.py（唯一配置）、schemas.py（数据契约）
│   ├── demo.py     示例数据统一装载器与演示态标记
│   ├── store.py    SQLite + JSON 幂等入库
│   └── web/index.html   四步演示台 + 常驻免责声明
├── ocr/            layout / recognize / table / normalize / pipeline
├── extract/        schema / rules / llm_local / evaluate
├── graph/          build / query / export
├── qa/             index / retrieve / generate / refuse / disclaimer
├── data/
│   ├── samples/    古籍页样本（占位 + 来源/脱敏说明）
│   ├── annotation/ 人工标注 ground truth（占位 + 字段规范）
│   ├── demo/       示例数据（保证零样本可演示）
│   ├── uploads/    运行期上传件（不进仓库）
│   ├── store/      本地库与 JSON 镜像（不进仓库）
│   └── reports/    评测报告（不进仓库）
├── eval/           metrics（指标口径）/ audit（反向溯源审计）/ run_all（一键重跑）
└── docs/           赛题二原文.md / 改造说明.md / archive/ / 项目改造提示词.md
```

## 5. 数据契约与溯源

字段定义集中在 `app/schemas.py`，**任何文本块、实体、三元组、答案证据都必须携带**：

`page_id` · `bbox[x1,y1,x2,y2]` · `line_id` · `quote` · `source_hash`

核心对象：`Block` / `Entity` / `Triple` / `Evidence` / `Answer` / `OCRResult` / `GraphData`。
存储层沿用旧工程的「摘抄 + 坐标 + 批次幂等」设计：每条入库记录带原文摘抄与 bbox，`run_id` 由来源指纹确定性生成，重复跑不膨胀。

## 6. 指标口径（**全部待测**）

`eval/metrics.py` 只定义口径，不给数字；无标注数据时 `eval/run_all.py` 一律输出「待测」并列出缺什么。

| 指标 | 口径 | 需要什么才能实测 |
|---|---|---|
| CER 字符错误率 | 编辑距离(识别文本, 人工转写) / 人工转写字数，按页加权 | `data/annotation/` 逐页转写 GT |
| 字段级 P / R / F1 | 按 `(page_id, type, name, value)` 对齐预测与 GT，微平均 | 实体与关系人工标注 |
| 表格结构还原准确率 | 行列结构 + 单元格内容同时正确才算对 | 方剂表 / 剂量表结构 GT |
| 可溯源率 | 证据 `quote` 能回原文命中的条数 / 证据总条数 | 真实问答日志 |
| 拒答率 | 无依据问题中正确拒答数 / 无依据问题总数 | 人工构造的无依据问题集 |

> 禁止把 `data/demo/` 的示例数据当评测结果；禁止填任何未经实测的百分比。

## 7. 本地化、安全与配置

- **推理全本地、零密钥、零公网调用**：配置只允许指向 `127.0.0.1` / `localhost`，不读取任何 API Key。
- 配置集中在 `app/config.py`，环境变量统一前缀 `TCM_`：

  | 变量 | 默认 | 说明 |
  |---|---|---|
  | `TCM_OCR_BACKEND` | `rapidocr` | 可选 `rapidocr` / `paddleocr` / `stub`；依赖缺失时自动降级为桩并标注 |
  | `TCM_LLM_BACKEND` | `none` | `none`（规则+桩）/ `ollama` / `vllm`，后两者走本机 OpenAI 兼容 HTTP |
  | `TCM_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | 本机 Ollama 默认地址 |
  | `TCM_VECTOR_ENABLED` | `0` | 是否启用本地向量通道（占位实现，非语义向量） |
  | `TCM_PORT` | `8000` | 服务端口 |

- 切换本地 LLM 示例：`set TCM_LLM_BACKEND=ollama && set TCM_LLM_MODEL=qwen2.5:7b`，然后重启服务。
- 每个页面常驻「仅供参考、非诊疗建议」免责声明；问答输出边界由 `qa/disclaimer.py` 统一收口。

## 8. 数据合规与保密义务

- `data/samples/` 中的**命题单位脱敏样本仅限赛事研发使用，不得外传**；真实扫描页、PDF 与人工标注文件已列入 `.gitignore`，不进仓库、不上传任何外部服务。
- `data/demo/` 为**合成示意数据**，不含真实古籍影像、不含患者信息，可自由用于演示。
- 参赛者须遵守赛事数据使用协议：不得复制、公开、转授样本；演示录屏与截图若含脱敏样本需自行复核。
- 旧 A23 资产（含合同、投诉、采购等办公文档）已从主链路移除，索引见 `docs/archive/`，完整内容可通过基线 tag 恢复。

## 9. 回退到改造前

```bash
git tag                      # 应能看到 baseline-a23
git show baseline-a23:<旧文件路径> > <恢复路径>
git reset --hard baseline-a23  # 需要整体回退时（会丢弃框架提交，谨慎）
```

## 10. 未完成项与下一步

按优先级：① 接入本地 OCR 模型与竖排版面后处理 → ② 古籍样本采集、脱敏与人工标注 → ③ 本地 LLM 抽取/生成与证据强约束 → ④ 五项指标实测 → ⑤ 报告 / PPT / 演示视频。详见 `docs/改造说明.md`。
