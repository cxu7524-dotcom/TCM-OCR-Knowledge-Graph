# 中医古籍 OCR 与知识图谱智能问答原型

面向中医古籍与中药资源的 **OCR 识别 → 结构化抽取 → 知识图谱 → 可溯源问答** 四步链路原型。

> ### 【免责声明】仅供参考、非诊疗建议
> 本系统为赛事研发原型，所有内容不构成诊断、用药或治疗建议，不得用于临床决策。
>
> ### 【数据声明】合成示意数据，用于工程自测，非命题单位样本、非官方评测结果
> 命题单位脱敏样本**尚未到位**；当前全部实测值来自本地合成的 24 页竖排繁体示意页，
> 只证明「指标计算链路可跑通」，**不是官方评测结论**。合成集与命题样本结果分列、不混算。

---

## 1. 当前阶段：阶段二（本地 OCR 接入 + 首轮实测）

阶段一交付了工程骨架与数据契约；阶段二把能验证的桩换成了真实实现。

| 项 | 状态 |
|---|---|
| 本地 OCR 真实推理 | **已接入**：`rapidocr-onnxruntime`（包内自带 ONNX 权重，CPU、离线、无密钥） |
| 竖排 / 版面后处理 | **已实现**：检测框 x 投影列聚类、从右向左阅读顺序、列内碎片合并、页眉前置 |
| 繁简 / 异体字 | **已实现**：`data/lexicon/异体字表.json` 单一字表，多字优先，通假只登记不改写 |
| 字形纠错层 | **已实现**：`ocr/correct.py` + `data/lexicon/ocr_confusions.json`，只做上下文可判定的纠错，`TCM_OCR_CORRECT=0` 可关闭，每次纠错写 trace |
| 方剂表结构还原 | **已实现**：bbox 行列聚类 + 缺失单元格补识别 + 短文本整格二次识别；实测结构 2/2、单元格级 0.9444 |
| 对齐口径 | **已修正**：`app/schemas.py:align_norm()` 唯一实现，GT 与预测对称归一化，报告双列披露修正前后 |
| 五项指标实测 | **已实现**：五个 `compute_*` + 正式报告（见第 3 节，未达标项如实标注） |
| 无依据问题集 | **35 条**，三类覆盖（原文未出现 / 个人诊疗 / 疗效剂量越界），正确拒答 35/35 |
| 本地 LLM | **仍未接入**（`TCM_LLM_BACKEND=none`，阶段三） |
| 语义向量 | **仍未接入**（`qa/index.py` 的 hash 后端是占位，非语义向量） |
| 命题单位样本 | **未到位**；到位后单独登记、单独跑、与合成集分列 |

## 2. 快速开始

环境：Python 3.10+（本机 3.13 实测）；纯 CPU；**不需要数据库服务、不需要密钥、不需要 GPU、不联网推理**。

```bash
pip install -r requirements.txt      # 含本地 OCR 依赖，纯 CPU
python app/main.py                   # 或双击 启动操作台.bat
```

- 页面地址：<http://127.0.0.1:8000/>　接口文档：<http://127.0.0.1:8000/docs>
- 上传一张古籍页 → **真实 OCR 识别**（bbox 可回原图叠加）；点「加载示例古籍页」→ 走 `data/demo/` 桩数据并标注
- 换端口：`python app/main.py --port 8100`（或设 `TCM_PORT=8100`）

重建合成数据与重跑评测（全部可复现）：

```bash
python tools/make_synthetic_pages.py     # 24 页竖排繁体/方剂表/异体字页 + 六类 GT
python -m eval.dataset --check           # GT 校验与版本哈希
python eval/run_all.py --refresh         # 真实 OCR + 抽取 + 问答跑批 + 五项指标
python -m unittest discover -s tests -v  # 版面 / 归一化 / 指标口径单测
```

## 3. 阶段三实测结果（首轮 + 口径修正 + 字形纠错）

> 数据来源：`SYN-2026-001` 合成示意集（24 页，测试页 7 页，按 page_id 哈希切分，种子 2026）。
> GT 版本：`514f99246f103a95`（六类 GT 内容 sha256 前 16 位，**本阶段未改 GT**）。
> 完整报告与逐项分子分母：`data/reports/正式评测报告.md`、`data/reports/评测明细.json`；
> OCR 错误诊断：`data/reports/ocr_error_analysis.md`。

| 指标 | 实测值 | 参考阈值 | 达标情况 | 数据来源 |
|---|---|---|---|---|
| CER 字符错误率 | **0.0890**（阶段二 0.1911） | 无参考阈值 | 无阈值 | 合成集测试页（真实 OCR vs GT 转写） |
| 字段级 P / R / F1 | **0.6471**<br>（修正前对照 0.4314） | ≥ 0.80 | **未达标** | 合成集测试页 |
| 方剂表结构还原 | **0.0000**（整表全对口径）<br>结构 2/2、**单元格级 0.9444** | ≥ 0.80 | **未达标** | 合成集 2 张测试表 |
| 问答可溯源率 | **1.0000**（106/106） | ≥ 0.90 | **达标**（口径局限见下） | 真实问答日志（67 条跑批） |
| 无依据正确拒答率 | **1.0000**（35/35） | ≥ 0.85 | **达标** | 35 条无依据问题集 |
| 应答题正确作答率（配对） | **0.9062**（29/32） | 无（配对观察） | 配对观察 | 32 条有依据问题 |
| 繁体页 CER / 含异体字页 CER | **0.0890 / 0.0669** | 单列统计 | 单列 | 繁体页与异体字页分别统计 |
| 异体字归一化命中率 | **0.4791**（183/382 覆盖字符） | 单列统计 | 单列 | 合成集异体字页 |

### 3.1 本阶段做了什么（对应 P0/P1/P2）

- **P0 对齐口径修正**：新增唯一实现 `app/schemas.py:align_norm()` / `entity_align_key()`，
  `eval/dataset.py` 与 `extract/evaluate.py` 的 `align_key()` 都只是委托包装；
  GT 与预测**对称**归一化（去空白标点 + 异体/繁简映射）。GT 原文与 `quote` 一字未改。
  报告**双列披露**修正前后 F1。
- **P1 繁体 OCR**：先诊断（`eval/ocr_error_analysis.py`）后改代码。诊断结论：错误块中
  识别错字 71.8%、检测漏框 23.1%、切分错位 5.1%，繁体字错误占 62.7% → **瓶颈是字形能力，不是切图策略**。
  因此走「字形纠错层 + 表格补格/整格二次识别」路线（本地包只有简体 rec 模型，无网络无法换繁体模型，已如实登记）。
  CER 从 0.1911 降到 0.0890。
- **P2 抽取粒度**：统计 FP 24 条中粒度类 7 条（29.2%），其中多数源于 OCR 漏字（缺「气」「咳」），
  不是规则切分粒度；已在 `data/annotation/标注指南.md` 第五节写成字段粒度规范，**未改 GT**。

### 3.2 未达标原因（如实，不美化）

- **字段 F1 0.6471（阈值 0.80）**：剩余误差主要来自 ① OCR 漏字导致主治短语缺字
  （「五脏六腑寒热邪气」→「五脏六腑寒热邪」）；② GT 功效实体是机器生成的截断词
  （如「主上气咳」），词典无法覆盖；③ 少量繁体字形差异（如「温」/「溫」不做全局回写）。
  **未通过改写 GT、只归一化预测、或放宽阈值来提分。**
- **表格整表 0.0000**：口径要求「整表所有单元格全对」。结构 2/2 正确、单元格级 0.9444
  （18 格中 17 格正确），2 张测试表各差 1 个炮制单元格（「擘」在低置信二次识别中输出「璧」，
  已登记纠错规则但仍未稳定命中）。**未调整口径。**
- **可溯源率 1.0000 是构造性口径**：`qa/generate.py` 只输出通过 `quote` 校验的证据，
  未通过的已转拒答，因此该指标接近闭环自校验；它衡量「输出证据能否回原文」，
  **不等于「答案是否回答了问题」**，须与「应答题正确作答率 0.9062」配对解读。

### 3.3 真实古籍页实测（外部评测集，非命题单位样本）

合成集数字不能外推。在 60 张真实木刻扫描页（兵家 / 天文算法 / 传记类，无医家类）上，
按书切分（调参 = 天文問答 14 页，验证 = 其余 34 页）：

| 配置 | CER 中位 | 覆盖率中位 | 块数中位 |
|---|---|---|---|
| 修前：整页 + 简体 rec | 0.4451 | 0.6451 | 19 |
| 修后：整页 + **繁体 cht** rec | **0.2911** | **0.7643** | 18 |
| 修后：竖排逐列 + 整页择优 + 繁体 | 0.2911 | 0.7643 | 18 |

- **主要提升来自繁体识别模型**（`chinese_cht_PP-OCRv3_rec_mobile.onnx`，字典内嵌在 ONNX metadata），
  不是竖排切列；竖排逐列在版式规整页更好（CER 0.53→0.40），但在表格/窄列页会字数暴降，
  因此加了「整页结果兜底 + 两路择优」，保证不劣化。
- 诊断文档里的 0.8890 是**未做阅读顺序排序**的拼接结果；同 pipeline 基线是 0.4796，
  两者不可直接比，详见 `docs/真实古籍页诊断.md` 第七节。
- 未达标：块数 18 < 60（与 CER 存在 trade-off，优先 CER）；PP-OCRv5 模型下载损坏未接入。
- 复现：`python tools/eval_guji.py --vertical 1 --split validation`（需 `TCM_OCR_REC_MODEL` 指向本地模型）。

## 4. 四步链路与接口

| 步骤 | 前端 | 后端接口 | 负责模块 |
|---|---|---|---|
| ① 上传古籍页 | 步骤一 | `POST /api/pages`（`multipart` 上传真实识别，或 `demo=1` 走示例数据） | `ocr/pipeline.py` + `app/store.py` |
| ② 显示 OCR 与结构化结果 | 步骤二 | `GET /api/pages/{id}/ocr` | `ocr/pipeline.py` + `extract/rules.py` |
| ③ 图谱可视化 | 步骤三 | `GET /api/graph` | `graph/build.py` + `graph/export.py` |
| ④ 可溯源问答 | 步骤四 | `POST /api/qa` | `qa/retrieve.py` → `qa/generate.py` |

辅助接口：`GET /api/health`（配置与模块状态）、`GET /api/pages`（页面清单）、`GET /api/meta`（文案与示例问题）、`GET /uploads/{name}`（上传页预览）。

行为约束：
- **没有原文证据必须拒答**，返回「原文未检索到依据」并带 `refused=true`；
- 生成前逐条校验 `Evidence.quote` 能在 `Block.text` 中命中，命中失败即转拒答（`qa/generate.py:verify_evidences()`，阶段二未放宽）；
- 个人诊疗 / 疗效 / 剂量建议类问题按输出边界直接拒答。

## 5. 目录结构

```
README.md / requirements.txt / 启动操作台.bat
├── app/            main.py（FastAPI）、config.py（唯一配置）、schemas.py（数据契约）
│   ├── demo.py     示例数据装载器与演示态标记
│   ├── store.py    SQLite + JSON 幂等入库
│   └── web/index.html   四步演示台（按 is_demo 动态标注）+ 常驻免责声明
├── ocr/            recognize（真实 ONNX 推理）/ layout（列聚类·阅读顺序）/ table（bbox 行列还原）
│                   normalize（字表归一化）/ pipeline（真实宽高·低置信复核队列）
├── extract/        schema（类型与词典）/ rules / llm_local（阶段三）/ evaluate
├── graph/          build / query / export
├── qa/             index / retrieve / generate / refuse / disclaimer
├── tools/          make_synthetic_pages.py（合成页 + 六类 GT 生成器）
├── tests/          test_layout.py（版面 / 归一化 / 指标口径单测）
├── data/
│   ├── lexicon/    异体字表.json（繁简 / 异体 / 通假，含来源与置信等级）
│   ├── samples/    来源登记.md（仅元数据）+ synthetic/pages.jsonl（页面元数据）
│   ├── annotation/ 六类 GT（*.jsonl）+ 标注指南.md + 数据集版本.md
│   ├── demo/       示例数据（is_demo=True，不计分）
│   ├── uploads/ store/ reports/   运行产物（不进仓库）
├── eval/           metrics（五个 compute_*）/ dataset（GT 校验与切分）/ predict（真实跑批）
│                   audit（反向溯源审计）/ run_all（一键正式评测）
└── docs/           赛题二原文.md / 改造说明.md / 交付物清单.md / archive/ / 阶段二提示词.md
```

## 6. 数据契约与溯源

字段定义集中在 `app/schemas.py`，**任何文本块、实体、三元组、答案证据都必须携带**：

`page_id` · `bbox[x1,y1,x2,y2]` · `line_id` · `quote` · `source_hash`

核心对象：`Block` / `Entity` / `Triple` / `Evidence` / `Answer` / `OCRResult` / `GraphData`。
存储层沿用「摘抄 + 坐标 + 批次幂等」：`run_id` 由来源指纹确定性生成，同页重复跑零膨胀（schema v2 唯一键不含 run_id）。

## 7. 指标口径与计算方式

`eval/metrics.py` 的五个 `compute_*` 全部用标准库实现，**拒绝 `is_demo=True` 与缺溯源字段的记录**，无有效样本时返回 `status="待测"`（不返回 0 分）：

| 指标 | 计算口径 |
|---|---|
| CER | 按页分组，页内 GT 按 line_id、预测按 reading_order 拼接，`Σ 编辑距离 / Σ GT 字数`；繁体页、含异体字页单列 |
| 字段 F1 | 按 `(page_id, type, name, value)` 集合对齐，TP/FP/FN 微平均 |
| 表格结构 | 行列结构序列 + 单元格文本序列都相同才算该表正确；另报单元格级正确率 |
| 可溯源率 | 每条证据 `quote` 回原文 `Block.text` 命中数 / 证据总数 |
| 拒答率 | 期望拒答的问题中实际 `refused=true` 的比例；未跑批计入分母 |

默认只在**测试页**计分（`--all-pages` 可全量）。GT 版本哈希由 `eval/dataset.py:version_hash()` 计算，报告必须注明。

## 8. 本地化、安全与配置

- **推理全本地、零密钥、零公网调用**：`LlmConfig.is_local()` 硬拦截非 `127.0.0.1` / `localhost` 地址，本阶段未放宽；OCR 模型随包分发，运行时不下载。
- 配置集中在 `app/config.py`，环境变量前缀 `TCM_`：

  | 变量 | 默认 | 说明 |
  |---|---|---|
  | `TCM_OCR_BACKEND` | `rapidocr` | `rapidocr`（默认，真实推理）/ `paddleocr`（未实现，选中即降级）/ `stub` |
  | `TCM_OCR_MIN_CONF` | `0.35` | 低置信过滤阈值；低于阈值写入 `data/reports/ocr_review_queue.jsonl` 复核队列 |
  | `TCM_OCR_BOX_THRESH` / `TCM_OCR_TEXT_SCORE` | `0.30` / `0.35` | 检测阶段阈值（下调以提升表格小单元格召回） |
  | `TCM_OCR_CORRECT` | `1` | 字形纠错层开关；设 `0` 可完全回滚，纠错记录见 `data/reports/ocr_corrections.jsonl` |
  | `TCM_LLM_BACKEND` | `none` | 阶段二不部署；`ollama` / `vllm` 仅允许本机地址 |
  | `TCM_VECTOR_ENABLED` | `0` | 语义向量阶段三启用 |
  | `TCM_PORT` | `8000` | 服务端口 |

- 每个页面常驻「仅供参考、非诊疗建议」；问答输出边界由 `qa/disclaimer.py` 统一收口。

## 9. 数据合规与保密义务

- 命题单位脱敏样本**仅限赛事研发使用，不得外传**；真实扫描页 / PDF / 命题标注已列入 `.gitignore`（`data/samples/*.png`、`data/annotation/official/`），不进仓库、不上传任何外部服务。
- 合成影像（`data/samples/synthetic/pages/`）不进仓库，可用 `tools/make_synthetic_pages.py` 固定种子重建；合成 GT 与页面元数据进仓库以保证评测可复现。
- 合成数据在界面、接口、报告、数据集版本说明中**全程标注**「合成示意数据，用于工程自测，非命题单位样本、非官方评测结果」。
- 旧 A23 资产已从主链路移除，索引见 `docs/archive/`，内容可由基线 tag 恢复。

## 10. 回退到改造前

```bash
git tag                          # baseline-a23 与阶段一框架提交均保留
git show baseline-a23:<旧路径> > <恢复路径>
git reset --hard baseline-a23    # 整体回退（会丢弃框架与阶段二提交，谨慎）
```

## 11. 未完成项与下一步（阶段三）

按优先级：① 命题单位样本接收登记 + 双人独立标注 + Kappa 与人工抽检 → ② 本地 LLM（Ollama/vLLM）抽取与生成 + 输出二次 quote 校验 → ③ 语义向量检索 → ④ 技术报告 / PPT / 演示视频（见 `docs/交付物清单.md`）→ ⑤ 用命题样本重跑五项指标并与合成集分列。详见 `docs/改造说明.md`。
