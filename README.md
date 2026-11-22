# 中医古籍 OCR 与知识图谱问答原型

面向中医古籍整理与知识库录入的本地原型。系统支持上传扫描页、识别文字并还原版面、抽取药材和方剂等知识，通过图谱与可回查原文的问答辅助检索。

> **仅供参考、非诊疗建议。** 系统输出不构成诊断、用药或治疗建议。
>
> 当前评测使用合成示意集和公开古籍数据，命题单位样本尚未到位。已有结果用于研发验证，并非官方评测成绩。

## 当前进度

四步演示链路已跑通：

1. **上传页面**：支持常见图片格式，也可加载内置示例；真实上传使用本机 OCR 推理。
2. **识别与结构化**：查看原图与文字框、阅读顺序、实体和三元组。方剂表按行列展示，有可靠出处的单元格可定位到原图。
3. **图谱可视化**：按当前页展示实体及关系，点击节点查看来源。
4. **可溯源问答**：先用**语义向量**在全库检索最相关的原文段落，再把段落交给本机大模型生成解释，答案里的 `[编号]` 可回查到「书名 / 页码 / 行范围 / bbox / 原文」；库中没有依据时明确声明并改用通用中医常识作答，涉及个人诊疗、剂量、疗效的问题一律拒答。

OCR 使用本地 ONNX 模型；语义向量使用本地 ONNX 模型（bge-small-zh-v1.5，512 维）；生成使用本机 Ollama 上的大模型。三者全部本机推理，无 API 密钥、无云端调用。图谱问答基于已抽取的关系，适合逐页核对。

公开医书扫描页已验证上传、识别和问答回跳。真实页的实体与关系准确性仍需人工核验，例如《思濟堂方書》`page_35` 存在主治跨段归属错误。合成集指标不能直接代表真实古籍或命题样本表现。

## 语义检索与 RAG 问答

问答不是"关键词命中就摘一段"，而是完整的检索增强生成（RAG）链路：

```text
用户问题
  → 查询扩展（现代词 → 古籍用词：失眠→不寐/不得眠，腹泻→泄泻/下利 …）
  → 语义向量编码（本地 bge-small-zh-v1.5，512 维）
  → 全库余弦检索 top-k（知识库 = 所有已入库页面切成的语义段落）
  → 双阈值过滤：绝对相似度 ≥ 0.49，且 ≥ 最高分 × 0.72
  → 字面锚点校验：语义相近但一个实词二元组都对不上 → 判为无依据
  → 专业提示词（区分「原文依据」与「中医常识补充」两种口径）
  → 本机大模型生成（qwen2.5:7b）
  → 解析答案中的 [编号] → 映射回 书名 / 页 / 行 / bbox / 原文
```

四个设计取舍：

- **段落切分而非按行检索**：古籍 OCR 一行只有十几个字，单行做向量召回噪声极大；系统把相邻文本块合并成约 160 字的段落检索，命中后再回落到底层文本块给出精确出处。
- **阈值是实测出来的**：在 12 条域内 + 8 条域外问题上扫描阈值，0.49 时域内召回 100%、误命中 12%，再叠加字面锚点把纯语义误命中压掉。阈值可用 `TCM_RAG_MIN_SCORE` 调整。
- **查询扩展补古代用词**：古籍很少出现"失眠""腹泻""便秘"，内置约 30 组现代词→古籍词映射，命中后拼进检索串，并允许等价词充当锚点。实测"失眠"从"无依据"变成命中《温疫方論》「不得眠」并给出出处。
- **检索不到不等于拒答**：库中无依据时仍作答（`TCM_RAG_ANSWER_WITHOUT_EVIDENCE=1`，默认开），但答案开头会明确写「本古籍库未检索到直接依据」，并全文不出现任何 `[编号]` 引用，避免把常识伪装成古籍原文。个人诊疗 / 剂量 / 疗效类问题仍在检索之前直接拒答。

知识库规模可以用 `GET /api/rag/status` 查看，也可在首页问答区顶部看到「知识库 N 页 / M 段」徽标。

### 导入古籍语料

```powershell
# 1) 下载本地语义向量模型（约 90MB，仅在首次需要）
python tools/download_embed_model.py

# 2) 导入 ModelScope 公开古籍评测集《医家类》176 页（10 部医书，约 5.6 万字）
#    标注表缺失时会自动从 ModelScope 下载
python tools/ingest_corpus.py --source ancientdoc

# 3)（可选）下载评测子集 59 页原图并用真实 OCR 重建，
#    使这 59 页拥有真实检测框、置信度与阅读顺序（而非推算行框）
python tools/download_eval_pages.py      # 下载 59 张原图（约 80MB）
python tools/reingest_eval59.py          # 走生产 OCR 链路重建同名页面

# 4) 重建语义索引（也可由服务启动 / 上传新页时自动触发）
curl -X POST http://127.0.0.1:8100/api/rag/rebuild
```

`ingest_corpus.py` 导入的是**转写文本**而非原图，因此每个块的 `bbox` 是按阅读顺序推算的行框，`engine` 记为 `dataset-transcription`，`notes` 中写明来源；不冒充真实版面检测结果。其中评测子集 59 页已用原图重跑真实 OCR（`engine=rapidocr`，bbox 为真实检测框、含置信度），其余 117 页仍为转写文本。若换成自有语料，用 `--source jsonl`，每行 `{"page_id": "...", "book": "...", "text": "整页原文"}`。

测过的问题类型与典型表现：

| 问题 | 表现 |
|---|---|
| 甘草的性味和功效是什么 | 命中 3 段，最高相似度 0.71，答案带 3 条出处 |
| 温疫病邪伏于膜原怎么治 | 命中《温疫方論》3 段，全部标 `[1][2][3]` |
| 脉象寸盛尺弱说明什么 | 命中《合併脉訣難經太素評林》4 段 |
| 今天天气怎么样 | 命中最高的 0.48 未过阈值 → 声明无依据 + 常识作答 |
| 我最近咳嗽，该吃什么药？剂量多少 | 检索前直接拒答（越界），0.0 秒返回 |

## 给协作者：从克隆到跑通

仓库不包含模型权重、古籍原图和运行数据（见 `.gitignore`），克隆后按下面四步即可复现全部功能：

```powershell
# 0) 环境（Python 3.10+，在项目根目录执行）
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 1) 模型（约 110MB，全部本机推理，无 API 密钥、无云端调用）
#    OCR 识别模型 ch_PP-OCRv5_rec_mobile.onnx 的获取方式见下文「识别模型」；
#    语义向量模型一条命令：
python tools/download_embed_model.py

# 2) 知识库语料：导入医家类 176 页建库（标注表缺失时自动从 ModelScope 下载）
python tools/ingest_corpus.py --source ancientdoc
#    可选：下载 59 页评测原图并用真实 OCR 原位重建（出处可点回原图真实位置）
python tools/download_eval_pages.py
python tools/reingest_eval59.py

# 3) 启动服务（问答生成走本机 Ollama；没装也能跑，问答降级为原文摘录）
ollama pull qwen2.5:7b
$env:TCM_LLM_BACKEND = "ollama"; $env:TCM_LLM_MODEL = "qwen2.5:7b"
.\.venv\Scripts\python.exe app/main.py --port 8100

# 4) 浏览器打开（改了前端后按 Ctrl+F5 强刷）
#    首页问答（全库语义检索）：http://127.0.0.1:8100/home
#    四步演示台：            http://127.0.0.1:8100/
```

- 问答的检索链路、阈值与出处格式见上文「语义检索与 RAG 问答」；全部环境变量见「配置」一节。
- 改动代码前后跑回归：`python -m unittest discover -s tests -t .`（当前 206 用例，6 项按环境跳过）。
- 只存在于本机、不进仓库的东西：`data/store/`（SQLite 与向量索引）、`模型权重/`、
  `古籍评测集/`（原图）、`data/uploads/`、`data/screenshots/`。重建它们就是上面第 1–2 步。

## 快速开始

### 环境准备

- Python 3.10 及以上，本地工作区已用 Python 3.12 验证。
- CPU 可运行，无需独立数据库服务或 API 密钥。
- 安装依赖时需要访问 Python 包源；安装完成后，默认 OCR 推理在本机运行。

在项目根目录创建虚拟环境并安装依赖。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app\main.py
```

macOS 或 Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app/main.py
```

打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)。交互式接口文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

Windows 安装依赖后也可双击 `启动操作台.bat`。脚本优先使用项目内 `.venv`，服务运行期间请保持窗口开启。需要更换端口时运行 `python app/main.py --port 8100`。

### 识别模型

默认自动探测 `模型权重/ch_PP-OCRv5_rec_mobile.onnx`。该文件不随仓库提交；不存在时使用依赖包内的识别模型，实际模型可在页面配置栏和 `/api/health` 中查看。也可通过 `TCM_OCR_REC_MODEL` 指定本地识别模型路径。

与已有报告相同的 v5 模型可从 [RapidOCR 模型仓库](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx) 获取，保存到上述路径。下载与依赖安装需要联网，系统识别时使用本地文件。更换模型会影响识别效果及评测结果。

### 使用流程

1. 在首页上传扫描页，或点击「加载示例古籍页」。
2. 进入识别结果，点击文字框查看原文和坐标；若识别到方剂表，可查看表格并点击有出处的单元格。
3. 打开图谱，查看当前页的节点与关系。
4. 输入问题，查看答案及证据，点击证据回到对应原文。页面列表支持切换历史页和删除页面。

支持 `png`、`jpg`、`jpeg`、`bmp`、`webp`、`tif`、`tiff`。默认上传大小上限为 20 MB。

## 语音输入（普通话 / 粤语 / English）

首页问答面板的输入框左边有一个话筒和一个语种下拉。说话 → 文字**填进输入框**，不自动发送：先自己看一眼，有听错或错别字直接改，确认后再按发送。

两条通道，前端自动选，不用手动切：

| 通道 | 实现 | 特点 |
|---|---|---|
| ① 浏览器原生识别 | `SpeechRecognition`（`zh-CN` / `zh-HK` / `en-US`） | 实时出字、快；但要连微软 / 谷歌的语音服务器，国内经常连不上 |
| ② 本机离线识别 | 录音 → `POST /api/asr` → `faster-whisper` small / int8 / CPU | 慢 1~3 秒，全程不出本机，断网可用 |

第一次点话筒走 ①；一旦浏览器报 `network`（外网不通），就把这台机器记进 `localStorage`，以后直接走 ②，不再白等。

语种档位与实测（本机 small 模型，2026-11-20）：

| 档位 | 底层 | 实测 |
|---|---|---|
| 普通话 `zh` | `language="zh"` + 中医/文档领域提示词 | 「甘草有什么功效和禁忌？」「麻黄汤有哪些药材组成？」「当归补血汤的组成和用法」全对；「伤寒论」会听成「商涵论」（小模型对书名弱，靠用户改字） |
| 粤语 `yue` | `language="zh"` + 最短提示 | 「帮我将这份文件的标题改成季度报告」全对。**不能**用 `language="yue"`：这颗模型直接吐空还白跑十几秒 |
| 英语 `en` | `language="en"` | 「Please change the title of this document to Quarterly Report.」全对；挂中文档位会把英文硬转成中文胡说 |

启用本机通道：

```bash
pip install -r requirements.txt          # 装了 faster-whisper 才会启用 ②
# 模型放到 模型权重/faster-whisper-small/（约 461MB，不入库）
#   model.bin / tokenizer.json / config.json / vocabulary.txt
curl -s http://127.0.0.1:8100/api/asr/status   # ready=true 即可用
```

接口：

| 接口 | 用途 |
|---|---|
| `GET /api/asr/status` | 模型与依赖自检、可选语种；打开页面时顺带后台预热模型 |
| `POST /api/asr` | 表单 `file`（webm / m4a / wav）+ `lang`（`zh` / `yue` / `en`）→ `{text, seconds, duration, lang}` |

识别层只做「声音 → 文字」，不改写语义；识别结果为空（没说清、只有静音、或在念提示词）时返回 422「没听清，再说一次」，不会把幻觉结果塞进输入框。

## 评测与测试

仓库中的[正式评测报告](data/reports/正式评测报告.md)和[评测明细](data/reports/评测明细.json)记录了阶段九合成集结果。合成集共 24 页，其中 7 页用于测试，GT 版本为 `514f99246f103a95`。

| 指标 | 结果 | 口径 |
|---|---:|---|
| 字符错误率 CER | 0.0445 | 编辑距离 17 / 参考转写 382 字 |
| 字段级 F1 | 0.8283 | 归一化对齐，TP 41、FP 6、FN 11 |
| 方剂表结构还原 | 1.0000 | 整表 2/2，单元格 36/36 |
| 问答可溯源率 | 1.0000 | 已返回证据 29/29 可回查 |
| 无依据正确拒答率 | 1.0000 | 35/35 |
| 应答题正确作答率 | 0.9062 | 29/32，与拒答率配对观察 |
| 繁体页 / 含异体字页 CER | 0.0445 / 0.0176 | 分别统计 7 页 / 5 页 |

可溯源率采用构造性口径，衡量已输出证据能否回到原文，应与应答题正确作答率一起理解。上述报告生成于阶段九，新增图谱关系问答后的页面交互尚未纳入该批问答评测。

公开古籍评测数据单独统计。医家类 59 页的严格 CER 中位数为 0.8485，顺序无关 CER 中位数为 0.4556，覆盖率中位数为 0.7195。公开集参考转写由机器生成，繁简差异、图注和阅读顺序会影响对照。详细分层与计算口径见[医家类 59 页评测报告](docs/医家类59页评测报告.md)。

在项目根目录运行单测：

```bash
python -m unittest discover -s tests -v
```

本次工作区验证为 **122 条测试通过**。以下命令可重建合成页并重跑评测，`--refresh` 会重新生成本地预测与报告，模型、依赖和运行环境可能影响结果：

```bash
python tools/make_synthetic_pages.py
python -m eval.dataset --check
python eval/run_all.py --refresh
```

## 接口概览

| 接口 | 用途 |
|---|---|
| `GET /api/health` | 服务、模型与配置状态 |
| `GET /api/meta` | 演示文案、示例问题和词表信息 |
| `GET /api/metrics` | 读取本地评测明细 |
| `POST /api/pages` | 上传图片，或使用 `demo=1` 加载示例 |
| `GET /api/pages` | 查看已入库页面 |
| `DELETE /api/pages/{page_id}` | 删除页面及其派生数据 |
| `GET /api/pages/{page_id}/ocr` | 获取文字块、实体、三元组和方剂表 |
| `GET /api/graph?page_id=...` | 获取当前页图谱 |
| `POST /api/qa` | 提问并返回答案、出处或拒答；支持 `history` 多轮上下文与 `scope=page\|library` |
| `GET /api/rag/status` | 知识库规模、向量模型、索引状态与阈值参数 |
| `POST /api/rag/rebuild` | 强制重建语义向量索引 |
| `GET /api/asr/status` | 语音识别自检：模型与依赖、可选语种 |
| `POST /api/asr` | 录音转文字（`lang=zh\|yue\|en`），结果只填进输入框由用户改字 |

`POST /api/qa` 的响应除 `text` / `evidences` / `trace` 外，新增两个字段：

- `sources`：答案实际引用的出处列表，每条含 `index`（对应答案里的 `[n]`）、`book`、`page_id`、`line_span`、`quote`、`bbox`、`score`（余弦相似度）、`cited`（是否被答案显式引用）；
- `retrieval`：检索元信息，含 `mode`（`semantic-rag`）、`embed_model`、`hits`、`top_score`、`floor`（实际生效阈值）、`anchor_dropped`、`answered_without_evidence`。

文本块、实体、三元组和证据使用 `page_id`、`bbox`、`line_id`、`quote`、`source_hash` 追溯来源。问答输出前会校验证据与原文，图谱关系问答还会检查关系方向和主体归属。

## 配置

配置集中在 `app/config.py`，环境变量使用 `TCM_` 前缀。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `TCM_PORT` | `8000` | 服务端口 |
| `TCM_MAX_UPLOAD_MB` | `20` | 上传大小上限，单位 MB |
| `TCM_OCR_REC_MODEL` | 自动探测 | 本地 OCR 识别模型路径 |
| `TCM_OCR_MIN_CONF` | `0.35` | OCR 低置信过滤阈值 |
| `TCM_OCR_CORRECT` | `1` | 字形纠错开关 |
| `TCM_OCR_VERTICAL_SPLIT` | `1` | 竖排逐列处理开关 |
| `TCM_OCR_PICK` | `auto` | 整页与逐列结果选择策略 |
| `TCM_LLM_BACKEND` | `none` | 生成后端：`none` / `ollama` / `vllm` / `cloud`；本机演示用 `ollama`，云端抽取用 `cloud` |
| `TCM_LLM_MODEL` | `qwen2.5:7b` | 生成模型名 |
| `TCM_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | 本机后端仅允许 127.0.0.1 / localhost；`cloud` 后端要求 HTTPS 地址 |
| `TCM_LLM_TIMEOUT` | `120` | 生成超时（秒）；7B 模型首次推理需加载权重，过短会误判为不可用 |
| `TCM_LLM_RESPONSE_FORMAT` | `json_object` | 云端返回格式，可选 `json_schema`（调用 `chat_result` 时需同时提供 schema） |
| `TCM_LLM_API_KEY` | 无 | 云端服务凭据，只从服务端环境变量读取，不写入仓库 |
| `TCM_EXTRACT_BACKEND` | `auto` | 上传抽取后端：`auto` / `rules` / `cloud`；选 `cloud` 需同时把 `TCM_LLM_BACKEND` 设为 `cloud` |
| `TCM_EMBED_BACKEND` | `local` | 向量后端：`local`（ONNX，默认）/ `ollama` / `none` |
| `TCM_EMBED_MODEL` | `bge-small-zh-v1.5` | 向量模型名（写入索引 manifest，换模型自动重建） |
| `TCM_EMBED_MODEL_DIR` | `模型权重/bge-small-zh-v1.5` | 本地向量模型目录 |
| `TCM_RAG_ENABLED` | `1` | RAG 开关；关闭后回退到原有字面检索链路 |
| `TCM_RAG_MIN_SCORE` | `0.49` | 余弦相似度绝对阈值（实测拐点） |
| `TCM_RAG_RELATIVE_KEEP` | `0.72` | 相对阈值：只保留 ≥ 最高分 × 该值的段落 |
| `TCM_RAG_ANSWER_WITHOUT_EVIDENCE` | `1` | 无原文依据时是否用通用常识作答并显式声明 |
| `TCM_RAG_TOP_K` | `6` | 送给大模型的段落数上限 |
| `TCM_VECTOR_ENABLED` | `0` | 旧哈希向量通道开关（非语义，仅自检用） |

云端调用需在服务端进程环境中配置 `TCM_LLM_BASE_URL`（HTTPS，包含服务商要求的版本路径）、`TCM_LLM_MODEL` 和 `TCM_LLM_API_KEY`，凭据不落盘、不进仓库。云端抽取默认由 `TCM_EXTRACT_BACKEND=auto` 在 `TCM_LLM_BACKEND=cloud` 时启用，否则走规则；每页的抽取状态与提示词 / 协议版本会随页面 OCR 接口返回，调用失败时返回错误而不是静默降级。

配置后可在仓库根目录做一次人工短样本联通检查（输出状态、错误类别与结构化内容，凭据自动脱敏）：

```powershell
@'
import json
import os
from extract.llm_local import LocalLLMExtractor

client = LocalLLMExtractor()
schema = {"type": "object", "properties": {"entities": {"type": "array", "items": {"type": "string"}}}, "required": ["entities"], "additionalProperties": False}
result = client.chat_result([{"role": "user", "content": "Return a JSON object with an entities array for: \u7518\u8349\u4e09\u94b1"}], schema)
secret = os.environ.get("TCM_LLM_API_KEY", "")
def redact(value):
    return value.replace(secret, "[redacted]") if secret else value
print(json.dumps({"model": redact(client.cfg.model), "status": result.status, "error": redact(result.error), "result": redact(result.content)}, ensure_ascii=False))
'@ | .\.venv\Scripts\python.exe -
```

第 2 步六个人工短样本语义验证入口为 `tools/llm_extraction_six_cases.py`，每行输出一个 JSON 对象（输入、模型输出、预期、自动判定、模型与协议版本），自动判定仅供人工复核：

```powershell
$env:TCM_LLM_BACKEND = 'cloud'
$env:TCM_LLM_BASE_URL = 'https://www.autodl.art/api/v1'
$env:TCM_LLM_MODEL = 'DeepSeek-V4.1-Flash-1017'
$env:TCM_LLM_RESPONSE_FORMAT = 'json_object'
$env:TCM_LLM_MAX_TOKENS = '4096'
.\.venv\Scripts\python.exe tools/llm_extraction_six_cases.py
```

## 项目结构

```text
app/        FastAPI 接口、配置、数据契约、存储与网页
ocr/        OCR、版面分析、竖排处理、表格还原、字形归一化
extract/    词表、实体与关系抽取规则
graph/      图谱构建、查询与导出
qa/         语义向量检索、RAG 生成、图谱关系问答、证据校验与拒答
eval/       数据集校验、预测和指标计算
tools/      合成页生成、古籍语料导入、向量模型下载、公开古籍评测与诊断脚本
tests/      OCR、抽取、表格展示与问答回归测试
data/       词表、示例、标注、向量索引及本地运行数据
docs/       技术报告、评测说明与项目记录
```

RAG 相关模块：

| 文件 | 职责 |
|---|---|
| `qa/embed_local.py` | 本地 ONNX 向量推理（tokenizers + onnxruntime，不依赖 torch） |
| `qa/embed.py` | 向量后端统一入口（local / ollama 双后端，失败即降级不伪造） |
| `qa/kb.py` | 知识库语料加载与段落切分（相邻块合并成 ~160 字段落） |
| `qa/index.py` | 向量索引：持久化到 `data/store/vectors/<model>/`，语料变化自动重建 |
| `qa/prompt.py` | RAG 提示词（区分原文依据与常识补充、禁止剂量与诊疗建议） |
| `qa/rag.py` | RAG 主流程：检索 → 阈值 → 锚点 → 生成 → 引用回映射 → 出处 |

进一步阅读：[技术报告](docs/技术报告.md)、[阶段十一 RAG 实现说明](docs/阶段十一RAG实现说明.md)、[阶段九完成报告](docs/阶段九完成报告.md)、[交付物清单](docs/交付物清单.md)、[交接摘要](docs/交接摘要.md)。

## 数据与后续工作

上传图片、SQLite 数据、模型权重、公开古籍评测影像和本地 `output/` 交付文件按 `.gitignore` 保存在本机。命题单位样本到位后需按授权范围登记、标注与独立评测，赛后按约定删除或归还。

后续重点是核验公开医书页的实体和关系准确性，修复跨段主体归属，继续改进密集小字与夹注识别，并在命题单位样本到位后分列报告结果。
