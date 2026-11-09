# -*- coding: utf-8 -*-
"""
app/config.py —— 全项目唯一配置入口

设计约定：
  1. 所有路径、后端开关、阈值集中在本文件；其他模块一律 `from app.config import CONFIG`。
  2. 不写死绝对路径：一切以 PROJECT_ROOT（本文件上一级）为基准推导。
  3. 零密钥、零公网：默认不读取任何 API Key，不访问任何公网地址。
     LLM 只指向本机 127.0.0.1 的 OpenAI 兼容服务（Ollama / vLLM），且默认后端为 none。
  4. 环境变量覆盖：统一前缀 TCM_，便于不改动源码即可切换后端 / 端口。
     例如：TCM_LLM_BACKEND=ollama、TCM_OCR_BACKEND=rapidocr、TCM_PORT=8100。
  5. 兜底原则：声明的重型依赖缺失时自动降级为可运行的规则 + 桩，
     并在输出中显式标记 model_loaded=False / is_demo=True，绝不静默假装识别成功。

TODO（下一阶段）：
  - 接入真实 OCR 后把 OCRConfig.backend 固定为 rapidocr，并补齐版面后处理阈值。
  - 接入本地 LLM 后补充模型健康检查与超时重试策略。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# 项目根目录：app/config.py → 上一级即仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(key: str, default: str) -> str:
    """读取 TCM_ 前缀环境变量，空值视为未设置。"""
    return os.environ.get("TCM_" + key, default) or default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


# 识别模型自动探测候选（按优先级）。路径相对 PROJECT_ROOT，模型权重不进仓库。
REC_MODEL_CANDIDATES = ("模型权重/ch_PP-OCRv5_rec_mobile.onnx",)
# 包内自带的后备模型（rapidocr-onnxruntime 分发，简体 ch_PP-OCRv3_rec）
BUNDLED_REC_MODEL = "ch_PP-OCRv3_rec（rapidocr 包内简体）"


def _resolve_rec_model() -> tuple:
    """识别模型三级回退：环境变量 → 本地 v5 探测 → 包内 v3。

    返回 (模型路径, 来源标记)。模型路径为空表示用 rapidocr 包内简体 v3。
    只做路径探测，不加载模型、不下载任何权重。
    """
    env = (os.environ.get("TCM_OCR_REC_MODEL", "") or "").strip()
    if env:
        return env, "env"
    for rel in REC_MODEL_CANDIDATES:
        if (PROJECT_ROOT / rel).is_file():
            return rel, "auto"
    return "", "bundled"


_REC_MODEL_PATH, _REC_MODEL_SOURCE = _resolve_rec_model()


@dataclass(frozen=True)
class PathConfig:
    """目录与文件位置。所有目录在首次运行时由 app/store.py 幂等创建。"""

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    annotation_dir: Path = PROJECT_ROOT / "data" / "annotation"
    demo_dir: Path = PROJECT_ROOT / "data" / "demo"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    store_dir: Path = PROJECT_ROOT / "data" / "store"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"
    db_path: Path = PROJECT_ROOT / "data" / "store" / "tcm.sqlite3"
    web_dir: Path = PROJECT_ROOT / "app" / "web"
    eval_dir: Path = PROJECT_ROOT / "eval"

    def ensure_dirs(self) -> None:
        """幂等建目录，可反复调用。"""
        for p in (
            self.data_dir, self.samples_dir, self.annotation_dir, self.demo_dir,
            self.uploads_dir, self.store_dir, self.reports_dir, self.web_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OcrConfig:
    """OCR 配置。

    backend 可选：
      rapidocr  —— 默认。rapidocr-onnxruntime，纯 CPU 可跑，需可选依赖（见 requirements.txt）。
      paddleocr —— 可选替代实现，需自行安装对应包。
      stub      —— 不加载任何模型，返回带完整结构的示例结果（演示 / 无依赖环境）。
    auto_fallback=True 时，声明的后端依赖不可用会自动降级到 stub 并显式标注。
    """

    backend: str = _env("OCR_BACKEND", "rapidocr")
    lang: str = _env("OCR_LANG", "ch")
    # 低置信过滤阈值。早期实测：默认 0.50 会把表格页 score≈0.37~0.47 的
    # 真实单元格全部剔除，导致结构还原失真；调整为 0.35 并在报告中登记该差异。
    min_conf: float = _env_float("OCR_MIN_CONF", 0.35)
    # 检测阶段阈值（下调以提升方剂表小单元格召回，均为本地推理参数）
    det_box_thresh: float = _env_float("OCR_BOX_THRESH", 0.30)
    det_text_score: float = _env_float("OCR_TEXT_SCORE", 0.35)
    auto_fallback: bool = _env_bool("OCR_AUTO_FALLBACK", True)
    vertical_layout: bool = _env_bool("OCR_VERTICAL_LAYOUT", True)
    table_enabled: bool = _env_bool("OCR_TABLE_ENABLED", True)
    normalize_enabled: bool = _env_bool("OCR_NORMALIZE_ENABLED", True)
    # 字形纠错层：只做上下文可判定的纠错，可整体关闭（TCM_OCR_CORRECT=0）
    correct_enabled: bool = _env_bool("OCR_CORRECT", True)
    # 竖排逐列检测（P1）：竖排页先切列再逐列识别；=0 时完全回到整页识别，便于 A/B
    vertical_split: bool = _env_bool("OCR_VERTICAL_SPLIT", True)
    # 两路候选的取舍策略（P2）：
    #   auto     —— 默认，按 ocr/score.py 的复合无监督分择优
    #   vertical —— 强制逐列（无逐列结果时回退整页），用于 A/B 对照
    #   page     —— 强制整页，用于 A/B 对照
    pick: str = _env("OCR_PICK", "auto")
    # 择优余量：逐列分需超过整页分 × margin 才采用。
    # 实测（v5 / 34 页 + 医家类 20 页）：margin=1.02 时逐列一页都选不上（等价整页，零劣化）；
    # margin=1.00 时改判 3 页，验证集中位不变、均值略降，但医家类有一页从 0.302 劣化到 0.449。
    # 结论：当前复合分不足以承担改判，默认保留 2% 余量，宁可不用也不劣化。
    pick_margin: float = _env_float("OCR_PICK_MARGIN", 1.02)
    # 阶段七 P0：版面方向判据（全部只用「检测框几何 / 图像墨迹」无监督特征，不读 GT / CER）
    #   old —— 与阶段六完全一致：竖长条框占比 ≥0.30 且投影列数 ≥3
    #   new —— 追加两条对 det 碎片鲁棒的信号：图像层规则竖列、密集页的投影列规则性
    dir_rule: str = _env("OCR_DIR_RULE", "new")
    dir_min_img_cols: int = _env_int("OCR_DIR_MIN_IMG_COLS", 5)
    dir_min_img_aspect: float = _env_float("OCR_DIR_MIN_IMG_ASPECT", 3.0)
    dir_min_boxes: int = _env_int("OCR_DIR_MIN_BOXES", 20)
    dir_min_cols: int = _env_int("OCR_DIR_MIN_COLS", 8)
    dir_max_gap_cv: float = _env_float("OCR_DIR_MAX_GAP_CV", 0.5)
    col_min_w: int = _env_int("OCR_COL_MIN_W", 12)
    col_gap_ratio: float = _env_float("OCR_COL_GAP_RATIO", 0.02)
    rotate: int = _env_int("OCR_ROTATE", 90)     # 90=逆时针（实测有效），270=顺时针
    # 识别模型路径：三级回退（见 _resolve_rec_model）——
    #   ① TCM_OCR_REC_MODEL 显式指定；② 自动探测 模型权重/ch_PP-OCRv5_rec_mobile.onnx；
    #   ③ 都没有则用包内简体 ch_PP-OCRv3_rec（CER 明显更差，会在 notes / health 里标注）。
    rec_model: str = _REC_MODEL_PATH
    rec_model_source: str = _REC_MODEL_SOURCE
    demo_fallback: bool = _env_bool("OCR_DEMO_FALLBACK", True)

    # ---- 抽取层开关（阶段九新增；阈值集中在这里，不散落在规则里）----
    # 剂量抽取是否要求药名命中词典：0=放宽（默认，真实古籍页才抽得到），1=回到阶段八行为
    extract_dose_strict: bool = _env_bool("EXTRACT_DOSE_STRICT", False)
    # 错字容忍：与词典项编辑距离 ≤1 的片段在低置信块上尝试匹配（默认开）
    extract_fuzzy: bool = _env_bool("EXTRACT_FUZZY", True)
    # 错字容忍触发的置信度上限：只对「OCR 自己都不确定」的块做模糊匹配，避免误报
    extract_fuzzy_max_conf: float = _env_float("EXTRACT_FUZZY_MAX_CONF", 0.90)


@dataclass(frozen=True)
class LlmConfig:
    """本地大模型配置。

    backend 可选：
      none   —— 默认。不加载模型、不联网，用规则 + 桩完成端到端演示。
      ollama —— 本机 Ollama 的 OpenAI 兼容接口（默认 http://127.0.0.1:11434/v1）。
      vllm   —— 本机 vLLM 的 OpenAI 兼容接口（自行设置 TCM_LLM_BASE_URL）。
    切换示例：set TCM_LLM_BACKEND=ollama && set TCM_LLM_MODEL=qwen2.5:7b
    安全约束：base_url 只允许指向 127.0.0.1 / localhost，见 validate()。
    """

    backend: str = _env("LLM_BACKEND", "none")
    base_url: str = _env("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    model: str = _env("LLM_MODEL", "qwen2.5:7b")
    timeout: int = _env_int("LLM_TIMEOUT", 30)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 512)
    temperature: float = _env_float("LLM_TEMPERATURE", 0.1)

    @property
    def enabled(self) -> bool:
        return self.backend.lower() in {"ollama", "vllm"} and self.is_local()

    def is_local(self) -> bool:
        """仅允许本机地址，杜绝任何云端在线推理。"""
        url = self.base_url.lower()
        return url.startswith("http://127.0.0.1") or url.startswith("http://localhost")


@dataclass(frozen=True)
class VectorConfig:
    """本地向量索引配置。

    backend 可选：
      none —— 默认。仅走稀疏检索。
      hash —— 内置哈希向量（标准库实现，零依赖，仅用于打通链路，非语义向量）。
    TODO：接入本地 embedding 模型后替换为真实向量并记录模型名与版本。
    """

    backend: str = _env("VECTOR_BACKEND", "none")
    dim: int = _env_int("VECTOR_DIM", 128)
    enabled: bool = _env_bool("VECTOR_ENABLED", False)


@dataclass(frozen=True)
class QaConfig:
    """问答检索与拒答阈值。阈值是口径占位，未实测前不得当作评测结论。"""

    sparse_weight: float = _env_float("QA_SPARSE_WEIGHT", 1.0)
    vector_weight: float = _env_float("QA_VECTOR_WEIGHT", 0.0)
    top_k: int = _env_int("QA_TOP_K", 5)
    min_score: float = _env_float("QA_MIN_SCORE", 18.0)
    # 相对阈值：只保留 >= 最高分 × relative_keep 的证据，减少低相关噪声
    relative_keep: float = _env_float("QA_RELATIVE_KEEP", 0.6)
    min_evidence: int = _env_int("QA_MIN_EVIDENCE", 1)
    refuse_on_no_evidence: bool = _env_bool("QA_REFUSE_ON_NO_EVIDENCE", True)
    max_evidence: int = _env_int("QA_MAX_EVIDENCE", 3)


@dataclass(frozen=True)
class ServerConfig:
    host: str = _env("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)
    reload: bool = _env_bool("RELOAD", False)
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 20)


@dataclass(frozen=True)
class Config:
    paths: PathConfig = PathConfig()
    ocr: OcrConfig = OcrConfig()
    llm: LlmConfig = LlmConfig()
    vector: VectorConfig = VectorConfig()
    qa: QaConfig = QaConfig()
    server: ServerConfig = ServerConfig()
    demo_mode: bool = _env_bool("DEMO_MODE", True)
    disclaimer: str = "仅供参考、非诊疗建议"
    # 示例数据标注：只描述数据性质；OCR 默认真跑，标签由 app/main.py 按实际状态生成
    demo_badge: str = "示例数据 · 非命题单位样本"

    def summary(self) -> dict:
        """给 /api/health 与界面用的配置快照（不含任何密钥）。"""
        from pathlib import Path as _Path
        rec = (self.ocr.rec_model or "").strip()
        rec_name = _Path(rec).name if rec else BUNDLED_REC_MODEL
        return {
            "ocr_backend": self.ocr.backend,
            "ocr_auto_fallback": self.ocr.auto_fallback,
            "ocr_rec_model": rec_name,
            "ocr_rec_model_source": self.ocr.rec_model_source,
            "extract_dose_strict": self.ocr.extract_dose_strict,
            "extract_fuzzy": self.ocr.extract_fuzzy,
            "llm_backend": self.llm.backend,
            "llm_enabled": self.llm.enabled,
            "llm_base_url": self.llm.base_url if self.llm.enabled else "",
            "vector_backend": self.vector.backend,
            "demo_mode": self.demo_mode,
            "disclaimer": self.disclaimer,
            "demo_badge": self.demo_badge,
            "server": f"{self.server.host}:{self.server.port}",
            "root": str(self.paths.root),
        }

    def to_dict(self) -> dict:
        def _plain(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_plain(v) for v in value]
            return value

        return _plain(asdict(self))


# 全局单例：其他模块只读它，不各自解析环境变量
CONFIG = Config()
