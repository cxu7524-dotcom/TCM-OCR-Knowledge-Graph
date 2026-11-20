# -*- coding: utf-8 -*-
"""app/asr_voice.py —— 本机离线语音识别（faster-whisper small / int8 / CPU）。

为什么要有它：
    浏览器自带的语音识别（SpeechRecognition）要把麦克风音频送到谷歌 / 微软的服务器，
    国内网络经常连不上，前端只会拿到一句 network error。这条通道把录音留在本机做识别，
    换掉那一次外网请求——断网也能用。

模型：模型权重/faster-whisper-small/（461MB，与研一项目同款 small 模型）
    单句 3~5 秒，CPU int8 一般 1~3 秒出结果；服务刚启动第一次约 10 秒。

语言档位（只保留实测准确的三档）：
    zh  普通话 · yue 粤语 · en English
    粤语不是换语言码，而是「zh + 粤语提示词」——这颗 small 模型上 language="yue"
    会输出空字符串还白跑 15~36 秒；挂中文模型反而全对，且自动出简体。
    英语必须走 en，挂中文档位会把英文硬转成中文胡说。
    档位与提示词的定档依据见研一项目 docs（语音-方言识别-1120.md）。

与 app/main.py 的关系：
    本模块只被 /api/asr 与 /api/asr/status 两个新增路由调用，不改动既有识别 / 抽取 /
    图谱 / 问答的任何逻辑；没装模型或没装依赖时接口返回 503，不影响其它功能。
"""
from __future__ import annotations

import difflib
import os
import re
import tempfile
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("TCM_ASR_MODEL_DIR") or (_ROOT / "模型权重" / "faster-whisper-small"))

# 中文转写里不该留英文那种词间空格；这几个字符之间才允许保留空格
_CJK = r"\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef"
_CJK_SPACE = re.compile(r"(?<=[%s])\s+(?=[%s])" % (_CJK, _CJK))
# whisper 爱输出半角标点（"金额,这两个"），中文里看着别扭，统一成全角
_HALF2FULL = str.maketrans({",": "，", "?": "？", "!": "！", ";": "；", ":": "："})
_PUNCT_SPACE = re.compile(r"\s*([，。？！；：、])\s*")

_lock = threading.Lock()
_model = None
_load_error = ""

# ==================== 语言档位 ====================
# 提示词纪律（实测，2026-11-20）：
#   方言档一律用最短提示。研一项目实测：给粤语塞「文档 / 表格 / 标题」这类领域词会把
#   「季度报告」带成「柜道报告」；本项目复测得粤语加中医词目前不影响，但仍按最保守的来。
#   普通话档相反——实测必须带领域词，才能把「甘草」从「干草」、把「季度报告」从
#   「嫉妒报告」拉回来。下面 P_ZH 是本机 5 句实测选出来的（4/5，唯一错的是「伤寒论」，
#   小模型对书名就是弱，交给用户改字 + 大模型按上下文纠正）。
P_ZH = "以下是中医古籍问答的普通话提问，可能提到甘草、麻黄、方剂、药材、功效、禁忌、文档、表格、标题等词。"
P_YUE = "以下是粤语口语。"

LANGS = {
    "zh": {"whisper": "zh", "prompt": P_ZH, "label": "普通话", "browser": "zh-CN", "group": "zh"},
    "yue": {"whisper": "zh", "prompt": P_YUE, "label": "粤语", "browser": "zh-HK", "group": "zh"},
    "en": {"whisper": "en", "prompt": None, "label": "English", "browser": "en-US", "group": "other"},
}
DEFAULT_LANG = "zh"


def langs() -> list:
    """给前端渲染语言下拉用（顺序即展示顺序）。"""
    return [{"value": k, "label": v["label"], "browser": v["browser"], "group": v["group"]}
            for k, v in LANGS.items()]


def normalize_lang(lang) -> str:
    """前端传什么都兜住，认不出就按普通话。"""
    key = str(lang or "").strip().lower()
    return key if key in LANGS else DEFAULT_LANG


def _install_pyav_compat() -> None:
    """PyAV >= 19 删掉了 faster-whisper 仍在传的 metadata_errors，不补这层直接 500。

    只在解码那一下临时替换 av.open，解完立刻还原，不动全局状态，也不升降级环境里的包。
    """
    try:
        import av
        import faster_whisper.transcribe as _fwt
        from faster_whisper.audio import decode_audio as _orig
    except Exception:
        return
    if getattr(_orig, "_tcm_compat", False):
        return

    def _major():
        try:
            return int(str(av.__version__).split(".")[0])
        except Exception:
            return 0

    def decode_audio(input_file, sampling_rate=16000, split_stereo=False):
        open0 = av.open

        def _open(*a, **kw):
            if _major() >= 19:
                kw.pop("metadata_errors", None)
            return open0(*a, **kw)

        av.open = _open
        try:
            return _orig(input_file, sampling_rate=sampling_rate, split_stereo=split_stereo)
        finally:
            av.open = open0

    decode_audio._tcm_compat = True
    _fwt.decode_audio = decode_audio


def ready() -> bool:
    """模型文件是否齐（不加载模型，秒回）。"""
    return (MODEL_DIR / "model.bin").is_file() and (MODEL_DIR / "tokenizer.json").is_file()


def info() -> dict:
    """给 /api/asr/status 用的自检信息。"""
    files = {}
    for n in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        p = MODEL_DIR / n
        files[n] = round(p.stat().st_size / 1048576, 1) if p.is_file() else None
    try:
        import faster_whisper  # noqa: F401
        dep = True
    except Exception:
        dep = False
    return {
        "model_dir": str(MODEL_DIR),
        "file_mb": files,
        "deps": dep,
        "ready": ready() and dep,
        "loaded": _model is not None,
        "error": _load_error,
        "langs": langs(),
    }


def _get_model():
    """第一次调用时加载模型（约 2~3 秒，冷盘更久），之后复用同一个实例。"""
    global _model, _load_error
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if not ready():
            _load_error = "离线语音模型缺失：模型权重/faster-whisper-small（需 model.bin 与 tokenizer.json）"
            raise RuntimeError(_load_error)
        try:
            from faster_whisper import WhisperModel
            threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            _model = WhisperModel(str(MODEL_DIR), device="cpu",
                                  compute_type="int8", cpu_threads=threads)
            _load_error = ""
        except Exception as e:                       # 依赖缺失 / 模型文件损坏
            _load_error = "离线语音引擎起不来：%s" % e
            raise RuntimeError(_load_error)
        return _model


def warm_up() -> None:
    """后台预加载模型：用户第一次点麦克风时就不用再等那几秒。失败不吵人。"""
    try:
        if ready():
            _install_pyav_compat()
            _get_model()
    except Exception:
        pass


# 音画空白或外语内容时，Whisper 会背出视频站字幕模板（训练数据残留）
_ECHO_BLACKLIST = re.compile(r"请不吝|点赞|订阅|打赏|明镜与点点|字幕由|Amara\.org|谢谢观看")


def _is_hallucination(text: str, prompt: str, max_overlap: int = 5) -> bool:
    """识别结果是不是「把提示词当内容念出来」的幻觉。

    Whisper 在音频语言与提示语言对不上时会拿 initial_prompt 续写（实测：英文音频选
    方言档，34 秒换来一句「请把这个文件改变成《闽南语口语》」）。宁可判空让用户重说，
    也不能把假话当识别结果塞进输入框。
    """
    if not text:
        return False
    if _ECHO_BLACKLIST.search(text):
        return True
    if not prompt:
        return False
    m = difflib.SequenceMatcher(None, prompt, text).find_longest_match(
        0, len(prompt), 0, len(text))
    return m.size >= max_overlap


def transcribe(audio_path: str | Path, lang: str = "zh") -> dict:
    """音频文件 → 文字。返回 {text, seconds, duration, lang, lang_label}。"""
    key = normalize_lang(lang)
    cfg = LANGS[key]
    _install_pyav_compat()
    model = _get_model()
    t0 = time.time()
    segments, meta = model.transcribe(
        str(audio_path),
        language=cfg["whisper"],
        beam_size=5,
        vad_filter=True,                              # 掐掉首尾静音，短句更准
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,             # 单句场景，别让它自己脑补上下文
        initial_prompt=cfg["prompt"],
    )
    parts = []
    for s in segments:
        parts.append(s.text)
    text = "".join(parts).strip()
    text = _CJK_SPACE.sub("", text)                   # 去掉中文之间的多余空格
    if len(re.findall(r"[\u4e00-\u9fa5]", text)) >= max(2, len(text) // 3):
        text = text.translate(_HALF2FULL)             # 中文句子里的半角标点 → 全角
    text = _PUNCT_SPACE.sub(r"\1", text)              # 中文标点两边不留空格
    text = re.sub(r"\s{2,}", " ", text).strip()
    if _is_hallucination(text, cfg["prompt"]):
        text = ""
    return {
        "text": text,
        "seconds": round(time.time() - t0, 2),
        "duration": round(float(getattr(meta, "duration", 0) or 0), 2),
        "lang": key,
        "lang_label": cfg["label"],
    }


def transcribe_bytes(data: bytes, filename: str = "voice.webm", lang: str = "zh") -> dict:
    """上传的字节流 → 文字。临时文件用完即删（录音不落地保存）。"""
    suffix = Path(filename or "").suffix.lower() or ".webm"
    fd, tmp = tempfile.mkstemp(prefix="tcm_asr_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return transcribe(tmp, lang)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
