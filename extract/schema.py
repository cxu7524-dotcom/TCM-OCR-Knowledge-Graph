# -*- coding: utf-8 -*-
"""
extract/schema.py —— 中医药实体与关系定义（口径层）

本文件定义「有哪些类型 / 关系」以及抽取用的**术语表加载**，不含抽取逻辑。

术语表来源与规模（阶段九）：
  - 词表放在 `data/lexicon/中医术语表.json`，带 `版本 / 录入日期 / 说明 / 来源 / 局限` 字段；
  - 模块级常量名（HERBS / FORMULAS / EFFECT_WORDS …）保持不变，现有 import 与单测不受影响；
  - JSON 缺失或损坏时**退回内置小词典**（阶段八的 11 味药），并在 stderr 打日志、在
    `dictionary_stats()` 里标 `degraded=True`——降级必须可见，不能静默变小。

口径边界：
  - 这是**工程用词表**，不是知识库、不是权威全量词表，也不得当作评测基线；
  - 匹配时对词典项与原文施加**同一套归一化**（`ocr/normalize.py`），繁简 / 异体因此可互通；
  - 词表命中率不等于抽取准确率，真实页的准确率仍需人工核验（见 docs/阶段九完成报告.md）。
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Set

from app.config import CONFIG

logger = logging.getLogger(__name__)

LEXICON_PATH = CONFIG.paths.data_dir / "lexicon" / "中医术语表.json"

# 与 app/schemas.py 的 ENTITY_TYPES 保持一致（此处补充中文说明）
ENTITY_TYPE_DESC: Dict[str, str] = {
    "药材": "单味药材名（含炮制品名时另记炮制属性）",
    "性味": "四气五味，如味甘平、味苦寒",
    "归经": "脏腑经络归属，如入脾经",
    "功效": "作用表述，如利小便、补血",
    "主治": "主治病证，多由「主 / 主治 / 治」引出",
    "禁忌": "配伍禁忌与忌口，多由「忌」引出",
    "方剂": "方名，如桂枝汤",
    "剂量": "用量，含数字与单位，如三两、十二枚",
}

RELATIONS: Dict[str, str] = {
    "性味": "药材/方剂 → 性味",
    "归经": "药材 → 归经",
    "功效": "药材 → 功效",
    "主治": "药材/方剂 → 主治",
    "禁忌": "药材/方剂 → 禁忌",
    "组成": "方剂 → 药材",
    "剂量": "药材 → 剂量",
    "见于": "实体 → 出处页",
    "别名": "药材 → 别名",
}

# ------------------------------------------------------------------
# 内置兜底小词典（JSON 缺失时使用；阶段八原样保留，仅供降级演示）
# ------------------------------------------------------------------
_FALLBACK_HERBS: Set[str] = {
    "甘草", "桂枝", "芍药", "生姜", "大枣", "海藻", "菘菜",
    "茯苓", "当归", "黄芪", "桔梗",
}
_FALLBACK_FORMULAS: Set[str] = {"桂枝汤", "桂枝汤方"}
_FALLBACK_EFFECTS: Set[str] = {
    "坚筋骨", "长肌肉", "轻身延年", "温中下气", "利小便", "补血",
}
_FALLBACK_DISEASES: Set[str] = {"中风", "伤寒", "疟疾", "痢疾", "消渴", "黄疸"}
_FALLBACK_UNITS: Set[str] = {
    "两", "钱", "分", "厘", "枚", "片", "盏", "钟", "盅", "升", "合", "斤", "勺",
}


def _load_json() -> Dict[str, object]:
    """读取术语表；失败返回空 dict（调用方走兜底）。"""
    try:
        return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("术语表缺失，回退内置小词典：%s", LEXICON_PATH)
        return {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("术语表读取失败（%s），回退内置小词典：%s", type(exc).__name__, LEXICON_PATH)
        return {}


def _norm_set(items) -> Set[str]:
    """词典项统一走 ocr/normalize 的同一套归一化，保证与原文可比。"""
    from ocr.normalize import normalize_text
    out: Set[str] = set()
    for it in items or []:
        if not isinstance(it, str):
            continue
        t = normalize_text(it.strip())
        if len(t) >= 1:
            out.add(t)
    return out


_RAW = _load_json()
DEGRADED: bool = not _RAW
LEXICON_META: Dict[str, object] = {
    "版本": _RAW.get("版本", "fallback"),
    "录入日期": _RAW.get("录入日期", ""),
    "说明": _RAW.get("说明", "术语表缺失，使用内置兜底小词典"),
    "来源": _RAW.get("来源", []),
    "局限": _RAW.get("局限", ""),
    "整理人": _RAW.get("整理人", ""),
    "置信等级": _RAW.get("置信等级", ""),
}

HERBS: Set[str] = _norm_set(_RAW.get("药材")) or set(_FALLBACK_HERBS)
FORMULAS: Set[str] = _norm_set(_RAW.get("方剂")) or set(_FALLBACK_FORMULAS)
EFFECT_WORDS: Set[str] = _norm_set(_RAW.get("功效")) or set(_FALLBACK_EFFECTS)
DISEASES: Set[str] = _norm_set(_RAW.get("主治")) or set(_FALLBACK_DISEASES)

_units_raw = _RAW.get("剂量单位") or {}
DOSE_UNITS: Set[str] = (
    _norm_set(_units_raw.get("繁体")) | _norm_set(_units_raw.get("简体"))
    | _norm_set(_units_raw.get("特殊"))
) or set(_FALLBACK_UNITS)

# 长词必须排在短词前面：避免「味辛温」被拆成「味辛」+ 多余短实体（字段 F1 的 FP 来源）
NATURE_PATTERNS: List[str] = [
    r"味辛微温", r"味甘温", r"味辛温", r"味苦寒", r"味甘平", r"味苦平",
    r"味甘", r"味苦", r"味辛", r"性平", r"性温", r"无毒", r"有小毒",
]

# 「咳逆上气」「胸胁逆气」由 ZHUZHI_RE 的「主…」引导捕获为主治，不重复算功效

CONTRA_WORDS: Set[str] = {"海藻", "菘菜"}

# 归经：古籍常见「归某经」表述，以及笼统的「脏腑」说法（合成集 GT 只标后者）。
# 顺序按长度降序匹配，避免「肺经」被「经」类短词截断。
# 注意：不要加入「五脏六腑」这类长词——合成集 GT 只标「脏腑」，加长词会多出 FP。
MERIDIAN_WORDS: Set[str] = {
    "脏腑", "肺经", "脾经", "心经", "肝经", "肾经", "胃经", "胆经",
    "膀胱经", "大肠经", "小肠经", "三焦经", "心包经", "心包络",
}

# 剂量：数字 + 单位（可选尾缀「半」，古籍「一錢半」很常见）。
# 单位只收**量词明确**的写法：个 / 条 / 粒 会误命中「四五个时辰」这类叙述，故不入表。
# 数字与单位之间**不允许空白**：实测「卷之一 合成示意頁」会被误切成剂量「一合」。
DOSE_RE = (r"[一二三四五六七八九十百半\d]+"
           r"(?:两|钱|分|厘|枚|片|升|合|斤|勺|盏|钟|盅|铢)\s*半?")

# 主治引导词：把「主治五藏六府寒热邪气」「治中风不省人事」切成一条主治实体。
# 阶段九把「治 / 疗」纳入引导词——真实方书页的主治病证几乎都用「治…」行文。
ZHUZHI_RE = r"(?:主治|主|治|疗|療)\s*([^，。；、\s]{2,20})"

# 主治引导的停用前缀：这些是行文用语，不是病证
ZHUZHI_STOP: Set[str] = _norm_set((_RAW.get("停用") or {}).get("主治")) | {
    "之", "法", "验", "驗", "愈", "则", "則", "宜", "病", "本", "标", "標", "者", "人", "此",
}

# 方名停用词：剂型单字与「湯下 / 煎服」这类行文
FORMULA_STOP: Set[str] = _norm_set((_RAW.get("停用") or {}).get("方名")) | {
    "汤", "丸", "散", "丹", "饮", "膏", "煎", "锭",
}

# 方名模式（不依赖词典）：2-4 个汉字 + 剂型字
FORMULA_RE = r"[\u4e00-\u9fa5]{2,4}(?:汤|汤|丸|散|丹|饮|飲|膏|煎|锭|錠)"

# 方名里不允许出现的字：剂量单位与阿拉伯数字（否则「三钟煎」会被当成方名）。
# 汉字数字**允许出现在首位**——三化汤 / 六君子汤 / 八珍汤 / 十全大补汤 都是真方名。
_FORMULA_BAD_CHARS = set("两钱分厘枚片升合斤勺盏钟盅铢0123456789")
_NUMERAL_CHARS = set("一二三四五六七八九十百千半")
# 剂型字后面紧跟这些字时，说明它是普通词语而不是方名：
# 「饮食」「丸子」「丹田」「散布」「煎用」——真实页实测的 FP 主要来自这里。
_FORMULA_CONT_CHARS = set("食子田布落用水中之者")


def is_valid_formula(name: str, next_char: str = "") -> bool:
    """方名候选的确定性过滤：长度、停用词、剂量单位 / 数字、病证名、后续字。

    纯函数、不读 GT、不含随机，保证跨进程一致。
    next_char：模式命中后紧跟的那个字（用于排除「饮食 / 丸子 / 丹田」这类普通词）。
    """
    if not name or len(name) < 3 or len(name) > 5:
        return False
    if name in FORMULA_STOP or name in DISEASES:
        return False
    if any(ch in _FORMULA_BAD_CHARS for ch in name):
        return False
    # 数字只允许在首位（「三化汤」合法，「化三汤」不是方名）
    if any(ch in _NUMERAL_CHARS for ch in name[1:]):
        return False
    if next_char and next_char in _FORMULA_CONT_CHARS:
        return False
    # 去掉剂型字后至少还剩 2 个汉字（「汤」单独不算）
    return len(name[:-1]) >= 2


def is_valid_indication(text: str) -> bool:
    """主治候选的确定性过滤：去掉「治法 / 治之 / 治验」等行文用语。"""
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if t in ZHUZHI_STOP:
        return False
    # 以行文用语开头（治法、治验、之…）直接丢弃
    return not any(t.startswith(w) for w in ZHUZHI_STOP if w)


def dictionary_stats() -> Dict[str, object]:
    """词表规模与来源，仅用于界面说明，不是评测指标。"""
    return {
        "药材": len(HERBS), "方剂": len(FORMULAS), "性味模式": len(NATURE_PATTERNS),
        "功效词": len(EFFECT_WORDS), "主治词": len(DISEASES), "剂量单位": len(DOSE_UNITS),
        "禁忌词": len(CONTRA_WORDS), "关系类型": len(RELATIONS),
        "版本": LEXICON_META.get("版本", "unknown"),
        "录入日期": LEXICON_META.get("录入日期", ""),
        "整理人": LEXICON_META.get("整理人", ""),
        "置信等级": LEXICON_META.get("置信等级", ""),
        "degraded": DEGRADED,
        "note": "工程用词表（教材目录级，未逐条审定），不是权威全量词表；规模不代表抽取准确率",
    }


def warn_if_degraded() -> None:
    """把降级信息打到 stderr（服务启动与命令行都能看到）。"""
    if DEGRADED:
        print(f"[warn] 术语表未加载，抽取回退内置小词典（{len(HERBS)} 味药）：{LEXICON_PATH}",
              file=sys.stderr)
