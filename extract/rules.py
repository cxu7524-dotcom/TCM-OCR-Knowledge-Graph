# -*- coding: utf-8 -*-
"""
extract/rules.py —— ② 文本 → 结构化（词典 + 模式，零模型）

保留旧工程 `matcher/extract.py` / `evaluate.py` 的分层思路：抽取与计分分离，
抽取只做确定性规则，不调用任何模型，结果可复现、可审计。

输出两类对象（字段定义见 app/schemas.py）：
  - Entity：药材 / 性味 / 归经 / 功效 / 主治 / 禁忌 / 方剂 / 剂量
  - Triple：候选三元组（去重与冲突处理交给 graph/build.py）

溯源约定：quote 一律取该块的**原文整行** `Block.text`，保证百分百能回原文命中；
          归一后的匹配词只放在 value / name，不改写 quote。

阶段九改造（真实古籍页链路修复）：
  1. 词表从 `data/lexicon/中医术语表.json` 加载（见 extract/schema.py），不再硬编码 11 味药；
  2. 新增**不依赖词典**的模式抽取：方名模式、药名 + 剂量配对、扩展的主治引导词；
  3. 新增**错字容忍**：仅对「本行一个实体都没抽到」且 OCR 置信度偏低的块，用编辑距离 ≤1
     去比词典项，命中记 extractor="rules_fuzzy"；
  4. is_demo 由调用方透传（示例页 True / 真实页 False），不再写死。

确定性要求（tests/test_extract_determinism.py）：
  所有集合遍历都必须显式排序；命中按 (起点, -长度, 词) 排序后去重叠，跨进程可复现。

TODO（下一阶段）：
  - 跨行主语归属仍是启发式（「上文最近药材」+「方名后紧跟组成」），需由本地 LLM 抽取 +
    人工标注共同校验；
  - 增补剂量单位换算、炮制法、配伍关系（君臣佐使）、反面表述（"不可用"）识别。
"""

from __future__ import annotations

import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import CONFIG
from app.schemas import Block, Entity, Triple

from .schema import (
    CONTRA_WORDS, DISEASES, DOSE_RE, DOSE_UNITS, EFFECT_WORDS, FORMULA_RE, FORMULAS,
    HERBS, MERIDIAN_WORDS, NATURE_PATTERNS, ZHUZHI_RE,
    is_valid_formula, is_valid_indication,
)

# 旧口径：药名必须紧邻剂量且**命中词典**（TCM_EXTRACT_DOSE_STRICT=1 时启用，便于 A/B）
_DOSE_R = re.compile(r"([一-龥]{1,6}?)\s*(" + DOSE_RE + r")")
# 新口径：任意 1-6 个汉字紧邻剂量，药名段再回查词典；查不到时按尾部汉字兜底
_DOSE_TAIL_R = re.compile(r"([一-龥]{1,6})\s*(" + DOSE_RE + r")")
_ZHUZHI_R = re.compile(ZHUZHI_RE)
_FORMULA_R = re.compile(FORMULA_RE)

# 表格表头：不参与实体抽取（否则「劑量」会被当成剂量值、「藥名」被当成药材）
TABLE_HEADERS = {"藥名", "药名", "劑量", "剂量", "炮製", "炮制", "主治", "功效"}

# 剂量兜底时要去掉的尾部行文用字（「半夏各一錢」的「各」、「共三錢」的「共」）
_TAIL_STOP = set("各共右俱為为入用服下每又加再後后先")
# 药名兜底时要去掉的头部行文用字（「加煉蜜二兩」的「加」）
_HEAD_STOP = set("加共求用入又再每右和調调煮煎為为將将")
# 药名兜底时不允许出现的字（数字与剂量单位）
_NUM_UNITS = set("一二三四五六七八九十百千半0123456789") | DOSE_UNITS
# 药名兜底时不允许出现的虚词 / 版式字：否则「卷之一」会被当成药名、进而把「一合」算成剂量
_NAME_STOP_CHARS = set("之第頁页卷共各右左上下中前后後者也不无無甚每其此")


def _index(terms: Iterable[str]) -> Dict[str, List[str]]:
    """首字索引：只比对同首字的词，避免逐词 in 判断的 O(词表 × 块数) 开销。"""
    idx: Dict[str, List[str]] = {}
    for t in terms:
        if t:
            idx.setdefault(t[0], []).append(t)
    for key in idx:
        idx[key].sort(key=lambda s: (-len(s), s))
    return idx


_HERB_IDX = _index(HERBS)
_FORMULA_IDX = _index(FORMULAS)
_EFFECT_IDX = _index(EFFECT_WORDS)
_MERIDIAN_IDX = _index(MERIDIAN_WORDS)

# 错字容忍用：按词长分桶（只在低置信、且本行零命中的块上跑）
_FUZZY_BY_LEN: Dict[int, List[str]] = {}
for _t in sorted(HERBS | FORMULAS, key=lambda s: (len(s), s)):
    _FUZZY_BY_LEN.setdefault(len(_t), []).append(_t)


def _scan(text: str, idx: Dict[str, List[str]]) -> List[Tuple[int, str]]:
    """词表命中，返回 [(起点, 词)]，**去重叠且长词优先**，顺序完全确定。

    去重叠的意义：`桂枝` 不再从 `桂枝汤` 里被重复算成一个独立药材；
    同一位置多个候选时取最长的那个（`甘草梢` 优先于 `甘草`）。
    """
    if not text:
        return []
    raw: List[Tuple[int, str]] = []
    for i, ch in enumerate(text):
        for term in idx.get(ch, ()):
            if text.startswith(term, i):
                raw.append((i, term))
    raw.sort(key=lambda p: (p[0], -len(p[1]), p[1]))
    out: List[Tuple[int, str]] = []
    last_end = -1
    for pos, term in raw:
        if pos < last_end:
            continue
        out.append((pos, term))
        last_end = pos + len(term)
    return out


def _table_dose_blocks(blocks: List[Block]) -> Optional[set]:
    """判定「三列表格页」并返回其剂量列的块 id 集合；不是表格页返回 None。

    为什么需要：表格页的药名/剂量/炮制是**分开的检测框**，
    `_DOSE_R` 要求「药名紧邻剂量」连写，分开的单元格根本匹配不到；
    而正文里的「一合」又会被误抽成剂量。实测合成表格页因此产生
    9 个剂量 FP + 6 个剂量 FN。

    判据（纯几何，不读 GT）：按 x 中心聚类后恰好 3 列、每列 ≥3 块，
    取 x 升序的中间列为剂量列。列间距阈值取框宽中位数的 0.8 倍。
    """
    # 先剔除页眉/页脚（宽高比 ≥4 的长条），否则页眉的 x 中心会落进剂量列、
    # 把列聚类污染成 4 列，导致整个判据失效。
    items = []
    for b in blocks:
        if not isinstance(b, Block) or not b.text:
            continue
        w = max(1.0, b.bbox[2] - b.bbox[0])
        h = max(1.0, b.bbox[3] - b.bbox[1])
        if w / h >= 4.0:
            continue
        items.append(b)
    if len(items) < 9:
        return None
    widths = [max(1.0, b.bbox[2] - b.bbox[0]) for b in items]
    gap = max(40.0, statistics.median(widths) * 0.8)
    xs = sorted((b.bbox[0] + b.bbox[2]) / 2 for b in items)
    groups: List[List[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    if len(groups) != 3 or min(len(g) for g in groups) < 3:
        return None
    centers = [sum(g) / len(g) for g in groups]
    mid = centers[1]
    tol = gap / 2
    return {id(b) for b in items
            if abs((b.bbox[0] + b.bbox[2]) / 2 - mid) <= tol}


def _pick_main(text: str, words: Iterable[str]) -> str:
    """在多个命中的词典词里挑「主」词，结果**完全确定**。

    阶段五 P0 根因修复：原先用 `max(hits, key=len)`，等长命中时由 set 迭代顺序决定，
    而 Python 的字符串哈希逐进程随机（PYTHONHASHSEED），同一份 OCR 输出会抽出不同实体，
    字段 F1 在 0.5882 / 0.6471 之间随机跳动（与逐列切分无关，OCR 输出逐字节一致）。
    现改为显式排序：**最长优先 → 文本中最先出现 → 字典序兜底**。
    选「最先出现」的依据是古籍行文主次（先述者为主功效），不依赖任何 GT。
    """
    best = ""
    best_key = None
    for w in set(words):
        pos = text.find(w)
        if pos < 0:
            continue
        key = (-len(w), pos, w)
        if best_key is None or key < best_key:
            best, best_key = w, key
    return best


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """等长或差 1 的两个串是否编辑距离 ≤1（早退实现，只用于错字容忍）。"""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return diff == 1
    # 长度差 1：短串是否为长串删一字所得
    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def _fuzzy_terms(text: str, max_len: int = 4) -> List[str]:
    """把「整块文本」当成一个候选词，找编辑距离 ≤1 的词典项（顺序确定）。

    为什么只比整块、不滑窗：500+ 味药的词表上做 2 字滑窗，任意两个字都能找到
    距离 ≤1 的药材（「胎前」→ 白前/前胡、「小见」→ 小蓟），误报率极高。
    错字容忍真正要救的是**短单元格**（如「白朮」被识别成「白木」），
    因此限定：块文本长度 2–8、整块参与比对、首字相同（OCR 极少把首字认错到无法对齐）。
    """
    t = (text or "").strip()
    if not (2 <= len(t) <= 8):
        return []
    terms = _FUZZY_BY_LEN.get(len(t)) or []
    out = [term for term in terms
           if term[0] == t[0] and term != t and _edit_distance_at_most_one(t, term)]
    return sorted(out)


def _resolve_herb(prefix: str) -> Tuple[str, str]:
    """把「剂量前的汉字段」解析成药名，返回 (药名, 抽取器)。

    优先词典：取**最靠右**（最贴近剂量）的命中，同尾位置取更长者——
    「川大黄洛三錢」→ 命中 `川大黄`，而不是 `大黄` 或整段乱码。
    词典没命中时按尾部汉字兜底（标记 rules_dose_ctx，便于人工复核时区分）。
    """
    hits = _scan(prefix, _HERB_IDX)
    if hits:
        best = max(hits, key=lambda p: (p[0] + len(p[1]), len(p[1]), p[1]))
        return best[1], "rules"
    tail = prefix
    while tail and tail[-1] in _TAIL_STOP:
        tail = tail[:-1]
    while tail and tail[0] in _HEAD_STOP:
        tail = tail[1:]
    if len(tail) < 2:
        return "", ""
    cand = tail[-3:] if len(tail) >= 3 else tail
    if len(cand) < 2 or len(cand) > 4:
        return "", ""
    if any(ch in _NUM_UNITS for ch in cand):
        return "", ""
    if any(ch in _NAME_STOP_CHARS for ch in cand):
        return "", ""
    if cand in DISEASES or cand in EFFECT_WORDS or cand in FORMULAS:
        return "", ""
    return cand, "rules_dose_ctx"


def _pair_herb_dose(text: str) -> List[Tuple[str, str, str]]:
    """药名 + 剂量配对（不要求药名在词典内），返回 [(药名, 剂量, 抽取器)]。"""
    out: List[Tuple[str, str, str]] = []
    for m in _DOSE_TAIL_R.finditer(text):
        herb, extractor = _resolve_herb(m.group(1))
        if not herb:
            continue
        out.append((herb, m.group(2).strip(), extractor))
    return out


def _entity(name: str, etype: str, b: Block, value: str = "",
            is_demo: bool = True, extractor: str = "rules") -> Entity:
    """构造带完整溯源字段的实体。"""
    return Entity(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
        quote=b.text, source_hash=b.source_hash,
        name=name, type=etype, value=value or name,
        block_ref=b.block_ref, extractor=extractor, is_demo=is_demo,
    )


def _triple(head: str, relation: str, tail: str, b: Block,
            is_demo: bool = True) -> Triple:
    """构造带完整溯源字段的三元组。"""
    return Triple(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
        quote=b.text, source_hash=b.source_hash,
        head=head, relation=relation, tail=tail,
        evidence_refs=[b.block_ref or b.line_id], is_demo=is_demo,
    )


# 方名模式的前置字白名单：句首 / 标点 / 数量词 / 「治用服」这类方名引导字。
# 没有这条限制时，句尾片段会被误判成方名（真实页实测：「…四肢無力飲」→ 方名「四肢无力饮」）。
_FORMULA_PREV_OK = set("治療疗用服以與与加入下送吞和調调配方丸散丹湯汤飲饮膏")
_FORMULA_PREV_OK |= set("一二三四五六七八九十百0123456789")
_FORMULA_PREV_PUNCT = set("，。、；：,.;:（）()《》「」『』 \t\n")


def _formula_hits(text: str) -> List[Tuple[int, str]]:
    """方名命中：词典优先，模式补漏，两者按区间去重（长词 / 词典词优先）。

    模式抽取是真实古籍页的关键——方书页的方名大多不在教材方剂目录里
    （如「三化湯」「回生丹」），只靠词典会整页归零。
    """
    dict_hits = _scan(text, _FORMULA_IDX)
    spans = [(pos, pos + len(term)) for pos, term in dict_hits]
    out = list(dict_hits)
    for m in _FORMULA_R.finditer(text):
        name = m.group(0)
        nxt = text[m.end()] if m.end() < len(text) else ""
        prev = text[m.start() - 1] if m.start() > 0 else ""
        if not is_valid_formula(name, nxt):
            continue
        if prev and prev not in _FORMULA_PREV_OK and prev not in _FORMULA_PREV_PUNCT:
            continue
        s, e = m.start(), m.end()
        if any(s < de and ds < e for ds, de in spans):   # 与词典命中重叠 → 跳过
            continue
        spans.append((s, e))
        out.append((s, name))
    out.sort(key=lambda p: (p[0], -len(p[1]), p[1]))
    return out


def extract_entities(blocks: Iterable[Block], is_demo: bool = True) -> List[Entity]:
    """规则抽取入口。

    输入：Block 列表；输出：Entity 列表（去重、字段完整、不抛异常）。
    is_demo：示例页 True / 真实页 False，由调用方（app/main.py:_ingest）透传。
    """
    out: List[Entity] = []
    blocks = list(blocks)
    # 预计算：每页若判定为三列表格，记录其剂量列的块 id（见 _table_dose_blocks）
    dose_cols: dict = {}
    _by_page: dict = {}
    for _b in blocks:
        if isinstance(_b, Block):
            _by_page.setdefault(_b.page_id, []).append(_b)
    for _pid, _bs in _by_page.items():
        _s = _table_dose_blocks(_bs)
        if _s:
            dose_cols[_pid] = _s

    seen = set()
    try:
        for b in blocks:
            if not isinstance(b, Block):
                continue
            text = b.text_norm or b.text
            if not text:
                continue

            def add(name: str, etype: str, value: str = "",
                    extractor: str = "rules") -> None:
                key = (b.page_id, b.line_id, etype, name, value)
                if key not in seen:
                    seen.add(key)
                    out.append(_entity(name, etype, b, value,
                                       is_demo=is_demo, extractor=extractor))

            # 禁忌行里的「海藻 / 菘菜」是忌口对象，不再当独立药材入图
            contra_ctx = "忌" in text

            herb_hits = _scan(text, _HERB_IDX)
            for _pos, herb in herb_hits:
                if not (contra_ctx and herb in CONTRA_WORDS):
                    add(herb, "药材")
            for _pos, formula in _formula_hits(text):
                add(formula.rstrip("方"), "方剂", formula)
            # 性味：去掉被更长命中包含的短词（如「味辛」被「味辛温」包含），
            # 但保留互不包含的并列词（如「无毒」），与 GT 的一行多实体口径一致。
            nature_hits = [m.group(0) for pat in NATURE_PATTERNS
                           for m in [re.search(pat, text)] if m]
            if nature_hits:
                longest = _pick_main(text, nature_hits)
                seen_nature = set()
                for h in sorted(set(nature_hits),
                                key=lambda s: (-len(s), text.find(s), s)):
                    if h not in longest or h == longest:
                        if h not in seen_nature:
                            seen_nature.add(h)
                            add(h, "性味")
            # 剂量：表格页按列位置直取（单元格是分开的检测框，正则匹配不到），
            # 正文页才跑正则。
            #   TCM_EXTRACT_DOSE_STRICT=1 → 回到阶段八行为：药名必须命中词典
            #   默认 0 → 放宽：药名段回查词典，查不到按尾部汉字兜底（标记 rules_dose_ctx）
            if b.page_id in dose_cols:
                # 表头「劑量」本身不是剂量值，排除
                if id(b) in dose_cols[b.page_id] and text not in TABLE_HEADERS:
                    add(text, "剂量", text)
            elif CONFIG.ocr.extract_dose_strict:
                for herb_txt, dose_val in _DOSE_R.findall(text):
                    # 只认词典内的药名：否则「温服一合」「一合」这类会被误抽成剂量
                    if herb_txt.strip() in HERBS:
                        add(dose_val, "剂量", dose_val)
            else:
                for herb_txt, dose_val, _ext in _pair_herb_dose(text):
                    add(dose_val, "剂量", dose_val)
                    if _ext == "rules_dose_ctx":
                        # 词典外药名：按需求保留为药材候选，但标记抽取器便于人工复核
                        add(herb_txt, "药材", extractor=_ext)
            # 功效：GT 每行只标一个主功效，取「最长 + 最先出现」（见 _pick_main）
            eff = _pick_main(text, EFFECT_WORDS)
            if eff:
                add(eff, "功效")
            m = _ZHUZHI_R.search(text)
            if m and is_valid_indication(m.group(1)):
                add(m.group(1), "主治", m.group(1))
            for word in sorted(CONTRA_WORDS, key=lambda s: (-len(s), s)):
                if contra_ctx and word in text:
                    add(word, "禁忌")
            # 归经：古籍常见「归某经」，合成集 GT 只标笼统的「脏腑」
            for _pos, word in _scan(text, _MERIDIAN_IDX):
                add(word, "归经")

            # 错字容忍：只在本行「一个药材 / 方剂都没抽到」且 OCR 置信度偏低时启用。
            # 命中记 extractor="rules_fuzzy"，quote 仍是原文整行，不做任何改写。
            # 表格单元格 / 表头 / 纯剂量串一律不参与：「三兩」→「三七/三棱」、「炮製」→「炮姜」
            # 都是实测出来的误报（单元格文本太短，任何两字都能对上某味药）。
            if (CONFIG.ocr.extract_fuzzy
                    and b.kind != "table_cell"
                    and text not in TABLE_HEADERS
                    and not re.fullmatch(DOSE_RE, text)
                    and not any(e.line_id == b.line_id and e.type in {"药材", "方剂"}
                                for e in out)
                    and float(b.conf or 0) < float(CONFIG.ocr.extract_fuzzy_max_conf)):
                for term in _fuzzy_terms(text)[:2]:
                    add(term, "方剂" if term in FORMULAS else "药材",
                        extractor="rules_fuzzy")
    except Exception:
        return out  # 桩不得抛异常，返回已抽到的部分
    return out


def candidate_triples(blocks: Iterable[Block],
                      entities: Optional[Iterable[Entity]] = None,
                      is_demo: bool = True) -> List[Triple]:
    """从原文行生成候选三元组（规则）。

    归属策略（启发式，已在模块说明中标注为 TODO）：
      - 行内出现方名 → 记为「当前方」，其后含剂量的行视为该方组成；
      - 行内首个药材记为「当前主语」，跨行的主治 / 功效 / 禁忌归到它；
        本行没有药材但有「当前方」时，主治 / 功效归到方剂（方书页的主治病证是方的）；
      - 药名紧邻剂量 → (药材, 剂量, 剂量值)，同时补 (方剂, 组成, 药材)。
    输入：Block 列表（entities 参数保留给后续 LLM 抽取结果复用）
    输出：Triple 列表，字段完整，不抛异常。
    """
    out: List[Triple] = []
    current_formula = ""
    current_herb = ""
    strict = CONFIG.ocr.extract_dose_strict
    try:
        for b in blocks:
            if not isinstance(b, Block):
                continue
            text = b.text_norm or b.text
            if not text:
                continue

            f_hits = _formula_hits(text)
            if f_hits:
                current_formula = f_hits[0][1].rstrip("方")

            herbs = [h for _p, h in _scan(text, _HERB_IDX)]
            if herbs and not text.startswith("忌"):
                current_herb = herbs[0]

            # 方剂组成 + 剂量
            pairs: List[Tuple[str, str]] = []
            if strict:
                for herb, dose in _DOSE_R.findall(text):
                    herb = herb.strip(" ·，,")
                    if herb:
                        pairs.append((herb, dose.strip()))
            else:
                pairs = [(h, d) for h, d, _e in _pair_herb_dose(text)]
            for herb, dose in pairs:
                out.append(_triple(herb or "（未识别药名）", "剂量", dose, b, is_demo))
                if current_formula and herb != current_formula:
                    out.append(_triple(current_formula, "组成", herb, b, is_demo))

            # 主语：本行药材 → 上文药材 → 当前方（方书页的主治属于方）
            subject = (herbs[0] if herbs else "") or current_herb or current_formula

            # 性味
            for pat in NATURE_PATTERNS:
                m = re.search(pat, text)
                if m and subject:
                    out.append(_triple(subject, "性味", m.group(0), b, is_demo))
                    break

            # 主治 / 功效
            m = _ZHUZHI_R.search(text)
            if m and subject and is_valid_indication(m.group(1)):
                out.append(_triple(subject, "主治", m.group(1), b, is_demo))
            eff = _pick_main(text, EFFECT_WORDS)
            if eff and subject:
                out.append(_triple(subject, "功效", eff, b, is_demo))

            # 禁忌：优先取本行主语（避免跨行猜测），行内无主语才回退到上文主语
            if "忌" in text:
                subj = next((h for h in herbs if h not in CONTRA_WORDS), current_herb)
                for word in sorted(CONTRA_WORDS, key=lambda s: (-len(s), s)):
                    if word in text and subj:
                        out.append(_triple(subj, "禁忌", word, b, is_demo))
    except Exception:
        return out
    return out


def extract_all(blocks: Iterable[Block],
                is_demo: bool = True) -> Tuple[List[Entity], List[Triple]]:
    """便捷入口：一次拿到实体与候选三元组（is_demo 默认 True，保持向后兼容）。"""
    items = list(blocks)
    return (extract_entities(items, is_demo=is_demo),
            candidate_triples(items, is_demo=is_demo))
