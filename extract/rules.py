# -*- coding: utf-8 -*-
"""
extract/rules.py —— ② 文本 → 结构化（词典 + 规则，先跑通）

保留旧工程 `matcher/extract.py` / `evaluate.py` 的分层思路：抽取与计分分离，
抽取只做确定性规则，不调用任何模型，结果可复现、可审计。

输出两类对象（字段定义见 app/schemas.py）：
  - Entity：药材 / 性味 / 归经 / 功效 / 主治 / 禁忌 / 方剂 / 剂量
  - Triple：候选三元组（去重与冲突处理交给 graph/build.py）

溯源约定：quote 一律取该块的**原文整行** `Block.text`，保证百分百能回原文命中；
          归一后的匹配词只放在 value / name，不改写 quote。

TODO（下一阶段）：
  - 跨行主语归属目前是启发式（「上文最近药材」+「方名后紧跟组成」），
    需由本地 LLM 抽取 + 人工标注共同校验；
  - 增补剂量单位换算、炮制法、配伍关系（君臣佐使）、反面表述（"不可用"）识别；
  - 词典与规则命中需输出 trace，便于 eval/audit.py 逐条审计。
"""

from __future__ import annotations

import re
import statistics
from typing import Iterable, List, Optional, Tuple

from app.schemas import Block, Entity, Triple

from .schema import (
    CONTRA_WORDS, DOSE_RE, EFFECT_WORDS, FORMULAS, HERBS,
    MERIDIAN_WORDS, NATURE_PATTERNS, ZHUZHI_RE,
)

_DOSE_R = re.compile(r"([一-龥]{1,6}?)\s*(" + DOSE_RE + r")")
_ZHUZHI_R = re.compile(ZHUZHI_RE)

# 表格表头：不参与实体抽取（否则「劑量」会被当成剂量值、「藥名」被当成药材）
TABLE_HEADERS = {"藥名", "药名", "劑量", "剂量", "炮製", "炮制", "主治", "功效"}


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


def _entity(name: str, etype: str, b: Block, value: str = "") -> Entity:
    """构造带完整溯源字段的实体。"""
    return Entity(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
        quote=b.text, source_hash=b.source_hash,
        name=name, type=etype, value=value or name,
        block_ref=b.block_ref, extractor="rules", is_demo=True,
    )


def _triple(head: str, relation: str, tail: str, b: Block) -> Triple:
    """构造带完整溯源字段的三元组。"""
    return Triple(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id,
        quote=b.text, source_hash=b.source_hash,
        head=head, relation=relation, tail=tail,
        evidence_refs=[b.block_ref or b.line_id], is_demo=True,
    )


def extract_entities(blocks: Iterable[Block]) -> List[Entity]:
    """规则抽取入口（桩：小词典 + 正则）。

    输入：Block 列表；输出：Entity 列表（去重、字段完整、不抛异常）。
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

            def add(name: str, etype: str, value: str = "") -> None:
                key = (b.page_id, b.line_id, etype, name, value)
                if key not in seen:
                    seen.add(key)
                    out.append(_entity(name, etype, b, value))

            # 禁忌行里的「海藻 / 菘菜」是忌口对象，不再当独立药材入图
            contra_ctx = "忌" in text

            for herb in sorted(HERBS, key=lambda s: (-len(s), s)):
                if herb in text and not (contra_ctx and herb in CONTRA_WORDS):
                    add(herb, "药材")
            for formula in sorted(FORMULAS, key=lambda s: (-len(s), s)):
                if formula in text:
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
            # 正文页才跑正则，且要求药名在词典内 —— 否则「一合」这类会被误抽成剂量。
            if b.page_id in dose_cols:
                # 表头「劑量」本身不是剂量值，排除
                if id(b) in dose_cols[b.page_id] and text not in TABLE_HEADERS:
                    add(text, "剂量", text)
            else:
                for herb_txt, dose_val in _DOSE_R.findall(text):
                    # 只认词典内的药名：否则「温服一合」「一合」这类会被误抽成剂量
                    if herb_txt.strip() in HERBS:
                        add(dose_val, "剂量", dose_val)
            # 功效：GT 每行只标一个主功效，取「最长 + 最先出现」（见 _pick_main）
            eff = _pick_main(text, EFFECT_WORDS)
            if eff:
                add(eff, "功效")
            m = _ZHUZHI_R.search(text)
            if m:
                add(m.group(1), "主治", m.group(1))
            for word in sorted(CONTRA_WORDS, key=lambda s: (-len(s), s)):
                if contra_ctx and word in text:
                    add(word, "禁忌")
            # 归经：古籍常见「归某经」，合成集 GT 只标笼统的「脏腑」
            for word in sorted(MERIDIAN_WORDS, key=lambda s: (-len(s), s)):
                if word in text:
                    add(word, "归经")
    except Exception:
        return out  # 桩不得抛异常，返回已抽到的部分
    return out


def candidate_triples(blocks: Iterable[Block],
                      entities: Optional[Iterable[Entity]] = None) -> List[Triple]:
    """从原文行生成候选三元组（规则桩）。

    归属策略（启发式，已在模块说明中标注为 TODO）：
      - 行内出现方名 → 记为「当前方」，其后含剂量的行视为该方组成；
      - 行内首个药材记为「当前主语」，跨行的主治 / 功效 / 禁忌归到它；
      - 药名紧邻剂量 → (药材, 剂量, 剂量值)。
    输入：Block 列表（entities 参数保留给后续 LLM 抽取结果复用）
    输出：Triple 列表，字段完整，不抛异常。
    """
    out: List[Triple] = []
    current_formula = ""
    current_herb = ""
    try:
        for b in blocks:
            if not isinstance(b, Block):
                continue
            text = b.text_norm or b.text
            if not text:
                continue

            formula_hit = next((f.rstrip("方") for f in sorted(FORMULAS, key=lambda s: (-len(s), s))
                                if f in text), "")
            if formula_hit:
                current_formula = formula_hit

            herbs = [h for h in sorted(HERBS, key=lambda s: (-len(s), s)) if h in text]
            if herbs and not text.startswith("忌"):
                current_herb = herbs[0]

            # 方剂组成 + 剂量
            for herb, dose in _DOSE_R.findall(text):
                herb = herb.strip(" ·，,")
                out.append(_triple(herb or "（未识别药名）", "剂量", dose.strip(), b))
                if current_formula:
                    out.append(_triple(current_formula, "组成", herb, b))

            # 性味
            for pat in NATURE_PATTERNS:
                m = re.search(pat, text)
                if m and current_herb:
                    out.append(_triple(current_herb, "性味", m.group(0), b))
                    break

            # 主治 / 功效
            m = _ZHUZHI_R.search(text)
            if m and current_herb:
                out.append(_triple(current_herb, "主治", m.group(1), b))
            eff = _pick_main(text, EFFECT_WORDS)
            if eff and current_herb:
                out.append(_triple(current_herb, "功效", eff, b))

            # 禁忌：优先取本行主语（避免跨行猜测），行内无主语才回退到上文主语
            if "忌" in text:
                subject = next((h for h in herbs if h not in CONTRA_WORDS), current_herb)
                for word in sorted(CONTRA_WORDS, key=lambda s: (-len(s), s)):
                    if word in text and subject:
                        out.append(_triple(subject, "禁忌", word, b))
    except Exception:
        return out
    return out


def extract_all(blocks: Iterable[Block]) -> Tuple[List[Entity], List[Triple]]:
    """便捷入口：一次拿到实体与候选三元组。"""
    items = list(blocks)
    return extract_entities(items), candidate_triples(items)
