# -*- coding: utf-8 -*-
"""
qa/kb.py —— 知识库（语料）加载与段落切分

职责：把 SQLite 里所有页面的文本块，按「页 → 阅读顺序」拼成**语义段落**，
供向量化与检索使用。古籍 OCR 出来的块平均只有十几个字（一行一列），
单块做向量检索召回质量很差，所以这里把相邻块合并成 ~160 字的段落，
检索命中段落后，再回落到段内**具体文本块**给出精确出处（page_id / line_id / bbox）。

设计要点：
  - 只读 SQLite，不写库；带轻量缓存，库变化（页数/块数/时间戳）时自动失效；
  - 段落保留 blocks 列表，因此任何引用都能回原文核对，满足可溯源要求；
  - 向量化用 `text_norm`（简体归一），提示词与引用用 `text`（原文），
    既提升与现代提问的语义匹配，又不破坏原文引用的真实性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app import store
from app.config import CONFIG
from app.schemas import Block

# 段落分隔符：这些行天然是一段的开头，遇到就断开
_SECTION_HINTS = ("卷", "篇", "章", "第一", "第二", "第三", "方", "治", "論", "论")


@dataclass
class Passage:
    """语义段落：检索与向量化的最小单位，内部仍指向可溯源的原文块。"""

    pid: str                      # 全局唯一：f"{page_id}#c{序号}"
    page_id: str
    book: str
    text: str                     # 原文（繁体/异体原样）
    embed_text: str               # 归一后的简体文本，仅用于向量化
    blocks: List[Block] = field(default_factory=list)

    @property
    def line_ids(self) -> List[str]:
        return [b.line_id for b in self.blocks]

    @property
    def is_demo(self) -> bool:
        return bool(getattr(self.blocks[0], "is_demo", True)) if self.blocks else True

    def cite_label(self) -> str:
        """人读的出处标签：书 / 页 / 行范围。"""
        lines = self.line_ids
        span = lines[0] if len(lines) <= 1 else f"{lines[0]}–{lines[-1]}"
        return f"{self.book} · {self.page_id} · {span}"

    def to_meta(self) -> dict:
        return {
            "pid": self.pid,
            "page_id": self.page_id,
            "book": self.book,
            "text": self.text,
            "blocks": [
                {"line_id": b.line_id, "bbox": list(b.bbox), "text": b.text,
                 "source_hash": b.source_hash, "block_ref": b.block_ref or ""}
                for b in self.blocks
            ],
        }


def book_of(page_id: str) -> str:
    """从 page_id 推书名，供出处展示。

    约定 `书名__页` 是古籍页；其余情况给出可读的分类名，
    避免前端出现「《P20261116193054》」这种机器 ID 当书名的尴尬。
    """
    pid = str(page_id or "")
    if not pid:
        return ""
    if "__" in pid:
        return pid.split("__", 1)[0] or pid
    if pid.upper().startswith("DEMO"):
        return "示例页"
    if pid.upper().startswith("SYN"):
        return "合成示例"
    if re.match(r"^P\d{8,}", pid):
        return "本地上传"
    return pid


def corpus_signature() -> str:
    """语料指纹：页数/块数/最新时间。变化即认为索引需要重建。"""
    try:
        pages = store.list_pages() or []
        latest = max((str(p.get("created_at") or "") for p in pages), default="")
        n_blocks = 0
        try:
            conn = store.connect()
            try:
                n_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0])
            finally:
                conn.close()
        except Exception:
            n_blocks = 0
        return f"{len(pages)}p-{n_blocks}b-{latest}"
    except Exception:
        return "0p-0b-"


_cache: Dict[str, object] = {"sig": None, "passages": None}


def _split_blocks(blocks: List[Block], target: int) -> List[List[Block]]:
    """把一页的块按阅读顺序切成若干组，每组约 target 字。"""
    groups: List[List[Block]] = []
    cur: List[Block] = []
    cur_len = 0
    for b in blocks:
        t = (b.text or "").strip()
        if not t:
            continue
        # 明显是小标题/书名行：单独成段，且断开当前段
        if cur and (b.kind in {"title", "heading"} or t.startswith(_SECTION_HINTS)):
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += len(t)
        if cur_len >= target:
            groups.append(cur)
            cur, cur_len = [], 0
    if cur:
        groups.append(cur)
    return groups


def build_passages(force: bool = False) -> List[Passage]:
    """加载全库语料并切成段落（带缓存）。任何异常都返回空列表，不抛。"""
    sig = corpus_signature()
    if not force and _cache.get("sig") == sig and _cache.get("passages") is not None:
        return _cache["passages"]  # type: ignore[return-value]
    try:
        target = max(40, int(CONFIG.rag.passage_chars))
        passages: List[Passage] = []
        for p in store.list_pages() or []:
            pid_page = str(p.get("page_id") or "")
            if not pid_page:
                continue
            if p.get("is_demo"):
                continue  # 示例页不进语义知识库：全库检索不该把演示页当"头号出处"
            blocks = store.get_blocks(pid_page) or []
            blocks = sorted(blocks, key=lambda b: (getattr(b, "reading_order", 0) or 0,
                                                   getattr(b, "line_id", "") or ""))
            for i, group in enumerate(_split_blocks(blocks, target)):
                text = "".join((b.text or "").strip() for b in group)
                embed_text = "".join((b.text_norm or b.text or "").strip() for b in group)
                if not text:
                    continue
                passages.append(Passage(
                    pid=f"{pid_page}#c{i:03d}", page_id=pid_page,
                    book=book_of(pid_page), text=text,
                    embed_text=embed_text or text, blocks=group,
                ))
        _cache["sig"], _cache["passages"] = sig, passages
        return passages
    except Exception:
        return []


def page_blocks(page_id: str) -> List[Block]:
    try:
        return store.get_blocks(page_id) or []
    except Exception:
        return []


def find_block(passages: List[Passage], pid: str, line_id: str) -> Optional[Block]:
    """在段落里精确定位某个块（用于把模型引用映射回 bbox）。"""
    for ps in passages:
        if ps.pid != pid:
            continue
        for b in ps.blocks:
            if b.line_id == line_id:
                return b
    return None


def library_stats() -> Dict[str, object]:
    """知识库规模统计，供 /api/rag/status 与前端展示。"""
    passages = build_passages()
    books: Dict[str, int] = {}
    pages = set()
    for ps in passages:
        books[ps.book] = books.get(ps.book, 0) + 1
        pages.add(ps.page_id)
    return {
        "pages": len(pages),
        "passages": len(passages),
        "blocks": sum(len(ps.blocks) for ps in passages),
        "books": books,
        "signature": corpus_signature(),
    }
