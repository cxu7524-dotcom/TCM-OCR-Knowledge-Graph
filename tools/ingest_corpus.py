# -*- coding: utf-8 -*-
"""
tools/ingest_corpus.py —— 把古籍转写语料导入知识库（供语义检索 / RAG 使用）

支持两种来源：

  1. `--source ancientdoc`：ModelScope 公开古籍评测集 `bytedance-community/AncientDoc`
     的 `label.csv`（含每页机器转写 `OCR` 列）。默认只取 `医家类`，
     共 176 页 / 10 部医书 / 约 5.7 万字；其中 59 页正是本项目评测用的三部书子集。

  2. `--source jsonl`：自定义语料，每行
     {"page_id": "...", "book": "...", "text": "整页原文"}

导入后每页会被切成若干文本块（按句读与长度），并跑一遍实体 / 三元组抽取，
因此知识图谱与问答证据都能直接引用这些真实古籍页。

⚠ 溯源诚实性说明：
   本工具导入的是**转写文本**，没有原图。因此每个块的 bbox 是按阅读顺序推算的
   行框（等分行高），**不是原图坐标**，`engine` 记为 `dataset-transcription`、
   `notes` 里写明来源。前端据此提示"无原图，bbox 为行序推算"，
   绝不冒充真实版面检测结果。

用法：
    python tools/ingest_corpus.py --source ancientdoc --dry-run
    python tools/ingest_corpus.py --source ancientdoc
    python tools/ingest_corpus.py --source ancientdoc --rebuild
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store                                              # noqa: E402
from app.schemas import Block, OCRResult, source_hash_of           # noqa: E402
from extract.rules import extract_all                              # noqa: E402
from graph.build import build_triples                              # noqa: E402

DEFAULT_CSV = ROOT / "data" / "annotation" / "official_label.csv"
LABEL_URL = ("https://modelscope.cn/api/v1/datasets/bytedance-community/AncientDoc/repo"
             "?Revision=master&FilePath=label.csv")
PAGE_W, PAGE_H = 1024, 1024          # 合成版面尺寸（仅用于推算行框）
LINE_TARGET = 26                     # 每块目标字数（接近一行竖排古籍）
_SENT_SPLIT = re.compile(r"(?<=[。！？；，、])")


def download_label(dest: Path) -> bool:
    """标注表不在本机时，从 ModelScope 公开数据集拉取（约 11MB）。"""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[i] 标注表不存在，正在从 ModelScope 下载…")
    try:
        req = urllib.request.Request(LABEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        print(f"[√] 已保存 {dest}（{dest.stat().st_size / 1024 / 1024:.1f} MB）")
        return True
    except Exception as exc:
        print(f"[×] 下载失败：{type(exc).__name__}: {exc}")
        return False


def _norm(text: str) -> str:
    """繁简 / 异体归一；失败时原样返回（不影响入库）。"""
    try:
        from ocr.normalize import normalize_text
        return normalize_text(text)
    except Exception:
        return text


def split_blocks(text: str) -> List[str]:
    """把整页转写切成"列"级文本块。

    两种页面结构分别处理：
      · 带换行的页（表格页 / 分栏页）：换行本身就是列边界，优先按行切，
        过长的行再按 LINE_TARGET 硬切，过短的行并入上一列，避免出现单字块；
      · 整页无换行的页（连续竖排正文）：按句读切，再把长段按 LINE_TARGET 硬切。
    这样每页能还原十几"列"，引用才能定位到具体行而不是整页。
    """
    raw = (text or "").replace("\r", "").strip()
    if not raw:
        return []

    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if len(lines) > 1:
        out: List[str] = []
        for ln in lines:
            if len(ln) <= LINE_TARGET * 2:
                out.append(ln)
            else:
                for i in range(0, len(ln), LINE_TARGET):
                    out.append(ln[i:i + LINE_TARGET])
        merged: List[str] = []
        for s in out:
            if merged and len(s) <= 3 and len(merged[-1]) < LINE_TARGET:
                merged[-1] += s          # 单字/双字行并入上一列
            else:
                merged.append(s)
        return [s for s in merged if s]

    parts = [p for p in _SENT_SPLIT.split(raw) if p and p.strip()]
    fixed: List[str] = []
    for p in parts:
        if len(p) <= LINE_TARGET * 2:
            fixed.append(p)
            continue
        for i in range(0, len(p), LINE_TARGET):
            fixed.append(p[i:i + LINE_TARGET])
    out = []
    buf = ""
    for p in fixed:
        buf += p
        if len(buf) >= LINE_TARGET:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        if out and len(buf.strip()) < 8:
            out[-1] += buf.strip()
        else:
            out.append(buf.strip())
    return [s for s in out if s]


def build_page(page_id: str, text: str) -> Tuple[OCRResult, str]:
    """由整页文本构造 OCRResult（含块与推算行框）。"""
    pieces = split_blocks(text)
    n = max(1, len(pieces))
    line_h = PAGE_H / n
    blocks: List[Block] = []
    for i, piece in enumerate(pieces):
        y1 = int(i * line_h)
        y2 = int(min(PAGE_H, (i + 1) * line_h))
        line_id = f"L{i:02d}"
        blocks.append(Block(
            page_id=page_id,
            bbox=[40, y1, PAGE_W - 40, max(y2, y1 + 8)],
            line_id=line_id,
            quote=piece,
            source_hash=source_hash_of(page_id, line_id, piece),
            text=piece,
            text_norm=_norm(piece),
            kind="title" if i == 0 and len(piece) <= 18 else "paragraph",
            conf=0.0,
            reading_order=i,
            col_id=i,
            block_ref="",
        ))
    ocr = OCRResult(
        page_id=page_id, image_path="", width=PAGE_W, height=PAGE_H,
        blocks=blocks, engine="dataset-transcription", model_loaded=False,
        is_demo=False,
        source_hash=source_hash_of(page_id, text[:64]),
        notes=["来源：AncientDoc(label.csv) 机器转写；无原图，bbox 为按行序推算，非版面检测结果"],
        created_at="",
    )
    return ocr, text


def rows_from_csv(path: Path, category: str) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, str]] = []
    for r in rows:
        if category and (r.get("type") or "").strip() != category:
            continue
        name = (r.get("name") or "").strip()
        text = (r.get("OCR") or "").strip()
        if not name or not text:
            continue
        book, page = (name.split("page_") + [""])[:2] if "page_" in name else (name, "")
        page_no = re.sub(r"\.(png|jpg|jpeg)$", "", page, flags=re.I)
        page_id = f"{book}__p{page_no}" if page_no else book
        out.append({"page_id": page_id, "book": book, "text": text})
    return out


def rows_from_jsonl(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(d.get("page_id") or "").strip()
            text = str(d.get("text") or d.get("ocr") or "").strip()
            if pid and text:
                out.append({"page_id": pid, "book": str(d.get("book") or pid), "text": text})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="导入古籍语料到知识库")
    ap.add_argument("--source", choices=["ancientdoc", "jsonl"], default="ancientdoc")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--category", default="医家类")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="导入后立即重建语义索引")
    ap.add_argument("--reset", action="store_true",
                    help="先删除本工具此前导入的页（engine=dataset-transcription）再重新导入")
    args = ap.parse_args()

    if args.source == "ancientdoc":
        path = Path(args.csv)
        if not path.exists() and not download_label(path):
            print(f"[×] 找不到标注表：{path}")
            return 2
        rows = rows_from_csv(path, args.category)
    else:
        path = Path(args.jsonl)
        if not path.exists():
            print(f"[×] 找不到语料：{path}")
            return 2
        rows = rows_from_jsonl(path)

    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        print("[×] 没有可导入的记录")
        return 1

    total_chars = sum(len(r["text"]) for r in rows)
    books = {}
    for r in rows:
        books[r["book"]] = books.get(r["book"], 0) + 1
    print(f"[√] 待导入 {len(rows)} 页 / {len(books)} 部书 / {total_chars} 字")
    for b, n in sorted(books.items(), key=lambda x: -x[1]):
        print(f"     {b}: {n} 页")

    if args.dry_run:
        print("[i] dry-run：未写入数据库")
        return 0

    store.init_db()

    if args.reset:
        removed = 0
        for p in store.list_pages() or []:
            if str(p.get("engine") or "") == "dataset-transcription":
                try:
                    store.delete_page(str(p.get("page_id")))
                    removed += 1
                except Exception:
                    pass
        print(f"[i] --reset：已删除旧导入页 {removed} 个")

    ok = skip = fail = 0
    n_blocks = n_entities = n_triples = 0
    for i, r in enumerate(rows, start=1):
        try:
            ocr, _ = build_page(r["page_id"], r["text"])
            entities, cands = extract_all(ocr.blocks, is_demo=False)
            triples = build_triples(entities, ocr.blocks, cands)
            stat = store.save_page(ocr, entities, triples)
            n_blocks += stat.get("blocks", 0)
            n_entities += stat.get("entities", 0)
            n_triples += stat.get("triples", 0)
            if stat.get("skipped") and stat.get("blocks", 0) == 0:
                skip += 1
            else:
                ok += 1
        except Exception as exc:
            fail += 1
            print(f"    [!] {r['page_id']} 导入失败：{type(exc).__name__}: {exc}")
        if i % 20 == 0 or i == len(rows):
            print(f"    进度 {i}/{len(rows)} …")

    print(f"[√] 完成：新增页 {ok}，已存在 {skip}，失败 {fail}")
    print(f"    新增文本块 {n_blocks}，实体 {n_entities}，三元组 {n_triples}")

    if args.rebuild:
        from qa.index import rebuild_now
        print("[i] 正在重建语义索引（首次会调用本机 embedding 模型）…")
        st = rebuild_now(force=True)
        print(f"[√] 索引：{st.get('vectors')} 段 / {st.get('dim')} 维 / {st.get('health')}")
    else:
        print("[i] 索引将在服务启动或下次上传时自动重建；也可运行 "
              "python tools/ingest_corpus.py --source ancientdoc --limit 1 --rebuild 强制重建")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
