# -*- coding: utf-8 -*-
"""tools/reingest_eval59.py —— 用真实 OCR 重建 59 页评测子集（替换数据集转写版）。

背景：知识库此前用 ModelScope label.csv 的机器转写文本建库（engine=dataset-transcription），
每块 bbox 是按阅读顺序推算的行框，不是真实检测框。59 页评测原图已通过
tools/download_eval_pages.py 下载到 古籍评测集/医家类/，本脚本把这些页走一遍与
/api/pages 上传完全相同的 OCR 链路后重建，使这 59 页拥有真实 bbox / 置信度 / 阅读顺序。

要点：
  1. page_id 沿用转写版命名（书名__p页码），先 store.delete_page 再 save_page，
     避免 INSERT OR IGNORE 留下双份 blocks；
  2. 原图复制到 data/uploads/{page_id}.png，演示台步骤二可直接预览原图；
  3. run_ocr 全链路与生产一致（低置信过滤、竖排逐列、表格还原、字形纠错）；
  4. 结束后重建语义向量索引（qa.index.rebuild_now）。

用法（项目根目录）：
    python tools/reingest_eval59.py --dry-run   # 只核对 page_id 对得上，不删不跑
    python tools/reingest_eval59.py             # 全量重建 59 页
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store                                               # noqa: E402
from app.config import CONFIG                                       # noqa: E402
from extract.rules import extract_all_audited                       # noqa: E402
from extract.formula_segments import extract_formula_candidates     # noqa: E402
from extract.formula_rows import extract_formula_rows               # noqa: E402
from graph.build import build_graph, build_triples                  # noqa: E402
from ocr.pipeline import run_ocr                                    # noqa: E402
from ocr.recognize import get_adapter, rec_model_name               # noqa: E402

LIST_PATH = ROOT / "古籍评测集" / "医家类下载清单.csv"
IMG_DIR = ROOT / "古籍评测集" / "医家类"
UPLOADS = ROOT / "data" / "uploads"


def page_id_of(name: str) -> str:
    """'丹方彚编page_12.png' → '丹方彚编__p12'（与 ingest_corpus 的命名一致）。"""
    stem = Path(name).stem                       # 丹方彚编page_12
    if "page_" not in stem:
        raise ValueError(f"文件名不含 page_：{name}")
    book, page = stem.split("page_", 1)
    return f"{book}__p{page}"


def main() -> int:
    ap = argparse.ArgumentParser(description="真实 OCR 重建 59 页评测子集")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not LIST_PATH.exists():
        print(f"清单不存在：{LIST_PATH}", file=sys.stderr)
        return 2
    rows = [r for r in csv.DictReader(LIST_PATH.open(encoding="utf-8-sig"))
            if (r.get("文件名(必须原样保存)") or "").strip()]
    print(f"目标 {len(rows)} 页")

    # 第一步：核对 page_id 与库中转写页一一对应，防止命名错位造成孤儿页
    missing, wrong_engine = [], []
    for r in rows:
        pid = page_id_of(r["文件名(必须原样保存)"])
        img = IMG_DIR / r["文件名(必须原样保存)"]
        if not img.exists():
            missing.append(img.name)
            continue
        page = store.get_page(pid)
        inner = (page or {}).get("page") or {}
        if page is None:
            missing.append(f"库中无 {pid}")
        elif inner.get("engine") != "dataset-transcription":
            wrong_engine.append(f"{pid} engine={inner.get('engine')}")
    if missing or wrong_engine:
        for m in missing:
            print(f"  [!] {m}", file=sys.stderr)
        for w in wrong_engine:
            print(f"  [!] 非转写页：{w}", file=sys.stderr)
        if missing:
            print("page_id 对不上，中止（不删任何数据）", file=sys.stderr)
            return 2

    if args.dry_run:
        print("[dry-run] 59 个 page_id 全部对上，图片齐全。实际执行请去掉 --dry-run")
        return 0

    adapter, notes = get_adapter()
    print(f"OCR 引擎就绪：{rec_model_name() or adapter}")
    UPLOADS.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    n_block = n_ent = n_tri = 0
    for i, r in enumerate(rows, 1):
        name = r["文件名(必须原样保存)"].strip()
        pid = page_id_of(name)
        src = IMG_DIR / name
        dest = UPLOADS / f"{pid}{src.suffix.lower()}"

        store.delete_page(pid)                   # 删旧页（blocks/entities/triples 一并删）
        shutil.copyfile(src, dest)
        t0 = time.time()
        ocr = run_ocr(str(dest), pid)
        # ↓↓↓ 与 app/main.py:_ingest 的入库流程逐行对齐 ↓↓↓
        # 幂等：重复入库前剔除上一轮附着的小字证据块
        ocr.blocks = [b for b in ocr.blocks if b.kind != 'small_column_evidence']
        entities, candidates, unresolved = extract_all_audited(ocr.blocks, is_demo=False)
        triples = build_triples(entities, ocr.blocks, candidates)
        build_graph(triples, entities)           # 校验可构建（demo台按页现查 store）
        # 正文方剂表：候选分段 → 行抽取（无条件，随页入库）
        segmented = extract_formula_candidates(ocr.blocks)
        formula_extraction = extract_formula_rows(ocr.blocks, segmented['candidates'])
        formula_extraction['continuation_hints'] = segmented['continuation_hints']
        # 双列小字复识：实验性开关 TCM_OCR_SMALL_COLUMNS=1 才启用（默认关，
        # 与生产行为一致；失败只降级不阻断入库）
        if CONFIG.ocr.small_columns_enabled:
            try:
                from ocr.small_columns_attach import attach_small_columns
                evidence, formula_extraction, small_audit = attach_small_columns(
                    ocr, formula_extraction, server_fallback=CONFIG.ocr.small_columns_server_enabled)
                ocr.blocks.extend(evidence)
                formula_extraction['small_columns_audit'] = small_audit
            except Exception as exc:
                formula_extraction['small_columns_audit'] = {'enabled': True, 'fallback': True,
                                                             'reason': type(exc).__name__}
        formula_extraction['_printed_tables'] = [t.to_dict() for t in ocr.tables]
        store.save_page(ocr, entities, triples, unresolved=unresolved,
                        formula_extraction=formula_extraction)
        n_block += len(ocr.blocks)
        n_ent += len(entities)
        n_tri += len(triples)
        n_frows = sum(len(f.get('rows') or []) for f in (formula_extraction.get('formulas') or []))
        print(f"  [{i}/{len(rows)}] {pid}  块 {len(ocr.blocks)}  实体 {len(entities)}"
              f"  关系 {len(triples)}  方剂 {len(formula_extraction.get('formulas') or [])}/行 {n_frows}"
              f"  待核 {len(unresolved)}  {time.time() - t0:.1f}s")

    print(f"\n59 页重建完成，总耗时 {time.time() - t_all:.0f}s："
          f"块 {n_block} / 实体 {n_ent} / 关系 {n_tri}")

    # 第二步：重建语义索引（passage 全量重切 + 向量重编码）
    from qa.index import rebuild_now
    st = rebuild_now(force=True)
    print(f"语义索引已重建：{st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
