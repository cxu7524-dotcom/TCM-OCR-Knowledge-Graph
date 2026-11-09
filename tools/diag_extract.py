# -*- coding: utf-8 -*-
"""
tools/diag_extract.py —— 真实古籍页「抽取产出量」诊断（阶段九新增）

回答一个具体问题：**真实古籍页上传后，②③④ 三步到底产出了多少东西？**
只统计「抽出来多少」，不评价「抽得准不准」——准确率需要人工核验，本脚本给不出来。

统计口径：
  实体 = extract/rules.py:extract_entities() 的输出条数（去重后）
  三元组 = graph/build.py:build_triples() 去重后的条数
  节点 = build_graph() 的节点数
  页均 = 总数 / 实际处理的页数（不剔除任何失败页，失败页计 0）

OCR 结果默认缓存到 data/reports/diag_ocr_cache/<页名>.json（OCR 慢且确定性，
缓存只影响跑批速度，不影响抽取结果）；--no-cache 强制重跑。

用法：
  python tools/diag_extract.py --dir 古籍评测集/医家类/思濟堂方書 --limit 10
  python tools/diag_extract.py --dir 古籍评测集/医家类 --limit 12 --no-cache
  python tools/diag_extract.py --dir 古籍评测集/医家类 --out data/reports/diag_extract.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import CONFIG                                    # noqa: E402
from app.schemas import Block, OCRResult                          # noqa: E402
from extract.rules import extract_all                             # noqa: E402
from graph.build import build_graph, build_triples                # noqa: E402
from ocr.pipeline import run_ocr                                  # noqa: E402

CACHE_DIR = CONFIG.paths.reports_dir / "diag_ocr_cache"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(folder: Path, limit: int) -> List[Path]:
    files = sorted((p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.name)
    return files[:limit] if limit > 0 else files


def _ocr_cached(image: Path, no_cache: bool) -> OCRResult:
    """带缓存的 OCR：缓存按「图片名 + 大小 + 修改时间」失效，避免串页。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stat = image.stat()
    key = f"{image.parent.name}__{image.stem}__{stat.st_size}_{int(stat.st_mtime)}.json"
    path = CACHE_DIR / key
    if not no_cache and path.exists():
        try:
            return OCRResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    ocr = run_ocr(str(image), f"DIAG-{image.parent.name}-{image.stem}")
    path.write_text(json.dumps(ocr.to_dict(), ensure_ascii=False), encoding="utf-8")
    return ocr


def diagnose(folder: Path, limit: int, no_cache: bool) -> Dict[str, Any]:
    images = _list_images(folder, limit)
    rows: List[Dict[str, Any]] = []
    for image in images:
        ocr = _ocr_cached(image, no_cache)
        blocks: List[Block] = list(ocr.blocks)
        entities, candidates = extract_all(blocks, is_demo=ocr.is_demo)
        triples = build_triples(entities, blocks, candidates)
        graph = build_graph(triples, entities)
        rows.append({
            "page": f"{image.parent.name}/{image.name}",
            "engine": ocr.engine, "model_loaded": ocr.model_loaded,
            "is_demo": ocr.is_demo, "blocks": len(blocks),
            "entities": len(entities), "triples": len(triples),
            "nodes": len(graph.nodes),
            "by_type": _count_types(entities),
            "by_extractor": _count_extractors(entities),
        })
    n = len(rows) or 1
    return {
        "dir": str(folder),
        "pages": len(rows),
        "rec_model": (CONFIG.ocr.rec_model or "ch_PP-OCRv3_rec（包内简体）"),
        "dose_strict": CONFIG.ocr.extract_dose_strict,
        "fuzzy": CONFIG.ocr.extract_fuzzy,
        "totals": {
            "blocks": sum(r["blocks"] for r in rows),
            "entities": sum(r["entities"] for r in rows),
            "triples": sum(r["triples"] for r in rows),
            "nodes": sum(r["nodes"] for r in rows),
        },
        "per_page": {
            "blocks": round(sum(r["blocks"] for r in rows) / n, 2),
            "entities": round(sum(r["entities"] for r in rows) / n, 2),
            "triples": round(sum(r["triples"] for r in rows) / n, 2),
            "nodes": round(sum(r["nodes"] for r in rows) / n, 2),
        },
        "rows": rows,
    }


def _count_types(entities) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entities:
        out[e.type] = out.get(e.type, 0) + 1
    return dict(sorted(out.items()))


def _count_extractors(entities) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entities:
        out[e.extractor] = out.get(e.extractor, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description="真实古籍页抽取产出量诊断")
    ap.add_argument("--dir", required=True, help="图片目录（可递归一层书名的上级目录）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少页（0=全部）")
    ap.add_argument("--no-cache", action="store_true", help="忽略 OCR 缓存，强制重跑")
    ap.add_argument("--out", default="", help="把完整结果写到该 JSON 路径")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_absolute():
        folder = _ROOT / folder
    if not folder.is_dir():
        print(f"目录不存在：{folder}", file=sys.stderr)
        return 2

    result = diagnose(folder, args.limit, args.no_cache)
    print(f"目录：{result['dir']}")
    print(f"页数：{result['pages']}　识别模型：{result['rec_model']}　"
          f"剂量严格模式：{result['dose_strict']}　错字容忍：{result['fuzzy']}")
    print(f"合计：文本块 {result['totals']['blocks']}　实体 {result['totals']['entities']}　"
          f"三元组 {result['totals']['triples']}　图谱节点 {result['totals']['nodes']}")
    print(f"页均：文本块 {result['per_page']['blocks']}　实体 {result['per_page']['entities']}　"
          f"三元组 {result['per_page']['triples']}　节点 {result['per_page']['nodes']}")
    print("")
    print(f"{'页面':44} {'块':>4} {'实体':>5} {'三元组':>6} {'节点':>5}  类型分布 / 抽取器")
    for r in result["rows"]:
        print(f"{r['page'][:44]:44} {r['blocks']:>4} {r['entities']:>5} {r['triples']:>6} "
              f"{r['nodes']:>5}  {r['by_type']} {r['by_extractor']}")
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = _ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
