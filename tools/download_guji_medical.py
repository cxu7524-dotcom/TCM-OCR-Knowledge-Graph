# -*- coding: utf-8 -*-
"""
tools/download_guji_medical.py —— 下载医家类小规模验证样本（阶段五 P4）

来源：ModelScope `bytedance-community/AncientDoc`（公开古籍理解评测集，
      本项目 `古籍评测集/label.csv` 即该数据集的标注表）。
路径：`imgs/医家类/<书名>/page_N.png`

硬约束（写在代码里，防止手滑下多）：
  - **最多 20 张**（10 本医书 × 2 页），超过直接拒绝；
  - 只下 `医家类下载清单.csv` 里登记过的页（这些页有机器转写 OCR 文本，可算 CER）；
  - 文件名保持 `书名page_N.png`（`tools/import_guji_eval.py --build-gt` 按此匹配）；
  - 图片只落 `古籍评测集/医家类/`（`.gitignore` 已排除），不进仓库、不上传外部服务。

用法：
    python tools/download_guji_medical.py --dry-run     # 只看会下哪些
    python tools/download_guji_medical.py               # 实际下载 20 张
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
LIST_PATH = ROOT / "古籍评测集" / "医家类下载清单.csv"
OUT_DIR = ROOT / "古籍评测集" / "医家类"
MANIFEST = ROOT / "data" / "reports" / "guji_medical_manifest.json"

REPO = "bytedance-community/AncientDoc"
MAX_IMAGES = 20
PER_BOOK = 2


def file_url(book: str, page_file: str) -> str:
    """ModelScope 单文件下载地址（路径需整体 URL 编码，中文书名才能通过）。"""
    path = f"imgs/医家类/{book}/{page_file}"
    return ("https://modelscope.cn/api/v1/datasets/" + REPO + "/repo?Revision=master&FilePath="
            + urllib.parse.quote(path))


def pick_targets() -> List[Dict[str, str]]:
    """按清单顺序，每本书取前 PER_BOOK 页，总量不超过 MAX_IMAGES。"""
    rows = list(csv.DictReader(LIST_PATH.open(encoding="utf-8-sig")))
    per_book: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        book = (r.get("书名") or "").strip()
        name = (r.get("文件名(必须原样保存)") or "").strip()
        page = (r.get("页码") or "").strip()
        if book and name and page:
            per_book[book].append({"book": book, "page": page, "name": name})
    out: List[Dict[str, str]] = []
    for book, items in per_book.items():
        out.extend(items[:PER_BOOK])
    return out[:MAX_IMAGES]


def main() -> int:
    ap = argparse.ArgumentParser(description="下载医家类验证样本（最多 20 张）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-book", type=int, default=PER_BOOK)
    args = ap.parse_args()

    targets = pick_targets() if args.per_book == PER_BOOK else []
    if not targets:
        print("清单为空", file=sys.stderr)
        return 2
    if len(targets) > MAX_IMAGES:
        print(f"拒绝：目标 {len(targets)} 张 > 上限 {MAX_IMAGES} 张", file=sys.stderr)
        return 2
    print(f"目标 {len(targets)} 张 / {len({t['book'] for t in targets})} 本：")
    for t in targets:
        print(f"  {t['name']}")
    if args.dry_run:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: List[Dict[str, object]] = []
    for t in targets:
        dst = OUT_DIR / t["name"]
        url = file_url(t["book"], f"page_{t['page']}.png")
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  跳过（已存在）{t['name']}")
            done.append({**t, "url": url, "bytes": dst.stat().st_size, "status": "cached"})
            continue
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:
            print(f"  失败 {t['name']}：{type(exc).__name__} {exc}", file=sys.stderr)
            if dst.exists():
                dst.unlink()
            done.append({**t, "url": url, "status": f"error:{type(exc).__name__}"})
            continue
        size = dst.stat().st_size
        ok = False
        try:
            from PIL import Image
            with Image.open(dst) as im:
                ok = im.width > 100 and im.height > 100
                wh = [im.width, im.height]
        except Exception:
            wh = [0, 0]
        if not ok:
            print(f"  校验失败 {t['name']}（{size} 字节，{wh}）", file=sys.stderr)
            done.append({**t, "url": url, "bytes": size, "status": "invalid"})
            continue
        print(f"  OK {t['name']}  {size} 字节  {wh[0]}x{wh[1]}")
        done.append({**t, "url": url, "bytes": size, "wh": wh, "status": "ok"})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source": f"ModelScope {REPO} / imgs/医家类",
        "notice": "外部公开古籍评测集，非命题单位样本；GT 为数据集机器转写，非人工校对",
        "count": len(done),
        "items": done,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    ok_n = sum(1 for d in done if d["status"] in {"ok", "cached"})
    print(f"\n完成 {ok_n}/{len(done)} 张，清单：{MANIFEST}")
    return 0 if ok_n == len(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
