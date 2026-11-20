# -*- coding: utf-8 -*-
"""tools/download_eval_pages.py —— 下载医家类 59 页评测原图（ModelScope 公开数据集）。

与 download_guji_medical.py 同一来源（bytedance-community/AncientDoc），
但取消 20 张上限，按清单全量下载；支持断点续传与图片完整性校验。

用法：
    python tools/download_eval_pages.py --dry-run   # 只看会下哪些
    python tools/download_eval_pages.py             # 全量下载 59 张
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST_PATH = ROOT / "古籍评测集" / "医家类下载清单.csv"
OUT_DIR = ROOT / "古籍评测集" / "医家类"
MANIFEST = ROOT / "data" / "reports" / "guji_eval59_manifest.json"

REPO = "bytedance-community/AncientDoc"


def file_url(book: str, page_file: str) -> str:
    """ModelScope 单文件下载地址（路径整体 URL 编码，中文书名才能通过）。"""
    path = f"imgs/医家类/{book}/{page_file}"
    return ("https://modelscope.cn/api/v1/datasets/" + REPO + "/repo?Revision=master&FilePath="
            + urllib.parse.quote(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="下载医家类 59 页评测原图")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not LIST_PATH.exists():
        print(f"清单不存在：{LIST_PATH}", file=sys.stderr)
        return 2
    rows = [r for r in csv.DictReader(LIST_PATH.open(encoding="utf-8-sig"))
            if (r.get("文件名(必须原样保存)") or "").strip()]
    print(f"目标 {len(rows)} 张 / {len({r['书名'] for r in rows})} 本")
    if args.dry_run:
        for r in rows:
            print(f"  {r['文件名(必须原样保存)']}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for i, r in enumerate(rows, 1):
        book = r["书名"].strip()
        name = r["文件名(必须原样保存)"].strip()
        page = r["页码"].strip()
        dst = OUT_DIR / name
        url = file_url(book, f"page_{page}.png")
        if dst.exists() and dst.stat().st_size > 0:
            done.append({"name": name, "book": book, "url": url,
                         "bytes": dst.stat().st_size, "status": "cached"})
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, dst.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] 失败 {name}：{type(exc).__name__} {exc}", file=sys.stderr)
            if dst.exists():
                dst.unlink()
            done.append({"name": name, "book": book, "url": url,
                         "status": f"error:{type(exc).__name__}"})
            continue
        size = dst.stat().st_size
        ok, wh = False, [0, 0]
        try:
            from PIL import Image
            with Image.open(dst) as im:
                im.load()
                ok = im.width > 100 and im.height > 100
                wh = [im.width, im.height]
        except Exception:
            pass
        if not ok:
            print(f"  [{i}/{len(rows)}] 校验失败 {name}（{size} 字节）", file=sys.stderr)
            done.append({"name": name, "book": book, "url": url,
                         "bytes": size, "status": "invalid"})
            continue
        print(f"  [{i}/{len(rows)}] OK {name}  {size/1024:.0f}KB  {wh[0]}x{wh[1]}")
        done.append({"name": name, "book": book, "url": url,
                     "bytes": size, "wh": wh, "status": "ok"})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source": f"ModelScope {REPO} / imgs/医家类",
        "notice": "外部公开古籍评测集，非命题单位样本；GT 为数据集机器转写，非人工校对",
        "count": len(done),
        "ok": sum(1 for d in done if d["status"] in {"ok", "cached"}),
        "items": done,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    ok_n = sum(1 for d in done if d["status"] in {"ok", "cached"})
    print(f"\n完成 {ok_n}/{len(done)} 张，清单：{MANIFEST}")
    return 0 if ok_n == len(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
