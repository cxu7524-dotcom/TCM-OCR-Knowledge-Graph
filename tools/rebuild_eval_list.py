# -*- coding: utf-8 -*-
"""从 official_label.csv 重建 医家类下载清单.csv（59 页评测子集）。"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "annotation" / "official_label.csv"
OUT_DIR = ROOT / "古籍评测集"
LIST_PATH = OUT_DIR / "医家类下载清单.csv"

# 评测报告用的三部书（繁体，以 label.csv 里 name 的实际写法为准）
EVAL_BOOKS = {"丹方彚编", "合併脉訣難經太素評林", "思濟堂方書"}

with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if (r.get("type") or "").strip() == "医家类"]

per_book = defaultdict(list)
for r in rows:
    name = (r.get("name") or "").strip()
    ocr = (r.get("OCR") or "").strip()
    if not name or not ocr or "page_" not in name:
        continue
    book = name.split("page_")[0]
    per_book[book].append(name)

print("医家类各书页数：")
for book in sorted(per_book):
    tag = "  ← 评测子集" if book in EVAL_BOOKS else ""
    print(f"  {book}: {len(per_book[book])} 页{tag}")

targets = []
for book in EVAL_BOOKS:
    for name in sorted(per_book.get(book, []), key=lambda n: int(re.search(r"page_(\d+)", n).group(1))):
        page = re.search(r"page_(\d+)", name).group(1)
        targets.append({"书名": book, "文件名(必须原样保存)": name, "页码": page})

print(f"\n评测子集合计：{len(targets)} 页")
if len(sys.argv) > 1 and sys.argv[1] == "--write":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LIST_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["书名", "文件名(必须原样保存)", "页码"])
        w.writeheader()
        w.writerows(targets)
    print(f"清单已写入 {LIST_PATH}")
