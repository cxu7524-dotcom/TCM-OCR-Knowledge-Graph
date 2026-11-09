# -*- coding: utf-8 -*-
"""
tools/import_guji_eval.py —— 导入外部古籍评测集（label.csv）为项目可用格式

背景：
  用户在 `古籍评测集/label.csv` 提供了一份外部古籍多模态评测集：
  2584 条 / 14 个四部分类目，每条含整页繁体 OCR 原文、白话译文、三类问答。
  其中 **医家类 176 条 / 10 本医书** 与赛题二「面向中医古籍与中药资源」直接对口。

为什么分两步：
  label.csv 只有**整页** OCR 文本，没有 bbox / line_id，也没有图片。
  直接把整页当成一个 Block 虽然能过校验，但必须先知道图片真实宽高，
  否则 bbox 是假的。所以：
    第 1 步 `--export`：导出「下载清单」与「原始转写 jsonl」（不含 bbox）
    第 2 步 `--build-gt`：图片到位后，读真实宽高，转成正式 GT（整页一个 Block）

合规：
  - 原始影像与外部标注放 `data/annotation/official/`（`.gitignore` 已排除），不进仓库。
  - 本脚本只做格式转换，不改写任何原文；`quote` 一字不动。
  - 外部数据集的来源与授权须由使用者确认，见 `data/annotation/official/来源登记.md`。

用法（项目根目录）：
    python tools/import_guji_eval.py --stats
    python tools/import_guji_eval.py --export
    python tools/import_guji_eval.py --build-gt --images "古籍评测集/医家类"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "古籍评测集" / "label.csv"
OUT_DIR = ROOT / "data" / "annotation" / "official"
LIST_PATH = ROOT / "古籍评测集" / "医家类下载清单.csv"
RAW_PATH = OUT_DIR / "guji_raw.jsonl"

# 目标类目：医家类与赛题二对口；其余类目默认不导入（避免场景不匹配）
TARGET_TYPE = "医家类"

PAGE_RE = re.compile(r"page_(\d+)\.(?:png|jpg|jpeg)$", re.I)


# ----------------------------------------------------------------------
# 一、读取与解析
# ----------------------------------------------------------------------

def load_rows(path: Path = CSV_PATH) -> List[Dict[str, str]]:
    """读 label.csv（BOM + 多行字段，必须用 csv 模块，不能按行切）。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到标注文件：{path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def book_of(name: str) -> str:
    """'温疫方論page_23.png' → '温疫方論'。"""
    return PAGE_RE.sub("", (name or "").strip())


def page_of(name: str) -> int:
    """'温疫方論page_23.png' → 23；解析不出返回 0。"""
    m = PAGE_RE.search((name or "").strip())
    return int(m.group(1)) if m else 0


def parse_questions(text: str) -> List[str]:
    """从问答列里抽出「问题N：...」，答案不要（只用于构造有依据问题集）。"""
    if not text:
        return []
    out: List[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        m = re.match(r"^问题\s*\d+\s*[：:]\s*(.+)$", line)
        if m:
            q = m.group(1).strip().strip('"').strip()
            if q:
                out.append(q)
    return out


def normalize_row(r: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """单行 → 中间结构；缺 OCR 原文的条目直接丢弃（无法当 GT）。"""
    name = (r.get("name") or "").strip()
    ocr = (r.get("ocr") or r.get("OCR") or "").strip()
    if not name or not ocr:
        return None
    return {
        "name": name,
        "type": (r.get("type") or "").strip(),
        "book": book_of(name),
        "page_no": page_of(name),
        "ocr": ocr,
        "translation": (r.get("translation") or r.get("TRANSLATION") or "").strip(),
        "qa": {
            "knowledge": (r.get("knowlage") or r.get("KNOWLAGE") or "").strip(),
            "language": (r.get("language") or r.get("LANGUAGE") or "").strip(),
            "reasoning": (r.get("reasoning") or r.get("REASONING") or "").strip(),
        },
        "questions": (
            parse_questions(r.get("knowlage") or r.get("KNOWLAGE") or "")
            + parse_questions(r.get("language") or r.get("LANGUAGE") or "")
            + parse_questions(r.get("reasoning") or r.get("REASONING") or "")
        ),
    }


# ----------------------------------------------------------------------
# 二、统计
# ----------------------------------------------------------------------

def cmd_stats() -> int:
    rows = load_rows()
    print(f"总记录：{len(rows)}")
    print("\n类目分布：")
    for t, c in Counter((r.get("type") or "").strip() for r in rows).most_common():
        print("  %-14s %4d" % (t, c))

    med = [x for x in (normalize_row(r) for r in rows) if x and x["type"] == TARGET_TYPE]
    print(f"\n{TARGET_TYPE}：{len(med)} 条")
    books = Counter(x["book"] for x in med)
    print(f"涉及医书：{len(books)} 本")
    for b, c in books.most_common():
        print("  %-26s %2d 条" % (b, c))

    lens = sorted(len(x["ocr"]) for x in med)
    if lens:
        print("\nOCR 字数：最短 %d / 中位 %d / 最长 %d" % (lens[0], lens[len(lens) // 2], lens[-1]))
    print("缺译文的条目：", sum(1 for x in med if not x["translation"]))
    print("抽出问题总数：", sum(len(x["questions"]) for x in med))
    return 0


# ----------------------------------------------------------------------
# 三、导出：下载清单 + 原始转写
# ----------------------------------------------------------------------

def cmd_export(limit: Optional[int] = None) -> int:
    rows = load_rows()
    med = [x for x in (normalize_row(r) for r in rows) if x and x["type"] == TARGET_TYPE]
    med.sort(key=lambda x: (x["book"], x["page_no"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 原始转写（不含 bbox，图片到位后再 build-gt）
    with RAW_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for x in med:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 2) 下载清单：按书分组，方便逐本下载并保持原文件名
    with LIST_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "书名", "页码", "文件名(必须原样保存)", "OCR字数", "有译文", "问题数"])
        for i, x in enumerate(med, start=1):
            w.writerow([i, x["book"], x["page_no"], x["name"],
                        len(x["ocr"]), "是" if x["translation"] else "否", len(x["questions"])])

    print(f"原始转写：{RAW_PATH}（{len(med)} 条）")
    print(f"下载清单：{LIST_PATH}")
    print("\n按书统计（下载时按书名建子目录，文件名保持原样）：")
    for b, c in Counter(x["book"] for x in med).most_common():
        print("  %-26s %2d 张" % (b, c))
    print("\n提醒：浏览器另存会把 '书名page_23.png' 改成 'page_23 (1).png'，"
          "文件名一改就再也对不上标注，务必手动保持原名。")
    return 0


# ----------------------------------------------------------------------
# 四、图片到位后：转正式 GT（整页一个 Block）
# ----------------------------------------------------------------------

def _image_size(path: Path):
    """读图片宽高；Pillow 不可用时返回 (0, 0)，由调用方决定是否跳过。"""
    try:
        from PIL import Image
    except ImportError:
        return 0, 0
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _scan_images(images_dir: Path) -> Dict[str, Path]:
    """扫描图片目录 → {逻辑文件名: 实际路径}，兼容两种布局。

      ① 扁平布局：`医家类/丹方彚编page_23.png`            → 键 `丹方彚编page_23.png`
      ② 子目录布局：`医家类/丹方彚编/page_23.png`          → 键 `丹方彚编page_23.png`（书名取目录名）

    两种布局都归一到 `书名page_N.png`，正好与 `guji_raw.jsonl` 的 `name` 字段对齐。
    """
    out: Dict[str, Path] = {}
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not images_dir.is_dir():
        return out
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.name] = p
        elif p.is_dir():
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in exts:
                    out[f"{p.name}{q.name}"] = q
    return out


def cmd_build_gt(images_dir: Path) -> int:
    """把 guji_raw.jsonl + 已下载图片 → 正式 GT（整页 Block + 有依据问题集）。

    输出到 data/annotation/official/（gitignore 已排除），不污染合成集 GT。
    只处理「图片目录里实际存在的书」，无标注页跳过并列出清单。
    """
    from app.schemas import source_hash_of

    if not RAW_PATH.exists():
        print(f"先跑 --export 生成 {RAW_PATH}", file=sys.stderr)
        return 2
    raw = [json.loads(l) for l in RAW_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    found = _scan_images(images_dir)
    books = {book_of(n) for n in found}
    if not found:
        print(f"图片目录里没有图片：{images_dir}", file=sys.stderr)
        return 2

    ocr_out: List[Dict[str, Any]] = []
    qa_out: List[Dict[str, Any]] = []
    missing: List[str] = []
    skipped_no_size = 0
    used: set = set()

    for x in raw:
        if x["book"] not in books:          # 只做有图片的书，其余 7 本不参与
            continue
        img = found.get(x["name"])
        if img is None:
            missing.append(x["name"])
            continue
        w, h = _image_size(img)
        if not w or not h:
            skipped_no_size += 1
            continue
        used.add(x["name"])
        page_id = f'{x["book"]}-p{x["page_no"]:04d}'
        bbox = [0, 0, w, h]
        ocr_out.append({
            "page_id": page_id,
            "bbox": bbox,
            "line_id": "L00",
            "quote": x["ocr"],                 # 原文一字不改
            "source_hash": source_hash_of("guji", page_id, x["name"], x["ocr"][:64]),
            "annotator": "external-guji-eval",
            "version": "v1",
            "is_synthetic": False,
            "is_demo": False,
            "text": x["ocr"],
            "text_norm": x["ocr"],             # 归一化交给 ocr/normalize 与 align_norm
            "kind": "page",
            "source_name": x["name"],
            "image_path": str(img),
            "book": x["book"],
            "page_no": x["page_no"],
        })
        for q in x["questions"]:
            qa_out.append({
                "question": q,
                "page_id": page_id,
                "expected_refused": False,
                "annotator": "external-guji-eval",
                "version": "v1",
                "is_demo": False,
            })

    ocr_path = OUT_DIR / "ocr_transcription.jsonl"
    qa_path = OUT_DIR / "qa_log_gold.jsonl"
    for p, items in ((ocr_path, ocr_out), (qa_path, qa_out)):
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    unlabeled = sorted(n for n in found if n not in used)
    print(f"GT 转写：{ocr_path}（{len(ocr_out)} 页）")
    print(f"有依据问题：{qa_path}（{len(qa_out)} 条）")
    print(f"涉及书目：{len(books)} 本 —— " + "、".join(sorted(books)))
    print(f"跳过无标注页：{len(unlabeled)} 张")
    if unlabeled:
        print("  " + "、".join(unlabeled))
    print(f"未找到图片：{len(missing)} 张" + (f"，例如：{missing[:5]}" if missing else ""))
    if skipped_no_size:
        print(f"图片宽高读取失败 {skipped_no_size} 张（Pillow 未安装？）")
    print("\n注意：这批 GT 是**外部数据集的机器转写**，不是人工逐字校对；")
    print("      当基准用前须抽检 20%，错字会直接算进 CER。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="导入外部古籍评测集（label.csv）")
    ap.add_argument("--stats", action="store_true", help="统计类目 / 书目 / 字数")
    ap.add_argument("--export", action="store_true", help="导出下载清单与原始转写")
    ap.add_argument("--build-gt", action="store_true", help="图片到位后转正式 GT")
    ap.add_argument("--images", default="古籍评测集/医家类", help="图片目录（默认 古籍评测集/医家类）")
    args = ap.parse_args()

    if args.stats:
        return cmd_stats()
    if args.export:
        return cmd_export()
    if args.build_gt:
        return cmd_build_gt(Path(args.images))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
