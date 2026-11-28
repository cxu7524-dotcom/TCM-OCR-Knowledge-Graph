# -*- coding: utf-8 -*-
"""《温疫方論》逐页 OCR 原字与繁简兼容评测。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REMOVE = set(" \t\r\n\u3000，。、；：？！“”‘’（）《》〈〉「」『』【】〔〕…—·,.;:?!()[]<>\"\'`~～|｜/\\-_=+•●○◦□")
# Each group starts with its canonical form.
_GROUPS = ("为爲為", "于於", "温溫", "热熱", "则則", "满滿", "参參叁", "减減", "里裏裡", "饮飲", "渐漸", "真眞", "数數", "户戶", "风風", "蜕蛻", "颐頤", "时時", "属屬", "测測", "滞滯", "内內", "卫衛衞", "绝絶絕", "强強", "脉脈", "脱脫", "将將", "杂雜雑", "凉涼", "众衆眾", "门門", "鹤鶴", "钟鍾鐘", "效効", "别別", "浅淺", "万萬", "雾霧", "岂豈", "鸡雞鷄", "剂劑", "进進", "陈陳", "赖賴", "谬謬", "喷噴", "状狀", "协協", "结結", "盖蓋", "见見", "频頻", "值値", "载載", "恶惡", "清淸", "畅暢", "盏盞", "谵譫", "证證", "误誤", "变變", "发發", "与與", "无無", "经經", "药藥", "实實", "连連", "传傳", "导導", "复復", "患", "昼晝", "书書", "虽雖", "气氣", "伤傷", "胀脹", "肿腫", "辞辭")


def clean_text(s: str) -> str:
    return "".join(ch for ch in s if ch not in _REMOVE and not ch.isspace())


@lru_cache(maxsize=1)
def _single_mapping():
    # Importing the ocr package imports pipeline; defer this until CLI overrides are set.
    from ocr.normalize import mapping
    result = {k: v for k, v in mapping().items() if len(k) == len(v) == 1}
    for group in _GROUPS:
        result.update({ch: group[0] for ch in group})
    return result


def is_ts_equiv(a: str, b: str) -> bool:
    if a == b:
        return True
    mapping = _single_mapping()
    def canonical(ch):
        seen = set()
        while ch in mapping and ch not in seen and mapping[ch] != ch:
            seen.add(ch)
            ch = mapping[ch]
        return ch
    return canonical(a) == canonical(b)


def levenshtein_ops(gt: str, pred: str, equiv_fn=None) -> tuple[int, int, int, int]:
    """Wagner–Fischer distance and deterministic backtrace (diagonal, deletion, insertion)."""
    n, m = len(gt), len(pred)
    previous = list(range(m + 1))
    trace = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        trace[0][j] = 2
    for i, a in enumerate(gt, 1):
        row = [i] + [0] * m
        trace[i][0] = 1
        for j, b in enumerate(pred, 1):
            same = a == b or (equiv_fn is not None and equiv_fn(a, b))
            candidates = (previous[j - 1] + (not same), previous[j] + 1, row[j - 1] + 1)
            best = min(range(3), key=candidates.__getitem__)
            row[j] = candidates[best]
            trace[i][j] = best
        previous = row
    i, j = n, m
    subs = dels = ins = 0
    while i or j:
        op = trace[i][j]
        if op == 0:
            subs += not (gt[i - 1] == pred[j - 1] or (equiv_fn is not None and equiv_fn(gt[i - 1], pred[j - 1])))
            i -= 1
            j -= 1
        elif op == 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return previous[m], subs, dels, ins


def evaluate_records(records, body_only=False):
    """Evaluate in-memory dicts: page, gt, blocks, engine and direction.

    CER is undefined (None) for an empty reference; such pages remain in totals.
    """
    pages = []
    for record in records:
        blocks = sorted(record["blocks"], key=lambda b: b["reading_order"])
        pred = clean_text("".join(b["text"] or "" for b in blocks
                                  if not body_only or b["kind"] not in ("margin_note", "spine")))
        gt = clean_text(record["gt"])
        strict = levenshtein_ops(gt, pred)
        ts = levenshtein_ops(gt, pred, is_ts_equiv)
        page = {"page": record["page"], "gt_chars": len(gt), "pred_chars": len(pred),
                "engine": record.get("engine"), "direction": record.get("direction")}
        for suffix, values in (("", strict), ("_ts", ts)):
            dist, sub, delete, insert = values
            page.update({"dist" + suffix: dist, "S" + suffix: sub, "D" + suffix: delete,
                         "I" + suffix: insert, "cer" + suffix: dist / len(gt) if gt else None})
        pages.append(page)
    total_chars = sum(p["gt_chars"] for p in pages)
    totals = {"page_count": len(pages), "gt_chars": total_chars,
              "pred_chars": sum(p["pred_chars"] for p in pages),
              "engine_distribution": dict(Counter(p["engine"] or "unknown" for p in pages)),
              "direction_distribution": dict(Counter(p["direction"] or "unknown" for p in pages))}
    for suffix in ("", "_ts"):
        for key in ("dist", "S", "D", "I"):
            totals[key + suffix] = sum(p[key + suffix] for p in pages)
        totals["cer" + suffix] = totals["dist" + suffix] / total_chars if total_chars else None
        cers = [p["cer" + suffix] for p in pages if p["cer" + suffix] is not None]
        totals["median_cer" + suffix] = statistics.median(cers) if cers else None
    return {"summary": totals, "pages": pages}


def load_gt(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["page"]].append((int(row["order"]), row["text"]))
    return {page: "".join(text for _, text in sorted(rows)) for page, rows in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-dir", type=Path, default=ROOT / "古籍评测集/医家类/温疫方論")
    parser.add_argument("--gt-csv", type=Path, default=ROOT / "output/wyfl_kg/逐列原字.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/wyfl_eval/run")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rec-model", default=None)
    parser.add_argument("--pick", choices=("auto", "vertical", "page"))
    parser.add_argument("--correct", choices=("0", "1"))
    parser.add_argument("--ctc-rerank", choices=("0", "1"))
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--frame-isolate", choices=("0", "1"))
    parser.add_argument("--remove-grid", choices=("0", "1"))
    parser.add_argument("--strip-kunten", choices=("0", "1"))
    parser.add_argument("--merge-jiazhu", choices=("0", "1"))
    parser.add_argument("--dual-track-detector", choices=("stage5", "stage8"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    for name, value in (("TCM_OCR_CTC_RERANK", args.ctc_rerank), ("TCM_OCR_REC_MODEL", args.rec_model), ("TCM_OCR_PICK", args.pick), ("TCM_OCR_CORRECT", args.correct), ("TCM_OCR_FRAME_ISOLATE", args.frame_isolate), ("TCM_OCR_REMOVE_GRID", args.remove_grid), ("TCM_OCR_STRIP_KUNTEN", args.strip_kunten), ("TCM_OCR_MERGE_JIAZHU", args.merge_jiazhu), ("TCM_OCR_DUAL_TRACK_DETECTOR", args.dual_track_detector)):
        if value:
            os.environ[name] = value
    from ocr.pipeline import run_ocr
    from app.config import CONFIG, BUNDLED_REC_MODEL

    gt = load_gt(args.gt_csv)
    numbers = list(range(4, 25))[:args.limit or 21]
    for number in numbers:
        image = args.img_dir / f"page_{number}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        if f"p{number}" not in gt:
            raise ValueError(f"GT missing for p{number}")
    start = time.perf_counter()
    records = []
    raw = []
    for number in numbers:
        page = f"p{number}"
        result = run_ocr(str(args.img_dir / f"page_{number}.png"), page)
        blocks = [{key: getattr(b, key) for key in ("line_id", "reading_order", "bbox", "conf", "kind", "text")} for b in result.blocks]
        records.append({"page": page, "gt": gt[page], "blocks": blocks, "engine": result.engine,
                        "direction": result.layout.direction})
        raw.append({"page": page, "blocks": blocks, "engine": result.engine,
                    "direction": result.layout.direction, "model_loaded": result.model_loaded,
                    "is_demo": result.is_demo, "notes": result.notes})
    report = evaluate_records(records, args.body_only)
    report["summary"]["elapsed_seconds"] = time.perf_counter() - start
    report["summary"]["rec_model"] = CONFIG.ocr.rec_model or BUNDLED_REC_MODEL
    report["config"] = {"img_dir": str(args.img_dir), "gt_csv": str(args.gt_csv), "out_dir": str(args.out_dir),
                        "tag": args.tag, "limit": args.limit, "body_only": args.body_only,
                        "rec_model": CONFIG.ocr.rec_model or BUNDLED_REC_MODEL,
                        "rec_model_source": CONFIG.ocr.rec_model_source,
                        "pick": CONFIG.ocr.pick, "correct": CONFIG.ocr.correct_enabled, "ctc_rerank": CONFIG.ocr.ctc_rerank,
                        "frame_isolate": CONFIG.ocr.frame_isolate, "remove_grid": CONFIG.ocr.remove_grid,
                        "strip_kunten": CONFIG.ocr.strip_kunten, "merge_jiazhu": CONFIG.ocr.merge_jiazhu,
                        "dual_track_detector": CONFIG.ocr.dual_track_detector}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("summary.json", report), ("raw_ocr.json", raw)):
        (args.out_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
