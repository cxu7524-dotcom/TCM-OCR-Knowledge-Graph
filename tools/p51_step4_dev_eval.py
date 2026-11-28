# -*- coding: utf-8 -*-
"""第 4 步：在开发页上验证采纳规则（只评估，不改规则）。

对每个开发页：套用同一套冻结规则 → 得到修订后文本 → 与 docs 标注的
herb_bbox / herb_verbatim 逐味比对，统计改对、误改、待核。

用法：
  python tools/p51_step4_dev_eval.py --pools <pool1.json> <pool2.json> ... \
      --annot docs/正文方剂表_步骤1标注.json --out <report.json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.normalize import normalize_text  # noqa: E402
from ocr.vertical import imread_unicode    # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_adopt", ROOT / "tools/p51_step4_adopt.py")
_ad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ad)


def norm(s: str) -> str:
    return _ad.eq_text(s)


def corrected_texts(pool: dict, adopted: list) -> dict:
    texts = {b["line_id"]: b["text"] for b in pool["blocks"]}
    for d in adopted:
        s, e = d["span"]
        t = texts[d["line_id"]]
        texts[d["line_id"]] = t[:s] + d["proposal"] + t[e:]
    return texts


def span_of(pool: dict, gray, block: dict, bbox: list) -> tuple:
    boxes, method = _ad.char_ranges(gray, block)
    y1, y2 = bbox[1], bbox[3]
    idx = [i for i, (a, b) in enumerate(boxes) if _ad.overlap(a, b, y1, y2) >= 0.5]
    if not idx:
        return None, method
    return (idx[0], idx[-1] + 1), method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", type=Path, required=True)
    ap.add_argument("--annot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    annot = json.loads(args.annot.read_text(encoding="utf8"))
    pages = {(p["book"], p["page_no"]): p for p in annot["pages"]}

    report = {"pages": [], "totals": {"ingredients": 0, "ok_before": 0, "fixed": 0,
                                      "broken": 0, "pending_touched": 0, "adopted": 0},
              "samples": []}
    for pool_path in args.pools:
        pool = json.loads(pool_path.read_text(encoding="utf8"))
        gray = imread_unicode(pool["image"])
        if gray is not None and gray.ndim == 3:
            gray = np.mean(gray[:, :, :3], axis=2).astype(np.uint8)
        res = _ad.apply_rules(pool)
        texts = corrected_texts(pool, res["adopted"])
        book = "思濟堂方書" if "/思濟堂方書/" in pool["image"].replace("\\", "/") else "丹方彚编"
        page_no = int(Path(pool["image"]).stem.split("_")[-1])
        ann = pages.get((book, page_no), {"instances": [], "negative_spans": []})
        blocks = {b["line_id"]: b for b in pool["blocks"]}

        prows = []
        for inst in ann["instances"]:
            for ing in inst.get("ingredients", []):
                hb = ing.get("herb_bbox")
                herb = ing.get("herb_verbatim") or ""
                if not hb or not herb:
                    continue
                # 找覆盖该 bbox 的块
                best, best_area = None, 0
                for b in pool["blocks"]:
                    bx1, by1, bx2, by2 = b["bbox"]
                    ox = min(bx2, hb[2]) - max(bx1, hb[0])
                    oy = min(by2, hb[3]) - max(by1, hb[1])
                    if ox > 0 and oy > 0 and ox * oy > best_area:
                        best, best_area = b, ox * oy
                if best is None:
                    continue
                sp, method = span_of(pool, gray, best, hb)
                if sp is None:
                    continue
                before = best["text"][sp[0]:sp[1]]
                after = texts[best["line_id"]][sp[0]:sp[1]]
                # 块内字位切分受 OCR 缺字影响，判定改用「药名是否仍在块文本中」；
                # before/after 切片仅作参考展示
                tb = norm(best["text"])
                ta = norm(texts[best["line_id"]])
                ok_b = norm(herb) in tb
                ok_a = norm(herb) in ta
                row = {"page": "%s-%s" % (book, page_no), "instance": inst["id"],
                       "herb": herb, "line_id": best["line_id"], "span": list(sp),
                       "before": before, "after": after,
                       "ok_before": ok_b, "ok_after": ok_a, "changed": before != after,
                       "pos_method": method}
                if ok_b and not ok_a and row["changed"]:
                    row["verdict"] = "误改"
                    report["totals"]["broken"] += 1
                elif not ok_b and ok_a:
                    row["verdict"] = "改对"
                    report["totals"]["fixed"] += 1
                elif ok_b:
                    row["verdict"] = "保持正确"
                    report["totals"]["ok_before"] += 1
                else:
                    row["verdict"] = "未改对"
                report["totals"]["ingredients"] += 1
                prows.append(row)
                report["samples"].append(row)

        # 负例：修订是否落进标注的非配伍片段
        neg_hit = 0
        for neg in ann.get("negative_spans", []):
            nb = neg.get("bbox")
            if not nb:
                continue
            for d in res["adopted"]:
                bx1, by1, bx2, by2 = d["bbox"]
                if _ad.overlap(by1, by2, nb[1], nb[3]) > 0.5 and nb[0] <= bx2 and nb[2] >= bx1:
                    neg_hit += 1
        report["totals"]["adopted"] += len(res["adopted"])
        report["pages"].append({"page": "%s-%s" % (book, page_no), "pool": str(pool_path),
                                "blocks": len(pool["blocks"]), "decisions": len(res["decisions"]),
                                "adopted": len(res["adopted"]),
                                "pending": sum(1 for d in res["decisions"] if d["status"] == "待核"),
                                "ingredients": len(prows),
                                "neg_span_adopted": neg_hit,
                                "rows": prows})
        print(json.dumps({"page": "%s-%s" % (book, page_no), "adopted": len(res["adopted"]),
                          "ingredients": len(prows),
                          "verdicts": {v: sum(1 for r in prows if r["verdict"] == v)
                                       for v in ("改对", "误改", "保持正确", "未改对")}},
                         ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"totals": report["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
