# -*- coding: utf-8 -*-
"""第 6 步：留出页整页验收（原始 OCR → 冻结自动校正 → 组成抽取 → 逐项评分）。

逐页流程：
  1. 从原图跑一次真实整页 OCR（关闭文字纠错），**先保存完整 OCR 文档**（含 block_ref /
     reading_order / bbox / source_hash），再构建候选池；
  2. B 组：同一份原始 OCR 生成 4H/2H 窗 + 两模型 CTC 前 5，套第四步冻结采纳规则；
  3. A/B 各自调用 extract.segment_composition.extract_composition()，并按 ID 15 口径
     合并原规则关系（本方「组成」由逐味入口接管）；
  4. 全部落盘后，才读人工标注评分（药名位置 + 组成关系）。

用法：
  python tools/p51_step6_holdout.py --image <原图> --page-no 37 --annot docs/...json --out <目录>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                                  # noqa: E402
from extract.rules import extract_all_audited                   # noqa: E402
from extract.segment_composition import extract_composition     # noqa: E402
from extract.formula_segments import _Stream                    # noqa: E402
from graph.build import build_triples                           # noqa: E402
from ocr.normalize import normalize_text                        # noqa: E402
from ocr.vertical import imread_unicode                         # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_pool", ROOT / "tools/p51_step4_pool.py")
_pool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pool)
_spec2 = importlib.util.spec_from_file_location("p51_adopt", ROOT / "tools/p51_step4_adopt.py")
_adopt = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_adopt)
_spec3 = importlib.util.spec_from_file_location("p51_ctc", ROOT / "tools/p51_step3_ctc_candidates.py")
_ctc = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_ctc)

EQUIV = {"殼": "壳", "羗": "羌"}
TRANS = str.maketrans(EQUIV)


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s: str) -> str:
    return normalize_text(s or "").translate(TRANS)


# ---------- 1. 整页 OCR ----------
def page_ocr(image: Path, page_id: str, out: Path) -> dict:
    tmp = out / "_tmp"
    ocr = _pool.run_page_ocr(image, page_id, tmp)
    doc = {"page_id": page_id, "image_path": str(image),
           "width": ocr["width"], "height": ocr["height"],
           "engine": ocr["engine"], "is_demo": False, "model_loaded": True,
           "blocks": ocr["blocks"], "notes": ["第6步留出页：真实整页 OCR，纠错关闭"]}
    (out / "ocr.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return doc


# ---------- 2. 候选池 + 冻结采纳 ----------
def build_pool(doc: dict, image: Path) -> dict:
    img = imread_unicode(str(image))
    blocks = []
    for b in doc["blocks"]:
        n = _pool.hz(b["text"])
        if n == 0:
            continue
        x1, y1, x2, y2 = b["bbox"]
        blocks.append({"line_id": b["line_id"], "block_ref": b["block_ref"], "bbox": b["bbox"],
                       "text": b["text"], "n_hanzi": n,
                       "est_char_h": max(1, int(round((y2 - y1) / n)))})
    windows = []
    for mult, tag in ((4, "H4"), (2, "H2")):
        for b in blocks:
            for w in _pool._win.windows_for(b, mult):
                w = dict(w)
                w["window_id"] = "%s:%s:%s" % (tag, b["line_id"], w["window_id"].split(":")[-1])
                w["mult"] = mult
                windows.append(w)
    cands = _pool.infer_candidates(img, windows, doc["page_id"])
    return {"page_id": doc["page_id"], "image": str(image.resolve()),
            "image_sha256": sha256_of(image), "image_size": [doc["width"], doc["height"]],
            "blocks": blocks,
            "windows": {w["window_id"]: {"line_id": w["line_id"], "x1": w["x1"], "y1": w["y1"],
                                         "x2": w["x2"], "y2": w["y2"],
                                         "height": w["height"], "mult": w["mult"]} for w in windows},
            "candidates": cands}


# ---------- 4. 抽取 + 合并 ----------
def relations_for(blocks: list) -> dict:
    ext = extract_composition(blocks)
    ents, cands, _a = extract_all_audited(blocks, is_demo=False)
    old = build_triples(ents, blocks, cands)
    heads = {norm(s["title"]) for s in ext["segments"]}
    kept = [t for t in old if not (t.relation == "组成" and norm(t.head) in heads)]
    return {"extract": ext, "merged": kept + ext["relations"],
            "old_kept": len(kept), "old_all": len(old)}


# ---------- 5. 评分 ----------
def map_span(gray, block, y1, y2):
    boxes, method = _adopt.char_ranges(gray, {"bbox": list(block.bbox), "text": block.text})
    idx = [i for i, (a, b) in enumerate(boxes) if _adopt.overlap(a, b, y1, y2) >= 0.5]
    if not idx:
        return None, method
    return (idx[0], idx[-1] + 1), method


def score_names(blocks: list, gray, gold_herbs: list) -> dict:
    """逐味：在覆盖其 bbox 的块里按字位切片核对（strict），并记录近似命中。"""
    rows = []
    for g in gold_herbs:
        hb = g["herb_bbox"]
        best, area = None, 0
        for b in blocks:
            bx1, by1, bx2, by2 = b.bbox
            ox = min(bx2, hb[2]) - max(bx1, hb[0])
            oy = min(by2, hb[3]) - max(by1, hb[1])
            if ox > 0 and oy > 0 and ox * oy > area:
                best, area = b, ox * oy
        row = {"page": g["page"], "instance": g["instance"], "herb": g["herb"],
               "bbox": hb, "line_id": best.line_id if best else None,
               "strict": False, "approx": False, "slice": None, "pos_method": None}
        if best is not None:
            sp, method = map_span(gray, best, hb[1], hb[3])
            row["pos_method"] = method
            if sp:
                sl = best.text[sp[0]:sp[1]]
                row["slice"] = sl
                row["span"] = list(sp)
                row["strict"] = norm(sl) == norm(g["herb"])
            row["approx"] = norm(g["herb"]) in norm(best.text)
        rows.append(row)
    return {"rows": rows,
            "strict_hits": sum(1 for r in rows if r["strict"]),
            "approx_hits": sum(1 for r in rows if r["approx"]),
            "total": len(rows)}


def score_composition(merged: list, gold_pairs: list, gold_herbs: list) -> dict:
    """组成关系：药名 + 方剂 + 证据位置三者相符才算命中。"""
    herbs_by_key = {}
    for g in gold_herbs:
        herbs_by_key.setdefault((norm(g["formula"]), norm(g["herb"])), []).append(g)
    page_herbs = {(norm(g["herb"])) for g in gold_herbs}
    formulas = {norm(g["formula"]) for g in gold_herbs}
    hits, wrong, extra, rows = {}, [], [], []
    for t in merged:
        if t.relation != "组成":
            continue
        key = (norm(t.head), norm(t.tail))
        ev = (t.relation_evidence or [{}])[0]
        bb = ev.get("bbox") or list(t.bbox)
        best = None
        for g in herbs_by_key.get(key, []):
            if not bb:
                continue
            ox = min(bb[2], g["herb_bbox"][2]) - max(bb[0], g["herb_bbox"][0])
            oy = min(bb[3], g["herb_bbox"][3]) - max(bb[1], g["herb_bbox"][1])
            if ox > 0 and oy > 0 and (best is None or ox * oy > best):
                best = ox * oy
        row = {"head": t.head, "tail": t.tail, "block_ref": ev.get("block_ref"),
               "quote": ev.get("quote"), "span": [ev.get("start"), ev.get("end")],
               "bbox": bb, "overlap": best}
        if best:
            hits[key] = row
            row["verdict"] = "命中"
        elif key in herbs_by_key:
            row["verdict"] = "位置不符"
            wrong.append(row)
        elif norm(t.tail) in page_herbs:
            row["verdict"] = "错归"
            wrong.append(row)
        else:
            row["verdict"] = "误增"
            extra.append(row)
        rows.append(row)
    missing = [{"formula": g["formula"], "herb": g["herb"]}
               for g in gold_herbs if (norm(g["formula"]), norm(g["herb"])) not in hits]
    return {"rows": rows, "hits": len(hits), "missing": missing,
            "wrong": wrong, "extra": extra, "total": len(gold_herbs),
            "extra_count": len(extra) + len(wrong)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--page-no", type=int, required=True)
    ap.add_argument("--annot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    page_id = "HOLDOUT-P%d" % args.page_no
    doc = page_ocr(args.image, page_id, args.out)
    pool = build_pool(doc, args.image)
    (args.out / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf8")

    res = _adopt.apply_rules(pool)
    for d in res["decisions"]:
        d["proposal_original"] = d["proposal"]
    corrected = _adopt.apply_to_ocr(doc, res["adopted"])
    (args.out / "decisions.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in res["decisions"]), encoding="utf8")
    (args.out / "corrected_ocr.json").write_text(json.dumps(corrected, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf8")

    blocks_a = [Block.from_dict(b) for b in doc["blocks"]]
    blocks_b = [Block.from_dict(b) for b in corrected["blocks"]]
    rel_a = relations_for(blocks_a)
    rel_b = relations_for(blocks_b)
    for tag, rel in (("A", rel_a), ("B", rel_b)):
        (args.out / ("relations_%s.json" % tag)).write_text(json.dumps(
            {"merged": len(rel["merged"]), "old_kept": rel["old_kept"],
             "segments": rel["extract"]["segments"], "pending": rel["extract"]["pending"],
             "triples": [t.to_dict() for t in rel["merged"]]}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf8")

    # ---- 评分阶段：此时才读人工标注 ----
    ann = json.loads(args.annot.read_text(encoding="utf8"))
    page = next(p for p in ann["pages"] if p["book"] == "思濟堂方書" and p["page_no"] == args.page_no)
    gold_herbs = []
    for inst in page["instances"]:
        for i in inst.get("ingredients", []):
            if i.get("display_status") != "应出表":
                continue
            gold_herbs.append({"page": args.page_no, "instance": inst["id"],
                               "formula": inst.get("name_verbatim") or "",
                               "herb": i["herb_verbatim"], "herb_bbox": i["herb_bbox"],
                               "role": i.get("role")})
    img = imread_unicode(str(args.image))
    gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8) if img.ndim == 3 else img
    names_a = score_names(blocks_a, gray, gold_herbs)
    names_b = score_names(blocks_b, gray, gold_herbs)
    comp_a = score_composition(rel_a["merged"], gold_herbs, gold_herbs)
    comp_b = score_composition(rel_b["merged"], gold_herbs, gold_herbs)

    result = {"page": args.page_no, "image": str(args.image.resolve()),
              "image_sha256": sha256_of(args.image), "annot_sha256": sha256_of(args.annot),
              "blocks": len(blocks_a),
              "names": {"A": names_a, "B": names_b},
              "composition": {"A": comp_a, "B": comp_b},
              "decisions": {"total": len(res["decisions"]),
                            "adopted": len(res["adopted"]),
                            "pending": sum(1 for d in res["decisions"] if d["status"] == "待核"),
                            "items": [{"line": d["line_id"], "span": d["span"],
                                       "from": d["original"], "to": d["proposal"],
                                       "status": d["status"], "rule": d["rule"],
                                       "reasons": d["reasons"], "bbox": d["bbox"]}
                                      for d in res["decisions"]]}}
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf8")
    print(json.dumps({"page": args.page_no, "blocks": len(blocks_a),
                      "names_A": names_a["strict_hits"], "names_B": names_b["strict_hits"],
                      "gold": names_a["total"],
                      "comp_A": comp_a["hits"], "comp_B": comp_b["hits"],
                      "extra_A": comp_a["extra_count"], "extra_B": comp_b["extra_count"],
                      "adopted": len(res["adopted"]), "pending": result["decisions"]["pending"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
