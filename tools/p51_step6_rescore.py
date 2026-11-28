# -*- coding: utf-8 -*-
"""第 6 步：重算评分（不重跑 OCR/推理）。

原判据要求「按字框切出的切片 == 药名」，但留出页的字框普遍偏宽（含剂量字或邻字），
会把本来正确的识别判错。改为位置约束下的出现判据：

  药名命中 = 该药名在覆盖字框的块内出现，且出现位置与字框映射区间重叠（容许 ±1 字）

组成命中 = 方剂名相符 + 药名相符 + 证据块落在该方剂实例的配伍区域（quote_bbox）内。

用法：
  python tools/p51_step6_rescore.py --dirs <页目录...> --annot docs/...json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                          # noqa: E402
from ocr.normalize import normalize_text                # noqa: E402
from ocr.vertical import imread_unicode                 # noqa: E402
import importlib.util                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_adopt", ROOT / "tools/p51_step4_adopt.py")
_adopt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adopt)

EQUIV = {"殼": "壳", "羗": "羌"}
TRANS = str.maketrans(EQUIV)


def norm(s: str) -> str:
    return normalize_text(s or "").translate(TRANS)


def map_span(gray, block, y1, y2):
    boxes, method = _adopt.char_ranges(gray, {"bbox": list(block.bbox), "text": block.text})
    idx = [i for i, (a, b) in enumerate(boxes) if _adopt.overlap(a, b, y1, y2) >= 0.5]
    if not idx:
        return None, method
    return (idx[0], idx[-1] + 1), method


def hit_near(text: str, herb: str, span, tol: int = 1) -> dict:
    """药名是否出现在字框映射区间附近（±tol 字）。"""
    t, h = norm(text), norm(herb)
    if not h:
        return {"hit": False, "offset": None, "note": "空药名"}
    occ = []
    i = t.find(h)
    while i >= 0:
        occ.append(i)
        i = t.find(h, i + 1)
    if not occ:
        return {"hit": False, "offset": None, "note": "块内无该药名"}
    if span is None:
        return {"hit": False, "offset": occ[0], "note": "字框无法映射到块内位置"}
    s, e = span
    near = [i for i in occ if i < e + tol and i + len(h) > s - tol]
    if near:
        return {"hit": True, "offset": near[0], "note": "位置相符"}
    return {"hit": False, "offset": occ[0], "note": "块内有该药名但不在标注位置"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--annot", type=Path, required=True)
    args = ap.parse_args()
    ann = json.loads(args.annot.read_text(encoding="utf8"))

    for d in args.dirs:
        res = json.loads((d / "result.json").read_text(encoding="utf8"))
        page_no = res["page"]
        page = next(p for p in ann["pages"] if p["book"] == "思濟堂方書" and p["page_no"] == page_no)
        img = imread_unicode(res["image"])
        gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8) if img.ndim == 3 else img

        gold = []
        for inst in page["instances"]:
            for i in inst.get("ingredients", []):
                if i.get("display_status") != "应出表":
                    continue
                gold.append({"instance": inst["id"], "formula": inst.get("name_verbatim") or "",
                             "herb": i["herb_verbatim"], "herb_bbox": i["herb_bbox"],
                             "quote_bbox": inst.get("quote_bbox")})

        for tag, fname in (("A", "ocr.json"), ("B", "corrected_ocr.json")):
            blocks = [Block.from_dict(b) for b in json.loads((d / fname).read_text(encoding="utf8"))["blocks"]]
            rows = []
            for g in gold:
                hb = g["herb_bbox"]
                best, area = None, 0
                for b in blocks:
                    bx1, by1, bx2, by2 = b.bbox
                    ox = min(bx2, hb[2]) - max(bx1, hb[0])
                    oy = min(by2, hb[3]) - max(by1, hb[1])
                    if ox > 0 and oy > 0 and ox * oy > area:
                        best, area = b, ox * oy
                row = {"instance": g["instance"], "formula": g["formula"], "herb": g["herb"],
                       "bbox": hb, "line_id": best.line_id if best else None}
                if best is None:
                    row.update({"hit": False, "note": "无覆盖块"})
                else:
                    span, method = map_span(gray, best, hb[1], hb[3])
                    info = hit_near(best.text, g["herb"], span)
                    row.update({"hit": info["hit"], "offset": info["offset"], "note": info["note"],
                                "pos_method": method, "text": best.text})
                rows.append(row)
            res.setdefault("names2", {})[tag] = {
                "rows": rows, "hits": sum(1 for r in rows if r["hit"]), "total": len(rows)}

        # 组成：加入配伍区域约束
        for tag, fname in (("A", "relations_A.json"), ("B", "relations_B.json")):
            triples = json.loads((d / fname).read_text(encoding="utf8"))["triples"]
            rows, hits = [], set()
            for t in triples:
                if t["relation"] != "组成":
                    continue
                ev = (t.get("relation_evidence") or [{}])[0]
                bb = ev.get("bbox") or t.get("bbox") or [0, 0, 0, 0]
                cands = [g for g in gold if norm(g["formula"]) == norm(t["head"])
                         and norm(g["herb"]) == norm(t["tail"])]
                in_region = None
                for g in cands:
                    qb = g.get("quote_bbox") or g["herb_bbox"]
                    ox = min(bb[2], qb[2]) - max(bb[0], qb[0])
                    oy = min(bb[3], qb[3]) - max(bb[1], qb[1])
                    if ox > 0 and oy > 0:
                        in_region = g["instance"]
                        break
                page_herbs = {norm(g["herb"]) for g in gold}
                formulas = {norm(g["formula"]) for g in gold}
                if cands and in_region:
                    hits.add((norm(t["head"]), norm(t["tail"])))
                    verdict = "命中"
                elif cands:
                    verdict = "区域不符"
                elif norm(t["tail"]) in page_herbs:
                    verdict = "错归"
                elif norm(t["head"]) in formulas or True:
                    verdict = "误增"
                rows.append({"head": t["head"], "tail": t["tail"], "block_ref": ev.get("block_ref"),
                             "verdict": verdict, "in_region": in_region,
                             "quote": ev.get("quote"), "span": [ev.get("start"), ev.get("end")]})
            missing = [{"formula": g["formula"], "herb": g["herb"]} for g in gold
                       if (norm(g["formula"]), norm(g["herb"])) not in hits]
            res.setdefault("composition2", {})[tag] = {
                "rows": rows, "hits": len(hits), "total": len(gold), "missing": missing,
                "extra": [r for r in rows if r["verdict"] in ("误增", "错归", "区域不符")]}

        (d / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
        print(json.dumps({"page": page_no,
                          "names2": {k: v["hits"] for k, v in res["names2"].items()},
                          "comp2": {k: v["hits"] for k, v in res["composition2"].items()},
                          "extra2": {k: len(v["extra"]) for k, v in res["composition2"].items()},
                          "gold": len(gold)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
