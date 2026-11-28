# -*- coding: utf-8 -*-
"""第 7 步：邻字保护的单变量回归（冻结候选池，只重放采纳规则）。

唯一改动：tools/p51_step4_adopt.py 的 R6 从「左侧邻字」扩到「两侧邻字」。
OCR、候选池、阈值、词表、抽取器全部沿用第 6 步冻结产物，不重跑推理。

用法：
  python tools/p51_step7_guard.py --pool <pool.json> --ocr <ocr.json> --out <目录> --page 40
  python tools/p51_step7_guard.py --pool <pool.json> --ocr <ocr.json> --out <目录> --p51
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                        # noqa: E402
from extract.rules import extract_all_audited         # noqa: E402
from extract.segment_composition import extract_composition  # noqa: E402
from graph.build import build_triples                 # noqa: E402
from ocr.normalize import normalize_text              # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_adopt", ROOT / "tools/p51_step4_adopt.py")
_adopt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adopt)


def norm(s: str) -> str:
    return normalize_text(s or "").translate(str.maketrans({"殼": "壳", "羗": "羌"}))


def relations_for(blocks: list) -> list:
    ext = extract_composition(blocks)
    ents, cands, _a = extract_all_audited(blocks, is_demo=False)
    old = build_triples(ents, blocks, cands)
    heads = {norm(s["title"]) for s in ext["segments"]}
    kept = [t for t in old if not (t.relation == "组成" and norm(t.head) in heads)]
    return kept + ext["relations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--page", type=int)
    ap.add_argument("--p51", action="store_true")
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf8"))
    ocr = json.loads(args.ocr.read_text(encoding="utf8"))
    res = _adopt.apply_rules(pool)
    for d in res["decisions"]:
        d["proposal_original"] = d["proposal"]
    corrected = _adopt.apply_to_ocr(ocr, res["adopted"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decisions.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in res["decisions"]), encoding="utf8")
    (args.out / "corrected_ocr.json").write_text(json.dumps(corrected, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf8")

    blocks_a = [Block.from_dict(b) for b in ocr["blocks"]]
    blocks_b = [Block.from_dict(b) for b in corrected["blocks"]]
    rel_a, rel_b = relations_for(blocks_a), relations_for(blocks_b)

    if args.p51:
        spec = importlib.util.spec_from_file_location(
            "p51_verify", ROOT / "output/page51_acceptance/verify.py")
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        gold = json.loads((ROOT / "output/page51_acceptance/gold.json").read_text(encoding="utf8"))
        out = {}
        for tag, blocks, rel in (("A", blocks_a, rel_a), ("B", blocks_b, rel_b)):
            sc = verify.score(gold, blocks, rel)
            out[tag] = {"matched": sc["matched"], "fp": sc["false_positives"],
                        "missing": sc["missing"],
                        "tails": [m["tail"] for m in sc["matches"]]}
        payload = {"p51": out, "decisions": [
            {"line": d["line_id"], "span": d["span"], "from": d["original"],
             "to": d["proposal"], "status": d["status"], "rule": d["rule"],
             "guard": d.get("guard", [])} for d in res["decisions"]]}
        (args.out / "p51_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf8")
        print(json.dumps({"p51": {k: (v["matched"], v["fp"]) for k, v in out.items()},
                          "decisions": [(d["line_id"], d["original"], d["proposal"], d["status"])
                                        for d in res["decisions"]]}, ensure_ascii=False))
        return 0

    (args.out / "relations_A.json").write_text(json.dumps(
        {"triples": [t.to_dict() for t in rel_a]}, ensure_ascii=False) + "\n", encoding="utf8")
    (args.out / "relations_B.json").write_text(json.dumps(
        {"triples": [t.to_dict() for t in rel_b]}, ensure_ascii=False) + "\n", encoding="utf8")
    result = {"page": args.page, "image": pool["image"], "blocks": len(blocks_a),
              "decisions": {"total": len(res["decisions"]), "adopted": len(res["adopted"]),
                            "pending": sum(1 for d in res["decisions"] if d["status"] == "待核"),
                            "items": [{"line": d["line_id"], "span": d["span"],
                                       "from": d["original"], "to": d["proposal"],
                                       "status": d["status"], "rule": d["rule"],
                                       "reasons": d["reasons"], "bbox": d["bbox"],
                                       "guard": d.get("guard", [])} for d in res["decisions"]]}}
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf8")
    print(json.dumps({"page": args.page, "adopted": len(res["adopted"]),
                      "decisions": [(d["line_id"], d["original"], d["proposal"], d["status"])
                                    for d in res["decisions"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
