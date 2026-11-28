# -*- coding: utf-8 -*-
"""第 5 步：同一冻结抽取流程分别跑 A（原始 OCR）与 B（第 4 步修订后 OCR）。

后端：本地规则（extract.rules.extract_all_audited + graph.build.build_triples）。
唯一变量：送入抽取器的 OCR 文本版本。

评分复用 output/page51_acceptance/verify.py 的 score() 与同目录 gold.json。

用法：
  python tools/p51_step5_reextract.py --ocr <ocr.json> --label A --out <目录>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                    # noqa: E402
from extract.rules import extract_all_audited    # noqa: E402
from graph.build import build_triples            # noqa: E402

ACCEPT = ROOT / "output/page51_acceptance"


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_verify():
    spec = importlib.util.spec_from_file_location("p51_verify", ACCEPT / "verify.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--label", required=True, choices=("A", "B"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    page = json.loads(args.ocr.read_text(encoding="utf8"))
    blocks = [Block.from_dict(b) for b in page["blocks"]]
    ents, cands, audits = extract_all_audited(blocks, is_demo=False)
    triples = build_triples(ents, blocks, cands)

    verify = load_verify()
    gold = json.loads((ACCEPT / "gold.json").read_text(encoding="utf8"))
    result = verify.score(gold, blocks, triples)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps(
        {"n": len(cands), "entities": [e.to_dict() for e in ents],
         "triples": [t.to_dict() for t in cands]}, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (args.out / "triples.json").write_text(json.dumps(
        {"n": len(triples), "triples": [t.to_dict() for t in triples]},
        ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (args.out / "audits.json").write_text(json.dumps(
        {"n": len(audits), "items": audits}, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (args.out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf8")
    cfg = {"label": args.label, "ocr": str(args.ocr.resolve()), "ocr_sha256": sha256_of(args.ocr),
           "backend": "local_rules",
           "backend_detail": "extract.rules.extract_all_audited(is_demo=False) + graph.build.build_triples",
           "gold": str((ACCEPT / "gold.json").resolve()),
           "score_fn": str((ACCEPT / "verify.py").resolve()) + "::score",
           "blocks": len(blocks), "block_order": [b.block_ref for b in blocks],
           "n_candidates": len(cands), "n_triples": len(triples), "n_audits": len(audits)}
    (args.out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf8")

    # 两处校正的定位核对：引文必须能在本组文本里逐字命中
    text_by_ref = {b.block_ref: b.text for b in blocks}
    focus = {}
    for t in triples:
        if t.relation != "组成" or t.tail not in ("柴胡", "川芎", "茯苓", "羗活", "羌活"):
            continue
        evid = t.relation_evidence or [{"block_ref": b.block_ref, "quote": t.quote}
                                       for b in blocks if b.line_id == t.line_id]
        rows = []
        for e in evid:
            ref = e.get("block_ref", "")
            q = e.get("quote", "")
            txt = text_by_ref.get(ref, "")
            rows.append({"block_ref": ref, "quote": q, "offset": txt.find(q),
                         "quote_in_text": q in txt})
        focus.setdefault(t.tail, []).append({"head": t.head, "relation": t.relation,
                                             "tail": t.tail, "qualifier": t.qualifier,
                                             "evidence": rows})
    (args.out / "focus_relations.json").write_text(json.dumps(focus, ensure_ascii=False, indent=1) + "\n",
                                                   encoding="utf8")

    print(json.dumps({"label": args.label, "passed": result["passed"],
                      "matched": result["matched"], "expected": result["expected"],
                      "fp": result["false_positives"], "missing": result["missing"],
                      "candidates": len(cands), "triples": len(triples), "audits": len(audits),
                      "focus": sorted(focus)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
