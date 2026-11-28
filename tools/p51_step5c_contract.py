# -*- coding: utf-8 -*-
"""第 5 步收口：通用入口 + 证据契约校验 + 第 51 页评分。

抽取（extract/segment_composition.py）与评分严格分开：
本工具先调用通用入口产出关系并落盘，随后才读取第 51 页金标与 verify.score 评分。

用法：
  python tools/p51_step5c_contract.py --ocr <ocr.json> --label A --out <目录>
  python tools/p51_step5c_contract.py --blocks <pool.json> --label DEV-P32 --out <目录> --no-score
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
from app.schemas import Block, source_hash_of          # noqa: E402
from extract.rules import extract_all_audited          # noqa: E402
from extract.segment_composition import extract_composition  # noqa: E402
from graph.build import build_triples                  # noqa: E402
from ocr.normalize import normalize_text               # noqa: E402

ACCEPT = ROOT / "output/page51_acceptance"
EQUIV = {"殼": "壳", "羗": "羌"}


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s: str) -> str:
    return normalize_text(s).translate(str.maketrans(EQUIV))


def load_verify():
    spec = importlib.util.spec_from_file_location("p51_verify", ACCEPT / "verify.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_evidence(triples: list, block_by_ref: dict) -> list:
    """逐条校验：block_ref 存在、block.text[start:end] == quote、Triple.validate()。"""
    rows = []
    for t in triples:
        problems = []
        subj = t.subject_evidence[0] if t.subject_evidence else None
        rel = t.relation_evidence[0] if t.relation_evidence else None
        if not subj:
            problems.append("缺 subject_evidence")
        else:
            b = block_by_ref.get(subj.get("block_ref"))
            if b is None:
                problems.append("subject block_ref 不存在")
            elif b.text[subj["start"]:subj["end"]] != subj["quote"]:
                problems.append("subject 引文与切片不符")
        if not rel:
            problems.append("缺 relation_evidence")
        else:
            b = block_by_ref.get(rel.get("block_ref"))
            if b is None:
                problems.append("relation block_ref 不存在")
            elif b.text[rel["start"]:rel["end"]] != rel["quote"]:
                problems.append("relation 引文与切片不符")
            if not rel.get("source_hash"):
                problems.append("relation 缺 source_hash")
        if not t.source_hash:
            problems.append("三元组缺 source_hash")
        problems.extend(t.validate())
        rows.append({"head": t.head, "tail": t.tail,
                     "subject": {"block_ref": subj.get("block_ref"), "quote": subj.get("quote"),
                                 "range": [subj.get("start"), subj.get("end")]} if subj else None,
                     "relation": {"block_ref": rel.get("block_ref"), "quote": rel.get("quote"),
                                  "range": [rel.get("start"), rel.get("end")],
                                  "surface": rel.get("surface")} if rel else None,
                     "triple_source_hash": t.source_hash,
                     "ok": not problems, "problems": problems})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=Path)
    ap.add_argument("--blocks", type=Path, help="开发页 pool.json（取其中 blocks）")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    if args.ocr:
        page = json.loads(args.ocr.read_text(encoding="utf8"))
        blocks = [Block.from_dict(b) for b in page["blocks"]]
        src = {"ocr": str(args.ocr.resolve()), "sha256": sha256_of(args.ocr)}
    else:
        pool = json.loads(args.blocks.read_text(encoding="utf8"))
        blocks = [Block.from_dict(b) for b in pool["blocks"]]
        for b in blocks:                      # 开发页 pool 未存 page_id，补上以便溯源与契约校验
            if not b.page_id:
                b.page_id = pool.get("page_id", "")
        src = {"pool": str(args.blocks.resolve()), "sha256": sha256_of(args.blocks)}

    # ① 抽取（只用 OCR 块）
    ext = extract_composition(blocks)
    block_by_ref = {b.block_ref or b.line_id: b for b in blocks}
    checks = check_evidence(ext["relations"], block_by_ref)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "segments.json").write_text(json.dumps(
        {"n": len(ext["segments"]), "segments": ext["segments"]}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf8")
    (args.out / "pending.json").write_text(json.dumps(
        {"n": len(ext["pending"]), "items": ext["pending"]}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf8")
    (args.out / "relations.json").write_text(json.dumps(
        {"n": len(ext["relations"]), "triples": [t.to_dict() for t in ext["relations"]]},
        ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (args.out / "contract.json").write_text(json.dumps(
        {"n": len(checks), "passed": sum(1 for c in checks if c["ok"]), "rows": checks},
        ensure_ascii=False, indent=1) + "\n", encoding="utf8")

    result = {"label": args.label, "segments": len(ext["segments"]),
              "relations": len(ext["relations"]),
              "contract_passed": sum(1 for c in checks if c["ok"]),
              "pending": len(ext["pending"])}

    if not args.no_score:
        # ② 与原规则合并：本方组成由本入口接管，原剂量与其他关系保留
        ents, cands, _audits = extract_all_audited(blocks, is_demo=False)
        old = build_triples(ents, blocks, cands)
        heads = {norm(s["title"]) for s in ext["segments"]}
        kept = [t for t in old if not (t.relation == "组成" and norm(t.head) in heads)]
        merged = kept + ext["relations"]
        verify = load_verify()
        gold = json.loads((ACCEPT / "gold.json").read_text(encoding="utf8"))
        score = verify.score(gold, blocks, merged)
        (args.out / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf8")
        (args.out / "merged_triples.json").write_text(json.dumps(
            {"n": len(merged), "old_kept": len(kept), "new": len(ext["relations"]),
             "triples": [t.to_dict() for t in merged]}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf8")
        result.update({"matched": score["matched"], "fp": score["false_positives"],
                       "missing": score["missing"]})

    (args.out / "run_config.json").write_text(json.dumps(
        {"label": args.label, "input": src, "entry": "extract.segment_composition.extract_composition",
         "backend": "local_rules + segment_scan", "blocks": len(blocks),
         "score_fn": (str((ACCEPT / "verify.py").resolve()) + "::score") if not args.no_score else None,
         "gold": (str((ACCEPT / "gold.json").resolve()) if not args.no_score else None)},
        ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
