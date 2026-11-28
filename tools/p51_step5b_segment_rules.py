# -*- coding: utf-8 -*-
"""第 5 步补测：配伍段逐味抽取（实验路径，可切换）。

与原规则的区别只有一处：在**已确认方剂的配伍段内**按顺序逐味生成「组成」，
允许同块多味、允许共用剂量、允许无独立剂量；原规则的剂量配对逻辑保留不动，
但该方已确认条目的「组成」由本路径接管，避免两套组成候选叠加。

范围定位复用 extract/formula_segments.py 的 _Stream（原文↔归一化映射）与 _starts（方题定位），
边界识别补一条通用的制法/服法规则——现有实现的 _STOP 认不出「水搗碎温服」这类无剂量的制法，
导致配伍段一路吞到方论。

用法：
  python tools/p51_step5b_segment_rules.py run --ocr <ocr.json> --label B --out <目录>
  python tools/p51_step5b_segment_rules.py inspect --ocr <ocr.json>          # 只看范围与逐味
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                                    # noqa: E402
from extract.rules import extract_all_audited                     # noqa: E402
from extract.formula_segments import _Stream, _starts, _herb_hits  # noqa: E402
from extract.schema import HERBS                                  # noqa: E402
from graph.build import build_triples                             # noqa: E402
from ocr.normalize import normalize_text                          # noqa: E402

ACCEPT = ROOT / "output/page51_acceptance"
FORMULA = "人參敗毒散"
EQUIV = {"殼": "壳", "羗": "羌"}

# 通用制法/服法边界：无剂量的「水搗碎温服」也要认
PREP = re.compile(
    r"^(?:"
    r"(?:水|酒|米湯|米汤|薑湯|姜汤|童便|井水|甘瀾水|甘澜水|東流水|东流水)[^钱錢]{0,10}(?:搗|捣|研|磨|煎|煮|熬|調|调|服)"
    r"|(?:搗|捣)碎|(?:研|磨)(?:[为為爲])?末|(?:[为為爲])末"
    r"|每服|空心服|食[前后]服|(?:温|溫|熱|热|冷|凉|涼)服|煎服|(?:水|酒)?煎[^钱錢]{0,4}服"
    r"|(?:右|上)[为為爲]|(?:右|上)(?:件|味)|共[为為爲捣搗]"
    r")")

CANON = {}
for h in HERBS:
    CANON.setdefault(normalize_text(h).translate(str.maketrans(EQUIV)), h)


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_verify():
    spec = importlib.util.spec_from_file_location("p51_verify", ACCEPT / "verify.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def locate(stream: _Stream, formula: str = FORMULA) -> dict:
    """方题 → 配伍段起止。返回标题块、范围、边界证据。"""
    starts = _starts(stream)
    target = normalize_text(formula).translate(str.maketrans(EQUIV))
    hit = next((s for s in starts if s["name_norm"] == target), None)
    if hit is None:
        raise SystemExit("未定位到方题：%s（starts=%s）" % (formula, [s["name_norm"] for s in starts]))
    begin = hit["end"]
    # 从方题之后逐块前进，遇到制法/服法即止
    parts = [p for p in stream.parts if p["end"] > begin]
    end, boundary = len(stream.norm), None
    for p in parts:
        text = stream.norm[p["start"]:p["end"]].strip()
        if not text:
            continue
        if PREP.match(text):
            end = p["start"]
            boundary = {"line_id": p["block"].line_id, "block_ref": p["block"].block_ref,
                        "type": "制法/服法", "cue": text[:12], "stage": "段内首块即制法"}
            break
    # 兜底：段内若出现过早的 STO P 触发点，也在这里记录
    return {"title": stream.source(hit["start"], hit["end"]), "begin": begin, "end": end,
            "boundary": boundary, "parts": [p for p in parts if p["start"] >= begin and p["end"] <= end]}


def scan_herbs(stream: _Stream, begin: int, end: int, parts: list) -> list:
    """段内按顺序取互不重叠的药名，同位置优先完整长词。"""
    out = []
    for p in parts:
        a, b = max(begin, p["start"]), min(end, p["end"])
        if a >= b:
            continue
        norm = stream.norm[a:b]
        align = p["alignment"][a - p["start"]:b - p["start"]]
        i = 0
        while i < len(norm):
            cand = None
            for L in (4, 3, 2):
                if i + L > len(norm):
                    continue
                key = norm[i:i + L].translate(str.maketrans(EQUIV))
                if key in CANON:
                    cand = (L, CANON[key], key)
                    break
            if cand is None:
                i += 1
                continue
            L, herb, key = cand
            ra, rb = align[i][0], align[i + L - 1][1]
            raw = p["block"].text[ra:rb]
            out.append({"herb": herb, "surface": raw, "line_id": p["block"].line_id,
                        "block_ref": p["block"].block_ref, "bbox": list(p["block"].bbox),
                        "start": ra, "end": rb, "ctx": p["block"].text,
                        "slice_ok": p["block"].text[ra:rb] == raw})
            i += L
    return out


def compose_triples(blocks: list, old_triples: list, seg: dict, herbs: list, formula: str) -> list:
    """接管本方「组成」：删掉原规则由剂量配对派生的组成，换成逐味关系；其余保留。"""
    from ocr.normalize import normalize_text as nt
    keep = []
    for t in old_triples:
        if t.relation == "组成" and nt(t.head).translate(str.maketrans(EQUIV)) == \
                nt(formula).translate(str.maketrans(EQUIV)):
            continue
        keep.append(t)
    from app.schemas import Triple
    new = []
    for h in herbs:
        ev = [{"block_ref": h["block_ref"], "line_id": h["line_id"], "bbox": h["bbox"],
               "quote": h["ctx"], "start": h["start"], "end": h["end"],
               "surface": h["surface"], "mapping_precision": "exact"}]
        new.append(Triple(head=formula, relation="组成", tail=h["herb"],
                          quote=h["ctx"], evidence_refs=[h["block_ref"]],
                          line_id=h["line_id"], page_id=blocks[0].page_id,
                          bbox=h["bbox"], source_hash="", is_demo=False,
                          subject_rule="segment_scan", relation_evidence=ev,
                          condition_evidence=[]))
    return keep + new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "inspect"))
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--label", default="?")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    page = json.loads(args.ocr.read_text(encoding="utf8"))
    blocks = [Block.from_dict(b) for b in page["blocks"]]
    stream = _Stream(blocks)
    seg = locate(stream)
    herbs = scan_herbs(stream, seg["begin"], seg["end"], seg["parts"])

    if args.cmd == "inspect":
        print(json.dumps({"label": args.label, "title": seg["title"]["text"],
                          "segment_text": stream.norm[seg["begin"]:seg["end"]],
                          "boundary": seg["boundary"],
                          "blocks": [p["block"].line_id for p in seg["parts"]],
                          "herbs": [(h["herb"], h["surface"], h["line_id"], h["start"], h["end"])
                                    for h in herbs]}, ensure_ascii=False, indent=1))
        return 0

    if not args.out:
        raise SystemExit("run 需要 --out")
    ents, cands, audits = extract_all_audited(blocks, is_demo=False)
    old = build_triples(ents, blocks, cands)
    new = compose_triples(blocks, old, seg, herbs, FORMULA)

    verify = load_verify()
    gold = json.loads((ACCEPT / "gold.json").read_text(encoding="utf8"))
    result = verify.score(gold, blocks, new)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf8")
    (args.out / "triples.json").write_text(json.dumps(
        {"n": len(new), "triples": [t.to_dict() for t in new]}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf8")
    (args.out / "segment.json").write_text(json.dumps(
        {"title": seg["title"], "begin": seg["begin"], "end": seg["end"],
         "segment_norm": stream.norm[seg["begin"]:seg["end"]],
         "segment_raw": "".join(p["block"].text for p in seg["parts"]),
         "boundary": seg["boundary"],
         "blocks": [{"line_id": p["block"].line_id, "block_ref": p["block"].block_ref,
                     "bbox": list(p["block"].bbox),
                     "text": p["block"].text} for p in seg["parts"]],
         "herbs": herbs}, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (args.out / "run_config.json").write_text(json.dumps(
        {"label": args.label, "ocr": str(args.ocr.resolve()), "ocr_sha256": sha256_of(args.ocr),
         "backend": "local_rules + segment_scan",
         "change": "配伍段内逐味生成组成；本方组成由本路径接管，原剂量与其他关系保留",
         "formula": FORMULA, "gold": str((ACCEPT / "gold.json").resolve()),
         "score_fn": str((ACCEPT / "verify.py").resolve()) + "::score",
         "n_old_triples": len(old), "n_new_triples": len(new),
         "n_segment_herbs": len(herbs)}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    print(json.dumps({"label": args.label, "matched": result["matched"], "fp": result["false_positives"],
                      "missing": result["missing"], "passed": result["passed"],
                      "segment_blocks": [p["block"].line_id for p in seg["parts"]],
                      "boundary": seg["boundary"]["cue"] if seg["boundary"] else None,
                      "herbs": [h["herb"] for h in herbs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
