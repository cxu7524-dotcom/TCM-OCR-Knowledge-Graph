# -*- coding: utf-8 -*-
"""第 8 步：引药边界的直接检查（构造块集，覆盖四种情形）。

1. 「引加」位于块首 → 后续引药不生成基本组成
2. 「引加」位于块中 → 标记前的正常药味仍生成组成
3. 「不加引」结束基本配伍 → 后续服法文字不进入组成
4. 同页另一方的生薑属于基本配伍 → 仍能独立产出关系

构造块集只验证入口行为，不参与第 6 步口径的统计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                                  # noqa: E402
from extract.segment_composition import extract_composition     # noqa: E402
from ocr.normalize import normalize_text                        # noqa: E402


def mk(lid: str, text: str, order: int, bbox: list, kind: str = "paragraph") -> Block:
    return Block(page_id="CHECK8", bbox=bbox, line_id=lid, block_ref="CHECK8:%s" % lid,
                 text=text, text_norm=normalize_text(text), kind=kind, reading_order=order,
                 conf=1.0, source_hash="chk8_%s" % lid)


def summarize(res: dict) -> dict:
    return {"segments": [{"title": s["title"], "n": s["herb_count"],
                          "boundary": (s["boundary"] or {}).get("cue"), "text": s["text"]}
                         for s in res["segments"]],
            "relations": [(t.head, t.tail) for t in res["relations"]],
            "pending": res["pending"]}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/p51_step8_check"
    out.mkdir(parents=True, exist_ok=True)
    cases = []

    # 1. 引加在块首
    c1 = [mk("T1", "防風通聖散", 1, [960, 100, 1000, 180], kind="title"),
          mk("C1", "川芎當歸赤芍", 2, [900, 120, 950, 400]),
          mk("P1", "引加生薑二錢葱白二寸", 3, [850, 130, 890, 260])]
    r1 = extract_composition(c1)
    cases.append({"case": "1_引加位于块首", "input": [b.text for b in c1],
                  "expect": "3 味；边界=引加；无生姜/葱白",
                  "result": summarize(r1),
                  "passed": [t.tail for t in r1["relations"]] == ["川芎", "当归", "赤芍"]
                  and (r1["segments"][0]["boundary"] or {}).get("cue") == "引加"})

    # 2. 引加在块中
    c2 = [mk("T1", "防風通聖散川芎當歸赤芍引加生薑二錢", 1, [960, 100, 1000, 360], kind="title")]
    r2 = extract_composition(c2)
    cases.append({"case": "2_引加位于块中", "input": [b.text for b in c2],
                  "expect": "3 味（川芎/当归/赤芍）；引加之后的生薑不计",
                  "result": summarize(r2),
                  "passed": [t.tail for t in r2["relations"]] == ["川芎", "当归", "赤芍"]
                  and (r2["segments"][0]["boundary"] or {}).get("cue") == "引加"})

    # 3. 不加引
    c3 = [mk("T1", "防風通聖散川芎當歸不加引水二盞煎服", 1, [960, 100, 1000, 340], kind="title")]
    r3 = extract_composition(c3)
    cases.append({"case": "3_不加引结束配伍", "input": [b.text for b in c3],
                  "expect": "2 味；边界=不加引",
                  "result": summarize(r3),
                  "passed": [t.tail for t in r3["relations"]] == ["川芎", "当归"]
                  and (r3["segments"][0]["boundary"] or {}).get("cue") == "不加引"})

    # 4. 同页两方：一方的生薑是基本配伍，另一方的生薑是引药
    c4 = [mk("T1", "桂枝湯", 1, [960, 100, 1000, 160], kind="title"),
          mk("C1", "桂枝生姜甘草大枣", 2, [900, 120, 950, 420]),
          mk("P1", "水煎温服", 3, [850, 130, 890, 220]),
          mk("T2", "九味羌活湯", 4, [800, 100, 840, 220], kind="title"),
          mk("C2", "防風白芷川芎生地", 5, [740, 120, 790, 360]),
          mk("P2", "引加生薑二錢葱白二寸水煎熱服", 6, [690, 130, 730, 300])]
    r4 = extract_composition(c4)
    rel = [(t.head, t.tail) for t in r4["relations"]]
    cases.append({"case": "4_同页两方的生薑", "input": [b.text for b in c4],
                  "expect": "桂枝湯保留生姜；九味羌活湯不含生姜/葱白",
                  "result": summarize(r4),
                  "passed": ("桂枝湯", "生姜") in rel
                  and ("九味羌活湯", "生姜") not in rel
                  and ("九味羌活湯", "葱白") not in rel})

    report = {"cases": cases, "all_passed": all(c["passed"] for c in cases)}
    (out / "entry_checks_step8.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf8")
    print(json.dumps({"cases": [(c["case"], c["passed"]) for c in cases],
                      "all_passed": report["all_passed"]}, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
