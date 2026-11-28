# -*- coding: utf-8 -*-
"""第 5 步收口：通用入口的直接检查（含构造用例与真实开发片段）。

覆盖两种本批开发页里不存在的排版：
  A. 方题与配伍同块 —— 垂直排版的方名独立成列，本批 6 个开发页 0 例，
     故用构造块集直接调用入口；
  B. 同页两个方剂各自归属 —— 构造块集 + 真实开发页 32（两个方题）。

构造块集只用于「入口行为」的契约检查，不参与第 51 页统计。

用法：
  python tools/p51_step5c_entry_checks.py --pool32 <pool_p32.json> --out <目录>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block                      # noqa: E402
from extract.segment_composition import extract_composition  # noqa: E402
from ocr.normalize import normalize_text                      # noqa: E402


def mk(line_id: str, text: str, order: int, bbox: list, kind: str = "paragraph") -> Block:
    return Block(page_id="CHECK", bbox=bbox, line_id=line_id, block_ref="CHECK:%s" % line_id,
                 text=text, text_norm=normalize_text(text), kind=kind, reading_order=order, conf=1.0,
                 source_hash="chk_%s" % line_id)


def summarize(res: dict) -> dict:
    return {"segments": [{"title": s["title"], "herbs": s["herb_blocks"], "n": s["herb_count"],
                          "boundary": (s["boundary"] or {}).get("cue"),
                          "next_title": s["next_title"], "text": s["text"]}
                         for s in res["segments"]],
            "relations": [{"head": t.head, "tail": t.tail} for t in res["relations"]],
            "pending": res["pending"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool32", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"cases": []}

    # A. 方题与配伍同块
    case_a = [
        mk("T1", "防風通聖散川芎當歸赤芍", 1, [900, 100, 960, 320], kind="title"),
        mk("T2", "水二盞煎服", 2, [700, 120, 760, 240]),
    ]
    res_a = extract_composition(case_a)
    a_ok = (len(res_a["segments"]) == 1
            and res_a["segments"][0]["title"] == "防風通聖散"
            and [t.tail for t in res_a["relations"]] == ["川芎", "当归", "赤芍"]
            and (res_a["segments"][0]["boundary"] or {}).get("cue", "").startswith("水二盞"))
    report["cases"].append({"case": "A_方题与配伍同块", "kind": "构造块集",
                            "input": [b.text for b in case_a], "expect": "1 段、方题防風通聖散、三味药、遇水二盞煎服止",
                            "result": summarize(res_a), "passed": bool(a_ok)})

    # B. 同页两个方剂各自归属
    case_b = [
        mk("T1", "防風通聖散", 1, [960, 100, 1000, 180], kind="title"),
        mk("C1", "川芎當歸赤芍", 2, [900, 120, 950, 400]),
        mk("P1", "水二盞煎服", 3, [850, 130, 890, 240]),
        mk("T2", "通關散", 4, [800, 100, 840, 160], kind="title"),
        mk("C2", "天南星生半夏甘草", 5, [740, 120, 790, 420]),
        mk("P2", "共爲細末", 6, [690, 130, 730, 200]),
    ]
    res_b = extract_composition(case_b)
    heads = [t.head for t in res_b["relations"]]
    b_ok = (len(res_b["segments"]) == 2
            and heads.count("防風通聖散") == 3 and heads.count("通關散") == 3
            and {t.tail for t in res_b["relations"] if t.head == "通關散"} == {"天南星", "生半夏", "甘草"})
    report["cases"].append({"case": "B_同页两方各自归属", "kind": "构造块集",
                            "input": [b.text for b in case_b], "expect": "2 段，各 3 味，方名不串",
                            "result": summarize(res_b), "passed": bool(b_ok)})

    # C. 真实开发页 32（两个方题）
    pool = json.loads(args.pool32.read_text(encoding="utf8"))
    blocks = [Block.from_dict(b) for b in pool["blocks"]]
    for b in blocks:
        if not b.page_id:
            b.page_id = pool.get("page_id", "")
    res_c = extract_composition(blocks)
    c_ok = (len(res_c["segments"]) == 2
            and {s["title"] for s in res_c["segments"]} == {"防風通聖散", "通關散"}
            and all(t.head in {"防風通聖散", "通關散"} for t in res_c["relations"]))
    report["cases"].append({"case": "C_真实开发页32", "kind": "真实片段",
                            "input": str(args.pool32), "expect": "2 段，方名各自归属",
                            "result": summarize(res_c), "passed": bool(c_ok)})

    report["all_passed"] = all(c["passed"] for c in report["cases"])
    (args.out / "entry_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf8")
    print(json.dumps({"cases": [(c["case"], c["passed"]) for c in report["cases"]],
                      "all_passed": report["all_passed"]}, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
