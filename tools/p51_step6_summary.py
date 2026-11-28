# -*- coding: utf-8 -*-
"""第 6 步：五页留出结果汇总（药名 / 组成 / 修订 / 误改）。

用法：
  python tools/p51_step6_summary.py --dirs <p37目录> <p39目录> ... --out <汇总.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pages, tot = [], {"gold": 0, "names_A": 0, "names_B": 0, "names_A_approx": 0,
                      "names_B_approx": 0, "comp_A": 0, "comp_B": 0,
                      "extra_A": 0, "extra_B": 0, "adopted": 0, "pending": 0,
                      "decisions": 0, "fixed": 0, "broken": 0}
    for d in args.dirs:
        r = json.loads((d / "result.json").read_text(encoding="utf8"))
        names = r["names2"] if "names2" in r else r["names"]
        comp = r["composition2"] if "composition2" in r else r["composition"]
        a, b = names["A"], names["B"]
        ca, cb = comp["A"], comp["B"]
        fixed, broken = [], []
        for ra, rb in zip(a["rows"], b["rows"]):
            ka = ra.get("hit", ra.get("strict"))
            kb = rb.get("hit", rb.get("strict"))
            if not ka and kb:
                fixed.append(rb["herb"])
            if ka and not kb:
                broken.append(rb["herb"])
        page = {"page": r["page"], "blocks": r["blocks"],
                "gold": a["total"],
                "names_A": a["hits"], "names_B": b["hits"],
                "comp_A": ca["hits"], "comp_B": cb["hits"],
                "extra_A": len(ca["extra"]), "extra_B": len(cb["extra"]),
                "missing_A": ca["missing"], "missing_B": cb["missing"],
                "decisions": r["decisions"]["total"], "adopted": r["decisions"]["adopted"],
                "pending": r["decisions"]["pending"],
                "fixed": fixed, "broken": broken,
                "adopted_items": [x for x in r["decisions"]["items"] if x["status"] == "自动采纳"],
                "pending_items": [x for x in r["decisions"]["items"] if x["status"] == "待核"],
                "comp_extra_B": [(x["head"], x["tail"], x["verdict"]) for x in cb["extra"]],
                "comp_extra_A": [(x["head"], x["tail"], x["verdict"]) for x in ca["extra"]],
                "dir": str(d)}
        pages.append(page)
        tot["gold"] += page["gold"]
        tot["names_A"] += page["names_A"]
        tot["names_B"] += page["names_B"]
        tot["comp_A"] += page["comp_A"]
        tot["comp_B"] += page["comp_B"]
        tot["extra_A"] += page["extra_A"]
        tot["extra_B"] += page["extra_B"]
        tot["adopted"] += page["adopted"]
        tot["pending"] += page["pending"]
        tot["decisions"] += page["decisions"]
        tot["fixed"] += len(fixed)
        tot["broken"] += len(broken)

    summary = {"pages": pages, "totals": tot,
               "verdict": {
                   "names_B_gt_A": tot["names_B"] > tot["names_A"],
                   "no_broken": tot["broken"] == 0,
                   "comp_B_ge_A": tot["comp_B"] >= tot["comp_A"],
                   "extra_B_le_A": tot["extra_B"] <= tot["extra_A"]}}
    summary["passed"] = all(summary["verdict"].values())
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(json.dumps({"totals": tot, "verdict": summary["verdict"],
                      "passed": summary["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
