# -*- coding: utf-8 -*-
"""第 8 步：冻结 OCR 上的抽取重跑（只重放抽取器，不重跑 OCR/推理/采纳）。

输入用第 7 步保存的 ocr.json（A）与 corrected_ocr.json（B），
关系由新的 extract/segment_composition.py 重新生成，决策沿用第 7 步。

用法：
  python tools/p51_step8_regress.py --dir output/.../p50 --page 50 --out output/.../p50
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block  # noqa: E402

_spec = importlib.util.spec_from_file_location("p51_g7", ROOT / "tools/p51_step7_guard.py")
_g7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ocr = json.loads((args.dir / "ocr.json").read_text(encoding="utf8"))
    fixed = json.loads((args.dir / "corrected_ocr.json").read_text(encoding="utf8"))
    decisions = [json.loads(l) for l in (args.dir / "decisions.jsonl").open(encoding="utf8")]

    for name in ("ocr.json", "corrected_ocr.json"):
        (args.out / name).write_text((args.dir / name).read_text(encoding="utf8"), encoding="utf8")
    (args.out / "decisions.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf8")

    rel_a = _g7.relations_for([Block.from_dict(b) for b in ocr["blocks"]])
    rel_b = _g7.relations_for([Block.from_dict(b) for b in fixed["blocks"]])
    for tag, rel in (("A", rel_a), ("B", rel_b)):
        (args.out / ("relations_%s.json" % tag)).write_text(json.dumps(
            {"triples": [t.to_dict() for t in rel]}, ensure_ascii=False) + "\n", encoding="utf8")

    result = {"page": args.page, "image": ocr.get("image_path", ""),
              "blocks": len(ocr["blocks"]),
              "decisions": {"total": len(decisions),
                            "adopted": sum(1 for d in decisions if d["status"] == "自动采纳"),
                            "pending": sum(1 for d in decisions if d["status"] == "待核"),
                            "items": decisions}}
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf8")
    print(json.dumps({"page": args.page, "A": len(rel_a), "B": len(rel_b)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
