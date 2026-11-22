"""用真实 OCR 与留存关系重建候选，离线复验 37 页条目归属。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block
from extract.llm_validate import validate_candidates
from ocr.normalize import normalize_text


def source(loc):
    return {"block_ref": loc["block_ref"], "quote": loc["quote"], "start": loc["start"]}


def reconstruct():
    directory = ROOT / "output/llm_eval_step5_nothink"
    dataset = json.loads((directory / "fixed_dataset.json").read_text(encoding="utf8"))
    page = next(page for page in dataset["pages"] if page["number"] == 37)
    records = [json.loads(line) for line in (directory / "cloud_results.jsonl").read_text(encoding="utf8").splitlines()]
    record = next(record for record in records if record["page"] == 37)
    target = {"薄荷叶", "防风", "石斛", "木香"}
    triples = [t for t in record["predictions"]["triples"] if t["relation"] == "剂量" and t["head"] in target]
    if len(triples) != 4 or {t["head"] for t in triples} != target:
        raise ValueError("固定评测结果不包含四条目标关系")
    anchors = [e for e in record["predictions"]["entities"] if e["type"] == "方剂" and e["name"] == "四物汤"]
    if len(anchors) != 1 or any(t["entry_id"] != anchors[0]["entry_id"] for t in triples):
        raise ValueError("固定评测结果的锚点不一致")
    anchor = anchors[0]
    entities = [{"id": "anchor", "name": anchor["value"], "type": "方剂", "evidence": [source(x) for x in anchor["evidence"]]}]
    relations = []
    for index, triple in enumerate(triples):
        subject = triple["subject_evidence"][0]
        dose = triple["relation_evidence"][0]
        matches = [(i, j) for i in range(len(dose["quote"])) for j in range(i + 1, len(dose["quote"]) + 1)
                   if normalize_text(dose["quote"][i:j]) == normalize_text(triple["tail"])]
        if len(matches) != 1:
            raise ValueError("留存剂量无法唯一映射到原文证据")
        a, b = matches[0]
        dose_entity = {"block_ref": dose["block_ref"], "quote": dose["quote"][a:b], "start": dose["start"] + a}
        # 原始模型响应未保存，依据已留存的证据摘录重建临时候选。
        entities.append({"id": f"herb-{index}", "name": subject["quote"], "type": "药材", "evidence": [source(subject)]})
        entities.append({"id": f"dose-{index}", "name": dose_entity["quote"], "type": "剂量", "evidence": [dose_entity]})
        relations.append({"subject_id": f"herb-{index}", "object_id": f"dose-{index}", "entry_anchor_id": "anchor",
                          "type": "剂量", "qualifier": triple["qualifier"],
                          "subject_evidence": [source(x) for x in triple["subject_evidence"]],
                          "relation_evidence": [source(x) for x in triple["relation_evidence"]]})
    blocks = [Block.from_dict(block) for block in page["blocks"]]
    return blocks, {"error": "", "output": {"entities": entities, "relations": relations, "unresolved": []}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("legacy", "fixed"), default="fixed")
    args = parser.parse_args()
    blocks, candidates = reconstruct()
    result = validate_candidates(blocks, [b.block_ref for b in blocks], candidates, is_demo=False)
    accepted = [t for t in result.triples if t.head in {"薄荷叶", "防风", "石斛", "木香"} and t.relation == "剂量"]
    rejected = [u for u in result.unresolved if u.get("reason") == "continuation_entry_conflict"]
    print(json.dumps({"method": "真实 OCR 与留存关系重建候选", "accepted": len(accepted),
                      "scope_rejected": len(rejected), "rejected": rejected}, ensure_ascii=False))
    return 0 if (len(accepted) == 4 if args.expect == "legacy" else len(accepted) == 0 and len(rejected) == 4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
