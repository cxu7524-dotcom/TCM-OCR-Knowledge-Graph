"""Offline replay of reconstructed saved candidates; no model request and no DB writes."""
import copy
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block
from extract.llm_validate import validate_candidates
from ocr.normalize import normalize_text

OLD = ROOT / "output/page51_acceptance"
OUT = ROOT / "output/page51_repair_offline"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(name, obj):
    with (OUT / name).open("x", encoding="utf8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)


def match(quote, name):
    matches = [(i, j) for i in range(len(quote)) for j in range(i + 1, len(quote) + 1)
               if normalize_text(quote[i:j]) == normalize_text(name)]
    if len(matches) != 1:
        raise ValueError("saved evidence cannot be uniquely mapped")
    return matches[0]


def source(loc):
    return {k: loc[k] for k in ("block_ref", "quote", "start")}


def reconstruct(compositions):
    entities, relations = [], []
    for i, triple in enumerate(compositions):
        subject, target = triple["subject_evidence"][0], triple["relation_evidence"][0]
        a, b = match(subject["quote"], triple["head"])
        c, d = match(target["quote"], triple["tail"])
        anchor = {"block_ref": subject["block_ref"], "quote": subject["quote"][a:b], "start": subject["start"] + a}
        herb = {"block_ref": target["block_ref"], "quote": target["quote"][c:d], "start": target["start"] + c}
        f, h = f"f{i}", f"h{i}"
        entities.extend([{"id": f, "name": anchor["quote"], "type": "方剂", "evidence": [anchor]},
                         {"id": h, "name": herb["quote"], "type": "药材", "evidence": [herb]}])
        relations.append({"subject_id": f, "object_id": h, "entry_anchor_id": f, "type": "组成",
                          "subject_evidence": [source(x) for x in triple["subject_evidence"]],
                          "relation_evidence": [source(x) for x in triple["relation_evidence"]],
                          "qualifier": triple["qualifier"]})
    return {"entities": entities, "relations": relations, "unresolved": []}


def replay(page, candidates, scorer, gold):
    blocks = [Block.from_dict(item) for item in page["blocks"]]
    result = validate_candidates(blocks, [b.block_ref for b in blocks], {"error": "", "output": candidates}, is_demo=False)
    return {"method": "重建候选经生产 validate_candidates；非原始模型完整响应、未重新调用模型",
            "candidate_relations": len(candidates["relations"]), "accepted_relations": len(result.triples),
            "status": result.status, "error": result.error, "score": scorer(gold, blocks, result.triples),
            "composition": [t.to_dict() for t in result.triples], "unresolved": result.unresolved}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=OUT)
    OUT = parser.parse_args().dir
    page, gold, saved = read(OLD / "ocr.json"), read(OLD / "gold.json"), read(OLD / "cloud.json")
    spec = importlib.util.spec_from_file_location("page51_original_verifier", OLD / "verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    image = ROOT / gold["image"]
    if hashlib.sha256(image.read_bytes()).hexdigest() != gold["image_sha256"]:
        raise ValueError("原图指纹不匹配")
    candidates = reconstruct(saved["composition"])
    baseline = replay(page, candidates, verifier.score, gold)
    # Gold-derived corrections are an explicit offline control, never a model input.
    edits = [("R020", "柴朝", "柴胡"), ("R019", "川芦", "川芎"), ("R019", "茯芬", "茯苓")]
    corrected, controlled = copy.deepcopy(page), copy.deepcopy(candidates)
    trace = []
    for line, before, after in edits:
        block = next(b for b in corrected["blocks"] if b["line_id"] == line)
        start = block["text"].index(before)
        trace.append({"block_ref": block["block_ref"], "start": start, "original": before, "corrected": after,
                      "source": "用户提供的验收金标准，仅用于离线受控对照，非自动识别或独立图像核验"})
        block["text"] = block["text"].replace(before, after, 1)
        block["quote"] = block["text"]
        block["text_norm"] = normalize_text(block["text"])
        for entity in controlled["entities"]:
            if entity["type"] == "药材" and entity["name"] == before and entity["evidence"][0]["block_ref"] == block["block_ref"]:
                entity["name"] = after
                entity["evidence"][0]["quote"] = after
        for rel in controlled["relations"]:
            for ev in rel["relation_evidence"]:
                if ev["block_ref"] == block["block_ref"] and before in ev["quote"]:
                    ev["quote"] = ev["quote"].replace(before, after)
                    rel["qualifier"] = "受控人工金标准校正，来源见独立修订记录"
    control = replay(corrected, controlled, verifier.score, gold)
    OUT.mkdir(parents=True, exist_ok=False)
    write("raw_ocr.json", page)
    write("gold_separate.json", gold)
    write("reconstructed_candidates.json", candidates)
    write("baseline_replay.json", baseline)
    write("controlled_corrected_ocr.json", corrected)
    write("controlled_corrections.json", trace)
    write("controlled_candidates.json", controlled)
    write("controlled_replay.json", control)
    print(json.dumps({"baseline_score": baseline["score"], "controlled_score": control["score"],
                      "baseline_reasons": [u["reason"] for u in baseline["unresolved"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
