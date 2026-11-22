"""新提示词的两页真实调用入口，结果与历史评测隔离。"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block
from extract.llm_extraction import PROMPT_VERSION, PROTOCOL_VERSION, extract_candidates
from extract.llm_local import LocalLLMExtractor
from extract.llm_validate import validate_candidates
from ocr.normalize import normalize_text

SOURCE = ROOT / "output/llm_eval_step5_nothink/fixed_dataset.json"
HISTORICAL = ROOT / "output/llm_eval_step5_nothink/cloud_results.jsonl"
SOURCE_SHA256 = "b762bf57ee73554f13cea81fb9c04ab7323f1bd4ea78f54fd96959a9da82e587"
DEFAULT = ROOT / "output/entry_boundary_v2"
MODEL = "DeepSeek-V4.1-Flash-1017"
BASE_URL = "https://www.autodl.art/api/v1"
PARAMETERS = {"response_format": "json_object", "temperature": 0.1, "max_tokens": 16384, "timeout": 180, "thinking": "disabled"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[redacted]") if secret else value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def relevant_refs(evidence, refs):
    return [item for item in evidence if item.get("block_ref") in refs]


def historical_wrong_count():
    records = [json.loads(line) for line in HISTORICAL.read_text(encoding="utf8").splitlines() if line.strip()]
    pages = [record for record in records if record.get("page") == 37]
    if len(pages) != 1:
        raise ValueError("历史第 37 页记录不唯一")
    predictions = pages[0]["predictions"]
    anchors = {e["entry_id"] for e in predictions["entities"]
               if e["type"] == "方剂" and normalize_text(e["name"]) == "四物汤"}
    names = {"薄荷叶", "防风", "石斛", "木香"}
    wrong = [t for t in predictions["triples"] if t["relation"] == "剂量"
             and normalize_text(t["head"]) in names and t["entry_id"] in anchors
             and any(e.get("line_id") in {"R018", "R017"} for e in t["subject_evidence"] + t["relation_evidence"])]
    if len(wrong) != 4 or {normalize_text(t["head"]) for t in wrong} != names:
        raise ValueError("历史四条错挂与固定记录不符")
    return len(wrong)


def project(page, output, validated):
    target_refs = {b["block_ref"] for b in page["blocks"] if b["line_id"] in {"R018", "R017"}} if page["number"] == 37 else set()
    entities = {e["id"]: e for e in output["entities"]}
    projected = []
    for rel in output["relations"]:
        subject, obj, anchor = (entities[rel[key]] for key in ("subject_id", "object_id", "entry_anchor_id"))
        evidence = rel["subject_evidence"] + rel["relation_evidence"] + subject["evidence"] + obj["evidence"]
        selected = relevant_refs(evidence, target_refs)
        if selected:
            projected.append({"subject": subject["name"], "relation": rel["type"], "object": obj["name"],
                              "anchor_id": rel["entry_anchor_id"], "anchor_name": anchor["name"], "anchor_type": anchor["type"],
                              "anchor_evidence": anchor["evidence"], "qualifier": rel["qualifier"], "target_evidence": selected})
    mentions = [u for u in output["unresolved"] if relevant_refs(u["evidence"], target_refs)]
    names = {"薄荷叶", "防风", "石斛", "木香"}
    guessed = [x for x in projected if x["relation"] == "剂量" and normalize_text(x["subject"]) in names
               and normalize_text(x["anchor_name"]) == "四物汤"]
    accepted_anchors = {e.entry_id for e in validated.entities if e.type == "方剂" and e.name == "四物汤"}
    admitted = [t for t in validated.triples if t.relation == "剂量" and t.head in names and t.entry_id in accepted_anchors
                and any(e["block_ref"] in target_refs for e in t.subject_evidence + t.relation_evidence)]
    audit = [u for u in validated.unresolved if (u.get("block_ref") in target_refs or
             any(x.get("block_ref") in target_refs for x in u.get("subject_evidence", [])))]
    title_counts = {}
    for title in ("三化汤", "六君子汤"):
        entries = {e.entry_id for e in validated.entities if e.type == "方剂" and e.name == title}
        title_counts[title] = sum(t.entry_id in entries for t in validated.triples)
    return {"target_refs": sorted(target_refs), "before_validation": projected,
            "model_unresolved": mentions, "guessed_later_formula_count": len(guessed),
            "guessed_self_subject_count": sum(x["anchor_type"] == "药材" and normalize_text(x["anchor_name"]) == normalize_text(x["subject"]) for x in projected),
            "after_validation_wrong_entry_count": len(admitted), "validation_unresolved": audit,
            "positive_relations_by_entry": title_counts}


def run_case(page, client):
    blocks = [Block.from_dict(item) for item in page["blocks"]]
    start = time.perf_counter()
    call = extract_candidates(blocks, [b.block_ref for b in blocks], client)
    result = validate_candidates(blocks, [b.block_ref for b in blocks], call, is_demo=False)
    projection = project(page, call["output"], result) if call["output"] is not None else None
    return {"page": page["number"], "role": "development" if page["number"] == 35 else "regression_not_holdout",
            "elapsed_seconds": round(time.perf_counter() - start, 4), "http_status": call["status"],
            "complete_json": call["output"] is not None, "finished_normally": call["diagnostics"].get("finish_reason") == "stop", "error": call["error"], "diagnostics": call["diagnostics"],
            "validation_status": result.status, "validation_error": result.error,
            "accepted_entities": len(result.entities), "accepted_relations": len(result.triples),
            "unresolved_count": len(result.unresolved), "projection": projection}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    if sha(source_bytes) != SOURCE_SHA256:
        raise SystemExit("固定 OCR 数据指纹不符，停止")
    source = json.loads(source_bytes)
    pages = [p for p in source["pages"] if p["number"] in (35, 37)]
    if [p["number"] for p in pages] != [35, 37]:
        raise SystemExit("固定页面范围不符，停止")
    args.dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dir / "manifest.json"
    historical_count = historical_wrong_count()
    expected = {"source_sha256": SOURCE_SHA256, "historical_sha256": sha(HISTORICAL.read_bytes()),
                "historical_wrong_after_validation": historical_count,
                "pages": [{"number": 35, "role": "development"}, {"number": 37, "role": "regression_not_holdout"}],
                "model": MODEL, "base_url": BASE_URL, "prompt_version": PROMPT_VERSION,
                "protocol_version": PROTOCOL_VERSION, "parameters": PARAMETERS}
    if args.action == "prepare":
        with manifest_path.open("x", encoding="utf8") as f:
            json.dump(expected, f, ensure_ascii=False, indent=2)
        print(json.dumps(expected, ensure_ascii=False))
        return 0
    if json.loads(manifest_path.read_text(encoding="utf8")) != expected:
        raise SystemExit("清单与固定输入、版本或参数不一致，未发送页面")
    client = LocalLLMExtractor()
    cfg = client.cfg
    if (not cfg.enabled or cfg.backend.lower() != "cloud" or cfg.base_url.rstrip("/") != BASE_URL or cfg.model != MODEL
            or cfg.response_format != "json_object" or cfg.temperature != 0.1 or cfg.max_tokens != 16384
            or cfg.timeout != 180 or cfg.thinking.strip().lower() != "disabled"):
        raise SystemExit("云端环境不符合固定参数，未发送页面")
    target = args.dir / "results.jsonl"
    secret = os.environ.get("TCM_LLM_API_KEY", "")
    failed = False
    with target.open("x", encoding="utf8") as stream:
        for page in pages:
            started = time.perf_counter()
            try:
                result = run_case(page, client)
            except Exception as exc:
                result = {"page": page["number"], "error": type(exc).__name__, "complete_json": False,
                          "elapsed_seconds": round(time.perf_counter() - started, 4)}
            result.update({"source_sha256": SOURCE_SHA256, "model": MODEL, "prompt_version": PROMPT_VERSION,
                           "protocol_version": PROTOCOL_VERSION, "parameters": PARAMETERS,
                           "historical_wrong_after_validation": historical_count if page["number"] == 37 else None})
            stream.write(json.dumps(redact(result, secret), ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(redact({"page": result["page"], "complete_json": result["complete_json"],
                                     "error": result["error"], "diagnostics": result.get("diagnostics")}, secret), ensure_ascii=False), flush=True)
            failed = failed or bool(result["error"]) or not result.get("complete_json") or not result.get("finished_normally") or result.get("validation_status") == "failure"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
