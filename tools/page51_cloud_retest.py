"""Isolated page-51 production OCR and optional authorized cloud retest."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block
from extract.llm_extraction import PROMPT_VERSION, PROTOCOL_VERSION, extract_candidates
from extract.llm_local import LocalLLMExtractor
from extract.llm_validate import validate_candidates
from ocr import pipeline

OLD = ROOT / "output/page51_acceptance"
DEFAULT = ROOT / "output/page51_retest_new"
IMAGE = ROOT / "古籍评测集/医家类/思濟堂方書/page_51.png"
IMAGE_SHA256 = "1ece3630fa897c88fbc8c2216310ad568e09249ca092dd317789fe600a44ad84"


def write(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    secret = os.environ.get("TCM_LLM_API_KEY", "")
    if secret:
        text = text.replace(secret, "[redacted]")
    with path.open("x", encoding="utf8") as stream:
        stream.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("ocr", "cloud"))
    parser.add_argument("--dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if hashlib.sha256(IMAGE.read_bytes()).hexdigest() != IMAGE_SHA256:
        raise SystemExit("原图指纹不符")
    if args.action == "ocr":
        args.dir.mkdir(parents=True, exist_ok=False)
        config = replace(pipeline.CONFIG, paths=replace(pipeline.CONFIG.paths, reports_dir=args.dir),
                         ocr=replace(pipeline.CONFIG.ocr, correct_enabled=False))
        with patch.object(pipeline, "CONFIG", config), patch.object(pipeline.correct, "CONFIG", config):
            result = pipeline.run_ocr(str(IMAGE), page_id="RETEST-P51")
        if result.is_demo or not result.model_loaded or not result.blocks:
            raise SystemExit("未获得真实 OCR；未发起云调用")
        write(args.dir / "ocr.json", result.to_dict())
        print(json.dumps({"ocr": "complete", "blocks": len(result.blocks), "engine": result.engine}, ensure_ascii=False))
        return 0
    if (args.dir / "cloud_result.json").exists() or (args.dir / "cloud_candidates.json").exists():
        raise SystemExit("已有云端结果，拒绝覆盖")
    page = json.loads((args.dir / "ocr.json").read_text(encoding="utf8"))
    client = LocalLLMExtractor()
    cfg = client.cfg
    if (not cfg.enabled or cfg.backend.lower() != "cloud" or cfg.base_url.rstrip("/") != "https://www.autodl.art/api/v1"
            or cfg.model != "DeepSeek-V4.1-Flash-1017" or cfg.thinking != "disabled"
            or cfg.response_format != "json_object" or cfg.temperature != 0.1 or cfg.max_tokens != 16384 or cfg.timeout != 180):
        raise SystemExit("云端凭据或固定参数不符合要求，未发送")
    blocks = [Block.from_dict(x) for x in page["blocks"]]
    refs = [b.block_ref for b in blocks]
    # Gold is loaded only after the model call and never enters its request.
    call = extract_candidates(blocks, refs, client=client)
    result = validate_candidates(blocks, refs, call, is_demo=False)
    write(args.dir / "cloud_candidates.json", {"output": call.get("output"), "error": call.get("error"),
                                               "diagnostics": call.get("diagnostics"), "http_status": call.get("status")})
    spec = importlib.util.spec_from_file_location("page51_original_verifier", OLD / "verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    gold = json.loads((OLD / "gold.json").read_text(encoding="utf8"))
    write(args.dir / "gold_separate.json", gold)
    record = {"prompt_version": PROMPT_VERSION, "protocol_version": PROTOCOL_VERSION,
              "http_status": call.get("status"), "diagnostics": call.get("diagnostics"),
              "complete_json": call.get("output") is not None, "validation_status": result.status,
              "error": result.error, "candidate_count": len(call["output"]["relations"]) if call.get("output") else 0,
              "accepted_count": len(result.triples), "composition": [t.to_dict() for t in result.triples if t.relation == "组成"],
              "unresolved": result.unresolved, "score": verifier.score(gold, blocks, result.triples)}
    write(args.dir / "cloud_result.json", record)
    print(json.dumps({"complete_json": record["complete_json"], "status": result.status,
                      "candidate_count": record["candidate_count"], "accepted_count": record["accepted_count"],
                      "score": {k: record["score"][k] for k in ("passed", "matched", "expected", "false_positives", "missing")}}, ensure_ascii=False))
    return 0 if record["complete_json"] and record["diagnostics"].get("finish_reason") == "stop" and record["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
