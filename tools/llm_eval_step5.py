"""固定历史 OCR，离线评测规则或使用进程凭据评测云端候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schemas import Block, Triple
from extract.llm_extraction import PROMPT_VERSION, PROTOCOL_VERSION, extract_candidates
from extract.llm_local import LocalLLMExtractor
from extract.llm_validate import validate_candidates
from extract.rules import extract_all_audited
from graph.build import build_triples
from ocr.normalize import normalize_text

ANNOTATION = ROOT / "docs/正文方剂表_步骤1标注.json"
RUN = ROOT / "output/formula_stage6/run-eea0ubbv"
DEV = ROOT / "output/small_columns_step1_dev/page35_full_ocr.json"
EXTENDED_DATASET = ROOT / "output/llm_eval_step5_extended/fixed_dataset.json"
DEFAULT = ROOT / "output/llm_eval_step5"
PAGES = (35, 37, 39, 40, 41, 50)
MODEL = "DeepSeek-V4.1-Flash-1017"
BASE_URL = "https://www.autodl.art/api/v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return re.sub(r"\s+", "", normalize_text(str(text or "")))


def dose(text):
    return re.sub(r"^各", "", norm(text))


def intersects(a, b):
    return len(a or []) == 4 and len(b or []) == 4 and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def prepare():
    annotation_bytes = ANNOTATION.read_bytes()
    annotation = json.loads(annotation_bytes)
    by_page = {p["page_no"]: p for p in annotation["pages"] if p.get("book") == "思濟堂方書"}
    pages = []
    for number in PAGES:
        truth = by_page[number]
        if number == 35:
            source = DEV
            ocr = json.loads(source.read_text(encoding="utf8"))
            if ocr.get("baseline_metadata", {}).get("image_sha256") != truth["sha256"]:
                raise ValueError("development OCR image hash does not match annotation")
        else:
            record = json.loads((RUN / f"page_{number}_record.json").read_text(encoding="utf8"))
            source = RUN / "store/pages" / (record["page_id"] + ".json")
            snapshot = json.loads(source.read_text(encoding="utf8"))
            ocr = snapshot["page"]
            uploaded_image = RUN / "uploads" / Path(ocr["image_path"]).name
            if sha(uploaded_image.read_bytes()) != truth["sha256"]:
                raise ValueError(f"uploaded image hash mismatch for page {number}")
        if sha((ROOT / truth["image"]).read_bytes()) != truth["sha256"]:
            raise ValueError(f"source image hash mismatch for page {number}")
        blocks = sorted(ocr["blocks"], key=lambda b: (b["reading_order"], b["line_id"], b["block_ref"]))
        rows = []
        for instance in truth["instances"]:
            if number != 35 and instance.get("display_status") != "应出表":
                continue
            for index, ingredient in enumerate(instance.get("ingredients", []), 1):
                if number == 35 and ingredient.get("role") != "基本组成":
                    continue
                if number != 35 and ingredient.get("display_status") != "应出表":
                    continue
                rows.append({"id": f"{instance['id']}:{index}", "entry": instance["name_verbatim"],
                             "entry_bbox": instance["name_bbox"], "herb": ingredient["herb_verbatim"],
                             "herb_bbox": ingredient["herb_bbox"], "dose": ingredient["dose_verbatim"],
                             "dose_bbox": ingredient["dose_bbox"], "role": ingredient.get("role"),
                             "passage": instance.get("quote", "")})
        pages.append({"number": number, "split": "development" if number == 35 else "holdout",
                      "source_image": truth["image"], "image_sha256": truth["sha256"],
                      "ocr_source": source.relative_to(ROOT).as_posix(), "ocr_source_sha256": sha(source.read_bytes()),
                      "ocr_page_id": ocr["page_id"], "blocks": blocks, "gold_rows": rows,
                      "reviewed_annotations": truth})
    dataset = {"version": "tcm-step5-1.0", "annotation": ANNOTATION.relative_to(ROOT).as_posix(),
               "annotation_sha256": sha(annotation_bytes), "model": MODEL, "base_url": BASE_URL,
               "prompt_version": PROMPT_VERSION, "protocol_version": PROTOCOL_VERSION,
               "parameters": {"response_format": "json_object", "temperature": 0.1, "max_tokens": 4096, "timeout": 120},
               "scope": "仅人工核对的基本组成与剂量行；其余区域未覆盖，不作为负例", "pages": pages}
    return dataset


def boxes(record, by_ref, field):
    locs = record.get(field) or []
    if locs:
        return [x.get("bbox") for x in locs if isinstance(x, dict)]
    ref = record.get("block_ref") or next(iter(record.get("evidence_refs") or []), "")
    return [by_ref[ref]["bbox"]] if ref in by_ref else [record.get("bbox")]


def maximum_matching(edges):
    """一对一最大匹配，避免多个候选共享 OCR 块时受枚举顺序影响。"""
    owner = {}
    def visit(gold, seen):
        for prediction in edges[gold]:
            if prediction in seen:
                continue
            seen.add(prediction)
            if prediction not in owner or visit(owner[prediction], seen):
                owner[prediction] = gold
                return True
        return False
    for gold in range(len(edges)):
        visit(gold, set())
    return {gold: prediction for prediction, gold in owner.items()}


def score_page(page, result):
    """在已标注的名称与位置内计分；其他预测单列待人工复核。"""
    rows, predictions = page["gold_rows"], result.get("predictions") or {}
    entities, triples = predictions.get("entities", []), predictions.get("triples", [])
    blocks = {b["block_ref"]: b for b in page["blocks"]}
    anchors = {}
    for entity in entities:
        if entity.get("type") == "方剂":
            anchors.setdefault(entity.get("entry_id", ""), []).append(entity)
    def entity_matches(e, name, kind, bbox):
        return norm(e.get("name")) == norm(name) and e.get("type") == kind and any(intersects(x, bbox) for x in boxes(e, blocks, "evidence"))
    def anchor_matches(t, row):
        if t.get("entry_id"):
            return any(entity_matches(e, row["entry"], "方剂", row["entry_bbox"]) for e in anchors.get(t["entry_id"], []))
        return any(entity_matches(e, row["entry"], "方剂", row["entry_bbox"]) for e in entities)
    def matches(row, t, kind):
        if t.get("relation") != kind or not anchor_matches(t, row):
            return False
        evidence_boxes = boxes(t, blocks, "relation_evidence")
        if any(x in t.get("qualifier", "") for x in ("若", "条件", "加减", "引药")):
            return False
        if kind == "组成":
            return (norm(t.get("head")) == norm(row["entry"]) and norm(t.get("tail")) == norm(row["herb"])
                    and any(intersects(x, row["herb_bbox"]) for x in evidence_boxes)
                    and not any(x in t.get("qualifier", "") for x in ("若", "条件", "加减", "引药")))
        if not (norm(t.get("head")) == norm(row["herb"]) and dose(t.get("tail")) == dose(row["dose"])
                and any(intersects(x, row["dose_bbox"]) for x in evidence_boxes)):
            return False
        if t.get("entry_id"):
            return True
        return any(matches(row, composition, "组成") for composition in triples)
    matches_by_kind = {kind: maximum_matching([[i for i, t in enumerate(triples) if matches(row, t, kind)]
                                               for row in rows]) for kind in ("组成", "剂量")}
    matched = {kind: set(mapping.values()) for kind, mapping in matches_by_kind.items()}
    ledger = []
    for row_index, row in enumerate(rows):
        item = {"gold": row}
        for kind in ("组成", "剂量"):
            hit = matches_by_kind[kind].get(row_index)
            item[kind] = {"matched": hit is not None, "prediction_index": hit,
                          "prediction": triples[hit] if hit is not None else None,
                          "error": "" if hit is not None else "未找到同时满足类型、条目、名称和出处的预测"}
        ledger.append(item)
    # 不完整的标注范围不足以认定其余输出为错误，全部列为待复核。
    unscored = [t for i, t in enumerate(triples) if i not in matched.get(t.get("relation"), set())]
    entity_gold = []
    entity_keys = set()
    seen_entries = set()
    for row in rows:
        key = (norm(row["entry"]), tuple(row["entry_bbox"]))
        if key not in seen_entries:
            entity_gold.append((row["entry"], "方剂", row["entry_bbox"]))
            seen_entries.add(key)
        for name, kind, bbox in ((row["herb"], "药材", row["herb_bbox"]), (row["dose"], "剂量", row["dose_bbox"])):
            entity_key = (key, kind, dose(name) if kind == "剂量" else norm(name), tuple(bbox))
            if entity_key not in entity_keys:
                entity_keys.add(entity_key)
                entity_gold.append((name, kind, bbox))
    entity_edges = []
    for name, kind, bbox in entity_gold:
        possible = []
        for i, entity in enumerate(entities):
            same_dose = (kind == "剂量" and entity.get("type") == kind and dose(entity.get("name")) == dose(name)
                         and any(intersects(x, bbox) for x in boxes(entity, blocks, "evidence")))
            if entity_matches(entity, name, kind, bbox) or same_dose:
                possible.append(i)
        entity_edges.append(possible)
    entity_matches_by_gold = maximum_matching(entity_edges)
    entity_ledger = []
    for i, (name, kind, bbox) in enumerate(entity_gold):
        hit = entity_matches_by_gold.get(i)
        entity_ledger.append({"expected": {"name": name, "type": kind, "bbox": bbox},
                              "matched": hit is not None, "prediction_index": hit,
                              "prediction": entities[hit] if hit is not None else None})
    return {"gold_rows": len(rows), "composition_hits": len(matched["组成"]), "dose_hits": len(matched["剂量"]),
            "entity_gold": len(entity_gold), "entity_hits": len(entity_matches_by_gold), "entity_unscored": len(entities) - len(entity_matches_by_gold),
            "relation_unscored": len(unscored), "unscored_predictions": unscored, "ledger": ledger,
            "entity_ledger": entity_ledger,
            "precision": None, "f1": None, "reason": "未匹配预测未完成逐项人工核验，不能计作负例；精确率和 F1 待复核"}


def review_triple_evidence(page, triples):
    """逐条复核已接受关系的主体与关系引文，不改变候选结果。"""
    by_ref = {block["block_ref"]: block for block in page["blocks"]}
    problems = []
    for index, raw in enumerate(triples):
        if not isinstance(raw, dict):
            problems.append({"index": index, "reason": "invalid_triple"})
            continue
        triple = Triple.from_dict(raw)
        if triple.validate() or triple.page_id != page["ocr_page_id"] or not triple.entry_id:
            problems.append({"index": index, "reason": "invalid_triple"})
            continue
        if triple.subject_evidence and all(isinstance(loc, dict) for loc in triple.subject_evidence):
            if norm(triple.head) not in norm("".join(str(loc.get("quote", "")) for loc in triple.subject_evidence)):
                problems.append({"index": index, "field": "subject_evidence", "reason": "subject_not_in_evidence"})
        if triple.relation_evidence and isinstance(triple.relation_evidence[0], dict):
            first = triple.relation_evidence[0]
            if any(getattr(triple, key) != first.get(key) for key in ("page_id", "line_id", "bbox", "source_hash", "quote")):
                problems.append({"index": index, "field": "relation_evidence", "reason": "primary_provenance_mismatch"})
        for field in ("subject_evidence", "relation_evidence"):
            locations = getattr(triple, field)
            if not isinstance(locations, list) or not locations:
                problems.append({"index": index, "field": field, "reason": "missing_evidence"})
                continue
            for loc in locations:
                if not isinstance(loc, dict):
                    problems.append({"index": index, "field": field, "reason": "invalid_evidence"})
                    continue
                block = by_ref.get(loc.get("block_ref"))
                start, end, quote = loc.get("start"), loc.get("end"), loc.get("quote")
                if (not block or loc.get("page_id") != page["ocr_page_id"]
                        or loc.get("line_id") != block.get("line_id")
                        or loc.get("bbox") != block.get("bbox")
                        or loc.get("source_hash") != block.get("source_hash")
                        or type(start) is not int or type(end) is not int or not isinstance(quote, str)
                        or not 0 <= start < end <= len(block.get("text", "")) or block["text"][start:end] != quote):
                    problems.append({"index": index, "field": field, "reason": "evidence_mismatch"})
    return problems


def review_probe(dataset, dataset_hash, record):
    """核验开发页的技术状态、冻结元数据和已接受关系证据。"""
    page = next((p for p in dataset["pages"] if p["number"] == 35 and p["split"] == "development"), None)
    diagnostics = record.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    predictions = record.get("predictions") or {}
    triples = predictions.get("triples") if isinstance(predictions, dict) else None
    unresolved = predictions.get("unresolved") if isinstance(predictions, dict) else None
    metadata_ok = (page is not None and record.get("page") == 35 and record.get("split") == "development"
                   and record.get("dataset_sha256") == dataset_hash and record.get("parameters") == dataset["parameters"]
                   and record.get("model") == MODEL and record.get("backend") == "probe" and record.get("prompt_version") == PROMPT_VERSION
                   and record.get("protocol_version") == PROTOCOL_VERSION
                   and diagnostics.get("response_model") in (None, MODEL))
    technical = (metadata_ok and record.get("http_status") == 200 and diagnostics.get("finish_reason") == "stop"
                 and record.get("error") == "" and record.get("status") == "success_with_results"
                 and isinstance(triples, list) and bool(triples) and isinstance(unresolved, list))
    problems = review_triple_evidence(page, triples) if technical else []
    evidence_ok = bool(technical and not problems)
    return {"page": 35, "dataset_sha256": dataset_hash, "metadata_valid": metadata_ok,
            "technical_probe_passed": bool(technical), "evidence_validation_passed": evidence_ok,
            "probe_passed": evidence_ok, "accepted_relations": len(triples) if isinstance(triples, list) else 0,
            "candidate_review_count": len(unresolved) if isinstance(unresolved, list) else None,
            "condition_missing_count": sum(u.get("reason") == "condition_missing" for u in unresolved if isinstance(u, dict)) if isinstance(unresolved, list) else None,
            "evidence_problems": problems}


def run_page(page, backend, client=None):
    blocks = [Block.from_dict(b) for b in page["blocks"]]
    start = time.perf_counter()
    if backend == "rules":
        entities, candidates, unresolved = extract_all_audited(blocks, is_demo=False)
        triples = build_triples(entities, blocks, candidates)
        result = {"error": "", "status": "success", "predictions": {"entities": [x.to_dict() for x in entities],
                  "triples": [x.to_dict() for x in triples], "unresolved": unresolved}}
    else:
        call = extract_candidates(blocks, [b.block_ref for b in blocks], client)
        validated = validate_candidates(blocks, [b.block_ref for b in blocks], call, is_demo=False)
        diagnostics = call["diagnostics"]
        triples = [x.to_dict() for x in validated.triples]
        technical = (call["status"] == 200 and diagnostics["finish_reason"] == "stop"
                     and not validated.error and validated.status == "success_with_results" and bool(triples))
        evidence_passed = bool(technical and not review_triple_evidence(page, triples))
        result = {"error": validated.error, "status": validated.status, "http_status": call["status"],
                  "diagnostics": diagnostics, "evidence_validation_passed": evidence_passed,
                  "technical_probe_passed": bool(technical), "candidate_review_count": len(validated.unresolved),
                  "probe_passed": evidence_passed,
                  "predictions": {"entities": [x.to_dict() for x in validated.entities],
                  "triples": triples, "unresolved": validated.unresolved}}
    result["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    result["page"] = page["number"]
    result["split"] = page["split"]
    result["score"] = score_page(page, result)
    return result


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[redacted]") if secret else value
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    return value


def load_results(path):
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    if len({r["page"] for r in records}) != len(records):
        raise ValueError("duplicate page result")
    return {r["page"]: r for r in records}


def summarize(dataset, directory, dataset_hash):
    rules = load_results(directory / "rules_results.jsonl")
    cloud = load_results(directory / "cloud_results.jsonl")
    for results in (rules, cloud):
        if results and any(r.get("dataset_sha256") != dataset_hash for r in results.values()):
            raise ValueError("result dataset hash mismatch")
    pages = []
    for page in dataset["pages"]:
        number = page["number"]
        pages.append({"page": number, "split": page["split"], "source_image": page["source_image"],
                      "ocr_source": page["ocr_source"], "original_blocks": page["blocks"],
                      "human_gold_rows": page["gold_rows"], "rules": rules.get(number) if rules else None,
                      "cloud": cloud.get(number) if cloud else None})
    summary = {}
    for backend, source in (("rules", rules), ("cloud", cloud)):
        if source is None:
            summary[backend] = {"status": "not_run"}
            continue
        holdout = [source.get(page["number"]) for page in dataset["pages"] if page["split"] == "holdout"]
        complete = [r for r in holdout if r and r.get("status") != "failure" and r.get("score")]
        score = [r["score"] for r in complete]
        all_gold = [page for page in dataset["pages"] if page["split"] == "holdout"]
        gold_rows = sum(len(page["gold_rows"]) for page in all_gold)
        # 对失败页保留分母；仅当全部页都成功时给出可比的召回率。
        hits = sum(s["composition_hits"] + s["dose_hits"] for s in score)
        entity_hits = sum(s["entity_hits"] for s in score)
        entity_gold = sum(score_page(page, {"predictions": {}})["entity_gold"] for page in all_gold)
        audits = sum(len(r["predictions"]["unresolved"]) for r in complete)
        accepted = sum(len(r["predictions"]["triples"]) for r in complete)
        failures = len(holdout) - len(complete)
        summary[backend] = {"status": "complete" if not failures else "incomplete",
                            "pages": len(holdout), "failed_or_missing_pages": failures,
                            "failure_rate": failures / len(holdout),
                            "elapsed_seconds_per_page": {str(r["page"]): r.get("elapsed_seconds") for r in holdout if r},
                            "relation_gold": 2 * gold_rows, "relation_hits": hits,
                            "relation_recall": hits / (2 * gold_rows) if not failures and gold_rows else None,
                            "relation_recall_including_failures_as_misses": hits / (2 * gold_rows) if gold_rows else None,
                            "entity_gold": entity_gold, "entity_hits": entity_hits,
                            "entity_recall": entity_hits / entity_gold if not failures and entity_gold else None,
                            "entity_recall_including_failures_as_misses": entity_hits / entity_gold if entity_gold else None,
                            "entity_precision": None, "entity_f1": None,
                            "relation_precision": None, "relation_f1": None, "wrong_subject_count": None,
                            "audit_records": audits, "accepted_relations": accepted,
                            "audit_ratio": audits / (audits + accepted) if audits + accepted else None,
                            "unscored_relations": sum(s["relation_unscored"] for s in score),
                            "unscored_entities": sum(s["entity_unscored"] for s in score)}
    return {"dataset_sha256": dataset_hash, "model": dataset.get("model", MODEL),
            "prompt_version": dataset.get("prompt_version", PROMPT_VERSION),
            "protocol_version": dataset.get("protocol_version", PROTOCOL_VERSION), "parameters": dataset["parameters"],
            "scope": dataset["scope"], "summary": summary, "pages": pages,
            "limitations": ["未匹配预测尚未逐条人工复核，实体与关系的精确率、F1 和主体错挂数无法确认",
                            "存在失败页时，单独列出将失败页视为未命中的端到端召回，不当作模型语义召回",
                            "待核比例分母为成功页的待核记录数加通过校验的关系数，含实体层拒收记录",
                            "开发页仅作指定片段检查，留出页不用于调参",
                            "否定与条件加减误报需在完整预测上人工逐项复核，不能据未标注区域自动判负例"],
            "coverage": {"同页多方": "50 页桂枝汤与九味羌活汤均有基本配伍行",
                         "跨行主体": "50 页方名与配伍分列，可检查原文块；主治关系未纳入金标准",
                         "跨列共用剂量": "50 页九味羌活汤苍术、生地共用各一钱半",
                         "条件加减": "37 页页首按症加味保留原文，前页主体未在本页确认",
                         "禁忌对象": "本批无已确认禁忌主体及关系金标准，待补人工标注",
                         "OCR缺字": "37 页正〔?〕榔为疑字，已排除基本组成分母",
                         "疑似续页": "40 页三味均承接前页，本页没有可确认起方"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "prepare-extended", "prepare-no-thinking", "probe", "recheck", "rules", "cloud", "report"))
    parser.add_argument("--dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dir / "fixed_dataset.json"
    if args.action in ("prepare", "prepare-extended", "prepare-no-thinking"):
        if dataset_path.exists():
            raise SystemExit("固定数据已存在，拒绝覆盖；可选择新的 --dir")
        if args.action in ("prepare-extended", "prepare-no-thinking"):
            original = (EXTENDED_DATASET
                        if args.action == "prepare-no-thinking" else DEFAULT / "fixed_dataset.json")
            dataset = json.loads(original.read_text(encoding="utf8"))
            dataset["parameters"] = {"response_format": "json_object", "temperature": 0.1, "max_tokens": 16384, "timeout": 180}
            if args.action == "prepare-no-thinking":
                dataset["parameters"]["thinking"] = "disabled"
            dataset["derived_from_dataset_sha256"] = sha(original.read_bytes())
        else:
            dataset = prepare()
        dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"fixed dataset: {dataset_path}")
        return
    dataset = json.loads(dataset_path.read_text(encoding="utf8"))
    if args.action != "report" and (dataset.get("model") != MODEL or dataset.get("base_url") != BASE_URL
            or dataset.get("prompt_version") != PROMPT_VERSION or dataset.get("protocol_version") != PROTOCOL_VERSION):
        raise SystemExit("固定数据的模型或协议版本不一致，停止评测")
    if [page.get("number") for page in dataset.get("pages", [])] != list(PAGES):
        raise SystemExit("固定页范围不符，停止评测")
    if args.action in ("probe", "recheck", "cloud") and dataset["parameters"].get("thinking") == "disabled":
        original_bytes = EXTENDED_DATASET.read_bytes()
        original = json.loads(original_bytes)
        if (dataset.get("derived_from_dataset_sha256") != sha(original_bytes) or dataset["pages"] != original["pages"]
                or dataset["parameters"] != {**original["parameters"], "thinking": "disabled"}):
            raise SystemExit("固定 OCR 与原批次不一致，未发送页面")
    if args.action == "recheck":
        probe_path = args.dir / "probe_results.jsonl"
        probe = load_results(probe_path)
        if not probe or len(probe) != 1 or 35 not in probe:
            raise SystemExit("开发探测记录缺失或页范围不符")
        review = review_probe(dataset, sha(dataset_path.read_bytes()), probe[35])
        review["probe_sha256"] = sha(probe_path.read_bytes())
        destination = args.dir / "probe_review.json"
        with destination.open("x", encoding="utf8") as stream:
            json.dump(review, stream, ensure_ascii=False, indent=2)
        print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0 if review["probe_passed"] else 1
    if args.action == "report":
        report = summarize(dataset, args.dir, sha(dataset_path.read_bytes()))
        if dataset.get("prompt_version") == PROMPT_VERSION and dataset.get("protocol_version") == PROTOCOL_VERSION:
            destination = args.dir / "comparison.json"
            with destination.open("x", encoding="utf8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return
    client = None
    if args.action in ("cloud", "probe"):
        client = LocalLLMExtractor()
        cfg = client.cfg
        parameters = dataset["parameters"]
        if (not cfg.enabled or cfg.backend.lower() != "cloud" or cfg.base_url.rstrip("/") != BASE_URL or cfg.model != MODEL
                or cfg.response_format != parameters["response_format"] or cfg.temperature != parameters["temperature"]
                or cfg.max_tokens != parameters["max_tokens"] or cfg.timeout != parameters["timeout"]
                or cfg.thinking.strip().lower() != parameters.get("thinking", "")):
            raise SystemExit("云端配置未满足冻结参数，未发送任何页面")
        if args.action == "cloud" and parameters.get("thinking") == "disabled":
            probe = load_results(args.dir / "probe_results.jsonl")
            record = probe.get(35) if probe and len(probe) == 1 else None
            if not record or not review_probe(dataset, sha(dataset_path.read_bytes()), record)["probe_passed"]:
                raise SystemExit("开发页探测尚未通过，未发送留出页")
    destination = args.dir / (args.action + "_results.jsonl")
    if destination.exists():
        raise SystemExit("结果文件已存在，拒绝覆盖；可选择新的 --dir 并复制固定数据")
    secret = os.environ.get("TCM_LLM_API_KEY", "")
    failed = False
    selected_pages = [page for page in dataset["pages"] if page["number"] == 35 and page["split"] == "development"] if args.action == "probe" else dataset["pages"]
    if args.action == "probe" and len(selected_pages) != 1:
        raise SystemExit("开发页范围不符，未发送页面")
    with destination.open("x", encoding="utf8") as stream:
        for page in selected_pages:
            started = time.perf_counter()
            try:
                result = run_page(page, "cloud" if args.action == "probe" else args.action, client)
            except Exception as exc:
                result = {"page": page["number"], "split": page["split"], "status": "failure",
                          "error": type(exc).__name__, "predictions": None,
                          "elapsed_seconds": round(time.perf_counter() - started, 4)}
            result.update({"backend": args.action, "model": MODEL if client else "rules", "dataset_sha256": sha(dataset_path.read_bytes()),
                           "prompt_version": PROMPT_VERSION, "protocol_version": PROTOCOL_VERSION,
                           "parameters": parameters if args.action == "probe" else dataset["parameters"]})
            stream.write(json.dumps(redact(result, secret), ensure_ascii=False) + "\n")
            stream.flush()
            progress = {"page": page["number"], "status": result["status"], "error": result["error"]}
            if args.action == "probe":
                progress.update({"probe_passed": result.get("probe_passed", False), "diagnostics": result.get("diagnostics")})
            print(json.dumps(redact(progress, secret), ensure_ascii=False), flush=True)
            failed = failed or result["status"] == "failure" or (args.action == "probe" and not result.get("probe_passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
