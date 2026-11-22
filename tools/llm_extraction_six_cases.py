"""六个人工短样本的云端语义检查，JSON Lines 输出。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Block
from extract.llm_extraction import PROMPT_VERSION, PROTOCOL_VERSION, build_input, extract_candidates
from extract.llm_local import LocalLLMExtractor


CASES = [
    ("indication_owner", ["桂枝汤", "桂枝三钱，甘草二钱。", "主治风寒。"], "主治风寒属于桂枝汤，不属于甘草"),
    ("two_formulas", ["桂枝汤", "桂枝三钱。主治风寒。", "四君子汤", "人参二钱。主治气虚。"], "两方主治分别归属本方"),
    ("shared_dose", ["桂枝汤", "桂枝、", "芍药各三钱，甘草一钱。"], "跨列的桂枝和芍药各三钱，甘草一钱，甘草不应另有三钱"),
    ("conditional", ["桂枝汤", "桂枝三钱。若咳嗽，加杏仁二钱。"], "杏仁组成关系保留咳嗽条件，或条件加减进入待核"),
    ("contraindication", ["甘草", "忌海藻。主治咳嗽。"], "禁忌及主治主体为甘草，海藻不是主体"),
    ("continuation", ["各三钱，水煎服。主治风寒。"], "页首主体不明的续行进入待核，不臆造主体关系"),
]


def make_blocks(case_id, lines):
    return [Block(page_id=case_id, line_id=f"L{i}", block_ref=f"{case_id}:L{i}",
                  reading_order=i, col_id=i, kind="title" if i == 1 and len(lines) > 1 else "paragraph", text=text)
            for i, text in enumerate(lines, 1)]


def judge(case_id, data):
    entities = {item["id"]: item for item in data["entities"]}
    relations = data["relations"]
    def has(subject, kind, obj, anchor=None, condition=None):
        return any(entities[r["subject_id"]]["name"] == subject and r["type"] == kind
                   and obj in entities[r["object_id"]]["name"]
                   and (anchor is None or entities[r["entry_anchor_id"]]["name"] == anchor)
                   and (condition is None or condition in r["qualifier"]) for r in relations)
    if case_id == "indication_owner":
        return has("桂枝汤", "主治", "风寒", "桂枝汤") and not has("甘草", "主治", "风寒")
    if case_id == "two_formulas":
        return (has("桂枝汤", "主治", "风寒", "桂枝汤") and has("四君子汤", "主治", "气虚", "四君子汤")
                and not has("桂枝汤", "主治", "气虚") and not has("四君子汤", "主治", "风寒"))
    if case_id == "shared_dose":
        return (all(has(herb, "剂量", dose, "桂枝汤") for herb, dose in [("桂枝", "三钱"), ("芍药", "三钱"), ("甘草", "一钱")])
                and not has("甘草", "剂量", "三钱"))
    if case_id == "conditional":
        unresolved = any("杏仁" in "".join(e["quote"] for e in item["evidence"]) for item in data["unresolved"])
        unconditional = any(r["type"] == "组成" and "杏仁" in entities[r["object_id"]]["name"]
                            and "咳嗽" not in r["qualifier"] for r in relations)
        return not unconditional and (has("桂枝汤", "组成", "杏仁", "桂枝汤", "咳嗽") or unresolved)
    if case_id == "contraindication":
        return has("甘草", "禁忌", "海藻", "甘草") and has("甘草", "主治", "咳嗽", "甘草") and not any(entities[r["subject_id"]]["name"] == "海藻" for r in relations)
    return bool(data["unresolved"]) and not relations


def main():
    client = LocalLLMExtractor()
    if client.cfg.backend.lower() != "cloud" or not client.available():
        print(json.dumps({"error": "cloud_configuration_unavailable"}))
        return 2
    secret = os.environ.get("TCM_LLM_API_KEY", "")
    passed = True
    for case_id, lines, expected in CASES:
        blocks = make_blocks(case_id, lines)
        targets = [b.block_ref for b in blocks[1:]] if len(blocks) > 1 else [blocks[0].block_ref]
        payload = build_input(blocks, targets)
        result = extract_candidates(blocks, targets, client)
        decision = judge(case_id, result["output"]) if result["output"] is not None else False
        record = {"case": case_id, "model": client.cfg.model, "prompt_version": PROMPT_VERSION,
                  "protocol_version": PROTOCOL_VERSION, "input": payload, "output": result["output"], "raw_output": result["raw_output"],
                  "status": result["status"], "error": result["error"], "expected": expected, "passed": decision}
        def redact(value):
            if isinstance(value, str):
                return value.replace(secret, "[redacted]") if secret else value
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            return value
        serialized = json.dumps(redact(record), ensure_ascii=False)
        print(serialized, flush=True)
        passed = passed and decision
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
