"""将云端候选定位到 OCR 原文，产生独立的可审计记录。"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from app.schemas import Block, Entity, Triple, ENTITY_TYPES, source_hash_of
from extract.schema import RELATIONS
from extract.formula_segments import extract_entry_scopes
from ocr.normalize import normalize_text


@dataclass
class ValidationResult:
    status: str
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


_DIRECTIONS = {
    "组成": ({"方剂"}, {"药材"}), "剂量": ({"药材"}, {"剂量"}),
    "主治": ({"方剂", "药材"}, {"主治"}), "功效": ({"方剂", "药材"}, {"功效"}),
    "性味": ({"方剂", "药材"}, {"性味"}), "归经": ({"药材"}, {"归经"}),
    "禁忌": ({"方剂", "药材"}, {"禁忌", "药材"}), "别名": ({"药材"}, {"药材"}),
}


def validate_candidates(blocks: Iterable[Block], target_refs: Iterable[str], call: dict, *, is_demo: bool = True) -> ValidationResult:
    """仅核验契约与原文位置，不推定模型的医学语义正确。"""
    blocks = list(blocks)
    by_ref = {b.block_ref: b for b in blocks}
    targets = set(target_refs)
    if (not blocks or len(by_ref) != len(blocks) or any(not b.block_ref or not b.page_id or not b.line_id for b in blocks)
            or len({b.page_id for b in blocks}) != 1 or not targets or not targets.issubset(by_ref)):
        return ValidationResult("failure", error="invalid_input")
    page_id = blocks[0].page_id
    if not isinstance(call, dict) or call.get("error") or not isinstance(call.get("output"), dict):
        return ValidationResult("failure", error=(call.get("error") or "invalid_output") if isinstance(call, dict) else "invalid_output")
    data = call["output"]
    if set(data) != {"entities", "relations", "unresolved"} or any(not isinstance(data[k], list) for k in data):
        return ValidationResult("failure", error="invalid_protocol")

    try:
        scopes = extract_entry_scopes(blocks)
    except Exception:
        return ValidationResult("failure", error="scope_detection_failed")

    audits = []
    def audit(reason, refs=(), entry_id="", relation="", tail=""):
        ref = next((r.get("block_ref") for r in refs if isinstance(r, dict) and r.get("block_ref") in by_ref), "")
        b = by_ref.get(ref)
        audits.append({"page_id": page_id, "block_ref": ref, "line_id": b.line_id if b else "",
                       "bbox": list(b.bbox) if b else [], "source_hash": b.source_hash if b else "",
                       "reading_order": b.reading_order if b else 0, "entry_id": entry_id,
                       "reason": reason, "relation": relation, "tail": tail,
                       "quote": next((r.get("quote", "") for r in refs if isinstance(r, dict)), ""),
                       "evidence": refs})

    def locate(evidence, target_only=False):
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("missing_evidence")
        found = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) not in ({"block_ref", "quote"}, {"block_ref", "quote", "start"}):
                raise ValueError("invalid_evidence")
            ref, quote = item.get("block_ref"), item.get("quote")
            if not isinstance(ref, str) or ref not in by_ref:
                raise ValueError("unknown_block")
            if target_only and ref not in targets:
                raise ValueError("outside_target")
            if not isinstance(quote, str) or not quote:
                raise ValueError("empty_quote")
            block = by_ref[ref]
            starts = []
            pos = block.text.find(quote)
            while pos >= 0:
                starts.append(pos)
                pos = block.text.find(quote, pos + 1)
            if not starts:
                raise ValueError("quote_not_found")
            if "start" in item:
                start = item["start"]
                if isinstance(start, bool) or not isinstance(start, int) or start not in starts:
                    raise ValueError("invalid_quote_position")
            elif len(starts) == 1:
                start = starts[0]
            else:
                raise ValueError("ambiguous_quote")
            found.append({"page_id": block.page_id, "block_ref": ref, "line_id": block.line_id,
                          "bbox": list(block.bbox), "source_hash": block.source_hash,
                          "quote": quote, "start": start, "end": start + len(quote)})
        return found

    candidates = {}
    invalid_ids = set()
    def name_spans(name, evidence):
        text = "".join(loc["quote"] for loc in evidence)
        start = text.find(name)
        if start < 0:
            raise ValueError("entity_name_not_in_evidence")
        if text.find(name, start + 1) >= 0:
            raise ValueError("ambiguous_entity_name")
        end = start + len(name)
        cursor = 0
        spans = []
        for loc in evidence:
            length = len(loc["quote"])
            left, right = max(start, cursor), min(end, cursor + length)
            if left < right:
                spans.append(f"{loc['block_ref']}:{loc['start'] + left - cursor}:{loc['start'] + right - cursor}")
            cursor += length
        return spans
    for item in data["entities"]:
        if not isinstance(item, dict) or set(item) != {"id", "name", "type", "evidence"}:
            audit("invalid_entity_structure")
            continue
        ident = item["id"]
        if not isinstance(ident, str) or not ident or ident in candidates or ident in invalid_ids:
            if isinstance(ident, str):
                invalid_ids.add(ident)
                candidates.pop(ident, None)
            audit("duplicate_or_invalid_entity_id", item.get("evidence", []))
            continue
        try:
            if not isinstance(item["name"], str) or not item["name"] or item["type"] not in ENTITY_TYPES:
                raise ValueError("invalid_entity_type_or_name")
            evidence = locate(item["evidence"])
            name_spans(item["name"], evidence)
            candidates[ident] = (item, evidence)
        except (ValueError, TypeError) as exc:
            invalid_ids.add(ident)
            audit(str(exc), item.get("evidence", []) if isinstance(item.get("evidence"), list) else [])

    entities = {}
    triples = {}
    referenced = set()
    def stable_entry(ident):
        item, evidence = candidates[ident]
        return "entry-" + source_hash_of(page_id, normalize_text(item["name"]),
                                         *name_spans(item["name"], evidence))

    def overlap(a, b):
        return (a["block_ref"] == b["block_ref"] and a["start"] < b["block_raw_range"][1]
                and b["block_raw_range"][0] < a["end"])

    def continuation_conflict(anchor_id, evidence):
        item, anchor_locs = candidates[anchor_id]
        spans = []
        for span in name_spans(item["name"], anchor_locs):
            ref, start, end = span.rsplit(":", 2)
            spans.append({"block_ref": ref, "start": int(start), "end": int(end)})
        for entry in scopes["entries"]:
            if normalize_text(entry["text"]) != normalize_text(item["name"]):
                continue
            if not any(overlap(loc, src) for loc in spans for src in entry["sources"]):
                continue
            anchor_start = min(offsets[loc["block_ref"]] + loc["start"] for loc in spans)
            for continuation in scopes["continuation"]:
                matched = [src for src in continuation["sources"] if any(
                    overlap(loc, src) and offsets[loc["block_ref"]] + max(loc["start"], src["block_raw_range"][0]) < anchor_start
                    for loc in evidence)]
                if matched:
                    return matched
        return []

    ordered = sorted(blocks, key=lambda b: (b.reading_order, b.line_id, b.block_ref))
    offsets = {}
    raw_text = ""
    for block in ordered:
        offsets[block.block_ref] = len(raw_text)
        raw_text += block.text
    boundaries = [offsets[src["block_ref"]] + src["block_raw_range"][1]
                  for entry in scopes["entries"] for src in entry["sources"] if src["block_ref"] in offsets]
    # A preparation instruction establishes a local boundary; it does not prove
    # that the following prose is a composition list. Never use a model's role
    # label as the sole evidence of that boundary.
    preparation = re.compile(r'(?:水|酒)?[捣搗煎研煮炒炙][^。；;\n]{0,18}?(?:服|饮|飲)')

    def discussion_boundary(anchor_id, object_id, relation_evidence):
        anchor_item, anchor_evidence = candidates[anchor_id]
        object_item, object_evidence = candidates[object_id]
        anchor_spans = name_spans(anchor_item["name"], anchor_evidence)
        object_spans = name_spans(object_item["name"], object_evidence)
        def positions(spans):
            result = []
            for span in spans:
                ref, start, end = span.rsplit(":", 2)
                result.append((ref, int(start), int(end)))
            return result
        anchor_position = min(offsets[ref] + start for ref, start, _ in positions(anchor_spans))
        for ref, start, end in positions(object_spans):
            if not any(e["block_ref"] == ref and e["start"] < end and start < e["end"] for e in relation_evidence):
                continue
            position = offsets[ref] + start
            # A later title alone cannot turn an earlier discussion into a
            # composition. Without a preceding preparation boundary, explicit
            # backward links remain eligible for their usual checks.
            start = anchor_position if position > anchor_position else max(
                (point for point in boundaries if point <= position), default=0)
            marker = list(preparation.finditer(raw_text, start, position))
            if marker:
                m = marker[-1]
                return [{"block_ref": b.block_ref, "block_raw_range": [max(0, m.start()-offsets[b.block_ref]),
                        min(len(b.text), m.end()-offsets[b.block_ref])], "text": b.text[max(0, m.start()-offsets[b.block_ref]):min(len(b.text), m.end()-offsets[b.block_ref])]}
                        for b in ordered if offsets[b.block_ref] < m.end() and offsets[b.block_ref] + len(b.text) > m.start()]
        return []
    def addition_context(locations):
        """只识别与目标药名相邻的加味线索；无法判定角色的加字保守待核。"""
        symptom = re.compile(r'(?:若|如|如果)[^，。；;\n]{1,18}|[^，。；;\n]{1,14}(?:痛|癢|痒|咳|嗽|熱|热|寒|腫|肿|喘|攣|挛|渴|泄|汗)')
        for loc in locations:
            block = by_ref[loc["block_ref"]]
            position = offsets[block.block_ref] + loc["start"]
            boundary = max((point for point in boundaries if point <= position), default=0)
            prior_preparation = list(preparation.finditer(raw_text, boundary, position))
            if prior_preparation:
                boundary = max(boundary, prior_preparation[-1].end())
            prefix = raw_text[boundary:position]
            match = re.search(r'([^。；;\n]{0,24})加([^。；;\n]*)$', prefix)
            if not match:
                continue
            before_add = match.group(1).strip(" ，、")
            context_start = boundary + match.start()
            context_blocks = [item for item in ordered
                              if offsets[item.block_ref] < position + 1
                              and offsets[item.block_ref] + len(item.text) > context_start]
            if before_add.endswith(("不", "勿", "禁", "忌", "不可", "不宜")):
                return "negated", "", context_blocks
            if before_add.endswith(("引", "水", "酒", "湯", "汤")):
                return "role_uncertain", "", context_blocks
            conditions = list(symptom.finditer(before_add))
            if conditions:
                condition = conditions[-1].group(0).strip(" ，、")
                return "conditional", condition, context_blocks
            return "role_uncertain", "", context_blocks
        return "", "", []
    def make_entity(ident, entry):
        item, evidence = candidates[ident]
        key = (entry, normalize_text(item["name"]), item["type"])
        if key in entities:
            for loc in evidence:
                if loc not in entities[key].evidence:
                    entities[key].evidence.append(loc)
            return
        first = evidence[0]
        entities[key] = Entity(page_id=first["page_id"], bbox=first["bbox"], line_id=first["line_id"],
                               quote=first["quote"], source_hash=first["source_hash"], block_ref=first["block_ref"],
                               name=key[1], value=item["name"], type=item["type"], extractor="llm_local",
                               is_demo=is_demo, entry_id=entry, evidence=list(evidence))

    for relation in data["relations"]:
        if not isinstance(relation, dict) or set(relation) != {"subject_id", "object_id", "type", "entry_anchor_id", "subject_evidence", "relation_evidence", "qualifier"}:
            audit("invalid_relation_structure")
            continue
        refs = relation.get("relation_evidence", [])
        refs = refs if isinstance(refs, list) else []
        for key in ("subject_id", "object_id", "entry_anchor_id"):
            if isinstance(relation.get(key), str):
                referenced.add(relation[key])
        anchor_id = relation.get("entry_anchor_id")
        entry = stable_entry(anchor_id) if isinstance(anchor_id, str) and anchor_id in candidates else ""
        subject_evidence, relation_evidence, condition_evidence, scope_evidence = [], [], [], []
        try:
            sid, oid, aid = (relation[k] for k in ("subject_id", "object_id", "entry_anchor_id"))
            if any(not isinstance(x, str) or x not in candidates for x in (sid, oid, aid)):
                raise ValueError("unknown_endpoint_or_anchor")
            typ = relation["type"]
            if not isinstance(typ, str) or typ not in RELATIONS or typ not in _DIRECTIONS:
                raise ValueError("invalid_relation_type")
            subject, obj, anchor = (candidates[x][0] for x in (sid, oid, aid))
            allowed_subject, allowed_object = _DIRECTIONS[typ]
            if subject["type"] not in allowed_subject or obj["type"] not in allowed_object:
                raise ValueError("invalid_relation_direction")
            if anchor["type"] not in {"方剂", "药材"}:
                raise ValueError("invalid_entry_anchor")
            if typ == "组成" and sid != aid:
                raise ValueError("composition_anchor_mismatch")
            if typ == "剂量" and anchor["type"] == "药材" and sid != aid:
                raise ValueError("dose_anchor_mismatch")
            if typ != "剂量" and subject["type"] in {"方剂", "药材"} and sid != aid:
                raise ValueError("subject_anchor_mismatch")
            qualifier = relation["qualifier"]
            if not isinstance(qualifier, str):
                raise ValueError("invalid_qualifier")
            subject_evidence = locate(relation["subject_evidence"])
            relation_evidence = locate(relation["relation_evidence"], target_only=True)
            if typ == "组成":
                spans = name_spans(obj["name"], candidates[oid][1])
                for span in spans:
                    ref, start, end = span.rsplit(":", 2)
                    if not any(loc["block_ref"] == ref and loc["start"] < int(end) and int(start) < loc["end"]
                               for loc in relation_evidence):
                        raise ValueError("composition_herb_evidence_mismatch")
            kind = ""
            if typ in {"组成", "剂量"} and anchor["type"] == "方剂":
                scope_evidence = continuation_conflict(aid, subject_evidence + relation_evidence + candidates[oid][1])
                if scope_evidence:
                    raise ValueError("continuation_entry_conflict")
                herb_locations = candidates[oid][1] if typ == "组成" else subject_evidence + candidates[sid][1]
                kind, condition, condition_blocks = addition_context(herb_locations)
                if kind:
                    condition_evidence = [{"page_id": block.page_id, "block_ref": block.block_ref,
                                           "line_id": block.line_id, "bbox": list(block.bbox),
                                           "source_hash": block.source_hash, "quote": block.text,
                                           "start": 0, "end": len(block.text)} for block in condition_blocks]
                if kind == "conditional":
                    expected = re.sub(r'^(?:若|如果|如)', '', normalize_text(condition))
                    if not expected or expected not in normalize_text(qualifier) or any(word in qualifier for word in ("基础配伍", "基本组成")):
                        raise ValueError("condition_missing")
                elif kind == "negated":
                    raise ValueError("negated_addition")
                elif kind == "role_uncertain":
                    if (not any(word in qualifier for word in ("引药", "引藥", "制法", "送服", "炮制", "炮製"))
                            or any(word in qualifier for word in ("基础配伍", "基本组成"))):
                        raise ValueError("addition_role_uncertain")
                if typ == "组成":
                    role_boundary = discussion_boundary(aid, oid, relation_evidence)
                    if role_boundary and kind != "conditional":
                        scope_evidence = role_boundary
                        raise ValueError("composition_after_preparation_uncertain")
                    if re.search(r'疑(?:为|為|是|作)|(?:OCR|识别|識別).{0,12}(?:有误|有誤|误读|誤讀|不清|存疑)', qualifier, re.I):
                        raise ValueError("uncertain_ocr_herb")
            if "条件" in qualifier and kind != "conditional" and not any(word in qualifier for word in ("若", "如", "如果", "遇", "兼")):
                raise ValueError("condition_missing")
            if subject["name"] not in "".join(e["quote"] for e in subject_evidence):
                raise ValueError("subject_not_in_evidence")
            for ident in (aid, sid, oid):
                make_entity(ident, entry)
            head, tail = normalize_text(subject["name"]), normalize_text(obj["name"])
            key = (entry, head, typ, tail, qualifier)
            if key not in triples:
                first = relation_evidence[0]
                triples[key] = Triple(page_id=first["page_id"], line_id=first["line_id"], bbox=first["bbox"],
                                      source_hash=first["source_hash"], quote=first["quote"], head=head,
                                      relation=typ, tail=tail, evidence_refs=[e["block_ref"] for e in relation_evidence],
                                      is_demo=is_demo, entry_id=entry, qualifier=qualifier,
                                      subject_evidence=subject_evidence, relation_evidence=relation_evidence,
                                      condition_evidence=condition_evidence)
            else:
                triple = triples[key]
                for name, evidence in (("subject_evidence", subject_evidence), ("relation_evidence", relation_evidence), ("condition_evidence", condition_evidence)):
                    for loc in evidence:
                        if loc not in getattr(triple, name):
                            getattr(triple, name).append(loc)
                triple.evidence_refs = list(dict.fromkeys(triple.evidence_refs + [e["block_ref"] for e in relation_evidence]))
        except (ValueError, TypeError, KeyError) as exc:
            audit(str(exc), refs, entry_id=entry, relation=str(relation.get("type", "")))
            if str(exc) in {"continuation_entry_conflict", "condition_missing", "addition_role_uncertain", "negated_addition", "composition_after_preparation_uncertain", "uncertain_ocr_herb", "composition_herb_evidence_mismatch"} and all(isinstance(relation.get(k), str) and relation[k] in candidates for k in ("subject_id", "object_id", "entry_anchor_id")):
                sid, oid, aid = (relation[k] for k in ("subject_id", "object_id", "entry_anchor_id"))
                audits[-1].update({"candidate_anchor": {"name": candidates[aid][0]["name"], "entry_id": entry},
                                   "subject": candidates[sid][0]["name"], "tail": candidates[oid][0]["name"],
                                   "subject_evidence": candidates[sid][1], "object_evidence": candidates[oid][1],
                                   "anchor_evidence": candidates[aid][1], "qualifier": relation.get("qualifier", ""),
                                   "condition_evidence": condition_evidence, "scope_evidence": scope_evidence,
                                   "reason_detail": {"continuation_entry_conflict": "页首续文与后续另起方剂范围冲突，原主体待核",
                                                     "condition_missing": "原文存在条件加味，限定条件未完整保留",
                                                     "addition_role_uncertain": "加字可能涉及引药、制法或加减，作用角色待核",
                                                     "negated_addition": "原文为不加或禁加，不能转成肯定配伍",
                                                     "composition_after_preparation_uncertain": "候选位于已定位制法之后，缺少独立配伍证据，原文角色待核",
                                                     "uncertain_ocr_herb": "模型标注药名识读存疑，保留 OCR 原文等待独立证据",
                                                     "composition_herb_evidence_mismatch": "组成关系的定位证据未覆盖该药名，不能用其他位置代替"}[str(exc)]})

    for item in data["unresolved"]:
        if not isinstance(item, dict) or set(item) != {"evidence", "reason", "candidate_subject_ids"}:
            audit("invalid_unresolved_structure")
            continue
        refs = item.get("evidence", []) if isinstance(item.get("evidence"), list) else []
        try:
            if not isinstance(item["reason"], str) or not item["reason"] or not isinstance(item["candidate_subject_ids"], list):
                raise ValueError("invalid_unresolved")
            if any(not isinstance(x, str) or x not in candidates for x in item["candidate_subject_ids"]):
                raise ValueError("unknown_candidate_subject")
            evidence = locate(refs, target_only=True)
            audit(item["reason"], evidence)
            audits[-1]["candidate_entry_ids"] = [stable_entry(ident) for ident in item["candidate_subject_ids"]]
        except (ValueError, TypeError) as exc:
            audit(str(exc), refs)

    if triples or not data["relations"]:
        for ident, (item, evidence) in candidates.items():
            if ident not in referenced and any(loc["block_ref"] in targets for loc in evidence):
                make_entity(ident, stable_entry(ident))

    result_status = "success_with_results" if entities or triples else "success_empty"
    return ValidationResult(result_status, list(entities.values()), list(triples.values()), audits)
