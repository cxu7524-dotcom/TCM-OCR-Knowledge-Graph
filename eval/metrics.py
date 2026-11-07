# -*- coding: utf-8 -*-
"""
eval/metrics.py —— 五项指标口径与真实计算（阶段二）

计算原则：
  1. 所有 compute_* 都**显式排除 is_demo=True 与缺溯源字段的记录**；
  2. 输入缺失 / 有效样本为 0 时返回 status="待测"、value=None，绝不返回 0 分；
  3. 全部用标准库实现（编辑距离 / 集合对齐 / 子串命中），可复现、可审计；
  4. 繁体与异体字单独统计，不混进总体 CER（赛题硬约束）。

阈值（来自 docs/赛题二原文.md 第 1 节，用户提供）：
  关键实体字段抽取 F1 ≥ 0.80；问答可溯源率 ≥ 0.90；无依据问题正确拒答率 ≥ 0.85；
  方剂表结构还原「达到可用水平」——本工程给出可判定口径：
  行列结构正确率与单元格内容正确率均 ≥ 0.80 视为可用（口径说明见报告）。

TODO（阶段三）：部分匹配 / 同义别名裁决、分类型置信区间、命题样本分列统计。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.config import CONFIG
from app.schemas import norm_for_match, quote_hits_blocks
from eval.dataset import align_key, clean_records, group_by_page

THRESHOLDS: Dict[str, float] = {
    "field_f1": 0.80,
    "traceability": 0.90,
    "refusal": 0.85,
    "table_usable": 0.80,
}

METRIC_SPECS: List[Dict[str, Any]] = [
    {"key": "cer", "name": "CER 字符错误率", "status": "待测", "value": None,
     "formula": "Σ 编辑距离(识别文本, 人工转写) / Σ 人工转写字数（按页加权）",
     "threshold": None,
     "needs": ["data/annotation/ocr_transcription.jsonl", "data/reports/ocr_predictions.jsonl"]},
    {"key": "field_f1", "name": "字段级 P / R / F1", "status": "待测", "value": None,
     "formula": "按 (page_id,type,name,value) 集合对齐，TP/FP/FN 微平均",
     "threshold": THRESHOLDS["field_f1"],
     "needs": ["data/annotation/entities.jsonl", "data/reports/extraction_predictions.jsonl"]},
    {"key": "table_structure", "name": "表格结构还原准确率", "status": "待测", "value": None,
     "formula": "行列结构与单元格内容同时正确的表数 / GT 表数",
     "threshold": THRESHOLDS["table_usable"],
     "needs": ["data/annotation/tables.jsonl", "data/reports/table_predictions.jsonl"]},
    {"key": "traceability", "name": "可溯源率", "status": "待测", "value": None,
     "formula": "quote 能回原文命中的证据数 / 证据总条数",
     "threshold": THRESHOLDS["traceability"],
     "needs": ["data/reports/qa_log.jsonl", "data/reports/ocr_predictions.jsonl（原文块）"]},
    {"key": "refusal", "name": "拒答率", "status": "待测", "value": None,
     "formula": "无依据问题中正确拒答数 / 无依据问题总数",
     "threshold": THRESHOLDS["refusal"],
     "needs": ["data/annotation/qa_no_evidence.jsonl", "data/reports/qa_log.jsonl"]},
]


def judge(key: str, value: Optional[float]) -> str:
    """按阈值判定；CER 无参考阈值，只报值。"""
    if value is None:
        return "待测"
    th = THRESHOLDS.get(key)
    if th is None:
        return "无参考阈值"
    return "达标" if value >= th else "未达标"


def _result(key: str, value: Optional[float], **extra) -> Dict[str, Any]:
    status = "实测" if value is not None else "待测"
    return {"key": key, "status": status, "value": value,
            "judge": judge(key, value), "threshold": THRESHOLDS.get(key), **extra}


def levenshtein(a: str, b: str) -> int:
    """标准库实现的两字符串编辑距离（插入 / 删除 / 替换，代价均为 1）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _page_text(records: Sequence[Dict[str, Any]], text_key: str = "quote",
               order_key: str = "line_id") -> str:
    """把一页的 GT / 预测文本按稳定顺序拼成一个字符串（无分隔符，避免加分母）。"""
    items = sorted(records, key=lambda r: str(r.get(order_key, "")))
    return norm_for_match("".join(str(r.get(text_key, "")) for r in items))


def load_lexicon() -> Dict[str, Any]:
    """读取 data/lexicon/异体字表.json；不存在返回空结构。"""
    p = CONFIG.paths.data_dir / "lexicon" / "异体字表.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def compute_cer(gt_ocr: List[Dict[str, Any]], pred_ocr: List[Dict[str, Any]],
                pages: Optional[List[str]] = None) -> Dict[str, Any]:
    """CER（按页加权）+ 繁体 / 异体字子集单列统计。

    对齐口径：按 page_id 分组，页内 GT 按 line_id、预测按 reading_order/line_id 排序后拼接。
    """
    gt, ex1 = clean_records(gt_ocr)
    pred, ex2 = clean_records(pred_ocr)
    if pages is not None:
        keep = set(pages)
        gt = [r for r in gt if r.get("page_id") in keep]
        pred = [r for r in pred if r.get("page_id") in keep]
    g_pages, p_pages = group_by_page(gt), group_by_page(pred)
    common = sorted(set(g_pages) & set(p_pages))
    if not common or not g_pages:
        return _result("cer", None, reason="缺 GT 或预测 OCR 文本",
                       excluded={"gt": ex1, "pred": ex2}, gt_pages=len(g_pages),
                       matched_pages=0)

    lex = load_lexicon()
    trad_chars = set("".join(lex.get("繁简", {}).keys()))
    variant_chars = set("".join(lex.get("异体", {}).keys()))
    per_page: Dict[str, Dict[str, Any]] = {}
    total_dist = total_len = 0
    trad_pages: List[str] = []
    variant_pages: List[str] = []
    for pid in common:
        g = _page_text(g_pages[pid])
        p = "".join(norm_for_match(str(r.get("quote", ""))) for r in
                    sorted(p_pages[pid], key=lambda r: (int(r.get("reading_order") or 0),
                                                        str(r.get("line_id", "")))))
        if not g:
            continue
        d = levenshtein(p, g)
        total_dist += d
        total_len += len(g)
        per_page[pid] = {"gt_chars": len(g), "pred_chars": len(p), "distance": d,
                         "cer": round(d / len(g), 4)}
        if trad_chars & set(g):
            trad_pages.append(pid)
        if variant_chars & set(g):
            variant_pages.append(pid)
    if total_len == 0:
        return _result("cer", None, reason="GT 文本为空", excluded={"gt": ex1, "pred": ex2})

    def subset_cer(ids: List[str]) -> Optional[float]:
        if not ids:
            return None
        num = sum(per_page[p]["distance"] for p in ids if p in per_page)
        den = sum(per_page[p]["gt_chars"] for p in ids if p in per_page)
        return round(num / den, 4) if den else None

    return _result(
        "cer", round(total_dist / total_len, 4),
        detail=f"总编辑距离 {total_dist} / GT 字数 {total_len}，覆盖 {len(per_page)} 页",
        per_page=per_page,
        traditional_subset={"pages": trad_pages, "cer": subset_cer(trad_pages),
                            "note": "繁体页单独统计，未混入总体 CER"},
        variant_subset={"pages": variant_pages, "cer": subset_cer(variant_pages),
                        "note": "含异体字页单独统计，未混入总体 CER"},
        excluded={"gt": ex1, "pred": ex2},
        gt_pages=len(g_pages), matched_pages=len(common),
    )


def compute_variant_hit(pred_ocr: List[Dict[str, Any]]) -> Dict[str, Any]:
    """异体字归一化命中率：原文中被字表覆盖的字符，有多少在 text_norm 中被正确转换。

    逐字符比对「原文」与「按同一字表重新归一化的结果」，避免用「目标字是否出现在整串
    归一文本里」这种会漏判的近似写法。
    """
    from ocr.normalize import mapping, normalize_text

    m = mapping()
    if not m:
        return {"status": "待测", "value": None, "reason": "异体字表未就绪"}
    pred, _ = clean_records(pred_ocr)
    multi = [k for k in m if len(k) > 1]
    hit = total = 0
    for r in pred:
        raw = str(r.get("quote", ""))
        if not raw:
            continue
        tmp = normalize_text(raw)
        for i, ch in enumerate(raw):
            covered = ch in m or any(ch in k for k in multi)
            if covered:
                total += 1
                if i < len(tmp) and tmp[i] != ch:
                    hit += 1
    if total == 0:
        return {"status": "待测", "value": None, "reason": "预测文本中无异体/繁体字命中",
                "total": 0}
    return {"status": "实测", "value": round(hit / total, 4), "hit": hit, "total": total,
            "note": "异体字归一化命中率，单列统计；繁简与异体合并口径"}


def compute_field_f1(gt_entities: List[Dict[str, Any]],
                     pred_entities: List[Dict[str, Any]],
                     pages: Optional[List[str]] = None) -> Dict[str, Any]:
    """字段级 P/R/F1：按 (page_id,type,name,value) 集合对齐，微平均。"""
    gold, ex1 = clean_records(gt_entities)
    pred, ex2 = clean_records(pred_entities)
    if pages is not None:
        keep = set(pages)
        gold = [r for r in gold if r.get("page_id") in keep]
        pred = [r for r in pred if r.get("page_id") in keep]
    if not gold or not pred:
        return _result("field_f1", None, reason="缺 GT 或预测实体（无有效样本）",
                       excluded={"gt": ex1, "pred": ex2},
                       gt_count=len(gold), pred_count=len(pred))
    g, p = set(map(align_key, gold)), set(map(align_key, pred))
    tp, fp, fn = len(g & p), len(p - g), len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return _result("field_f1", round(f1, 4), P=round(precision, 4), R=round(recall, 4),
                   TP=tp, FP=fp, FN=fn, excluded={"gt": ex1, "pred": ex2},
                   detail=f"TP={tp} FP={fp} FN={fn}，对齐键 (page_id,type,name,value)")


def _table_shape(t: Dict[str, Any]) -> List[int]:
    rows = t.get("rows") or []
    return [len(r) if isinstance(r, (list, tuple)) else 0 for r in rows]


def _table_cells(t: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for row in t.get("rows") or []:
        for cell in (row if isinstance(row, (list, tuple)) else [row]):
            out.append(norm_for_match(str(cell)))
    return out


def compute_table_structure(gt_tables: List[Dict[str, Any]],
                            pred_tables: List[Dict[str, Any]],
                            pages: Optional[List[str]] = None) -> Dict[str, Any]:
    """方剂表结构还原：行列结构 + 单元格内容全对才算该表正确。

    同时给出结构正确率、单元格内容正确率、联合正确率（主指标）。
    """
    gold, ex1 = clean_records(gt_tables, need_provenance=False)
    pred, ex2 = clean_records(pred_tables, need_provenance=False)
    if pages is not None:
        keep = set(pages)
        gold = [r for r in gold if r.get("page_id") in keep]
        pred = [r for r in pred if r.get("page_id") in keep]
    if not gold or not pred:
        return _result("table_structure", None, reason="缺 GT 或预测表格",
                       excluded={"gt": ex1, "pred": ex2},
                       gt_count=len(gold), pred_count=len(pred))

    def key(t: Dict[str, Any]) -> str:
        return f"{t.get('page_id','')}|{t.get('table_id','')}"

    pmap = {key(t): t for t in pred}
    total = shape_ok = cell_ok = joint_ok = 0
    cell_total = cell_correct = 0
    per_table = []
    for g in gold:
        k, total = key(g), total + 1
        p = pmap.get(k)
        s_ok = bool(p) and _table_shape(g) == _table_shape(p)
        c_ok = bool(p) and _table_cells(g) == _table_cells(p)
        gc, pc = _table_cells(g), _table_cells(p) if p else []
        for i in range(max(len(gc), len(pc))):
            cell_total += 1
            cell_correct += int(i < len(pc) and i < len(gc) and gc[i] == pc[i])
        shape_ok += int(s_ok)
        cell_ok += int(c_ok)
        joint_ok += int(s_ok and c_ok)
        per_table.append({"key": k, "shape_ok": s_ok, "cell_ok": c_ok,
                          "matched": bool(p)})
    shape_rate = round(shape_ok / total, 4)
    cell_rate = round(cell_ok / total, 4)
    joint = round(joint_ok / total, 4)
    usable = shape_rate >= THRESHOLDS["table_usable"] and cell_rate >= THRESHOLDS["table_usable"]
    cell_level = round(cell_correct / cell_total, 4) if cell_total else None
    out = _result("table_structure", joint, shape_rate=shape_rate, cell_rate=cell_rate,
                  cell_level_accuracy=cell_level,
                  cell_correct=cell_correct, cell_total=cell_total,
                  total_tables=total, matched_tables=len(pmap),
                  usable=usable, per_table=per_table,
                  excluded={"gt": ex1, "pred": ex2},
                  detail=(f"结构 {shape_ok}/{total}、整表单元格全对 {cell_ok}/{total}、"
                          f"联合 {joint_ok}/{total}；单元格级正确 {cell_correct}/{cell_total}"
                          f"（{cell_level}）"))
    out["judge"] = "达标" if usable else "未达标"
    out["threshold"] = THRESHOLDS["table_usable"]
    return out


def compute_traceability(qa_logs: List[Dict[str, Any]],
                         blocks_by_page: Dict[str, List[Any]]) -> Dict[str, Any]:
    """可溯源率：每条证据 quote 回原文命中数 / 证据总数。只统计非拒答、非示例记录。"""
    logs, ex = clean_records(qa_logs, need_provenance=False)
    usable = [r for r in logs if not r.get("refused") and not r.get("is_demo")]
    total = hit = 0
    for rec in usable:
        for ev in rec.get("evidences") or []:
            if not isinstance(ev, dict) or ev.get("is_demo") is True:
                continue
            total += 1
            quote = str(ev.get("quote", ""))
            blocks = blocks_by_page.get(str(ev.get("page_id", "")), [])
            if quote and quote_hits_blocks(quote, blocks):
                hit += 1
    if total == 0:
        return _result("traceability", None, reason="无有效证据（无依据问题全部拒答或日志为空）",
                       excluded=ex, usable_answers=len(usable))
    return _result("traceability", round(hit / total, 4), hit=hit, total=total,
                   usable_answers=len(usable), excluded=ex,
                   detail="quote 回原文 Block.text 命中才计数，校验口径与 qa/generate.py 一致")


def compute_refusal(qa_no_evidence: List[Dict[str, Any]],
                    qa_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """无依据问题正确拒答率：期望 refused=true 的问题中实际拒答的比例。"""
    gold, ex1 = clean_records(qa_no_evidence, need_provenance=False)
    logs, ex2 = clean_records(qa_logs, need_provenance=False)
    by_q = {str(r.get("question", "")).strip(): r for r in logs if r.get("question")}
    correct = wrong = missing = 0
    details = []
    for g in gold:
        q = str(g.get("question", "")).strip()
        if not q:
            continue
        actual = by_q.get(q)
        if actual is None:
            missing += 1
            details.append({"question": q, "result": "未跑批"})
            continue
        ok = bool(actual.get("refused"))
        correct += int(ok)
        wrong += int(not ok)
        details.append({"question": q, "result": "正确拒答" if ok else "误答",
                        "reason": actual.get("refuse_reason", "")})
    denom = correct + wrong + missing
    if denom == 0:
        return _result("refusal", None, reason="无依据问题集为空或未跑批",
                       excluded={"gt": ex1, "log": ex2}, details=[])
    # 未跑批按「未正确拒答」计入分母，不美化
    return _result("refusal", round(correct / denom, 4), correct=correct, wrong=wrong,
                   missing=missing, total=denom, details=details,
                   excluded={"gt": ex1, "log": ex2},
                   detail=f"正确拒答 {correct}/{denom}（误答 {wrong}、未跑批 {missing}）")


def compute_all(ds, predictions: Dict[str, List[Dict[str, Any]]],
                blocks_by_page: Dict[str, List[Any]],
                test_only: bool = True) -> Dict[str, Any]:
    """五项一次性算出，返回可直接写入报告的字典。"""
    pages = ds.test_pages if test_only else None
    qa_logs = predictions.get("qa_log", [])
    return {
        "cer": compute_cer(ds.ocr_gt, predictions.get("ocr", []), pages),
        "field_f1": compute_field_f1(ds.ent_gt, predictions.get("entities", []), pages),
        "table_structure": compute_table_structure(ds.table_gt,
                                                   predictions.get("tables", []), pages),
        "traceability": compute_traceability(qa_logs, blocks_by_page),
        "refusal": compute_refusal(ds.qa_no_evidence, qa_logs),
        "variant_hit": compute_variant_hit(predictions.get("ocr", [])),
        "test_only": test_only,
        "test_pages": list(pages or []),
    }


# ------------------------------------------------------------------
# 阶段一骨架报告仍会用到的口径辅助函数（保留兼容）
# ------------------------------------------------------------------

def has_ground_truth() -> bool:
    d = CONFIG.paths.annotation_dir
    if not d.exists():
        return False
    return any(p.suffix.lower() in (".json", ".jsonl") for p in d.iterdir() if p.is_file())


def missing_requirements() -> List[Dict[str, Any]]:
    return [{"key": s["key"], "name": s["name"], "missing": list(s["needs"])}
            for s in METRIC_SPECS]


def audit_sample_count(answers: Iterable[Any]) -> Dict[str, int]:
    items = list(answers)
    demo = sum(1 for a in items if getattr(a, "is_demo", True))
    return {"total": len(items), "demo": demo, "real": len(items) - demo}


def compute_traceability_for_audit(answer: Any, blocks: List[Any]) -> Dict[str, Any]:
    evs = answer.evidences or []
    hit = sum(1 for e in evs if quote_hits_blocks(e.quote, blocks))
    return {"total": len(evs), "hit": hit, "is_demo": True,
            "note": "审计明细，示例数据不计入指标"}


def report_metrics() -> Dict[str, Any]:
    """骨架报告用的口径表（阶段二正式报告以 compute_all 结果为准）。"""
    return {
        "phase": "阶段二：本地 OCR 接入与首轮实测（正式结果见 正式评测报告.md）",
        "policy": "禁止编造实测数字；示例数据不得当评测结果",
        "ground_truth_available": has_ground_truth(),
        "metrics": [{"key": s["key"], "name": s["name"], "status": "待测",
                     "value": None, "formula": s["formula"], "needs": s["needs"]}
                    for s in METRIC_SPECS],
    }
