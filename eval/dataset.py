# -*- coding: utf-8 -*-
"""
eval/dataset.py —— GT 加载 / 校验 / 对齐 / 切分 / 版本哈希

设计目标：让评测报告里的每个数字都能追到具体输入文件与 GT 版本。
  - 六类 GT 统一用 JSONL，字段口径严格对齐 data/annotation/README.md；
  - 缺字段 / 缺溯源五元组 / is_demo=True 的记录视为无效，直接丢弃并计数；
  - 按 page_id 做确定性切分（固定种子），训练页与测试页不交叉；
  - 版本哈希 = 六类 GT 文件内容拼接后的 sha256 前 16 位，报告必须注明。

命令行：
    python -m eval.dataset --check          校验六类 GT 是否齐备、字段是否合规
    python -m eval.dataset --version        打印 GT 版本哈希
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.config import CONFIG

GT_DIR = CONFIG.paths.annotation_dir
GT_FILES = {
    "ocr": "ocr_transcription.jsonl",
    "entities": "entities.jsonl",
    "relations": "relations.jsonl",
    "tables": "tables.jsonl",
    "qa_gold": "qa_log_gold.jsonl",
    "qa_no_evidence": "qa_no_evidence.jsonl",
}
PROVENANCE = ("page_id", "bbox", "line_id", "quote", "source_hash")
SPLIT_SEED = 2026
TEST_RATIO = 0.30


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL；不存在 / 行格式错误时返回空列表或跳过坏行，不抛异常。"""
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    except OSError:
        return []
    return out


def is_valid_record(rec: Dict[str, Any], need_provenance: bool = True) -> Tuple[bool, str]:
    """单条 GT / 预测记录的合规性：必须带溯源五元组，且不得是示例数据。"""
    if not isinstance(rec, dict):
        return False, "非字典"
    if rec.get("is_demo") is True:
        return False, "示例数据"
    if need_provenance:
        for k in PROVENANCE:
            if k not in rec or rec[k] in (None, "", []):
                return False, f"缺 {k}"
        bbox = rec.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False, "bbox 非 4 元"
    return True, ""


def clean_records(records: Iterable[Dict[str, Any]],
                  need_provenance: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """过滤无效记录并返回（有效记录, 被排除条数）。"""
    ok, excluded = [], 0
    for r in records:
        valid, _ = is_valid_record(r, need_provenance)
        if valid:
            ok.append(r)
        else:
            excluded += 1
    return ok, excluded


def version_hash(gt_dir: Path = GT_DIR) -> str:
    """六类 GT 文件内容 → 版本哈希（只取前 16 位，便于报告书写）。"""
    h = hashlib.sha256()
    for name in GT_FILES.values():
        p = gt_dir / name
        if p.exists():
            h.update(name.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def split_pages(page_ids: Iterable[str], test_ratio: float = TEST_RATIO,
                seed: int = SPLIT_SEED) -> Tuple[List[str], List[str]]:
    """按 page_id 的确定性哈希切分训练 / 测试页，同一 GT 版本结果恒定。"""
    uniq = sorted(set(str(p) for p in page_ids if p))
    scored = []
    for pid in uniq:
        v = int(hashlib.sha256(f"{seed}|{pid}".encode("utf-8")).hexdigest()[:8], 16)
        scored.append((v / 0xFFFFFFFF, pid))
    scored.sort()
    n_test = max(1, int(round(len(scored) * test_ratio))) if scored else 0
    test = [p for _, p in scored[:n_test]]
    train = [p for _, p in scored[n_test:]]
    return train, test


@dataclass
class Dataset:
    """一次评测所用 GT 与预测的集合。"""

    gt_dir: Path = GT_DIR
    version: str = ""
    ocr_gt: List[Dict[str, Any]] = field(default_factory=list)
    ent_gt: List[Dict[str, Any]] = field(default_factory=list)
    rel_gt: List[Dict[str, Any]] = field(default_factory=list)
    table_gt: List[Dict[str, Any]] = field(default_factory=list)
    qa_gold: List[Dict[str, Any]] = field(default_factory=list)
    qa_no_evidence: List[Dict[str, Any]] = field(default_factory=list)
    train_pages: List[str] = field(default_factory=list)
    test_pages: List[str] = field(default_factory=list)
    excluded: Dict[str, int] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, gt_dir: Path | None = None) -> "Dataset":
        d = gt_dir or GT_DIR
        ds = cls(gt_dir=d, version=version_hash(d))
        for key, name in GT_FILES.items():
            p = d / name
            if not p.exists() or p.stat().st_size == 0:
                ds.missing.append(name)
            raw = load_jsonl(p)
            # 表格 / 问题集不强制带 bbox，其余必须带溯源五元组
            need_prov = key in ("ocr", "entities", "relations")
            cleaned, excluded = clean_records(raw, need_provenance=need_prov)
            setattr(ds, {
                "ocr": "ocr_gt", "entities": "ent_gt", "relations": "rel_gt",
                "tables": "table_gt", "qa_gold": "qa_gold",
                "qa_no_evidence": "qa_no_evidence",
            }[key], cleaned)
            ds.excluded[name] = excluded
        pages = [r.get("page_id") for r in ds.ocr_gt if r.get("page_id")]
        if not pages:
            pages = [r.get("page_id") for r in ds.ent_gt if r.get("page_id")]
        ds.train_pages, ds.test_pages = split_pages(pages)
        return ds

    def check(self) -> Dict[str, Any]:
        """GT 齐备性与字段校验结果，供报告与 CLI 使用。"""
        counts = {k: len(getattr(self, v)) for k, v in {
            "ocr": "ocr_gt", "entities": "ent_gt", "relations": "rel_gt",
            "tables": "table_gt", "qa_gold": "qa_gold",
            "qa_no_evidence": "qa_no_evidence"}.items()}
        return {
            "version": self.version,
            "gt_dir": str(self.gt_dir),
            "missing_files": self.missing,
            "counts": counts,
            "excluded_demo_or_invalid": self.excluded,
            "pages_total": len(set(self.train_pages) | set(self.test_pages)),
            "train_pages": len(self.train_pages),
            "test_pages": len(self.test_pages),
            "ready": not self.missing and counts["ocr"] > 0,
        }


def align_key(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """字段对齐键 (page_id, type, name, value)，与 extract/evaluate.py 口径一致。"""
    return (str(rec.get("page_id", "")), str(rec.get("type", "")),
            str(rec.get("name", "")), str(rec.get("value", "")))


def group_by_page(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        out.setdefault(str(r.get("page_id", "")), []).append(r)
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="GT 数据集校验")
    ap.add_argument("--check", action="store_true", help="校验六类 GT")
    ap.add_argument("--version", action="store_true", help="打印 GT 版本哈希")
    args = ap.parse_args()
    ds = Dataset.load()
    if args.version:
        print(ds.version)
    else:
        print(json.dumps(ds.check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
