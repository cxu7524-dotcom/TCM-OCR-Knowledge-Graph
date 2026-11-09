# -*- coding: utf-8 -*-
"""
app/schemas.py —— 全链路统一数据契约（溯源是一等公民）

硬性约束（改造提示词第 4.4 条）：
    任何文本块 Block、实体 Entity、三元组 Triple、答案证据 Evidence
    都必须携带 page_id / bbox / line_id / quote / source_hash。
    字段定义只在本文件出现一次；任何模块返回缺字段的数据视为 bug。

设计取舍：
  - 用标准库 dataclass 而非 pydantic：让 ocr/extract/graph/qa/eval 在
    「没装 FastAPI」的命令行环境（如 python eval/run_all.py）也能独立跑通。
  - 所有字段都有默认值，桩函数因此可以构造结构完整的示例数据，不抛异常。
  - 统一 to_dict / from_dict / validate / provenance，全链路不许丢字段。
  - bbox 统一为 [x1, y1, x2, y2]，原点在图像左上角，单位像素。

类型口径：
  Block.kind      : paragraph | table_cell | title | dose
  Entity.type     : 药材 | 性味 | 归经 | 功效 | 主治 | 禁忌 | 方剂 | 剂量
  Triple.relation : 见 extract/schema.py 的 RELATIONS
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List

# bbox: [x1, y1, x2, y2]
BBox = List[float]
EMPTY_BBOX: BBox = [0.0, 0.0, 0.0, 0.0]

BLOCK_KINDS = ("paragraph", "table_cell", "title", "dose")
ENTITY_TYPES = ("药材", "性味", "归经", "功效", "主治", "禁忌", "方剂", "剂量")


def source_hash_of(*parts: Any) -> str:
    """由若干稳定片段计算来源指纹（sha256 前 16 位），用于幂等入库与溯源绑定。"""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def bbox_valid(bbox: Any) -> bool:
    """bbox 形状校验：四个数，且 x2>=x1、y2>=y1。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return x2 >= x1 and y2 >= y1


def _clean(d: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """from_dict 时丢弃未知键，避免外部 JSON 污染内部契约。"""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in known}


class _Record:
    """所有契约对象的公共能力：序列化 / 反序列化 / 溯源字段校验。"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None):
        return cls(**_clean(d or {}, cls))

    def validate(self) -> List[str]:
        """返回问题清单；空列表表示契约合规。子类可追加字段级规则。"""
        problems: List[str] = []
        for f in fields(self):
            if not hasattr(self, f.name):
                problems.append(f"缺字段 {f.name}")
        if isinstance(self, Provenance):
            if not self.page_id:
                problems.append("缺 page_id")
            if not bbox_valid(self.bbox):
                problems.append("bbox 非法")
            if not self.line_id and getattr(self, "_need_line_id", True):
                problems.append("缺 line_id")
            if not self.quote:
                problems.append("缺 quote")
            if not self.source_hash:
                problems.append("缺 source_hash")
        return problems


@dataclass
class Provenance(_Record):
    """溯源五元组。所有对外产出的文本型对象都继承它。"""

    page_id: str = ""
    bbox: BBox = field(default_factory=lambda: list(EMPTY_BBOX))
    line_id: str = ""
    quote: str = ""
    source_hash: str = ""


@dataclass
class Block(Provenance):
    """① OCR 产出的最小文本块：一行 / 一格 / 一个剂量单元。"""

    col_id: int = 0                 # 竖排古籍的列序号（从右向左递增）
    reading_order: int = 0         # 版面分析给出的阅读顺序
    text: str = ""                 # 原始识别文本
    text_norm: str = ""            # 繁简 / 异体 / 通假归一化后文本
    kind: str = "paragraph"        # paragraph | table_cell | title | dose
    conf: float = 0.0              # OCR 置信度；桩数据固定 0.0 并标注示例
    block_ref: str = ""            # 页内稳定引用，形如 p001:b003

    def __post_init__(self) -> None:
        # 文本块自身即原文：quote 缺省时回填 text，保证溯源字段永不缺失
        if not self.quote:
            self.quote = self.text
        if not self.text_norm:
            self.text_norm = self.text
        if not self.block_ref and self.line_id:
            self.block_ref = f"{self.page_id}:{self.line_id}"

    def validate(self) -> List[str]:
        problems = super().validate()
        if self.kind not in BLOCK_KINDS:
            problems.append(f"kind 非法: {self.kind}")
        if not self.text:
            problems.append("text 为空")
        return problems


@dataclass
class TableStruct(_Record):
    """方剂表 / 剂量表的结构还原结果（表格结构还原指标的对象）。"""

    table_id: str = ""
    page_id: str = ""
    rows: List[List[str]] = field(default_factory=list)
    cells: List[str] = field(default_factory=list)   # Block.block_ref 列表
    source_hash: str = ""
    is_demo: bool = True


@dataclass
class LayoutResult(_Record):
    """版面分析结果：竖排列切分与阅读顺序。"""

    page_id: str = ""
    cols: List[Dict[str, Any]] = field(default_factory=list)
    block_order: List[str] = field(default_factory=list)
    direction: str = "vertical-rtl"
    engine: str = "stub"
    is_demo: bool = True
    note: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult(_Record):
    """ocr/pipeline.py 的完整产出。"""

    page_id: str = ""
    image_path: str = ""
    width: int = 0
    height: int = 0
    blocks: List[Block] = field(default_factory=list)
    tables: List[TableStruct] = field(default_factory=list)
    layout: LayoutResult = field(default_factory=LayoutResult)
    engine: str = "stub"           # rapidocr | paddleocr | stub
    model_loaded: bool = False     # False = 未加载模型（示例数据 / 依赖降级）
    is_demo: bool = True           # True = 示例数据
    source_hash: str = ""
    notes: List[str] = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None):
        d = dict(d or {})
        d["blocks"] = [b if isinstance(b, Block) else Block.from_dict(b)
                       for b in d.get("blocks", [])]
        d["tables"] = [t if isinstance(t, TableStruct) else TableStruct.from_dict(t)
                       for t in d.get("tables", [])]
        lay = d.get("layout")
        d["layout"] = lay if isinstance(lay, LayoutResult) else LayoutResult.from_dict(lay)
        return cls(**_clean(d, cls))

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.page_id:
            problems.append("缺 page_id")
        for i, b in enumerate(self.blocks):
            problems.extend(f"blocks[{i}].{p}" for p in b.validate())
        return problems


@dataclass
class Entity(Provenance):
    """② 结构化抽取的实体。"""

    name: str = ""
    type: str = ""                 # 药材 | 性味 | ... | 剂量
    value: str = ""
    block_ref: str = ""
    extractor: str = "rules"       # rules | llm_local | stub
    is_demo: bool = True

    def __post_init__(self) -> None:
        if not self.quote:
            self.quote = self.value or self.name

    def validate(self) -> List[str]:
        problems = super().validate()
        if not self.name:
            problems.append("name 为空")
        if self.type and self.type not in ENTITY_TYPES:
            problems.append(f"type 非法: {self.type}")
        return problems


@dataclass
class Triple(Provenance):
    """③ 知识图谱三元组：head -relation-> tail，携带原文证据。"""

    head: str = ""
    relation: str = ""
    tail: str = ""
    evidence_refs: List[str] = field(default_factory=list)  # Block.block_ref / Evidence.quote
    is_demo: bool = True

    def __post_init__(self) -> None:
        if not self.quote and self.evidence_refs:
            self.quote = self.evidence_refs[0]

    def validate(self) -> List[str]:
        problems = super().validate()
        if not (self.head and self.relation and self.tail):
            problems.append("head/relation/tail 不全")
        if not self.evidence_refs:
            problems.append("缺 evidence_refs")
        return problems


@dataclass
class GraphNode(_Record):
    id: str = ""
    label: str = ""
    type: str = ""                 # 药材 | 方剂 | 主治 | ...
    weight: int = 1
    page_id: str = ""
    source_hash: str = ""
    is_demo: bool = True


@dataclass
class GraphEdge(_Record):
    source: str = ""
    target: str = ""
    relation: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    page_id: str = ""
    source_hash: str = ""
    is_demo: bool = True


@dataclass
class GraphData(_Record):
    """前端力导向图可直接消费的 nodes / links 结构。"""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    is_demo: bool = True
    stats: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None):
        d = dict(d or {})
        d["nodes"] = [n if isinstance(n, GraphNode) else GraphNode.from_dict(n)
                      for n in d.get("nodes", [])]
        d["edges"] = [e if isinstance(e, GraphEdge) else GraphEdge.from_dict(e)
                      for e in d.get("edges", [])]
        return cls(**_clean(d, cls))

    def to_vis_payload(self) -> Dict[str, Any]:
        """输出 {nodes:[{id,label,type,value}], links:[{source,target,relation}]}。"""
        return {
            "nodes": [{"id": n.id, "label": n.label, "type": n.type, "value": n.weight}
                      for n in self.nodes],
            "links": [{"source": e.source, "target": e.target, "relation": e.relation,
                       "evidence_refs": e.evidence_refs} for e in self.edges],
            "is_demo": self.is_demo,
            "stats": self.stats,
        }


@dataclass
class Evidence(Provenance):
    """④ 问答证据：必须能在原文 Block.text 中命中，否则答案转拒答。"""

    score: float = 0.0
    block_ref: str = ""
    channel: str = "sparse"        # sparse | vector | graph
    is_demo: bool = True

    def validate(self) -> List[str]:
        problems = super().validate()
        if not self.quote:
            problems.append("quote 为空")
        return problems


@dataclass
class Answer(_Record):
    """问答最终产物。所有输出（含拒答）都带免责声明与 trace。"""

    text: str = ""
    evidences: List[Evidence] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str = ""
    disclaimer: str = "仅供参考、非诊疗建议"
    trace: List[str] = field(default_factory=list)
    backend: str = "stub"
    is_demo: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None):
        d = dict(d or {})
        d["evidences"] = [e if isinstance(e, Evidence) else Evidence.from_dict(e)
                          for e in d.get("evidences", [])]
        return cls(**_clean(d, cls))

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.text:
            problems.append("text 为空")
        if not self.disclaimer:
            problems.append("缺 disclaimer")
        for i, e in enumerate(self.evidences):
            problems.extend(f"evidences[{i}].{p}" for p in e.validate())
        return problems


# ------------------------------------------------------------------
# 全链路校验工具：供 qa/generate.py 与 eval/audit.py 复用
# ------------------------------------------------------------------

def norm_for_match(text: str) -> str:
    """比对用的轻量归一化：去空白与常见标点。仅用于命中判定，不改原文。"""
    table = str.maketrans("", "", " \t\r\n\u3000，。、；：,.;:（）()《》「」『』\"'")
    return (text or "").translate(table)


def align_norm(text: Any) -> str:
    """**唯一的**对齐归一化函数（GT 与预测必须对称调用，禁止各写一份）。

    顺序：去空白与常见标点（norm_for_match）→ 异体字表映射 → 繁简映射。
    只用于「判定两个字段是否等价」，不改写 GT 原文、不改写预测输出。
    口径说明见 data/annotation/标注指南.md 第四节。
    """
    base = norm_for_match("" if text is None else str(text))
    if not base:
        return ""
    try:
        # 延迟导入：ocr.normalize 顶部依赖本模块，函数内导入避免循环
        from ocr.normalize import normalize_text
        return normalize_text(base)
    except Exception:
        return base


def entity_align_key(rec: Any, normalized: bool = True) -> tuple:
    """字段对齐键的**唯一实现**：`(page_id, type, name, value)`。

    `eval/dataset.py:align_key()` 与 `extract/evaluate.py:align_key()` 都委托到这里，
    避免口径漂移；`normalized=False` 只用于报告「修正前」对照。
    """
    get = rec.get if isinstance(rec, dict) else (lambda k, d=None: getattr(rec, k, d))
    page = str(get("page_id", "") or "")
    etype = str(get("type", "") or "")
    name = str(get("name", "") or "")
    value = str(get("value", "") or "")
    if not normalized:
        return (page, etype, name, value)
    return (page, align_norm(etype), align_norm(name), align_norm(value))


def quote_hits_blocks(quote: str, blocks: Iterable[Block], normalized: bool = True) -> bool:
    """证据摘抄是否能在任一原文块的文本中命中。命中判定不允许模型自由发挥。"""
    q = norm_for_match(quote) if normalized else (quote or "")
    if not q:
        return False
    for b in blocks:
        t = norm_for_match(b.text) if normalized else b.text
        if not t:
            continue
        if q in t or t in q:
            return True
    return False


def validate_chain(items: Iterable[Any]) -> List[str]:
    """批量校验全链路对象；空列表表示全部合规。"""
    problems: List[str] = []
    for i, item in enumerate(items):
        if hasattr(item, "validate"):
            problems.extend(f"[{i}] {p}" for p in item.validate())
        else:
            problems.append(f"[{i}] 非契约对象: {type(item).__name__}")
    return problems


def records_to_dicts(items: Iterable[Any]) -> List[Dict[str, Any]]:
    """统一序列化，便于 JSON 落盘与接口返回。"""
    return [i.to_dict() if hasattr(i, "to_dict") else dict(i) for i in items]
