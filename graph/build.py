# -*- coding: utf-8 -*-
"""
graph/build.py —— ③ 结构化 → 图谱：三元组构建 / 去重 / 冲突处理

输入：extract/rules.py 的实体与候选三元组；输出：规范三元组 + GraphData。
原型阶段用「文件 + SQLite + 前端原生 Canvas 力导向图」，不引入图数据库。

确定性规则（零模型，可复现）：
  - 去重键 (head, relation, tail)；重复项的 evidence_refs 合并，source_hash 取首个；
  - 冲突：同一 (head, relation) 出现多个 tail 时全部保留，并在 notes 里列出，
    **不自动裁决**（古籍异文需要人工 / 本地 LLM 考据）；
  - 任何三元组都保留 page_id / bbox / line_id / quote / source_hash。

TODO（下一阶段）：
  - 引入实体消歧与别名合并，冲突按版本 / 出处加权裁决并留痕；
  - 支持 N 元关系与方剂组成的有序性（君臣佐使）；
  - 图谱规模上去后改为增量构建与分批导出。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from app.schemas import (
    Block, Entity, GraphData, GraphEdge, GraphNode, Triple,
)


def dedup_triples(triples: Iterable[Triple]) -> List[Triple]:
    """按 (head, relation, tail) 去重，合并证据引用。"""
    bucket: Dict[Tuple[str, str, str], Triple] = {}
    for t in triples:
        if not isinstance(t, Triple):
            continue
        key = (t.head, t.relation, t.tail)
        if key not in bucket:
            bucket[key] = Triple(**{**t.to_dict(), "evidence_refs": list(t.evidence_refs)})
        else:
            for ref in t.evidence_refs:
                if ref not in bucket[key].evidence_refs:
                    bucket[key].evidence_refs.append(ref)
    return list(bucket.values())


def find_conflicts(triples: Iterable[Triple]) -> List[Dict[str, str]]:
    """列出同一 (head, relation) 的多值冲突，只报告不裁决。"""
    grouped: Dict[Tuple[str, str], set] = defaultdict(set)
    for t in triples:
        if isinstance(t, Triple):
            grouped[(t.head, t.relation)].add(t.tail)
    return [{"head": h, "relation": r, "tails": " / ".join(sorted(v))}
            for (h, r), v in grouped.items() if len(v) > 1]


def build_triples(entities: Iterable[Entity], blocks: Iterable[Block] | None = None,
                  candidates: Iterable[Triple] | None = None) -> List[Triple]:
    """三元组构建入口。

    candidates 为空时，从 extract/rules.py 依 blocks 现场生成候选（避免循环导入，延迟引入）。
    返回去重后的规范三元组；不抛异常。
    """
    try:
        items = list(candidates) if candidates is not None else []
        if not items and blocks is not None:
            from extract.rules import candidate_triples
            items = candidate_triples(list(blocks))
        return dedup_triples(items)
    except Exception:
        return []


def infer_node_type(name: str, entities: Iterable[Entity]) -> str:
    """由实体表推断节点类型；找不到则按字面做保守推断。"""
    for e in entities:
        if isinstance(e, Entity) and e.name == name:
            return e.type or "实体"
    if name.endswith(("汤", "散", "丸", "方")):
        return "方剂"
    if any(u in name for u in ("两", "钱", "枚", "升", "分", "片")):
        return "剂量"
    if name.startswith(("味", "性")) or name in ("无毒", "有小毒"):
        return "性味"
    return "实体"


def build_graph(triples: Iterable[Triple],
                entities: Iterable[Entity] | None = None) -> GraphData:
    """构图入口：三元组 → nodes / edges（前端可直接渲染）。"""
    entities = list(entities or [])
    triples = [t for t in triples if isinstance(t, Triple)]
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    for t in triples:
        for name in (t.head, t.tail):
            if name not in nodes:
                nodes[name] = GraphNode(
                    id=name, label=name, type=infer_node_type(name, entities),
                    page_id=t.page_id, source_hash=t.source_hash, is_demo=t.is_demo,
                )
            nodes[name].weight += 1
        edges.append(GraphEdge(
            source=t.head, target=t.tail, relation=t.relation,
            evidence_refs=list(t.evidence_refs), page_id=t.page_id,
            source_hash=t.source_hash, is_demo=t.is_demo,
        ))

    relations: Dict[str, int] = defaultdict(int)
    for e in edges:
        relations[e.relation] += 1
    return GraphData(
        nodes=list(nodes.values()), edges=edges, is_demo=True,
        stats={
            "node_count": len(nodes), "edge_count": len(edges),
            "relations": dict(relations),
            "conflicts": find_conflicts(triples),
            "note": "示例数据 / 模型未接入；规模不是评测指标",
        },
    )
