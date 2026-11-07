# -*- coding: utf-8 -*-
"""
graph/export.py —— 导出前端可视化 JSON

输出结构直接给 app/web/index.html 的原生 Canvas 力导向图消费：
    {nodes:[{id,label,type,value}], links:[{source,target,relation,evidence_refs}],
     is_demo, stats}
零外部依赖：不引 D3/ECharts CDN，不联网也能渲染（离线验收要求）。

TODO（下一阶段）：
  - 大图分层抽样、按关系类型过滤、节点聚类配色；
  - 导出标准图格式（GraphML / JSON-LD）并带出处与版本元数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import CONFIG
from app.schemas import GraphData


def to_vis_payload(graph: GraphData, badge: Optional[str] = None) -> Dict[str, Any]:
    """图谱 → 前端渲染负载。badge 由调用方按 is_demo 动态给出。"""
    try:
        payload = graph.to_vis_payload()
        payload["disclaimer"] = CONFIG.disclaimer
        payload["badge"] = badge or CONFIG.demo_badge
        return payload
    except Exception as exc:
        return {"nodes": [], "links": [], "is_demo": True,
                "stats": {"note": f"导出降级：{type(exc).__name__}"},
                "disclaimer": CONFIG.disclaimer, "badge": CONFIG.demo_badge}


def export_graph_json(graph: GraphData, path: Optional[Path] = None) -> Path:
    """落盘到 data/store/graph.json（幂等覆盖，可反复跑）。"""
    out = Path(path) if path else CONFIG.paths.store_dir / "graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_graph_json(path: Optional[Path] = None) -> Optional[GraphData]:
    """读取已导出的图谱；不存在返回 None（调用方决定是否现场构建）。"""
    p = Path(path) if path else CONFIG.paths.store_dir / "graph.json"
    if not p.exists():
        return None
    try:
        return GraphData.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def empty_payload(reason: str = "暂无图谱数据") -> Dict[str, Any]:
    """空图负载，保证前端永远拿得到结构完整的响应。"""
    return {"nodes": [], "links": [], "is_demo": True,
            "stats": {"note": reason}, "disclaimer": CONFIG.disclaimer,
            "badge": CONFIG.demo_badge}
