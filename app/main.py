# -*- coding: utf-8 -*-
"""
app/main.py —— FastAPI 入口与路由（四步演示台的服务端）

承接旧 `agent/api_server.py` 的 FastAPI 骨架、上传与静态资源组织方式，
剥离客服 / 聊天室 / 三角色登录 / 模板填表等全部旧业务路由。

最小接口：
    GET  /                        四步演示台页面
    GET  /api/health              配置与模块状态（不含任何密钥）
    GET  /api/meta                免责文案 + 示例问题 + 步骤说明
    POST /api/pages               上传古籍页（或 demo=1 走示例数据）
    GET  /api/pages               页面清单
    GET  /api/pages/{id}/ocr      OCR 与结构化结果
    GET  /api/graph               图谱可视化 JSON
    POST /api/qa                  可溯源问答
    GET  /uploads/{name}          上传页预览
    GET  /docs                    FastAPI 自带接口文档

运行：python app/main.py [--port 8100]  或  双击 启动操作台.bat
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许 `python app/main.py` 直接启动（此时 sys.path[0] 是 app/）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse                  # noqa: E402
from fastapi.staticfiles import StaticFiles                              # noqa: E402

from app.config import CONFIG                                            # noqa: E402
from app.demo import demo_flags, load_demo_ocr, load_demo_questions        # noqa: E402
from app.schemas import Answer, Block, GraphData, OCRResult                # noqa: E402
from app import store                                                     # noqa: E402
from extract.rules import extract_all                                      # noqa: E402
from graph.build import build_graph, build_triples                         # noqa: E402
from graph.export import (                                                 # noqa: E402
    empty_payload, export_graph_json, load_graph_json, to_vis_payload,
)
from ocr.pipeline import new_page_id, run_demo, run_ocr                    # noqa: E402
from qa.disclaimer import DISCLAIMER, page_banner                          # noqa: E402
from qa.generate import answer as qa_answer                                # noqa: E402
from qa.index import build_index                                           # noqa: E402

app = FastAPI(
    title="中医古籍 OCR 与知识图谱智能问答原型",
    description="四步链路：上传古籍页 → 显示 OCR 与结构化结果 → 图谱可视化 → 可溯源问答。"
                "仅供参考、非诊疗建议。当前阶段为示例数据 / 模型未接入。",
    version="0.2.0-framework",
)

# 进程内演示状态（原型阶段够用；真实数据以 SQLite / JSON 镜像为准）
_STATE: Dict[str, Any] = {
    "page_id": None,
    "blocks": [],
    "entities": [],
    "triples": [],
    "graph": None,
    "vector_index": None,
    "last_ingest": None,
}


# ------------------------------------------------------------------
# 内部：一次完整入库（OCR → 抽取 → 图谱 → 存储）
# ------------------------------------------------------------------

def _ingest(ocr: OCRResult) -> Dict[str, Any]:
    """把 OCRResult 走完 ②③④ 的构建并幂等入库，更新演示状态。"""
    entities, candidates = extract_all(ocr.blocks)
    triples = build_triples(entities, ocr.blocks, candidates)
    graph = build_graph(triples, entities)
    stat = store.save_page(ocr, entities, triples)
    export_graph_json(graph)

    _STATE.update({
        "page_id": ocr.page_id,
        "blocks": ocr.blocks,
        "entities": entities,
        "triples": triples,
        "graph": graph,
        "vector_index": build_index(ocr.blocks),
        "last_ingest": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {
        "page_id": ocr.page_id,
        "blocks": len(ocr.blocks),
        "entities": len(entities),
        "triples": len(triples),
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "ingest": stat,
        "engine": ocr.engine,
        "model_loaded": ocr.model_loaded,
        "notes": ocr.notes,
        **demo_flags(),
        "disclaimer": DISCLAIMER,
    }


def _blocks_for(page_id: Optional[str]) -> List[Block]:
    """取问答用的原文块：优先内存状态，其次 SQLite，最后示例数据兜底。"""
    if page_id and page_id == _STATE["page_id"] and _STATE["blocks"]:
        return _STATE["blocks"]
    if page_id:
        rows = store.get_blocks(page_id)
        if rows:
            return rows
    if _STATE["blocks"]:
        return _STATE["blocks"]
    return load_demo_ocr("DEMO-001").blocks


def _current_graph() -> Optional[GraphData]:
    if _STATE["graph"] is not None:
        return _STATE["graph"]
    return load_graph_json()


# ------------------------------------------------------------------
# 页面与静态资源
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(CONFIG.paths.web_dir / "index.html", media_type="text/html; charset=utf-8")


CONFIG.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(CONFIG.paths.uploads_dir)), name="uploads")
app.mount("/static", StaticFiles(directory=str(CONFIG.paths.web_dir)), name="static")


# ------------------------------------------------------------------
# 元信息
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    """配置快照与模块状态；不含密钥、不含任何实测指标。"""
    try:
        db = store.stats()
    except Exception as exc:
        db = {"error": type(exc).__name__}
    return {
        "status": "ok",
        "config": CONFIG.summary(),
        "db": db,
        "current_page_id": _STATE["page_id"],
        "last_ingest": _STATE["last_ingest"],
        "modules": {
            "ocr": {"backend": CONFIG.ocr.backend, "model_loaded": False, "is_demo": True},
            "extract": {"backend": "rules", "llm": CONFIG.llm.backend,
                        "llm_loaded": False, "is_demo": True},
            "graph": {"nodes": len(_STATE["graph"].nodes) if _STATE["graph"] else 0,
                      "is_demo": True},
            "qa": {"backend": CONFIG.llm.backend, "vector": CONFIG.vector.backend,
                   "is_demo": True},
        },
        "metrics": "待测（模型未接入、标注未采集）",
        **demo_flags(),
    }


@app.get("/api/meta")
def meta():
    """前端常驻文案、示例问题与四步说明。"""
    return {
        **page_banner(),
        "steps": [
            {"id": 1, "name": "上传古籍页", "api": "POST /api/pages"},
            {"id": 2, "name": "识别与结构化", "api": "GET /api/pages/{id}/ocr"},
            {"id": 3, "name": "图谱可视化", "api": "GET /api/graph"},
            {"id": 4, "name": "可溯源问答", "api": "POST /api/qa"},
        ],
        "questions": load_demo_questions(),
        "config": CONFIG.summary(),
    }


# ------------------------------------------------------------------
# ① 上传
# ------------------------------------------------------------------

@app.post("/api/pages")
async def upload_page(
    file: Optional[UploadFile] = File(default=None),
    demo: str = Form(default="0"),
    page_id: Optional[str] = Form(default=None),
):
    """上传古籍页图像。

    - 传 demo=1（或不传文件）→ 加载 data/demo/ 示例数据，保证零样本可演示；
    - 传文件 → 保存到 data/uploads/ 后走 OCR 流水线；
      本阶段模型未接入，仍返回结构完整的示例识别结果并显式标注。
    """
    use_demo = demo.strip().lower() in {"1", "true", "yes", "on"} or file is None
    try:
        if use_demo:
            ocr = run_demo(page_id or "DEMO-001")
        else:
            pid = page_id or new_page_id()
            suffix = Path(file.filename or "page.png").suffix.lower() or ".png"
            safe_name = f"{pid}{suffix}"
            dest = CONFIG.paths.uploads_dir / safe_name
            with dest.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            ocr = run_ocr(str(dest), pid)
            ocr.notes.append("已接收上传图像；OCR 模型未接入，返回示例识别结果")
        return JSONResponse(_ingest(ocr))
    except Exception as exc:
        # 服务不因演示输入崩溃：降级到示例数据并如实标注
        ocr = run_demo("DEMO-001")
        ocr.notes.append(f"上传处理降级：{type(exc).__name__}")
        return JSONResponse(_ingest(ocr))


@app.get("/api/pages")
def list_pages():
    return {"pages": store.list_pages(), "is_demo": True,
            "disclaimer": DISCLAIMER, "badge": CONFIG.demo_badge}


# ------------------------------------------------------------------
# ② OCR 与结构化
# ------------------------------------------------------------------

@app.get("/api/pages/{page_id}/ocr")
def get_ocr(page_id: str):
    snap = store.get_page(page_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="未找到该页面，请先在步骤一上传或加载示例数据")
    page = snap.get("page", {})
    return {
        "page_id": page_id,
        "page": page,
        "entities": snap.get("entities", []),
        "triples": snap.get("triples", []),
        "blocks": page.get("blocks", []),
        "is_demo": snap.get("is_demo", True),
        "engine": page.get("engine", "stub"),
        "model_loaded": page.get("model_loaded", False),
        "notes": page.get("notes", []),
        **demo_flags(),
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------
# ③ 图谱
# ------------------------------------------------------------------

@app.get("/api/graph")
def get_graph(page_id: Optional[str] = None):
    graph = _current_graph()
    if graph is None:
        return empty_payload("暂无图谱数据，请先在步骤一加载示例古籍页")
    payload = to_vis_payload(graph)
    payload["page_id"] = _STATE["page_id"]
    return payload


# ------------------------------------------------------------------
# ④ 可溯源问答
# ------------------------------------------------------------------

@app.post("/api/qa")
def ask(payload: Dict[str, Any] = Body(default={})):
    """可溯源问答。无证据 / 越界一律拒答，所有输出带免责声明。"""
    payload = payload or {}
    question = str(payload.get("question") or payload.get("q") or "").strip()
    page_id = payload.get("page_id") or _STATE["page_id"]
    if not question:
        ans = Answer(
            text="问题为空，原文未检索到依据。" + DISCLAIMER,
            refused=True, refuse_reason="原文未检索到依据",
            disclaimer=DISCLAIMER, trace=["空问题直接拒答"],
            backend="stub", is_demo=True,
        )
    else:
        blocks = _blocks_for(page_id)
        ans = qa_answer(question, blocks, vector_index=_STATE["vector_index"])
    try:
        store.log_qa(question, ans)
    except Exception:
        pass
    return {**ans.to_dict(), **page_banner(), "page_id": page_id}


# ------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(description="赛题二原型 · 本地 Web 服务")
    ap.add_argument("--host", default=CONFIG.server.host)
    ap.add_argument("--port", type=int, default=CONFIG.server.port)
    ap.add_argument("--reload", action="store_true", default=CONFIG.server.reload)
    args = ap.parse_args()

    store.init_db()
    import uvicorn
    print(f"服务启动： http://{args.host}:{args.port}/")
    print(f"免责声明： {DISCLAIMER}")
    print(f"演示状态： {CONFIG.demo_badge}（指标一律待测）")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    _cli()
