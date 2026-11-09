# -*- coding: utf-8 -*-
"""
app/main.py —— FastAPI 入口与路由（四步演示台的服务端）

承接旧 `agent/api_server.py` 的 FastAPI 骨架、上传与静态资源组织方式，
剥离客服 / 聊天室 / 三角色登录 / 模板填表等全部旧业务路由。

最小接口：
    GET    /                        四步演示台页面
    GET    /api/health              配置与模块状态（不含任何密钥）
    GET    /api/meta                免责文案 + 示例问题 + 步骤说明 + 词表规模
    POST   /api/pages               上传古籍页（或 demo=1 走示例数据）
    GET    /api/pages               页面清单
    DELETE /api/pages/{id}          删除一页及其派生数据（前端二次确认）
    GET    /api/pages/{id}/ocr      OCR 与结构化结果（blocks 按 reading_order 排序）
    GET    /api/graph               图谱可视化 JSON（支持 ?page_id= 按页重建）
    POST   /api/qa                  可溯源问答
    GET    /uploads/{name}          上传页预览
    GET    /docs                    FastAPI 自带接口文档

运行：python app/main.py [--port 8100]  或  双击 启动操作台.bat

阶段九改造要点：
  - 上传失败**不再静默降级成示例页**：按 400 / 413 / 503 / 500 分类返回，界面明确报错；
  - is_demo 从页面透传到实体 / 三元组 / 图谱节点，示例页与真实页标签不再混淆；
  - 图谱支持按 page_id 重建，块按 reading_order 排序；
  - 文案按实际状态生成（示例页 / 真实 OCR 识别 + 实际模型名）。
"""

from __future__ import annotations

import argparse
import io
import logging
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
from app.schemas import Answer, Block, Entity, GraphData, OCRResult, Triple  # noqa: E402
from app import store                                                     # noqa: E402
from extract.rules import extract_all                                      # noqa: E402
from extract.schema import dictionary_stats, warn_if_degraded              # noqa: E402
from graph.build import build_graph, build_triples                         # noqa: E402
from graph.export import (                                                 # noqa: E402
    empty_payload, export_graph_json, load_graph_json, to_vis_payload,
)
from ocr.pipeline import new_page_id, run_demo, run_ocr                    # noqa: E402
from ocr.recognize import get_adapter, rec_model_name, resolve_rec_model   # noqa: E402
from qa.disclaimer import DISCLAIMER, page_banner                          # noqa: E402
from qa.generate import answer as qa_answer                                # noqa: E402
from qa.index import build_index                                           # noqa: E402

logger = logging.getLogger("tcm.app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

app = FastAPI(
    title="中医古籍 OCR 与知识图谱智能问答原型",
    description="四步链路：上传古籍页 → 显示 OCR 与结构化结果 → 图谱可视化 → 可溯源问答。"
                "仅供参考、非诊疗建议。OCR 为本地 ONNX 模型真实推理；示例页会明确标注为示例数据。",
    version="0.3.0-stage9",
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
    "ocr": None,
}


def real_badge(model: str = "") -> str:
    """真实识别页的标注：模型名 + 数据性质，按实际状态生成。"""
    name = model or rec_model_name()
    return f"本地 OCR 真实识别 · {name} · 非命题单位样本"


def _flags_for(ocr: Optional[OCRResult]) -> Dict[str, Any]:
    """按识别结果动态给演示态标记：示例页与真实页用不同标签，均按实际状态生成。"""
    if ocr is None or ocr.is_demo:
        return demo_flags()
    return {
        "is_demo": False,
        "badge": real_badge(),
        "note": (f"本地 ONNX OCR 真实识别（模型：{rec_model_name()}）；"
                 "页面为上传的古籍扫描页，非命题单位样本、非官方评测结果"),
        "model_loaded": bool(ocr.model_loaded),
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------
# 内部：一次完整入库（OCR → 抽取 → 图谱 → 存储）
# ------------------------------------------------------------------

def _ingest(ocr: OCRResult) -> Dict[str, Any]:
    """把 OCRResult 走完 ②③④ 的构建并幂等入库，更新演示状态。

    is_demo 从页面**透传**给抽取层：示例页 True、真实页 False，
    不再像阶段八那样在 rules.py 里写死 True。
    """
    entities, candidates = extract_all(ocr.blocks, is_demo=ocr.is_demo)
    triples = build_triples(entities, ocr.blocks, candidates)
    graph = build_graph(triples, entities)
    stat = store.save_page(ocr, entities, triples)
    export_graph_json(graph)

    _STATE.update({
        "page_id": ocr.page_id,
        "ocr": ocr,
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
        "rec_model": rec_model_name(),
        "model_loaded": ocr.model_loaded,
        "notes": ocr.notes,
        **_flags_for(ocr),
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


def _snapshot_objects(page_id: str):
    """按页取实体 / 三元组（JSON 快照优先，SQLite 兜底），用于按页重建图谱。"""
    snap = store.get_page(page_id) or {}
    ents_raw = snap.get("entities") or []
    tris_raw = snap.get("triples") or []
    entities = [e if isinstance(e, Entity) else Entity.from_dict(e) for e in ents_raw]
    triples = [t if isinstance(t, Triple) else Triple.from_dict(t) for t in tris_raw]
    if not entities:
        entities = store.get_entities(page_id)
    if not triples:
        triples = store.get_triples(page_id)
    return entities, triples


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
    _rec_path, _rec_name, _rec_src = resolve_rec_model()
    return {
        "status": "ok",
        "config": CONFIG.summary(),
        "db": db,
        "current_page_id": _STATE["page_id"],
        "last_ingest": _STATE["last_ingest"],
        "modules": {
            "ocr": {"backend": (_STATE["ocr"].engine if _STATE["ocr"] else CONFIG.ocr.backend),
                    "rec_model_name": _rec_name,
                    "rec_model_source": _rec_src,
                    "model_loaded": bool(_STATE["ocr"] and _STATE["ocr"].model_loaded),
                    "is_demo": bool(_STATE["ocr"] is None or _STATE["ocr"].is_demo)},
            "extract": {"backend": "rules", "llm": CONFIG.llm.backend,
                        "llm_loaded": False, "is_demo": False,
                        "dictionary": dictionary_stats()},
            "graph": {"nodes": len(_STATE["graph"].nodes) if _STATE["graph"] else 0,
                      "is_demo": bool(_STATE["graph"].is_demo) if _STATE["graph"] else True},
            "qa": {"backend": CONFIG.llm.backend, "vector": CONFIG.vector.backend,
                   "is_demo": False},
        },
        "metrics": "五项实测指标见 data/reports/正式评测报告.md（合成集，非官方评测结果）",
        **(_flags_for(_STATE["ocr"]) if _STATE["ocr"] else {
            "is_demo": True,
            "badge": "本地 OCR 后端已就绪 · 请上传页面或加载示例",
            "note": "尚无已加载页面；后端能力由 /api/health 的 modules 字段给出",
            "model_loaded": False, "disclaimer": DISCLAIMER}),
    }


@app.get("/api/meta")
def meta():
    """前端常驻文案、示例问题、四步说明与词表规模。"""
    return {
        **page_banner(),
        "steps": [
            {"id": 1, "name": "上传古籍页", "api": "POST /api/pages"},
            {"id": 2, "name": "识别与结构化", "api": "GET /api/pages/{id}/ocr"},
            {"id": 3, "name": "图谱可视化", "api": "GET /api/graph?page_id=..."},
            {"id": 4, "name": "可溯源问答", "api": "POST /api/qa"},
        ],
        "questions": load_demo_questions(),
        "config": CONFIG.summary(),
        "dictionary": dictionary_stats(),
        "rec_model_name": rec_model_name(),
    }


# ------------------------------------------------------------------
# ① 上传
# ------------------------------------------------------------------

def _check_image_bytes(raw: bytes) -> None:
    """校验上传内容确实是可解码的图片；不是则 400（不猜、不降级）。"""
    try:
        from PIL import Image
    except Exception:      # 没装 Pillow 时不做解码校验，交给 OCR 阶段报错
        return
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="图像无法解码：文件不是有效的图片（常见原因是把别的文件改了扩展名）。"
                   "请上传 png / jpg / bmp / webp / tif 格式的古籍扫描页。")


@app.post("/api/pages")
async def upload_page(
    file: Optional[UploadFile] = File(default=None),
    demo: str = Form(default="0"),
    page_id: Optional[str] = Form(default=None),
):
    """上传古籍页图像。

    - 传 demo=1（或不传文件）→ 加载 data/demo/ 示例数据，保证零样本可演示；
    - 传文件 → 保存到 data/uploads/ 后走真实 OCR 流水线。

    失败一律**不降级成示例页**（阶段九 P0-4）：按原因返回 400 / 413 / 503 / 500，
    界面据此显示可读错误，而不是假装成功。
    """
    use_demo = demo.strip().lower() in {"1", "true", "yes", "on"} or file is None
    if use_demo:
        # 示例路径允许 demo_fallback：它就是示例数据本身
        ocr = run_demo(page_id or "DEMO-001")
        return JSONResponse(_ingest(ocr))

    suffix = Path(file.filename or "page.png").suffix.lower() or ".png"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式 {suffix}：只接受 "
                   + " / ".join(sorted(s.lstrip('.') for s in ALLOWED_SUFFIXES)) + "。")

    try:
        raw = await file.read()
    except Exception as exc:                     # 读上传流失败：客户端问题
        logger.warning("读取上传流失败：%s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="上传数据读取失败，请重新上传。")

    limit = int(CONFIG.server.max_upload_mb) * 1024 * 1024
    if len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大：{len(raw) / 1048576:.1f} MB，上限 "
                   f"{CONFIG.server.max_upload_mb} MB（可用 TCM_MAX_UPLOAD_MB 调整）。")

    _check_image_bytes(raw)

    # OCR 依赖 / 模型可用性：不可用直接 503，不返回示例数据冒充识别结果
    adapter, notes = get_adapter()
    if adapter.name == "stub":
        raise HTTPException(
            status_code=503,
            detail="OCR 依赖不可用：未检测到 rapidocr-onnxruntime。"
                   "请先 pip install -r requirements.txt，再重启服务。"
                   f"（诊断：{'；'.join(notes)}）")

    pid = page_id or new_page_id()
    dest = CONFIG.paths.uploads_dir / f"{pid}{suffix}"
    try:
        with dest.open("wb") as fh:
            fh.write(raw)
        ocr = run_ocr(str(dest), pid)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("上传处理失败：%s", type(exc).__name__)
        raise HTTPException(status_code=500,
                            detail=f"服务端处理异常（{type(exc).__name__}），详见服务端日志。")

    if ocr.is_demo or not ocr.model_loaded:
        # 真实上传路径**不允许**降级成示例数据（阶段九 P0-4）
        logger.error("真实上传被降级为示例数据：page_id=%s notes=%s", pid, ocr.notes)
        raise HTTPException(
            status_code=503,
            detail="OCR 未能完成真实识别（模型加载失败或依赖缺失），本次不上传、不返回示例数据。"
                   f"诊断信息：{'；'.join(ocr.notes[-2:])}")

    ocr.notes.append(f"已接收上传图像：{file.filename}（{len(raw) / 1024:.0f} KB）；"
                     f"识别模型 {rec_model_name()}，耗时见日志")
    return JSONResponse(_ingest(ocr))


@app.get("/api/pages")
def list_pages():
    flags = _flags_for(_STATE["ocr"]) if _STATE["ocr"] else demo_flags()
    return {"pages": store.list_pages(), "current_page_id": _STATE["page_id"],
            "is_demo": flags["is_demo"], "disclaimer": DISCLAIMER, "badge": flags["badge"]}


@app.delete("/api/pages/{page_id}")
def delete_page(page_id: str):
    """删除一页及其 OCR / 实体 / 三元组 / 上传图片（前端必须二次确认）。"""
    stat = store.delete_page(page_id)
    if not stat.get("deleted"):
        raise HTTPException(status_code=404, detail="未找到该页面（可能已被删除）")
    if _STATE["page_id"] == page_id:
        _STATE.update({"page_id": None, "ocr": None, "blocks": [], "entities": [],
                       "triples": [], "graph": None, "vector_index": None})
    return {"ok": True, **stat, "pages": store.list_pages()}


# ------------------------------------------------------------------
# ② OCR 与结构化
# ------------------------------------------------------------------

@app.get("/api/pages/{page_id}/ocr")
def get_ocr(page_id: str):
    snap = store.get_page(page_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="未找到该页面，请先在步骤一上传或加载示例数据")
    page = snap.get("page", {})
    # blocks 必须按 reading_order 排序（阶段九 P1）：JSON 快照里的顺序是 OCR 原始返回顺序，
    # SQLite 已 ORDER BY reading_order, id，优先用它；取不到才退回快照并就地排序。
    blocks = [b.to_dict() for b in store.get_blocks(page_id)]
    if not blocks:
        blocks = sorted(page.get("blocks", []),
                        key=lambda b: (b.get("reading_order", 0), b.get("line_id", "")))
    ocr_obj = OCRResult.from_dict(page)
    return {
        "page_id": page_id,
        "page": page,
        "entities": snap.get("entities", []),
        "triples": snap.get("triples", []),
        "blocks": blocks,
        "is_demo": snap.get("is_demo", True),
        "engine": page.get("engine", "stub"),
        "rec_model": rec_model_name(),
        "model_loaded": page.get("model_loaded", False),
        "notes": page.get("notes", []),
        **_flags_for(ocr_obj),
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------
# ③ 图谱
# ------------------------------------------------------------------

@app.get("/api/graph")
def get_graph(page_id: Optional[str] = None):
    """图谱 JSON。传 page_id 时**按该页**重建，不传才用内存里最后一次 ingest 的图。"""
    if page_id and page_id != _STATE["page_id"]:
        entities, triples = _snapshot_objects(page_id)
        if not entities and not triples:
            return empty_payload(f"页面 {page_id} 暂无图谱数据（可能未抽到实体，或页面已被删除）")
        graph = build_graph(triples, entities)
        snap = store.get_page(page_id) or {}
        is_demo = bool(snap.get("is_demo", True))
        badge = CONFIG.demo_badge if is_demo else real_badge()
        payload = to_vis_payload(graph, badge=badge)
        payload.update({"page_id": page_id, "is_demo": is_demo})
        return payload

    graph = _current_graph()
    if graph is None:
        return empty_payload("暂无图谱数据，请先在步骤一上传古籍页或加载示例")
    flags = _flags_for(_STATE["ocr"]) if _STATE["ocr"] else demo_flags()
    payload = to_vis_payload(graph, badge=flags["badge"])
    payload["page_id"] = _STATE["page_id"]
    payload["is_demo"] = bool(flags["is_demo"])
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
    if page_id and page_id != _STATE["page_id"]:
        snap = store.get_page(page_id) or {}
        flags = {"is_demo": bool(snap.get("is_demo", True)),
                 "badge": CONFIG.demo_badge if snap.get("is_demo", True) else real_badge()}
    else:
        flags = _flags_for(_STATE["ocr"]) if _STATE["ocr"] else demo_flags()
    if not question:
        ans = Answer(
            text="问题为空，原文未检索到依据。" + DISCLAIMER,
            refused=True, refuse_reason="原文未检索到依据",
            disclaimer=DISCLAIMER, trace=["空问题直接拒答"],
            backend="stub", is_demo=flags["is_demo"],
        )
    else:
        blocks = _blocks_for(page_id)
        ans = qa_answer(question, blocks, vector_index=_STATE["vector_index"],
                        is_demo=flags["is_demo"])
    try:
        store.log_qa(question, ans)
    except Exception:
        pass
    return {**ans.to_dict(), **page_banner(), "page_id": page_id,
            "badge": flags["badge"], "is_demo": flags["is_demo"]}


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
    warn_if_degraded()
    import uvicorn
    print(f"服务启动： http://{args.host}:{args.port}/")
    print(f"免责声明： {DISCLAIMER}")
    print(f"阶段九：本地 OCR 真实推理（识别模型 {rec_model_name()}）；"
          "抽取为词典 + 规则，未接入 LLM")
    print("五项实测指标见 data/reports/正式评测报告.md（合成集，非官方评测结果）")
    print("数据声明：合成示意数据用于工程自测；命题单位样本未到位，不得外传")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    _cli()
