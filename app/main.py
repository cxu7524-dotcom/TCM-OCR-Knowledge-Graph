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
    GET    /api/metrics             评测概览（只读 data/reports/评测明细.json）
    POST   /api/qa                  可溯源问答
    GET    /uploads/{name}          上传页预览
    GET    /docs                    FastAPI 自带接口文档
    GET    /lounge                  聊天室页面（跨设备公开聊天，纯内存）
    GET    /api/chat/messages       聊天室：增量消息 + 在室成员
    POST   /api/chat/send           聊天室：发言

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
import json
import json
import logging
import mimetypes
import shutil
import sys
import threading
import time
from functools import wraps
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

# 允许 `python app/main.py` 直接启动（此时 sys.path[0] 是 app/）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi import Response                                                # noqa: E402
from fastapi.responses import FileResponse, JSONResponse                  # noqa: E402
from fastapi.staticfiles import StaticFiles                              # noqa: E402

from app.config import CONFIG                                            # noqa: E402
from app.demo import demo_flags, load_demo_ocr, load_demo_questions        # noqa: E402
from app.schemas import Answer, Block, Entity, GraphData, OCRResult, Triple  # noqa: E402
from app import store                                                     # noqa: E402
from extract.rules import (candidate_triples,                           # noqa: E402
                           extract_all_audited,
                           extract_entities as candidate_entities)
from extract.formula_segments import extract_formula_candidates
from extract.formula_rows import extract_formula_rows
from extract.formula_tables import text_formula_tables
from extract.llm_extraction import extract_candidates, PROMPT_VERSION, PROTOCOL_VERSION
from extract.llm_validate import validate_candidates
from extract.schema import dictionary_stats, warn_if_degraded              # noqa: E402
from graph.build import build_graph, build_triples                         # noqa: E402
from graph.export import (                                                 # noqa: E402
    empty_payload, export_graph_json, load_graph_json, to_vis_payload,
)
from graph.query import run_query                                          # noqa: E402
from ocr.pipeline import new_page_id, run_demo, run_ocr                    # noqa: E402
from ocr import table as table_mod                                         # noqa: E402
from ocr.recognize import get_adapter, rec_model_name, resolve_rec_model   # noqa: E402
from qa.disclaimer import DISCLAIMER, page_banner                          # noqa: E402
from qa.disclaimer import out_of_scope_note, wrap                          # noqa: E402
from qa.generate import answer as qa_answer                                # noqa: E402
from qa.graph_qa import detect_intent, graph_answer as graph_qa_answer     # noqa: E402
from qa.index import build_index                                           # noqa: E402
from qa.index import kick_rebuild, rebuild_now                             # noqa: E402
from qa.rag import rag_answer, rag_status                                  # noqa: E402
from qa.refuse import is_out_of_scope                                      # noqa: E402

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
    "unresolved": [],
    "graph": None,
    "vector_index": None,
    "last_ingest": None,
    "ocr": None,
    "extraction": None,
}
_INGEST_LOCK = threading.RLock()


def _serialized_ingest(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _INGEST_LOCK:
            return fn(*args, **kwargs)
    return wrapped


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

@_serialized_ingest
def _ingest(ocr: OCRResult) -> Dict[str, Any]:
    """把 OCRResult 走完 ②③④ 的构建并幂等入库，更新演示状态。

    is_demo 从页面**透传**给抽取层：示例页 True、真实页 False，
    不再像阶段八那样在 rules.py 里写死 True。
    """
    # 同一 OCRResult 重复入库时，从原始块重新生成本轮证据，防止证据重复进抽取流。
    ocr.blocks = [b for b in ocr.blocks if b.kind != 'small_column_evidence']
    selected = CONFIG.llm.extract_backend.lower()
    if selected not in {"auto", "rules", "cloud"}:
        raise HTTPException(status_code=503, detail="抽取配置无效，请检查 TCM_EXTRACT_BACKEND。")
    cloud = selected == "cloud" or (selected == "auto" and CONFIG.llm.backend.lower() == "cloud")
    if cloud:
        if CONFIG.llm.backend.lower() != "cloud" or not CONFIG.llm.enabled:
            raise HTTPException(status_code=503, detail="云端抽取配置不完整，请检查服务端配置。")
        try:
            refs = [b.block_ref for b in ocr.blocks]
            call = extract_candidates(ocr.blocks, refs)
            validated = validate_candidates(ocr.blocks, refs, call, is_demo=ocr.is_demo)
        except Exception:
            raise HTTPException(status_code=503, detail="云端抽取调用或结果处理失败，可重试。")
        if validated.status == "failure":
            safe_errors = {"authentication", "model_unavailable", "rate_limit", "timeout", "server_error", "invalid_response", "truncated", "connection_error", "configuration", "invalid_protocol", "invalid_output", "invalid_input", "http_error", "scope_detection_failed"}
            code = validated.error if validated.error in safe_errors else "unknown_error"
            raise HTTPException(status_code=503, detail=f"云端抽取失败（{code}），可检查配置后重试。")
        entities, triples, unresolved = validated.entities, validated.triples, validated.unresolved
        extraction = {"mode": "cloud", "model": CONFIG.llm.model, "prompt_version": PROMPT_VERSION,
                      "protocol_version": PROTOCOL_VERSION,
                      "status": "success_with_relations" if triples else "success_no_relations"}
    else:
        entities, candidates, unresolved = extract_all_audited(ocr.blocks, is_demo=ocr.is_demo)
        triples = build_triples(entities, ocr.blocks, candidates)
        extraction = {"mode": "rules", "model": "", "prompt_version": "", "protocol_version": "",
                      "status": "success_with_relations" if triples else "success_no_relations"}
    graph = build_graph(triples, entities)
    # 未确认归属随页入库：这些行「找不到可确认主体」，不产出关系，
    # 但记录要能事后核查（page_id / line_id / 关系 / 客体 / 原因 / bbox）。
    segmented = extract_formula_candidates(ocr.blocks)
    formula_extraction = extract_formula_rows(ocr.blocks, segmented['candidates'])
    formula_extraction['continuation_hints'] = segmented['continuation_hints']
    if CONFIG.ocr.small_columns_enabled:
        try:
            from ocr.small_columns_attach import attach_small_columns
            evidence, formula_extraction, small_audit = attach_small_columns(
                ocr, formula_extraction, server_fallback=CONFIG.ocr.small_columns_server_enabled)
            ocr.blocks.extend(evidence)
            formula_extraction['small_columns_audit'] = small_audit
        except Exception as exc:
            # 原 OCR 与基线方表不变；实验失败不能导致错写剂量。
            logger.exception('小字证据回退：%s', type(exc).__name__)
            formula_extraction['small_columns_audit'] = {'enabled': True, 'fallback': True,
                                                          'reason': type(exc).__name__}
    # 保存原书表结构以便 JSON 快照不可读时从 SQLite 恢复。
    formula_extraction['_printed_tables'] = [t.to_dict() for t in ocr.tables]
    stat = store.save_page(ocr, entities, triples, unresolved=unresolved,
                           formula_extraction=formula_extraction, extraction_meta=extraction)
    try:
        export_graph_json(graph)
    except OSError:
        logger.warning("图谱镜像写入失败，页面数据仍可从数据库读取")

    # 新页面入库 → 语义索引后台增量重建（不阻塞本次请求）
    try:
        kick_rebuild()
    except Exception:
        pass

    _STATE.update({
        "page_id": ocr.page_id,
        "ocr": ocr,
        "blocks": ocr.blocks,
        "entities": entities,
        "triples": triples,
        "unresolved": unresolved,
        "extraction": extraction,
        "graph": graph,
        "vector_index": build_index([b for b in ocr.blocks if b.kind != 'small_column_evidence']),
        "last_ingest": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {
        "page_id": ocr.page_id,
        "blocks": len(ocr.blocks),
        "entities": len(entities),
        "triples": len(triples),
        "unresolved": len(unresolved),
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "ingest": stat,
        "extraction": extraction,
        **_tables_payload(ocr.to_dict(), [b.to_dict() for b in ocr.blocks], formula_extraction),
        "formula_continuation_hints": formula_extraction['continuation_hints'],
        "formula_extraction_status": "stored",
        "engine": ocr.engine,
        "rec_model": rec_model_name(),
        "model_loaded": ocr.model_loaded,
        "notes": ocr.notes,
        **_flags_for(ocr),
        "disclaimer": DISCLAIMER,
    }


def _blocks_for(page_id: Optional[str]) -> List[Block]:
    """指定页只检索该页。无选页时使用内置示例。"""
    if not page_id:
        return load_demo_ocr("DEMO-001").blocks
    if page_id == _STATE["page_id"]:
        return _STATE["blocks"]
    return store.get_blocks(page_id)


def _current_graph() -> Optional[GraphData]:
    if _STATE["graph"] is not None:
        return _STATE["graph"]
    return load_graph_json()


def _objects_for(page_id: Optional[str], blocks: List[Block],
                 is_demo: bool = True):
    """问答用的实体 / 三元组：**只用当前页**。

    优先用入库时保存的图谱（JSON 快照 → SQLite）；该页快照里没有三元组时
    （早期入库的页），用与入库同一套确定性规则基于本页原文就地重建，
    不引入任何新规则，也不跨页拼凑。
    """
    ent: List[Entity] = []
    tri: List[Triple] = []
    if page_id:
        ent, tri = _snapshot_objects(page_id)
    elif _STATE["triples"]:
        ent, tri = _STATE["entities"], _STATE["triples"]
    snap = store.get_page(page_id) if page_id else None
    if tri or not blocks or (snap and "extraction" in snap):
        return ent, tri
    try:
        blocks = [b for b in blocks if b.kind != 'small_column_evidence']
        ent = ent or candidate_entities(blocks, is_demo=is_demo)
        return ent, build_triples(ent, blocks,
                                  candidate_triples(blocks, is_demo=is_demo))
    except Exception:
        return ent, tri


def _cross_page_hits(page_id: str, question: str) -> int:
    """统计「同一问题在其他页命中多少条关系」——只写进 trace，绝不作为证据。"""
    try:
        intent, term = detect_intent(question)
        if not intent:
            return 0
        total = 0
        for p in store.list_pages():
            pid = p.get("page_id") if isinstance(p, dict) else getattr(p, "page_id", "")
            if not pid or pid == page_id:
                continue
            _e, tris = _snapshot_objects(pid)
            if not tris:
                continue
            total += int(run_query(intent, term, tris, page_id=pid).get("count") or 0)
        return total
    except Exception:
        return 0


def _snapshot_objects(page_id: str):
    """按页取实体 / 三元组（JSON 快照优先，SQLite 兜底），用于按页重建图谱。"""
    snap = store.get_page(page_id) or {}
    ents_raw = snap.get("entities") or []
    tris_raw = snap.get("triples") or []
    entities = [e if isinstance(e, Entity) else Entity.from_dict(e) for e in ents_raw]
    triples = [t if isinstance(t, Triple) else Triple.from_dict(t) for t in tris_raw]
    if not entities and "extraction" not in snap:
        entities = store.get_entities(page_id)
    if not triples and "extraction" not in snap:
        triples = store.get_triples(page_id)
    return entities, triples


# ------------------------------------------------------------------
# 页面与静态资源
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(CONFIG.paths.web_dir / "index.html", media_type="text/html; charset=utf-8")


@app.get("/home", include_in_schema=False)
def home_page():
    """简洁主页：古籍页上传 + AI 大模型对话。"""
    return FileResponse(CONFIG.paths.web_dir / "home.html", media_type="text/html; charset=utf-8")


@app.get("/lounge", include_in_schema=False)
def lounge_page():
    """聊天室页面（新增）：游客之间跨设备互通，纯前端渲染 + 两条内存接口。"""
    return FileResponse(CONFIG.paths.web_dir / "lounge.html", media_type="text/html; charset=utf-8")


CONFIG.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(CONFIG.paths.uploads_dir)), name="uploads")
app.mount("/static", StaticFiles(directory=str(CONFIG.paths.web_dir)), name="static")


# ------------------------------------------------------------------
# 元信息
# ------------------------------------------------------------------

@app.get("/api/llm")
def llm_status():
    """本机大模型（Ollama）状态探活，供主页展示「AI 助手是否就绪」。

    只读探测，不触发推理；任一步失败都返回 enabled=False，前端据此提示。
    """
    import urllib.request

    cfg = {
        "backend": CONFIG.llm.backend,
        "model": CONFIG.llm.model,
        "enabled": CONFIG.llm.enabled,
    }
    if not CONFIG.llm.enabled:
        return {**cfg, "reachable": False, "models": [],
                "ready": False, "note": "本机 LLM 未启用（TCM_LLM_BACKEND 未设置为 ollama/vllm）"}

    base = (CONFIG.llm.base_url or "").rstrip("/")
    if base.endswith("/v1"):
        tags_url = base[:-3] + "/api/tags"
    else:
        tags_url = base + "/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("models", [])]
        target_ready = any((m == CONFIG.llm.model or m.startswith(CONFIG.llm.model + ":"))
                           for m in models)
        return {
            **cfg, "reachable": True, "models": models,
            "ready": target_ready,
            "note": "AI 助手已就绪" if target_ready else
                    f"可连接 Ollama，但模型 {CONFIG.llm.model} 尚未就绪（现有：{', '.join(models) or '无'}）",
        }
    except Exception as exc:
        return {**cfg, "reachable": False, "models": [],
                "ready": False,
                "note": f"无法连接本机 Ollama：{type(exc).__name__}"}


@app.get("/api/rag/status")
def rag_state():
    """知识库与语义索引状态：多少页 / 多少段 / 多少向量 / 模型 / 是否需重建。"""
    try:
        return rag_status()
    except Exception as exc:
        return {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/rag/rebuild")
def rag_rebuild(payload: Dict[str, Any] = Body(default={})):
    """强制重建语义索引（同步）。语料大时请留足时间，前端应显示进度提示。"""
    try:
        force = bool((payload or {}).get("force", True))
        return {"ok": True, "index": rebuild_now(force=force)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建失败：{type(exc).__name__}: {exc}")


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
            "extract": {"backend": CONFIG.llm.extract_backend, "llm": CONFIG.llm.backend,
                        "llm_loaded": False, "is_demo": False,
                        "current_page": _STATE.get("extraction"),
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


_METRIC_ORDER = [
    ("cer", "字符错误率 CER", "lower"),
    ("field_f1", "字段级 F1", "higher"),
    ("table_structure", "表格结构还原", "higher"),
    ("traceability", "可溯源率", "higher"),
    ("refusal", "无依据拒答率", "higher"),
    ("answer_rate", "应答题正确作答率", "higher"),
]


@app.get("/api/metrics")
def metrics():
    """读取 eval/run_all.py 产出的 data/reports/评测明细.json，供前端展示评测概览。

    只读、不重算；文件缺失时返回 available=False，前端据此提示「尚未评测」。
    """
    path = CONFIG.paths.reports_dir / "评测明细.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "items": [],
                "notice": "尚未生成评测明细，请先运行 python eval/run_all.py"}
    except (OSError, ValueError) as exc:
        return {"available": False, "items": [],
                "notice": f"评测明细读取失败（{type(exc).__name__}）"}
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        return {"available": False, "items": [], "notice": "评测明细格式不正确"}
    results = data["results"]
    items = []
    for key, label, better in _METRIC_ORDER:
        r = results.get(key)
        if not isinstance(r, dict):
            continue
        items.append({"key": key, "label": label, "better": better,
                      "value": r.get("value"), "threshold": r.get("threshold"),
                      "judge": r.get("judge"), "status": r.get("status"),
                      "detail": r.get("detail")})
    if not items:
        return {"available": False, "items": [], "notice": "评测明细没有可展示的指标"}
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    test_pages = results.get("test_pages")
    return {"available": True, "items": items,
            "generated_at": data.get("generated_at"),
            "gt_version": data.get("gt_version"),
            "test_only": results.get("test_only"),
            "test_pages": len(test_pages) if isinstance(test_pages, list) else None,
            "notice": data.get("notice"),
            "counts": dataset.get("counts"),
            "test_pages": dataset.get("test_pages"),
            "pages_total": dataset.get("pages_total"),
            "gt_version": data.get("gt_version"),
            "reproduce": data.get("reproduce")}


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
    dest = None
    try:
        while True:
            dest = CONFIG.paths.uploads_dir / f"{new_page_id()}{suffix}"
            try:
                with dest.open("xb") as fh:
                    fh.write(raw)
                break
            except FileExistsError:
                continue
        ocr = run_ocr(str(dest), pid)
    except HTTPException:
        if dest is not None:
            dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if dest is not None:
            dest.unlink(missing_ok=True)
        logger.exception("上传处理失败：%s", type(exc).__name__)
        raise HTTPException(status_code=500,
                            detail=f"服务端处理异常（{type(exc).__name__}），详见服务端日志。")

    if ocr.is_demo or not ocr.model_loaded:
        dest.unlink(missing_ok=True)
        # 真实上传路径**不允许**降级成示例数据（阶段九 P0-4）
        logger.error("真实上传被降级为示例数据：page_id=%s notes=%s", pid, ocr.notes)
        raise HTTPException(
            status_code=503,
            detail="OCR 未能完成真实识别（模型加载失败或依赖缺失），本次不上传、不返回示例数据。"
                   f"诊断信息：{'；'.join(ocr.notes[-2:])}")

    ocr.notes.append(f"已接收上传图像：{file.filename}（{len(raw) / 1024:.0f} KB）；"
                     f"识别模型 {rec_model_name()}，耗时见日志")
    try:
        return JSONResponse(_ingest(ocr))
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


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
                       "triples": [], "unresolved": [], "graph": None,
                       "vector_index": None, "extraction": None})
    return {"ok": True, **stat, "pages": store.list_pages()}


# ------------------------------------------------------------------
# ② OCR 与结构化
# ------------------------------------------------------------------

_TABLE_NO_FIELD = ("历史记录缺少表格字段：该页入库时尚未保存表格结构"
                   "（早于方剂表还原功能的旧记录），无法还原表格。请重新识别该页。")
_TABLE_NONE = "本页未识别到方剂表（bbox 行列聚类要求至少 2 行且多数行 ≥2 格，竖排正文页不会误判）。"


def _printed_tables_payload(page: Dict[str, Any],
                            blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把页面快照里的表格整理成前端可直接渲染、且**逐格可定位**的结构。

    复用 OCR 已产出的 TableStruct，不改动识别结果：
      - rows 与 cell_refs 同形，行列位置与空白单元格原样保留；
      - cell_refs 缺失（旧记录）时用 align_cell_refs 逐格核验补齐，
        核不上就留空，前端标「无法定位」，不拿文本相似度硬凑；
      - 快照里**没有 tables 字段**时返回 tables_status="no_table_field"，
        与「本页确实无表」（tables_status="none"）区分开，不用空表冒充。
    """
    if "tables" not in page:
        return {"tables": [], "tables_status": "no_table_field",
                "tables_note": _TABLE_NO_FIELD}
    raw_tables = page.get("tables") or []
    if not isinstance(raw_tables, list):
        return {"tables": [], "tables_status": "no_table_field",
                "tables_note": _TABLE_NO_FIELD}
    if not raw_tables:
        return {"tables": [], "tables_status": "none", "tables_note": _TABLE_NONE}

    block_objs = [Block.from_dict(b) if isinstance(b, dict) else b for b in blocks]
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_tables):
        t = raw if isinstance(raw, dict) else (
            raw.to_dict() if hasattr(raw, "to_dict") else {})
        rows = [list(r) if isinstance(r, (list, tuple)) else [r]
                for r in (t.get("rows") or [])]
        refs = t.get("cell_refs") or []
        refs_source = "ocr"
        if not refs:
            refs = table_mod.align_cell_refs(rows, t.get("cells") or [], block_objs)
            refs_source = "derived" if refs else "missing"
        # 行列数按实际 rows 归一，缺列的行补空，保证前端网格不错位
        width = max((len(r) for r in rows), default=0)
        refs_norm = []
        for r_i, row in enumerate(rows):
            rows[r_i] = row + [""] * (width - len(row))
            line = list(refs[r_i]) if r_i < len(refs) and isinstance(refs[r_i], list) else []
            refs_norm.append((line + [""] * width)[:width])
        total = sum(1 for row in rows for text in row if str(text or "").strip())
        located = sum(1 for line in refs_norm for ref in line if str(ref or "").strip())
        out.append({
            "source_type": "原书表格",
            "table_id": t.get("table_id") or f"{page.get('page_id', '')}-T{i + 1:02d}",
            "page_id": t.get("page_id") or page.get("page_id", ""),
            "rows": rows,
            "cell_refs": refs_norm,
            "cells": t.get("cells") or [],
            "source_hash": t.get("source_hash", ""),
            "is_demo": bool(t.get("is_demo", True)),
            "refs_source": refs_source,
            "stats": {"rows": len(rows), "cols": width,
                      "filled_cells": total, "located_cells": located},
        })
    note = ""
    if any(t["refs_source"] == "derived" for t in out):
        note = "出处由 rows / cells 逐格核对补齐（旧记录无 cell_refs 字段），核对不上即标「无法定位」。"
    elif all(t["refs_source"] == "missing" for t in out):
        note = "该页表格缺少可核验的出处字段，单元格内容照原样展示，但无法定位到原图。"
    return {"tables": out, "tables_status": "ok", "tables_note": note}


def _tables_payload(page: Dict[str, Any], blocks: List[Dict[str, Any]],
                    formula_extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并原书表格及入库的正文表；旧快照无正文结果时沿用原有状态。"""
    result = _printed_tables_payload(page, blocks)
    if formula_extraction is None:
        return result
    objects = [Block.from_dict(b) if isinstance(b, dict) else b for b in blocks]
    derived, suppressed = text_formula_tables(formula_extraction, result['tables'], objects,
                                               bool(page.get('is_demo', True)))
    result['tables'] += derived
    result['duplicate_formula_tables_suppressed'] = suppressed
    if derived:
        if result['tables_status'] == 'none':
            result['tables_note'] = '本页未识别到原书规整表；以下为正文抽取。'
        elif result['tables_status'] == 'no_table_field':
            result['tables_note'] += ' 另显示已保存的正文抽取结果。'
        result['tables_status'] = 'ok'
    return result


def _unresolved_payload(page_id: str, snap: Dict[str, Any],
                        blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把「未确认归属」审计记录整理成前端 / 核查脚本可直接用的结构。

    优先读页面快照里入库时存下的记录（普通上传路径就会有）；
    旧记录没有该字段时，用**与入库同一套确定性规则**在本页原文上重算，
    只补审计、不改三元组、不新增关系——算出来也标 derived，不冒充入库结果。
    """
    raw = snap.get("unresolved") if isinstance(snap, dict) else None
    source = "stored"
    if raw is None:
        source = "derived"
        try:
            items = [Block.from_dict(b) if isinstance(b, dict) else b for b in blocks]
            _ents, _tris, raw = extract_all_audited(items, is_demo=bool(snap.get("is_demo", True)))
        except Exception:
            raw = []
    records = [store._norm_audit(u, page_id) for u in (raw or [])]
    note = ""
    if source == "derived":
        note = ("该页入库时尚未保存未确认记录（早于本功能的旧记录），"
                "此处按同一套确定性规则基于本页原文重算，仅用于审计，未新增任何关系。")
    return {"unresolved": records, "unresolved_count": len(records),
            "unresolved_source": source, "unresolved_note": note}


@app.get("/api/pages/{page_id}/ocr")
def get_ocr(page_id: str):
    snap = store.get_page(page_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="未找到该页面，请先在步骤一上传或加载示例数据")
    page = snap.get("page", {})
    # blocks 必须按 reading_order 排序（阶段九 P1）：JSON 快照里的顺序是 OCR 原始返回顺序，
    # SQLite 已 ORDER BY reading_order, id，优先用它；取不到才退回快照并就地排序。
    blocks = ([] if "extraction" in snap else [b.to_dict() for b in store.get_blocks(page_id)])
    if not blocks:
        blocks = sorted(page.get("blocks", []),
                        key=lambda b: (b.get("reading_order", 0), b.get("line_id", "")))
    ocr_obj = OCRResult.from_dict(page)
    return {
        "page_id": page_id,
        "extraction": snap.get("extraction"),
        "page": page,
        "entities": snap.get("entities", []),
        "triples": snap.get("triples", []),
        "blocks": blocks,
        # 方剂表：rows + 同形 cell_refs，供步骤二「方剂表还原」渲染与定位
        **_tables_payload(page, blocks, snap.get("formula_extraction")),
        "formula_continuation_hints": (snap.get("formula_extraction") or {}).get("continuation_hints", []),
        "formula_extraction_status": "stored" if "formula_extraction" in snap else "legacy_missing",
        # 未确认归属：普通上传即入库，这里原样返回，供步骤二核查
        **_unresolved_payload(page_id, snap, blocks),
        "is_demo": snap.get("is_demo", True),
        "engine": page.get("engine", "stub"),
        "rec_model": rec_model_name(),
        "model_loaded": page.get("model_loaded", False),
        "notes": page.get("notes", []),
        **_flags_for(ocr_obj),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/pages/{page_id}/unresolved")
def get_page_unresolved(page_id: str):
    """单页「未确认归属」审计记录：这些行没有产出关系，原因与坐标在此可查。"""
    snap = store.get_page(page_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="未找到该页面，请先在步骤一上传或加载示例数据")
    blocks = [b.to_dict() for b in store.get_blocks(page_id)]
    if not blocks:
        blocks = sorted(snap.get("page", {}).get("blocks", []),
                        key=lambda b: (b.get("reading_order", 0), b.get("line_id", "")))
    return {"page_id": page_id, **_unresolved_payload(page_id, snap, blocks),
            "disclaimer": DISCLAIMER}


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
    """可溯源问答（阶段十一：语义 RAG 为主链路）。

    链路优先级：
      0. 越界题（个人诊疗 / 剂量 / 疗效）→ 直接拒答，不检索不生成；
      1. **RAG 主链路**：语义向量检索知识库 → 阈值过滤 → 专业提示词 → 本机大模型生成
         → 解析 [编号] 回映射到 书/页/行/bbox，输出可点击的出处；
         检索不到时按配置给「通用中医常识」回答并显式声明无原文依据；
      2. 图谱关系题的结构化结果（qa.graph_qa）作为 objects 附带透出，不覆盖 RAG 答案；
      3. RAG 不可用（未启用 / 索引缺失）时，回退到原有字面检索链路，口径不变。
    """
    payload = payload or {}
    question = str(payload.get("question") or payload.get("q") or "").strip()
    mode = str(payload.get("mode") or "standard").strip().lower()
    if mode not in ("fast", "standard", "expert"):
        mode = "standard"
    t_start = time.time()
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
    elif is_out_of_scope(question):
        # 输出边界优先于一切检索：个人诊疗 / 剂量 / 疗效一律拒答
        note = out_of_scope_note()
        ans = Answer(
            text=wrap(note, "refusal"), evidences=[], refused=True, refuse_reason=note,
            disclaimer=DISCLAIMER,
            trace=["命中越界规则（个人诊疗 / 疗效 / 用药剂量）→ 拒答，未进入检索与生成"],
            backend="boundary", is_demo=flags["is_demo"],
        )
    else:
        blocks = _blocks_for(page_id)
        entities, triples = _objects_for(page_id, blocks, is_demo=flags["is_demo"])
        graph_ans = graph_qa_answer(
            question, blocks, triples, entities=entities, page_id=page_id or "",
            is_demo=flags["is_demo"],
            page_exists=bool(page_id) and store.get_page(page_id) is not None,
            other_page_hits=_cross_page_hits(page_id, question) if page_id else 0,
        )
        ans = graph_ans

        # 只有在"没有指定当前页"或"当前页确实有文本"时才走全库 RAG；
        # 请求一个不存在 / 空白的页时不做全库作答，避免答非所问。
        if CONFIG.rag.enabled and (not page_id or bool(blocks)):
            # 演示台选定了页 → 页内检索（出处只来自该页）；首页聊天 → 全库检索
            scope = str(payload.get("scope") or ("page" if page_id else "library")).strip().lower()
            history = payload.get("history")
            if not isinstance(history, list):
                history = None
            rag_ans = rag_answer(question, history=history, page_id=page_id,
                                 scope=scope, is_demo=flags["is_demo"], mode=mode)
            has_ev = bool(rag_ans.sources or rag_ans.evidences)
            # 页内模式：检索到该页原文才采纳（保住"页内无依据 → 正确拒答"的评测语义）；
            # 全库模式：无依据也可用常识作答（rag-open，显式声明）
            if not rag_ans.refused and (has_ev or scope == "library"):
                # RAG 答案更完整：优先采用；图谱关系只作为结构化对象附带
                if graph_ans is not None and graph_ans.objects:
                    rag_ans.objects = list(graph_ans.objects)
                    rag_ans.trace = list(rag_ans.trace) + ["已附带图谱关系结构化结果"]
                ans = rag_ans
            # RAG 拒答时不覆盖已有答案，留给下面的字面检索兜底

        if ans is None:
            index = _STATE["vector_index"] if page_id and page_id == _STATE["page_id"] else None
            ans = qa_answer(question, blocks, vector_index=index, is_demo=flags["is_demo"])
    try:
        store.log_qa(question, ans)
    except Exception:
        pass
    return {**ans.to_dict(), **page_banner(), "page_id": page_id,
            "badge": flags["badge"], "is_demo": flags["is_demo"],
            "qa_mode": mode, "elapsed": round(time.time() - t_start, 2)}


# ------------------------------------------------------------------
# 语音输入（本机离线识别）—— 新增路由，只调用 app/asr_voice.py
# 浏览器原生识别连不上外网时，前端自动降级到这两条接口；识别结果只填进
# 输入框，由用户自己改错别字后再发送，不直接替用户提问。
# ------------------------------------------------------------------

@app.get("/api/asr/status")
def api_asr_status():
    """语音通道自检：模型是否就位、依赖是否装好、可选语言档位。"""
    from app import asr_voice
    info = asr_voice.info()
    # 页面一打开就后台预热模型，用户真按麦克风时不用再等那几秒
    if info["ready"] and not info["loaded"]:
        threading.Thread(target=asr_voice.warm_up, daemon=True).start()
    return info


@app.post("/api/asr")
async def api_asr(file: UploadFile = File(...), lang: str = Form("zh")):
    """录音（webm / m4a / wav …）→ 文字。

    lang：zh 普通话 / yue 粤语 / en English（与 app/asr_voice.LANGS 同一张表）
    """
    from app import asr_voice
    info = asr_voice.info()
    if not info["ready"]:
        raise HTTPException(status_code=503,
                            detail="本机语音识别未就绪：" + (info["error"] or "模型或依赖缺失"))
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="没有收到音频数据")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="录音过大（上限 25MB）")
    try:
        r = asr_voice.transcribe_bytes(data, file.filename or "voice.webm", lang)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="识别失败：%s" % e)
    if not r.get("text"):
        raise HTTPException(status_code=422, detail="没听清，再说一次")
    return r


# ------------------------------------------------------------------
# 聊天室（新增路由，只调用 app/chat_room.py；纯内存，不进数据库）
# 目的：让「游客 + 五位编码」这套前端身份能真的跨设备说话。
# 边界：公开一条消息流，重启即清空；删掉本段与 app/chat_room.py 即回退。
# ------------------------------------------------------------------

@app.get("/api/chat/messages")
def api_chat_messages(room: str = "", since: int = 0, id: str = ""):
    """取某间聊天室里 since 之后的新消息与在室列表（前端每 2.5 秒拉一次）。

    room：聊天室 id（官方如 o-fangji，游客自建如 r7k2m）。
    id：当前游客编码。带上就等于登记在场，所以「在室」是真实心跳，
    不是「发过言的人」。
    """
    from app import chat_room
    try:
        return chat_room.snapshot(room, since, id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/chat/send")
def api_chat_send(payload: Dict[str, Any] = Body(default={})):
    """发一条消息。

    payload: {"room": "o-fangji", "id": "88129", "text": "..."}
    昵称由服务端按 id 生成（游客88129），前端不传名字，避免伪造。
    """
    from app import chat_room
    payload = payload or {}
    try:
        msg = chat_room.post(str(payload.get("room") or ""),
                             str(payload.get("id") or ""),
                             str(payload.get("text") or ""),
                             reply=payload.get("reply"))
    except chat_room.SlowDown as e:
        raise HTTPException(status_code=429,
                            detail="说得太快了，歇 %.1f 秒再发" % e.wait)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "seq": msg["seq"], "message": msg}


@app.post("/api/chat/ask")
def api_chat_ask(payload: Dict[str, Any] = Body(default={})):
    """在聊天室里向古籍知识库提问：跑一遍问答链路，把结果作为一条消息落进房间。

    payload: {"room": "o-fangji", "id": "88129", "question": "..."}
    - 走的是 /api/qa 同一条链路（全库检索，不走页内），所以出处仍是 书/页/行；
    - 提问按人限速（12 秒一条、五分钟三条），因为一次 RAG 要跑本机模型；
    - 问答结果是对整间房可见的一条消息（kind=qa），不是私聊。
    """
    from app import chat_room
    payload = payload or {}
    room = str(payload.get("room") or "")
    gid = str(payload.get("id") or "")
    q = str(payload.get("question") or "").strip()
    mode = str(payload.get("mode") or "standard").strip().lower()
    if mode not in ("fast", "standard", "expert"):
        mode = "standard"
    if not q:
        raise HTTPException(status_code=400, detail="先写一句要问的")
    if len(q) > chat_room.MAX_ASK:
        raise HTTPException(status_code=400,
                            detail="问题太长（%d 字以内）" % chat_room.MAX_ASK)
    try:
        chat_room.room_meta(room)
        chat_room.reserve_ask(gid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        ans = ask({"question": q, "scope": "library", "mode": mode})
    except Exception as e:                       # 模型没起来 / 索引坏了，别把房间弄挂
        raise HTTPException(status_code=503, detail="问答暂时不可用：%s" % e)
    cites = []
    for s in (ans.get("sources") or []):
        if len(cites) >= 4:
            break
        cites.append({"book": s.get("book") or "",
                      "page_id": s.get("page_id") or "",
                      "line": s.get("line_span") or s.get("line_id") or "",
                      "quote": s.get("quote") or ""})
    text = str(ans.get("text") or "").strip() or "这次没有取到答案。"
    try:
        msg = chat_room.post_bot(room, q, text, cites=cites,
                                 refused=bool(ans.get("refused")), by=gid,
                                 backend=str(ans.get("backend") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "message": msg, "refused": bool(ans.get("refused")),
            "backend": ans.get("backend") or "", "cites": len(cites),
            "mode": mode,
            "self_check": (ans.get("retrieval") or {}).get("self_check") or {},
            "elapsed": ans.get("elapsed")}


@app.post("/api/chat/invite")
def api_chat_invite(payload: Dict[str, Any] = Body(default={})):
    """按编号邀请某位游客。

    payload: {"from": "88129", "to": "47217", "room": "o-fangji"}
    只进内存队列，对方拉消息时取走；对方不在线也不会丢（十分钟内有效）。
    """
    from app import chat_room
    payload = payload or {}
    try:
        inv = chat_room.invite(str(payload.get("from") or ""),
                               str(payload.get("to") or ""),
                               str(payload.get("room") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "invite": inv, "to_name": inv["to_name"]}


@app.post("/api/chat/invite/respond")
def api_chat_invite_respond(payload: Dict[str, Any] = Body(default={})):
    """被邀请人处理邀请。payload: {"id": "i3", "me": "93074", "action": "accept"}

    action 取 accept / decline / ignore，结果转给邀请人。
    """
    from app import chat_room
    payload = payload or {}
    try:
        res = chat_room.respond(str(payload.get("id") or ""),
                                str(payload.get("me") or ""),
                                str(payload.get("action") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "result": res}


@app.get("/api/rooms")
def api_rooms(id: str = ""):
    """聊天室列表：官方（固定 4 间）+ 游客自建（按创建时间倒序）。

    带上 id 就一并算出每间的未读数与我的角色（左侧列表的红点用）。
    """
    from app import chat_room
    return chat_room.list_rooms(id)


@app.post("/api/rooms")
def api_rooms_create(payload: Dict[str, Any] = Body(default={})):
    """游客开一间。payload: {"name": "...", "desc": "...", "owner": "88129"}"""
    from app import chat_room
    payload = payload or {}
    try:
        room = chat_room.create_room(str(payload.get("name") or ""),
                                     str(payload.get("desc") or ""),
                                     str(payload.get("owner") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "room": room}


@app.delete("/api/rooms")
def api_rooms_delete(id: str = "", owner: str = ""):
    """创建者删掉自己的聊天室。"""
    from app import chat_room
    try:
        return chat_room.delete_room(id, owner)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/room")
def api_room(id: str = ""):
    """单间信息（房间页标题、类型、简介）。"""
    from app import chat_room
    try:
        return chat_room.room_meta(id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/chat/inbox")
def api_chat_inbox(id: str = ""):
    """收件箱：只取发给我的邀请与处理结果（列表页也会轮询这个）。"""
    from app import chat_room
    try:
        return chat_room.inbox(id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/chat/invites")
def api_chat_invites(me: str = ""):
    """我发出的邀请及当前状态（对话框里的记录）。"""
    from app import chat_room
    try:
        rows = chat_room.sent(me)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"invites": rows}


@app.get("/api/chat/notice")
def api_chat_notice(room: str = "", id: str = ""):
    """看房间公告。id 只用来判断「我能不能改这一条」。"""
    from app import chat_room
    try:
        n = chat_room.notice(room)
        n["can_manage"] = chat_room.can_manage(room, id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return n


@app.post("/api/chat/notice")
def api_chat_set_notice(payload: Dict[str, Any] = Body(default={})):
    """设置 / 撤下公告（官方房任何人都能改，游客房只有房主）。"""
    from app import chat_room
    payload = payload or {}
    try:
        n = chat_room.set_notice(str(payload.get("room") or ""),
                                 str(payload.get("id") or ""),
                                 str(payload.get("text") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, **n}


@app.get("/api/chat/files")
def api_chat_files(room: str = ""):
    """房间文件清单。"""
    from app import chat_room
    try:
        return chat_room.list_files(room)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/chat/files")
async def api_chat_upload(room: str = Form(""), id: str = Form(""),
                          file: UploadFile = File(...)):
    """上传一个文件到房间（≤5 MB，单间 12 个）。"""
    from app import chat_room
    try:
        data = await file.read()
        row = chat_room.save_file(room, id, file.filename or "", data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="存文件失败：%s" % e)
    return {"ok": True, "file": row}


@app.get("/api/chat/files/{fid}")
def api_chat_download(fid: str, room: str = "", inline: int = 0):
    """房间文件：默认按原始文件名另存；inline=1 就在浏览器里直接看。

    图片 / PDF / 纯文本走 inline，其余类型仍按下载给（避免浏览器乱解释可执行文件）。
    """
    from app import chat_room
    try:
        info = chat_room.file_path(room, fid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    path = str(info["path"])
    name = str(info["name"])
    if not inline:
        return FileResponse(path, filename=name, media_type="application/octet-stream")
    mt = mimetypes.guess_type(name)[0] or "application/octet-stream"
    viewable = (mt.startswith("image/") or mt == "application/pdf" or
                mt.startswith("text/") or mt in ("application/json", "application/xml"))
    if not viewable:
        return FileResponse(path, filename=name, media_type="application/octet-stream")
    if mt.startswith("text/") or mt in ("application/json", "application/xml"):
        mt = mt + "; charset=utf-8"
    return FileResponse(path, media_type=mt, headers={
        "Content-Disposition": "inline; filename*=utf-8''%s" % quote(name)})


@app.post("/api/chat/role")
def api_chat_role(payload: Dict[str, Any] = Body(default={})):
    """房主任命 / 撤销管理员。payload: {"room", "id", "target", "role": "admin|member"}"""
    from app import chat_room
    payload = payload or {}
    try:
        return chat_room.set_role(str(payload.get("room") or ""),
                                  str(payload.get("id") or ""),
                                  str(payload.get("target") or ""),
                                  str(payload.get("role") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat/mute")
def api_chat_mute(payload: Dict[str, Any] = Body(default={})):
    """禁言 N 分钟（0 = 解除）。payload: {"room", "id", "target", "minutes": 10}"""
    from app import chat_room
    payload = payload or {}
    try:
        return chat_room.mute(str(payload.get("room") or ""),
                              str(payload.get("id") or ""),
                              str(payload.get("target") or ""),
                              payload.get("minutes", 10))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat/kick")
def api_chat_kick(payload: Dict[str, Any] = Body(default={})):
    """把一个人移出这一间。payload: {"room", "id", "target"}"""
    from app import chat_room
    payload = payload or {}
    try:
        return chat_room.kick(str(payload.get("room") or ""),
                              str(payload.get("id") or ""),
                              str(payload.get("target") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/chat/roles")
def api_chat_roles(room: str = ""):
    """这一间的角色 / 禁言 / 移出名单（界面画标签与按钮用）。"""
    from app import chat_room
    try:
        return chat_room.roles_of(room)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/chat/export")
def api_chat_export(room: str = "", id: str = ""):
    """把整间房导成一份 Markdown（附件下载）。"""
    from app import chat_room
    try:
        info = chat_room.export_markdown(room, id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    body = info["text"].encode("utf-8")
    return Response(content=body,
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition":
                             "attachment; filename*=utf-8''%s" % quote(info["filename"]),
                             "Content-Length": str(len(body))})


@app.get("/api/chat/search")
def api_chat_search(room: str = "", q: str = "", limit: int = 40):
    """在房间消息里搜字（含问答的问句）。"""
    from app import chat_room
    try:
        return chat_room.search(room, q, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/chat/invites/sent")
def api_chat_invites_sent(id: str = "", room: str = ""):
    """我发出的邀请（带「已进房 / 在线」），聊天室里用来审核。"""
    from app import chat_room
    try:
        rows = chat_room.sent(id, room)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"invites": rows}


@app.post("/api/chat/invites/revoke")
def api_chat_invites_revoke(payload: Dict[str, Any] = Body(default={})):
    """撤回一条还没被回应的邀请。"""
    from app import chat_room
    payload = payload or {}
    try:
        out = chat_room.revoke(str(payload.get("invite") or ""),
                               str(payload.get("id") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return out


@app.get("/api/chat/status")
def api_chat_status():
    """聊天室自检：条数、在室人数、限速与长度上限。"""
    from app import chat_room
    return chat_room.stats()


# ------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------

def _warm_index() -> None:
    """开机把检索索引热起来。

    索引是懒加载的：进程刚起时第一问会边加载边触发重建，那一次常常取不到片段，
    答出来就是「依据不足」。这里在端口开始监听前先把它建好，第一问就正常。
    """
    try:
        t0 = time.time()
        from qa.index import get_index, rebuild_now
        idx = get_index()
        if (idx.status() or {}).get("stale"):
            rebuild_now(force=False)
            idx = get_index()
        st = idx.status() or {}
        print("  [预热] 检索索引：%s 段 · stale=%s · 用时 %.1f 秒"
              % (st.get("vectors"), st.get("stale"), time.time() - t0))
    except Exception as e:                       # 预热失败不该拦住服务启动
        print("  [预热] 跳过：%s" % e)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="赛题二原型 · 本地 Web 服务")
    ap.add_argument("--host", default=CONFIG.server.host)
    ap.add_argument("--port", type=int, default=CONFIG.server.port)
    ap.add_argument("--reload", action="store_true", default=CONFIG.server.reload)
    args = ap.parse_args()

    store.init_db()
    warn_if_degraded()
    _warm_index()
    import uvicorn
    print(f"服务启动： http://{args.host}:{args.port}/")
    print(f"免责声明： {DISCLAIMER}")
    print(f"本地 OCR 识别模型：{rec_model_name()}；抽取模式：{CONFIG.llm.extract_backend}")
    print("五项实测指标见 data/reports/正式评测报告.md（合成集，非官方评测结果）")
    print("数据声明：合成示意数据用于工程自测；命题单位样本未到位，不得外传")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    _cli()
