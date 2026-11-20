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
from extract.rules import (candidate_triples,                           # noqa: E402
                           extract_all_audited,
                           extract_entities as candidate_entities)
from extract.formula_segments import extract_formula_candidates
from extract.formula_rows import extract_formula_rows
from extract.formula_tables import text_formula_tables
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
from qa.generate import answer as qa_answer                                # noqa: E402
from qa.graph_qa import detect_intent, graph_answer as graph_qa_answer     # noqa: E402
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
    "unresolved": [],
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
    # 同一 OCRResult 重复入库时，从原始块重新生成本轮证据，防止证据重复进抽取流。
    ocr.blocks = [b for b in ocr.blocks if b.kind != 'small_column_evidence']
    entities, candidates, unresolved = extract_all_audited(ocr.blocks, is_demo=ocr.is_demo)
    triples = build_triples(entities, ocr.blocks, candidates)
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
                           formula_extraction=formula_extraction)
    export_graph_json(graph)

    _STATE.update({
        "page_id": ocr.page_id,
        "ocr": ocr,
        "blocks": ocr.blocks,
        "entities": entities,
        "triples": triples,
        "unresolved": unresolved,
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
    if tri or not blocks:
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
                       "triples": [], "unresolved": [], "graph": None,
                       "vector_index": None})
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
    """可溯源问答。无证据 / 越界一律拒答，所有输出带免责声明。

    入口分两路（阶段十）：
      1. 图谱关系题（症状→药材 / 症状→方剂 / 药材→禁忌）：走 qa.graph_qa，
         只引用当前页三元组，逐条校验出处与主体归属；
      2. 其他题：沿用原有文本混合检索（qa.generate.answer），口径不变。
    """
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
        entities, triples = _objects_for(page_id, blocks, is_demo=flags["is_demo"])
        ans = graph_qa_answer(
            question, blocks, triples, entities=entities, page_id=page_id or "",
            is_demo=flags["is_demo"],
            page_exists=bool(page_id) and store.get_page(page_id) is not None,
            other_page_hits=_cross_page_hits(page_id, question) if page_id else 0,
        )
        if ans is None:
            index = _STATE["vector_index"] if page_id and page_id == _STATE["page_id"] else None
            ans = qa_answer(question, blocks, vector_index=index, is_demo=flags["is_demo"])
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
