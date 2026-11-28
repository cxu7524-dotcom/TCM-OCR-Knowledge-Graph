# -*- coding: utf-8 -*-
"""
app/store.py —— SQLite + JSON 幂等入库层

继承自旧 A23 `agent/store.py` 的设计思想（原样保留三条）：
  1. 保真：每条数据都带 quote（原文摘抄）+ 定位坐标，绝不改写原文；
  2. 规范：幂等入库（同来源同内容重复跑不膨胀、不重复）；
  3. 批次：每次入库生成 run_id，同批数据同一批次号，可追溯。
改造点：
  - 旧关系型数据库及其驱动 → 标准库 sqlite3，零装机门槛；
  - 旧 loc（行号/段落号）扩展为 bbox[x1,y1,x2,y2] + line_id + col_id；
  - 表从「办公文档五表」换成「古籍页 / 文本块 / 实体 / 三元组 / 未确认归属 / 问答日志」；
  - 同时写一份 JSON 镜像到 data/store/，便于人工核对与前端直接消费。

幂等实现：run_id 默认由「来源指纹」确定性生成（同一输入 → 同一批次号），
配合 UNIQUE(page_id, ref, run_id) 约束 + INSERT OR IGNORE，重复跑零膨胀。

命令行：
    python -m app.store --init     建表（幂等）
    python -m app.store --stats    库统计
    python -m app.store --pages    页面清单
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from functools import wraps
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG
from .schemas import (
    Answer, Block, Entity, GraphData, OCRResult, Triple,
    source_hash_of,
)

PATHS = CONFIG.paths
_PAGE_LOCK = threading.RLock()


def _page_locked(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _PAGE_LOCK:
            return fn(*args, **kwargs)
    return wrapped


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    """每次新建连接（FastAPI 多线程下更安全），开启外键与 WAL。"""
    PATHS.ensure_dirs()
    conn = sqlite3.connect(str(PATHS.db_path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


SCHEMA_VERSION = "2"   # v2：唯一键去掉 run_id，保证「同一页重复跑不膨胀」

DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    page_id       TEXT PRIMARY KEY,
    image_path    TEXT,
    width         INTEGER DEFAULT 0,
    height        INTEGER DEFAULT 0,
    engine        TEXT,
    model_loaded  INTEGER DEFAULT 0,
    is_demo       INTEGER DEFAULT 1,
    source_hash   TEXT,
    run_id        TEXT,
    created_at    TEXT NOT NULL,
    dual_track_status   TEXT DEFAULT 'not_run',
    dual_track_evidence TEXT DEFAULT '',
    dual_track_audit    TEXT DEFAULT '',
    dual_track_run      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS blocks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       TEXT NOT NULL,
    block_ref     TEXT NOT NULL,
    bbox          TEXT NOT NULL,
    line_id       TEXT,
    col_id        INTEGER DEFAULT 0,
    reading_order INTEGER DEFAULT 0,
    text          TEXT,
    text_norm     TEXT,
    kind          TEXT,
    conf          REAL DEFAULT 0,
    quote         TEXT,
    source_hash   TEXT,
    run_id        TEXT,
    UNIQUE(page_id, block_ref)
);
CREATE INDEX IF NOT EXISTS ix_blocks_page ON blocks(page_id);

CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       TEXT NOT NULL,
    name          TEXT,
    etype         TEXT,
    value         TEXT,
    block_ref     TEXT,
    bbox          TEXT,
    line_id       TEXT,
    quote         TEXT,
    source_hash   TEXT,
    extractor     TEXT,
    is_demo       INTEGER DEFAULT 1,
    run_id        TEXT,
    fingerprint   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_entities_page ON entities(page_id);

CREATE TABLE IF NOT EXISTS triples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       TEXT NOT NULL,
    head          TEXT,
    relation      TEXT,
    tail          TEXT,
    evidence_refs TEXT,
    bbox          TEXT,
    line_id       TEXT,
    quote         TEXT,
    source_hash   TEXT,
    is_demo       INTEGER DEFAULT 1,
    run_id        TEXT,
    fingerprint   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_triples_page ON triples(page_id);

-- 未确认归属：抽取时「找不到可确认主体」的关系，不产出三元组，但留审计记录，
-- 供上传后核查（哪一页 / 哪一行 / 什么关系 / 什么客体 / 为什么没归）。
CREATE TABLE IF NOT EXISTS unresolved (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       TEXT NOT NULL,
    line_id       TEXT,
    relation      TEXT,
    tail          TEXT,
    reason        TEXT,
    bbox          TEXT,
    quote         TEXT,
    source_hash   TEXT,
    reading_order INTEGER DEFAULT 0,
    run_id        TEXT,
    created_at    TEXT NOT NULL,
    fingerprint   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_unresolved_page ON unresolved(page_id);

-- 正文方剂候选及字段溯源：按页整体覆盖，重复处理保持幂等。
CREATE TABLE IF NOT EXISTS formula_extractions (
    page_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question      TEXT,
    answer_text   TEXT,
    refused       INTEGER DEFAULT 0,
    refuse_reason TEXT,
    evidence_n    INTEGER DEFAULT 0,
    backend       TEXT,
    is_demo       INTEGER DEFAULT 1,
    created_at    TEXT NOT NULL
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """轻量迁移：列不存在就 ALTER TABLE 补上，**不重建表、不丢已有数据**。"""
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return
    if cols and column not in cols:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        except sqlite3.Error:
            pass


def init_db() -> str:
    """幂等建表 / 建索引，可反复调用。返回库文件路径。

    轻量迁移：若检测到旧版 schema（唯一键含 run_id），重建三张明细表。
    这些表只存运行期演示数据，重建不会丢真实标注或样本（二者不进 SQLite）。
    阶段九新增 is_demo 列：用 ALTER TABLE 补齐，不重建（保留已入库的页面）。
    """
    PATHS.ensure_dirs()
    conn = connect()
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone() \
            if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name='schema_meta'").fetchone() else None
        version = row[0] if row else ""
        conn.executescript(DDL)
        if version != SCHEMA_VERSION:
            for t in ("blocks", "entities", "triples"):
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.executescript(DDL)
            conn.execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES('version',?)",
                         (SCHEMA_VERSION,))
        # 列级迁移（幂等）：旧库补 is_demo，保留数据
        _ensure_column(conn, "entities", "is_demo", "INTEGER DEFAULT 1")
        _ensure_column(conn, "triples", "is_demo", "INTEGER DEFAULT 1")
        _ensure_column(conn, "blocks", "evidence_role", "TEXT DEFAULT ''")
        _ensure_column(conn, "blocks", "evidence_status", "TEXT DEFAULT ''")
        _ensure_column(conn, "blocks", "parent_block_ref", "TEXT DEFAULT ''")
        _ensure_column(conn, "blocks", "evidence_engine", "TEXT DEFAULT ''")
        _ensure_column(conn, "entities", "llm_metadata", "TEXT DEFAULT ''")
        _ensure_column(conn, "triples", "llm_metadata", "TEXT DEFAULT ''")
        _ensure_column(conn, "unresolved", "llm_metadata", "TEXT DEFAULT ''")
        _ensure_column(conn, "pages", "extraction_meta", "TEXT DEFAULT ''")
        _ensure_column(conn, "pages", "ocr_payload", "TEXT DEFAULT ''")
        # 阶段七：双轨状态与证据单独成列，便于不解析整份 payload 就能查「这页跑没跑、跑成什么」
        _ensure_column(conn, "pages", "dual_track_status", "TEXT DEFAULT 'not_run'")
        _ensure_column(conn, "pages", "dual_track_evidence", "TEXT DEFAULT ''")
        _ensure_column(conn, "pages", "dual_track_audit", "TEXT DEFAULT ''")
        # 「是否尝试过」必须独立成列：failed/skipped 也是有 run 状态的组合，
        # 只靠 status 反推会把 failed+run=true 读成 false（stage7-v2 修的缺陷）。
        _ensure_column(conn, "pages", "dual_track_run", "INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    return str(PATHS.db_path)


def make_run_id(source_hash: str, kind: str = "ingest") -> str:
    """确定性批次号：同一来源指纹 → 同一 run_id，从根上保证重复跑不膨胀。"""
    return "run-" + source_hash_of(kind, source_hash) + "-" + (source_hash or "demo")[:8]


def _bbox_json(value: Any) -> str:
    try:
        return json.dumps(list(value or [0, 0, 0, 0]), ensure_ascii=False)
    except TypeError:
        return "[0,0,0,0]"


def _dual_json(value: Any) -> str:
    """双轨 payload 落库：None → ''（没跑/失败/跳过），[] → '[]'（跑了没检出）。

    两种「空」必须分开存，否则读回后无法区分 not_run 与 completed 无结果。
    """
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def _dual_evidence_from(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


AUDIT_FIELDS = ("page_id", "line_id", "relation", "tail", "reason",
                "bbox", "quote", "source_hash", "reading_order")
AUDIT_EXTRA_FIELDS = ("block_ref", "entry_id", "evidence", "candidate_entry_ids", "candidate_anchor",
                      "subject", "subject_evidence", "object_evidence", "anchor_evidence", "qualifier", "condition_evidence",
                      "scope_evidence", "reason_detail")


def _norm_audit(record: Any, page_id: str) -> Dict[str, Any]:
    """把一条「未确认归属」记录整理成固定字段，快照 / 接口 / 审计三处同一形状。"""
    u = record if isinstance(record, dict) else {}
    try:
        bbox = [float(v) for v in (u.get("bbox") or [0, 0, 0, 0])][:4]
    except (TypeError, ValueError):
        bbox = [0, 0, 0, 0]
    if len(bbox) != 4:
        bbox = [0, 0, 0, 0]
    out = {key: (u.get(key) if key != "bbox" else bbox) for key in AUDIT_FIELDS}
    out["page_id"] = str(out.get("page_id") or page_id)
    for key in ("line_id", "relation", "tail", "reason", "quote", "source_hash"):
        out[key] = str(out.get(key) or "")
    out["reading_order"] = int(out.get("reading_order") or 0)
    for key in AUDIT_EXTRA_FIELDS:
        if key in u:
            out[key] = u[key]
    return out


@_page_locked
def save_page(
    ocr: OCRResult,
    entities: Optional[List[Entity]] = None,
    triples: Optional[List[Triple]] = None,
    run_id: Optional[str] = None,
    unresolved: Optional[List[Dict[str, Any]]] = None,
    formula_extraction: Optional[Dict[str, Any]] = None,
    extraction_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """一个事务内写入 page / blocks / entities / triples，并返回写入统计。

    run_id 为空时按 source_hash 确定性生成；重复调用不会新增任何行。
    unresolved 为「未确认归属」审计记录：传入即整页替换（先删后插），
    保证同一页重复处理不堆积、也不会留下上一版 OCR 的旧记录；
    传 None 表示本次不涉及（例如早期调用方），表内容原样保留。
    """
    init_db()
    entities = entities or []
    triples = triples or []
    rid = run_id or make_run_id(ocr.source_hash or ocr.page_id, "ingest")
    sh = ocr.source_hash or source_hash_of(ocr.page_id, "page")
    stat = {"run_id": rid, "page_id": ocr.page_id,
            "blocks": 0, "entities": 0, "triples": 0, "skipped": 0,
            "unresolved": 0}

    conn = connect()
    try:
        conn.execute("INSERT OR IGNORE INTO runs(run_id,kind,note,created_at) VALUES(?,?,?,?)",
                     (rid, "ingest", f"{ocr.page_id} · engine={ocr.engine}", _now()))
        cur = conn.execute(
            "INSERT OR IGNORE INTO pages"
            "(page_id,image_path,width,height,engine,model_loaded,is_demo,source_hash,run_id,created_at,"
            " dual_track_status,dual_track_evidence,dual_track_audit,dual_track_run)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ocr.page_id, ocr.image_path, ocr.width, ocr.height, ocr.engine,
             int(ocr.model_loaded), int(ocr.is_demo), sh, rid, _now(),
             ocr.dual_track_status or "not_run",
             _dual_json(ocr.dual_track_evidence),
             _dual_json(ocr.dual_track_audit),
             int(bool(getattr(ocr, "dual_track_run", False)))))
        stat["skipped"] += 1 if cur.rowcount == 0 else 0

        if extraction_meta is not None:
            conn.execute("UPDATE pages SET image_path=?,width=?,height=?,engine=?,model_loaded=?,is_demo=?,source_hash=?,run_id=?,extraction_meta=?,ocr_payload=?,dual_track_status=?,dual_track_evidence=?,dual_track_audit=?,dual_track_run=? WHERE page_id=?",
                         (ocr.image_path, ocr.width, ocr.height, ocr.engine, int(ocr.model_loaded), int(ocr.is_demo), sh, rid,
                          json.dumps(extraction_meta, ensure_ascii=False), json.dumps(ocr.to_dict(), ensure_ascii=False),
                          ocr.dual_track_status or "not_run",
                          _dual_json(ocr.dual_track_evidence),
                          _dual_json(ocr.dual_track_audit),
                          int(bool(getattr(ocr, "dual_track_run", False))), ocr.page_id))
            for table in ("blocks", "entities", "triples"):
                conn.execute(f"DELETE FROM {table} WHERE page_id=?", (ocr.page_id,))

        # 子框证据与开关状态同步替换；不触碰原 OCR 块或其他页面。
        conn.execute("DELETE FROM blocks WHERE page_id=? AND kind=?", (ocr.page_id, "small_column_evidence"))
        for b in ocr.blocks:
            cur = conn.execute(
                "INSERT OR IGNORE INTO blocks"
                "(page_id,block_ref,bbox,line_id,col_id,reading_order,text,text_norm,kind,conf,quote,source_hash,run_id,evidence_role,evidence_status,parent_block_ref,evidence_engine)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, b.block_ref or b.line_id, _bbox_json(b.bbox), b.line_id,
                 b.col_id, b.reading_order, b.text, b.text_norm, b.kind, b.conf,
                 b.quote, b.source_hash, rid, b.evidence_role, b.evidence_status, b.parent_block_ref, b.evidence_engine))
            stat["blocks"] += cur.rowcount

        for e in entities:
            # 唯一键只由「页 + 内容」决定，与批次 / 来源指纹无关，重复跑不膨胀
            fp = (source_hash_of("entity-llm", ocr.page_id, e.entry_id, e.name, e.type, e.value)
                  if e.entry_id else source_hash_of("entity", ocr.page_id, e.name, e.type, e.value, e.quote))
            cur = conn.execute(
                "INSERT OR IGNORE INTO entities"
                "(page_id,name,etype,value,block_ref,bbox,line_id,quote,source_hash,extractor,is_demo,run_id,fingerprint,llm_metadata)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, e.name, e.type, e.value, e.block_ref, _bbox_json(e.bbox),
                 e.line_id, e.quote, e.source_hash, e.extractor, int(e.is_demo), rid, fp,
                 json.dumps({"entry_id": e.entry_id, "evidence": e.evidence}, ensure_ascii=False)))
            stat["entities"] += cur.rowcount

        for t in triples:
            fp = (source_hash_of("triple-llm", ocr.page_id, t.entry_id, t.head, t.relation, t.tail, t.qualifier)
                  if t.entry_id else source_hash_of("triple", ocr.page_id, t.head, t.relation, t.tail))
            cur = conn.execute(
                "INSERT OR IGNORE INTO triples"
                "(page_id,head,relation,tail,evidence_refs,bbox,line_id,quote,source_hash,is_demo,run_id,fingerprint,llm_metadata)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, t.head, t.relation, t.tail,
                 json.dumps(t.evidence_refs, ensure_ascii=False), _bbox_json(t.bbox),
                 t.line_id, t.quote, t.source_hash, int(t.is_demo), rid, fp,
                 json.dumps({"entry_id": t.entry_id, "qualifier": t.qualifier,
                             "subject_evidence": t.subject_evidence, "relation_evidence": t.relation_evidence,
                             "condition_evidence": t.condition_evidence}, ensure_ascii=False)))
            stat["triples"] += cur.rowcount

        if unresolved is not None:
            # 整页替换：避免同一页重跑后新旧两版审计记录并存
            conn.execute("DELETE FROM unresolved WHERE page_id=?", (ocr.page_id,))
            for u in unresolved:
                line_id = str(u.get("line_id") or "")
                relation = str(u.get("relation") or "")
                tail = str(u.get("tail") or "")
                metadata = {key: u[key] for key in AUDIT_EXTRA_FIELDS if key in u}
                fp = (source_hash_of("unresolved-llm", ocr.page_id, line_id, relation, tail,
                                     str(u.get("reason") or ""), json.dumps(metadata, ensure_ascii=False, sort_keys=True))
                      if metadata else source_hash_of("unresolved", ocr.page_id, line_id, relation, tail))
                cur = conn.execute(
                    "INSERT OR IGNORE INTO unresolved"
                    "(page_id,line_id,relation,tail,reason,bbox,quote,source_hash,"
                    "reading_order,run_id,created_at,fingerprint,llm_metadata)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ocr.page_id, line_id, relation, tail, str(u.get("reason") or ""),
                     _bbox_json(u.get("bbox")), str(u.get("quote") or ""),
                     str(u.get("source_hash") or ""), int(u.get("reading_order") or 0),
                     rid, _now(), fp, json.dumps(metadata, ensure_ascii=False)))
                stat["unresolved"] += cur.rowcount

        if formula_extraction is not None:
            conn.execute("INSERT OR REPLACE INTO formula_extractions(page_id,payload) VALUES(?,?)",
                         (ocr.page_id, json.dumps(formula_extraction, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()

    # JSON 镜像：完整快照，人工可直接核对（含 bbox / quote / source_hash）
    audit = ([_norm_audit(u, ocr.page_id) for u in unresolved]
             if unresolved is not None else get_unresolved(ocr.page_id))
    snapshot = {
        "page": ocr.to_dict(),
        "entities": [e.to_dict() for e in entities],
        "triples": [t.to_dict() for t in triples],
        # 未确认归属审计：不产出关系，但入库留存，上传后可核查
        "unresolved": audit,
        "run_id": rid,
        "is_demo": ocr.is_demo,
        "note": CONFIG.demo_badge if ocr.is_demo else "本地 OCR 真实识别（非命题单位样本）",
    }
    if extraction_meta is not None:
        snapshot["extraction"] = extraction_meta
    if formula_extraction is not None:
        snapshot["formula_extraction"] = formula_extraction
    else:
        previous = get_formula_extraction(ocr.page_id)
        if previous is not None:
            snapshot["formula_extraction"] = previous
    try:
        snap = PATHS.store_dir / "pages" / f"{ocr.page_id}.json"
        # page_id 可能带书页层级（如「丹方彚编/page_12」），先建父目录再写
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        if extraction_meta is None:
            raise
    return stat


def save_graph(graph: GraphData) -> Path:
    """图谱 JSON 镜像（前端 /api/graph 也读它）。"""
    PATHS.ensure_dirs()
    out = PATHS.store_dir / "graph.json"
    out.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def log_qa(question: str, answer: Answer) -> None:
    """问答日志入库，供拒答率 / 可溯源率审计使用。"""
    init_db()
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO qa_logs(question,answer_text,refused,refuse_reason,evidence_n,backend,is_demo,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (question, answer.text, int(answer.refused), answer.refuse_reason,
             len(answer.evidences), answer.backend, int(answer.is_demo), _now()))
        conn.commit()
    finally:
        conn.close()


def list_pages() -> List[Dict[str, Any]]:
    init_db()
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT page_id,image_path,engine,model_loaded,is_demo,source_hash,run_id,created_at"
            " FROM pages ORDER BY created_at DESC, page_id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@_page_locked
def get_page(page_id: str) -> Optional[Dict[str, Any]]:
    """优先读 JSON 快照（结构完整），缺失时回退 SQLite 页面表。"""
    init_db()
    conn = connect()
    try:
        conn.execute("BEGIN")
        managed = conn.execute("SELECT extraction_meta,ocr_payload,is_demo,dual_track_status,dual_track_evidence,dual_track_audit,dual_track_run FROM pages WHERE page_id=?", (page_id,)).fetchone()
        if managed and managed["extraction_meta"]:
            entities = []
            for row in conn.execute("SELECT * FROM entities WHERE page_id=? ORDER BY id", (page_id,)):
                data = dict(row)
                data.update(json.loads(data.get("llm_metadata") or "{}"))
                data["bbox"] = json.loads(data.get("bbox") or "[0,0,0,0]")
                data["type"] = data.pop("etype", "")
                data["is_demo"] = bool(data.get("is_demo", 1))
                entities.append(Entity.from_dict(data).to_dict())
            triples = []
            for row in conn.execute("SELECT * FROM triples WHERE page_id=? ORDER BY id", (page_id,)):
                data = dict(row)
                data.update(json.loads(data.get("llm_metadata") or "{}"))
                data["bbox"] = json.loads(data.get("bbox") or "[0,0,0,0]")
                data["evidence_refs"] = json.loads(data.get("evidence_refs") or "[]")
                data["is_demo"] = bool(data.get("is_demo", 1))
                triples.append(Triple.from_dict(data).to_dict())
            unresolved = []
            for row in conn.execute("SELECT * FROM unresolved WHERE page_id=? ORDER BY reading_order,id", (page_id,)):
                data = dict(row)
                data.update(json.loads(data.get("llm_metadata") or "{}"))
                data["bbox"] = json.loads(data.get("bbox") or "[0,0,0,0]")
                unresolved.append(_norm_audit(data, page_id))
            formula = conn.execute("SELECT payload FROM formula_extractions WHERE page_id=?", (page_id,)).fetchone()
            # 负载可能为空/损坏（历史页、手工清库）；退回页面列重建，保证双轨状态仍可读回
            page: Optional[Dict[str, Any]] = None
            if managed["ocr_payload"]:
                try:
                    parsed = json.loads(managed["ocr_payload"])
                    page = parsed if isinstance(parsed, dict) else None
                except (TypeError, ValueError):
                    page = None
            if page is None:
                row = conn.execute("SELECT * FROM pages WHERE page_id=?", (page_id,)).fetchone()
                page = {k: row[k] for k in row.keys()
                        if k not in ("extraction_meta", "ocr_payload")}
            # 旧页面的 payload 里没有双轨字段：用列值补齐，保持「None 与 [] 不同」的语义
            page.setdefault("dual_track_status", managed["dual_track_status"] or "not_run")
            if "dual_track_evidence" not in page:
                page["dual_track_evidence"] = _dual_evidence_from(managed["dual_track_evidence"])
            elif not isinstance(page["dual_track_evidence"], (list, type(None))):
                page["dual_track_evidence"] = _dual_evidence_from(page["dual_track_evidence"])
            if "dual_track_audit" not in page:
                page["dual_track_audit"] = _dual_evidence_from(managed["dual_track_audit"])
            elif not isinstance(page["dual_track_audit"], (list, type(None))):
                page["dual_track_audit"] = _dual_evidence_from(page["dual_track_audit"])
            # 「是否尝试过」以列为准（failed + None + run=true 不能靠 status 反推）
            page["dual_track_run"] = bool(page.get("dual_track_run",
                                                   managed["dual_track_run"]))
            return {"page": page, "extraction": json.loads(managed["extraction_meta"]),
                    "entities": entities, "triples": triples, "unresolved": unresolved,
                    "formula_extraction": json.loads(formula[0]) if formula else None,
                    "is_demo": bool(managed["is_demo"])}
    except (TypeError, ValueError):
        return None
    finally:
        conn.close()
    snap = PATHS.store_dir / "pages" / f"{page_id}.json"
    if snap.exists():
        try:
            return json.loads(snap.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM pages WHERE page_id=?", (page_id,)).fetchone()
        if not row:
            return None
        page = dict(row)
        # 无 payload 的旧页：双轨字段只可能来自列值；缺列（极旧库）同样是 not_run/None
        page["dual_track_status"] = page.get("dual_track_status") or "not_run"
        page["dual_track_evidence"] = _dual_evidence_from(page.get("dual_track_evidence"))
        page["dual_track_audit"] = _dual_evidence_from(page.get("dual_track_audit"))
        page["dual_track_run"] = bool(page.get("dual_track_run"))
        result = {"page": page, "entities": [], "triples": [], "is_demo": bool(page.get("is_demo", 1))}
        saved = conn.execute("SELECT payload FROM formula_extractions WHERE page_id=?", (page_id,)).fetchone()
        if saved:
            try:
                extraction = json.loads(saved[0])
            except (TypeError, ValueError):
                extraction = None
            if isinstance(extraction, dict):
                result["formula_extraction"] = extraction
                if isinstance(extraction.get("_printed_tables"), list):
                    page["tables"] = extraction["_printed_tables"]
        return result
    finally:
        conn.close()



def get_formula_extraction(page_id: str) -> Optional[Dict[str, Any]]:
    """读取入库的正文结果；None 表示旧页没有该字段，空结果与之不同。"""
    init_db()
    conn = connect()
    try:
        row = conn.execute("SELECT payload FROM formula_extractions WHERE page_id=?", (page_id,)).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    finally:
        conn.close()


def get_blocks(page_id: str) -> List[Block]:
    """从 SQLite 还原 Block 对象（保证溯源字段完整）。"""
    init_db()
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM blocks WHERE page_id=? ORDER BY reading_order, id",
                            (page_id,)).fetchall()
        out: List[Block] = []
        for r in rows:
            d = dict(r)
            try:
                d["bbox"] = json.loads(d.get("bbox") or "[0,0,0,0]")
            except json.JSONDecodeError:
                d["bbox"] = [0, 0, 0, 0]
            d["page_id"] = page_id
            out.append(Block.from_dict(d))
        return out
    finally:
        conn.close()


def get_triples(page_id: Optional[str] = None) -> List[Triple]:
    init_db()
    conn = connect()
    try:
        sql = "SELECT * FROM triples" + (" WHERE page_id=?" if page_id else "")
        rows = conn.execute(sql, (page_id,) if page_id else ()).fetchall()
        out: List[Triple] = []
        for r in rows:
            d = dict(r)
            try:
                d.update(json.loads(d.get("llm_metadata") or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            try:
                d["evidence_refs"] = json.loads(d.get("evidence_refs") or "[]")
                d["bbox"] = json.loads(d.get("bbox") or "[0,0,0,0]")
            except json.JSONDecodeError:
                d["evidence_refs"], d["bbox"] = [], [0, 0, 0, 0]
            d["is_demo"] = bool(d.get("is_demo", 1))
            out.append(Triple.from_dict(d))
        return out
    finally:
        conn.close()


def get_entities(page_id: str) -> List[Entity]:
    """从 SQLite 还原 Entity（JSON 快照缺失时的兜底；快照优先，见 get_page）。"""
    init_db()
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM entities WHERE page_id=? ORDER BY id",
                            (page_id,)).fetchall()
        out: List[Entity] = []
        for r in rows:
            d = dict(r)
            try:
                d.update(json.loads(d.get("llm_metadata") or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            try:
                d["bbox"] = json.loads(d.get("bbox") or "[0,0,0,0]")
            except json.JSONDecodeError:
                d["bbox"] = [0, 0, 0, 0]
            d["page_id"] = page_id
            d["type"] = d.pop("etype", d.get("type", ""))
            d["is_demo"] = bool(d.get("is_demo", 1))
            out.append(Entity.from_dict(d))
        return out
    finally:
        conn.close()


def get_unresolved(page_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """「未确认归属」审计记录：可按页过滤（不传则全库），按原文顺序返回。

    这些记录**不是**三元组，不参与图谱与问答；只用于上传后核查
    「这一行本来该归给谁、为什么没归」。
    """
    init_db()
    conn = connect()
    try:
        sql = ("SELECT page_id,line_id,relation,tail,reason,bbox,quote,source_hash,"
               "reading_order,llm_metadata FROM unresolved")
        if page_id:
            sql += " WHERE page_id=?"
        sql += " ORDER BY reading_order, id"
        rows = conn.execute(sql, (page_id,) if page_id else ()).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d.update(json.loads(d.get("llm_metadata") or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            try:
                d["bbox"] = json.loads(d.get("bbox") or "[0,0,0,0]")
            except json.JSONDecodeError:
                d["bbox"] = [0, 0, 0, 0]
            out.append(_norm_audit(d, d.get("page_id") or page_id or ""))
        return out
    finally:
        conn.close()


def delete_page(page_id: str) -> Dict[str, Any]:
    """删除一页及其全部派生数据（SQLite 五表 + JSON 快照 + 上传图片）。

    只删本系统命名的上传文件（`P<时间戳>-<hash>.<ext>`），不碰任何其他文件。
    返回删除统计；页面不存在时 `deleted=False`。
    """
    init_db()
    stat: Dict[str, Any] = {"page_id": page_id, "deleted": False,
                            "db": {}, "snapshot": False, "image": ""}
    conn = connect()
    try:
        row = conn.execute("SELECT image_path FROM pages WHERE page_id=?", (page_id,)).fetchone()
        if row is None:
            return stat
        image_path = row[0] or ""
        for table in ("blocks", "entities", "triples", "unresolved", "formula_extractions"):
            cur = conn.execute(f"DELETE FROM {table} WHERE page_id=?", (page_id,))
            stat["db"][table] = cur.rowcount
        conn.execute("DELETE FROM pages WHERE page_id=?", (page_id,))
        conn.commit()
        stat["deleted"] = True
    finally:
        conn.close()

    snap = PATHS.store_dir / "pages" / f"{page_id}.json"
    if snap.exists():
        try:
            snap.unlink()
            stat["snapshot"] = True
        except OSError:
            pass

    # 只删 data/uploads/ 内、由本系统命名的文件
    name = Path(image_path).name if image_path else ""
    if name and _is_managed_upload(name):
        target = (PATHS.uploads_dir / name).resolve()
        try:
            if target.is_file() and PATHS.uploads_dir.resolve() in target.parents:
                target.unlink()
                stat["image"] = name
        except OSError:
            pass
    return stat


def _is_managed_upload(name: str) -> bool:
    """本系统命名的上传文件：P<14 位时间戳>-<6 位 hash>.<ext>。"""
    import re
    return bool(re.fullmatch(r"P\d{14}-[0-9a-f]{6}\.(png|jpg|jpeg|bmp|webp|tif|tiff)", name))


def stats() -> Dict[str, Any]:
    init_db()
    conn = connect()
    try:
        out: Dict[str, Any] = {}
        for table in ("pages", "blocks", "entities", "triples", "unresolved",
                      "qa_logs", "runs"):
            out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        out["db_path"] = str(PATHS.db_path)
        return out
    finally:
        conn.close()


def _main() -> None:
    ap = argparse.ArgumentParser(description="TCM 原型 · 本地 SQLite 存储")
    ap.add_argument("--init", action="store_true", help="幂等建表")
    ap.add_argument("--stats", action="store_true", help="库统计")
    ap.add_argument("--pages", action="store_true", help="页面清单")
    args = ap.parse_args()
    if args.stats:
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
    elif args.pages:
        print(json.dumps(list_pages(), ensure_ascii=False, indent=2))
    else:
        print("已建库（幂等）:", init_db())
        print(json.dumps(stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
