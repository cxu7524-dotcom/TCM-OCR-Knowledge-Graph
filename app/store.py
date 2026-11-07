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
  - 表从「办公文档五表」换成「古籍页 / 文本块 / 实体 / 三元组 / 问答日志」；
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG
from .schemas import (
    Answer, Block, Entity, GraphData, OCRResult, Triple,
    source_hash_of,
)

PATHS = CONFIG.paths


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
    created_at    TEXT NOT NULL
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
    run_id        TEXT,
    fingerprint   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_triples_page ON triples(page_id);

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


def init_db() -> str:
    """幂等建表 / 建索引，可反复调用。返回库文件路径。

    轻量迁移：若检测到旧版 schema（唯一键含 run_id），重建三张明细表。
    这些表只存运行期演示数据，重建不会丢真实标注或样本（二者不进 SQLite）。
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


def save_page(
    ocr: OCRResult,
    entities: Optional[List[Entity]] = None,
    triples: Optional[List[Triple]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """一个事务内写入 page / blocks / entities / triples，并返回写入统计。

    run_id 为空时按 source_hash 确定性生成；重复调用不会新增任何行。
    """
    init_db()
    entities = entities or []
    triples = triples or []
    rid = run_id or make_run_id(ocr.source_hash or ocr.page_id, "ingest")
    sh = ocr.source_hash or source_hash_of(ocr.page_id, "page")
    stat = {"run_id": rid, "page_id": ocr.page_id,
            "blocks": 0, "entities": 0, "triples": 0, "skipped": 0}

    conn = connect()
    try:
        conn.execute("INSERT OR IGNORE INTO runs(run_id,kind,note,created_at) VALUES(?,?,?,?)",
                     (rid, "ingest", f"{ocr.page_id} · engine={ocr.engine}", _now()))
        cur = conn.execute(
            "INSERT OR IGNORE INTO pages"
            "(page_id,image_path,width,height,engine,model_loaded,is_demo,source_hash,run_id,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ocr.page_id, ocr.image_path, ocr.width, ocr.height, ocr.engine,
             int(ocr.model_loaded), int(ocr.is_demo), sh, rid, _now()))
        stat["skipped"] += 1 if cur.rowcount == 0 else 0

        for b in ocr.blocks:
            cur = conn.execute(
                "INSERT OR IGNORE INTO blocks"
                "(page_id,block_ref,bbox,line_id,col_id,reading_order,text,text_norm,kind,conf,quote,source_hash,run_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, b.block_ref or b.line_id, _bbox_json(b.bbox), b.line_id,
                 b.col_id, b.reading_order, b.text, b.text_norm, b.kind, b.conf,
                 b.quote, b.source_hash, rid))
            stat["blocks"] += cur.rowcount

        for e in entities:
            # 唯一键只由「页 + 内容」决定，与批次 / 来源指纹无关，重复跑不膨胀
            fp = source_hash_of("entity", ocr.page_id, e.name, e.type, e.value, e.quote)
            cur = conn.execute(
                "INSERT OR IGNORE INTO entities"
                "(page_id,name,etype,value,block_ref,bbox,line_id,quote,source_hash,extractor,run_id,fingerprint)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, e.name, e.type, e.value, e.block_ref, _bbox_json(e.bbox),
                 e.line_id, e.quote, e.source_hash, e.extractor, rid, fp))
            stat["entities"] += cur.rowcount

        for t in triples:
            fp = source_hash_of("triple", ocr.page_id, t.head, t.relation, t.tail)
            cur = conn.execute(
                "INSERT OR IGNORE INTO triples"
                "(page_id,head,relation,tail,evidence_refs,bbox,line_id,quote,source_hash,run_id,fingerprint)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (ocr.page_id, t.head, t.relation, t.tail,
                 json.dumps(t.evidence_refs, ensure_ascii=False), _bbox_json(t.bbox),
                 t.line_id, t.quote, t.source_hash, rid, fp))
            stat["triples"] += cur.rowcount

        conn.commit()
    finally:
        conn.close()

    # JSON 镜像：完整快照，人工可直接核对（含 bbox / quote / source_hash）
    snapshot = {
        "page": ocr.to_dict(),
        "entities": [e.to_dict() for e in entities],
        "triples": [t.to_dict() for t in triples],
        "run_id": rid,
        "is_demo": ocr.is_demo,
        "note": CONFIG.demo_badge,
    }
    (PATHS.store_dir / "pages").mkdir(parents=True, exist_ok=True)
    (PATHS.store_dir / "pages" / f"{ocr.page_id}.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
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


def get_page(page_id: str) -> Optional[Dict[str, Any]]:
    """优先读 JSON 快照（结构完整），缺失时回退 SQLite 页面表。"""
    init_db()
    snap = PATHS.store_dir / "pages" / f"{page_id}.json"
    if snap.exists():
        try:
            return json.loads(snap.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM pages WHERE page_id=?", (page_id,)).fetchone()
        return dict(row) if row else None
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
                d["evidence_refs"] = json.loads(d.get("evidence_refs") or "[]")
                d["bbox"] = json.loads(d.get("bbox") or "[0,0,0,0]")
            except json.JSONDecodeError:
                d["evidence_refs"], d["bbox"] = [], [0, 0, 0, 0]
            out.append(Triple.from_dict(d))
        return out
    finally:
        conn.close()


def stats() -> Dict[str, Any]:
    init_db()
    conn = connect()
    try:
        out: Dict[str, Any] = {}
        for table in ("pages", "blocks", "entities", "triples", "qa_logs", "runs"):
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
