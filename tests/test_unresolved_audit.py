# -*- coding: utf-8 -*-
"""
tests/test_unresolved_audit.py —— 「未确认归属」从抽取到入库的护栏

锁死的五件事：
  1. extract_all_audited 与 extract_all 的实体 / 三元组逐条一致，只多出审计记录；
  2. 没有可确认主体时**不产出关系**，但审计记录带齐
     page_id / line_id / 关系 / 客体 / 原因 / bbox / quote；
  3. 正常入库（store.save_page 传 unresolved）后可按页查到，页与页之间互不串；
  4. 同一页重复处理是整页替换，不堆积、不留下上一版记录；
  5. 删除页面后，该页审计记录一并清除。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import store                                      # noqa: E402
from app.schemas import Block, OCRResult, source_hash_of    # noqa: E402
from extract.rules import (candidate_triples_audited,        # noqa: E402
                          extract_all, extract_all_audited)
from ocr.normalize import normalize_text                    # noqa: E402

PID_A = "P-AUDIT-A"
PID_B = "P-AUDIT-B"


def _block(text, i, pid=PID_A):
    return Block(page_id=pid, bbox=[10 * i, 20, 10 * i + 9, 300],
                 line_id=f"R{i:03d}", quote=text, source_hash=source_hash_of(pid, text),
                 text=text, text_norm=normalize_text(text), kind="paragraph",
                 conf=0.9, col_id=0, reading_order=i, block_ref=f"{pid}:R{i:03d}")


def _ocr(pid=PID_A, blocks=None):
    blocks = blocks or [_block("此方不但治真陽虚損", 1, pid)]
    return OCRResult(page_id=pid, image_path=f"{pid}.png", width=800, height=1200,
                     engine="rapidocr", model_loaded=True, is_demo=False,
                     source_hash=source_hash_of("page", pid), blocks=blocks)


def _temp_paths():
    """临时 store 目录：结构与 PATHS 一致，避免测试写进真实库。"""
    root = Path(tempfile.mkdtemp(prefix="tcm-audit-"))
    paths = SimpleNamespace(
        store_dir=root / "store", uploads_dir=root / "uploads",
        db_path=root / "store" / "tcm.sqlite3",
        ensure_dirs=lambda: [(root / "store").mkdir(parents=True, exist_ok=True),
                             (root / "uploads").mkdir(parents=True, exist_ok=True)],
    )
    (root / "store" / "pages").mkdir(parents=True, exist_ok=True)
    return root, paths


class TestExtractAudit(unittest.TestCase):
    def test_audited_matches_plain_extract(self):
        blocks = [_block("甘草 味甘平 無毒", 1), _block("主治五藏六府寒熱邪氣", 2),
                  _block("此方不但治真陽虚損", 3)]
        ents_plain, tris_plain = extract_all(blocks, is_demo=False)
        ents_audit, tris_audit, skipped = extract_all_audited(blocks, is_demo=False)
        self.assertEqual([e.to_dict() for e in ents_plain],
                         [e.to_dict() for e in ents_audit])
        self.assertEqual([t.to_dict() for t in tris_plain],
                         [t.to_dict() for t in tris_audit])
        # 前两行有主体，第三行的「此方」本页无方名可指代 → 只进审计
        self.assertTrue(skipped)

    def test_no_subject_produces_no_triple(self):
        blocks = [_block("此方不但治真陽虚損", 1)]
        triples, skipped = candidate_triples_audited(blocks, is_demo=False)
        self.assertEqual(triples, [])
        self.assertEqual(len(skipped), 1)
        rec = skipped[0]
        self.assertEqual(rec["page_id"], PID_A)
        self.assertEqual(rec["line_id"], "R001")
        self.assertEqual(rec["relation"], "主治")
        self.assertIn("真阳虚损", rec["tail"])   # tail 取自 text_norm（繁简归一）
        self.assertIn("无方名可指代", rec["reason"])
        # 审计记录必须能回原图：坐标与原文都要在
        self.assertEqual(rec["bbox"], [10, 20, 19, 300])
        self.assertEqual(rec["quote"], "此方不但治真陽虚損")
        self.assertTrue(rec["source_hash"])

    def test_subject_present_no_audit(self):
        blocks = [_block("理中湯", 1), _block("治中氣不運腹中不實", 2)]
        triples, skipped = candidate_triples_audited(blocks, is_demo=False)
        self.assertEqual(skipped, [])
        self.assertIn(("理中汤", "主治"), {(t.head, t.relation) for t in triples})


class TestAuditPersistence(unittest.TestCase):
    def setUp(self):
        self.root, self.paths = _temp_paths()
        self.patch = mock.patch.object(store, "PATHS", self.paths)
        self.patch.start()
        store.init_db()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _save(self, pid, texts, extra=None):
        blocks = [_block(t, i + 1, pid) for i, t in enumerate(texts)]
        if extra:
            blocks += extra
        ents, tris, skipped = extract_all_audited(blocks, is_demo=False)
        stat = store.save_page(_ocr(pid, blocks), ents, tris, unresolved=skipped)
        return stat, skipped

    def test_saved_and_readable(self):
        stat, skipped = self._save(PID_A, ["此方不但治真陽虚損"])
        self.assertEqual(stat["unresolved"], len(skipped))
        rows = store.get_unresolved(PID_A)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["line_id"], "R001")
        self.assertEqual(rows[0]["relation"], "主治")
        self.assertIn("真阳虚损", rows[0]["tail"])
        self.assertTrue(rows[0]["reason"])

    def test_snapshot_carries_audit(self):
        self._save(PID_A, ["此方不但治真陽虚損"])
        snap = store.get_page(PID_A)
        self.assertIn("unresolved", snap)
        self.assertEqual(snap["unresolved"][0]["line_id"], "R001")

    def test_page_isolation(self):
        self._save(PID_A, ["此方不但治真陽虚損"])
        self._save(PID_B, ["此方治傷寒頭痛"])
        self.assertEqual(len(store.get_unresolved(PID_A)), 1)
        self.assertEqual(len(store.get_unresolved(PID_B)), 1)
        self.assertEqual({r["page_id"] for r in store.get_unresolved(PID_A)}, {PID_A})
        self.assertEqual(len(store.get_unresolved()), 2)

    def test_repeat_processing_replaces_not_appends(self):
        self._save(PID_A, ["此方不但治真陽虚損"])
        first = len(store.get_unresolved(PID_A))
        self._save(PID_A, ["此方不但治真陽虚損"])          # 同一页再跑一次
        second = len(store.get_unresolved(PID_A))
        self.assertEqual(first, second)
        # 内容变了（重新 OCR 后行文不同）：旧记录不应残留
        self._save(PID_A, ["前方治中風不省人事"])
        rows = store.get_unresolved(PID_A)
        self.assertEqual(len(rows), 1)
        self.assertIn("中风", rows[0]["tail"])

    def test_delete_page_clears_audit(self):
        self._save(PID_A, ["此方不但治真陽虚損"])
        self.assertTrue(store.get_unresolved(PID_A))
        stat = store.delete_page(PID_A)
        self.assertTrue(stat["deleted"])
        self.assertEqual(stat["db"].get("unresolved"), 1)
        self.assertEqual(store.get_unresolved(PID_A), [])


class TestOcrPayloadCarriesAudit(unittest.TestCase):
    """接口层：快照里有记录就原样返回，旧记录重算并标 derived。"""

    def test_stored_records_surface(self):
        from app import main
        snap = {"page": {"page_id": PID_A}, "is_demo": False,
                "unresolved": [{"page_id": PID_A, "line_id": "R001", "relation": "主治",
                                "tail": "真陽虚損", "reason": "无方名可指代",
                                "bbox": [1, 2, 3, 4], "quote": "此方不但治真陽虚損",
                                "source_hash": "h", "reading_order": 1}]}
        with mock.patch.object(main.store, "get_page", return_value=snap), \
             mock.patch.object(main.store, "get_blocks", return_value=[]):
            res = main.get_ocr(PID_A)
        self.assertEqual(res["unresolved_source"], "stored")
        self.assertEqual(res["unresolved_count"], 1)
        self.assertEqual(res["unresolved"][0]["line_id"], "R001")
        self.assertEqual(res["unresolved"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])

    def test_legacy_snapshot_derives_and_marks(self):
        from app import main
        blocks = [_block("此方不但治真陽虚損", 1).to_dict()]
        snap = {"page": {"page_id": PID_A}, "is_demo": False}
        res = main._unresolved_payload(PID_A, snap, blocks)
        self.assertEqual(res["unresolved_source"], "derived")
        self.assertEqual(res["unresolved_count"], 1)
        self.assertIn("重算", res["unresolved_note"])

    def test_no_audit_is_empty_not_error(self):
        from app import main
        snap = {"page": {"page_id": PID_A}, "is_demo": False, "unresolved": []}
        res = main._unresolved_payload(PID_A, snap, [])
        self.assertEqual(res["unresolved"], [])
        self.assertEqual(res["unresolved_count"], 0)
        self.assertEqual(res["unresolved_note"], "")


if __name__ == "__main__":
    unittest.main()
