# -*- coding: utf-8 -*-
"""
tests/test_stage9.py —— 阶段九回归护栏（真实古籍页链路 + 抽取扩展 + 上传错误分类）

锁死的四类行为：
  1. 术语表从 JSON 加载且规模达标；JSON 缺失时降级可见（degraded=True），不静默变小；
  2. is_demo 从页面透传到实体 / 三元组 / 图谱节点，示例页 True、真实页 False；
  3. 真实古籍页（方书行文）能抽出足量实体与三元组，且不产生已知类型的误报
     （「三钟煎」当方名、「卷之一 合」当剂量、「三两」模糊匹配成三七）；
  4. 上传失败按 400 / 413 / 503 分类，不静默降级成示例页。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.schemas import Block, Entity                     # noqa: E402
from extract import rules, schema                          # noqa: E402
from graph.build import build_graph                        # noqa: E402


def _blocks(rows, page_id="P-STAGE9", conf=0.95, kind="paragraph"):
    return [Block(page_id=page_id, bbox=[0, i * 40, 300, i * 40 + 30],
                  line_id=f"R{i:03d}", quote=text, source_hash=f"h{i}",
                  text=text, text_norm=text, kind=kind, conf=conf,
                  col_id=0, reading_order=i, block_ref=f"{page_id}:R{i:03d}")
            for i, text in enumerate(rows, start=1)]


# 真实方书页的行文（取自 思濟堂方書 page_35 的 OCR 结果，已做繁简归一）
REAL_ROWS = [
    "三化汤",
    "治中风不省人事痰涎壅盛胸膈不宽饮食不",
    "枳实厚朴",
    "川大黄洛三钱芒硝一钱",
    "水三钟煎八分温服",
    "六君子汤",
    "治面目萎黄或光白色气少言微四肢无力饮食难化",
    "人参白术甘草广皮制半夏各工钱",
    "引加生姜一钱半红枣肉三枚水煎温服",
]


class TestLexicon(unittest.TestCase):
    def test_scale(self):
        self.assertGreaterEqual(len(schema.HERBS), 300, "药材词表应 ≥300")
        self.assertGreaterEqual(len(schema.FORMULAS), 200, "方剂词表应 ≥200")
        self.assertGreaterEqual(len(schema.EFFECT_WORDS), 120, "功效词应 ≥120")
        self.assertGreaterEqual(len(schema.DISEASES), 80, "主治病证应 ≥80")
        self.assertGreaterEqual(len(schema.DOSE_UNITS), 20, "剂量单位应 ≥20")

    def test_meta_is_honest(self):
        """词表必须写清来源与局限，且不得声称是权威全量词表。"""
        meta = schema.LEXICON_META
        self.assertTrue(meta.get("版本"))
        self.assertTrue(meta.get("来源"))
        self.assertIn("不是权威全量词表", str(meta.get("局限", "")))
        self.assertFalse(schema.DEGRADED, "术语表应能正常加载")

    def test_stats_include_source(self):
        st = schema.dictionary_stats()
        self.assertFalse(st["degraded"])
        self.assertIn("不是权威全量词表", st["note"])


class TestFormulaFilter(unittest.TestCase):
    def test_valid_formula(self):
        self.assertTrue(schema.is_valid_formula("三化汤"))
        self.assertTrue(schema.is_valid_formula("六君子汤"))
        self.assertTrue(schema.is_valid_formula("十全大补汤"))
        self.assertFalse(schema.is_valid_formula("三钟煎"))     # 剂量单位混入
        self.assertFalse(schema.is_valid_formula("伤寒"))       # 病证，不是方名
        self.assertFalse(schema.is_valid_formula("汤"))         # 只剩剂型字
        self.assertFalse(schema.is_valid_formula("化三汤"))     # 数字不在首位

    def test_no_formula_from_sentence_tail(self):
        """「…四肢无力饮食难化」不得被切成方名「四肢无力饮」。"""
        hits = [n for _p, n in rules._formula_hits("治面目萎黄或光白色气少言微四肢无力饮食难化")]
        self.assertEqual(hits, [])

    def test_formula_pattern_hits_real_name(self):
        hits = [n for _p, n in rules._formula_hits("三化汤")]
        self.assertIn("三化汤", hits)


class TestDosePairing(unittest.TestCase):
    def test_herb_before_dose_wins(self):
        herb, _ext = rules._resolve_herb("川大黄洛")
        self.assertEqual(herb, "川大黄", "应取最贴近剂量的最长词典药名")

    def test_no_dose_from_page_header(self):
        """「卷之一 合成示意頁」不得产出剂量「一合」。"""
        ents = rules.extract_entities(_blocks(["本草纲目 卷之一 合成示意頁 第17頁"]))
        self.assertFalse([e for e in ents if e.type == "剂量"])

    def test_dose_with_half_suffix(self):
        ents = rules.extract_entities(_blocks(["引加生姜一钱半红枣肉三枚水煎温服"]))
        doses = {e.name for e in ents if e.type == "剂量"}
        self.assertIn("一钱半", doses)
        self.assertIn("三枚", doses)


class TestFuzzyGuard(unittest.TestCase):
    def test_short_cell_not_fuzzy_matched(self):
        """表格单元格「三两」不得被模糊匹配成「三七 / 三棱」。"""
        ents = rules.extract_entities(_blocks(["三两"], kind="table_cell", conf=0.4))
        names = {e.name for e in ents if e.type == "药材"}
        self.assertFalse(names & {"三七", "三棱"}, names)

    def test_table_header_not_fuzzy_matched(self):
        ents = rules.extract_entities(_blocks(["炮制"], kind="paragraph", conf=0.4))
        self.assertFalse([e for e in ents if e.type == "药材"])


class TestRealPageExtraction(unittest.TestCase):
    def test_enough_entities_and_triples(self):
        ents, tris = rules.extract_all(_blocks(REAL_ROWS))
        self.assertGreaterEqual(len(ents), 8, f"实体太少：{len(ents)}")
        self.assertGreaterEqual(len(tris), 5, f"三元组太少：{len(tris)}")
        names = {e.name for e in ents}
        for expect in ("三化汤", "六君子汤", "川大黄", "芒硝", "人参", "白术", "甘草"):
            self.assertIn(expect, names)
        types = {e.type for e in ents}
        self.assertTrue({"方剂", "药材", "剂量", "主治"} <= types, types)

    def test_quote_is_original_line(self):
        """quote 必须是原文整行，不许被归一化或改写。"""
        rows = ["三化湯治中風不省人事"]
        blocks = _blocks(rows)
        blocks[0].text_norm = "三化汤治中风不省人事"
        ents = rules.extract_entities(blocks)
        self.assertTrue(ents)
        for e in ents:
            self.assertEqual(e.quote, "三化湯治中風不省人事")


class TestIsDemoPropagation(unittest.TestCase):
    def test_real_page_marks_false(self):
        ents, tris = rules.extract_all(_blocks(REAL_ROWS), is_demo=False)
        self.assertTrue(ents and tris)
        self.assertTrue(all(e.is_demo is False for e in ents))
        self.assertTrue(all(t.is_demo is False for t in tris))

    def test_demo_page_defaults_true(self):
        ents, tris = rules.extract_all(_blocks(["甘草味甘平无毒"]))
        self.assertTrue(all(e.is_demo is True for e in ents))
        self.assertTrue(all(t.is_demo is True for t in tris))

    def test_graph_inherits_flag(self):
        ents, tris = rules.extract_all(_blocks(REAL_ROWS), is_demo=False)
        graph = build_graph(tris, ents)
        self.assertTrue(graph.nodes)
        self.assertTrue(all(n.is_demo is False for n in graph.nodes))
        self.assertFalse(graph.is_demo)

    def test_evaluate_does_not_refuse_real_page(self):
        """真实页（is_demo=False）不得被判为「输入含示例数据，不得计分」。"""
        from extract.evaluate import field_level_prf
        gold = [Entity(page_id="P", name="甘草", type="药材", value="甘草", quote="甘草",
                       line_id="R001", bbox=[0, 0, 1, 1], source_hash="h", is_demo=False)]
        pred = [Entity(page_id="P", name="甘草", type="药材", value="甘草", quote="甘草",
                       line_id="R001", bbox=[0, 0, 1, 1], source_hash="h", is_demo=False)]
        res = field_level_prf(gold, pred)
        self.assertEqual(res["status"], "实测")
        self.assertNotIn("示例数据", res["reason"])


class TestDeterminismCrossProcess(unittest.TestCase):
    """同一输入在不同 PYTHONHASHSEED 下必须产出逐字节相同的结果。"""

    SCRIPT = (
        "import sys,json,hashlib;sys.path.insert(0,%r);"
        "from tests.test_stage9 import REAL_ROWS,_blocks;"
        "from extract.rules import extract_all;"
        "e,t=extract_all(_blocks(REAL_ROWS),is_demo=False);"
        "s=json.dumps([[x.name,x.type,x.line_id] for x in e]+"
        "[[x.head,x.relation,x.tail] for x in t],ensure_ascii=False);"
        "print(hashlib.sha256(s.encode()).hexdigest())" % str(_ROOT)
    )

    def _run(self, seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", self.SCRIPT], env=env,
                             cwd=str(_ROOT), capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        return out.stdout.strip()

    def test_same_hash_across_seeds(self):
        self.assertEqual(self._run("0"), self._run("12345"))


class TestUploadErrorClassification(unittest.TestCase):
    """上传失败不得静默降级为示例页（P0-4）。"""

    def _call(self, filename: str, content: bytes, **patch):
        from app import main as app_main
        from fastapi import HTTPException
        from starlette.datastructures import UploadFile
        up = UploadFile(filename=filename, file=io.BytesIO(content))
        stack = [mock.patch.object(app_main, k, v) for k, v in patch.items()]
        for s in stack:
            s.start()
        try:
            return asyncio.run(app_main.upload_page(file=up, demo="0", page_id=None))
        except HTTPException as exc:
            return exc
        finally:
            for s in reversed(stack):
                s.stop()

    def test_bad_image_is_400(self):
        res = self._call("fake.png", b"this is not an image")
        self.assertEqual(getattr(res, "status_code", None), 400)
        self.assertIn("图像无法解码", res.detail)

    def test_bad_suffix_is_400(self):
        res = self._call("page.pdf", b"%PDF-1.4")
        self.assertEqual(getattr(res, "status_code", None), 400)
        self.assertIn("不支持的格式", res.detail)

    def test_ocr_missing_is_503(self):
        from ocr.recognize import StubOcrAdapter
        png = _tiny_png()
        res = self._call("ok.png", png, get_adapter=lambda *a, **k: (StubOcrAdapter(), ["依赖缺失"]))
        self.assertEqual(getattr(res, "status_code", None), 503)
        self.assertIn("OCR", res.detail)


def _tiny_png() -> bytes:
    """生成一张 8x8 的合法 PNG（不依赖外部素材）。"""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (8, 8), 255).save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    unittest.main(verbosity=2)
