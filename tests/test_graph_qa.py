# -*- coding: utf-8 -*-
"""
tests/test_graph_qa.py —— 图谱关系问答回归（第 2 步）

覆盖四类硬约束：
  1. 关系方向：症状只认 tail，药名只认 head；
  2. 主体类型：药材 / 方剂 各归各的题型，互不串味；
  3. 出处校验：page_id / line_id / bbox / quote / source_hash 任一不符即丢弃；
  4. 主体归属：跨行必须有相邻的主体引文，中途出现别的药材即判归属不明。
另测题型识别、缺页 / 空页、无证据拒答与越界问题优先。

这些用例只构造内存对象，不读写数据库、不依赖 OCR 后端。
"""

from __future__ import annotations

import unittest
from typing import List, Tuple

from app.schemas import Block, Triple, source_hash_of
from graph.query import (query_contraindications, query_formulas_by_symptom,
                         query_herbs_by_symptom)
from qa.graph_qa import (INTENT_CONTRA, INTENT_FORMULAS, INTENT_HERBS,
                         detect_intent, graph_answer)

PID = "TESTPAGE-001"


def mk_blocks(lines: List[str], page_id: str = PID) -> List[Block]:
    out = []
    for i, text in enumerate(lines):
        lid = f"L{i + 1:02d}"
        out.append(Block(
            page_id=page_id, bbox=[10.0, 10.0 + i * 60, 500.0, 50.0 + i * 60],
            line_id=lid, quote=text, source_hash=source_hash_of(page_id, lid, text),
            text=text, text_norm=text, kind="paragraph", reading_order=i + 1,
            block_ref=f"{page_id}:{lid}",
        ))
    return out


def mk_triple(head: str, relation: str, tail: str, b: Block, **kw) -> Triple:
    t = Triple(
        page_id=b.page_id, bbox=list(b.bbox), line_id=b.line_id, quote=b.text,
        source_hash=b.source_hash, head=head, relation=relation, tail=tail,
        evidence_refs=[b.block_ref], is_demo=False,
    )
    for k, v in kw.items():
        setattr(t, k, v)
    return t


# 一张「药名行 + 主治续行 + 功效续行 + 禁忌行」的典型本草页
LINES = [
    "本草網目 卷之一",
    "甘草味甘平無毒",
    "主治五藏六府寒熱邪氣",
    "堅筋骨長肌肉",
    "忌海藻菘菜",
    "茯苓味甘平主胸脅逆氣",
]
# 一张「方名行 + 治…行」的方书页
FLINES = [
    "方書 卷之二",
    "三化湯",
    "治中風不省人事痰涎壅盛",
]


class TestRelationDirection(unittest.TestCase):
    """方向：症状必须在 tail；出现在 head 或邻近正文都不算。"""

    def setUp(self):
        self.blocks = mk_blocks(LINES)
        self.triples = [
            mk_triple("甘草", "主治", "五脏六腑寒热邪气", self.blocks[2]),
            mk_triple("甘草", "禁忌", "海藻", self.blocks[4]),
            mk_triple("茯苓", "主治", "胸胁逆气", self.blocks[5]),
        ]

    def test_symptom_matches_tail(self):
        res = query_herbs_by_symptom("五脏六腑寒热邪气", self.triples, page_id=PID)
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["results"][0]["head"], "甘草")

    def test_symptom_only_in_head_is_not_a_hit(self):
        # 「甘草」出现在 head，不能因为问题里提到药名就当成症状命中
        res = query_herbs_by_symptom("甘草", self.triples, page_id=PID)
        self.assertEqual(res["count"], 0)

    def test_contraindication_tail_is_not_a_symptom(self):
        # 禁忌对象「海藻」不是主治病证，不能被「按症状查药材」召回
        res = query_herbs_by_symptom("海藻", self.triples, page_id=PID)
        self.assertEqual(res["count"], 0)

    def test_herb_subject_not_in_formula_query(self):
        res = query_formulas_by_symptom("五脏六腑寒热邪气", self.triples, page_id=PID)
        self.assertEqual(res["count"], 0)

    def test_formula_subject_not_in_herb_query(self):
        blocks = mk_blocks(FLINES)
        triples = [mk_triple("三化汤", "主治", "中风不省人事痰涎壅盛", blocks[2])]
        self.assertEqual(query_herbs_by_symptom("中风不省人事", triples, page_id=PID)["count"], 0)
        self.assertEqual(query_formulas_by_symptom("中风不省人事", triples, page_id=PID)["count"], 1)

    def test_contraindication_head_must_match_exactly(self):
        # 问「茯苓」的禁忌，不能把「甘草 忌海藻」错归给它
        res = query_contraindications("茯苓", self.triples, page_id=PID)
        self.assertEqual(res["count"], 0)


class TestPageScope(unittest.TestCase):
    def test_other_page_triples_excluded(self):
        blocks = mk_blocks(FLINES)
        other = mk_blocks(LINES, page_id="OTHER-PAGE")      # 同一批文字，不同页
        triples = [mk_triple("三化汤", "主治", "中风不省人事", other[2])]
        res = query_formulas_by_symptom("中风不省人事", triples, page_id=PID)
        self.assertEqual(res["count"], 0)
        self.assertEqual(query_formulas_by_symptom(
            "中风不省人事", triples, page_id="OTHER-PAGE")["count"], 1)


class TestIntentDetection(unittest.TestCase):
    def test_three_positive_intents(self):
        self.assertEqual(detect_intent("五脏六腑寒热邪气可用哪些药材？"), (INTENT_HERBS, "五脏六腑寒热邪气"))
        self.assertEqual(detect_intent("中风不省人事用什么方？"), (INTENT_FORMULAS, "中风不省人事"))
        self.assertEqual(detect_intent("甘草有什么禁忌？"), (INTENT_CONTRA, "甘草"))

    def test_other_question_types_fall_back(self):
        for q in ("桂枝汤由哪些药材组成？", "甘草的性味是什么？", "甘草主治什么？",
                  "这个方子能治好我的失眠吗？该吃多少？"):
            self.assertEqual(detect_intent(q), ("", ""), q)

    def test_personal_question_not_routed_to_graph(self):
        self.assertEqual(detect_intent("我头痛用什么药"), ("", ""))


class TestProvenanceCheck(unittest.TestCase):
    """五个溯源字段任一不符，该关系一律丢弃并转拒答。"""

    def setUp(self):
        self.blocks = mk_blocks(FLINES)
        self.good = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])

    def _ask(self, triple: Triple):
        return graph_answer("中风不省人事用什么方？", self.blocks, [triple],
                            page_id=PID, page_exists=True, is_demo=False)

    def test_good_triple_answered(self):
        ans = self._ask(self.good)
        self.assertFalse(ans.refused)
        self.assertEqual(len(ans.objects), 1)
        self.assertEqual(ans.objects[0]["head"], "三化汤")

    def test_quote_mismatch_refused(self):
        bad = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])
        bad.quote = "治中風不省人事"          # 少了后半句，与原文块不一致
        self.assertTrue(self._ask(bad).refused)

    def test_line_id_not_on_page_refused(self):
        bad = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])
        bad.line_id = "L99"
        self.assertTrue(self._ask(bad).refused)

    def test_bbox_mismatch_refused(self):
        bad = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])
        bad.bbox = [0.0, 0.0, 1.0, 1.0]
        self.assertTrue(self._ask(bad).refused)

    def test_page_id_mismatch_refused(self):
        bad = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])
        bad.page_id = "OTHER-PAGE"
        self.assertTrue(self._ask(bad).refused)

    def test_source_hash_mismatch_refused(self):
        bad = mk_triple("三化汤", "主治", "中风不省人事", self.blocks[2])
        bad.source_hash = "deadbeefdeadbeef"
        self.assertTrue(self._ask(bad).refused)


class TestSubjectAttribution(unittest.TestCase):
    """跨行关系必须能确认主体；归错主体或超出回溯距离一律拒答。"""

    def test_carry_subject_adds_subject_evidence(self):
        blocks = mk_blocks(FLINES)
        triples = [mk_triple("三化汤", "主治", "中风不省人事痰涎壅盛", blocks[2])]
        ans = graph_answer("中风不省人事用什么方？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertFalse(ans.refused)
        self.assertEqual(ans.objects[0]["support"], "carry")
        self.assertEqual(ans.objects[0]["subject_line"], "L02")
        roles = sorted(e.role for e in ans.evidences)
        self.assertEqual(roles, ["relation", "subject"])
        self.assertIn("本页原文记载", ans.text)
        self.assertIn("主体引文", ans.text)

    def test_intervening_other_subject_refused(self):
        # 关系行上一条是另一味药：主体归属不明，不能答
        blocks = mk_blocks(["本草網目", "甘草味甘平", "當歸味甘溫", "主治五藏六府寒熱邪氣"])
        triples = [mk_triple("甘草", "主治", "五脏六腑寒热邪气", blocks[3])]
        ans = graph_answer("五脏六腑寒热邪气可用哪些药材？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.objects, [])

    def test_subject_beyond_window_refused(self):
        lines = ["本草網目", "甘草味甘平"] + [f"續行{i}不可辨" for i in range(8)] + ["主治五藏六府寒熱邪氣"]
        blocks = mk_blocks(lines)
        triples = [mk_triple("甘草", "主治", "五脏六腑寒热邪气", blocks[-1])]
        ans = graph_answer("五脏六腑寒热邪气可用哪些药材？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)

    def test_subject_not_on_page_refused(self):
        blocks = mk_blocks(LINES)
        triples = [mk_triple("當歸", "主治", "五脏六腑寒热邪气", blocks[2])]
        ans = graph_answer("五脏六腑寒热邪气可用哪些药材？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)


class TestRefusalPaths(unittest.TestCase):
    def test_no_evidence_refused(self):
        blocks = mk_blocks(LINES)
        ans = graph_answer("黄疸可用哪些药材？", blocks, [],
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.evidences, [])
        self.assertIn("本页原文没有能支持该关系的证据", ans.text)

    def test_missing_page_refused(self):
        ans = graph_answer("甘草有什么禁忌？", [], [],
                           page_id="NO-SUCH-PAGE", page_exists=False, is_demo=False)
        self.assertTrue(ans.refused)
        self.assertIn("页面不存在", ans.refuse_reason)

    def test_empty_page_refused(self):
        ans = graph_answer("甘草有什么禁忌？", [], [],
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)
        self.assertIn("空页", ans.refuse_reason)

    def test_other_page_hits_never_cited(self):
        blocks = mk_blocks(LINES)
        ans = graph_answer("甘草有什么禁忌？", blocks, [],
                           page_id=PID, page_exists=True, is_demo=False,
                           other_page_hits=3)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.evidences, [])
        self.assertTrue(any("不引用" in t for t in ans.trace))

    def test_out_of_scope_question_keeps_refusal(self):
        blocks = mk_blocks(LINES)
        triples = [mk_triple("甘草", "主治", "五脏六腑寒热邪气", blocks[2])]
        ans = graph_answer("这个方子能治好我的失眠吗？该吃多少？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertIsNone(ans)      # 非三类题型：交回原文本检索 / 原拒答流程


class TestNonGraphQuestion(unittest.TestCase):
    def test_returns_none_for_plain_question(self):
        blocks = mk_blocks(LINES)
        self.assertIsNone(graph_answer("甘草的性味是什么？", blocks, [], page_id=PID))


# ------------------------------------------------------------------
# 第 2 步补验：禁忌主体必须同名，不允许子串重合
# ------------------------------------------------------------------

CONTRA_LINES = [
    "甘草 味甘平 無毒",
    "忌海藻 菘菜",
]


def contra_page():
    """只有「甘草 忌海藻 / 菘菜」两行证据的构造页。"""
    blocks = mk_blocks(CONTRA_LINES)
    triples = [
        mk_triple("甘草", "禁忌", "海藻", blocks[1]),
        mk_triple("甘草", "禁忌", "菘菜", blocks[1]),
    ]
    return blocks, triples


class TestContraindicationSubjectExact(unittest.TestCase):
    """药名 / 炮制品名必须完全同名；子串重合不算同一主体。"""

    def setUp(self):
        self.blocks, self.triples = contra_page()

    def test_same_name_hits(self):
        res = query_contraindications("甘草", self.triples, page_id=PID)
        self.assertEqual(res["count"], 2)

    def test_traditional_and_variant_forms_still_match(self):
        # 同一名称的繁简 / 已定义异体字归一化仍可命中：川大黃 → 川大黄
        blocks = mk_blocks(["川大黃 味苦寒", "忌菘菜"])
        triples = [mk_triple("川大黃", "禁忌", "菘菜", blocks[1])]
        self.assertEqual(query_contraindications("川大黄", triples, page_id=PID)["count"], 1)

    def test_processed_variants_do_not_hit(self):
        for term in ("生甘草", "炙甘草", "甘草節", "甘草片", "甘草梢"):
            self.assertEqual(
                query_contraindications(term, self.triples, page_id=PID)["count"], 0, term)

    def test_answer_for_exact_name(self):
        ans = graph_answer("甘草有什么禁忌？", self.blocks, self.triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertFalse(ans.refused)
        self.assertEqual(sorted(o["tail"] for o in ans.objects), ["海藻", "菘菜"])
        self.assertTrue(ans.evidences)

    def test_processed_variants_refused_with_empty_evidence(self):
        for q in ("生甘草有什么禁忌？", "炙甘草有什么禁忌？"):
            ans = graph_answer(q, self.blocks, self.triples,
                               page_id=PID, page_exists=True, is_demo=False)
            self.assertTrue(ans.refused, q)
            self.assertEqual(ans.evidences, [], q)
            self.assertEqual(ans.objects, [], q)
            self.assertIn("本页原文没有能支持该关系的证据", ans.text, q)

    def test_variant_head_not_reused_for_base_name(self):
        # 反向：页面只记「炙甘草」的禁忌时，问「甘草」不能把它当成甘草的禁忌
        blocks = mk_blocks(["炙甘草 味甘 無毒", "忌海藻"])
        triples = [mk_triple("炙甘草", "禁忌", "海藻", blocks[1])]
        self.assertEqual(query_contraindications("甘草", triples, page_id=PID)["count"], 0)
        ans = graph_answer("甘草有什么禁忌？", blocks, triples,
                           page_id=PID, page_exists=True, is_demo=False)
        self.assertTrue(ans.refused)
        self.assertEqual(ans.evidences, [])


class TestTraditionalQuestion(unittest.TestCase):
    """繁体提问与简体同等识别（繁简 / 已定义异体字归一化）。"""

    def test_traditional_contra(self):
        self.assertEqual(detect_intent("甘草有什麼禁忌？"), (INTENT_CONTRA, "甘草"))
        self.assertEqual(detect_intent("甘草忌什麼"), (INTENT_CONTRA, "甘草"))

    def test_traditional_formula(self):
        self.assertEqual(detect_intent("中風不省人事用什麼方？"),
                         (INTENT_FORMULAS, "中风不省人事"))

    def test_traditional_herb(self):
        # 繁体提问：归一化后与简体页文本可比（藏府→脏腑、邪氣→邪气）
        self.assertEqual(detect_intent("五藏六府寒熱邪氣可用哪些藥材？"),
                         (INTENT_HERBS, "五脏六腑寒热邪气"))


if __name__ == "__main__":
    unittest.main()
