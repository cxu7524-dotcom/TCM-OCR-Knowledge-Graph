# -*- coding: utf-8 -*-
"""回答模式（快速 / 标准 / 专家）：只改检索广度与生成方式，不改输出边界。

口径：
- 快速：不调模型，只摘原文（1 秒级）；
- 标准：现行链路（检索 + 本机模型）；
- 专家：检索范围放宽、条数更多、提示词要求分条对照异文；
- 三档都走同一套越界拒答规则，档位原样写进 retrieval 供前端显示。
"""
import unittest
from unittest.mock import patch

from qa import prompt as prompt_mod
from qa import rag


class FakePassage:
    book = "伤寒论"
    page_id = "p1"
    line_ids = ["L1"]
    text = "麻黄三两，桂枝二两，甘草一两"
    blocks = []

    def cite_label(self):
        return "p1 · L1"


HITS = [(FakePassage(), 0.61)]
META = {"index": {"vectors": 10}, "candidates": 3, "hits": 1, "top_score": 0.61,
        "has_evidence": True, "floor": 0.49, "mode": "semantic-rag"}


def fake_retrieve_factory(seen, hits=None, meta=None):
    def fake(query, top_k=None, page_id=None, scope="library", relative_keep=None):
        seen.update({"top_k": top_k, "relative_keep": relative_keep})
        return list(HITS if hits is None else hits), dict(META if meta is None else meta)
    return fake


class ModeTest(unittest.TestCase):
    def run_mode(self, mode, reply="甘草调和诸药 [1]。"):
        seen = {}
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen)), \
                patch.object(rag.llm_client, "chat_messages", return_value=reply) as chat:
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode=mode)
        return ans, chat, seen

    def test_fast_never_calls_model(self):
        ans, chat, seen = self.run_mode("fast")
        self.assertFalse(chat.called, "快速模式不该调用本机模型")
        self.assertEqual(ans.backend, "rag-fast")
        self.assertIn("快速模式", ans.text)
        self.assertFalse(ans.refused)
        self.assertTrue(ans.sources)

    def test_fast_narrows_retrieval(self):
        ans, chat, seen = self.run_mode("fast")
        self.assertEqual(seen["top_k"], 4)
        self.assertEqual(ans.retrieval["qa_mode"], "fast")
        self.assertLessEqual(len(ans.sources), 3)

    def test_standard_keeps_current_behaviour(self):
        ans, chat, seen = self.run_mode("standard")
        self.assertIsNone(seen["top_k"])              # 沿用配置里的 top_k
        self.assertIsNone(seen["relative_keep"])      # 沿用配置里的 relative_keep
        self.assertTrue(chat.called)
        self.assertEqual(ans.backend, "rag")
        self.assertEqual(ans.retrieval["qa_mode"], "standard")
        self.assertNotIn("异文", chat.call_args[0][0][0]["content"])

    def test_expert_widens_and_switches_prompt(self):
        ans, chat, seen = self.run_mode("expert")
        self.assertEqual(seen["top_k"], 12)
        self.assertAlmostEqual(seen["relative_keep"], 0.60)
        first = chat.call_args_list[0]                 # 第一次是作答，第二次是自查
        self.assertIn("异文", first[0][0][0]["content"])     # 专家提示词要求对照异文
        self.assertEqual(first[1].get("temperature"), 0.1)
        self.assertEqual(ans.retrieval["qa_mode"], "expert")

    def test_unknown_mode_falls_back_to_standard(self):
        ans, chat, seen = self.run_mode("随你")
        self.assertEqual(ans.retrieval["qa_mode"], "standard")
        self.assertTrue(chat.called)

    def test_fast_without_evidence_refuses_instead_of_guessing(self):
        seen = {}
        empty = dict(META, hits=0, has_evidence=False, note="没有命中")
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen, hits=[], meta=empty)), \
                patch.object(rag.llm_client, "chat_messages") as chat:
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode="fast")
        self.assertFalse(chat.called)
        self.assertTrue(ans.refused)
        self.assertIn("快速模式", ans.refuse_reason)

    def test_out_of_scope_still_refused_in_every_mode(self):
        for mode in ("fast", "standard", "expert"):
            with patch.object(rag.llm_client, "chat_messages") as chat:
                ans = rag.rag_answer("我这个咳嗽该吃多少克麻黄", mode=mode)
            self.assertTrue(ans.refused, mode)
            self.assertFalse(chat.called, mode)

    def test_prompt_style_switch(self):
        std = prompt_mod.build_messages("问", HITS)
        exp = prompt_mod.build_messages("问", HITS, style="expert")
        self.assertNotIn("异文", std[0]["content"])
        self.assertIn("异文", exp[0]["content"])
        self.assertEqual(std[-1]["content"], exp[-1]["content"])   # 原文与问题部分一致

    def test_mode_label(self):
        self.assertEqual(rag.mode_label("fast"), "快速")
        self.assertEqual(rag.mode_label("expert"), "专家")
        self.assertEqual(rag.mode_label("nope"), "标准")
        self.assertEqual(rag.mode_label(""), "标准")


ANSWER = ("甘草调和诸药 [1]。甘草还能治百病。"
          "（据中医常识）甘草味甘性平，归心肺脾胃经 [1]。")


class SelfCheckTest(unittest.TestCase):
    """专家模式第二遍：答完回查，原文不支持的句子就地标〔未核〕。"""

    def run_expert(self, verdict, answer=ANSWER):
        seen = {}
        replies = [answer, verdict]
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen)), \
                patch.object(rag.llm_client, "chat_messages", side_effect=replies) as chat:
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode="expert")
        return ans, chat

    def test_expert_calls_model_twice(self):
        ans, chat = self.run_expert("全部有据")
        self.assertEqual(chat.call_count, 2)              # 一遍作答 + 一遍自查
        self.assertIn("待审答案", chat.call_args[0][0][-1]["content"])

    def test_flagged_sentence_gets_marker(self):
        ans, chat = self.run_expert("无据|甘草还能治百病。")
        self.assertIn("〔未核〕", ans.text)
        self.assertIn("甘草还能治百病〔未核〕。", ans.text)    # 标注在句号之前
        sc = ans.retrieval["self_check"]
        self.assertTrue(sc["ran"])
        self.assertEqual(sc["flagged"], 1)
        self.assertEqual(sc["marked"], 1)
        self.assertIn("自查：", ans.text)

    def test_common_sense_sentences_are_not_re_marked(self):
        # 模型把「（据中医常识）」那句也指认为无据，不该重复标注
        ans, chat = self.run_expert("无据|甘草味甘性平，归心肺脾胃经")
        sc = ans.retrieval["self_check"]
        self.assertEqual(sc["marked"], 0)
        self.assertEqual(sc["flagged_common"], 1)
        self.assertNotIn("〔未核〕", ans.text)
        self.assertIn("常识补充", ans.text)

    def test_clean_verdict_marks_nothing(self):
        ans, chat = self.run_expert("全部有据")
        sc = ans.retrieval["self_check"]
        self.assertTrue(sc["ok"])
        self.assertEqual(sc["marked"], 0)
        self.assertNotIn("〔未核〕", ans.text)

    def test_garbage_verdict_is_harmless(self):
        ans, chat = self.run_expert("我觉得还行吧")
        sc = ans.retrieval["self_check"]
        self.assertEqual(sc["flagged"], 0)
        self.assertNotIn("〔未核〕", ans.text)

    def test_model_silent_keeps_answer(self):
        ans, chat = self.run_expert("")
        sc = ans.retrieval["self_check"]
        self.assertFalse(sc["ran"])
        self.assertIn("模型没有回应", sc["note"])
        self.assertTrue(ans.text)

    def test_standard_mode_does_not_self_check(self):
        seen = {}
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen)), \
                patch.object(rag.llm_client, "chat_messages", return_value=ANSWER) as chat:
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode="standard")
        self.assertEqual(chat.call_count, 1)
        self.assertNotIn("self_check", ans.retrieval)
        self.assertNotIn("〔未核〕", ans.text)

    def test_marker_goes_after_citation_number(self):
        seen = {}
        replies = ["结论 [1]。另一句没有出处。", "无据|另一句没有出处。"]
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen)), \
                patch.object(rag.llm_client, "chat_messages", side_effect=replies):
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode="expert")
        self.assertIn("另一句没有出处〔未核〕。", ans.text)
        self.assertIn("结论 [1]。", ans.text)

    def test_marker_lands_after_trailing_citation(self):
        seen = {}
        replies = ["结论 [1]。这句有出处 [1]", "无据|这句有出处 [1]"]
        with patch.object(rag, "retrieve", fake_retrieve_factory(seen)), \
                patch.object(rag.llm_client, "chat_messages", side_effect=replies):
            ans = rag.rag_answer("甘草在方剂里起什么作用", mode="expert")
        self.assertIn("这句有出处 [1]〔未核〕", ans.text)

    def test_paraphrased_flag_is_reported_not_marked(self):
        ans, chat = self.run_expert("无据|完全不在答案里的另一句话")
        sc = ans.retrieval["self_check"]
        self.assertEqual(sc["marked"], 0)
        self.assertNotIn("〔未核〕", ans.text)
        self.assertIn("未能定位", sc["note"])


if __name__ == "__main__":
    unittest.main()
