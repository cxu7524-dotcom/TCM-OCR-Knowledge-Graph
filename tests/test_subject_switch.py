# -*- coding: utf-8 -*-
"""
tests/test_subject_switch.py —— 第 4 步：方剂主治跨段主体归属的回归用例

覆盖：
  1. page_35 两个真实失败样例的最小复现（杏仁→三化湯、川大黄→六君子湯）；
  2. 方名里含药名（藿香正氣湯）；
  3. 药材条目连续主治（甘草 / 主治 / 坚筋骨）；
  4. 同页另一主体的禁忌不被错归；
  5. 跨页续文 / 指代词无方名时不强行归属（未确认，不产出关系）；
  6. 组成与剂量的归属口径不变。
"""

from __future__ import annotations

import unittest

from app.schemas import Block, source_hash_of
from extract.rules import candidate_triples, candidate_triples_audited
from ocr.normalize import normalize_text

PID = "SUBJ-TEST-001"


def mk_blocks(lines):
    out = []
    for i, text in enumerate(lines):
        lid = f"L{i + 1:02d}"
        out.append(Block(page_id=PID, bbox=[10.0, 10.0 + i * 40, 60.0, 44.0 + i * 40],
                         line_id=lid, quote=text,
                         source_hash=source_hash_of(PID, lid, text),
                         text=text, text_norm=normalize_text(text), kind="paragraph",
                         reading_order=i + 1, block_ref=f"{PID}:{lid}"))
    return out


def rels(triples, relation=None):
    return [(t.head, t.relation, t.tail, t.line_id, t.subject_rule)
            for t in triples if relation is None or t.relation == relation]


class TestPage35MinimalRepro(unittest.TestCase):
    """药名出现在方名之前时，方名后的主治必须归给方剂。"""

    def test_sanhua_tang_owns_indication(self):
        blocks = mk_blocks([
            "共爲極細末加煉蜜二兩合棗肉杏仁泥爲丸",   # 煎服法，含药名「杏仁」
            "三化湯",
            "治中風不省人事痰涎壅盛胸膈不寬飲食不",
            "羌活枳實厚朴",
        ])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        heads = [g[0] for g in got]
        self.assertEqual(heads, ["三化汤"], got)
        self.assertEqual(got[0][3], "L03")
        self.assertEqual(got[0][4], "carry")      # 跨行承接方名
        self.assertNotIn("杏仁", heads)

    def test_liujunzi_tang_owns_indication(self):
        blocks = mk_blocks([
            "川大黄各三錢芒硝一錢半",                 # 组成行，含药名「川大黄」
            "六君子湯",
            "治面目萎黄或光白色氣少言微四肢無力飲",
        ])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        self.assertEqual([g[0] for g in got], ["六君子汤"], got)
        self.assertNotIn("川大黄", [g[0] for g in got])

    def test_formula_name_containing_herb(self):
        # 「藿香正氣湯」里的「藿香」不能抢走方剂的主治
        blocks = mk_blocks(["藿香正氣湯", "治感胃傷寒頭疼無汗惡寒發熱"])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        self.assertEqual([g[0] for g in got], ["藿香正气汤"], got)


class TestHerbEntries(unittest.TestCase):
    def test_consecutive_indications_stay_with_herb(self):
        blocks = mk_blocks(["甘草 味甘平 無毒", "主治五藏六府寒熱邪氣", "堅筋骨長肌肉",
                            "久服輕身延年溫中下氣"])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        self.assertEqual([(g[0], g[3]) for g in got], [("甘草", "L02")], got)

    def test_next_herb_takes_over(self):
        blocks = mk_blocks(["甘草 味甘平", "茯苓 味甘平 主胸脅逆氣", "久服輕身延年"])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        self.assertEqual([g[0] for g in got], ["茯苓"], got)


class TestContraindicationOwner(unittest.TestCase):
    def test_same_line_herb_wins(self):
        blocks = mk_blocks(["甘草 忌海藻 菘菜"])
        got = rels(candidate_triples(blocks, is_demo=False), "禁忌")
        self.assertEqual({g[0] for g in got}, {"甘草"})
        self.assertEqual({g[2] for g in got}, {"海藻", "菘菜"})

    def test_nearest_herb_not_previous_one(self):
        # 同页另一味药的禁忌：甘草条目在前，茯苓条目之后出现「忌…」，主体应为茯苓
        blocks = mk_blocks(["甘草 味甘平 無毒", "茯苓 味甘平 主胸脅逆氣", "忌海藻菘菜"])
        got = rels(candidate_triples(blocks, is_demo=False), "禁忌")
        self.assertEqual({g[0] for g in got}, {"茯苓"}, got)

    def test_contra_object_not_chosen_as_subject(self):
        # 「忌海藻 菘菜」行里，海藻 / 菘菜是忌口对象，主体应沿用上文的甘草
        blocks = mk_blocks(["甘草 味甘平 無毒", "忌海藻 菘菜"])
        got = rels(candidate_triples(blocks, is_demo=False), "禁忌")
        self.assertEqual({g[0] for g in got}, {"甘草"}, got)
        self.assertEqual({g[2] for g in got}, {"海藻", "菘菜"})


class TestUnconfirmedSubject(unittest.TestCase):
    """主体无法确认时：不产出关系，并在 audit 中留下可追踪记录。"""

    def test_anaphora_without_formula_is_skipped(self):
        blocks = mk_blocks(["人參 川附子 各二錢", "此方不但治真陽虚損亦治少陰證惡寒遍身"])
        triples, skipped = candidate_triples_audited(blocks, is_demo=False)
        self.assertEqual(rels(triples, "主治"), [])
        self.assertTrue(any("指代" in s["reason"] and s["line_id"] == "L02" for s in skipped),
                        skipped)

    def test_anaphora_with_formula_uses_formula(self):
        blocks = mk_blocks(["六味地黃湯", "此方專治啓精不足虚火炎上腰膝痿軟"])
        got = rels(candidate_triples(blocks, is_demo=False), "主治")
        self.assertEqual([(g[0], g[4]) for g in got], [("六味地黄汤", "formula_scope")], got)

    def test_page_start_indication_without_subject_is_skipped(self):
        blocks = mk_blocks(["治中風不省人事痰涎壅盛", "羌活 厚朴"])
        triples, skipped = candidate_triples_audited(blocks, is_demo=False)
        self.assertEqual(rels(triples, "主治"), [])
        self.assertTrue(any("跨页续文" in s["reason"] for s in skipped), skipped)


class TestCompositionAndDoseUnchanged(unittest.TestCase):
    def test_dose_goes_to_adjacent_herb_and_composition_to_formula(self):
        blocks = mk_blocks(["三化湯", "羌活 枳實 厚朴", "川大黄各三錢 芒硝一錢半"])
        triples = candidate_triples(blocks, is_demo=False)
        dose = [(h, t, l) for h, _r, t, l, _s in rels(triples, "剂量")]
        self.assertIn(("川大黄", "三钱", "L03"), dose)
        self.assertIn(("芒硝", "一钱半", "L03"), dose)
        comp = {(h, t) for h, _r, t, _l, _s in rels(triples, "组成")}
        self.assertEqual(comp, {("三化汤", "川大黄"), ("三化汤", "芒硝")})

    def test_subject_rule_is_recorded(self):
        blocks = mk_blocks(["甘草 味甘平 無毒", "主治五藏六府寒熱邪氣", "甘草 忌海藻"])
        triples = candidate_triples(blocks, is_demo=False)
        rules = {(t.relation, t.line_id): t.subject_rule for t in triples}
        self.assertEqual(rules[("主治", "L02")], "carry")
        self.assertEqual(rules[("禁忌", "L03")], "same_line")


if __name__ == "__main__":
    unittest.main()
