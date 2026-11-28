"""第二阶段静态数据：丹方 p12 补标、待核页裁定、复核记录、负例与栏线样例、分区。

所有坐标均为原图像素、左上角原点、半开区间 [x1,y1,x2,y2)。
框由 tools/dual_track_stage2_measure.py 的列窗测量给出，再按 6 倍叠框目验核对；
不确定的部分写进 certainty / candidates，不假装精确。
"""
from __future__ import annotations

from typing import Any, Dict, List

RELEASE = 'stage2-v3'
REVIEWER = 'single_reviewer_ai_visual（单人在 6 倍叠框图上目验，无第二人独立复核）'

# ---------------------------------------------------------------- 丹方彚编 p12

DF_BOOK = '丹方彚编'
DF_PAGE = 12
DF_REGION = 'right_half'
DF_GUTTER: Dict[str, int] = {
    'DF12-01': 960, 'DF12-02': 909, 'DF12-03': 860, 'DF12-04': 810,
    'DF12-05': 760, 'DF12-06': 711, 'DF12-07': 664, 'DF12-08': 613,
    'DF12-09': 759, 'DF12-10': 822, 'DF12-11': 882,
}

# 列窗测量结果（tools/dual_track_stage2_measure.py + 目验），已按半开区间记录。
#
# DF12-v2 修正要点（2026-11-24）：
#   上一版把「列窗 + 中缝」的整段下半栏当成两条长轨，导致上方组的轨框一直延伸到 y=818，
#   把下方独立条目（花藍石/硃砂/陽起石）整块吞进去。本版按「上方组在本栏下一个独立标题之前结束」
#   重新截断；长轨栏（滴乳石 934-986、翠膽礬 885-934、木香 589-637）保持到版框，
#   短轨栏（雲母 836-885、琥珀 786-836、蛤蚧 735-786）截在下方标题上沿之前。
DF_GROUPS: List[Dict[str, Any]] = [
    {'region': 'DF12-01', 'main_text': '滴乳石', 'main_bbox': [934, 204, 986, 303],
     'left_bbox': [934, 307, 959, 794], 'right_bbox': [962, 307, 986, 818],
     'right_text': '白者要如雪冬淋〔?〕長五六寸明亮中空色', 'left_text': '必真如乳〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '三字药名终结于 y=350，两轨自 351 起；本栏下方无新标题，两轨一直延续到版框（左 794 / 右 818）'},
    {'region': 'DF12-02', 'main_text': '翠膽礬', 'main_bbox': [885, 204, 931, 305],
     'left_bbox': [885, 309, 908, 539], 'right_bbox': [911, 309, 934, 818],
     'right_text': '成〔?〕真居可多色以青碧如琉璃擊碎縱橫解', 'left_text': '成〔?〕如〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'ambiguous_lane_extent',
     'note': '左轨在 y≈540 之后被下方「陽起石」条目的药名压住（陽起石 x 855-905 与左轨 x 885-908 在横向上重叠），'
             '无法判断左轨是结束还是被遮断，故左轨截到 539；右轨 x 911-934 不与之冲突，保持到 818',
     'candidates': [
         {'id': 'a', 'reading': '左轨止于 y=539（被下方条目打断，未再续）', 'basis': '叠框显示 540 以下左轨位置被「陽起石」占用'},
         {'id': 'b', 'reading': '左轨在陽起石下方继续，左轨应为两段', 'basis': '双行夹注通常连续；但本页下部栏位整体错位，无法目验确认'},
     ]},
    {'region': 'DF12-03', 'main_text': '雲母', 'main_bbox': [836, 204, 875, 275],
     'left_bbox': [836, 278, 859, 536], 'right_bbox': [862, 278, 885, 536],
     'right_text': '要〔?〕白〔?〕', 'left_text': '折〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '两轨止于 536，下一独立标题「陽起石」自 y≈540 起；两轨高度一致'},
    {'region': 'DF12-04', 'main_text': '琥珀', 'main_bbox': [786, 204, 825, 272],
     'left_bbox': [786, 277, 809, 539], 'right_bbox': [812, 277, 836, 539],
     'right_text': '要如血色熟透于〔?〕', 'left_text': '揩〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '两轨止于 539，下一独立标题「硃砂」自 y≈548 起'},
    {'region': 'DF12-05', 'main_text': '蛤蚧', 'main_bbox': [735, 204, 786, 278],
     'left_bbox': [735, 282, 759, 536], 'right_bbox': [762, 282, 786, 531],
     'text_status': 'partial_unverified',
     'right_text': '以四脚〔?〕', 'left_text': '有花斑起發光亮者〔?〕',
     'certainty': 'medium',
     'note': '两轨止于 531-536，下一独立标题「花藍石」自 y≈540 起'},
    {'region': 'DF12-06', 'main_text': '輕粉', 'main_bbox': [686, 204, 735, 275],
     'left_bbox': [686, 280, 710, 555], 'right_bbox': [713, 280, 735, 555],
     'right_text': '明如雪片〔?〕', 'left_text': '真色暗〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '两轨止于 555；本栏下方为独立条目「蟾酥」（药名自 y≈560 起），未纳入本组'},
    {'region': 'DF12-07', 'main_text': '沉香', 'main_bbox': [637, 204, 686, 275],
     'left_bbox': [637, 278, 661, 557], 'right_bbox': [666, 278, 686, 559],
     'right_text': '油潤切〔?〕', 'left_text': '黑色如牛角〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '两轨止于 557/559；本栏下方为独立条目「母丁香」（药名自 y≈560 起，下方另有两轨），未纳入本组'},
    {'region': 'DF12-08', 'main_text': '木香', 'main_bbox': [589, 204, 629, 275],
     'left_bbox': [589, 280, 612, 815], 'right_bbox': [615, 280, 637, 813],
     'right_text': '多有〔?〕用青木香〔?〕', 'left_text': '苦口粘牙者真〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'medium',
     'note': '两轨延续到版框（左 815 / 右 813）；该栏左侧 x 548-580 为单轨列，另作负例 NEG-DF-SINGLE-01'},
    {'region': 'DF12-09', 'main_text': '花藍石', 'main_bbox': [745, 540, 795, 655],
     'left_bbox': [745, 660, 758, 818], 'right_bbox': [762, 660, 795, 818],
     'right_text': '打開有花白斑', 'left_text': '燃者真無者假',
     'text_status': 'read_from_overlay', 'certainty': 'medium',
     'note': '下方独立条目：药名 540-668，两轨 669-818；与上方「蛤蚧」组已完全分离'},
    {'region': 'DF12-10', 'main_text': '硃砂', 'main_bbox': [800, 548, 850, 625],
     'left_bbox': [800, 630, 821, 809], 'right_bbox': [824, 630, 850, 818],
     'right_text': '有以〔?〕宜細研', 'left_text': '入〔?〕香丸',
     'text_status': 'partial_unverified', 'certainty': 'low',
     'note': '下方独立条目：药名 548-645、两轨 646 起；两字药名与首列小字纵向紧邻，边界 ±5px'},
    {'region': 'DF12-11', 'main_text': '陽起石', 'main_bbox': [855, 540, 905, 650],
     'left_bbox': [855, 655, 881, 816], 'right_bbox': [884, 655, 905, 818],
     'right_text': '鷄毛者真如鷹〔?〕', 'left_text': '要明亮如鷄〔?〕',
     'text_status': 'partial_unverified', 'certainty': 'low',
     'note': '下方独立条目：药名 540-672、两轨 673 起；该条目与左上「翠膽礬」左轨横向重叠，'
             '是 DF12-02 只能截到 539 的原因'},
]


# 逐组字形归属核对（6 倍/8 倍叠框目验，证据图见 evidence/fix_df12/_glyph_T*.png 与各组 before_after）
# 记录的是「这个边界上的字形属于大字标题还是小字轨道」「末笔是否被切」，不要求逐字识别准确。
GLYPH_REVIEW: Dict[str, Dict[str, Any]] = {
    'DF12-01': {'main_last_glyph': {'text': '石', 'bottom_y': 303, 'glyph_complete': True},
                'left_first_glyph': {'text': '必', 'top_y': 307, 'glyph_complete': True},
                'right_first_glyph': {'text': '白', 'top_y': 307, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': '三字药名 204-303；两轨自 307 起，首字「必／白」完整；两轨均到版框前收尾'},
    'DF12-02': {'main_last_glyph': {'text': '礬', 'bottom_y': 305, 'glyph_complete': True},
                'left_first_glyph': {'text': '皆', 'top_y': 309, 'glyph_complete': True},
                'right_first_glyph': {'text': '成', 'top_y': 309, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked_with_ambiguity',
                'review_notes': '左轨止于 539（下方被「陽起石」压住），末字完整但延续性不明，见 candidates'},
    'DF12-03': {'main_last_glyph': {'text': '母', 'bottom_y': 275, 'glyph_complete': True},
                'left_first_glyph': {'text': '折', 'top_y': 278, 'glyph_complete': True},
                'right_first_glyph': {'text': '要', 'top_y': 278, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 29},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 29},
                'review_status': 'checked',
                'review_notes': 'v2 把「折／要」这一首行小字误算进主字框（旧 main 到 305）；'
                                '本版主字止于「母」底 275、两轨自 278 起。轨尾止于「陽起石」标题前，'
                                '末字完整，框内含约 29px 尾部留白（目验）'},
    'DF12-04': {'main_last_glyph': {'text': '珀', 'bottom_y': 272, 'glyph_complete': True},
                'left_first_glyph': {'text': '揩', 'top_y': 277, 'glyph_complete': True},
                'right_first_glyph': {'text': '要', 'top_y': 277, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 33},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 33},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 305、把「揩／要」吞入；本版主字止于 272、两轨自 277 起；'
                                '轨尾止于「硃砂」标题前，含约 33px 留白'},
    'DF12-05': {'main_last_glyph': {'text': '蚧', 'bottom_y': 278, 'glyph_complete': True},
                'left_first_glyph': {'text': '有', 'top_y': 282, 'glyph_complete': True},
                'right_first_glyph': {'text': '以', 'top_y': 282, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 8},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 8},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 305、把「有／以」吞入；本版主字止于 278、两轨自 282 起；'
                                '轨尾止于「花藍石」标题前，含约 8px 留白'},
    'DF12-06': {'main_last_glyph': {'text': '粉', 'bottom_y': 275, 'glyph_complete': True},
                'left_first_glyph': {'text': '真', 'top_y': 280, 'glyph_complete': True},
                'right_first_glyph': {'text': '明', 'top_y': 280, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 3},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 3},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 305、把「真／明」吞入；本版止于 275、两轨自 280 起；'
                                '轨尾止于「蟾酥」标题前'},
    'DF12-07': {'main_last_glyph': {'text': '香', 'bottom_y': 275, 'glyph_complete': True},
                'left_first_glyph': {'text': '黑', 'top_y': 278, 'glyph_complete': True},
                'right_first_glyph': {'text': '油', 'top_y': 278, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 305、把「黑／油」吞入；本版止于 275、两轨自 278 起；'
                                '轨尾止于「母丁香」标题前'},
    'DF12-08': {'main_last_glyph': {'text': '香', 'bottom_y': 275, 'glyph_complete': True},
                'left_first_glyph': {'text': '苦', 'top_y': 280, 'glyph_complete': True},
                'right_first_glyph': {'text': '多', 'top_y': 280, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 305、把「苦／多」吞入；本版止于 275、两轨自 280 起；'
                                '本栏下方无新标题，两轨延续到版框'},
    'DF12-09': {'main_last_glyph': {'text': '石', 'bottom_y': 655, 'glyph_complete': True},
                'left_first_glyph': {'text': '燃', 'top_y': 660, 'glyph_complete': True},
                'right_first_glyph': {'text': '打', 'top_y': 660, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 668、把「燃／打」上沿吞入；本版止于 655、两轨自 660 起'},
    'DF12-10': {'main_last_glyph': {'text': '砂', 'bottom_y': 625, 'glyph_complete': True},
                'left_first_glyph': {'text': '灰〔?〕', 'top_y': 630, 'glyph_complete': True},
                'right_first_glyph': {'text': '有', 'top_y': 630, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 645、把「灰／有」吞入；本版止于 625、两轨自 630 起；'
                                '左列首字字形存疑（疑「灰」），归属已确认为小字轨道'},
    'DF12-11': {'main_last_glyph': {'text': '石', 'bottom_y': 650, 'glyph_complete': True},
                'left_first_glyph': {'text': '要', 'top_y': 655, 'glyph_complete': True},
                'right_first_glyph': {'text': '鷄〔?〕', 'top_y': 655, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked',
                'review_notes': 'v2 主字框到 672、把「要／鷄」吞入；本版止于 650、两轨自 655 起；'
                                '右列首字字形存疑，归属已确认为小字轨道'},
}

# 上一版（v1）DF12 组框，仅用于生成「修正前后」对照图与修改映射，不参与评测
DF_GROUPS_V1: Dict[str, Dict[str, List[int]]] = {
    'DF12-01': {'main': [934, 208, 986, 350], 'left': [934, 351, 959, 794], 'right': [962, 351, 986, 818]},
    'DF12-02': {'main': [885, 208, 934, 350], 'left': [885, 351, 908, 818], 'right': [911, 351, 934, 818]},
    'DF12-03': {'main': [836, 208, 884, 305], 'left': [836, 306, 856, 801], 'right': [859, 306, 885, 818]},
    'DF12-04': {'main': [786, 208, 834, 305], 'left': [786, 306, 809, 791], 'right': [812, 306, 836, 818]},
    'DF12-05': {'main': [735, 208, 786, 305], 'left': [735, 306, 759, 818], 'right': [762, 306, 786, 818]},
    'DF12-06': {'main': [686, 208, 735, 305], 'left': [686, 306, 710, 818], 'right': [713, 306, 735, 818]},
    'DF12-07': {'main': [637, 208, 686, 305], 'left': [637, 306, 663, 818], 'right': [666, 306, 686, 818]},
    'DF12-08': {'main': [589, 208, 634, 305], 'left': [589, 307, 612, 815], 'right': [615, 306, 637, 813]},
    'DF12-09': {'main': [745, 540, 795, 668], 'left': [745, 668, 758, 818], 'right': [761, 668, 795, 818]},
    'DF12-10': {'main': [800, 552, 848, 645], 'left': [800, 646, 821, 809], 'right': [824, 646, 848, 818]},
    'DF12-11': {'main': [855, 540, 905, 672], 'left': [855, 672, 881, 816], 'right': [884, 672, 905, 818]},
}

# 丹方 p12 右半页的检查结论：8 栏顶部组已逐栏核对，栏内下部条目只标注了 3 组
DF_PAGE_COVERAGE = {
    'status': 'partial_positions',
    'checked_columns': ['x 589-637 木香栏', 'x 637-686 沉香栏', 'x 686-735 輕粉栏',
                        'x 735-786 蛤蚧栏', 'x 786-836 琥珀栏', 'x 836-885 雲母栏',
                        'x 885-934 翠膽礬栏', 'x 934-986 滴乳石栏'],
    'annotated_groups': 11,
    'not_exhaustive_reason': '8 栏顶部各一组已核对并标注；下部另有独立标题条目，'
                             '本阶段确认并标注 3 组（花藍石、硃砂、陽起石）；'
                             '另已目验到但未标注的下部条目至少还有「母丁香」（沉香栏）、「蟾酥」（輕粉栏），'
                             '其余栏下部未逐一清点，故不声明 exhaustive_page',
    'whole_page_recall': 'not_computable_current_version（下部未穷尽，整页召回不可计算）',
    'ambiguous_is_record_level': '本页的歧义挂在记录上（DF12-02 的左轨延续），'
                                 '不设矩形 ambiguous 区：下部栏位与上部错位，任何矩形都会与长轨正例重叠，'
                                 '强行设置会把确定正例排除在外',
    'ambiguous_groups': ['DF12-02'],
    'ambiguous_region_ids': [],
}

# ---------------------------------------------------------------- 待核页裁定

NEW_SINGLE_GROUPS: List[Dict[str, Any]] = [
    {'region': 'S38-01', 'book': '思濟堂方書', 'page': 38, 'main_text': '桃仁',
     'main_bbox': [553, 574, 592, 661], 'right_bbox': [563, 661, 596, 745],
     'left_bbox': [536, 661, 560, 745],
     'right_text': '二十枚去', 'left_text': '皮尖',
     'certainty': 'ambiguous_lane_split',
     'note': '双轨成立（栏内并列两列小字）。右轨「二十枚去」四字、左轨「皮尖」二字；'
             '按「右轨后左轨」拼接为「二十枚去皮尖」，与清单一致。'
             '「去」归右轨末字还是左轨首字存在两种切分，见 candidates',
     'candidates': [
         {'id': 'a', 'reading': '右「二十枚去」／左「皮尖」', 'basis': '叠框显示「去」与「二 十 枚」同一列、间距一致'},
         {'id': 'b', 'reading': '右「二十枚」／左「去皮尖」', 'basis': '语义上「去皮尖」为完整炮制成词'},
     ]},
]

PENDING_VERDICTS: List[Dict[str, Any]] = [
    {'page': 38, 'page_id': '思濟堂方書/page_38', 'old_status': '待核', 'new_status': '转正（纳入正集）',
     'decision': '双轨成立，转正为 1 组（S38-01），切分待核记 ambiguous_lane_split'},
    {'page': 51, 'page_id': '思濟堂方書/page_51', 'old_status': '待核', 'new_status': '转排除',
     'decision': '复核后判定为单轨：枳殼下方只有「各一錢半」一条小字列，右侧为邻栏「羌活／獨活」，'
                 '未发现并排第二轨；证据见 evidence/ruler/p51_wide.png。'
                 '清单原记「各一錢半／麩炒」，其中「麩炒」在本页该栏未定位到',
     'reversible': True},
]

# ---------------------------------------------------------------- 负例区域

NEGATIVE_REGIONS: List[Dict[str, Any]] = [
    {'id': 'NEG-ST-01', 'book': '思濟堂方書', 'page': 32, 'bbox': [896, 300, 927, 640],
     'category': 'single_track', 'status': 'confirmed',
     'basis': '说明.md 排除理由「藥名後僅單條小字（劑量或炮制擇一）」；本阶段复核该栏确为单轨',
     'evidence': 'existing:说明.md 排除条目 + 分栏 bbox'},
    {'id': 'NEG-ST-02', 'book': '思濟堂方書', 'page': 40, 'bbox': [951, 260, 986, 700],
     'category': 'single_track', 'status': 'confirmed',
     'basis': '说明.md「澤瀉一錢二分、牡丹皮長注均為單軌」；栏内无并列第二轨',
     'evidence': 'existing:说明.md 排除条目'},
    {'id': 'NEG-BT-01', 'book': '思濟堂方書', 'page': 30, 'bbox': [337, 60, 365, 720],
     'category': 'body_text', 'status': 'confirmed',
     'basis': '卷首序正文，字号与行距为正文规格，无大字下并排两小字',
     'evidence': 'existing:说明.md「卷首序，無方劑」'},
    {'id': 'NEG-BT-02', 'book': '思濟堂方書', 'page': 44, 'bbox': [400, 80, 430, 700],
     'category': 'body_text', 'status': 'confirmed',
     'basis': '中風論說正文（说明.md 排除理由），无药名剂量双轨',
     'evidence': 'existing:说明.md 排除条目'},
    {'id': 'NEG-TI-01', 'book': '思濟堂方書', 'page': 30, 'bbox': [902, 20, 1021, 90],
     'category': 'title', 'status': 'confirmed',
     'basis': '卷首标题行（大字跨栏），属 title 类，不是药名 + 双轨',
     'evidence': 'existing:page_layout 归为跨栏宽块 + 说明.md「卷首序」'},
    {'id': 'NEG-RU-01', 'book': '丹方彚编', 'page': 12, 'bbox': [530, 194, 534, 824],
     'category': 'rule', 'status': 'confirmed',
     'basis': '版框左边线：整列墨量占可用高度 82%，为独立竖线，非字形',
     'evidence': 'output/dual_tracks/stage2/evidence/ruler/df_p12_bottom.png'},
    {'id': 'NEG-RU-02', 'book': '丹方彚编', 'page': 12, 'bbox': [637, 204, 640, 820],
     'category': 'rule', 'status': 'confirmed',
     'basis': '栏间界行：列窗测量时在 x 637-638 出现贯穿全高的墨量峰，与两侧字距明显',
     'evidence': 'output/dual_tracks/stage2/evidence/measure/丹方彚编_p12_c637_686.png'},
    {'id': 'NEG-NO-01', 'book': '丹方彚编', 'page': 12, 'bbox': [700, 20, 900, 180],
     'category': 'noise', 'status': 'confirmed',
     'basis': '天头空白带内的污点与透印，不含任何字列',
     'evidence': 'output/dual_tracks/stage2/evidence/ruler/丹方彚编_p12_500_0_1024_450_s1.6.png'},
    {'id': 'NEG-NO-02', 'book': '思濟堂方書', 'page': 30, 'bbox': [4, 10, 43, 200],
     'category': 'noise', 'status': 'unknown',
     'basis': '页面右缘（书耳位置）留白与残墨，是否含残缺字未核；保留 unknown',
     'evidence': 'existing:page_layout 首列窄块'},
    {'id': 'NEG-DF-SINGLE-01', 'book': '丹方彚编', 'page': 12, 'bbox': [536, 210, 585, 815],
     'category': 'single_track', 'status': 'confirmed',
     'basis': '丹方 p12 右半页最左列：藿合油香上手詩……为单一列，未分两轨，'
              '与相邻「木香」栏的双轨形成对照',
     'evidence': 'output/dual_tracks/stage2/evidence/ruler/丹方彚编_p12_530_180_640_460_s4.0.png'},
]

RULE_SAMPLES: List[Dict[str, Any]] = [
    {'id': 'RULE-IND-01', 'kind': 'independent_rule', 'book': '丹方彚编', 'page': 12,
     'bbox': [637, 204, 640, 820], 'status': 'confirmed',
     'description': '独立界行：与两侧字形有明显留白，笔直且贯穿全栏',
     'evidence': 'evidence/measure/丹方彚编_p12_c637_686.png',
     'expected_algo_behavior': '应识别为栏线并从字形聚合中排除'},
    {'id': 'RULE-TOUCH-01', 'kind': 'touching_rule', 'book': '丹方彚编', 'page': 12,
     'bbox': [686, 300, 689, 780], 'status': 'pending_visual',
     'description': '贴字界行：该竖线与「香」「粉」等笔画最近处不足 2px，'
                    '归属（栏线 or 字形长竖）需按字形聚合结果复核',
     'evidence': 'evidence/ruler/df_p12_A.png',
     'expected_algo_behavior': '不得直接并入字形，也不得当作纯噪声丢弃；应保留标记待复核'},
    {'id': 'RULE-LONG-01', 'kind': 'long_glyph_stroke', 'book': '思濟堂方書', 'page': 49,
     'bbox': [718, 520, 738, 690], 'status': 'confirmed',
     'description': 'P49-13 右轨「表裏寒熱虛實」中「表」的竖笔接近栏高，'
                    '易被竖直投影误判为栏线',
     'evidence': 'evidence/../stage1/evidence/思濟堂方書__p49__P49-13.png',
     'expected_algo_behavior': '长竖笔属字形，不得从字形掩膜中剔除'},
    {'id': 'RULE-BROKEN-01', 'kind': 'broken_rule', 'book': '丹方彚编', 'page': 12,
     'bbox': [885, 210, 888, 700], 'status': 'pending_visual',
     'description': '断线界行：该处竖线被淡墨与折痕打断，投影峰不连续',
     'evidence': 'evidence/ruler/zoom_cuidanfan.png',
     'expected_algo_behavior': '允许按分段峰恢复栏线，但须记录置信度，不得强行连通'},
]

# ---------------------------------------------------------------- 分区与标签

SPLITS: Dict[str, Dict[str, Any]] = {
    'development': {
        'purpose': '允许调参、看结果、反复试验',
        'pages': [(('思濟堂方書'), 33), ('思濟堂方書', 35), ('思濟堂方書', 36), ('思濟堂方書', 42),
                  ('丹方彚编', 12)],
        'basis': '历史用途：p33/p35/p36/p42 为第一阶段开发页；丹方 p12 用于异常宽框开发验证',
    },
    'historical_regression': {
        'purpose': '回归对照：不得因新算法改动而退化，但不作为独立结论依据',
        'pages': [('思濟堂方書', 34), ('思濟堂方書', 37), ('思濟堂方書', 39), ('思濟堂方書', 43),
                  ('思濟堂方書', 45), ('思濟堂方書', 46), ('思濟堂方書', 49), ('思濟堂方書', 50)],
        'basis': '第一阶段固定回归集 8 页（含曾被其他方剂任务验收过的 37/39/50）',
    },
    'independent_acceptance': {
        'purpose': '独立验收：只看指标，不用于调参',
        'pages': [('温疫方論', 12), ('思濟堂方書', 38)],
        'basis': '温疫 p12 为跨书诊断页、从未参与本任务调参；思濟堂 p38 本阶段才转正、'
                 '此前未进入任何开发或回归',
    },
}

PAGE_EXPOSURE: Dict[str, Dict[str, Any]] = {
    '思濟堂方書/page_33': {'exposure_history': 'used_in_development', 'used_for': ['双轨标注初测']},
    '思濟堂方書/page_35': {'exposure_history': 'used_in_development', 'used_for': ['dev 样例 dev-35-001']},
    '思濟堂方書/page_36': {'exposure_history': 'used_in_development', 'used_for': ['墨迹测量调试']},
    '思濟堂方書/page_42': {'exposure_history': 'used_in_development', 'used_for': ['墨迹测量调试']},
    '丹方彚编/page_12': {'exposure_history': 'used_in_development', 'used_for': ['异常宽框开发验证（本阶段补标）']},
    '温疫方論/page_12': {'exposure_history': 'unknown',
                         'used_for': ['跨书诊断'],
                         'note': '第一阶段已作为跨书样本出现；是否曾用于阈值选择无法确证，独立性按保守口径限定'},
    '思濟堂方書/page_38': {'exposure_history': 'unknown',
                           'used_for': ['第二阶段转正'],
                           'note': '长期列为待核页、未进入任何开发；但第一阶段曾被目验过是否双轨，'
                                   '不能声称从未曝光'},
}

PAGE_LABELS: Dict[str, Dict[str, Any]] = {
    '思濟堂方書/page_33': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['single_group']},
    '思濟堂方書/page_34': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple', 'unequal_lane_length']},
    '思濟堂方書/page_35': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['unequal_lane_length']},
    '思濟堂方書/page_36': {'rule_state': 'complete', 'wide_box_state': 'normal', 'features': []},
    '思濟堂方書/page_37': {'rule_state': 'complete', 'wide_box_state': 'normal', 'features': []},
    '思濟堂方書/page_38': {'rule_state': 'complete', 'wide_box_state': 'unknown',
                           'features': ['ambiguous_lane_split', 'faint_ink']},
    '思濟堂方書/page_39': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple', 'unequal_lane_length']},
    '思濟堂方書/page_42': {'rule_state': 'complete', 'wide_box_state': 'normal', 'features': []},
    '思濟堂方書/page_43': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple']},
    '思濟堂方書/page_45': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple', 'short_group']},
    '思濟堂方書/page_46': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple', 'short_group']},
    '思濟堂方書/page_49': {'rule_state': 'complete', 'wide_box_state': 'normal',
                           'features': ['same_column_multiple', 'long_glyph_stroke']},
    '思濟堂方書/page_50': {'rule_state': 'complete', 'wide_box_state': 'normal', 'features': []},
    '温疫方論/page_12': {'rule_state': 'partial_or_broken', 'wide_box_state': 'abnormal',
                         'features': ['cross_book', 'approximate_box']},
    '丹方彚编/page_12': {'rule_state': 'partial_or_broken', 'wide_box_state': 'abnormal',
                         'features': ['cross_book', 'same_column_multiple', 'unequal_lane_length',
                                      'faint_ink', 'three_char_main']},
}
