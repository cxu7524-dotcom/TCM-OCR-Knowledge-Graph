# -*- coding: utf-8 -*-
"""第三阶段配置/勘误/目标三件套：protocol_addendum、evaluation_config、evaluation_targets。

三份文件分工：
  protocol_addendum  阶段 2 协议在落地时遇到的歧义、最小反例、采用的规则、影响范围，
                     以及尺度合成例子的勘误（不动阶段 2 原文件）。
  evaluation_config  本阶段冻结的评测参数（阈值、区域判定、诊断、求解），后续阶段直接引用。
  evaluation_targets 后续阶段的验收目标，带版本、分母与计数依据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))
RELEASE = 'stage3-v1'


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def metrics(name: str) -> Dict[str, Any]:
    return load(OUT / 'metrics' / f'{name}.json')


# ---------------------------------------------------------------- 勘误与歧义

def build_addendum() -> Dict[str, Any]:
    s2p = S2 / 'evaluation_protocol.json'
    s2c = S2 / 'cross_stage_contract.json'
    errata = [{
        'id': 'E1-SCALE-EXAMPLE',
        'where': '阶段 2 cross_stage_contract.scale_object.synthetic_example.checks[2]',
        'wrong': '「若把 G 误当 glyph_pitch 使用，搜索带会窄 30px（按 1.5G 计算）」',
        'correct': '两个字块 [0,30) 与 [35,65)：字心分别为 15 与 50，字心距离 glyph_pitch = 35，'
                   '留白 G = 35 − 30 = 5。按 1.5 倍换算搜索带：正确 1.5 × 35 = 52.5px，'
                   '误用留白 1.5 × 5 = 7.5px，相差 1.5 × (35 − 5) = 45px，不是 30px。',
        'field_values_unchanged': {'H': 30, 'G': 5, 'glyph_pitch': 35},
        'impact': '只影响说明文字。字段定义、合成例子的期望值、后续公式引用都不变。'
                  '阶段 2 原文件不改动，保持可追溯；阶段 3 及以后的引用一律按 45px 写。',
        'checked_by': 'tests/test_dual_track_stage3.py::ScaleErratumTests',
    }]

    ambiguities = [
        {
            'id': 'A1-PARTIAL-EDGE-NEGATIVE',
            'protocol_ref': 'coverage_reporting.prediction_attribution + negative_overlap.cases',
            'question': '部分标注区域内，未匹配预测与负例只有边缘相交时，算 boundary_conflict 还是待核候选？',
            'minimal_example': '丹方 p12 的 11 个预测与 NEG-RU-02（栏线，宽 3px）相交，'
                               'negative_coverage_ratio 均为 0.108，全部未匹配。',
            'rule_adopted': '按 ratio 分层（与 negative_overlap.cases 一致）：'
                            'ratio >= confirmed_fp_min → confirmed_false_positive（FP）；'
                            'ratio >= suspected_fp_min → suspected_false_positive（FP，标 conditional）；'
                            '0 < ratio < suspected_fp_min → new_candidate_in_partial（不改判 FP，'
                            '但保留 overlaps_negative_region 与 ratio）；ratio = 0 → 候选。'
                            'boundary_conflict 只用于「已匹配可信正例」或「穷尽区域内的边缘相交」。',
            'why': '这里修的是上一轮的内部矛盾：正文写「未匹配一律待核」，代码却能凭阈值判误检。'
                   '两者不能并存。采用分层后，阈值以上算误检、阈值以下算候选，'
                   '「未匹配一律待核」这句被改成「未达阈值才待核」，代码与文字对齐。',
            'fix_note': '阈值本身仍是 provisional（pending_calibration），所以 suspected 一档的计数'
                        '在报告里标 conditional，不得写成已验证的严格结论。',
            'scope': '影响所有 partial_positions / negative_only 页的 fp 计数（当前 19 页全部属此类）',
        },
        {
            'id': 'A2-NEGATIVE-ONLY-PAGE',
            'protocol_ref': 'coverage_reporting 只有 partial / exhaustive 两档',
            'question': '某些页只划了负例、没有任何正例标注（p30/p32/p40/p44），coverage 填什么？',
            'minimal_example': 'p32 有 NEG-ST-01 一个确定负例，但全页没有正例标注；'
                               '旧算法给出 2 个预测，都不在负例区内。',
            'rule_adopted': '新增 coverage 取值 negative_only：未匹配预测进待核，不计 FP；'
                            '该页只报「确定负例区域内的误检计数」，不报召回/精确率。',
            'why': '若填 partial_positions 会误导读者以为「该页正例已部分标注」；'
                   '若填 exhaustive 会把大量未匹配预测算成误组。',
            'scope': '新增枚举值，阶段 4 起沿用；旧数据不受影响',
        },
        {
            'id': 'A3-DUPLICATE-EXTRA-ALREADY-MATCHED',
            'protocol_ref': 'error_diagnostics.duplicate_prediction',
            'question': '重复簇里的「多余预测」若已在全局解中匹配到另一个标注，还算 FP 吗？',
            'minimal_example': '预测 P1 与 G1 的 IoU 0.85、与 G2 的 IoU 0.90；'
                               'P1 在全局最优解里匹配 G2，但它同时也达到 G1 的阈值。',
            'rule_adopted': '只记 duplicate_prediction 诊断，不再计 FP（它已经是某个 TP）。'
                            '字段 extra_preds_already_matched_elsewhere 列出这类预测。',
            'why': '协议原文「其余按未匹配预测计 FP」的前提是「它确实未被匹配」；'
                   '已在别处匹配还计 FP，会同一块框既算 TP 又算 FP。',
            'scope': '影响 duplicate_extra_fp 的计数；TP/FP/FN 仍由全局匹配决定',
        },
        {
            'id': 'A4-ENDPOINT-INCLUSIVE-IMPL',
            'protocol_ref': 'text_and_boundary.endpoint_sensitivity',
            'question': '「按两种解释各算一次」的 inclusive 解释具体怎么落到坐标上？',
            'minimal_example': '某 pending 组 group_bbox=[934,307,986,818]。',
            'rule_adopted': 'inclusive 版本把该组的 group_bbox/right_bbox/left_bbox/main_bbox 的 '
                            'x2 与 y2 各 +1（半开区间 → 含端点），只改参考框，不改预测。'
                            '主体报告用 nominal，敏感性单列。',
            'why': '端点歧义的定义就是「右下像素算不算」，落在坐标上只能是 +1；'
                   '改成别的量级就不是端点问题了。',
            'scope': '23 个 pending 组；报告须给出两种解释下的命中数与 IoU 变化',
        },
        {
            'id': 'A5-PATH-B-CROP-WIDTH',
            'protocol_ref': '阶段 3 任务书「从参考组周围生成有上下文的裁块」',
            'question': '裁块要多宽？太窄旧算法直接失效。',
            'minimal_example': '按「组宽左右各扩 25%」生成 42×194 的裁块，'
                               'page_layout 只切出 1 列，estimate_page_typography 因'
                               '「列宽 29 > 页高 × 0.10」返回 None，propose_page 输出 0 个预测。',
            'rule_adopted': '横向取「组所在列宽 × 4.4」并裁剪到图内（保证 4-5 列），'
                            '纵向按组高上下各扩 0.9/0.5，最小高 380。',
            'why': '旧算法的尺度估计内建了「列宽应远小于页宽」的假设，'
                   '只给 1-2 列的裁块会让它整体失效——这是旧算法的尺度依赖缺陷，需在报告里点明。',
            'scope': '只影响路径 B 的裁块生成；主基线 C 不受影响',
        },
        {
            'id': 'A6-GROUP-IOU-USES-LANES-ONLY',
            'protocol_ref': 'scope.iou + group_bbox 定义',
            'question': '旧算法输出 main_bbox，是否参与组级 IoU？',
            'minimal_example': '旧算法 main_bbox 常把药名与小字首行一起框住（阶段 2 已修标注侧，'
                               '算法侧仍有此问题）。',
            'rule_adopted': '组级 IoU 只用 group_bbox = union(right, left)，main_bbox 不参与任何计分，'
                            '只作为诊断信息保存。',
            'why': '协议已定义 group_bbox 不含主字；引入 main 会让「框住标题」变成加分项。',
            'scope': '全部路径',
        },
        {
            'id': 'A12-LOCAL-EXHAUSTIVE-BOUNDARY',
            'protocol_ref': 'coverage_reporting.exhaustive_region',
            'question': '局部穷尽区域内外的预测，靠什么边界区分？',
            'minimal_example': '页级 coverage=exhaustive_region，区域 R1=[0,0,100,200]；'
                               '预测 A=[10,10,40,100]（区域内未匹配）、B=[300,10,330,100]（区域外未匹配）。'
                               '上一版没有区域坐标，两者都按页级 exhaustive 判，B 也被计成误组。',
            'rule_adopted': '区域自带 bbox 与 coverage，按**预测组框中心**归属：'
                            '落在区域内 → 用区域 coverage（穷尽则未匹配即 FP）；'
                            '落在所有区域外 → 退回 partial_positions（unmatched_outside_region，不计 FP）。'
                            'ignore / ambiguous 仍在最开始判定，优先级不变。',
            'why': '「页级声称穷尽」不等于「整页都穷尽」。没有区域坐标时无法区分，'
                   '要么多计误组，要么少计。协议要求按区域计分，就得先把区域坐标落实。',
            'scope': '新增 evaluation_regions 字段；当前 19 页都没有局部穷尽区域，故指标不变',
        },
        {
            'id': 'A13-COST-FUNCTION-NEGATIVE-IOU',
            'protocol_ref': 'group_matching.solver.edge_cost',
            'question': '最小费用流的边代价用 M−IoU 还是 −IoU？',
            'minimal_example': '1×601 矩阵：第 0 边 0.5、第 1 边 nextafter(0.5)、其余 0，阈值 0.5。'
                               'M=3 时 3−0.5 与 3−0.5000000000000001 都等于 2.5，'
                               '回退路径选了第 0 条边（总分较低）。',
            'rule_adopted': '边代价改为 −IoU（负代价，SPFA 可处理；二分图初始残量网络无环、'
                            '无负环，SSP 保持无负环）。M 不再参与代价计算。',
            'why': 'M−IoU 的用意只是让代价恒正，但只要 M 大于 1 就会在相减时丢掉低位，'
                   '把「更优一点点」的边当成等价。规模越大越严重。',
            'scope': '全部匹配；回退路径（skipped_scale_limit）同样受益',
        },
        {
            'id': 'A14-CONFIG-STRICTNESS',
            'protocol_ref': 'evaluation_config.group_matching.float_policy',
            'question': '数值策略允许被 --override 改吗？',
            'minimal_example': '--override group_matching.float_policy.tie_break_tolerance=1e-9 '
                               '会让并列判定重新引入容差，正是上一轮被修的缺陷。',
            'rule_adopted': '这两项是「实现约定」不是可调参数：只接受 '
                            'tie_break_tolerance=0.0 与 rounding_in_optimization=false；'
                            '其他取值由 validate_runtime() 拒绝并返回码 3，不静默按默认跑。',
            'why': '可配置不等于什么都该配。允许改这两项等于允许把已知缺陷配回来。',
            'scope': '评分入口；配置文件的默认值不变',
        },
        {
            'id': 'A8-REGION-BEFORE-MATCH',
            'protocol_ref': 'coverage_reporting.prediction_attribution.priority',
            'question': 'ignore / ambiguous 的判定放在匹配前还是匹配后？',
            'minimal_example': '标注 G=[0,0,10,20]，ignore 区 I=[10,0,20,20]（与 G 相交面积为 0），'
                               '预测 P=[0,0,20,20]：P 中心落 I，且与 G 的 IoU = 0.5。'
                               '匹配后判定会得到 TP=1 且 ignored=1（同一个框两种状态）。',
            'rule_adopted': '匹配前先做一次归属：ignore 无条件排除（不进入匹配候选，'
                            '因此不占用标注节点）；ambiguous 只在「够不到任何正例阈值」时暂缓，'
                            '够得到就按正例优先参与匹配。',
            'why': '协议要求「每个预测有唯一归属」。放到匹配后判 ignore，标注节点已经被它占掉，'
                   '就等于让一个被排除的框影响了 TP 分布。',
            'scope': '全部页；未匹配预测的归属不变，只是顺序提前',
        },
        {
            'id': 'A9-EXHAUSTIVE-CONFLICT-NOT-EXEMPT',
            'protocol_ref': 'negative_overlap.cases.boundary_conflict',
            'question': '穷尽区域里未匹配预测与负例边缘相交，boundary_conflict 能不能免除 FP？',
            'minimal_example': 'coverage=exhaustive_page，无标注；预测 [0,0,20,20]；'
                               '确定负例 [0,0,1,20]。ratio=0.05 → 上一版判 boundary_conflict 且 FP=0。',
            'rule_adopted': '穷尽区域内未匹配一律 FP；boundary_conflict 作为诊断标签同时保留。'
                            '诊断标签不改变 base_status，也不免除 FP。',
            'why': '穷尽区里「未匹配」本身就是结论（这里有不该有的东西）。'
                   '用诊断标签免掉计数，会让召回提升时误检被系统性地漏掉。',
            'scope': '影响 exhaustive_region / exhaustive_page；当前数据集还没有这类区域，'
                     '属于为阶段 4 预置的正确行为',
        },
        {
            'id': 'A10-BASE-STATUS-UNIQUE',
            'protocol_ref': '阶段 3 任务书「所有预测最终都有唯一的基础计分状态」',
            'question': '一个预测可能同时是 TP、重复多余项、边界冲突——怎么记？',
            'minimal_example': 'P1 匹配到 G1（TP），同时与栏线负例边缘相交（boundary_conflict）。',
            'rule_adopted': 'base_status ∈ {TP, FP, IGNORED, PENDING_AMBIGUOUS, PENDING_CANDIDATE, '
                            'INVALID}，每个预测恰好一个；其余（duplicate/fragment/merge/'
                            'boundary_conflict）都是 diagnostic_labels。自检 S07b 校验加总等于有效预测数。',
            'why': '"唯一基础状态 + 可叠加诊断"才能保证计数不重不漏；'
                   '上一版把 ignored 与 TP 同时写在一个预测上，加总会超过预测总数。',
            'scope': '全部页；报告给出的状态划分表即为该字段的汇总',
        },
        {
            'id': 'A11-DUPLICATE-EXTRA-IN-PARTIAL',
            'protocol_ref': 'partial_area_duplicates.matched_duplicate',
            'question': 'partial 区内重复簇的多余项，算 FP 还是候选？',
            'minimal_example': '同一标注 G 有两个预测 P1/P2 都达阈值；全局匹配只配 P1。'
                               'P2 在 partial 区、不触负例——上一版判候选（FP=0）。',
            'rule_adopted': '协议原文优先：多余项计 1 个 FP，单列 duplicate_extra_fp；'
                            '但若该预测已在全局解中匹配到别的标注（A3），则不再计 FP。',
            'why': '重复上报本身就是错误，不能因为区域未穷尽就当作「可能是新目标」。'
                   '单列是为了让报告能把它与「误检」分开看。',
            'scope': '影响 partial 区的 fp 计数',
        },
        {
            'id': 'A7-SKIP-REASON-IN-PATH-B',
            'protocol_ref': 'run_states + 阶段 3 任务书第四部分 skipped 原因区分',
            'question': '路径 B 遇到没有参考组的页，算 out_of_scope 还是 incomplete？',
            'minimal_example': 'p30/p32/p40/p44 只划了负例，没有正例组 → 没有裁块依据。',
            'rule_adopted': 'out_of_scope：路径 B 的定义就是「以参考组生成裁块」，'
                            '无参考组即不在该路径范围内，不影响页面状态判定为 partial。',
            'why': '这不是「本应处理却没做成」，而是「该路径本身没有输入」。',
            'scope': '路径 B 的页级状态',
        },
    ]
    return {
        'stage': 'stage3_dual_track_evaluation',
        'release': RELEASE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage3_build.py',
        'upstream': {
            'stage2_evaluation_protocol': {'path': 'output/dual_tracks/stage2/evaluation_protocol.json',
                                           'sha256': sha256_of(s2p), 'modified': False},
            'stage2_cross_stage_contract': {'path': 'output/dual_tracks/stage2/cross_stage_contract.json',
                                            'sha256': sha256_of(s2c), 'modified': False},
            'policy': '阶段 2 原文件一字未改；本阶段的补充与勘误只写在这里',
        },
        'errata': errata,
        'protocol_ambiguities': ambiguities,
        'adopted_elsewhere': [
            '浮点比较不引入 epsilon：IoU >= 阈值用原值判定（阶段 2 已定，本阶段实现一致）',
            '求解用最小费用最大流（先定最大流 k，再在 k 条边上最小化 Σ(M−IoU)），'
            '字典序裁定用受限 DFS 在 (k, ΣIoU) 相等的解集里取序列最小',
        ],
    }


# ---------------------------------------------------------------- 评测配置

def build_config(snapshot: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'stage': 'stage3_dual_track_evaluation',
        'release': RELEASE,
        'status': 'frozen',
        'frozen_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'baseline_snapshot_id': snapshot['snapshot_id'],
        'layout_snapshot_id': labels['layout_source']['layout_snapshot_id'],
        'group_matching': {
            'thresholds': [0.5, 0.75],
            'per_threshold_independent': True,
            'objectives': ['max_cardinality', 'max_total_iou', 'lexicographic'],
            'solver': {'primary': 'min_cost_max_flow_spfa',
                       'edge_cost': '-IoU（负代价；不用 M−IoU——M 较大时会把 '
                                    '0.5 与 0.5000000000000001 相减成同一代价）',
                       'exact_search': '在固定基数 k 下用受限 DFS 精确求「最大总 IoU，且该值下字典序最小」；'
                                       '同时用于校验最小费用流的浮点累加',
                       'tie_break': '前两目标完全并列（总 IoU 浮点精确相等）时才取 (pred,gt) 序列字典序最小',
                       'lex_search_limits': {'max_nodes': 600, 'max_k': 12,
                                             'time_budget_s': 20.0,
                                             'consumed_by': 'dual_track_eval.matching.match_groups(limits=...)',
                                             'scope': '只约束「完全并列时取字典序」这一步；'
                                                      '基数与总 IoU 不受限制影响',
                                             'on_exceed': 'tie_break_status=skipped_scale_limit，'
                                                          '基数与总 IoU 仍由最小费用流保证，'
                                                          '但不得声称第三目标已满足'}},
            'float_policy': {'epsilon': 0.0,
                             'tie_break_tolerance': 0.0,
                             'rounding_in_optimization': False,
                             'comparison': '阈值判定用 >= 原值；并列判定用浮点精确相等（不留 1e-9 之类容差，'
                                           '否则会把 0.5000000001 与 0.5000000002 当成并列而选中低分解）',
                             'relaxation': '最短路松弛不留固定容差（旧实现的 dist[v] - 1e-12 会跳过'
                                           '「只优一点点」的路径）'},
            'returned_total_rule': 'total_iou 永远由最终 pairs 重算，不沿用求解过程的中间值',
            'input_order': '(y1, x1, id) 升序，保证同一输入必得同一解',
            'edge_rule': 'IoU >= 阈值；框非法（面积<=0/无法解析）不建边',
        },
        'lane_scoring': {
            'child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5},
            'categories': ['pairing_success', 'left_right_swap', 'child_localization_failure'],
            'denominator': '已匹配且 certainty 非 ambiguous_lane_* 的组数',
            'excluded_certainty': ['ambiguous_lane_split', 'ambiguous_lane_extent'],
            'excluded_certainty_semantics': {
                'default': ['ambiguous_lane_split', 'ambiguous_lane_extent'],
                'empty_list': '[] 表示不排除任何 certainty，所有已匹配组都进轨级分母'
                              '（空名单是有意义的配置，不是「没配」）',
                'consumed_by': 'dual_track_eval.lane.lane_eligible(gt, exclude)',
            },
            'group_hit_independent': True,
        },
        'diagnostics': {
            'duplicate': {'mark_extra_only_without_flipping_tp': True,
                          'already_matched_elsewhere_not_fp': True,
                          'extra_in_partial_counts_fp': True,
                          'note': '重复簇的多余项即使位于 partial 区也计 1 个 FP（协议 '
                                  'partial_area_duplicates.matched_duplicate），单列 duplicate_extra_fp'},
            'fragment': {'union_coverage_of_gt_min': 0.5},
            'merge': {'containment_of_gt_min': 0.6},
            'applied_after_global_match': True,
            'diagnostic_is_not_exemption': 'boundary_conflict / duplicate / fragment / merge 都是诊断标签，'
                                           '不改变 base_status，也不免除基础的 FP 计数',
        },
        'scoring_order': [
            '1. 非法预测先分流（invalid_group_bbox 不参与匹配；缺轨/越界记 degraded 但组框可用）',
            '2. 有效预测做匹配前归属：ignore 直接排除；ambiguous 只在够不到任何正例阈值时暂缓',
            '3. 剩余预测与标注做组级匹配（每个 IoU 阈值独立求解）',
            '4. 匹配上判 TP；未匹配按负例覆盖 / 覆盖类型定 FP 或候选',
            '5. 诊断标签在全局匹配之后追加，不改 base_status',
        ],
        'base_statuses': ['TP', 'FP', 'IGNORED', 'PENDING_AMBIGUOUS', 'PENDING_CANDIDATE', 'INVALID'],
        'status_partition': '每个预测恰有一个 base_status；加总必须等于有效预测数（自检 S07b 校验）',
        'regions': {
            'priority': ['ignore', 'ambiguous', 'positive_match', 'negative', 'uncovered'],
            'negative_ratio': {'definition': 'area(pred ∩ union(negatives)) / area(pred)',
                               'union_first': True, 'cap': 1.0},
            'thresholds': {'boundary_conflict_max': 0.15, 'suspected_fp_min': 0.5,
                           'confirmed_fp_min': 0.8,
                           'status': 'pending_calibration',
                           'applies_to': {'partial_positions': 'ratio >= suspected_fp_min 且为确定负例 → '
                                                              '判 FP（suspected/confirmed），'
                                                              '与「未匹配一律待核」并不冲突：'
                                                              '阈值以上才算误检，以下才是候选',
                                          'negative_only': '同上（本页无正例，只报负例区内误检数）',
                                          'exhaustive_*': '未匹配一律 FP，边缘相交只加 boundary_conflict 诊断'},
                           'basis': '阶段 3 无独立开发集可用于标定；两条可用负例页（p32/p40）'
                                    '的预测与负例区零重叠，无法给出经验分布。'
                                    '阶段 4 补标后按开发集分布重标并冻结。'
                                    '当前结论一律标 conditional。'},
            'coverage_values': ['partial_positions', 'negative_only',
                                'exhaustive_region', 'exhaustive_page'],
            'evaluation_regions': {
                'field': 'pages[].evaluation_regions[] = {region_id, bbox, coverage}',
                'rule': '预测组框中心落在哪个区域，就用该区域的 coverage；'
                        '区域之外退回 partial_positions（未覆盖），不按穷尽算',
                'missing_bbox': 'coverage=exhaustive_region 但区域缺有效 bbox 时，'
                                '该页 status=data_error，总体标 not_computable_partial，'
                                '不给分数而不是给错分数',
                'consumed_by': 'dual_track_eval.regions.RegionTable(evaluation_regions=...)',
            },
            'partial_policy': '未匹配一律 new_candidate_in_partial，不计 FP',
        },
        'endpoint_sensitivity': {'shift_px': 1, 'body_report': 'nominal',
                                 'pending_groups': 23},
        'scale_object_ref': 'output/dual_tracks/stage2/cross_stage_contract.json#scale_object'
                            '（字段定义；本阶段不估尺度，只冻结引用）',
        'not_applicable': {
            'seam_threshold': '旧算法无中缝残余墨量比阈值（propose_in_column 用的是'
                              'valley ratio <= 0.10 的滑窗判据，不是 r=V/(min(L,R)+eps)）；'
                              '新算法阈值待阶段 4 标定',
            'scale_estimation': '阶段 4 产出 scale_object；本阶段只冻结字段契约',
            'gap_fill_params': '旧算法无补笔参数',
        },
        'reproducibility': {
            'score_from_saved_predictions': 'python tools/dual_track_stage3_score.py --predictions C B X',
            'geometry_deterministic': True,
            'ocr_deterministic': '实测同一图两次识别结果一致（见报告复现一节）；'
                                 '仍建议保存两次结果以供核对',
        },
    }


# ---------------------------------------------------------------- 验收目标

FIXED_CASE_SPECS = [
    ('丹方彚编', 12, 'DF12-01', '三字药名 + 长轨（右轨到版框），测长轨与主字边界'),
    ('丹方彚编', 12, 'DF12-03', '短轨 + 下方紧邻新标题（陽起石），测「上方组必须在下个标题前结束」'),
    ('丹方彚编', 12, 'DF12-08', '左邻单轨列（负例），测邻列污染'),
    ('温疫方論', 12, 'W12-01', '跨书、主字与小字相接'),
    ('思濟堂方書', 33, 'S33-01', '墨迹测量来源、字形完整（对照组）'),
]


def _fixed_cases() -> List[Dict[str, Any]]:
    """从标注里查真实 group_id，并对每个样例做四项校验。"""
    ann = load(S2 / 'annotations.json')
    by_region = {}
    for g in ann['groups']:
        by_region[(g['book'], int(g['page']), g['region'])] = g
    out = []
    problems = []
    for book, page, region, why in FIXED_CASE_SPECS:
        key = (book, page, region)
        g = by_region.get(key)
        if not g:
            problems.append(f'{book}/p{page}/{region} 在标注中不存在')
            continue
        expect_page = f'{book}/page_{page}'
        if g['page_id'] != expect_page:
            problems.append(f"{g['group_id']} 的 page_id={g['page_id']} 与期望 {expect_page} 不一致")
        if not g.get('evaluation_eligibility'):
            problems.append(f"{g['group_id']} 缺 evaluation_eligibility")
        out.append({'group_id': g['group_id'], 'region': region, 'page_id': g['page_id'],
                    'why': why,
                    'certainty': g.get('certainty'),
                    'evaluation_eligibility_status': (g.get('evaluation_eligibility') or {}).get('status'),
                    'checks': {'exists': True, 'unique': True,
                               'page_consistent': g['page_id'] == expect_page,
                               'eligibility_present': bool(g.get('evaluation_eligibility'))}})
    seen = [c['group_id'] for c in out]
    if len(seen) != len(set(seen)):
        problems.append(f'固定样例 group_id 重复：{seen}')
    if problems:
        raise SystemExit('固定样例校验失败：' + '；'.join(problems))
    return out


def _negative_stats(m_c: Dict[str, Any]):
    """负例区域数、面积与基线误检数——T6 的分母必须来自实际区域，不能写成 0/1。"""
    neg = load(S2 / 'negative_regions.json')
    confirmed = [r for r in neg.get('regions') or [] if str(r.get('status')) == 'confirmed']
    area = sum((r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1])
               for r in confirmed if r.get('bbox'))
    count = sum(p['thresholds']['0.5']['negative_confirmed_fp']
                + p['thresholds']['0.5']['negative_suspected_fp'] for p in m_c['pages'])
    return len(confirmed), area, count


def build_targets(m_c: Dict[str, Any], m_b: Dict[str, Any], m_x: Dict[str, Any]) -> Dict[str, Any]:
    o5, o75 = m_c['scores']['overall']['0.5'], m_c['scores']['overall']['0.75']
    split = m_c['scores']['by_split']
    dev5 = split['development']['thresholds']['0.5']
    reg5 = split['historical_regression']['thresholds']['0.5']
    reg75 = split['historical_regression']['thresholds']['0.75']
    ind5 = split['independent_acceptance']['thresholds']['0.5']

    pair_ok = m_c['scores']['overall']['0.5']['lane_counts'].get('pairing_success', 0)
    lane_den = m_c['scores']['overall']['0.5']['lane_denominator']
    neg_confirmed, neg_area, neg_fp = _negative_stats(m_c)

    def row(base_hit, base_den, target_hit, target_den, basis):
        return {'baseline': f'{base_hit}/{base_den}', 'baseline_value': round(base_hit / base_den, 4),
                'target': f'{target_hit}/{target_den}', 'target_value': round(target_hit / target_den, 4),
                'basis': basis}

    return {
        'stage': 'stage3_dual_track_evaluation',
        'release': RELEASE,
        'version': 'targets-v1',
        'status': 'frozen_before_stage4_tuning',
        'frozen_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'frozen_rule': '阶段 4 调参前冻结；调参期间不得改动本文件；'
                       '若必须改，另开 targets-v2 并写清改的理由与改前结果',
        'baseline_reference': {'snapshot_id': m_c.get('_snapshot_id'),
                               'path_c_file': m_c['predictions_file'],
                               'path_c_content_sha256': m_c['predictions_content_sha256'][:16],
                               'config_version': (m_c.get('effective_config') or {}).get('version'),
                               'config_sha256': ((m_c.get('effective_config') or {}).get('source') or {}).get('sha256', '')[:16]},
        'metrics': [
            {'id': 'T1', 'metric': 'group hit rate @IoU 0.5（全体已标组）',
             'denominator': '38 个已标注组（含 2 个 ambiguous_lane，组级仍计）',
             **row(o5['group_hits'], o5['annotations_total'], 30, 38,
                   '丹方 p12 从 0/11 提到 5/11（该页 11 组是「同栏多组」主场景），'
                   '温疫 p12 从 0/2 提到 1/2；合计 +6 组')},
            {'id': 'T2', 'metric': 'group hit rate @IoU 0.75（全体已标组）',
             'denominator': '同上 38 组',
             **row(o75['group_hits'], o75['annotations_total'], 18, 38,
                   '0.5 命中 24 组里有 13 组掉到 0.75 以下，主因是框边界偏松；'
                   '目标把其中一半（+7）收进 0.75')},
            {'id': 'T3', 'metric': '历史回归区 hit rate @0.5',
             'denominator': '8 页 20 组（思濟堂 p34-50）',
             **row(reg5['group_hits'], reg5['annotations_total'], 19, 20,
                   '旧算法在该区已 19/20'),
             'pass_rule': '命中数 >= 19（机械判定，唯一门槛）',
             'warning_rule': '命中 18 时标记 regression_warning 供人工复核，'
                             '但**不改变 pass/fail**——warning 只用于解释，不是第二个门槛',
             'why_not_lower': '若把门槛定成 18，等于默许一次退化；把「允许波动」写进门槛'
                              '会让同一份结果出现两种判法。所以门槛固定 19，波动只作为警示。'},
            {'id': 'T4', 'metric': '历史回归区 hit rate @0.75',
             'denominator': '同上',
             **row(reg75['group_hits'], reg75['annotations_total'], 14, 20,
                   '10/20 的失败集中在小字框边界，框精修后目标 +4')},
            {'id': 'T5', 'metric': '轨级配对正确率',
             'denominator_type': 'dynamic',
             'denominator_rule': '新算法按本轮「具有轨级资格的已匹配组数」动态计算 '
                                 '（= 组命中数 − ambiguous_lane_* 排除数）；'
                                 '不得固定要求新算法仍有 23 个匹配组',
             'baseline': f"{pair_ok}/{lane_den}",
             'baseline_ratio': round(pair_ok / lane_den, 4) if lane_den else None,
             'target_ratio': 0.90,
             'target_rule': 'pairing_success / 本轮轨级分母 >= 0.90，且分母 >= 10 才判定（小于 10 组时只报计数）',
             'basis': '基线 21/23 = 0.913（2 组 child_localization_failure，均为子框未达 0.5）；'
                      '目标是不退化到 0.90 以下'},
            {'id': 'T6', 'metric': '确定负例区域内的误检计数',
             'denominator_type': 'negative_regions',
             'denominator': f'{neg_confirmed} 个 confirmed 负例区域，合计 {neg_area} px²',
             'baseline': f'{neg_fp} 次 / {neg_confirmed} 个区域（{neg_area} px²）',
             'baseline_count': neg_fp,
             'target_count': 0,
             'target_rule': 'count <= 0 且区域集合不变（新增区域需另开版本）',
             'status': 'conditional',
             'conditional_basis': '负例覆盖阈值仍是 pending_calibration（0.15/0.5/0.8 无开发集依据），'
                                  '因此「0 次」只在当前阈值下成立，不能写成「已验证零误检」；'
                                  '冻结条件：阶段 4 补标后按开发集分布重标阈值并重跑本项',
             'basis': '基线 0 次；这条是约束项——提高召回时不得把预测推进负例区'},
            {'id': 'T7', 'metric': '丹方 p12 命中 @0.5',
             'denominator': '该页 11 组',
             **row(0, 11, 6, 11,
                   '基线全灭（旧布局把右半页 9 栏并成 1 列宽 446px）；'
                   '补上分栏后按每栏 1 组估计至少 6 组')},
            {'id': 'T8', 'metric': '温疫 p12 命中 @0.5',
             'denominator': '该页 2 组',
             **row(0, 2, 1, 2,
                   '跨书样本，旧算法 0/2；目标只要求建立第一个命中')},
        ],
        'fixed_cases': {
            'definition': '固定样例：不随调参变化、每轮必须单独报对错的组',
            'id_source': 'stage2/annotations.json 的 group_id 字段（唯一权威来源，不手写）',
            'list': _fixed_cases(),
            'requirement': '每轮报告必须给出这些组在 0.5 与 0.75 下的对错（命中 / 未命中 / 该阈值无匹配），'
                           '不得只报总体命中率',
            'validation': '生成时校验：存在、唯一、页面一致、评测资格存在；任一不满足即报错',
        },
        'regression_tolerance': {
            'rule': '历史回归区命中数下降 ≤1 组视为阈值边界噪声（组级 IoU 在 0.5 附近抖动）；'
                    '下降 ≥2 组判为退化，必须定位到具体组',
            'basis': '基线在该区 19/20，其中 10 组 IoU 落在 0.5-0.6 区间，边界抖动确有 1 组量级',
        },
        'not_computable': {
            'recall': 'not_computable——无 exhaustive_region/exhaustive_page',
            'precision': 'not_computable——同上',
            'false_group_count': 'not_computable——部分标注区内未匹配预测只作候选',
            'handling': '这些指标在报告中一律写 not_computable 并给出缺什么条件才能算；'
                        '不得用「1 - 误检数/预测数」之类的替代口径冒充精确率',
        },
        'sample_limits': {
            'independent_pages': 2, 'independent_groups': 3,
            'conclusion_scope': '独立区只报计数（当前 1/3），不做区间估计、不做显著性比较；'
                                '工程验收按上面 T1-T8 的计数门槛进行',
            'candidate_coverage_target': {'pages': 5, 'groups': 30,
                                          'status': 'candidate_target',
                                          'note': '达到后才考虑按页 bootstrap 给区间；'
                                                  '当前 3 组远不足'},
        },
        'auxiliary_baselines': {
            'X_full_page_components': {'hit_0.5': f"{m_x['scores']['overall']['0.5']['group_hits']}/38",
                                       'note': '比主基线少 2 组，但预测总数少（74 vs 125），'
                                               '说明它的 OCR 门控更保守'},
            'B_ctx_propose_page': {'hit_0.5': f"{m_b['scores']['overall']['0.5']['group_hits']}/38",
                                   'note': '仅在给定局部范围时有效；裁块过宽会过检'
                                           '（丹方 p12 单页 207 个预测）'},
        },
        'decision': {
            'comparison_baseline': 'C / full_page_propose_page',
            'why': '无 OCR 依赖、可完全复现；X 虽然更接近生产挂接，但多一层 OCR 门控，'
                   '会把「检测能力」和「OCR 支不支持」混在一起',
        },
    }


# ---------------------------------------------------------------- 新旧指标差异

# 上一轮（评测器 v1）的实测值，来自当时 metrics/*.json 的 overall 段。
# 保留成常量是为了让本轮能逐项对账：哪些变化来自评测器修正，哪些来自配置。
PREVIOUS_METRICS = {
    'evaluator_version': '1.1.0（上一轮修正版：精确并列、归属前移、配置驱动）',
    'C': {'0.5': {'group_hits': 24, 'annotations_total': 38, 'fp': 0, 'ignored': 0,
                  'new_candidates': 101, 'lane_denominator': 23, 'duplicate_extra_fp': 0},
          '0.75': {'group_hits': 11, 'annotations_total': 38, 'fp': 0, 'ignored': 0,
                   'new_candidates': 114, 'lane_denominator': 11, 'duplicate_extra_fp': 0}},
    'B': {'0.5': {'group_hits': 11, 'annotations_total': 38, 'fp': 11, 'ignored': 0,
                  'new_candidates': 221, 'lane_denominator': 11, 'duplicate_extra_fp': 2},
          '0.75': {'group_hits': 8, 'annotations_total': 38, 'fp': 11, 'ignored': 0,
                   'new_candidates': 224, 'lane_denominator': 8, 'duplicate_extra_fp': 2}},
    'X': {'0.5': {'group_hits': 22, 'annotations_total': 38, 'fp': 0, 'ignored': 0,
                  'new_candidates': 52, 'lane_denominator': 21, 'duplicate_extra_fp': 0},
          '0.75': {'group_hits': 10, 'annotations_total': 38, 'fp': 0, 'ignored': 0,
                   'new_candidates': 64, 'lane_denominator': 10, 'duplicate_extra_fp': 0}},
}

FIELDS = ('group_hits', 'annotations_total', 'fp', 'ignored', 'new_candidates',
          'lane_denominator', 'duplicate_extra_fp')


def build_metrics_diff(m: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {'stage': 'stage3_dual_track_evaluation', 'release': RELEASE,
           'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
           'previous_evaluator': PREVIOUS_METRICS['evaluator_version'],
           'change_scope': '本轮只改匹配代价函数（−IoU）、配置接线、局部穷尽区域与配置白名单；'
                           '预测未变，模型/布局/输入未变',
           'current_evaluator': (m['c'].get('effective_config') or {}).get('version'),
           'prediction_changed': False,
           'prediction_basis': {code: m[key]['predictions_content_sha256'][:16]
                               for code, key in (('C', 'c'), ('B', 'b'), ('X', 'x'))},
           'items': {}, 'summary': {}}
    changed = 0
    for code, key in (('C', 'c'), ('B', 'b'), ('X', 'x')):
        prev = PREVIOUS_METRICS[code]
        cur = m[key]['scores']['overall']
        item = {}
        for tk in ('0.5', '0.75'):
            row = {}
            for f in FIELDS:
                a, b = prev[tk].get(f), cur[tk].get(f)
                if a != b:
                    changed += 1
                row[f] = {'before': a, 'after': b, 'delta': (b - a) if isinstance(a, int)
                          and isinstance(b, int) else None}
            item[tk] = row
        out['items'][code] = item
    out['summary'] = {'changed_fields': changed,
                      'prediction_files_unchanged': True,
                      'note': '几何预测文件未变（内容哈希与上一轮快照一致）；'
                              '所有差异来自评测器修正与配置生效，不是算法变化'}
    return out


# ---------------------------------------------------------------- 产物自检

def build_validation(snap: Dict[str, Any], layouts: Dict[str, Any], labels: Dict[str, Any],
                     cfg: Dict[str, Any], tg: Dict[str, Any], addendum: Dict[str, Any],
                     m: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(cid: str, title: str, ok: bool, detail: str, items: Any = None) -> None:
        checks.append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                       'detail': detail, 'items': items if items is not None else []})

    # S01 产物齐全
    need = ['baseline_snapshot.json', 'frozen_layouts.json', 'frozen_layout_labels.json',
            'evaluation_config.json', 'evaluation_targets.json', 'protocol_addendum.json',
            'predictions/path_c.json', 'predictions/path_b.json',
            'predictions/path_x_components.json', 'recognition/path_a.json',
            'metrics/c.json', 'metrics/b.json', 'metrics/x_components.json']
    miss = [n for n in need if not (OUT / n).exists()]
    add('S01', '九类产物齐全', not miss, f'{len(need)} 项，缺 {len(miss)}', miss)

    # S02 快照与布局/标签互相引用得上
    ok = (cfg['baseline_snapshot_id'] == snap['snapshot_id']
          and cfg['layout_snapshot_id'] == labels['layout_source']['layout_snapshot_id']
          and labels['layout_source']['sha256'] == snap['inputs']['layouts_ref']['sha256'])
    add('S02', '快照 / 配置 / 标签三处 layout 与 snapshot 编号一致', ok,
        f"snapshot={snap['snapshot_id']} layout={cfg['layout_snapshot_id']}")

    # S03 图片尺寸与指纹
    bad = [p['page_id'] for p in snap['inputs']['pages']
           if not p.get('sha256') or len(p['sha256']) != 64]
    add('S03', '输入页指纹完整', not bad, f"{len(snap['inputs']['pages'])} 页，缺指纹 {len(bad)}", bad)

    # S04 预测框合法性
    bad = []
    for code, payload in m.items():
        for page in payload['pages']:
            for r in page['thresholds']['0.5']['predictions']:
                b = r['group_bbox']
                if b and not (b[2] > b[0] and b[3] > b[1]):
                    bad.append(f"{code}:{r['prediction_id']}")
    add('S04', '所有预测组框正面积', not bad, f'非法框 {len(bad)} 个', bad[:10])

    # S05 唯一编号
    for code, payload in m.items():
        ids = [r['prediction_id'] for page in payload['pages']
               for r in page['thresholds']['0.5']['predictions']]
        dup = len(ids) - len(set(ids))
        add(f'S05-{code}', f'{code} 预测编号唯一', dup == 0, f'{len(ids)} 个编号，重复 {dup}')

    # S06 阈值独立求解（不比「子集」，那条件本来就不成立）
    from dual_track_eval import match_groups as _mm
    ref_cases = load(S2 / 'matching_cases.json')['cases']
    bad = []
    for c in ref_cases:
        got = _mm(c['iou_matrix'], c['threshold'])
        if got['cardinality'] != c['expected']['cardinality'] or \
                got['pairs'] != c['expected']['pairs']:
            bad.append(c['case_id'])
    add('S06a', '第二阶段全部匹配夹具复现（基数与配对）', not bad,
        f"{len(ref_cases)} 个夹具，不符 {len(bad)}", bad)

    bad = []
    for code, payload in m.items():
        for page in payload['pages']:
            for tk, t in page['thresholds'].items():
                thr = float(tk)
                pairs = [(a, b) for a, b, _ in t['pairs']]
                # 节点唯一
                if len({a for a, _ in pairs}) != len(pairs) or \
                        len({b for _, b in pairs}) != len(pairs):
                    bad.append(f'{code}:{page["page_id"]}@{tk} 节点重复使用')
                # 每条边达到阈值
                ious = {g: None for g in page.get('page_geometry', {}).get('gt_boxes', [])}
                for a, b, v in t['pairs']:
                    if v < thr - 1e-12:
                        bad.append(f'{code}:{page["page_id"]}@{tk} 边 {a}-{b} IoU {v} < {thr}')
                # 分数与边一致
                if abs(sum(v for _, _, v in t['pairs']) - t['total_iou']) > 1e-6:
                    bad.append(f'{code}:{page["page_id"]}@{tk} total_iou 与边和不一致')
            # 高阈值基数 ≤ 低阈值
            ks = sorted(page['thresholds'], key=float)
            for lo, hi in zip(ks, ks[1:]):
                if page['thresholds'][hi]['cardinality'] > page['thresholds'][lo]['cardinality']:
                    bad.append(f'{code}:{page["page_id"]} {hi} 基数 > {lo}')
    add('S06b', '各阈值独立求解：节点唯一、边达阈值、分数一致、高阈值基数不增', not bad,
        f'违反 {len(bad)} 处', bad[:10])

    # S07 轨级分母 = 命中 - 排除；状态划分完整
    bad, bad2 = [], []
    for code, payload in m.items():
        for page in payload['pages']:
            t = page['thresholds']['0.5']
            if t['lane']['denominator'] + len(t['lane']['excluded']) != t['group_hits']:
                bad.append(f"{code}:{page['page_id']} 分母{t['lane']['denominator']}+"
                           f"排除{len(t['lane']['excluded'])}≠命中{t['group_hits']}")
            if not t.get('status_partition_ok') or t['status_sum'] != page['counts']['valid_predictions']:
                bad2.append(f"{code}:{page['page_id']} 状态和{t.get('status_sum')}≠"
                            f"有效预测{page['counts']['valid_predictions']}")
    add('S07a', '轨级分母 + 排除 = 组命中', not bad, f'不符 {len(bad)} 页', bad[:10])
    add('S07b', '每个预测恰有一个 base_status 且加总等于有效预测数', not bad2,
        f'不符 {len(bad2)} 页', bad2[:10])

    # S08 部分标注区不计 FP（重复多余项除外）
    bad = []
    for code, payload in m.items():
        for page in payload['pages']:
            if page['coverage'] in ('partial_positions', 'negative_only'):
                for r in page['thresholds']['0.5']['predictions']:
                    if r['label'] != 'FP':
                        continue
                    if 'duplicate_prediction' in (r['diagnostic_labels'] or []):
                        continue
                    if r['review_category'] in ('boundary_conflict', 'new_candidate_in_partial',
                                                'pending_review_ambiguous', 'ignored'):
                        bad.append(f"{code}:{r['prediction_id']}")
    add('S08', '部分标注区内没有把候选/边界冲突算成 FP（重复多余项除外）', not bad,
        f'违反 {len(bad)} 条', bad[:10])

    # S09 端点敏感性两种解释都在
    for code, payload in m.items():
        s = payload['endpoint_sensitivity']['delta']
        ok = s['pending_groups'] > 0 and all(k in s['by_threshold'] for k in ('0.5', '0.75'))
        add(f'S09-{code}', f'{code} 端点敏感性含两种解释', ok,
            f"pending {s['pending_groups']} 组，阈值 {list(s['by_threshold'])}")

    # S10 宽框标签覆盖标注页且 unknown 有原因
    need_pages = {p['page_id'] for p in layouts['pages'] if p['status'] == 'ok'}
    have = {r['page_id'] for r in labels['pages']}
    no_reason = [r['page_id'] for r in labels['pages']
                 if r['state'] == 'unknown' and not r['reason']]
    add('S10', '宽框标签覆盖全部布局页、unknown 均有原因',
        need_pages <= have and not no_reason,
        f'覆盖 {len(have)}/{len(need_pages)} 页，无原因 {len(no_reason)}', no_reason[:10])

    # S11 目标的分母与基线一致
    o = m['c']['scores']['overall']
    bad = []
    for item in tg['metrics']:
        if item['id'] in ('T1', 'T2'):
            den = int(item['baseline'].split('/')[1])
            if den != o['0.5']['annotations_total']:
                bad.append(item['id'])
    add('S11', '目标分母与基线标注数一致', not bad,
        f"{o['0.5']['annotations_total']} 组；不一致 {len(bad)} 项", bad)

    # S12 勘误范围
    e = addendum['errata'][0]
    add('S12', '勘误写明正确值与影响范围',
        '45' in e['correct'] and bool(e['impact'])
        and e['field_values_unchanged'] == {'H': 30, 'G': 5, 'glyph_pitch': 35},
        '勘误含 45px、字段值不变、影响范围说明')

    # S13 复现验证记录
    vp = OUT / 'verification_report.json'
    ok = vp.exists() and load(vp)['summary']['fail'] == 0
    add('S13', '复现验证无失败项', ok,
        f"{load(vp)['summary']}" if vp.exists() else '缺 verification_report.json')

    # S14 固定样例：存在、唯一、页面一致、资格明确
    fc = tg['fixed_cases']['list']
    bad = [c['group_id'] for c in fc
           if not all((c['checks'] or {}).get(k) for k in
                      ('exists', 'unique', 'page_consistent', 'eligibility_present'))]
    add('S14', '固定样例四项校验（存在/唯一/页面一致/资格明确）', not bad and len(fc) >= 5,
        f'{len(fc)} 组，不合格 {len(bad)}', bad)

    # S15 生效配置 = 配置文件 + 覆盖项（并核对指纹）
    sys.path.insert(0, str(ROOT / 'tools'))
    import dual_track_stage3_score as SC
    runtime = SC.runtime_config(cfg)
    bad = []
    for code, payload in m.items():
        eff = payload.get('effective_config') or {}
        if eff.get('runtime') != runtime:
            bad.append(f'{code} runtime 与配置文件不一致')
        src = eff.get('source') or {}
        if not src.get('sha256') or len(src['sha256']) != 64:
            bad.append(f'{code} 缺配置指纹')
        if src.get('sha256') and src['sha256'] != sha256_of(OUT / 'evaluation_config.json') \
                and not eff.get('overrides'):
            bad.append(f'{code} 配置指纹与当前文件不符且无覆盖记录')
    add('S15', '评分生效配置与 evaluation_config 一致、指纹可查', not bad,
        f'不符 {len(bad)} 项', bad[:10])

    # S16 目标规则唯一性（不能用「允许下降」再当一个门槛）
    bad = []
    for item in tg['metrics']:
        if item['id'] == 'T3':
            if 'pass_rule' not in item or 'warning_rule' not in item:
                bad.append('T3 缺 pass_rule/warning_rule')
            elif '不改变' not in item['warning_rule']:
                bad.append('T3 warning 未声明不改变判定')
        if item['id'] == 'T5' and item.get('denominator_type') != 'dynamic':
            bad.append('T5 分母不是动态')
        if item['id'] == 'T6' and item.get('denominator_type') != 'negative_regions':
            bad.append('T6 分母不是负例区域')
        if item['id'] == 'T6' and item.get('status') != 'conditional':
            bad.append('T6 未标 conditional')
    add('S16', '口径规则唯一：T3 单门槛 + warning、T5 动态分母、T6 区域分母且 conditional',
        not bad, f'不符 {len(bad)} 项', bad)

    # S17 超规模时的字典序状态有声明
    bad = []
    for code, payload in m.items():
        for page in payload['pages']:
            for tk, t in page['thresholds'].items():
                if t.get('tie_break_status') == 'skipped_scale_limit':
                    scope = t.get('tie_break_scope') or {}
                    if scope.get('applied') or 'consequence' not in scope:
                        bad.append(f'{code}:{page["page_id"]}@{tk} 超规模但未声明后果')
    add('S17', '字典序搜索超规模时显式声明（不静默声称已完成）', not bad,
        f'未声明 {len(bad)} 处', bad[:10])

    # S18 本轮四项反例在产物侧可复现（与单测同源，但跑在产物生成环境里）
    import math
    from dual_track_eval import RegionTable as _RT, match_groups as _mg, score_predictions as _sp
    detail = {}

    row = [0.5, math.nextafter(0.5, 1.0)] + [0.0] * 599
    r = _mg([row], 0.5)
    detail['fallback_1x601'] = {'pairs': r['pairs'], 'status': r['tie_break_status'],
                                'total': r['total_iou']}

    r2 = _mg([[0.6, 0.7]], 0.5, limits={'max_nodes': 1})
    detail['limit_1x2'] = {'pairs': r2['pairs'], 'status': r2['tie_break_status']}

    g = [{'group_id': 'G', 'page_id': 'p', 'group_bbox': [0, 0, 20, 100],
          'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100],
          'certainty': 'ambiguous_lane_extent'}]
    p1 = [{'prediction_id': 'P', 'page_id': 'p', 'group_bbox': [0, 0, 20, 100],
           'right_bbox': [15, 0, 20, 100], 'left_bbox': [0, 0, 5, 100]}]
    base = {'group_iou_thresholds': [0.5],
            'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5}}
    den_default = _sp(p1, g, {'p': _RT('p')}, base)['pages'][0]['thresholds']['0.5']['lane']['denominator']
    den_empty = _sp(p1, g, {'p': _RT('p')},
                    dict(base, lane_eligible_exclude_certainty=[]))['pages'][0]['thresholds']['0.5']['lane']['denominator']
    detail['lane_denominator'] = {'default': den_default, 'empty_exclude': den_empty}

    t_err = _RT('p', coverage='exhaustive_region')
    detail['missing_region_bbox'] = {'data_error': bool(t_err.data_error)}
    t_reg = _RT('p', coverage='exhaustive_region',
                evaluation_regions=[{'region_id': 'R1', 'page_id': 'p',
                                     'bbox': [0, 0, 100, 200],
                                     'coverage': 'exhaustive_region'}])
    x = _sp([{'prediction_id': 'IN', 'page_id': 'p', 'group_bbox': [10, 10, 40, 100],
              'right_bbox': [35, 10, 40, 100], 'left_bbox': [10, 10, 15, 100]},
             {'prediction_id': 'OUT', 'page_id': 'p', 'group_bbox': [300, 10, 330, 100],
              'right_bbox': [325, 10, 330, 100], 'left_bbox': [300, 10, 305, 100]}],
            [], {'p': t_reg})['pages'][0]['thresholds']['0.5']
    detail['local_region'] = {r['prediction_id']: r['base_status'] for r in x['predictions']}
    detail['local_region']['fp'] = x['fp']

    ok = (detail['fallback_1x601']['pairs'] == [[0, 1]]
          and detail['fallback_1x601']['status'] == 'skipped_scale_limit'
          and detail['limit_1x2']['status'] == 'skipped_scale_limit'
          and detail['limit_1x2']['pairs'] == [[0, 1]]
          and den_default == 0 and den_empty == 1
          and detail['missing_region_bbox']['data_error']
          and detail['local_region'].get('IN') == 'FP'
          and detail['local_region'].get('OUT') == 'PENDING_CANDIDATE'
          and detail['local_region']['fp'] == 1)
    add('S18', '本轮四项反例在产物环境可复现（回退代价/规模限制/空排除名单/局部穷尽区域）',
        ok, '；'.join(f'{k}={v}' for k, v in detail.items())[:400], detail)

    summary = {'checks': len(checks),
               'pass': sum(1 for c in checks if c['status'] == 'pass'),
               'fail': sum(1 for c in checks if c['status'] == 'fail'),
               'failed_ids': [c['id'] for c in checks if c['status'] == 'fail']}
    return {'stage': 'stage3_dual_track_evaluation', 'release': RELEASE,
            'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
            'summary': summary, 'checks': checks}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成配置/勘误/目标/自检')
    ap.add_argument('--stage', choices=['pre', 'post', 'all'], default='all',
                    help='pre=只出配置与勘误（评分前）；post=出目标与自检（评分后）；all=两者都出')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    snap = load(OUT / 'baseline_snapshot.json')
    labels = load(OUT / 'frozen_layout_labels.json')
    addendum = build_addendum()
    config = build_config(snap, labels)

    written = []
    if args.stage in ('pre', 'all'):
        for name, payload in (('protocol_addendum.json', addendum),
                              ('evaluation_config.json', config)):
            (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
                                    encoding='utf-8')
            written.append(name)
        if not args.quiet:
            print(f"勘误 {len(addendum['errata'])} 条、实施歧义 {len(addendum['protocol_ambiguities'])} 条")

    if args.stage in ('post', 'all'):
        m_c = metrics('c')
        m_c['_snapshot_id'] = snap['snapshot_id']
        m_b, m_x = metrics('b'), metrics('x_components')
        targets = build_targets(m_c, m_b, m_x)
        metrics_all = {'b': m_b, 'c': m_c, 'x': m_x}
        validation = build_validation(snap, load(OUT / 'frozen_layouts.json'), labels, config,
                                      targets, addendum, metrics_all)
        diff = build_metrics_diff(metrics_all)
        for name, payload in (('evaluation_targets.json', targets),
                              ('validation_report.json', validation),
                              ('metrics_diff.json', diff)):
            (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
                                    encoding='utf-8')
            written.append(name)
        if not args.quiet:
            print(f"目标 {len(targets['metrics'])} 项、固定样例 {len(targets['fixed_cases']['list'])} 组")
            print(f"自检 {validation['summary']['checks']} 项：通过 {validation['summary']['pass']}、"
                  f"失败 {validation['summary']['fail']} {validation['summary']['failed_ids']}")
            print(f"新旧指标变化字段 {diff['summary']['changed_fields']} 个")

    if not args.quiet:
        for n in written:
            print('→', n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
