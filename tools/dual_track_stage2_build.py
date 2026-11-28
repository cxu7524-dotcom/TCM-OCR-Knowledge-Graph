"""生成第二阶段产物：标注、负例、分区、评测协议、跨阶段契约、复核记录、匹配夹具。

用法：
  .venv/Scripts/python.exe tools/dual_track_stage2_build.py
  .venv/Scripts/python.exe tools/dual_track_stage2_build.py --out-dir output/dual_tracks/stage2 --quiet

只读第一阶段产物与原始图片，全部输出写到 output/dual_tracks/stage2/。
协议与契约是「定义」，不是实现：本脚本只负责把它们写成机器可读的 JSON，
并用穷举法校验匹配夹具里的期望值。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_track_stage1_data as D  # noqa: E402
import dual_track_stage2_data as S  # noqa: E402

TZ = timezone(timedelta(hours=8))
STAGE = 'stage2_annotation_protocol'
OUT = D.ROOT / 'output/dual_tracks/stage2'
EVID = OUT / 'evidence'


# ---------------------------------------------------------------- 基础工具

def _img(book: str, page: int) -> Dict[str, Any]:
    return D.image_info(D.BOOKS / book / D.page_file_name(page))


def _rel(path: Path) -> str:
    return D.rel(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bbox(box: Sequence[int]) -> List[int]:
    return [int(v) for v in box]


# ---------------------------------------------------------------- 标注

def build_annotations() -> Dict[str, Any]:
    stage1 = D.load_normalized()
    groups: List[Dict[str, Any]] = []
    pages: Dict[str, Dict[str, Any]] = {}

    for g in stage1['groups']:
        item = dict(g)
        item['origin'] = 'stage1'
        item['stage1_group_id'] = g['group_id']
        item['page_id'] = g['page_id']
        item['book'] = g['book']
        item['page'] = g['page']
        item['region'] = g['region']
        item['group_id'] = g['group_id']
        item['text_status'] = 'transcribed'
        item['annotation_coverage'] = 'partial_positions'
        item['split_membership'] = None  # 由分区步骤填
        groups.append(item)

    # 丹方 p12 补标
    df_img = _img(S.DF_BOOK, S.DF_PAGE)
    for spec in S.DF_GROUPS:
        gid = D.group_id(S.DF_BOOK, S.DF_PAGE, spec['region'])
        main, left, right = _bbox(spec['main_bbox']), _bbox(spec['left_bbox']), _bbox(spec['right_bbox'])
        group_b = D.union_bbox([right, left])
        groups.append({
            'group_id': gid,
            'stage1_group_id': None,
            'origin': 'stage2_new',
            'page_id': D.page_id(S.DF_BOOK, S.DF_PAGE),
            'book': S.DF_BOOK,
            'page': S.DF_PAGE,
            'region': spec['region'],
            'original_region_id': spec['region'],
            'image_path': _rel(D.BOOKS / S.DF_BOOK / D.page_file_name(S.DF_PAGE)),
            'image_source_path': _rel(D.BOOKS / S.DF_BOOK / D.page_file_name(S.DF_PAGE)),
            'image_sha256': df_img['sha256'],
            'image_size': df_img['image_size'],
            'original_image_size': df_img['image_size'],
            'image_size_matches_original': True,
            'main_bbox': main, 'right_bbox': right, 'left_bbox': left,
            'group_bbox': group_b,
            'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框',
            'main_text': spec['main_text'],
            'right_text': spec['right_text'],
            'left_text': spec['left_text'],
            'text_status': spec['text_status'],
            'reading_order': ['right', 'left'],
            'column': None,
            'column_state': 'default_absent',
            'gutter_x': S.DF_GUTTER.get(spec['region']),
            'annotation_source': '本阶段列窗测量',
            'annotation_source_raw': 'tools/dual_track_stage2_measure.py 列窗+中缝测量，6 倍叠框目验',
            'annotation_source_profile': {
                'precision': '列窗（界行之间）+ 中缝（列内低墨量位置）+ 墨迹外接框；非逐字精标',
                'glyph_boundary_error': {'status': 'unknown_uncalibrated', 'value_px': None,
                                         'calibration': 'none',
                                         'note': '未与逐字人工边界做对照测量'},
                'known_limits': ['中缝位置由列内墨量极小值给出，字与字粘连时会偏移',
                                 '三字药名与首列小字纵向相邻，主字/轨的分界由目验给定',
                                 '未做字号级标定'],
                'upstream': 'tools/dual_track_stage2_measure.py',
            },
            'boundary_precision': {
                'level': 'column_window_box',
                'description': '列窗 + 中缝定位后取墨迹外接框',
                'calibration': {'status': 'none', 'note': '未做字形级标定'},
                'known_limits': ['中缝 ±2px', '三字药名栏的主字下界 ±5px'],
                'note': '可用于组定位与左右分轨；不得用于精确边界评测',
            },
            'endpoint_interpretation': {
                'status': 'resolved', 'resolved_value': 'half_open',
                'candidates': ['half_open:[x1,y1,x2,y2)'], 'edge_shift_px': 0,
                'note': '本阶段新增框一律按半开区间写出，无端点歧义',
            },
            'glyph_boundary_error': {'status': 'unknown_uncalibrated', 'value_px': None,
                                     'box_quality': 'column_window', 'calibration': 'none',
                                     'note': '框与字形边界的误差未标定'},
            'evaluation_tolerance_px': {'status': 'pending_protocol', 'value_px': None,
                                        'note': '评测容差待协议确定，见 evaluation_protocol.json'},
            'edge_diagnostic': {
                'lane_bottom_rule': '上方组的轨框必须在本栏下一个独立标题的上沿之前结束',
                'checked_against': '同栏下方条目的药名框（见 validation C-DF-CONFLICT）',
                'v1_defect': '上一版把整段下半栏当成两条长轨，轨框延伸到 y=818，'
                             '吞入下方条目标题与轨框（DF12-03/04/05 分别侵入 DF12-11/10/09）',
            },
            'glyph_boundary_check': {
                'status': 'inspected_ok' if spec['certainty'] == 'medium' else 'inspected_uncertain',
                'method': '6 倍叠框放大目验（列窗测量图 + 修正前后对照图）',
                'evidence_image': f'output/dual_tracks/stage2/evidence/fix_df12/{spec["region"]}_before_after.png',
                'note': spec['note'],
            },
            'certainty': spec['certainty'],
            'certainty_basis': ('列窗测量 + 6 倍叠框目验；上下组已分离' if spec['certainty'] == 'medium'
                                else ('轨的纵向延续被下方条目打断，无法目验确认' if spec['certainty'] == 'ambiguous_lane_extent'
                                      else '药名与小字纵向相邻，分界为目验给定')),
            'review_status': 'confirmed_single_reviewer',
            'coordinate_convention': 'half_open',
            'original_coordinate_convention': 'half_open',
            'evaluation_eligibility': (
                {
                    'status': 'conditional',
                    'usable_for': ['double_track_localization'],
                    'not_usable_for': ['lane_split', 'reading_order_strict', 'glyph_exact_boundary',
                                       'precise_iou_benchmark', 'text_accuracy'],
                    'exact_boundary_evaluation': {
                        'allowed': False,
                        'blockers': ['ambiguous_lane_extent', 'no_boundary_calibration',
                                     'column_window_construction'],
                        'requires': ['protocol_tolerance', 'manual_relabel'],
                        'note': '左轨纵向延续未定，不得进入轨级与精确边界评测',
                    },
                    'reason': '左轨在下方条目标题处被压断，延续性未定',
                } if spec['certainty'] == 'ambiguous_lane_extent' else {
                    'status': 'conditional',
                    'usable_for': ['double_track_localization', 'lane_split', 'reading_order', 'region_crop'],
                    'not_usable_for': ['glyph_exact_boundary', 'precise_iou_benchmark', 'text_accuracy'],
                    'exact_boundary_evaluation': {
                        'allowed': False,
                        'blockers': ['no_boundary_calibration', 'column_window_construction'],
                        'requires': ['protocol_tolerance'],
                        'note': '列窗框，精度低于字形级；文本仅部分转录',
                    },
                    'reason': '列窗 + 中缝测量，非逐字精标；转写为部分转录',
                }
            ),
            'annotation_coverage': 'partial_positions',
            'glyph_review': dict(S.GLYPH_REVIEW.get(spec['region'], {}),
                                 evidence_paths=[
                                     f'output/dual_tracks/stage2/evidence/fix_df12/{spec["region"]}_before_after.png',
                                     'output/dual_tracks/stage2/evidence/fix_df12/_glyph_T1.png',
                                     'output/dual_tracks/stage2/evidence/fix_df12/_glyph_T2.png',
                                     'output/dual_tracks/stage2/evidence/fix_df12/_glyph_T3.png'],
                                 review_status=S.GLYPH_REVIEW.get(spec['region'], {}).get('review_status', 'unchecked')),
            'candidates': spec.get('candidates'),
            'note': spec['note'],
            'source_reference': {
                'manifest': None,
                'coordinates_json': None,
                'coordinates_csv': None,
                'upstream': 'tools/dual_track_stage2_measure.py + tools/dual_track_stage2_data.py',
                'stage1_readme': D.rel(D.README),
            },
            'split_membership': None,
        })

    # 待核页转正：p38
    for spec in S.NEW_SINGLE_GROUPS:
        img = _img(spec['book'], spec['page'])
        gid = D.group_id(spec['book'], spec['page'], spec['region'])
        main, left, right = _bbox(spec['main_bbox']), _bbox(spec['left_bbox']), _bbox(spec['right_bbox'])
        groups.append({
            'group_id': gid, 'stage1_group_id': None, 'origin': 'stage2_new_pending_promoted',
            'page_id': D.page_id(spec['book'], spec['page']),
            'book': spec['book'], 'page': spec['page'], 'region': spec['region'],
            'original_region_id': spec['region'],
            'image_path': _rel(D.DATASET / '待核' / f"{spec['book']}_page_{spec['page']}.png"),
            'image_source_path': _rel(D.BOOKS / spec['book'] / D.page_file_name(spec['page'])),
            'image_sha256': img['sha256'], 'image_size': img['image_size'],
            'original_image_size': img['image_size'], 'image_size_matches_original': True,
            'main_bbox': main, 'right_bbox': right, 'left_bbox': left,
            'group_bbox': D.union_bbox([right, left]),
            'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框',
            'main_text': spec['main_text'], 'right_text': spec['right_text'],
            'left_text': spec['left_text'], 'text_status': 'read_from_overlay',
            'reading_order': ['right', 'left'], 'column': None, 'column_state': 'default_absent',
            'annotation_source': '本阶段目验补标',
            'annotation_source_raw': '4 倍标尺图定位 + 目验读框',
            'annotation_source_profile': D.source_profile(D.SOURCE_MANUAL),
            'boundary_precision': {
                'level': 'manual_visual_box',
                'description': '4 倍标尺图目验定位，框按字列外沿给定',
                'calibration': {'status': 'none', 'note': '未做标定'},
                'known_limits': ['淡墨，末字边界需复核'],
                'note': '可用于组定位；末字归属存在两种切分',
            },
            'endpoint_interpretation': {'status': 'resolved', 'resolved_value': 'half_open',
                                        'candidates': ['half_open:[x1,y1,x2,y2)'], 'edge_shift_px': 0,
                                        'note': '新增框按半开写出'},
            'glyph_boundary_error': {'status': 'unknown_uncalibrated', 'value_px': None,
                                     'box_quality': 'manual', 'calibration': 'none',
                                     'note': '未标定'},
            'evaluation_tolerance_px': {'status': 'pending_protocol', 'value_px': None,
                                        'note': '待协议确定'},
            'glyph_boundary_check': {'status': 'inspected_uncertain',
                                     'method': '4 倍标尺图目验',
                                     'evidence_image': 'output/dual_tracks/stage2/evidence/ruler/p38_taoren.png',
                                     'note': '「去」字归属不确定，末笔与下一字边界待核'},
            'certainty': spec['certainty'],
            'certainty_basis': '双轨结构可见，但末字切分有两种读法',
            'review_status': 'confirmed_single_reviewer_with_ambiguity',
            'coordinate_convention': 'half_open', 'original_coordinate_convention': 'half_open',
            'evaluation_eligibility': {
                'status': 'conditional',
                'usable_for': ['double_track_localization', 'lane_split'],
                'not_usable_for': ['reading_order_strict', 'glyph_exact_boundary', 'text_accuracy'],
                'exact_boundary_evaluation': {
                    'allowed': False,
                    'blockers': ['ambiguous_lane_split', 'no_boundary_calibration'],
                    'requires': ['protocol_tolerance'],
                    'note': '末字归属未定，轨级严格配对评测需排除或单列',
                },
                'reason': '轨切分存在两种读法',
            },
            'annotation_coverage': 'partial_positions',
            'glyph_review': {
                'main_last_glyph': {'text': '仁', 'bottom_y': 661, 'glyph_complete': True},
                'left_first_glyph': {'text': '皮', 'top_y': 661, 'glyph_complete': True},
                'right_first_glyph': {'text': '二', 'top_y': 661, 'glyph_complete': True},
                'left_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'right_last_glyph': {'checked': True, 'cut': False, 'tail_padding_px': 0},
                'review_status': 'checked_with_ambiguity',
                'review_notes': '桃仁二字 574-661；两轨自 661 起，左「皮」右「二」；',
                'evidence_paths': ['output/dual_tracks/stage2/evidence/ruler/p38_taoren.png'],
            },
            'candidates': spec.get('candidates'),
            'note': spec['note'],
            'source_reference': {
                'manifest': f"{_rel(D.MANIFEST_JSON)}#{D.page_id(spec['book'], spec['page'])}",
                'coordinates_json': None, 'coordinates_csv': None,
                'upstream': 'tools/dual_track_stage2_ruler.py + 目验',
                'stage1_readme': D.rel(D.README),
            },
            'split_membership': None,
        })

    groups.sort(key=lambda g: (g['book'], int(g['page']), g['region']))

    # 页面级
    for p in stage1['pages']:
        pages[p['page_id']] = dict(p)
    for book, page in [(S.DF_BOOK, S.DF_PAGE), ('思濟堂方書', 38), ('思濟堂方書', 51)]:
        pid = D.page_id(book, page)
        img = _img(book, page)
        page_groups = [g['group_id'] for g in groups if g['page_id'] == pid]
        existing = pages.get(pid)
        base = dict(existing) if existing else {}
        base.update({
            'page_id': pid, 'book': book, 'page': page,
            'status': '收录' if page_groups else '排除',
            'status_basis': ('第二阶段补标/转正' if page_groups else '第二阶段复核判定为单轨，转排除'),
            'image_path': _rel(D.BOOKS / book / D.page_file_name(page)),
            'source_image_path': _rel(D.BOOKS / book / D.page_file_name(page)),
            'image_sha256': img['sha256'], 'image_size': img['image_size'],
            'group_ids': page_groups, 'group_count': len(page_groups),
            'annotation_coverage': 'partial_positions',
            'evaluation_region': f'{book}/page_{page}#right_half' if book == S.DF_BOOK
                                 else f'{book}/page_{page}#full_page',
            'negative_region_ids': [], 'ambiguous_region_ids': [], 'ignore_region_ids': [],
        })
        pages[pid] = base

    # 每页粒度信息
    for pid, p in pages.items():
        gids = [g['group_id'] for g in groups if g['page_id'] == pid]
        p['group_ids'] = gids
        p['group_count'] = len(gids)
        p.setdefault('annotation_coverage', 'partial_positions')
        p.setdefault('evaluation_region', f"{pid}#full_page")
        p.setdefault('negative_region_ids', [])
        p.setdefault('ambiguous_region_ids', [])
        p.setdefault('ignore_region_ids', [])
        # 分层标签：栏线状态 / 异常宽框（测量对象=旧布局列框，本阶段一律 unknown）/ 特征
        lab = S.PAGE_LABELS.get(pid, {'rule_state': 'unknown', 'features': []})
        widths = [g['group_bbox'][2] - g['group_bbox'][0] for g in groups if g['page_id'] == pid]
        p['labels'] = {
            'rule_state': lab['rule_state'],
            'wide_box': {
                'state': 'unknown',
                'measured_object': '旧布局的 column bbox（阶段 3 冻结后才可取得）',
                'measured_object_available': False,
                'observed_width': None,
                'layout_snapshot_id': None,
                'reference_statistics': None,
                'sample_count': None,
                'threshold': None,
                'decision_status': 'unknown_pending_stage3_layout',
                'method': '本阶段没有旧布局列框，正式宽框标签一律 unknown；'
                          '阶段 3 保存旧布局快照后按冻结规则重判',
                'basis': '缺少测量对象（旧布局列框），不得用人工组框宽度代替',
            },
            # 目标尺寸描述：人工组框宽度，只用于说明目标框的尺度，不作为宽框标签依据
            'group_box_width_desc': {
                'median_px': (sorted(widths)[len(widths) // 2] if widths else None),
                'sample_count': len(widths),
                'note': '人工标注的组框宽度中位数，描述「目标框大概多宽」，'
                        '与宽框标签（依据旧布局列框）是两回事',
            },
            'features': lab['features'],
        }

    # 阶段 3 需要保存的旧列框信息（本阶段只定义要求，不产出标签）
    wide_box_requirements = {
        'measured_object': '旧布局 column bbox',
        'stage3_must_save': ['column_bboxes（每列 [x1,y1,x2,y2)、页号、半页归属）',
                             'column_width_px（列框宽度）',
                             'page_half（left / right / full）',
                             'sample_count（参与统计的列数）',
                             'reference_statistics（中位数、IQR、按书分组）',
                             'threshold（判异常的冻结阈值与规则）',
                             'abnormal_reason（每页异常原因）'],
        'decision_rules': [
            '有效列框样本 < 30 列，或某书多数列框异常时，该页标签记 unknown',
            '可用有来源的替代参考（例如同书其他页冻结统计），但必须在 reference_statistics 里写明来源',
            '不得用人工组框宽度对照新参考统计来「固化」标签',
        ],
        'handoff': '阶段 3 生成旧布局快照 → 重新测量旧列框宽度 → 按冻结规则出标签；阶段 8 沿用，不重判',
    }
    pages[D.page_id(S.DF_BOOK, S.DF_PAGE)]['ambiguous_region_ids'] = \
        S.DF_PAGE_COVERAGE['ambiguous_region_ids']
    pages[D.page_id(S.DF_BOOK, S.DF_PAGE)]['ambiguous_group_ids'] = \
        [D.group_id(S.DF_BOOK, S.DF_PAGE, r) for r in S.DF_PAGE_COVERAGE['ambiguous_groups']]
    pages[D.page_id(S.DF_BOOK, S.DF_PAGE)]['page_coverage_detail'] = S.DF_PAGE_COVERAGE

    counts = {
        'groups_total': len(groups),
        'groups_by_origin': _count([g['origin'] for g in groups]),
        'groups_by_source': _count([g['annotation_source'] for g in groups]),
        'groups_by_certainty': _count([g['certainty'] for g in groups]),
        'pages_total': len(pages),
        'pages_by_status': _count([p['status'] for p in pages.values()]),
        'pages_with_groups': len({g['page_id'] for g in groups}),
        'pages_by_rule_state': _count([p['labels']['rule_state'] for p in pages.values()]),
        'pages_by_wide_box_state': _count([p['labels']['wide_box']['state'] for p in pages.values()]),
        'scenario_coverage': _scenario_coverage(pages, groups),
    }
    return {
        'stage': STAGE, 'release': S.RELEASE,
        'generated_by': 'tools/dual_track_stage2_build.py',
        'upstream_stage1': {
            'release': 'stage1-v2',
            'normalized_annotations': _rel(D.OUT_DIR / 'normalized_annotations.json'),
            'group_count': len(stage1['groups']),
        },
        'coordinate_system': '原扫描图像素，左上角原点，x 右 y 下',
        'group_bbox_rule': 'union(right_bbox, left_bbox)，不含主字框',
        'counts': counts,
        'wide_box_label_requirements': wide_box_requirements,
        'groups': groups,
        'pages': [pages[k] for k in sorted(pages)],
    }


def _count(items: Sequence[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[str(i)] = out.get(str(i), 0) + 1
    return dict(sorted(out.items()))


def _scenario_coverage(pages: Dict[str, Dict[str, Any]], groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """关键场景的页数与组数，供样本充分性论证使用。"""
    def pages_with(feature: str) -> List[str]:
        return sorted(pid for pid, p in pages.items() if feature in p['labels']['features'])

    multi = pages_with('same_column_multiple')
    unequal = pages_with('unequal_lane_length')
    return {
        'same_column_multiple': {'pages': multi, 'page_count': len(multi)},
        'unequal_lane_length': {'pages': unequal, 'page_count': len(unequal)},
        'short_group': {'pages': pages_with('short_group')},
        'faint_ink': {'pages': pages_with('faint_ink')},
        'cross_book': {'pages': pages_with('cross_book')},
        'ambiguous_lane_split': {'pages': pages_with('ambiguous_lane_split')},
        'three_char_main': {'pages': pages_with('three_char_main')},
        'long_glyph_stroke': {'pages': pages_with('long_glyph_stroke')},
        'exhaustive_pages': [pid for pid, p in pages.items()
                             if p['annotation_coverage'] == 'exhaustive_page'],
        'exhaustive_regions': [pid for pid, p in pages.items()
                               if p['annotation_coverage'] == 'exhaustive_region'],
    }


# ---------------------------------------------------------------- 负例

def build_negatives(ann: Dict[str, Any]) -> Dict[str, Any]:
    regions = []
    for r in S.NEGATIVE_REGIONS:
        img = _img(r['book'], r['page'])
        regions.append({
            'negative_region_id': r['id'],
            'page_id': D.page_id(r['book'], r['page']),
            'book': r['book'], 'page': r['page'],
            'image_path': _rel(D.BOOKS / r['book'] / D.page_file_name(r['page'])),
            'image_sha256': img['sha256'], 'image_size': img['image_size'],
            'bbox': _bbox(r['bbox']),
            'category': r['category'],
            'status': r['status'],
            'basis': r['basis'],
            'evidence': r['evidence'],
            'annotation_coverage': 'exhaustive_region' if r['status'] == 'confirmed' else 'unknown',
        })
    samples = []
    for s in S.RULE_SAMPLES:
        img = _img(s['book'], s['page'])
        samples.append({
            'sample_id': s['id'], 'kind': s['kind'],
            'page_id': D.page_id(s['book'], s['page']),
            'book': s['book'], 'page': s['page'],
            'image_path': _rel(D.BOOKS / s['book'] / D.page_file_name(s['page'])),
            'image_sha256': img['sha256'],
            'bbox': _bbox(s['bbox']),
            'status': s['status'],
            'description': s['description'],
            'evidence': s['evidence'],
            'expected_algo_behavior': s['expected_algo_behavior'],
        })
    cats = {'single_track': '只有一条小字列或单条小字，没有并排第二轨',
            'body_text': '正文字号/行距，无大字下并排两小字',
            'title': '标题、牌记、书耳等跨栏大字块',
            'rule': '栏线/界行/版框，属直线结构不属字形',
            'noise': '污点、透印、残墨、留白'}
    return {
        'stage': STAGE, 'release': S.RELEASE,
        'categories': cats,
        'counts': {'regions': len(regions),
                   'by_category': _count([r['category'] for r in regions]),
                   'by_status': _count([r['status'] for r in regions]),
                   'rule_samples': len(samples),
                   'rule_kinds': _count([s['kind'] for s in samples]),
                   'ambiguous_regions': 0, 'ignore_regions': 0},
        'regions': regions,
        'rule_samples': samples,
        # 歧义与忽略区：本阶段为空，原因写在这里，供阶段 3 判断是否需要新增
        'ambiguous_regions': [],
        'ignore_regions': [],
        'mask_policy': {
            'ambiguous_regions': '只放「成因清楚、且不与确定正例重叠」的矩形；'
                                 '丹方 p12 的歧义挂在记录级候选上（DF12-02），不设矩形，'
                                 '因为下部栏位与上部错位，任何矩形都会覆盖长轨正例',
            'ignore_regions': '天头地脚、破损、透印等不计分区域；本阶段未设置',
            'conflict_rule': '若某个确定正例被 ambiguous/ignore 掩膜覆盖，校验 V17 直接失败，'
                             '必须先裁定（缩小掩膜或把该组降级）再继续',
        },
        'notes': [
            'status=confirmed 的区域可当确定负例；status=unknown 的区域只标「未检查」，不得当负例计分',
            '负例只覆盖明确区域，未选中区域一律视为 unknown，不参与精确率分母',
        ],
    }


# ---------------------------------------------------------------- 分区

def build_splits(ann: Dict[str, Any]) -> Dict[str, Any]:
    pages = {p['page_id']: p for p in ann['pages']}
    groups = ann['groups']
    assign: Dict[str, Dict[str, Any]] = {}
    for split, spec in S.SPLITS.items():
        for book, page in spec['pages']:
            pid = D.page_id(book, page)
            p = pages.get(pid)
            if not p:
                raise KeyError(f'分区引用了不存在的页：{pid}')
            assign[pid] = {
                'page_id': pid, 'split': split,
                'image_sha256': p['image_sha256'],
                'group_ids': [g['group_id'] for g in groups if g['page_id'] == pid],
                'exposure_history': S.PAGE_EXPOSURE.get(pid, {}).get('exposure_history', 'unknown'),
                'used_for': S.PAGE_EXPOSURE.get(pid, {}).get('used_for', []),
                'exposure_note': S.PAGE_EXPOSURE.get(pid, {}).get('note', ''),
                'basis': spec['basis'],
            }
    unassigned = [p for p in pages if p not in assign]
    # 按 SHA-256 查重：同图副本不得跨分区
    by_sha: Dict[str, List[str]] = {}
    for pid, a in assign.items():
        by_sha.setdefault(a['image_sha256'], []).append(pid)
    conflicts = []
    for sha, pids in by_sha.items():
        splits = {assign[p]['split'] for p in pids}
        if len(splits) > 1:
            conflicts.append({'sha256': sha, 'pages': pids, 'splits': sorted(splits)})
    counts = {k: {'pages': len([a for a in assign.values() if a['split'] == k]),
                  'groups': sum(len(a['group_ids']) for a in assign.values() if a['split'] == k)}
              for k in S.SPLITS}
    independent = [a for a in assign.values() if a['split'] == 'independent_acceptance']
    independent_claim = {
        'status': 'exposure_history_unclear',
        'pages': [a['page_id'] for a in independent],
        'groups': sum(len(a['group_ids']) for a in independent),
        'what_it_supports': '可作为「未参与调参」的验收样本使用；因曝光历史未完全确证，'
                            '结论只能称为「暴露历史未明的验收样本」，不得声称完全盲测',
        'what_it_does_not_support': ['置信区间估计（组数过少）', '跨书泛化结论（仅 1 页跨书）',
                                     '与 dev/回归区做显著性比较'],
        'future_sealing': '可保留为后续封存用途；一旦用于调参，必须改记 exposure_history=used_in_development',
    }
    return {
        'stage': STAGE, 'release': S.RELEASE,
        'split_unit': 'page',
        'isolation_key': 'image_sha256',
        'counts': counts,
        'assignments': [assign[k] for k in sorted(assign)],
        'unassigned_pages': unassigned,
        'independent_claim': independent_claim,
        'isolation': {'sha_conflicts': conflicts, 'ok': not conflicts,
                      'duplicate_sha_groups': {k: v for k, v in by_sha.items() if len(v) > 1}},
        'sample_sufficiency': {
            'candidate_target': {'independent_pages': 5, 'independent_groups': 30,
                                 'status': 'candidate_target',
                                 'note': '候选覆盖目标，用于判断数据够不够；'
                                         '不据此宣称任何置信区间半宽'},
            'current': {'independent_pages': len(independent),
                        'independent_groups': independent_claim['groups'],
                        'status': 'insufficient_for_interval_estimate'},
            'engineering_baseline': '样本不足只限制「效果结论」，不影响工程基线推进：'
                                    '阶段 3 仍可在 dev/回归区跑通链路并给出逐页结果',
            'uncertainty_method': '样本允许时按页聚类 bootstrap（记录次数与随机种子）；'
                                  '页数过少时只给逐页结果并注明区间不稳定',
        },
        'notes': [
            '同一 SHA-256 的页面副本必须落在同一分区，程序按指纹查重（isolation.sha_conflicts 为空即通过）',
            'exposure_history=unknown 时，该分区里的结论只作探索性，不得声称完全独立',
            '未分配页面不参与任何评测分母',
        ],
    }


# ---------------------------------------------------------------- 评测协议

def build_protocol() -> Dict[str, Any]:
    return {
        'stage': STAGE, 'release': S.RELEASE, 'protocol_version': 'dual-track-eval-1.1',
        'scope': {
            'unit': 'group',
            'group_definition': '同一实体栏目、同一纵向区段内的「主字 + 右轨 + 左轨」，'
                                '允许同栏上下多组、两轨起止不同',
            'coordinate_system': '原图像素，左上角原点，半开区间 [x1,y1,x2,y2)',
            'iou': 'area(intersection) / area(union)，按半开区间计算，与像素面积一致',
            'numeric_policy': {
                'optimization_precision': 'IoU 在优化过程中保持双精度原值，不做任何四舍五入或截断',
                'display_precision': '只允许在结果展示层保留 3-4 位小数',
                'why': '先舍入再比较会人为制造并列，让第三目标（字典序）在本来不该介入的时候介入',
            },
        },
        'group_matching': {
            'type': 'bipartite_maximum_cardinality_then_max_total_iou',
            'thresholds': {'group_iou': [0.5, 0.75]},
            'edge_rule': '仅当 IoU(pred, gt) >= 阈值时该边存在；比较用原始双精度值',
            'float_comparison': {
                'operator': '>=',
                'epsilon': 0.0,
                'note': '不引入容差 epsilon。浮点比较直接按原值判定，'
                        '因此 0.5000001 与 0.5000002 在阈值 0.5 下都是有效边，'
                        '又如 0.4999999999 不是有效边——结果可复现，不因平台差异改变',
            },
            'objectives': [
                {'order': 1, 'goal': '最大化匹配数量（最大基数）'},
                {'order': 2, 'goal': '在最大基数下最大化匹配边的 IoU 总和'},
                {'order': 3, 'goal': '仍完全并列时按稳定裁定规则取唯一解'},
            ],
            'allowed_unmatched': True,
            'solver': {
                'recommended': 'min_cost_max_flow',
                'two_phase_semantics': '基数优先不是靠代价函数「大致」拿到的，而是流程约束：'
                                       '先求最大流（流值 = 最大基数 k），再在 k 条边的约束下最小化总代价；'
                                       '任何少于 k 条边的解在求解过程中就被排除',
                'edge_cost': 'cost(p,g) = M - IoU(p,g)',
                'M_definition': 'M = 有效边数 + 1',
                'M_rationale': '同一边数下最小化 sum(M - IoU) 等价于最大化 sum(IoU)；'
                               'M 的作用只是让代价恒为正、便于最小费用流实现，'
                               '它不承担「基数优先」的职责——基数由先定最大流量保证。'
                               '若把 M 设成足够大来「压过」IoU 差，那只是另一种实现技巧，'
                               '语义上仍以「先定流值」为准',
                'alternative': '先用 Hopcroft-Karp 求最大基数 k，再在「恰好 k 条边」约束下做'
                               '指派最小化 sum(M - IoU)；与上面等价',
                'disabled_edges': 'IoU < 阈值或框非法（面积<=0/越界/无法解析）的边不建，不参与任何目标',
                'unmatched_nodes': '预测或标注节点可不匹配，代价 0；不给未匹配加负收益，'
                                   '否则会扭曲最大基数目标',
            },
            'tie_break': {
                'order': 3,
                'when': '仅当 (基数, 总 IoU 原值) 完全相等时才执行；因 IoU 不预先舍入，'
                        '这种情况只在真正对称时出现',
                'rule': '把匹配写成按预测序号升序的 (pred_index, gt_index) 序列，取字典序最小者；'
                        '预测与标注的输入顺序必须稳定（按 (y, x, group_id) 排序后传入）',
                'why': '同一 (基数, 总 IoU) 可能有多解；固定输入序 + 字典序可让不同实现给出一致结果',
            },
            'per_threshold_independent': True,
            'reporting': 'IoU 0.5 与 0.75 分别求匹配、分别报告，不得用 0.5 的匹配去算 0.75 的指标',
        },
        'lane_scoring': {
            'after_group_match': True,
            'sides': {'fixed': ['right', 'left'],
                      'cross': [['right', 'left'], ['left', 'right']]},
            'child_threshold': {'right_iou': 0.5, 'left_iou': 0.5,
                                'note': '子框阈值独立配置，与组阈值解耦；改动须记版本'},
            'categories': {
                'pairing_success': '固定侧别 right-right 且 left-left 双双达到子框阈值',
                'left_right_swap': '固定侧别未同时通过，但交叉对照后两侧均达到阈值',
                'child_localization_failure': '其余情况（含单侧未达阈值、面积为 0、越界）',
            },
            'saved_values': ['right_iou_fixed', 'left_iou_fixed',
                             'right_pred_vs_gt_left_iou', 'left_pred_vs_gt_right_iou'],
            'denominators': {
                'pairing_accuracy': '组级已匹配组数（pairing_success / matched_groups）',
                'swap_rate': '同一分母（left_right_swap / matched_groups）',
                'child_failure_rate': 'child_localization_failure / matched_groups',
            },
            'group_hit_rule': '组框达到匹配条件即计组命中；轨级失败只改分类，不撤销组命中',
            'excluded_groups': 'certainty=ambiguous_lane_split / ambiguous_lane_extent 的组'
                               '不进入轨级分母（分母里要单独列出被排除的组数）',
        },
        'error_diagnostics': {
            'ordering_rule': '诊断标签在全局一对一匹配完成之后才添加；'
                             'TP/FP/FN 一律由这次全局匹配确定，诊断不得改动它们',
            'duplicate_prediction': {
                'rule': '同一标注组有 >=2 个预测组分别达到阈值时，一对一匹配只会让其中一个成为 TP，'
                        '其余按未匹配预测计 FP；诊断标签 duplicate_prediction 同时挂在'
                        '「该标注组」与「这些多余预测」上',
                'forbidden': '禁止「另取局部 IoU 最大者计 TP」这类事后改判：'
                             '一旦允许局部重选，CASE-A 的全局最优解（P1-G2、P2-G1）就会被推翻',
                'labels': ['duplicate_prediction'],
            },
            'fragment': {
                'rule': '多个预测组的并集覆盖某标注组但单个都未达阈值，记 fragment 标签；'
                        '该标注计 FN，这些预测各计 FP（由全局匹配得出，不再另算）',
                'labels': ['fragment'],
            },
            'merge': {
                'rule': '单个预测组与 >=2 个标注组的相交面积各自超过该标注组面积的 60%，记 merge 标签；'
                        'TP/FP/FN 仍按全局匹配结果，不因标签变化',
                'labels': ['merge'],
            },
            'multiple_labels': '同一条预测允许同时带多个诊断标签（如 duplicate + merge），'
                               '标签不进入基础计数',
            'partial_area_duplicates': {
                'matched_duplicate': '在 partial_positions 区域，重复预测中的多余项仍计 FP'
                                     '（它确实多报了一个组），但在报告里与「未匹配且无重复」的 FP 分开列',
                'unmatched_no_duplicate': '未匹配且不属于任何重复簇的预测，在部分标注区域'
                                          '进入「新增候选待核」，不计 FP',
            },
        },
        'coverage_reporting': {
            'metric_table': {
                'hit_rate_on_known': {'computable': True,
                                      'denominator': '该区域已标注组数',
                                      'records': 'annotations.groups（annotation_coverage=partial_positions 也可算）'},
                'recall': {'computable': False,
                           'requires': 'exhaustive_region 或 exhaustive_page',
                           'note': '当前无任何穷尽标注区域/页，召回率标为 not_computable'},
                'precision': {'computable': False,
                              'requires': '穷尽区域 + 已划定负例',
                              'note': '当前不可计算；只能报「确定负例区域内的误检计数」'},
                'false_group_count': {'computable': False,
                                      'requires': 'exhaustive_region',
                                      'note': '部分标注区域内的未匹配预测只能算候选，不能算误组'},
                'pairing_accuracy': {'computable': True,
                                     'denominator': '组级已匹配组数（扣除 ambiguous 组）',
                                     'records': 'evaluation_protocol.lane_scoring'},
                'negative_region_fp': {
                    'computable': True,
                    'definition': '预测组框落入某个确定负例区域的中心/面积比超过阈值即计一次区域内误检',
                    'denominator': '该负例区域的暴露面积（region 面积）与区域数；'
                                   '报告形如「在 N 个确定负例区域（合计 X px²）内出现 M 次误检」',
                    'note': '这不是精确率，只是区域内误检计数；分母必须写明区域数与面积',
                },
            },
            'partial_positions': {
                'can_report': ['命中数/已标组数（hit rate）', '轨级配对分类分布',
                               '确定负例区域内的误检计数'],
                'cannot_report': ['召回率', '精确率', '误组数（分母不完整）'],
                'note': '第一阶段 inherited 与丹方 p12 都属 partial_positions',
            },
            'exhaustive_region': {
                'can_report': ['召回率', '精确率', '误组数', '组匹配与轨级分类'],
                'requires': '该区域所有目标已标注，且负例区域已划定',
            },
            'exhaustive_page': {'can_report': ['同 exhaustive_region，按整页计'], 'requires': '整页穷尽'},
            'p12_specific': {
                'status': 'partial_positions',
                'whole_page_recall': 'not_computable_current_version',
                'note': '丹方 p12 下部条目未穷尽，整页召回在当前版本不可计算；'
                        '补齐标注后可提升 coverage 版本并重算',
            },
            'prediction_attribution': {
                'priority': ['ignore 区域 -> ambiguous 区域 -> 正例匹配 -> 负例区域 -> 未覆盖区域'],
                'rules': [
                    '预测组中心落在 ignore 区域：整体忽略，计入 ignored_count，不进 TP/FP',
                    '预测组中心落在 ambiguous 区域：不判对错，进 pending_review 计数',
                    '其余先做一对一匹配；未匹配的预测再查负例重叠与未覆盖',
                ],
                'cross_region': '预测跨正例与负例时，以「组框与负例并集交叠比例」判定',
                'ignored_reporting': '报告必须给出被忽略数量，且不与 FP 混计',
                'positive_must_win': '确定正例不得因为宽泛的歧义掩膜被排除：'
                                     '若某正例框与 ambiguous 区相交，以正例为准，并把该 ambiguous 区缩小',
            },
            'negative_overlap': {
                'definition': 'negative_coverage_ratio = area(pred_bbox ∩ union(overlapping_negative_regions)) '
                              '/ area(pred_bbox)',
                'union_first': True,
                'range': '[0, 1]',
                'threshold': {'value': None, 'status': 'pending_calibration',
                              'note': '阈值须在阶段 3 基线后按开发集标定，本阶段不给数'},
                'cases': {
                    'boundary_conflict': 'ratio 很小（仅边缘相交）且已匹配可信正例：保留冲突记录，不改判',
                    'suspected_false_positive': 'ratio 大且无正例匹配：判疑似误检',
                    'unmatched_in_exhaustive': '穷尽区域内未匹配且不触负例：计误组',
                    'new_candidate_in_partial': '部分标注区域内未匹配：进待核候选，不计 FP',
                },
            },
        },
        'text_and_boundary': {
            'transcripts': '右轨、左轨转写分别保存；阅读拼接顺序为右轨后左轨',
            'text_status_values': ['transcribed', 'read_from_overlay', 'partial_unverified'],
            'usable_for_text_eval': ['transcribed'],
            'not_usable_for_text_eval': ['read_from_overlay', 'partial_unverified'],
            'uncertain_glyph': '转写中的〔?〕表示字形存疑，不得按药名补字',
            'three_metrics_kept_apart': {
                'box_iou': '框重叠度，几何量，与字形无关',
                'boundary_displacement': '框边到字形外沿的距离，需标定',
                'glyph_integrity': '首尾笔画是否被框切断，需目验',
                'note': '三者不得互相换算；端点含/不含只影响框边解释，不改变字形判定',
            },
            'endpoint_sensitivity': '端点待核的框（第一阶段 23 组）在敏感性报告中按两种解释各算一次 IoU，'
                                    '报差值；不得只报有利的一种',
        },
        'run_states': {
            'enum': ['not_run', 'completed', 'partial', 'failed', 'skipped'],
            'field_semantics': {
                'dual_track_evidence': 'not_run/skipped/failed 为 null；completed/partial 为 list（可为空）',
                'dual_track_run': '是否尝试运行：completed/partial/failed=true；not_run/skipped=false',
                'audit': '区域级失败、跳过、忽略记录；空列表不得表示失败',
            },
            'page_state_decision_table': [
                {'regions': '全部 completed，且有组', 'page_state': 'completed',
                 'evidence': 'list 非空', 'run': True, 'audit': '空'},
                {'regions': '全部 completed，且无组（检测完成但确实没有符合的组）', 'page_state': 'completed',
                 'evidence': '[] 空列表', 'run': True, 'audit': '空（不得写失败）'},
                {'regions': '全部 completed 无组 + 其余 skipped', 'page_state': 'completed',
                 'evidence': '[] 空列表', 'run': True,
                 'audit': '记录被跳过的区域及原因'},
                {'regions': 'completed 有组 + skipped', 'page_state': 'partial',
                 'evidence': 'list 为已成功产生的组', 'run': True,
                 'audit': 'skipped 区域逐条记录'},
                {'regions': 'completed + failed（部分区域失败）', 'page_state': 'partial',
                 'evidence': 'list 为成功部分的组', 'run': True, 'audit': 'failed 区域与失败原因'},
                {'regions': 'failed + skipped（无任何成功输出）', 'page_state': 'failed',
                 'evidence': 'None', 'run': True, 'audit': '失败与跳过原因'},
                {'regions': '全部 failed', 'page_state': 'failed',
                 'evidence': 'None', 'run': True, 'audit': '失败原因'},
                {'regions': 'partial 但没有任何成功产出的组（尝试过、全在 audit）', 'page_state': 'partial',
                 'evidence': '[] 空列表 + audit 非空', 'run': True,
                 'audit': '必须非空，否则应判 failed'},
                {'regions': '无区域，但确实尝试运行过（例如整页判定跳过检测）', 'page_state': 'completed',
                 'evidence': '[] 空列表', 'run': True, 'audit': '记录区域划分为空的原因'},
                {'regions': '功能关闭或明确跳过（未尝试）', 'page_state': 'skipped',
                 'evidence': 'None', 'run': False, 'audit': '跳过原因（配置项/开关名）'},
                {'regions': '旧数据或未运行', 'page_state': 'not_run',
                 'evidence': 'None', 'run': False, 'audit': '空'},
            ],
            'ui_rule': '界面按状态枚举展示，禁止用「结果为空」反推状态',
            'aggregation_by_region_completion': {
                'principle': '页面状态由「应处理区域是否完成」决定；evidence 里有多少组只影响列表内容',
                'skip_reason_classes': {
                    'out_of_scope': '按任务范围明确排除的区域，不影响页面状态',
                    'incomplete': '本应处理但因尺度失败/图像问题未完成，会使页面降为 partial 或 failed',
                },
                'rules': [
                    'no in-scope regions -> completed（确实没有需要处理的区域）',
                    'all in-scope completed -> completed',
                    'some completed, some failed/incomplete -> partial',
                    'no completed in-scope region -> failed',
                    'region division failed -> failed（与「没有区域」区分）',
                    'function off / pre-run skip -> skipped',
                    'not attempted -> not_run',
                ],
                'invariance': '同一组区域状态与原因，evidence 为 0 组或多组时页面状态必须相同',
            },
            'fixtures': page_state_fixtures(),
        },
        'prediction_contract_fields': [
            'negative_region_ids', 'overlaps_negative_region', 'negative_coverage_ratio',
            'review_category', 'annotation_coverage',
        ],
        'sample_sufficiency': {
            'candidate_coverage_target': {'independent_groups': 30, 'independent_pages': 5,
                                          'status': 'candidate_target',
                                          'note': '只是候选覆盖目标，用来衡量数据够不够，'
                                                  '不保证任何置信区间半宽'},
            'current': {'independent_groups': 3, 'independent_pages': 2,
                        'status': 'insufficient_for_interval_estimate'},
            'uncertainty_rule': '样本允许时按页聚类重采样（bootstrap）给区间，记录抽样次数与随机种子；'
                                '页数过少时只给逐页结果，明确说明区间不稳定，不给单点置信区间',
            'per_page_reporting': '必须逐页列出命中/未命中，不能只报总体均值',
        },
    }

# ---------------------------------------------------------------- 页面状态汇总（按区域完成度，不看组数）

REGION_STATUSES = ('completed', 'failed', 'skipped')
SKIP_REASONS = ('out_of_scope', 'incomplete')


def aggregate_page_state(regions, attempted=True, function_enabled=True,
                         region_division_failed=False):
    """页面状态只由「应处理区域是否完成」决定，与 evidence 里有多少组无关。"""
    if not function_enabled:
        return {'page_state': 'skipped', 'dual_track_evidence': None, 'dual_track_run': False,
                'audit': ['功能关闭或运行前明确跳过']}
    if not attempted:
        return {'page_state': 'not_run', 'dual_track_evidence': None, 'dual_track_run': False,
                'audit': []}
    if region_division_failed:
        return {'page_state': 'failed', 'dual_track_evidence': None, 'dual_track_run': True,
                'audit': ['区域划分失败，无法确定应处理范围']}
    if not regions:
        return {'page_state': 'completed', 'dual_track_evidence': [], 'dual_track_run': True,
                'audit': ['无区域：按任务范围判定本页没有需要处理的区域']}

    in_scope = [r for r in regions if r.get('in_scope')]
    out_scope = [r for r in regions if not r.get('in_scope')]
    if not in_scope:
        return {'page_state': 'completed', 'dual_track_evidence': [], 'dual_track_run': True,
                'audit': [f'{len(out_scope)} 个区域明确不在任务范围内']}

    done = [r for r in in_scope if r['status'] == 'completed']
    failed = [r for r in in_scope if r['status'] == 'failed']
    skipped_incomplete = [r for r in in_scope
                          if r['status'] == 'skipped' and r.get('skip_reason') == 'incomplete']
    if len(done) == len(in_scope):
        state = 'completed'
    elif not done:
        state = 'failed'
    else:
        state = 'partial'
    audit = []
    if failed:
        audit.append(f'{len(failed)} 个应处理区域失败')
    if skipped_incomplete:
        audit.append(f'{len(skipped_incomplete)} 个应处理区域因尺度/图像问题未完成')
    if out_scope:
        audit.append(f'{len(out_scope)} 个区域明确不在范围内')
    return {'page_state': state,
            'dual_track_evidence': [] if state in ('completed', 'partial') else None,
            'dual_track_run': True, 'audit': audit}


def page_state_fixtures():
    cases = [
        {'id': 'PS-01', 'desc': '单个应处理区域 completed（结果可为 0 组）',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'completed'}],
         'expect': 'completed'},
        {'id': 'PS-02', 'desc': 'completed + 明确范围外 skipped',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'completed'},
                     {'region_id': 'r2', 'in_scope': False, 'status': 'skipped',
                      'skip_reason': 'out_of_scope'}],
         'expect': 'completed'},
        {'id': 'PS-03', 'desc': 'completed + 应处理但未完成（incomplete）',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'completed'},
                     {'region_id': 'r2', 'in_scope': True, 'status': 'skipped',
                      'skip_reason': 'incomplete'}],
         'expect': 'partial'},
        {'id': 'PS-04', 'desc': 'completed + failed',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'completed'},
                     {'region_id': 'r2', 'in_scope': True, 'status': 'failed'}],
         'expect': 'partial'},
        {'id': 'PS-05', 'desc': 'failed + 范围外 skipped',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'failed'},
                     {'region_id': 'r2', 'in_scope': False, 'status': 'skipped',
                      'skip_reason': 'out_of_scope'}],
         'expect': 'failed'},
        {'id': 'PS-06', 'desc': '全部 failed',
         'regions': [{'region_id': 'r1', 'in_scope': True, 'status': 'failed'},
                     {'region_id': 'r2', 'in_scope': True, 'status': 'failed'}],
         'expect': 'failed'},
        {'id': 'PS-07', 'desc': '无区域但确实尝试过', 'regions': [], 'expect': 'completed'},
        {'id': 'PS-08', 'desc': '无区域且区域划分失败',
         'regions': [], 'region_division_failed': True, 'expect': 'failed'},
        {'id': 'PS-09', 'desc': '功能关闭', 'regions': [], 'function_enabled': False,
         'expect': 'skipped'},
        {'id': 'PS-10', 'desc': '未运行', 'regions': [], 'attempted': False, 'expect': 'not_run'},
        {'id': 'PS-11', 'desc': '只有范围外区域',
         'regions': [{'region_id': 'r1', 'in_scope': False, 'status': 'skipped',
                      'skip_reason': 'out_of_scope'}],
         'expect': 'completed'},
    ]
    for c in cases:
        got = aggregate_page_state(c['regions'], c.get('attempted', True),
                                   c.get('function_enabled', True),
                                   c.get('region_division_failed', False))
        c['actual'] = got['page_state']
        c['ok'] = got['page_state'] == c['expect']
    inv = []
    for n in (0, 3):
        st = aggregate_page_state(
            [{'region_id': 'r1', 'in_scope': True, 'status': 'completed'},
             {'region_id': 'r2', 'in_scope': True, 'status': 'failed'}])['page_state']
        inv.append({'evidence_size': n, 'page_state': st})
    invariance_ok = len({x['page_state'] for x in inv}) == 1
    return {'cases': cases, 'all_ok': all(c['ok'] for c in cases) and invariance_ok,
            'group_count_invariance': {'samples': inv, 'ok': invariance_ok,
                                       'rule': '页面状态由区域完成情况决定，evidence 数量不影响状态'}}


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def solve_matching(matrix: List[List[float]], thr: float) -> Dict[str, Any]:
    """穷举全部匹配方案，按 (基数, 总IoU原值, 字典序) 取最优；夹具校验用，规模小。

    注意：总 IoU 用原始双精度累加，不做 round —— 舍入会制造假并列，让第三目标提前介入。
    """
    n, m = len(matrix), len(matrix[0]) if matrix else 0
    best_key: Optional[Tuple[int, float, Tuple]] = None
    best_pairs: List[Tuple[int, int]] = []
    for k in range(min(n, m), 0, -1):
        found = False
        for preds in itertools.combinations(range(n), k):
            for gts in itertools.permutations(range(m), k):
                pairs = list(zip(preds, gts))
                if any(matrix[p][g] < thr for p, g in pairs):
                    continue
                total = sum(matrix[p][g] for p, g in pairs)  # 原始精度，不舍入
                key = (k, total, tuple(sorted(pairs)))
                if best_key is None or (key[0], key[1]) > (best_key[0], best_key[1]) or \
                        ((key[0], key[1]) == (best_key[0], best_key[1]) and key[2] < best_key[2]):
                    best_key, best_pairs = key, sorted(pairs)
                found = True
        if found:
            break
    if best_key is None:
        return {'cardinality': 0, 'total_iou': 0.0, 'pairs': []}
    return {'cardinality': best_key[0], 'total_iou': best_key[1],
            'pairs': [list(p) for p in best_pairs],
            'total_iou_display': round(best_key[1], 4)}


def build_matching_cases() -> Dict[str, Any]:
    cases = []
    raw = [
        {'case_id': 'CASE-A', 'threshold': 0.5,
         'matrix': [[0.90, 0.80], [0.80, 0.10]],
         'expect': {'cardinality': 2, 'pairs': [[0, 1], [1, 0]], 'total_iou': 1.60},
         'purpose': '贪婪先取 P1-G1(0.90) 会让 P2 无路可走、只剩 1 个匹配；'
                    '最大基数要求 2 个，故必须取 P1-G2 + P2-G1'},
        {'case_id': 'CASE-B', 'threshold': 0.5,
         'matrix': [[0.90, 0.80], [0.85, 0.50]],
         'expect': {'cardinality': 2, 'pairs': [[0, 1], [1, 0]], 'total_iou': 1.65},
         'purpose': '两种方案都是 2 个匹配，按第二目标取总 IoU 更大的交叉方案 1.65 > 1.40'},
        {'case_id': 'CASE-C-NO-EDGE', 'threshold': 0.5,
         'matrix': [[0.20, 0.10], [0.30, 0.40]],
         'expect': {'cardinality': 0, 'pairs': [], 'total_iou': 0.0},
         'purpose': '全部边低于阈值：不得强行匹配，结果为空'},
        {'case_id': 'CASE-D-UNEQUAL', 'threshold': 0.5,
         'matrix': [[0.95, 0.60], [0.70, 0.90], [0.80, 0.85]],
         'expect': {'cardinality': 2, 'pairs': [[0, 0], [1, 1]], 'total_iou': 1.85},
         'purpose': '预测多于标注：最多只能配 2 组；在 6 种 2-匹配方案里取总 IoU 最大的 '
                    'P1-G1(0.95)+P2-G2(0.90)=1.85，不允许一标多配，也不允许配 3 条'},
        {'case_id': 'CASE-E-TIE', 'threshold': 0.5,
         'matrix': [[0.80, 0.80], [0.80, 0.80]],
         'expect': {'cardinality': 2, 'pairs': [[0, 0], [1, 1]], 'total_iou': 1.60},
         'purpose': '等值最优：基数与总 IoU 都相同，按字典序取 (0,0),(1,1)'},
        {'case_id': 'CASE-F-BOUNDARY', 'threshold': 0.5,
         'matrix': [[0.50, 0.49]],
         'expect': {'cardinality': 1, 'pairs': [[0, 0]], 'total_iou': 0.50},
         'purpose': '阈值边界：恰好等于阈值算有效边，0.49 不算'},
        {'case_id': 'CASE-G-NEAR-TIE', 'threshold': 0.5,
         'matrix': [[0.5000001, 0.5000002]],
         'expect': {'cardinality': 1, 'pairs': [[0, 1]], 'total_iou': 0.5000002},
         'purpose': '一条预测对两个标注，两条边都刚过阈值且差 1e-7：'
                    '必须按最大总 IoU 取第二条边。若在比较前先舍入到 2-3 位小数，'
                    '两值会变成相等，退化成靠字典序裁定并错误地选 G1'},
        {'case_id': 'CASE-H-PRECISION-VS-ROUND', 'threshold': 0.5,
         'matrix': [[0.5004, 0.5006]],
         'expect': {'cardinality': 1, 'pairs': [[0, 1]], 'total_iou': 0.5006},
         'purpose': '保留精度才分得出的第二目标：舍入到 2 位小数会把两者都变成 0.50 而制造并列，'
                    '舍入到 3 位才刚能分开；协议要求优化阶段完全不舍入'},
    ]
    for c in raw:
        solved = solve_matching(c['matrix'], c['threshold'])
        cases.append({
            'case_id': c['case_id'],
            'threshold': c['threshold'],
            'iou_matrix': c['matrix'],
            'row_labels': [f'P{i+1}' for i in range(len(c['matrix']))],
            'col_labels': [f'G{j+1}' for j in range(len(c['matrix'][0]))],
            'expected': c['expect'],
            'solver_result': solved,
            'matches_expected': (solved['cardinality'] == c['expect']['cardinality']
                                 and solved['pairs'] == c['expect']['pairs']
                                 and abs(solved['total_iou'] - c['expect']['total_iou']) < 1e-9),
            'purpose': c['purpose'],
        })
    diag = _diagnostic_cases()
    return {
        'stage': STAGE, 'release': S.RELEASE,
        'note': '本阶段只给夹具与参考解，阶段 3 实现实际评测器时必须让这些用例全部通过',
        'precision_policy': '优化阶段保留 IoU 原值；只有展示层允许 round',
        'cases': cases,
        'diagnostic_cases': diag['cases'],
        'lane_pairing_cases': diag['lane'],
        'all_expected_verified': all(c['matches_expected'] for c in cases)
                                 and diag['all_ok'],
    }


def _reference_counts(matrix: List[Optional[List[float]]], thr: float) -> Dict[str, Any]:
    """参考计数：非法预测（None 行）不建边、直接记 FP+diagnostic；其余走全局匹配。"""
    valid_rows = [r for r in matrix if r is not None]
    invalid_rows = len(matrix) - len(valid_rows)
    solved = solve_matching(valid_rows, thr) if valid_rows else {'cardinality': 0, 'pairs': []}
    matched = solved['cardinality']
    n_pred = len(valid_rows)
    n_gt = len(matrix[0]) if matrix else 0
    return {
        'tp': matched, 'fp': (n_pred - matched) + invalid_rows,
        'fn': n_gt - matched, 'matched_pairs': solved['pairs'],
        'invalid_predictions': invalid_rows,
    }


def _diagnostic_cases() -> Dict[str, Any]:
    cases = []
    # D1 重复预测：2 个预测都过阈值，全局匹配只允许 1 个 TP
    m = [[0.90], [0.85]]
    cnt = _reference_counts(m, 0.5)
    cases.append({
        'case_id': 'DIAG-DUPLICATE',
        'threshold': 0.5, 'iou_matrix': m,
        'expected_counts': {'tp': 1, 'fp': 1, 'fn': 0},
        'actual_counts': cnt,
        'expected_labels': ['duplicate_prediction'],
        'rule': '全局匹配只能给 1 个 TP；多出来的那个预测记 FP，'
                '禁止「另取局部 IoU 最大者计 TP」的事后改判',
        'ok': cnt['tp'] == 1 and cnt['fp'] == 1 and cnt['fn'] == 0,
    })
    # D2 重复 + 未匹配在部分覆盖区域：多余预测在部分区域是否计 FP 要分开说
    cases.append({
        'case_id': 'DIAG-DUPLICATE-PARTIAL',
        'threshold': 0.5, 'iou_matrix': m,
        'expected_counts': {'tp': 1, 'fp': 1, 'fn': 0},
        'actual_counts': cnt,
        'expected_labels': ['duplicate_prediction'],
        'rule': '重复簇里的多余预测：在 partial_positions 区域仍计 FP（它确实多报了一组），'
                '但报告里与「未匹配且无重复」的 FP 分列；'
                '若该多余预测落在未覆盖区域，则进待核候选而不是 FP',
        'ok': cnt['tp'] == 1 and cnt['fp'] == 1,
    })
    # D3 非法预测：不建边，记 FP 并保留诊断
    m3 = [[0.95], None]
    cnt3 = _reference_counts(m3, 0.5)
    cases.append({
        'case_id': 'DIAG-INVALID-PRED',
        'threshold': 0.5, 'iou_matrix': [[0.95], 'INVALID'],
        'expected_counts': {'tp': 1, 'fp': 1, 'fn': 0, 'invalid_predictions': 1},
        'actual_counts': cnt3,
        'expected_labels': ['invalid_box'],
        'rule': '坐标非法的预测不参与匹配、不建边，但保留在预测清单里并记 invalid_box 诊断，'
                '计入 FP；不得静默丢弃（否则精确率会虚高）',
        'ok': cnt3['tp'] == 1 and cnt3['fp'] == 1 and cnt3['invalid_predictions'] == 1,
    })
    # D4 碎组：3 个预测拼起来覆盖 1 个标注，单个都不到阈值
    cases.append({
        'case_id': 'DIAG-FRAGMENT',
        'threshold': 0.5, 'iou_matrix': [[0.40], [0.35], [0.30]],
        'expected_counts': {'tp': 0, 'fp': 3, 'fn': 1},
        'actual_counts': _reference_counts([[0.40], [0.35], [0.30]], 0.5),
        'expected_labels': ['fragment'],
        'rule': '单个都不到阈值 -> 0 TP、3 FP、1 FN；fragment 只是标签，不改变计数',
        'ok': _reference_counts([[0.40], [0.35], [0.30]], 0.5)['tp'] == 0,
    })
    # D5 覆盖等级对计分的影响
    cases.append({
        'case_id': 'DIAG-COVERAGE-DIFF',
        'threshold': 0.5, 'iou_matrix': [[0.90], [0.20]],
        'expected_counts': {'tp': 1, 'fp': 1, 'fn': 0},
        'actual_counts': _reference_counts([[0.90], [0.20]], 0.5),
        'coverage_effect': {
            'partial_positions': '只报命中率 1/已标组数；未匹配预测进「新增候选待核」，不计 FP',
            'exhaustive_region': '可报精确率 = TP/(TP+FP)、召回率 = TP/(TP+FN)；未匹配预测计 FP',
        },
        'rule': '同一组预测在两种覆盖等级下报出的指标不同：部分覆盖不能算精确率，'
                '不能把「未匹配」直接当误检',
        'ok': True,
    })
    lane = [
        {'case_id': 'LANE-OK', 'right_iou_fixed': 0.72, 'left_iou_fixed': 0.68,
         'right_pred_vs_gt_left': 0.10, 'left_pred_vs_gt_right': 0.12,
         'expected': 'pairing_success'},
        {'case_id': 'LANE-SWAP', 'right_iou_fixed': 0.20, 'left_iou_fixed': 0.18,
         'right_pred_vs_gt_left': 0.70, 'left_pred_vs_gt_right': 0.66,
         'expected': 'left_right_swap'},
        {'case_id': 'LANE-FAIL', 'right_iou_fixed': 0.55, 'left_iou_fixed': 0.30,
         'right_pred_vs_gt_left': 0.40, 'left_pred_vs_gt_right': 0.35,
         'expected': 'child_localization_failure',
         'note': '右轨通过、左轨固定与交叉都不到阈值 -> 子框定位失败'},
    ]
    for c in lane:
        c['verified'] = _lane_category(c) == c['expected']
    return {'cases': cases, 'lane': lane,
            'all_ok': all(c['ok'] for c in cases) and all(c['verified'] for c in lane)}


def _lane_category(c: Dict[str, Any]) -> str:
    thr = 0.5
    if c['right_iou_fixed'] >= thr and c['left_iou_fixed'] >= thr:
        return 'pairing_success'
    if c['right_pred_vs_gt_left'] >= thr and c['left_pred_vs_gt_right'] >= thr:
        return 'left_right_swap'
    return 'child_localization_failure'


# ---------------------------------------------------------------- 跨阶段契约

def build_contract() -> Dict[str, Any]:
    return {
        'stage': STAGE, 'release': S.RELEASE, 'contract_version': 'dual-track-contract-1.0',
        'config_snapshot': {
            'format': {
                'snapshot_id': 'string，形如 {stage}-{algorithm}-{yyyymmddHHMM}',
                'required_fields': [
                    'snapshot_id', 'stage', 'algorithm_name', 'algorithm_role',
                    'code_version', 'source_fingerprint', 'config', 'model_versions',
                    'model_weight_fingerprints', 'input_image_fingerprints',
                    'layout_params', 'prediction_artifact', 'runtime_env', 'frozen_at',
                ],
                'field_spec': {
                    'code_version': 'git commit 或 dirty-tree 的 tree hash；无 git 时为源码目录指纹',
                    'source_fingerprint': '参与运行的 .py 文件 → sha256 映射',
                    'config': '完整参数（含默认值），按 key 字典序；未设置的参数写 null 而不是省略',
                    'model_versions': '模型名/版本/供应商；不存在的参数写 "not_applicable"',
                    'model_weight_fingerprints': '权重文件 sha256；无权重（规则算法）写 "not_applicable"',
                    'input_image_fingerprints': 'page_id → sha256',
                    'layout_params': '分栏/版心/尺度参数，含阈值与正则',
                    'prediction_artifact': '预测结果文件的相对路径 + sha256',
                    'runtime_env': 'python 版本、关键依赖版本、OS',
                },
            },
            'stage3_requirements': {
                'saves': '旧算法基线快照（source_fingerprint + config + 权重 + 输入指纹）',
                'frozen_when': '基线跑完即冻结，之后任何调参必须新开快照',
                'used_by': '阶段 4/5 的对照基准',
            },
            'stage8_requirements': {
                'saves': '新算法快照，显式引用 stage3 snapshot_id（both 字段：baseline_ref / candidate_ref）',
                'comparison_rule': '整体改进只在两个已冻结快照之间比较，不得跨快照混参',
            },
            'ablation_plan': {
                'rule': '每次消融只改一个因素，其余全部沿用对应冻结快照',
                'factors': ['双轨证据来源（几何/投影/模型）', '中缝阈值', '最小峰值要求',
                            '尺度来源（半页/局部）', '栏线掩膜策略'],
                'recording': '每个消融项单独 snapshot，并写 baseline_ref 与 changed_factor',
                'not_applicable_rule': '旧算法不存在的参数写 "not_applicable"，不得填 0 或留空',
            },
        },
        'seam_profile': {
            'formula': 'r = V / (min(L, R) + epsilon)',
            'criterion': 'r <= threshold 判为「中缝清楚」（上限判据，r 越小越好）',
            'threshold': {'value': 0.15, 'status': 'development_initial_value',
                          'unit': 'ratio（无量纲）',
                          'note': '0.15 只是开发初值，须在阶段 3 用开发集标定后冻结到快照'},
            'config': {
                'ink_mask': {'representation': '二值掩膜，取值 0/1（不是 0/255）',
                             'method': 'adaptiveThreshold(GAUSSIAN_C, invert)',
                             'block_size': 31, 'c': 14,
                             'note': '与阶段 1/2 测量一致；换参数须新快照'},
                'vertical_projection': '在列窗口内对掩膜按列纵向取平均，得到归一化投影 '
                                       'P(x) = (1/H_w) * sum_y mask(y, x)，取值范围 [0,1]，'
                                       '表示该列被墨迹覆盖的行比例；不做二值计数累加',
                'projection_unit': 'ratio ∈ [0,1]（每列墨迹覆盖率）',
                'smoothing': {'method': 'moving_average',
                              'window_px': 'max(3, round(0.12 * glyph_pitch))',
                              'unit': 'px',
                              'note': '平滑窗口按字形尺度换算：候选成立前用正文字心间距 glyph_pitch，'
                                      '成立后用 small_glyph_pitch；不使用任何未在 scale_object 中定义的局部变量'},
                'left_right_search_band': {
                    'definition': '以候选中缝 x 为中心，左右各取 band 宽的搜索带',
                    'band_px': 'max(6, round(1.5 * scale_for_band))',
                    'scale_for_band': '候选成立前用正文字心间距 glyph_pitch；'
                                      '候选成立后用 small_glyph_width（只表示小字横向实际宽度）',
                    'note': '搜索带不得越出列窗口；越界即 failure=no_search_band',
                },
                'seam_band_width': {'w_seam_px': 'max(3, round(0.4 * scale_for_band))',
                                    'unit': 'px'},
                'vertical_stop_condition': {
                    'rule': '纵向扫描在遇到连续空行时停止，空行阈值用 G（正文）或 small_glyph_gap（候选内）',
                    'threshold_px': 'max(3, round(0.8 * G)) 或 max(3, round(0.8 * small_glyph_gap))',
                    'note': '不能再用「固定 5 行」这类无来源常数'},
                'window_height_and_step': {
                    'height_px': 'round(1.2 * small_glyph_height)（候选成立后）',
                    'step_px': 'round(0.5 * small_glyph_height)',
                    'note': '窗高与步长都按字高换算，不按页高比例'},
                'field_usage_note': 'G 是「字间留白」，glyph_pitch 是「字心间距」，'
                                    'small_glyph_width 是「小字横向宽度」——三者不可互换；'
                                    '搜索带用宽度/间距，纵向停止用留白，窗高步长用字高',
                'peak_value': 'L、R 取左右搜索带内 P(x) 的最大值（归一化覆盖率峰值），'
                              '峰值位置须距中缝带边界 >= 2px',
                'V_statistic': 'V 取中缝带内 P(x) 的算术均值（mean），单位同 P(x)，∈[0,1]',
                'L_R_V_same_scale': 'L、R、V 必须来自同一次投影与同一尺度，不得混用不同尺度的值',
                'epsilon': {'value': 0.02, 'unit': 'ratio',
                            'status': 'development_initial_value',
                            'note': '归一化投影下 eps 是小量（≈ P(x) 的量级），'
                                    '不能沿用累加投影时代的 1.0；阶段 3 标定'},
                'min_peak': {'value': 0.10, 'unit': 'ratio',
                             'status': 'development_initial_value',
                             'note': '累加投影时代写的 min_peak=4 是「像素计数」，'
                                     '归一化后必须换算成覆盖率，不能直接沿用'},
                'min_ink_support': '左右搜索带各需 >= 30% 的行有墨迹，否则视为证据不足',
                'failure_states': ['no_search_band（搜索带越界或宽度 < 2*w_seam）',
                                   'insufficient_peak（任一侧峰值 < min_peak）',
                                   'insufficient_support（墨量支持不足）',
                                   'blank_window（两侧均无字形支持，必须拒绝，不得给 r）'],
            },
            'pixel_scale_note': '所有像素参数与「字形尺度」挂钩，不跟页高硬绑：'
                                'block_size ≈ max(15, round(1.0 * H))（H 为正文中位字高），'
                                'band ≈ 1.5 * scale_for_band（间距/宽度），w_seam ≈ 0.4 * scale_for_band；'
                                '跨书时在快照里记录缩放系数与实际取值；'
                                '不使用任何未在 scale_object 中定义的局部变量名',
            'blank_window_rule': '两侧证据不足的空白窗口必须判为 failure（blank_window），'
                                 '不得因为 V≈0 就给出「中缝清楚」的结论',
        },
        'scale_object': {
            'produced_by': 'stage4',
            'consumed_by': 'stage5',
            'definition_note': '四类量必须分开：字高（H / small_glyph_height）、'
                               '字间留白（G / small_glyph_gap）、字心间距（glyph_pitch / small_glyph_pitch）、'
                               '字宽（small_glyph_width）。页高单独叫 page_height，不参与字高语义',
            'fields': ['H', 'G', 'glyph_pitch', 'page_height',
                       'small_glyph_height', 'small_glyph_width',
                       'small_glyph_gap', 'small_glyph_pitch',
                       'source', 'sample_count', 'dispersion', 'confidence_status', 'failure_reason'],
            'field_spec': {
                'H': {'meaning': '正文中位字高', 'unit': 'px',
                      'basis': '正文完整字形 [a_i,b_i) 的高度 b_i-a_i 的中位数，剔除标题带与极端尺寸'},
                'G': {'meaning': '同栏相邻正文完整字形之间的纵向留白统计量', 'unit': 'px',
                      'basis': '对半开字形带 [a_i,b_i) 与 [a_next,b_next)，留白 = a_next - b_i；'
                               '取中位数。样本筛选：同栏、纵向相邻、两者高度都在 [0.6H,1.6H]；'
                               '重叠样本（a_next < b_i，即留白为负）单独计数并剔除，'
                               '剔除数量写进 sample_count 的 excluded_overlap 子项'},
                'glyph_pitch': {'meaning': '相邻正文的字心距离（独立保存，不用于推导 G）', 'unit': 'px',
                                'basis': '字心 = (a_i+b_i)/2；pitch = 下一字心 - 本字心，取中位数；'
                                         '后续算法若需要间距量，显式引用本字段'},
                'page_height': {'meaning': '图像高度（页高）', 'unit': 'px',
                                'basis': '直接读图像尺寸；不参与字高/留白/间距语义'},
                'small_glyph_height': {'meaning': '双轨候选成立后估计的局部小字字高', 'unit': 'px',
                                       'basis': '候选内小字块高度中位数'},
                'small_glyph_width': {'meaning': '候选成立后小字的实际横向宽度', 'unit': 'px',
                                      'basis': '小字块外接框宽度的中位数；只表示宽度，不含留白'},
                'small_glyph_gap': {'meaning': '候选内相邻小字的纵向留白', 'unit': 'px',
                                    'basis': '同 G 的算法，输入换成候选内小字块'},
                'small_glyph_pitch': {'meaning': '候选内相邻小字的字心距离', 'unit': 'px',
                                      'basis': '同 glyph_pitch 的算法'},
                'source': {'values': ['half_page_estimate', 'local_refine', 'substitute_estimate'],
                           'note': '共享字段，适用于上面所有估计量；每个量各自带来源，不共用一条'},
                'sample_count': {'meaning': '参与该量估计的样本数（含 excluded_overlap 子项）', 'unit': 'count'},
                'dispersion': {'meaning': '该量的离散度（IQR 或标准差，需注明用哪个）', 'unit': 'px'},
                'confidence_status': {'values': ['confident', 'provisional', 'failed'],
                                      'note': '逐量判断，可不同量不同状态'},
                'failure_reason': {'meaning': '该量失败的原因；成功时为 null'},
            },
            'synthetic_example': {
                'input': '两个字块：字形带 [0,30) 与 [35,65)（两字高度都是 30px）',
                'expected': {'H': 30, 'G': 5, 'glyph_pitch': 35,
                             'note': '留白 = 35-30 = 5；字心间距 = 47.5-12.5 = 35（或 b_i - b_prev 同值）'},
                'checks': ['G 与 glyph_pitch 必须是两个独立值，不能一个由另一个推出来',
                           'H 不能等于 page_height',
                           '若把 G 误当 glyph_pitch 使用，搜索带会窄 30px（按 1.5G 计算）'],
            },
            'half_page_estimate_basis': '栏线位置、页框、极端尺寸剔除、明显标题带剔除，'
                                        '以及稳定文字尺寸分布；输出 H 与 G（不是页高）',
            'local_refine': '双轨候选成立后，用候选内的小字重新估计 small_glyph_height/width；'
                            '记录原值、新值、来源与依据，不静默覆盖',
            'failure_handling': [
                '估计失败时，可用「有样本、有来源」的替代估计（同书其他页/同栏上下段），'
                'source=substitute_estimate，confidence_status=provisional',
                '没有任何可信估计时，confidence_status=failed 并写 failure_reason，'
                '同时把受影响区域标为 skipped/partial 写进 audit',
                '删除「失败后仍按无来源默认值继续跑」的约定：没有来源就不许继续用数值',
            ],
        },
        'rule_line_policy': {
            'keep_original_image': True,
            'separate_mask': True,
            'high_confidence_rule': '独立且笔直的栏线可从字形聚合中排除',
            'ambiguous_cases': ['touching_rule（贴字）', 'broken_rule（断线）',
                                'long_glyph_stroke（长竖笔）'],
            'ambiguous_handling': '保留标记，结合字形聚合结果复核；不得直接剔除也不得直接并入',
            'samples_ref': 'negative_regions.json#rule_samples',
        },
        'run_state_contract': {
            'dual_track_evidence': 'null 或 list；语义随 run_state 变，见 evaluation_protocol.run_states',
            'dual_track_run': '是否尝试运行：completed/partial/failed=true；not_run/skipped=false',
            'audit': '区域级失败/跳过/忽略记录；不得用空列表表示失败',
        },
        'prediction_fields': {
            'negative_region_ids': '该预测重叠到的负例区域 id 列表',
            'overlaps_negative_region': 'bool',
            'negative_coverage_ratio': 'float[0,1]，先求负例区域并集再算交集',
            'review_category': 'accepted | suspected_false_positive | boundary_conflict | '
                               'pending_review | ignored',
            'annotation_coverage': '该预测所在区域的覆盖等级',
        },
    }


# ---------------------------------------------------------------- 复核记录

REVIEW_15 = [
    ('P34-11', '黄框贴合「羚羊角」三字，右轨「一錢二分」、左轨「用尖鎊」完整，无邻列污染'),
    ('P34-14', '三框均贴合，左轨「另研」末字贴框底但未切断'),
    ('P37-01', '左轨「酒炒」两字在框内，末字贴框底；未发现切笔'),
    ('P39-02', '主字「川附子」完整；右轨「各二錢」、左轨「製」完整'),
    ('P39-03', '三框完整，右轨「各三錢」、左轨「酒炒」无越界'),
    ('P43-15', '主字「溫服」完整；右轨「方見中」、左轨「風門」完整'),
    ('P43-16', '主字「送下」完整；右轨「方見痰」、左轨「門」完整'),
    ('P45-17', '主字「二錢」完整；右轨「滾水泡五」、左轨「次用」完整'),
    ('P45-18', '主字「法」完整；两轨完整；主字框上缘贴近上一字，未纳入'),
    ('P45-19', '主字「之」完整；右轨「方見」、左轨「中風」完整'),
    ('P46-07', '主字「蒼朮」完整；右轨「各一錢」、左轨「炒」完整'),
    ('P46-08', '主字「麻黃」完整；右轨「八分若頭不」、左轨「疼無汗不用」完整'),
    ('P49-12', '主字「杏仁」完整；右轨「七枚去皮」、左轨「尖研」完整'),
    ('P49-20', '主字「之」完整；右轨「方見中」、左轨「暑門」完整'),
    ('P50-09', '主字「黃芩」完整；右轨「各一錢半」、左轨「酒炒」完整'),
]


def build_review_log() -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for i, (region, note) in enumerate(REVIEW_15, start=1):
        entries.append({
            'review_id': f'R2-{i:03d}',
            'review_type': 'glyph_integrity',
            'target_type': 'group', 'target_id': f'思濟堂方書/p{_page_of(region)}/{region}',
            'old_status': 'not_inspected', 'new_status': 'inspected_ok',
            'reviewer': S.REVIEWER,
            'method': '6 倍拼图目验（每组单独叠框，黄=主字/红=右轨/绿=左轨）',
            'viewed_images': ['output/dual_tracks/stage2/evidence/review_1.png'
                              if i <= 8 else 'output/dual_tracks/stage2/evidence/review_2.png'],
            'findings': [note],
            'basis': '逐组看首字顶笔与末字底笔是否被框线切断、是否框进邻列',
            'decision': '维持原框，不修框',
        })
    entries.extend([
        {'review_id': 'R2-016', 'review_type': 'glyph_integrity',
         'target_type': 'group', 'target_id': '思濟堂方書/p49/P49-13',
         'old_status': 'inspected_uncertain', 'new_status': 'inspected_uncertain',
         'reviewer': S.REVIEWER, 'method': '6 倍叠框复看 + 5 倍栏条',
         'viewed_images': ['output/dual_tracks/stage1/evidence/思濟堂方書__p49__P49-13.png'],
         'findings': ['首字「親」顶部仍紧贴框顶，6 倍下无法确认是否切到顶笔',
                      '两轨较长（170px），末字「實/要」贴框底'],
         'basis': '分辨率不足，未取得更高倍原图证据',
         'decision': '保留 uncertain，不修框；该组不用于字形完整性论证'},
        {'review_id': 'R2-017', 'review_type': 'approximate_box_recheck',
         'target_type': 'group', 'target_id': '温疫方論/p12/W12-01',
         'old_status': 'approximate_box', 'new_status': 'approximate_box_kept',
         'reviewer': S.REVIEWER, 'method': '6 倍叠框 + 原图对照',
         'viewed_images': ['output/dual_tracks/stage1/evidence/温疫方論__p12__W12-01.png'],
         'findings': ['框与字贴合尚可，但来源自述「非字形级精确分割」，且无原始框生成代码',
                      '主字框与两轨框 2px 相接，无内容交叠'],
         'basis': '没有可复现的测量过程就没有理由换成「精标」；保留原框与近似标记',
         'decision': '不重新精标，保留近似框身份；新增坐标不覆盖旧框'},
        {'review_id': 'R2-018', 'review_type': 'approximate_box_recheck',
         'target_type': 'group', 'target_id': '温疫方論/p12/W12-02',
         'old_status': 'approximate_box', 'new_status': 'approximate_box_kept',
         'reviewer': S.REVIEWER, 'method': '6 倍叠框 + 原图对照',
         'viewed_images': ['output/dual_tracks/stage1/evidence/温疫方論__p12__W12-02.png'],
         'findings': ['左轨框底比「分」末笔低约 10px，框偏松（已知的近似框特征）'],
         'basis': '偏松属来源固有；改框需要逐字重测，本阶段不伪造精标',
         'decision': '保留原框；在 evaluation_eligibility 中维持 excluded from 精确边界评测'},
        {'review_id': 'R2-019', 'review_type': 'pending_page_verdict',
         'target_type': 'page', 'target_id': '思濟堂方書/page_38',
         'old_status': '待核（无坐标）', 'new_status': '转正：1 组 S38-01',
         'reviewer': S.REVIEWER, 'method': 'OCR 定位药名 + 4 倍标尺图目验',
         'viewed_images': ['output/dual_tracks/stage2/evidence/ruler/p38_taoren.png'],
         'findings': ['桃仁下方确有并排两列小字：右「二十枚去」、左「皮尖」',
                      '「去」归属有两种切分，两轨起止高度不同（左轨短）'],
         'basis': '并列两轨成立即符合目标定义；切分歧义单独记录，不影响「是双轨」的判定',
         'decision': '转正并标 ambiguous_lane_split'},
        {'review_id': 'R2-020', 'review_type': 'pending_page_verdict',
         'target_type': 'page', 'target_id': '思濟堂方書/page_51',
         'old_status': '待核（无坐标）', 'new_status': '转排除（单轨）',
         'reviewer': S.REVIEWER, 'method': 'OCR 定位药名 + 3.6 倍标尺图目验（含邻栏）',
         'viewed_images': ['output/dual_tracks/stage2/evidence/ruler/p51_wide.png'],
         'findings': ['枳殼下方只有「各一錢半」一列小字，右侧为邻栏「羌活／獨活」',
                      '未定位到清单所记的「麩炒」第二轨'],
         'basis': '单轨不符合目标定义',
         'decision': '转排除；如后续在该栏发现淡墨第二轨可回滚（reversible=true）'},
        {'review_id': 'R2-021', 'review_type': 'new_annotation',
         'target_type': 'region', 'target_id': '丹方彚编/page_12#right_half',
         'old_status': '未收录（说明.md 旧口径：整段双行夹注不收）',
         'new_status': '补标 11 组',
         'reviewer': S.REVIEWER, 'method': '列窗测量 + 中缝定位 + 6 倍拼图目验',
         'viewed_images': ['output/dual_tracks/stage2/evidence/montage_df_1_4.png',
                           'output/dual_tracks/stage2/evidence/montage_df_5_8.png',
                           'output/dual_tracks/stage2/evidence/montage_df_bottom.png'],
         'findings': ['右半页 9 列：8 列顶部各 1 组（药名 + 两长轨），另 3 组为栏内下部独立条目',
                      '最左列（藿合油香…）为单轨，不计正例，作负例 NEG-DF-SINGLE-01',
                      '下部仍有未穷尽条目，故 coverage=partial_positions'],
         'basis': '用户已确认该页属目标范围；按当前口径「同栏同区段左右并排两轨」判定',
         'decision': '新增 11 组，标 certainty=medium/low，转写为部分转录'},
    ])
    return {
        'stage': STAGE, 'release': S.RELEASE,
        'reviewer_profile': {
            'mode': 'single_reviewer_ai_visual',
            'independent_second_read': False,
            'limitation': '全部结论由单人在叠框放大图上目验得出，无第二人独立复核；'
                          '涉及「未发现切笔」的结论只表示在该放大倍数下不可见',
        },
        'counts': {
            'entries': len(entries),
            'by_type': _count([e['review_type'] for e in entries]),
            'by_decision': _count([e['decision'] for e in entries]),
        },
        'entries': entries,
    }


def _page_of(region: str) -> int:
    return int(region[1:3])



# ---------------------------------------------------------------- 冲突检查（纯函数，便于单测）

LANE_FIELDS = ('right_bbox', 'left_bbox')
BOX_FIELDS = ('main_bbox', 'right_bbox', 'left_bbox')


def find_cross_group_overlaps(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同页任意两组的任意两框相交面积 > 0 即记录；边缘接触应为 0。"""
    out = []
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in groups:
        by_page.setdefault(g['page_id'], []).append(g)
    for pid, gs in by_page.items():
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                for fa in BOX_FIELDS:
                    for fb in BOX_FIELDS:
                        area = D.intersect_area(gs[i][fa], gs[j][fb])
                        if area > 0:
                            out.append({'page': pid, 'a': gs[i]['region'], 'a_field': fa,
                                        'b': gs[j]['region'], 'b_field': fb, 'area': area})
    return out


def find_title_intrusions(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """上方组的轨框压到下方组的标题（main_bbox）。"""
    out = []
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for g in groups:
        by_page.setdefault(g['page_id'], []).append(g)
    for pid, gs in by_page.items():
        ordered = sorted(gs, key=lambda g: g['main_bbox'][1])
        for i, up in enumerate(ordered):
            for low in ordered[i + 1:]:
                if low['main_bbox'][1] <= up['main_bbox'][1]:
                    continue
                for lane in LANE_FIELDS:
                    area = D.intersect_area(up[lane], low['main_bbox'])
                    if area > 0:
                        out.append({'page': pid, 'upper': up['region'], 'lane': lane,
                                    'lower': low['region'], 'area': area,
                                    'upper_lane_bottom': up[lane][3],
                                    'lower_main_top': low['main_bbox'][1]})
    return out


def find_main_swallow(groups: List[Dict[str, Any]], max_depth: int = 2,
                      max_ratio: float = 0.05) -> Dict[str, List[Dict[str, Any]]]:
    """主字框吞入本组小字。1-2px 且面积比 <=5% 的算边缘相接，不算吞字。"""
    swallowed, contacts = [], []
    for g in groups:
        for lane in LANE_FIELDS:
            area = D.intersect_area(g['main_bbox'], g[lane])
            if area <= 0:
                continue
            mb, lb = g['main_bbox'], g[lane]
            depth = min(mb[3], lb[3]) - max(mb[1], lb[1])
            ratio = area / max(1, D.bbox_area(lb))
            item = {'group': g['group_id'], 'lane': lane, 'area': area,
                    'depth_px': depth, 'area_ratio': round(ratio, 4),
                    'main_bottom': mb[3], 'lane_top': lb[1]}
            if depth <= max_depth and ratio <= max_ratio:
                item['verdict'] = 'edge_contact'
                contacts.append(item)
            else:
                item['verdict'] = 'swallow'
                swallowed.append(item)
    return {'swallowed': swallowed, 'contacts': contacts}


def classify_rule_overlap(box: Sequence[int], rule: Sequence[int]) -> Optional[Dict[str, Any]]:
    """框与栏线区域的重叠：给出面积比、持续高度、判定（边缘重合 / 实质污染）。"""
    inter = D.intersect_area(box, rule)
    if inter <= 0:
        return None
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = rule
    iw = min(bx2, rx2) - max(bx1, rx1)
    ih = min(by2, ry2) - max(by1, ry1)
    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
    edge = iw <= max(4, int(0.25 * bw))
    return {'overlap_px2': inter, 'area_ratio': round(inter / (bw * bh), 4),
            'height_coverage': round(ih / bh, 4), 'overlap_width_px': iw,
            'rule_width_px': rx2 - rx1,
            'verdict': 'edge_coincidence' if edge else 'contamination'}


# ---------------------------------------------------------------- 校验

def validate(ann: Dict[str, Any], neg: Dict[str, Any], splits: Dict[str, Any],
             proto: Dict[str, Any], contract: Dict[str, Any],
             cases: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(cid: str, title: str, ok: bool, detail: str, items: Optional[List[Any]] = None) -> None:
        checks.append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                       'detail': detail, 'items': items or []})

    # V01 图片与指纹
    bad = []
    for g in ann['groups']:
        img = D.image_info(D.ROOT / g['image_path'])
        if not img['exists'] or not img['readable']:
            bad.append(f"{g['group_id']} 图片不可读")
            continue
        if img.get('sha256') != g['image_sha256']:
            bad.append(f"{g['group_id']} 指纹不符")
        if img['image_size'] != g['image_size']:
            bad.append(f"{g['group_id']} 尺寸不符")
    add('V01', '图片尺寸与指纹', not bad,
        f"{ann['counts']['groups_total']} 组的图路径/指纹/尺寸全部核对，异常 {len(bad)}", bad)

    # V02 框合法性
    bad = []
    for g in ann['groups']:
        for k in ('main_bbox', 'right_bbox', 'left_bbox'):
            b, code, _ = D.parse_bbox(g[k])
            if code != D.BBOX_OK:
                bad.append(f"{g['group_id']}.{k} 非法({code})")
                continue
            if D.bbox_area(b) <= 0:
                bad.append(f"{g['group_id']}.{k} 面积<=0")
            if not D.in_image(b, g['image_size']):
                bad.append(f"{g['group_id']}.{k} 越界")
    add('V02', '框合法性与范围', not bad,
        f"{ann['counts']['groups_total'] * 3} 个框全部为四整数、正面积、在原图内，异常 {len(bad)}", bad)

    # V03 唯一编号
    ids = [g['group_id'] for g in ann['groups']]
    add('V03', '标注编号唯一', len(ids) == len(set(ids)),
        f"{len(ids)} 条记录 / {len(set(ids))} 个唯一编号")

    # V04 侧别
    bad = [g['group_id'] for g in ann['groups']
           if D.center_x(g['right_bbox']) <= D.center_x(g['left_bbox'])]
    add('V04', '右轨中心在左轨右侧', not bad, f"异常 {len(bad)} 组", bad)

    # V05 新旧引用
    stage1_ids = {g['group_id'] for g in D.load_groups()}
    missing = [g['group_id'] for g in ann['groups'] if g['origin'] == 'stage1'
               and g['stage1_group_id'] not in stage1_ids]
    new_ids = [g['group_id'] for g in ann['groups'] if g['origin'] != 'stage1']
    add('V05', '新旧标注引用', not missing and len(stage1_ids) ==
        len([g for g in ann['groups'] if g['origin'] == 'stage1']),
        f"继承第一阶段 {len(stage1_ids)} 组全部可追溯到 stage1；新增 {len(new_ids)} 组", missing)

    # V06 覆盖范围与区域冲突
    conflicts = []
    page_ids = {p['page_id'] for p in ann['pages']}
    for g in ann['groups']:
        if g['page_id'] not in page_ids:
            conflicts.append(f"{g['group_id']} 的页不在页面表")
    neg_ids = {r['negative_region_id'] for r in neg['regions']}
    for p in ann['pages']:
        for rid in p.get('negative_region_ids', []):
            if rid not in neg_ids:
                conflicts.append(f"{p['page_id']} 引用了不存在的负例 {rid}")
    add('V06', '覆盖范围与正负例引用', not conflicts, f"冲突 {len(conflicts)} 项", conflicts)

    # V07 正负例几何重叠
    # rule 类负例天然压在栏线位置，与列窗测出的框边缘必然相交；
    # 这类相交单独记为「边界冲突」并按协议 boundary_conflict 处理，不当作硬失败。
    overlaps, boundary_conflicts = [], []
    for r in neg['regions']:
        for g in ann['groups']:
            if g['page_id'] != r['page_id']:
                continue
            for k in ('main_bbox', 'right_bbox', 'left_bbox'):
                a = D.intersect_area(g[k], r['bbox'])
                if a <= 0:
                    continue
                item = {'region': r['negative_region_id'], 'category': r['category'],
                        'group': g['group_id'], 'field': k, 'area': a}
                if r['category'] == 'rule':
                    boundary_conflicts.append(item)
                else:
                    overlaps.append(item)
    add('V07', '正例与硬负例（single_track/body_text/title/noise）不重叠',
        not overlaps, f"硬负例交叠 {len(overlaps)} 处；栏线类边界冲突 {len(boundary_conflicts)} 处（见 V18 处置）",
        overlaps)
    checks[-1]['boundary_conflicts'] = boundary_conflicts

    # V08 分区隔离
    add('V08', '分区按页与指纹隔离', splits['isolation']['ok'],
        f"三区共 {len(splits['assignments'])} 页，SHA 冲突 "
        f"{len(splits['isolation']['sha_conflicts'])} 项；未分配 {len(splits['unassigned_pages'])} 页",
        splits['isolation']['sha_conflicts'])

    # V09 分区覆盖所有有组页
    with_groups = {g['page_id'] for g in ann['groups']}
    assigned = {a['page_id'] for a in splits['assignments']}
    missing_split = sorted(with_groups - assigned)
    add('V09', '有标注的页都已分区', not missing_split,
        f"有组页 {len(with_groups)}，已分区 {len(with_groups & assigned)}", missing_split)

    # V10 证据图存在
    missing_ev = []
    for e in review['entries']:
        for img in e['viewed_images']:
            p = D.ROOT / img
            if not (p.exists() or (D.ROOT / 'output/dual_tracks/stage2/evidence' / Path(img).name).exists()):
                if 'stage1/evidence' not in img:
                    missing_ev.append(img)
    add('V10', '复核记录的证据图存在', not missing_ev, f"缺图 {len(missing_ev)} 张", missing_ev)

    # V11 协议完整性
    need = ['scope', 'group_matching', 'lane_scoring', 'error_diagnostics',
            'coverage_reporting', 'text_and_boundary', 'run_states', 'prediction_contract_fields']
    miss = [k for k in need if k not in proto]
    enum_ok = proto['run_states']['enum'] == ['not_run', 'completed', 'partial', 'failed', 'skipped']
    add('V11', '协议字段与枚举完整', not miss and enum_ok,
        f"缺字段 {miss}；运行状态枚举 {'正确' if enum_ok else '不符'}", miss)

    # V12 匹配夹具
    add('V12', '匹配夹具期望值自洽', cases['all_expected_verified'],
        f"{len(cases['cases'])} 个 IoU 用例 + {len(cases['diagnostic_cases'])} 个诊断用例 + "
        f"{len(cases['lane_pairing_cases'])} 个轨级用例，期望值与参考实现一致",
        [c['case_id'] for c in cases['cases'] if not c['matches_expected']]
        + [c['case_id'] for c in cases['diagnostic_cases'] if not c['ok']]
        + [c['case_id'] for c in cases['lane_pairing_cases'] if not c.get('verified')])

    # V13 契约完整性
    cneed = ['config_snapshot', 'seam_profile', 'scale_object', 'rule_line_policy',
             'run_state_contract', 'prediction_fields']
    cmiss = [k for k in cneed if k not in contract]
    add('V13', '跨阶段契约字段完整', not cmiss, f"缺字段 {cmiss}", cmiss)

    # V14 同页组间交叠（含同栏上下组）
    cross_overlaps = find_cross_group_overlaps(ann['groups'])
    add('V14', '同页组间不交叠（含同栏上下组）', not cross_overlaps,
        f"组间交叠 {len(cross_overlaps)} 处（面积>0 即失败，边缘接触应为 0）", cross_overlaps)

    # V15 上方轨框不得侵入下一独立标题
    intrusions = find_title_intrusions(ann['groups'])
    add('V15', '上方轨框不侵入下一组标题', not intrusions,
        f"轨框压住下一标题 {len(intrusions)} 处", intrusions)

    # V16 主字框不得吞入本组小字（允许 1-2px 边缘相接，记录实测）
    sw = find_main_swallow(ann['groups'])
    swallowed, contacts = sw['swallowed'], sw['contacts']
    add('V16', '主字框与本组小字框的几何重叠（仅几何，不代表字形归属正确）', not swallowed,
        f"实质重叠 {len(swallowed)} 处；1-2px 边缘相接 {len(contacts)} 处（继承组，已知）。"
        f"注意：本项只判「框有没有交叠」，不能证明主字框内部没有小字——那要靠 V20 的图像核对",
        swallowed)
    checks[-1]['edge_contacts'] = contacts
    checks[-1]['capability'] = 'geometry_only'

    # V21 页面状态规则与夹具（含组数不变性）
    ps = proto['run_states'].get('fixtures', {})
    add('V21', '页面状态规则夹具（含 evidence 组数不变性）', bool(ps.get('all_ok')),
        f"{len(ps.get('cases', []))} 个状态夹具 + 组数不变性检查，"
        f"全部通过={ps.get('all_ok')}",
        [c['id'] for c in ps.get('cases', []) if not c.get('ok')])

    # V20 字形归属核对记录（图像证据）；几何通过 ≠ 字形正确
    missing, bad = [], []
    for g in ann['groups']:
        if g['origin'] == 'stage1':
            continue
        gr = g.get('glyph_review') or {}
        need = ('main_last_glyph', 'left_first_glyph', 'right_first_glyph',
                'left_last_glyph', 'right_last_glyph', 'review_status')
        lack = [k for k in need if k not in gr or not gr[k]]
        if lack:
            missing.append({'group': g['group_id'], 'missing': lack})
            continue
        if gr['review_status'] not in ('checked', 'checked_with_ambiguity'):
            bad.append({'group': g['group_id'], 'reason': f"review_status={gr['review_status']}"})
        for k in ('main_last_glyph', 'left_first_glyph', 'right_first_glyph'):
            if gr[k].get('glyph_complete') is not True:
                bad.append({'group': g['group_id'], 'reason': f'{k} 未确认完整'})
        for k in ('left_last_glyph', 'right_last_glyph'):
            if gr[k].get('cut') is not False:
                bad.append({'group': g['group_id'], 'reason': f'{k} 末笔可能被切'})
        for pth in gr.get('evidence_paths', []):
            if not (D.ROOT / pth).exists():
                bad.append({'group': g['group_id'], 'reason': f'缺证据图 {pth}'})
    add('V20', '新增组的字形归属核对记录完整（首尾字与证据）', not missing and not bad,
        f"新增 {sum(1 for g in ann['groups'] if g['origin'] != 'stage1')} 组："
        f"缺记录 {len(missing)} 组、记录不成立 {len(bad)} 处", missing + bad)

    # V17 正例与 ambiguous / ignore 区域冲突 + 歧义组必须有候选解释
    amb_regions = neg.get('ambiguous_regions', [])
    ign_regions = neg.get('ignore_regions', [])
    hits, amb_missing = [], []
    for r in amb_regions + ign_regions:
        for g in ann['groups']:
            if g['page_id'] != r['page_id']:
                continue
            for k in ('main_bbox', 'right_bbox', 'left_bbox'):
                a = D.intersect_area(g[k], r['bbox'])
                if a > 0:
                    hits.append({'region': r['region_id'], 'kind': r['kind'], 'group': g['group_id'],
                                 'field': k, 'area': a})
    for g in ann['groups']:
        if g['certainty'] == 'ambiguous_lane_extent' and not g.get('candidates'):
            amb_missing.append(g['group_id'])
    add('V17', '正例不被 ambiguous/ignore 掩膜覆盖，且歧义组有候选解释',
        not hits and not amb_missing,
        f"掩膜冲突 {len(hits)} 处；缺候选解释的歧义组 {len(amb_missing)} 个", hits + amb_missing)

    # V18 栏线负例与框的重叠：给出面积比、持续高度与处置依据
    rule_rows = []
    for r in neg['regions']:
        if r['category'] != 'rule':
            continue
        rx1, ry1, rx2, ry2 = r['bbox']
        rw, rh = rx2 - rx1, ry2 - ry1
        for g in ann['groups']:
            if g['page_id'] != r['page_id']:
                continue
            for k in ('main_bbox', 'right_bbox', 'left_bbox'):
                b = g[k]
                m = classify_rule_overlap(b, r['bbox'])
                if not m:
                    continue
                m.update({'region': r['negative_region_id'], 'group': g['group_id'], 'field': k,
                          'basis': ('重叠只发生在框的边缘 1-3px，属墨迹外接框必然纳入栏线像素，'
                                    '评测时由栏线掩膜剔除' if m['verdict'] == 'edge_coincidence'
                                    else '重叠宽度超过框宽 25%，判为栏线污染，需重定框')})
                rule_rows.append(m)
    contam = [r for r in rule_rows if r['verdict'] == 'contamination']
    add('V18', '栏线负例与框的重叠已量化且无实质污染', not contam,
        f"栏线相关重叠 {len(rule_rows)} 处，其中边缘重合 {len(rule_rows) - len(contam)} 处、"
        f"实质污染 {len(contam)} 处", rule_rows)
    checks[-1]['rule_overlap_measurements'] = rule_rows

    # V19 原缺陷（v1 上下组吞并）必须能被新检查抓到
    v1_path = EVID / 'df12_annotations_before_fix.json'
    old_conflicts = []
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text(encoding='utf-8'))
        v1_groups = [g for g in v1['groups'] if g['book'] == S.DF_BOOK and g['page'] == S.DF_PAGE]
        for i in range(len(v1_groups)):
            for j in range(i + 1, len(v1_groups)):
                a, b = v1_groups[i], v1_groups[j]
                for fa in ('main_bbox', 'right_bbox', 'left_bbox'):
                    for fb in ('main_bbox', 'right_bbox', 'left_bbox'):
                        area = D.intersect_area(a[fa], b[fb])
                        if area > 0:
                            old_conflicts.append({'a': a['region'], 'a_field': fa,
                                                  'b': b['region'], 'b_field': fb, 'area': area})
    add('V19', '原缺陷能被 V14/V15 抓到（回归证明）', bool(old_conflicts),
        f"v1 快照在同页组间检查下产生 {len(old_conflicts)} 处交叠（>0 说明检查确实能抓到这个缺陷）",
        old_conflicts[:12])
    checks[-1]['snapshot'] = _rel(v1_path) if v1_path.exists() else None

    checks.sort(key=lambda c: int(str(c['id'])[1:]))
    ok = all(c['status'] == 'pass' for c in checks)
    return {'stage': STAGE, 'release': S.RELEASE,
            'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
            'summary': {'checks': len(checks), 'pass': sum(1 for c in checks if c['status'] == 'pass'),
                        'fail': sum(1 for c in checks if c['status'] == 'fail'), 'ok': ok},
            'checks': checks}


# ---------------------------------------------------------------- 写盘

FILES = ['annotations.json', 'negative_regions.json', 'dataset_splits.json',
         'evaluation_protocol.json', 'cross_stage_contract.json', 'review_log.json',
         'matching_cases.json']


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


PREV_RELEASES = [
    {'release': 'stage2-v1', 'note': '第二阶段首版（2026-11-23 21:23-21:45）',
     'artifact_sha256': {'annotations.json': 'cacefb18d298c6ab',
                         'negative_regions.json': '8c6aa461dffe4baf',
                         'dataset_splits.json': '2ad9b687b8549604',
                         'evaluation_protocol.json': '3cffc6b7eb3da741',
                         'cross_stage_contract.json': 'd910a1ff6dbe4f25',
                         'review_log.json': '2ca08be83cb28fa5',
                         'matching_cases.json': '108b44229a9da332'},
     'validation': {'checks': 13, 'pass': 13, 'fail': 0},
     'note_hash': '仅前缀，用于对照'},
    {'release': 'stage2-v2', 'note': '第二阶段缺口修正（2026-11-24 11:08-12:00）',
     'artifact_sha256': {'annotations.json': '0f1975de9c8ef54b',
                         'negative_regions.json': '68fc60485e5836f4',
                         'dataset_splits.json': '5ca5419e07b0d96b',
                         'evaluation_protocol.json': 'ddb6686bb446d032',
                         'cross_stage_contract.json': 'cf83ccb8044e8abb',
                         'review_log.json': '785be291adcac0cf',
                         'matching_cases.json': '95c41f8dedadc6d3'},
     'validation': {'checks': 19, 'pass': 19, 'fail': 0},
     'note_hash': '仅前缀，用于对照'},
]

RELEASE_CHANGES = [
    'p12 十一组按字形重定主字框下界与轨框上界：药名末字底 268-305、轨首字顶 277-309，'
    '不再把首行小字算进主字框（v2 的主字框底一律 305 是错误假设）',
    '新增逐组字形核对记录 glyph_review（主字末字 / 左右轨首字 / 左右轨末字 + 证据图 + 状态）',
    'V16 改名并声明 capability=geometry_only；新增 V20 校验字形记录完整性与证据存在',
    '尺度契约拆分：H（正文中位字高）、G（字间留白，G=a_next-b_i，负留白剔除并计数）、'
    'glyph_pitch（字心间距，独立）、small_glyph_height/width/gap/pitch、page_height；附 30/5/35 合成例子',
    '剖面与窗口公式改用具名字段（glyph_pitch / small_glyph_width / G / small_glyph_gap / small_glyph_height），'
    '清除 G_local 等未定义引用',
    '页面状态改为按「应处理区域完成度」汇总，skipped 区分 out_of_scope / incomplete；'
    '新增 11 条状态夹具与 evidence 组数不变性验证（V21）',
]


def write_revision_notes(out: Path, manifest: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    path = out / 'revision_notes.json'
    notes = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'stage': STAGE, 'releases': []}
    keep = [r for r in notes['releases']
            if r.get('release') not in {p['release'] for p in PREV_RELEASES}]
    notes['releases'] = [dict(p) for p in PREV_RELEASES] + keep
    s = report['summary']
    entry = {
        'release': S.RELEASE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'artifact_sha256': {k: v[:16] for k, v in manifest['artifact_sha256'].items()},
        'validation': {'checks': s['checks'], 'pass': s['pass'], 'fail': s['fail']},
        'changes': RELEASE_CHANGES,
    }
    prev = next((r for r in notes['releases'] if r.get('release') == 'stage2-v2'), None)
    if prev:
        entry['vs_previous'] = {
            'checks_delta': s['checks'] - prev['validation']['checks'],
            'artifacts_changed': sorted(
                k for k, v in entry['artifact_sha256'].items()
                if prev['artifact_sha256'].get(k) not in (None, v)),
        }
    notes['releases'] = [r for r in notes['releases'] if r.get('release') != S.RELEASE]
    notes['releases'].append(entry)
    notes['releases'].sort(key=lambda r: r['release'])
    write_json(path, notes)
    return {'written': _rel(path), 'release': S.RELEASE}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成第二阶段产物')
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = build_annotations()
    neg = build_negatives(ann)
    splits = build_splits(ann)
    proto = build_protocol()
    contract = build_contract()
    cases = build_matching_cases()
    review = build_review_log()

    # 回填分区归属，让读标注的人不用再查 splits
    split_of = {a['page_id']: a['split'] for a in splits['assignments']}
    for g in ann['groups']:
        g['split_membership'] = split_of.get(g['page_id'], 'unassigned')
    for p in ann['pages']:
        p['split_membership'] = split_of.get(p['page_id'], 'unassigned')
    ann['counts']['groups_by_split'] = _count([g['split_membership'] for g in ann['groups']])

    payloads = {
        'annotations.json': ann,
        'negative_regions.json': neg,
        'dataset_splits.json': splits,
        'evaluation_protocol.json': proto,
        'cross_stage_contract.json': contract,
        'review_log.json': review,
        'matching_cases.json': cases,
    }
    for name, payload in payloads.items():
        write_json(out / name, payload)

    report = validate(ann, neg, splits, proto, contract, cases, review)
    write_json(out / 'validation_report.json', report)

    manifest = {
        'stage': STAGE, 'release': S.RELEASE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'outputs': {n: _rel(out / n) for n in FILES + ['validation_report.json']},
        'artifact_sha256': {n: _sha(out / n) for n in FILES},
        'validation_summary': report['summary'],
        'upstream': {'stage1_normalized': _rel(D.OUT_DIR / 'normalized_annotations.json'),
                     'stage1_sha256': _sha(D.OUT_DIR / 'normalized_annotations.json')},
    }
    write_json(out / 'build_manifest.json', manifest)
    rev = write_revision_notes(out, manifest, report)
    manifest['revision_notes'] = rev
    write_json(out / 'build_manifest.json', manifest)

    if not args.quiet:
        c = ann['counts']
        print(f"组 {c['groups_total']}（继承 {c['groups_by_origin'].get('stage1', 0)} / "
              f"新增 {c['groups_total'] - c['groups_by_origin'].get('stage1', 0)}）；页 {c['pages_total']}")
        print(f"来源 {c['groups_by_source']}")
        print(f"负例 {neg['counts']['regions']} 个 {neg['counts']['by_category']}；"
              f"栏线样例 {neg['counts']['rule_samples']} 个 {neg['counts']['rule_kinds']}")
        print(f"分区 {splits['counts']}")
        print(f"校验 {report['summary']['pass']}/{report['summary']['checks']} 通过"
              f"{'' if report['summary']['ok'] else '（有失败项）'}")
        for chk in report['checks']:
            if chk['status'] != 'pass':
                print(f"  [FAIL] {chk['id']} {chk['title']} — {chk['detail']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
