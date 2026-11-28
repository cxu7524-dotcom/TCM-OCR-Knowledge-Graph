"""生成双轨小字页第一阶段的规范化标注（只读来源，写入独立输出目录）。

用法：
  .venv/Scripts/python.exe tools/normalize_dual_track_dataset.py
  .venv/Scripts/python.exe tools/normalize_dual_track_dataset.py --out-dir output/dual_tracks/stage1

产物（默认 output/dual_tracks/stage1/）：
  normalized_annotations.json   规范化标注（页面级 + 组级）
  corrections.json              逐项修正记录（旧值 / 新值 / 依据 / 来源）
  source_fingerprints_before.json / source_fingerprints_after.json   来源指纹
  normalization_manifest.json   本次运行的输入指纹与产物指纹（用于重复运行一致性核对）

约定：
  - 坐标数值一律保留原值，不做无图像依据的挪动
  - image_size 改写为原图实测尺寸
  - 端点约定能由代码证据确认的标 half_open，其余标 pending_verification
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_track_stage1_data as D  # noqa: E402
from check_dual_track_dataset import Context, run_checks  # noqa: E402

TZ = timezone(timedelta(hours=8))

COORD_SYSTEM = '原扫描图像素，左上角原点，x 右 y 下'
GROUP_BBOX_RULE = 'union(right_bbox, left_bbox)，不含主字框'
TOLERANCE_NOTE = '端点两种读法只让框的右下边界相差 1px，这是坐标解释差异，不是标注误差'

RELEASE = 'stage1-v2'

# 上一轮（首轮）产物的指纹，用于「本轮 vs 上一轮」对照；值取自首轮落盘文件。
PREVIOUS_RELEASE = {
    'release': 'stage1-v1',
    'note': '第一阶段首轮（2026-11-23 20:46-21:00）',
    'artifacts_sha256': {
        'normalized_annotations.json': '6941fe2e43e5ff77e2bac3a25d4be45bc518a4f28d88da6fed5c5170f9ef4add',
        'corrections.json': '19a6f3e73b1929d451cd10b3581369279903a3e3f1cdf11453a1b58036a197e3',
    },
    'checks': {'total': 13, 'pass': 9, 'fail': 2, 'pending': 2,
               'failed_ids': ['C02', 'C13'], 'pending_ids': ['C09', 'C12']},
    'counts': {'groups_total': 26, 'pages_total': 23, 'corrections_total': 56},
}

RELEASE_CHANGES = [
    '补 C14：清单 JSON 与 CSV 逐字段一致性（数量、唯一键、重复/缺行/多行、'
    'status/confidence/source/dataset_path/sha256/main/right/left/note）',
    '修 C06/C07/C08/C09：坐标解析不再抛异常；inf/-inf/nan/bool/非整数/非法字符串一律判非法并出明细，'
    '其余记录继续校验',
    '拆精度语义：端点解释差异（定义性 1px）、字形边界误差（未标定）、评测容差（待协议）分三个字段',
    '删掉无标定依据的 expected_error_px（原「约 1」「约 1-3」）',
    '新增 glyph_boundary_check：逐组记录字形完整性是否实际看过、看的是哪张图、结论',
    '丹方彚編 p12 由「待裁定」改为第二阶段既定任务',
    '文案：「坐标数值本身没有错」改为「未缩放、未越界、左右顺序正确」，字形完整性单列',
]

# 人工目验记录（2026-11-23，叠框放大 6 倍；证据图见 output/dual_tracks/stage1/evidence/）。
# 这些是「看原图才能定」的结论，不参与自动判定，逐条写明依据。
MANUAL_FINDINGS = [
    {
        'id': 'K-P35-text',
        'scope': 'group', 'group_id': '思濟堂方書/p35/dev-35-001',
        'field': 'right_text / left_text', 'action': 'recorded_only_not_modified',
        'old': {'清单': {'right': '五錢', 'left': '泔炒'}, '说明.md': {'right': '五錢', 'left': '泔炒'}},
        'new': {'坐标': {'right': '五錢黃土', 'left': '拌炒'}},
        'basis': '6 倍叠框核对原图：白朮下方右轨为「五錢黃土」四字、左轨为「拌炒」二字，'
                 '「泔」与字形不符；坐标侧正确，清单与说明.md 的例证文字有误',
        'evidence_image': 'output/dual_tracks/stage1/evidence/思濟堂方書__p35__dev-35-001.png',
        'impact': '不影响坐标与几何；说明.md 第 11 行的例证文字与清单 p35 条目的 right/left 需按此更正，'
                  '本阶段不改原始来源',
    },
    {
        'id': 'K-P34-10-uncertain',
        'scope': 'group', 'group_id': '思濟堂方書/p34/P34-10',
        'field': 'main_text', 'action': 'pending_review',
        'old': '大〔?〕棗', 'new': '大〔?〕棗（保留；原图核对更接近「大膠棗」）',
        'basis': '6 倍叠框核对原图：中间字结构接近「膠」，与清单的「大膠棗」一致；'
                 '但单人目验不足以撤掉疑字标记，按保守口径保留〔?〕并置 certainty=待核',
        'evidence_image': 'output/dual_tracks/stage1/evidence/思濟堂方書__p34__P34-10.png',
        'impact': '该组主字转写仍按存疑处理，可参与几何评测，不参与药名逐字评测',
    },
    {
        'id': 'K-P39-04-split',
        'scope': 'group', 'group_id': '思濟堂方書/p39/P39-04',
        'field': 'right_text / left_text 切分', 'action': 'recorded_only_not_modified',
        'old': {'清单': {'right': '四錢', 'left': '黃土拌炒'}},
        'new': {'坐标': {'right': '四錢黃土', 'left': '拌炒'}},
        'basis': '6 倍叠框核对原图：右轨为「四錢黃土」四字、左轨为「拌炒」二字，'
                 '清单把「黃土」记进左轨与分栏不符；两轨字符序列相同（四錢黃土拌炒），'
                 '差别只在切分位置，坐标侧与图像一致',
        'evidence_image': 'output/dual_tracks/stage1/evidence/思濟堂方書__p39__P39-04.png',
        'impact': '不影响几何；文本比对时须按「右轨+左轨」拼接后再与清单比，不能逐轨对标',
    },
]

# 字形完整性核对记录：只写实际看图看过的组。
# 方法：6 倍叠框放大目验（evidence/ 下同名 PNG），逐组看首字顶笔与末字底笔是否被框线切断。
# 未列出的组一律标 not_inspected，不冒充已核。
GLYPH_CHECKS = {
    '思濟堂方書/p33/S33-01': ('inspected_ok', '「黃」末笔紧贴框底，未见切断；右轨「二錢半」、左轨「酒洗」完整'),
    '思濟堂方書/p36/S36-01': ('inspected_ok', '「藥」末笔紧贴框底；主字框底与两轨框顶有间隙'),
    '思濟堂方書/p42/S42-01': ('inspected_ok', '「麥冬」上下完整；右轨「各三錢」四字、左轨「去心」二字均完整入框'),
    '思濟堂方書/p34/P34-10': ('inspected_ok', '三框内字均完整；疑字中间字结构接近「膠」'),
    '思濟堂方書/p35/dev-35-001': ('inspected_ok', '右轨「五錢黃土」四字、左轨「拌炒」二字完整入框'),
    '思濟堂方書/p39/P39-04': ('inspected_ok', '右轨「四錢黃土」四字、左轨「拌炒」二字完整入框'),
    '思濟堂方書/p43/P43-05': ('inspected_ok', '「豆」底横在框内；两轨字完整'),
    '思濟堂方書/p46/P46-06': ('inspected_ok', '主字框与两轨框 1px 相接，属框层相接，字与字之间无内容交叠'),
    '思濟堂方書/p49/P49-13': ('inspected_uncertain',
                              '首字「親」顶部紧贴框顶，6 倍下无法判定是否切到顶笔，按存疑记录'),
    '温疫方論/p12/W12-01': ('inspected_ok', '主字框与两轨框 2px 相接，无内容交叠'),
    '温疫方論/p12/W12-02': ('inspected_ok', '无内容交叠；近似框偏松，左轨框底比「分」末笔低约 10px'),
}


PRECISION_SEMANTICS = {
    'why_three_fields': '三件事经常被混成一个「±1px」：'
                        '(1) 端点怎么读、(2) 框与字形边界差多少、(3) 评测允许差多少。'
                        '三者来源不同、可信度不同，必须分开记，不能互相推导',
    'endpoint_interpretation': {
        'what': '右下角是包含还是不包含（半开 [x1,y1,x2,y2) 与闭区间 [x1,y1,x2,y2] 两种读法）',
        'certitude': '待核（23 组）/ 已确认（3 组，由生成代码 tight() 的 +1 确定）',
        'magnitude': '定义性的 1px：只让框的右/下边界平移，不涉及字形',
        'pitfall': '不能由此推出「标注误差 ≤1px」',
    },
    'glyph_boundary_error': {
        'what': '框的边与真实字形外沿之间差多少像素',
        'certitude': 'unknown_uncalibrated：三类来源都没有做过标定测量',
        'magnitude': None,
        'pitfall': '近似框（W12 两组）连量级都不可给；人工框与墨迹框同样未标定，'
                   '任何「约 1-3px」都是猜测',
    },
    'evaluation_tolerance_px': {
        'what': '评测时人为允许的偏差阈值（IoU 阈值、边界容差带等）',
        'certitude': 'pending_protocol：评测协议未定，本阶段不设默认值',
        'magnitude': None,
        'pitfall': '容差是协议选择，不是数据属性；协议定了才写进评测，不能反过来当成数据误差',
    },
    'glyph_boundary_check': {
        'what': '人眼在 6 倍叠框图上核「首字顶笔 / 末字底笔有没有被框线切断」',
        'scope': f'实际核对 {len(GLYPH_CHECKS)} 组 / 共 26 组；其余标 not_inspected',
        'result': '见每组 glyph_boundary_check.status 与 note',
    },
}


def _glyph_boundary_check(gid: str) -> Dict[str, Any]:
    hit = GLYPH_CHECKS.get(gid)
    if not hit:
        return {
            'status': 'not_inspected',
            'method': '6 倍叠框放大目验',
            'evidence_image': f"output/dual_tracks/stage1/evidence/{gid.replace('/', '__')}.png",
            'note': '本轮未逐组核对字形完整性，仅有几何检查通过；首尾笔画是否被框线切断未核',
        }
    status, note = hit
    return {
        'status': status,
        'method': '6 倍叠框放大目验（原扫描图叠框，非裁图重绘）',
        'evidence_image': f"output/dual_tracks/stage1/evidence/{gid.replace('/', '__')}.png",
        'note': note,
    }


SCOPE_NOTES = {
    'target_definition': '当前任务口径：在同一实体栏目、同一纵向区段内定位左右并排的两轨小字；'
                         '允许同栏上下多组，允许两轨起止高度不同',
    'dataset_criterion': '说明.md 的收录判据更窄：只要「大字下方两条小字」的剂量/炮制形态，'
                         '整段双行小字夹注按旧口径不收',
    'stage2_tasks': [
        {
            'task': '补标 丹方彚編/page_12 右半页',
            'status': '第二阶段既定任务（用户已确认该页属目标范围，不再裁定是否纳入）',
            'why': '该书整页为「每个药名下一组双行小字」，按当前口径属目标形态，'
                   '但说明.md 第一节按旧口径（整段双行夹注不收）未收录',
            'scope': '右半页；需新增该页坐标标注与逐轨转写',
            'evidence': 'output/dual_tracks/stage1/evidence/_scope_丹方彚編_p12_右半頁.png',
            'impact_until_done': '当前 26 组不代表当前口径的全部形态；不得据此宣称跨书召回，'
                                 '也不能把丹方彚編类页面判为负例',
        }
    ],
}



def _convention_for(source_class: str) -> tuple:
    if source_class == D.SOURCE_MEASURED:
        return D.CONV_HALF_OPEN, (
            'tools/dual_track_boxes_from_window.py 的 tight() 以「末笔墨迹坐标 + 1」作 x2/y2，'
            '即半开区间 [x1,y1,x2,y2)')
    return D.CONV_PENDING, (
        '来源文档只声明「图像像素框」，未声明右下角是否含端点；来源侧生成代码未保留，'
        '数据本身无法区分（1px 差异低于二值化噪声）')


def _eligibility(source_class: str, convention: str) -> Dict[str, Any]:
    """评测资格。精确边界评测的阻断项单独列出，不混进「容差」一个数字里。"""
    usable = ['double_track_localization', 'lane_split', 'reading_order', 'region_crop']
    not_usable: List[str] = []
    blockers: List[str] = []
    reason = []
    if source_class == D.SOURCE_APPROX:
        blockers.append('approximate_box')
        not_usable.extend(['glyph_exact_boundary', 'precise_iou_benchmark'])
        reason.append('人工近似框，来源文档自述并非字形级精确分割')
    elif source_class == D.SOURCE_MANUAL:
        reason.append('人工逐框，已作原图与叠框检查，未做字形级标定')
    else:
        reason.append('墨迹外接框，比人工框略紧，字间留白未计入')
    if convention != D.CONV_HALF_OPEN:
        blockers.append('endpoint_unresolved')
        reason.append('右下角端点约定待核，坐标存在两种读法')
    # 三类来源都没有字形边界标定 -> 精确边界还需要协议定容差
    if not blockers:
        blockers.append('no_boundary_calibration')
    exact = {
        'allowed': False,
        'blockers': sorted(set(blockers)),
        'requires': ['protocol_tolerance'],
        'note': '阻断项是「为什么现在不能用」；requires 是「要用还缺什么」。'
                '容差数值本身尚未确定，见 evaluation_tolerance_px',
    }
    status = 'usable' if not (set(blockers) & {'endpoint_unresolved', 'approximate_box'}) else 'conditional'
    return {
        'status': status,
        'usable_for': usable,
        'not_usable_for': sorted(set(not_usable)),
        'exact_boundary_evaluation': exact,
        'reason': '；'.join(reason),
    }


def build(ctx: Optional[Context] = None) -> Dict[str, Any]:
    ctx = ctx or Context()
    pages_by_id = ctx.page_records()
    coords_by_page = ctx.coords_by_page()

    groups: List[Dict[str, Any]] = []
    corrections: List[Dict[str, Any]] = []

    for c in sorted(ctx.coords, key=lambda r: (r['book'], int(r['page']), r['region'])):
        book, page, region = c['book'], int(c['page']), c['region']
        gid = D.group_id(book, page, region)
        pid = D.page_id(book, page)
        image_path = D.DATASET / book / D.page_file_name(page)
        info = ctx.image(image_path)
        real_size = info['image_size']
        declared = c.get('image_size')
        cls = D.classify_source(c.get('source', ''))
        conv, conv_basis = _convention_for(cls)
        main_b, right_b, left_b = D.as_bbox(c['main_bbox']), D.as_bbox(c['right_bbox']), D.as_bbox(c['left_bbox'])
        group_b = D.union_bbox([right_b, left_b])

        if list(declared or []) != list(real_size or []):
            corrections.append({
                'id': f'K-{gid}-size',
                'scope': 'group', 'group_id': gid,
                'field': 'image_size', 'action': 'corrected',
                'old': declared, 'new': real_size,
                'basis': f'解码原图 {D.rel(image_path)} 得真实宽高；原值来自 '
                         f'tools/build_dual_track_coords.py 第 93 行的统一硬编码 [1024, 804]',
                'source': D.rel(D.COORDS_JSON),
                'impact': '仅尺寸字段；坐标数值未改，且经 C07 核对全部落在真实图内',
            })
        corrections.append({
            'id': f'K-{gid}-conv',
            'scope': 'group', 'group_id': gid,
            'field': 'coordinate_convention', 'action': 'convention_assigned',
            'old': 'unknown（来源文档未声明）', 'new': conv,
            'basis': conv_basis,
            'source': D.rel(D.COORDS_JSON) if cls != D.SOURCE_MEASURED else
                      'tools/dual_track_boxes_from_window.py:20-26',
            'impact': TOLERANCE_NOTE if conv == D.CONV_PENDING else '无',
        })

        groups.append({
            'group_id': gid,
            'page_id': pid,
            'book': book,
            'page': page,
            'region': region,
            'original_region_id': region,
            'image_path': D.rel(image_path),
            'image_source_path': D.rel(D.BOOKS / book / D.page_file_name(page)),
            'image_sha256': info['sha256'],
            'image_size': real_size,
            'original_image_size': declared,
            'image_size_matches_original': list(declared or []) == list(real_size or []),
            'main_bbox': list(main_b),
            'right_bbox': list(right_b),
            'left_bbox': list(left_b),
            'group_bbox': list(group_b),
            'group_bbox_rule': GROUP_BBOX_RULE,
            'main_text': c['main_text'],
            'right_text': c['right_text'],
            'left_text': c['left_text'],
            'reading_order': ['right', 'left'],
            'column': c.get('column'),
            'column_state': 'present' if c.get('column') is not None else 'default_absent',
            'annotation_source': cls,
            'annotation_source_raw': c.get('source', ''),
            'annotation_source_profile': D.source_profile(cls),
            'boundary_precision': _boundary_precision(cls, conv),
            'endpoint_interpretation': _endpoint_interpretation(conv),
            'glyph_boundary_error': _glyph_boundary_error(cls),
            'evaluation_tolerance_px': _evaluation_tolerance(),
            'glyph_boundary_check': _glyph_boundary_check(gid),
            'certainty': '待核' if _has_uncertain(c) or pages_by_id[pid].get('confidence') == '待核' else '确认',
            'certainty_basis': _certainty_basis(c, pages_by_id[pid]),
            'review_status': 'confirmed_single_reviewer' if pages_by_id[pid].get('confidence') == '确认'
                             else 'pending_review',
            'coordinate_convention': conv,
            'original_coordinate_convention': 'unknown',
            'evaluation_eligibility': _eligibility(cls, conv),
            'annotation_coverage': D.COVERAGE_PARTIAL,
            'scope_note': '已知位置标注：说明.md 明确「26 处是已确认并可定位的位置，不是全页双轨清单」',
            'source_reference': {
                'manifest': f"{D.rel(D.MANIFEST_JSON)}#{pid}",
                'coordinates_json': f"{D.rel(D.COORDS_JSON)}#{gid}",
                'coordinates_csv': f"{D.rel(D.COORDS_CSV)}#{book},{page},{region}",
                'upstream': D.source_profile(cls)['upstream'],
                'readme': D.rel(D.README),
            },
        })

    # 页面级记录：收录 / 待核 / 排除 都保留
    page_records: List[Dict[str, Any]] = []
    for m in sorted(ctx.manifest, key=lambda r: (r['book'], D.manifest_page_number(r))):
        n = D.manifest_page_number(m)
        pid = D.page_id(m['book'], n)
        ds_path = D.ROOT / m['dataset_path'] if m.get('dataset_path') else None
        img_path = ds_path if ds_path is not None else D.BOOKS / m['book'] / D.page_file_name(n)
        info = ctx.image(img_path)
        gids = [D.group_id(c['book'], int(c['page']), c['region']) for c in coords_by_page.get(pid, [])]
        has_boxes = bool(gids)
        convs = {g['coordinate_convention'] for g in groups if g['page_id'] == pid}
        page_records.append({
            'page_id': pid,
            'book': m['book'],
            'page': n,
            'status': m['status'],
            'confidence': m.get('confidence', ''),
            'reason': m.get('note', ''),
            'image_path': D.rel(img_path),
            'source_image_path': D.rel(D.BOOKS / m['book'] / D.page_file_name(n)),
            'image_sha256': info.get('sha256'),
            'image_size': info.get('image_size'),
            'manifest_sha256': m['sha256'],
            'sha256_matches': info.get('sha256') == m['sha256'],
            # 页面级不重复存框：框只在 groups 里，页面级留空并记录状态
            'main_bbox': None,
            'right_bbox': None,
            'left_bbox': None,
            'group_bbox': None,
            'box_state': 'group_level_only' if has_boxes else (
                'absent_pending_review' if m['status'] == D.STATUS_PENDING else 'not_applicable_excluded'),
            'group_ids': gids,
            'group_count': len(gids),
            'coordinate_conventions': sorted(convs),
            'annotation_coverage': D.COVERAGE_PARTIAL if gids else 'none',
        })

    # 说明文件与实际数据的口径差异（只记录，不改原始来源）
    corrections.extend(_readme_corrections(ctx))
    # 人工目验结论（看原图才能定的部分）
    corrections.extend(MANUAL_FINDINGS)

    counts = {
        'pages_total': len(page_records),
        'pages_by_status': _counts([p['status'] for p in page_records]),
        'groups_total': len(groups),
        'groups_by_source': _counts([g['annotation_source'] for g in groups]),
        'groups_by_convention': _counts([g['coordinate_convention'] for g in groups]),
        'groups_by_endpoint_status': _counts([g['endpoint_interpretation']['status'] for g in groups]),
        'groups_by_glyph_check': _counts([g['glyph_boundary_check']['status'] for g in groups]),
        'groups_by_certainty': _counts([g['certainty'] for g in groups]),
        'pages_with_groups': len({g['page_id'] for g in groups}),
        'pages_without_groups': sorted({p['page_id'] for p in page_records if not p['group_ids']}),
        'corrections_total': len(corrections),
        'corrections_by_action': _counts([c['action'] for c in corrections]),
    }

    return {
        'stage': D.STAGE,
        'generated_by': 'tools/normalize_dual_track_dataset.py',
        'dataset_root': D.rel(D.DATASET),
        'source_files': [D.rel(p) for p in (D.MANIFEST_JSON, D.MANIFEST_CSV, D.COORDS_JSON,
                                            D.COORDS_CSV, D.README)],
        'coordinate_system': COORD_SYSTEM,
        'group_bbox_rule': GROUP_BBOX_RULE,
        'annotation_coverage': D.COVERAGE_PARTIAL,
        'annotation_coverage_note': '现有 26 组是「已知位置标注」，不足以证明覆盖整页全部双轨；'
                                    '不能据此计算召回率（说明.md 第五节口径，与当前任务一致）',
        'scope_notes': SCOPE_NOTES,
        'precision_semantics': PRECISION_SEMANTICS,
        'conventions': {
            'half_open': '[x1,y1,x2,y2) 半开区间，右下角不含端点',
            'pending_verification': '保留原坐标，右下角有「含端点」与「半开」两种读法，端点约定待核；'
                                    '这只影响坐标怎么读，不给出评测容差',
        },
        'counts': counts,
        'groups': groups,
        'pages': page_records,
        'corrections': corrections,
    }


def _has_uncertain(rec: Dict[str, Any]) -> bool:
    return any('〔?〕' in (rec.get(k) or '') or '〔？〕' in (rec.get(k) or '')
               for k in ('main_text', 'right_text', 'left_text'))


def _certainty_basis(rec: Dict[str, Any], manifest_page: Dict[str, Any]) -> str:
    bits = []
    if _has_uncertain(rec):
        bits.append('转写含〔?〕疑字标记')
    if manifest_page.get('confidence') == '待核':
        bits.append('清单该页 confidence=待核')
    return '；'.join(bits) if bits else '清单 confidence=确认，框几何检查通过'


def _boundary_precision(cls: str, conv: str) -> Dict[str, Any]:
    """框是怎么画的、标定到哪一步。这里不给像素误差数字。"""
    prof = D.source_profile(cls)
    return {
        'level': {D.SOURCE_MANUAL: 'manual_tight_box',
                  D.SOURCE_MEASURED: 'ink_tight_box',
                  D.SOURCE_APPROX: 'approximate_box'}[cls],
        'description': prof['precision'],
        'calibration': {'status': 'none',
                        'note': '未做「同一组框 vs 逐字人工边界」的对照测量，无标定数据'},
        'known_limits': prof['known_limits'],
        'note': '几何检查只能证明框落在图内、面积为正、左右顺序正确；'
                '首尾笔画完整性见 glyph_boundary_check，不能由几何检查推出',
    }


def _endpoint_interpretation(conv: str) -> Dict[str, Any]:
    """坐标怎么读：右下角是包含还是不包含。定义性的 1px，与测量误差无关。"""
    if conv == D.CONV_HALF_OPEN:
        return {
            'status': 'resolved',
            'resolved_value': 'half_open',
            'candidates': ['half_open:[x1,y1,x2,y2)'],
            'edge_shift_px': 0,
            'note': '由生成代码确认为半开区间，无歧义',
        }
    return {
        'status': 'pending',
        'resolved_value': None,
        'candidates': list(D.ENDPOINT_CANDIDATES),
        'edge_shift_px': D.ENDPOINT_EDGE_SHIFT_PX,
        'note': '两种读法只让框的右/下边界相差 1px，属坐标解释差异；'
                '不能据此推断「标注误差 ≤1px」，也不能用作评测容差',
    }


def _glyph_boundary_error(cls: str) -> Dict[str, Any]:
    return {
        'status': 'unknown_uncalibrated',
        'value_px': None,
        'box_quality': {D.SOURCE_MANUAL: 'manual', D.SOURCE_MEASURED: 'ink',
                        D.SOURCE_APPROX: 'approximate'}[cls],
        'calibration': 'none',
        'note': '框与真实字形边界之间的误差没有标定测量；'
                '近似框连量级都无法给出，人工框/墨迹框同样未标定',
    }


def _evaluation_tolerance() -> Dict[str, Any]:
    return {
        'status': 'pending_protocol',
        'value_px': None,
        'note': '评测容差是评测协议人为选定的阈值，当前未定；'
                '在协议确定前，精确边界类指标一律不可用',
    }


def _counts(items: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))


def _readme_corrections(ctx: Context) -> List[Dict[str, Any]]:
    """说明.md 的统计口径与实际数据不符处：只记录，不修改原始来源。"""
    out = []
    pages = ctx.manifest
    n_inc = sum(1 for m in pages if m['status'] == D.STATUS_INCLUDED)
    n_pend = sum(1 for m in pages if m['status'] == D.STATUS_PENDING)
    n_exc = sum(1 for m in pages if m['status'] == D.STATUS_EXCLUDED)
    counts = {}
    for c in ctx.coords:
        counts[D.classify_source(c.get('source', ''))] = counts.get(D.classify_source(c.get('source', '')), 0) + 1
    out.append({
        'id': 'K-readme-manifest-count',
        'scope': 'readme', 'target': D.rel(D.README),
        'field': '清单记录条数', 'action': 'recorded_only_not_modified',
        'old': 15, 'new': len(pages),
        'basis': f'清单.json 实有 {len(pages)} 条（收录 {n_inc} / 待核 {n_pend} / 排除 {n_exc}）；'
                 f'说明.md 第三节目录树仍写「15 条记录」，是生成 8 条排除记录之前的旧值',
        'source': D.rel(D.MANIFEST_JSON),
        'impact': '仅文档描述；程序统计一律以清单.json 实读为准，不写死历史值',
    })
    out.append({
        'id': 'K-readme-source-count',
        'scope': 'readme', 'target': D.rel(D.README),
        'field': '人工标注组数', 'action': 'recorded_only_not_modified',
        'old': 23, 'new': counts.get(D.SOURCE_MANUAL, 0),
        'basis': f'按 source 字段归类，历史人工标注实际 {counts.get(D.SOURCE_MANUAL, 0)} 组'
                 f'（holdout 20 + dev 1）；说明.md 第三之二节写「23 处」，与同段列出的 20+1 自相矛盾',
        'source': D.rel(D.COORDS_JSON),
        'impact': '按来源分精度时以本文件 counts.groups_by_source 为准',
    })
    return out


def write_outputs(payload: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = out_dir / 'normalized_annotations.json'
    cor = out_dir / 'corrections.json'
    ann.write_text(json.dumps({
        'stage': payload['stage'],
        'generated_by': payload['generated_by'],
        'dataset_root': payload['dataset_root'],
        'source_files': payload['source_files'],
        'coordinate_system': payload['coordinate_system'],
        'group_bbox_rule': payload['group_bbox_rule'],
        'annotation_coverage': payload['annotation_coverage'],
        'annotation_coverage_note': payload['annotation_coverage_note'],
        'scope_notes': payload['scope_notes'],
        'precision_semantics': payload['precision_semantics'],
        'conventions': payload['conventions'],
        'counts': payload['counts'],
        'groups': payload['groups'],
        'pages': payload['pages'],
    }, ensure_ascii=False, indent=1, sort_keys=False) + '\n', encoding='utf-8')
    cor.write_text(json.dumps({
        'stage': payload['stage'],
        'policy': '只改可从原图或代码证据确认的字段；坐标数值一律保留原值；说明文件不改，只记录差异',
        'counts': payload['counts']['corrections_by_action'],
        'entries': payload['corrections'],
    }, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return {'normalized_annotations': str(ann), 'corrections': str(cor)}


def verify_consistency(ann: Dict[str, Any], cor: Dict[str, Any]) -> Dict[str, Any]:
    """规范化结果与原始记录逐项对应的自检；也用于重复运行一致性核对。"""
    coords = {D.group_id(c['book'], c['page'], c['region']): c for c in D.load_coords()}
    problems, checked = [], 0
    for g in ann['groups']:
        src = coords.get(g['group_id'])
        if src is None:
            problems.append(f"{g['group_id']}: 原始坐标记录缺失")
            continue
        checked += 1
        for field in ('main_bbox', 'right_bbox', 'left_bbox'):
            if list(src[field]) != list(g[field]):
                problems.append(f"{g['group_id']}.{field} 与原始记录不一致")
        for field in ('main_text', 'right_text', 'left_text'):
            if (src.get(field) or '') != (g.get(field) or ''):
                problems.append(f"{g['group_id']}.{field} 与原始记录不一致")
        if list(D.union_bbox([g['right_bbox'], g['left_bbox']])) != list(g['group_bbox']):
            problems.append(f"{g['group_id']}.group_bbox 不等于左右轨并集外接框")
        if not D.center_x(g['right_bbox']) > D.center_x(g['left_bbox']):
            problems.append(f"{g['group_id']}: 右轨中心未大于左轨中心")
        if not D.in_image(g['main_bbox'], g['image_size']):
            problems.append(f"{g['group_id']}.main_bbox 越出真实图")
    return {
        'checked_groups': checked,
        'expected_groups': len(coords),
        'corrections_entries': len(cor['entries']),
        'problems': problems,
        'ok': not problems and checked == len(coords),
    }


def update_revision_notes(out_dir: Path, payload: Dict[str, Any], report: Dict[str, Any],
                          paths: Dict[str, str]) -> Dict[str, Any]:
    """把本轮与上一轮的关键变化落到 revision_notes.json（按 release 名覆盖，不重复堆叠）。"""
    path = out_dir / 'revision_notes.json'
    notes = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'stage': D.STAGE, 'releases': []}
    if not any(r.get('release') == PREVIOUS_RELEASE['release'] for r in notes['releases']):
        notes['releases'].insert(0, PREVIOUS_RELEASE)

    s = report['summary']
    entry = {
        'release': RELEASE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'artifacts_sha256': {Path(v).name: D.sha256_file(Path(v)) for v in paths.values()},
        'checks': {'total': s['checks'], 'pass': s['pass'], 'fail': s['fail'],
                   'pending': s['pending'], 'failed_ids': s['failed_ids'],
                   'pending_ids': s['pending_ids']},
        'counts': {'groups_total': payload['counts']['groups_total'],
                   'pages_total': payload['counts']['pages_total'],
                   'corrections_total': payload['counts']['corrections_total'],
                   'groups_by_endpoint_status': payload['counts']['groups_by_endpoint_status'],
                   'groups_by_glyph_check': payload['counts']['groups_by_glyph_check']},
        'changes': RELEASE_CHANGES,
    }
    prev = next((r for r in notes['releases'] if r.get('release') == PREVIOUS_RELEASE['release']), None)
    if prev:
        entry['vs_previous'] = {
            'checks_delta': {'total': s['checks'] - prev['checks']['total'],
                             'pass': s['pass'] - prev['checks']['pass']},
            'corrections_delta': entry['counts']['corrections_total'] - prev['counts']['corrections_total'],
            'artifacts_changed': sorted(
                name for name, digest in entry['artifacts_sha256'].items()
                if prev.get('artifacts_sha256', {}).get(name) not in (None, digest)),
            'unchanged_counts': {k: (entry['counts'][k] == prev['counts'].get(k))
                                 for k in ('groups_total', 'pages_total', 'corrections_total')},
        }
    notes['releases'] = [r for r in notes['releases'] if r.get('release') != RELEASE]
    notes['releases'].append(entry)
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'written': D.rel(path), 'release': RELEASE}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='生成双轨小字页第一阶段规范化标注')
    ap.add_argument('--out-dir', default=str(D.OUT_DIR))
    ap.add_argument('--skip-fingerprint', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_before = fp_after = None
    if not args.skip_fingerprint:
        before = D.snapshot_tree(D.DATASET)
        (out_dir / 'source_fingerprints_before.json').write_text(
            json.dumps({'root': D.rel(D.DATASET), 'files': before}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        fp_before = len(before)

    ctx = Context()
    payload = build(ctx)
    paths = write_outputs(payload, out_dir)

    cor = json.loads(Path(paths['corrections']).read_text(encoding='utf-8'))
    ann = json.loads(Path(paths['normalized_annotations']).read_text(encoding='utf-8'))
    consistency = verify_consistency(ann, cor)
    report = run_checks(ctx)

    if not args.skip_fingerprint:
        after = D.snapshot_tree(D.DATASET)
        (out_dir / 'source_fingerprints_after.json').write_text(
            json.dumps({'root': D.rel(D.DATASET), 'files': after}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        fp_after = len(after)
        consistency['source_unchanged'] = D.diff_snapshots(before, after)

    manifest = {
        'stage': D.STAGE,
        'release': RELEASE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'outputs': {k: D.rel(Path(v)) for k, v in paths.items()},
        'artifact_sha256': {Path(v).name: D.sha256_file(Path(v)) for v in paths.values()},
        'validation_summary': report['summary'],
        'consistency': consistency,
        'source_file_count_before': fp_before,
        'source_file_count_after': fp_after,
    }
    (out_dir / 'normalization_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    rev = update_revision_notes(out_dir, payload, report, paths)

    if not args.quiet:
        c = payload['counts']
        print(f"组 {c['groups_total']} / 页 {c['pages_total']}；状态 {c['pages_by_status']}")
        print(f"来源 {c['groups_by_source']}；端点 {c['groups_by_convention']}")
        print(f"修正记录 {c['corrections_total']} 条 {c['corrections_by_action']}")
        print(f"自检 {'OK' if consistency['ok'] else 'FAIL'}：{consistency['checked_groups']}/{consistency['expected_groups']} 组逐项一致")
        print(f"字形完整性实际核对 {c['groups_by_glyph_check'].get('inspected_ok', 0) + c['groups_by_glyph_check'].get('inspected_uncertain', 0)} 组"
              f"（{c['groups_by_glyph_check']}）；端点状态 {c['groups_by_endpoint_status']}")
        if 'source_unchanged' in consistency:
            print(f"来源指纹未变：{consistency['source_unchanged']['identical']}")
        print('产物：' + '，'.join(f'{k}={v}' for k, v in manifest['outputs'].items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
