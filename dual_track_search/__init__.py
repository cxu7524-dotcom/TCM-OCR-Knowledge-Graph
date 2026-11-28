# -*- coding: utf-8 -*-
"""双轨搜索区域：从原图与自动布局得到「哪里找双轨」和「字有多大」。

三个阶段消费它：
  stage4（本阶段）产出 region + scale；
  stage5 在 region 内找中缝、确认双轨；
  stage6 精修轨框。

设计约束：
  1. 只用图像与自动布局；人工框、人工栏位只用于开发核对。
  2. 坐标统一为**原图** [x1,y1,x2,y2) 半开区间；半页裁块内计算的量都带 offset，
     输出时回映原图（`transform` 字段留证）。
  3. 同一图像 + 同一配置重复运行，区域顺序与编号必须一致。
  4. 估不出来就报失败原因，**不用无来源的默认值**顶替；继承同页尺度必须标
     `value_origin=inherited_from_page / partially_inherited`。
  5. 「范围外」与「应处理但未覆盖」分开记：前者不出区域，后者计入页面完成度。
"""

from __future__ import annotations

from .rules import (RuleCandidate, RuleGroup, detect_rule_candidates,
                    classify_rule_touching, merge_broken_rules, group_rules,
                    group_rule_runs, line_eligibility)
from .columns import (ColumnVerdict, validate_columns, reject_wide_boxes)
from .scale import ScaleEstimate, estimate_scale, glyph_bands, as_ink_mask
from .masks import (build_rule_mask, mask_digest, page_mask_record,
                    payload_mask_record, verify_record)
from .api import (RegionPlan, SearchRegion, plan_halfpage, plan_page,
                  regions_for_page, scale_for_page, rule_lines_for_page,
                  audit_for_page)
from .config import DEFAULT_CONFIG

__all__ = [
    'RuleCandidate', 'RuleGroup', 'detect_rule_candidates', 'classify_rule_touching',
    'merge_broken_rules', 'group_rules', 'group_rule_runs', 'line_eligibility',
    'ColumnVerdict', 'validate_columns', 'reject_wide_boxes',
    'ScaleEstimate', 'estimate_scale', 'glyph_bands', 'as_ink_mask',
    'build_rule_mask', 'mask_digest', 'page_mask_record', 'payload_mask_record',
    'verify_record',
    'SearchRegion', 'plan_halfpage', 'plan_page', 'RegionPlan',
    'regions_for_page', 'scale_for_page', 'rule_lines_for_page', 'audit_for_page',
    'DEFAULT_CONFIG',
]

__version__ = '2.0.0'
