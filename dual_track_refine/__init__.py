# -*- coding: utf-8 -*-
"""双轨子框收紧与首尾补笔（阶段 6）。

stage5-v4 的组 ID、成对关系、断点证据只读冻结；本阶段只精修左右轨外框。
"""

from .api import (audit_for_group, groups_for_page, load, refine_all,
                  refined_predictions)
from .config import DEFAULT_CONFIG
from .refine import refine_group

__all__ = ['DEFAULT_CONFIG', 'refine_all', 'refine_group', 'refined_predictions',
           'groups_for_page', 'audit_for_group', 'load']

__version__ = 'stage6-v1'
