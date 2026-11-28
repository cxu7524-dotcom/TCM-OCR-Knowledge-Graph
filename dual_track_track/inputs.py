# -*- coding: utf-8 -*-
"""第五阶段输入：只通过 `dual_track_search` 的公开读取入口拿第四阶段产物。

为什么不在本阶段重算搜索区域：
  阶段 4 的验收已经固定在 `search_regions.json`（内容指纹 67b14cf7…）。
  阶段 5 若自己去切区域，就等于偷偷改了输入，冻结对照会失效。

读取清单（全部记指纹）：
  stage4 search_regions.json / rule_masks.json
  stage2 cross_stage_contract.json
  stage3 evaluation_config.json / frozen_layouts.json / frozen_layout_labels.json
  stage3 baseline_snapshot.json（对照用）
  逐页原图 sha256
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dual_track_search import (audit_for_page, regions_for_page,
                               rule_lines_for_page, scale_for_page)

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'
S5 = ROOT / 'output' / 'dual_tracks' / 'stage5'

STAGE4_CONTENT_PREFIX = '67b14cf7f9a334ab'


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon_sha256(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


@dataclass
class PageInput:
    page_id: str
    image_path: str
    image_sha256: str
    image_size: List[int]
    halves: List[List[int]]
    regions: List[Dict[str, Any]]
    page_scale: Dict[str, Any]
    rule_lines: List[Dict[str, Any]]
    audit: Dict[str, Any]
    status: str = 'ok'

    def region_scale(self, region: Dict[str, Any]) -> Dict[str, Any]:
        sc = region.get('scale') or {}
        if sc.get('H') is None:
            sc = dict(self.page_scale or {})
            sc['value_origin'] = 'inherited_from_page'
            sc['inherited_note'] = '区域无 H，回落到页面级'
        return sc


@dataclass
class Bundle:
    payload: Dict[str, Any]
    masks: Dict[str, Any]
    contract: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    layouts: Dict[str, Any]
    labels: Dict[str, Any]
    baseline: Dict[str, Any]
    fingerprints: Dict[str, Any] = field(default_factory=dict)

    def pages(self) -> List[PageInput]:
        out: List[PageInput] = []
        lay = {p['page_id']: p for p in self.layouts.get('pages', [])}
        for p in self.payload['pages']:
            pid = p['page_id']
            row = lay.get(pid) or {}
            out.append(PageInput(
                page_id=pid,
                image_path=p['image_path'],
                image_sha256=row.get('sha256') or p.get('image_sha256') or '',
                image_size=list(p['image_size']),
                halves=[list(h) for h in p['halves']],
                regions=regions_for_page(self.payload, pid),
                page_scale=scale_for_page(self.payload, pid),
                rule_lines=rule_lines_for_page(self.payload, pid),
                audit=audit_for_page(self.payload, pid),
                status=p.get('status', 'ok')))
        return out

    def mask_record(self, page_id: str) -> Optional[Dict[str, Any]]:
        for rec in self.masks.get('pages', []):
            if rec['page_id'] == page_id:
                return rec
        return None


def load_bundle(root: Path = ROOT) -> Bundle:
    p4 = root / 'output' / 'dual_tracks' / 'stage4' / 'search_regions.json'
    m4 = root / 'output' / 'dual_tracks' / 'stage4' / 'rule_masks.json'
    payload = load(p4)
    masks = load(m4)
    contract = load(S2 / 'cross_stage_contract.json')
    eval_cfg = load(S3 / 'evaluation_config.json')
    layouts = load(S3 / 'frozen_layouts.json')
    labels = load(S3 / 'frozen_layout_labels.json')
    baseline = load(S3 / 'baseline_snapshot.json')

    fp_files = {
        'stage4_search_regions': p4,
        'stage4_rule_masks': m4,
        'stage2_cross_stage_contract': S2 / 'cross_stage_contract.json',
        'stage2_annotations': S2 / 'annotations.json',
        'stage2_dataset_splits': S2 / 'dataset_splits.json',
        'stage3_evaluation_config': S3 / 'evaluation_config.json',
        'stage3_frozen_layouts': S3 / 'frozen_layouts.json',
        'stage3_frozen_layout_labels': S3 / 'frozen_layout_labels.json',
        'stage3_baseline_snapshot': S3 / 'baseline_snapshot.json',
    }
    fingerprints = {k: {'path': str(p.relative_to(root)).replace('\\', '/'),
                        'sha256': sha256_of(p) if p.exists() else None,
                        'exists': p.exists()} for k, p in fp_files.items()}
    fingerprints['stage4_content_sha256'] = payload.get('determinism', {}).get(
        'content_sha256')
    fingerprints['stage4_content_prefix_ok'] = str(
        fingerprints['stage4_content_sha256']).startswith(STAGE4_CONTENT_PREFIX)
    fingerprints['stage4_release'] = payload.get('release')
    fingerprints['stage4_rule_masks_sha256_recorded'] = (payload.get('rule_masks') or {}).get(
        'sha256')
    return Bundle(payload=payload, masks=masks, contract=contract,
                  evaluation_config=eval_cfg, layouts=layouts, labels=labels,
                  baseline=baseline, fingerprints=fingerprints)


def page_image_path(page: PageInput, root: Path = ROOT) -> Path:
    return root / page.image_path


def image_sha256(path: Path) -> str:
    return sha256_of(path)
