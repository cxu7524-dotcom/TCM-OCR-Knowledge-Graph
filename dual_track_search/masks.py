# -*- coding: utf-8 -*-
"""栏线掩膜：给阶段 5 的可重建记录（不是必须读图才能复现）。

为什么不直接存 PNG：
  - 掩膜就是几十条 1~6px 宽的竖条，存 PNG 体积大、还依赖编码器版本；
  - 存「线段清单 + 编码说明 + sha256」，任何环境都能逐字节重建同一张掩膜，
    重建结果一比对哈希就知道有没有被改过。

v2 修正（交接缺口 1）——**掩膜按线段画，不按整条线的 bbox 画**：
  - 断线的断口不置位，贴字段不置位，原图墨迹原样保留；
  - `independent` 口径只画「合格线的独立段」，合格与否由 `rules.line_eligibility`
    统一判定（与成组定栏、列框拆分、尺度排除同一个判据）；
  - 含贴字段的混合断线整条不合格，只在 `all` 口径里留证据，不进 `independent`。

两种口径：
  all          所有线段（含贴字、含歧义混合线），供人工核对
  independent  合格线的独立段，阶段 5 定栏/尺度排除应该用这一张
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import numpy as np

ENCODING = ('uint8 0/1；按**线段**把 [x0,x1) × [y0,y1) 置 1（断口与贴字段不置位）；'
            '无平滑无缩放')

KEEP_FIELDS = ('x0', 'x1', 'y0', 'y1', 'kind', 'status', 'confidence', 'touches_text',
               'ambiguous', 'coverage', 'eligible_coverage', 'ink_density', 'width',
               'group_id', 'half_id', 'note', 'qualified')


def _segments(line: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = line.get('segments')
    if segs:
        return segs
    return [{'x0': line['x0'], 'x1': line['x1'], 'y0': line['y0'], 'y1': line['y1'],
             'touches_text': bool(line.get('touches_text')),
             'eligible': not bool(line.get('touches_text')),
             'confidence': line.get('confidence')}]


def _paint(mask: np.ndarray, segs: Sequence[Dict[str, Any]]) -> None:
    H, W = mask.shape
    for s in segs:
        x0 = max(0, int(s['x0']))
        x1 = min(W, int(s['x1']))
        y0 = max(0, int(s['y0']))
        y1 = min(H, int(s['y1']))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1


def build_rule_mask(lines: Sequence[Dict[str, Any]], image_size: Sequence[int],
                    scope: str = 'all') -> np.ndarray:
    """按记录重建 0/1 掩膜（逐段绘制，断口天然留空）。"""
    W, H = int(image_size[0]), int(image_size[1])
    mask = np.zeros((H, W), np.uint8)
    for ln in lines:
        segs = _segments(ln)
        if scope == 'independent':
            q = ln.get('qualified')
            if q is False:
                continue
            if q is None and ln.get('kind') not in ('independent', 'broken'):
                continue
            segs = [s for s in segs if s.get('eligible', not s.get('touches_text'))]
        _paint(mask, segs)
    return mask


def mask_digest(mask: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f'{mask.shape[0]}x{mask.shape[1]};uint8;0/1;{ENCODING}'.encode('utf-8'))
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return h.hexdigest()


def page_mask_record(page: Dict[str, Any]) -> Dict[str, Any]:
    """从一页的 stage4 记录生成可重建掩膜记录（含两种口径的哈希）。"""
    raw = page.get('rule_lines') or page.get('rule_candidates') or []
    lines: List[Dict[str, Any]] = []
    for i, ln in enumerate(raw):
        item = {k: ln.get(k) for k in KEEP_FIELDS if k in ln}
        item['line_index'] = i
        item['segments'] = _segments(ln)
        item['eligible_segments'] = [s for s in item['segments']
                                     if s.get('eligible', not s.get('touches_text'))]
        q = ln.get('qualified')
        item['qualified'] = (bool(q) if q is not None
                             else ln.get('kind') in ('independent', 'broken'))
        item['excluded_from_scale'] = item['qualified']
        lines.append(item)
    rec: Dict[str, Any] = {
        'page_id': page['page_id'],
        'image_size': list(page['image_size']),
        'halves': page.get('halves'),
        'encoding': ENCODING,
        'coord_system': 'original_image [x1,y1,x2,y2) 半开区间',
        'counts': {
            'all': len(lines),
            'qualified': sum(1 for x in lines if x['qualified']),
            'ambiguous': sum(1 for x in lines if x.get('ambiguous')),
            'touching': sum(1 for x in lines if x.get('status') == 'touching'),
            'segments': sum(len(x['segments']) for x in lines),
            'in_group': sum(1 for x in lines if x.get('group_id')),
        },
        'scale_exclusion_bands': [[int(s['x0']), int(s['x1']), int(s['y0']), int(s['y1'])]
                                  for x in lines if x['qualified']
                                  for s in x['eligible_segments']],
        'lines': lines,
        'masks': {},
    }
    for scope in ('all', 'independent'):
        m = build_rule_mask(lines, page['image_size'], scope)
        rec['masks'][scope] = {'sha256': mask_digest(m), 'ink_px': int(m.sum())}
    return rec


def payload_mask_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    pages = [page_mask_record(p) for p in payload.get('pages', [])]
    agg = {'total': 0, 'qualified': 0, 'ambiguous': 0, 'touching': 0,
           'segments': 0, 'in_group': 0}
    for p in pages:
        agg['total'] += p['counts'].get('all', 0)
        for k in ('qualified', 'ambiguous', 'touching', 'segments', 'in_group'):
            agg[k] += p['counts'].get(k, 0)
    return {'stage': 'stage4_rule_masks',
            'release': payload.get('release'),
            'generated_at': payload.get('generated_at'),
            'encoding': ENCODING,
            'note': ('阶段 5 用 independent 口径定栏；贴字线、含贴字段的混合断线只在 all '
                     '口径里留证据，不得拿来定栏，也不从字形投影中排除；断口与贴字段的'
                     '原图墨迹一律保留'),
            'summary': agg,
            'pages': pages}


def verify_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """重建并比对哈希；返回逐口径结果。"""
    out = {}
    for scope in ('all', 'independent'):
        m = build_rule_mask(record['lines'], record['image_size'], scope)
        d = mask_digest(m)
        out[scope] = {'rebuilt_sha256': d,
                      'recorded_sha256': (record.get('masks') or {}).get(scope, {}).get('sha256'),
                      'match': d == (record.get('masks') or {}).get(scope, {}).get('sha256'),
                      'ink_px': int(m.sum())}
    return out
