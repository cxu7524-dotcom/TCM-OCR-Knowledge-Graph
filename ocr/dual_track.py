# -*- coding: utf-8 -*-
"""ocr/dual_track.py —— 阶段 7：双轨小字框选 / 分轨复识接入整页 OCR 入口。

定位
----
双轨结果是**并列证据**，不替换正文：整页 OCR 的 blocks / tables / layout /
阅读顺序一律不动，本模块只在 OCRResult 上补两个字段
`dual_track_status` 与 `dual_track_evidence`（另加 `dual_track_run` 表示是否尝试过）。

链路（复用已冻结的阶段 4~6 实现，不另写一套检测）
-----------------------------------------------
  原图 + 整页 OCR blocks
    → 阶段 4 单页搜索区域（live: plan_page；replay: 冻结 search_regions.json）
    → 阶段 5 成对确认（dual_track_track.detect_page）
    → 阶段 6 精修（dual_track_refine.refine_group，含 final_validation）
    → 按侧判定：只有 `final_validation.ok` 且原图坐标有效的侧才送复识
    → 复识：裁原图 → cv2.resize(zoom) → np.rot90(90° 逆时针) → recognize_crop
    → 证据：组 ID / 区域 / 侧别 / 原图框 / 框选依据 / 文本 / 置信度 / 模型 / 单侧状态

单侧状态（evidence.side_status）
--------------------------------
  ok                 最终整框校验通过且原图坐标有效，已复识（text 可能为空）
  pending_validation 最终整框校验未通过 → 待核，不送复识，原因见 pending_reasons
  invalid_bbox       坐标非法 / 越界 → 待核，不送复识
  missing_side       该组只有一侧轨框
  recognize_error    送进复识但识别调用抛异常

状态枚举（dual_track_status）
----------------------------
  not_run    旧数据或本次未启用            evidence=None
  completed  正常完成（列表可为空=未检出）  evidence=list（可为 []）
  partial    部分区域完成，另有局部失败      evidence=已成功部分，失败位置进 audit
  failed     整体失败，保留整页 OCR          evidence=None
  skipped    因明确条件跳过                  evidence=None，原因进 audit
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S4 = ROOT / 'output' / 'dual_tracks' / 'stage4'

STATUSES = ('not_run', 'completed', 'partial', 'failed', 'skipped')
SIDE_STATUSES = ('ok', 'pending_validation', 'invalid_bbox', 'missing_side',
                 'recognize_error')

#: 证据记录的固定字段（顺序即报告与接口里的展示顺序）
EVIDENCE_FIELDS = (
    'evidence_id', 'group_id', 'page_id', 'region_id', 'region_source', 'side',
    'bbox', 'bbox_source', 'basis', 'text', 'conf', 'model', 'engine',
    'side_status', 'status_reason', 'crop', 'crop_path', 'crop_sha256',
    'rotation', 'crop_scale', 'pending_reasons', 'low_confidence',
    'fallback_source', 'group_bbox', 'region_bbox', 'ocr_blocks', 'review_evidence',
)

_DEFAULT_INK = {'seam_profile': {'config': {'ink_mask': {'block_size': 31, 'c': 14}}}}

_BOOK_STATS: Optional[Tuple[Any, Any]] = None


# --------------------------------------------------------------------- 小工具

def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _contract() -> Dict[str, Any]:
    """阶段 2 契约（只用来取 ink_mask 参数）；缺失时用与 stage4 默认一致的参数。"""
    try:
        return _load(S2 / 'cross_stage_contract.json')
    except (OSError, ValueError):
        return dict(_DEFAULT_INK)


def _write_png(path: Path, image: Any) -> bool:
    """写 PNG：项目路径含中文，cv2.imwrite 在 Windows 上会静默失败，改用 imencode+tofile。"""
    import cv2
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        return False
    buf.tofile(str(path))
    return Path(path).is_file()


def _book_reference() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """书级 / 全局列宽参考（与阶段 4 同一口径，来自阶段 3 冻结版面，不是本页 GT）。

    取不到就返回 (None, None)：阶段 4 的 plan_page 允许缺省，区域照样产出，
    只是 `narrow_columns` 参考变弱，会在审计里记 `reference=missing`。
    """
    global _BOOK_STATS
    if _BOOK_STATS is not None:
        return _BOOK_STATS
    try:
        import statistics
        layouts = _load(S3 / 'frozen_layouts.json')
        by_book: Dict[str, Dict[str, Any]] = {}
        all_medians: List[float] = []
        for book in layouts['by_book']:
            pages = [r for r in layouts['pages'] if r['book'] == book
                     and r['status'] == 'ok' and r.get('cols')]
            medians = [statistics.median([c['width'] for c in r['cols']]) for r in pages]
            all_medians.extend(medians)
            by_book[book] = {'cols': sum(r['n_cols'] for r in pages),
                             'median_width': round(statistics.median(medians), 3)
                             if medians else None,
                             'pages': len(pages)}
        glob = {'cols': sum(v['cols'] for v in by_book.values()),
                'median_width': round(statistics.median(all_medians), 3)
                if all_medians else None}
        _BOOK_STATS = (by_book, glob)
    except (OSError, ValueError, KeyError):
        _BOOK_STATS = (None, None)
    return _BOOK_STATS


def _bbox_ok(bbox: Any, width: int, height: int) -> Tuple[bool, str]:
    """原图坐标有效性（半开区间 `[x1,y1,x2,y2)`）。

    判据按**原始数值**比较，不先取整，避免 `x2 = width + 0.4` 被 round 成 width 而蒙混过关：
      - 必须是 4 元数组
      - 每个元素是有限数值（字符串、None、NaN、inf 全拒）
      - `0 <= x1 < x2 <= width`、`0 <= y1 < y2 <= height`
    因此 80 px 宽图像上 `x2=80` 合法、`x2=81` 拒绝（上一版容忍 `width+1`，是本次要修的漏洞）。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, 'not_quad'
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False, 'not_numeric'
    if not all(math.isfinite(v) for v in vals):
        return False, 'not_finite'
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return False, 'empty_or_inverted'
    if x1 < 0 or y1 < 0:
        return False, 'negative_origin'
    if width and x2 > width:
        return False, 'beyond_width'
    if height and y2 > height:
        return False, 'beyond_height'
    return True, ''


def _bbox_ints(bbox: Any) -> Optional[List[int]]:
    """转成半开区间的整数坐标；格式/数值不合法时返回 None（绝不抛异常）。

    `build_evidence` 在判定前就要把 bbox 写进证据，非数字坐标若直接 `int(float(v))`
    会让整页证据构造崩掉——单侧坏坐标不该拖垮另一侧。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return [int(round(v)) for v in vals]


def _crop_read(image: Any, bbox: Sequence[int], zoom: float = 1.0,
               min_conf: float = 0.15) -> Tuple[str, float, Dict[str, Any]]:
    """与 ocr/small_columns_attach._read 完全相同的裁图方式。

    竖排小字：裁原图框 → 可选 cv2.resize(CUBIC) 放大 → np.rot90 逆时针 → recognize_crop。
    不改预处理、不改阈值，保证与既有小字实验同一口径。
    """
    import cv2
    import numpy as np
    from ocr.recognize import recognize_crop

    x, y, X, Y = (int(round(float(v))) for v in bbox)
    h, w = image.shape[:2]
    # 不静默裁剪：越界就报错，交给调用方记 invalid_bbox（上一版 min/max 会把 81 裁成 80）
    if x < 0 or y < 0 or X > w or Y > h:
        raise ValueError(f'bbox_out_of_bounds: {[x, y, X, Y]} vs {[w, h]}')
    info: Dict[str, Any] = {'clip': [x, y, X, Y], 'rotation': 90, 'zoom': zoom}
    if X <= x or Y <= y:
        raise ValueError(f'empty_crop: {[x, y, X, Y]}')
    crop = image[y:Y, x:X]
    if zoom and float(zoom) != 1.0:
        crop = cv2.resize(crop, None, fx=float(zoom), fy=float(zoom),
                          interpolation=cv2.INTER_CUBIC)
    rot = np.ascontiguousarray(np.rot90(crop))
    info['crop_shape'] = [int(rot.shape[1]), int(rot.shape[0])]
    info['crop_png'] = rot
    text, conf = recognize_crop(rot, min_conf=min_conf)
    return text, float(conf), info


# ------------------------------------------------------------------ 单页上下文

def _book_of(page_id: str, image_path: str) -> str:
    """书级参考用哪本书：优先 `书/page_N` 形式的 page_id，否则退回图片所在目录名。

    生产通道的 page_id 是时间戳（`P2026…`），不含书名；若只用 page_id 取书，
    书级列宽参考会静默失效，同一张图会因 page_id 不同而算出不同的区域尺度。
    """
    if '/page_' in (page_id or ''):
        return page_id.split('/page_')[0]
    try:
        return Path(image_path).parent.name
    except (TypeError, ValueError):
        return ''


def _live_context(image: Any, image_path: str, page_id: str) -> Dict[str, Any]:
    """阶段 4 单页搜索：与 tools/dual_track_stage4_run.py:run_page 同一调用顺序。"""
    import cv2
    from dual_track_search import DEFAULT_CONFIG as S4_CFG, plan_page
    from ocr.vertical import page_layout

    h, w = image.shape[:2]
    lay = page_layout(image_path)
    bw = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 14)
    book, glob = _book_reference()
    book_name = _book_of(page_id, image_path)
    plan = plan_page(page_id, image_path, bw, (w, h), lay['halves'], lay['cols'],
                     S4_CFG, (book or {}).get(book_name), glob)
    d = plan.to_dict()
    return {
        'plan_source': 'live_stage4_plan_page',
        'book': book_name,
        'regions': d['regions'],
        'page_scale': d['page_scale'],
        'rule_lines': d['rule_lines'],
        'halves': d['halves'],
        'status': d['status'],
        'status_reason': d['status_reason'],
        'coverage': d['coverage'],
        'reference': 'stage3_frozen_layouts' if (book or {}).get(book_name) else 'missing',
    }


def _frozen_context(page_id: str, image: Any) -> Dict[str, Any]:
    """冻结回放：直接读阶段 4 search_regions.json 的该页（用于与在线结果对照）。"""
    payload = _load(S4 / 'search_regions.json')
    page = next((p for p in payload['pages'] if p['page_id'] == page_id), None)
    if page is None:
        raise KeyError(f'page not found in stage4 search_regions: {page_id}')
    return {
        'plan_source': 'frozen_stage4_search_regions',
        'regions': page.get('regions') or [],
        'page_scale': page.get('page_scale') or {},
        'rule_lines': page.get('rule_lines') or [],
        'halves': page.get('halves') or [],
        'status': page.get('status'),
        'status_reason': page.get('status_reason'),
        'coverage': page.get('coverage'),
        'reference': 'frozen',
    }


def _neighbor_bounds(group: Dict[str, Any],
                     groups: Sequence[Dict[str, Any]],
                     region_bbox: Sequence[int]) -> Dict[str, int]:
    """同区域相邻组约束首尾补笔范围（与 stage6 api.refine_all 一致）。"""
    gb = group['group_bbox']
    head_y, tail_y = int(region_bbox[1]), int(region_bbox[3])
    for other in groups:
        if other is group or other['region_id'] != group['region_id']:
            continue
        ob = other['group_bbox']
        if min(gb[2], ob[2]) - max(gb[0], ob[0]) <= 0:
            continue
        if ob[3] <= gb[1]:
            head_y = max(head_y, int(ob[3]))
        if ob[1] >= gb[3]:
            tail_y = min(tail_y, int(ob[1]))
    return {'head_y': head_y, 'tail_y': tail_y}


# -------------------------------------------------------------------- 证据构造

def _basis(record: Dict[str, Any], side: str) -> Dict[str, Any]:
    """框选依据：来源区域 + 尺度 + 接纳字形数 + 补笔 + 校验状态。"""
    s = (record.get('sides') or {}).get(side) or {}
    fv = s.get('final_validation') or {}
    sup = s.get('supplement') or {}
    edge = s.get('edge') or {}
    return {
        'lane': side,
        'region_id': record.get('region_id'),
        'region_source': record.get('region_source'),
        'scale_used': record.get('scale_used'),
        'validation_status': fv.get('status'),
        'validation_reason': fv.get('reason'),
        'flags': list(s.get('flags') or []),
        'edge_notes': [n.get('kind') for n in (edge.get('notes') or [])],
        'accepted_glyphs': len(s.get('accepted_glyphs') or []),
        'excluded_glyphs': len(s.get('excluded_glyphs') or []),
        'supplement_head': len((sup.get('head') or {}).get('accepted') or []),
        'supplement_tail': len((sup.get('tail') or {}).get('accepted') or []),
        'stage5_confirmed_span': record.get('confirmed_span'),
        'joint': ((record.get('sides') or {}).get(side) or {}).get(
            'final_validation', {}).get('joint_source'),
        'boundary_review': ((record.get('sides') or {}).get(side) or {}).get(
            'boundary_review'),
    }


def _overlapping_blocks(bbox: Sequence[int], blocks: Sequence[Any],
                        limit: int = 6) -> List[str]:
    out: List[str] = []
    for b in blocks or []:
        bb = getattr(b, 'bbox', None)
        if not bb or len(bb) < 4:
            continue
        if (min(bbox[2], bb[2]) - max(bbox[0], bb[0]) <= 0
                or min(bbox[3], bb[3]) - max(bbox[1], bb[1]) <= 0):
            continue
        out.append(getattr(b, 'block_ref', '') or getattr(b, 'line_id', ''))
        if len(out) >= limit:
            break
    return out


def build_evidence(records: Sequence[Dict[str, Any]], *,
                   image: Any, page_id: str, width: int, height: int,
                   recognizer: Optional[Callable[..., Tuple[str, float, Dict[str, Any]]]] = None,
                   crop_dir: Optional[Path] = None, zoom: float = 1.0,
                   blocks: Sequence[Any] = (),
                   rule_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """按侧判定 → 复识 → 证据。**纯函数式**：不做检测，只消费 stage6 记录。

    records 里每条是 refine_group 的返回结构；返回
    {evidence, audit, side_counts, failed_groups}。
    """
    recognizer = recognizer or _crop_read
    evidence: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    failed_groups: List[Dict[str, Any]] = []
    crop_dir = Path(crop_dir) if crop_dir else None
    if crop_dir is not None:
        if not crop_dir.is_absolute():
            crop_dir = ROOT / crop_dir
        crop_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        gid = rec.get('group_id')
        region_bbox = (rec.get('region_bbox') or rec.get('region_box')
                       or (rec.get('before') or {}).get('group_bbox'))
        sides = rec.get('sides') or {}
        for side in ('right', 'left'):
            s = sides.get(side)
            bbox = (s or {}).get('bbox_after')
            if not s or not bbox:
                evidence.append({
                    'evidence_id': f'{gid}:{side}', 'group_id': gid,
                    'page_id': page_id, 'region_id': rec.get('region_id'),
                    'region_source': rec.get('region_source'), 'side': side,
                    'bbox': None, 'bbox_source': None, 'basis': _basis(rec, side),
                    'text': '', 'conf': None, 'model': '', 'engine': '',
                    'side_status': 'missing_side',
                    'status_reason': '该组只有单侧轨框',
                    'crop': None, 'crop_path': None, 'crop_sha256': None,
                    'rotation': 90, 'crop_scale': zoom,
                    'pending_reasons': list(rec.get('pending_reasons') or []),
                    'low_confidence': bool(rec.get('low_confidence')),
                    'fallback_source': rec.get('region_source'),
                    'group_bbox': (rec.get('after') or {}).get('group_bbox'),
                    'region_bbox': region_bbox, 'ocr_blocks': [],
                    'review_evidence': None,
                })
                counts['missing_side'] = counts.get('missing_side', 0) + 1
                continue

            fv = s.get('final_validation') or {}
            ok = bool(fv.get('ok'))
            valid, why = _bbox_ok(bbox, width, height)
            ints = _bbox_ints(bbox)
            item = {
                'evidence_id': f'{gid}:{side}', 'group_id': gid, 'page_id': page_id,
                'region_id': rec.get('region_id'),
                'region_source': rec.get('region_source'), 'side': side,
                'bbox': ints,
                'bbox_source': ('refined' if ints is not None else None) if ok else 'tentative',
                'basis': _basis(rec, side),
                'text': '', 'conf': None, 'model': '', 'engine': '',
                'side_status': 'ok', 'status_reason': '',
                'crop': None, 'crop_path': None, 'crop_sha256': None,
                'rotation': 90, 'crop_scale': zoom,
                'pending_reasons': list(s.get('pending_reasons') or []),
                'low_confidence': bool(rec.get('low_confidence')),
                'fallback_source': rec.get('region_source'),
                'group_bbox': (rec.get('after') or {}).get('group_bbox'),
                'region_bbox': region_bbox,
                'ocr_blocks': _overlapping_blocks(ints, blocks) if ints else [],
                'review_evidence': None,
            }
            if not ok:
                item['side_status'] = 'pending_validation'
                item['status_reason'] = (fv.get('reason') or 'final_validation_not_ok')
                rv = fv.get('boundary_review') or {}
                if rv:
                    H = float((rec.get('scale_used') or {}).get('H') or 21)
                    img_path = _write_review_image(
                        image, item['bbox'] or bbox, crop_dir, gid, side, H)
                    item['review_evidence'] = {
                        'kind': 'pending_side_original_crop',
                        'original_bbox': rv.get('before'),
                        'reviewed_bbox': rv.get('after'),
                        'actions': rv.get('actions') or [],
                        'unresolved': rv.get('unresolved') or [],
                        'reason': item['status_reason'],
                        'image_path': img_path,
                    }
                    if rule_mask is not None and valid:
                        rule_evidence = _write_rule_evidence(
                            image, rule_mask, ints, crop_dir, gid, side, H,
                            rv.get('unresolved') or ())
                        if rule_evidence:
                            item['review_evidence']['rule_mask_review'] = rule_evidence
                audit.append({'kind': 'side_pending', 'group_id': gid, 'side': side,
                              'reason': item['status_reason'],
                              'validation_status': fv.get('status'),
                              'fallback_to_tentative': bool(fv.get('fallback_to_tentative')),
                              'bbox': ints})
            elif not valid:
                item['side_status'] = 'invalid_bbox'
                item['status_reason'] = why
                audit.append({'kind': 'side_invalid_bbox', 'group_id': gid,
                              'side': side, 'reason': why, 'bbox': ints,
                              'bbox_raw': repr(bbox)[:120]})
            else:
                try:
                    text, conf, info = recognizer(image, item['bbox'], zoom)
                    item['text'] = text or ''
                    item['conf'] = round(float(conf), 4) if conf else 0.0
                    item['crop'] = info.get('clip')
                    item['crop_sha256'] = (info.get('crop_sha256')
                                           or (_sha256_bytes(info['crop_png'].tobytes())
                                               if info.get('crop_png') is not None else None))
                    item['rotation'] = int(info.get('rotation', 90))
                    item['crop_scale'] = float(info.get('zoom', zoom))
                    if not item['text']:
                        audit.append({'kind': 'empty_reading', 'group_id': gid,
                                      'side': side, 'bbox': item['bbox']})
                except Exception as exc:                        # 单侧失败不影响另一侧
                    item['side_status'] = 'recognize_error'
                    item['status_reason'] = f'{type(exc).__name__}'
                    audit.append({'kind': 'recognize_error', 'group_id': gid,
                                  'side': side, 'reason': type(exc).__name__,
                                  'bbox': item['bbox']})
                else:
                    # 裁图落盘是**证据附属物**，写不出来只降级证据，不改判定
                    if crop_dir is not None and info.get('crop_png') is not None:
                        try:
                            p = crop_dir / (f'{_safe(gid)}_{_sha256_bytes(gid.encode())}'
                                            f'_{side}.png')
                            if _write_png(p, info['crop_png']):
                                item['crop_path'] = _rel(p)
                            else:
                                audit.append({'kind': 'crop_write_failed', 'group_id': gid,
                                              'side': side, 'reason': 'imencode_or_io'})
                        except Exception as exc:
                            audit.append({'kind': 'crop_write_failed', 'group_id': gid,
                                          'side': side, 'reason': type(exc).__name__})
            counts[item['side_status']] = counts.get(item['side_status'], 0) + 1
            evidence.append(item)

        statuses = {e['side_status'] for e in evidence if e['group_id'] == gid}
        if statuses and statuses <= {'pending_validation', 'invalid_bbox', 'missing_side'}:
            failed_groups.append({'group_id': gid, 'sides': sorted(statuses),
                                  'region_id': rec.get('region_id')})
    return {'evidence': evidence, 'audit': audit, 'side_counts': counts,
            'failed_groups': failed_groups}


def item_bbox(bbox: Sequence[Any]) -> List[int]:
    return [int(round(float(v))) for v in bbox]


def _rel(p: Path) -> str:
    """尽量给相对路径（便于报告引用）；不在项目内则原样返回绝对路径。"""
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')
    except (ValueError, OSError):
        return str(p)


def _safe(text: str) -> str:
    """文件名用 ASCII 安全字符：非 ASCII（如书名）折叠成下划线，避免跨平台编码坑。"""
    out = ''.join(c if (c.isascii() and (c.isalnum() or c in '-_.')) else '_'
                  for c in str(text))
    while '__' in out:
        out = out.replace('__', '_')
    return out.strip('_') or 'item'


def _write_review_image(image: Any, bbox: Sequence[int], crop_dir: Optional[Path],
                        gid: str, side: str, H: float) -> Optional[str]:
    """待核侧的原图证据：框上下左右各留一个字高，不做任何增强/改写。"""
    if crop_dir is None:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = max(6, int(round(H)))
        x0, y0 = max(0, x1 - pad), max(0, y1 - pad)
        X, Y = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
        if X <= x0 or Y <= y0:
            return None
        key = _sha256_bytes(gid.encode())
        p = Path(crop_dir) / f'review_{_safe(gid)}_{key}_{side}.png'
        if _write_png(p, image[y0:Y, x0:X]):
            return _rel(p)
    except Exception:
        return None
    return None


def _write_rule_evidence(image: Any, rule_mask: np.ndarray, bbox: Sequence[int],
                         crop_dir: Optional[Path], gid: str, side: str, H: float,
                         unresolved: Sequence[str]) -> Optional[Dict[str, Any]]:
    """输出与原图同坐标的栏线掩膜取证，保留待核判定。"""
    from dual_track_fix.ruling import masked_crop

    x1, y1, x2, y2 = bbox
    pad = max(6, round(H))
    rows, cols = rule_mask.shape
    near = {}
    for edge, y in (('head', y1), ('tail', y2)):
        if f'cut_{edge}' in unresolved:
            y0, y_stop = max(0, y - pad), min(rows, y + pad)
            near[edge] = int(rule_mask[y0:y_stop, x1:x2].sum())
    if not any(near.values()):
        return None
    context = [max(0, x1 - pad), max(0, y1 - pad),
               min(cols, x2 + pad), min(rows, y2 + pad)]
    a, b, c, d = context
    mask = rule_mask[b:d, a:c]
    result = {'status': 'needs_glyph_review', 'crop_bbox': context,
              'lane_bbox': list(bbox), 'rule_pixels_near_edges': near,
              'marked_pixels': int(mask.sum()), 'mask_path': None,
              'masked_crop_path': None}
    if crop_dir is None:
        return result
    try:
        edited = masked_crop(image[b:d, a:c], mask)
        if edited is None:
            return result
        crop, pixels = edited
        output = crop_dir / 'rule_masks'
        output.mkdir(parents=True, exist_ok=True)
        key = f'{_safe(gid)}_{_sha256_bytes(gid.encode())}_{side}'
        for field, content in (('mask_path', pixels), ('masked_crop_path', crop)):
            path = output / f'{key}_{field}.png'
            if _write_png(path, content):
                result[field] = _rel(path)
    except Exception:
        return result  # 图像附件遇到 I/O 错误时，保留坐标及已知证据。
    return result


def _fill_model(evidence: List[Dict[str, Any]], engine: str) -> None:
    from ocr.recognize import rec_model_name
    name = rec_model_name()
    for e in evidence:
        e['model'] = name if e['side_status'] == 'ok' else ''
        e['engine'] = engine if e['side_status'] == 'ok' else ''


# ---------------------------------------------------------------------- 主入口

def _stage8_validate_lane(b: Sequence[int], ink: np.ndarray, segs: Sequence[Any],
                          nb: Dict[str, Any], H: float) -> Dict[str, Any]:
    """复验已定轨框与可信栏线、相邻组的几何关系。"""
    x1, y1, x2, y2 = [int(v) for v in b]
    x1, x2 = max(0, x1), min(ink.shape[1], x2)
    conflict: List[str] = []
    for s in segs or []:
        sx1, sy1, sx2, sy2 = [int(v) for v in s]
        ix = min(x2, sx2) - max(x1, sx1)
        iy = min(y2, sy2) - max(y1, sy1)
        if ix > 0.25 * (x2 - x1) and iy > 0.5 * min(y2 - y1, sy2 - sy1):
            conflict.append('rule_conflict')
            break
    if nb:
        if nb.get('head_y') and y1 < int(nb['head_y']) - 2:
            conflict.append('neighbor_head_conflict')
        if nb.get('tail_y') and y2 > int(nb['tail_y']) + 2:
            conflict.append('neighbor_tail_conflict')
    return {'ok': not conflict, 'reason': ';'.join(conflict) or None,
            'status': 'rejected_to_tentative' if conflict else 'accepted',
            'fallback_to_tentative': bool(conflict),
            'review_reasons': [], 'conflict_reasons': conflict,
            'checks': {'rule_conflict': 'rule_conflict' not in conflict,
                       'neighbor_conflict': not any(
                           r in conflict for r in ('neighbor_head_conflict',
                                                  'neighbor_tail_conflict'))},
            'H': H}


def _stage8_ink_components(ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return labels, stats


def _ink_ids(labels: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> set[int]:
    import numpy as np

    return set(map(int, np.unique(labels[y1:y2, x1:x2]))) - {0}


def _crossing_ink_ids(labels: np.ndarray, x1: int, x2: int, y: int) -> set[int]:
    above, below = labels[y - 1:y + 1, x1:x2]
    return (set(map(int, above[above == below]))
            | set(map(int, above[1:][above[1:] == below[:-1]]))
            | set(map(int, above[:-1][above[:-1] == below[1:]]))) - {0}


def _stage8_pair_support(group: Dict[str, Any], labels: np.ndarray, stats: np.ndarray,
                         H: float) -> Dict[str, int]:
    """只统计确认区段内属于该侧的字形墨迹。"""
    import numpy as np

    ev = group.get('evidence') or {}
    x1, y1, x2, y2 = group['group_bbox']
    start, stop = ev.get('core_y') or (y1, y2)
    seam = ev.get('valley_x', (x1 + x2) / 2)
    support = {}
    for side in ('right', 'left'):
        a, lo, b, hi = group[f'{side}_bbox']
        slab = labels[max(start, lo):min(stop, hi), a:b]
        ids, counts = np.unique(slab[slab > 0], return_counts=True)
        support[side] = 0
        for i, count in zip(ids, counts):
            x, _, w, h, area = stats[i]
            if (0.25 * H <= w <= 1.65 * H and 0.22 * H <= h <= 1.6 * H
                    and area >= 0.035 * H * H
                    and (x >= seam - 2 if side == 'right' else x + w <= seam + 2)):
                support[side] += int(count)
    return support


def _stage8_support_without_rules(ink: np.ndarray, H: float
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """只在判断两侧字形存在时隔离连续细竖墨迹。"""
    from dual_track_fix.ruling import vertical_rule_mask

    return _stage8_ink_components(ink & ~vertical_rule_mask(ink, H))


def _outer_edge_rule_ids(labels: np.ndarray, stats: np.ndarray,
                         box: Sequence[int], H: float,
                         side: Optional[str]) -> set[int]:
    """返回完全位于轨框外缘细窄带内的独立细纵栏线段 ID（含共线短点），避免其阻塞标题修剪留白检测。"""
    if side not in ('right', 'left'):
        return set()
    x1, y1, x2, y2 = box
    margin = max(4, round(0.3 * H))
    max_w = max(3, round(0.25 * H))
    min_h = max(5, round(0.22 * H))
    out: set[int] = set()
    specks: List[Tuple[int, int, int]] = []
    rule_xs: List[Tuple[int, int]] = []
    for cid in _ink_ids(labels, x1, y1, x2, y2):
        sx, sy, sw, sh, _ = map(int, stats[cid])
        if sw > max_w:
            continue
        in_margin = (sx >= x2 - margin) if side == 'right' else (sx + sw <= x1 + margin)
        if not in_margin:
            continue
        if sh >= min_h and sh >= 2.0 * sw:
            out.add(cid)
            rule_xs.append((sx, sx + sw))
        elif sh < min_h:
            specks.append((cid, sx, sx + sw))
    if len(rule_xs) >= 2 and specks:
        rx_lo = min(a for a, _ in rule_xs)
        rx_hi = max(b for _, b in rule_xs)
        if rx_hi - rx_lo <= max_w + 1:
            for cid, sx1, sx2 in specks:
                if sx1 >= rx_lo and sx2 <= rx_hi:
                    out.add(cid)
    return out


def _title_components(labels: np.ndarray, stats: np.ndarray,
                      title: Optional[Sequence[int]], box: Sequence[int], H: float, seam_x: float,
                      core_y: Optional[Sequence[int]] = None,
                      side: Optional[str] = None
                      ) -> Dict[int, List[int]]:
    """从药名区域及有上方标题归属证据的跨中缝尾笔找侵入轨顶的笔画，保护确认区段首字。"""
    import numpy as np

    x1, y1, x2, y2 = box
    if y2 - y1 < 1.5 * H:
        return {}
    search_end = min(y2 - int(H), y1 + max(1, int(1.1 * H)))
    if search_end <= y1:
        return {}
    outer_rules = _outer_edge_rule_ids(labels, stats, box, H, side)
    members = {}
    has_title = bool(title and min(x2, title[2]) > max(x1, title[0]))
    ty1, ty2 = (int(title[1]), int(title[3])) if has_title else (0, 0)
    in_title_band = has_title and ty1 < y1 < ty2 + 0.35 * H
    core_start = int(core_y[0]) if (core_y and len(core_y) >= 2) else y1
    for i in _ink_ids(labels, x1, y1, x2, search_end):
        if i in outer_rules:
            continue
        x, y, w, h, _ = map(int, stats[i])
        # 必须从轨顶 y1 上方进入（绝不把确认区段内 y>=y1 起笔的完整首字当成药名）
        if y >= y1 or y >= core_start:
            continue
        crosses_seam = (x + 1 <= seam_x <= x + w - 1)
        if not crosses_seam:
            continue
        # 证据路径 1：显式落在检测到的 main_bbox 标题带内
        max_trim = 1.0 * H if w >= 0.9 * H else 0.6 * H
        from_main_box = (in_title_band
                         and ty1 <= y <= ty2 + 2
                         and min(y + h, int(ty2 + 0.25 * H)) - y >= 0.4 * h
                         and (w >= 3 or h >= 2)
                         and (y + h - y1) <= max_trim)
        # 证据路径 2：字形主体明确位于轨顶及确认区段上方（向上跨越至少 0.3H）且跨中缝
        from_upper_title = (y <= y1 - max(4, round(0.3 * H))
                            and w >= 0.5 * H and h >= 0.55 * H
                            and (y + h - y1) <= 0.85 * H)
        if from_main_box or from_upper_title:
            members[i] = [x, y, x + w, y + h]
    # 若存在明确位于轨顶上方（y <= y1 - 0.35H）且跨中缝的宽药名上部构件（如「蘇合油香」或「丁香」之「香」上部「禾」），
    # 将其正下方紧邻且下方紧跟整轨空白行（忽略外缘细纵栏线）的下部构件（如「香」之「日」）一并纳入药名归属
    canopy_xa = max(0, min(x1, int(seam_x - H)))
    canopy_xb = min(labels.shape[1], max(x2, int(seam_x + H)))
    canopy_ya = max(0, y1 - int(H))
    canopy = []
    for k in _ink_ids(labels, canopy_xa, canopy_ya, canopy_xb, min(y2 - int(H), y1 + 4)):
        if k in outer_rules:
            continue
        kx, ky, kw, kh, _ = map(int, stats[k])
        if (kx + 1 <= seam_x <= kx + kw - 1
                and kw >= 0.75 * H and kh >= 0.5 * H
                and ky <= y1 - max(4, round(0.35 * H))
                and ky < core_start):
            canopy.append([kx, ky, kx + kw, ky + kh])
    if canopy:
        cx1 = min(b[0] for b in canopy) - round(0.3 * H)
        cx2 = max(b[2] for b in canopy) + round(0.3 * H)
        cy2 = max(b[3] for b in canopy)
        sub_cands = {}
        for j in _ink_ids(labels, x1, y1, x2, min(y2 - int(H), y1 + round(1.0 * H))):
            if j in members or j in outer_rules:
                continue
            sx, sy, sw, sh, _ = map(int, stats[j])
            if sx >= cx1 and sx + sw <= cx2 and sy <= cy2 and (sy + sh - y1) <= 1.0 * H:
                sub_cands[j] = [sx, sy, sx + sw, sy + sh]
        if sub_cands:
            cand_end = max(b[3] for b in sub_cands.values())
            probe_end = min(y2, cand_end + max(8, round(0.5 * H)))
            probe = labels[cand_end:probe_end, x1:x2]
            if outer_rules:
                probe = np.where(np.isin(probe, list(outer_rules)), 0, probe)
            row_blank = ~np.any(probe > 0, axis=1)
            for off in (0, 1, 2):
                if off + 1 < len(row_blank) and row_blank[off] and row_blank[off + 1]:
                    gap_start = cand_end + off
                    gap_end = gap_start + 2
                    while gap_end - cand_end < len(row_blank) and row_blank[gap_end - cand_end]:
                        gap_end += 1
                    gap_len = gap_end - gap_start
                    # 空白带 >=3 行时将起点落在空白带内部，避免紧贴标题末行或首字顶行
                    stop_y = gap_start + min(2, gap_len // 2) if gap_len >= 3 else gap_start
                    members.update(sub_cands)
                    if stop_y > cand_end:
                        for k in list(sub_cands):
                            b0 = list(members[k])
                            b0[3] = max(b0[3], stop_y)
                            members[k] = b0
                    break
    return members


def _stage8_tail_before_frame(box: Sequence[int], ink: np.ndarray,
                              labels: np.ndarray, stats: np.ndarray,
                              crossing: set[int], H: float, tail_bound: int,
                              side: Optional[str], seam_x: Optional[float]
                              ) -> Optional[Tuple[List[int], List[List[int]]]]:
    """允许在可见的横版框前补全末字，保留与栏线粘连的疑点。"""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    pad = int(round(1.7 * H))
    xa, xb = max(0, x1 - pad), min(ink.shape[1], x2 + pad)
    yb = min(ink.shape[0], y2 + int(round(0.7 * H)) + 1)
    rule_width = max(3, int(round(3 * H)))
    if y2 - y1 < 2 * H or yb <= y2 or xb - xa < rule_width:
        return None
    local = ink[y2:yb, xa:xb]
    rule = cv2.morphologyEx(local, cv2.MORPH_OPEN,
                            np.ones((1, rule_width), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # 只采纳比单字宽得多、且横穿本轨的同一行墨迹。
    rows = np.flatnonzero(
        (rule.sum(axis=1) >= rule_width)
        & (rule[:, x1 - xa:x2 - xa].sum(axis=1) >= 0.7 * (x2 - x1)))
    if not rows.size:
        return None
    frame_top = y2 + int(rows[0])
    end = max(int(stats[i][1] + stats[i][3]) for i in crossing)
    if not y2 < end <= min(frame_top, tail_bound):
        return None
    left, right = x1, x2
    if seam_x is not None and side == 'right':
        left = min(x1, min(int(stats[i][0]) for i in crossing))
        if left < seam_x or x1 - left > 0.12 * H:
            return None
    if seam_x is not None and side == 'left':
        right = max(x2, max(int(stats[i][0] + stats[i][2]) for i in crossing))
        if right > seam_x or right - x2 > 0.12 * H:
            return None
    if (_ink_ids(labels, left, y1, x1, end) - crossing
            or _ink_ids(labels, x2, y1, right, end) - crossing):
        return None
    added = _ink_ids(labels, left, y2, right, end)
    boxes = []
    for i in sorted(added):
        x, y, w, h, _ = map(int, stats[i])
        # 非连在边界上的墨迹只接受同一末字内的分离笔画。
        if (x < left or x + w > right or y + h > end or w > 1.4 * H or h > 1.4 * H
                or (i not in crossing and y >= y2 + 0.25 * H)):
            return None
        boxes.append([x, y, x + w, y + h])
    if _crossing_ink_ids(labels, left, right, end):
        return None
    return [left, y1, right, end], boxes


def _trailing_title_char_trim(
    labels: np.ndarray, stats: np.ndarray,
    title: Optional[Sequence[int]], box: Sequence[int],
    group_x: Optional[Sequence[int]], H: float, seam_x: float,
    side: Optional[str],
) -> Optional[Dict[str, Any]]:
    """当 main_bbox 紧贴轨顶且上方存在跨中缝大字标题时，检测轨顶紧邻的单个居中标题续字（如「蟾酥」之「酥」）。

    必须同时满足五项独立证据，避免误裁真实双轨首字：
      1) main_bbox 底部紧贴 y1 且其正上方存在跨中缝大字标题；
      2) 轨顶紧邻的墨迹带下方存在 >=3 行整栏空白带；
      3) 该带内存在跨中缝大字连通域；
      4) 该带整栏墨迹宽度符合居中单字尺度（<=1.5H）且中心贴近中缝；
      5) 空白带下方左右两侧均具独立小字墨迹支持。
    """
    import numpy as np

    if not title or group_x is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    gx1, gx2 = int(group_x[0]), int(group_x[1])
    if y2 - y1 < 3.0 * H or gx2 <= gx1:
        return None
    if abs(int(title[3]) - y1) > max(4, round(0.2 * H)):
        return None

    # 1) 上方存在跨中缝大字标题
    above_ya = max(0, y1 - round(1.6 * H))
    above_xa = max(0, int(seam_x - H))
    above_xb = min(labels.shape[1], int(seam_x + H))
    has_upper_title_char = False
    for cid in _ink_ids(labels, above_xa, above_ya, above_xb, y1 + 2):
        cx, cy, cw, ch, _ = map(int, stats[cid])
        if (cw >= 0.7 * H and ch >= 0.85 * H
                and cx + 1 <= seam_x <= cx + cw - 1 and cy < y1):
            has_upper_title_char = True
            break
    if not has_upper_title_char:
        return None

    # 排除整栏左右外缘细纵栏线
    outer_r = _outer_edge_rule_ids(labels, stats, [gx1, y1, gx2, y2], H, 'right')
    outer_l = _outer_edge_rule_ids(labels, stats, [gx1, y1, gx2, y2], H, 'left')
    outer_all = outer_r | outer_l

    # 2) 在 [y1, y1 + 1.65*H) 内寻找首个连续墨迹带及其后 >=3 行整栏空白带
    scan_end = min(y2 - int(1.5 * H), y1 + round(1.65 * H))
    if scan_end <= y1 + round(0.8 * H):
        return None
    slab = labels[y1:scan_end, gx1:gx2]
    if outer_all:
        slab = np.where(np.isin(slab, list(outer_all)), 0, slab)
    has_ink = np.any(slab > 0, axis=1)
    nz = np.flatnonzero(has_ink)
    if not nz.size:
        return None
    blk_start_off = int(nz[0])
    if blk_start_off > max(5, round(0.35 * H)):
        return None
    # 寻找 blk_start_off 之后第一个长度 >=3 的空白带
    blk_end_off = None
    gap_end_off = None
    cur = blk_start_off
    n_rows = len(has_ink)
    while cur < n_rows:
        if not has_ink[cur]:
            g0 = cur
            while cur < n_rows and not has_ink[cur]:
                cur += 1
            if cur - g0 >= 3:
                blk_end_off = g0
                gap_end_off = cur
                break
        else:
            cur += 1
    if blk_end_off is None or gap_end_off is None:
        return None
    blk_y1 = y1 + blk_start_off
    blk_y2 = y1 + blk_end_off
    gap_end = y1 + gap_end_off
    if blk_y2 - blk_y1 < 0.75 * H or blk_y2 - blk_y1 > 1.45 * H:
        return None

    # 3) 带内存在跨中缝大字连通域
    band_slab = slab[blk_start_off:blk_end_off, :]
    band_ids = set(map(int, np.unique(band_slab))) - {0}
    if not any(
        stats[c][2] >= 0.55 * H and stats[c][3] >= 0.75 * H
        and stats[c][0] + 1 <= seam_x <= stats[c][0] + stats[c][2] - 1
        for c in band_ids
    ):
        return None

    # 4) 带内整栏墨迹符合居中单字尺度（排除左右双轨并列首字）
    xs_on = np.flatnonzero(np.any(band_slab > 0, axis=0))
    if not xs_on.size:
        return None
    ix1 = gx1 + int(xs_on[0])
    ix2 = gx1 + int(xs_on[-1]) + 1
    if (ix2 - ix1) > 1.5 * H or abs((ix1 + ix2) / 2.0 - seam_x) > 0.25 * H:
        return None

    # 5) 空白带下方左右两侧均具独立小字支持
    rem_slab = labels[gap_end:y2, gx1:gx2]
    if outer_all:
        rem_slab = np.where(np.isin(rem_slab, list(outer_all)), 0, rem_slab)
    seam_off = int(round(seam_x)) - gx1
    if not (2 <= seam_off <= gx2 - gx1 - 2):
        return None
    left_px = int((rem_slab[:, :seam_off] > 0).sum())
    right_px = int((rem_slab[:, seam_off:] > 0).sum())
    if min(left_px, right_px) < max(20, round(2.0 * H)):
        return None

    gap_len = gap_end - blk_y2
    stop_y = blk_y2 + min(2, gap_len // 2)
    return {
        'kind': 'trailing_title_char_trim',
        'from': y1,
        'to': stop_y,
        'title_char_bbox': [ix1, blk_y1, ix2, blk_y2],
        'blank_gap': [blk_y2, gap_end],
    }


def _stage8_following_title_stop(group: Dict[str, Any], box: Sequence[int],
                                 following: Sequence[Dict[str, Any]],
                                 labels: np.ndarray, stats: np.ndarray, H: float
                                 ) -> Optional[Dict[str, Any]]:
    """同栏另有稳定双轨或可见跨槽标题时，在断组留白处停止旧轨。"""
    import numpy as np

    x1, y1, x2, y2 = box
    mate_end = min(g[3] for s in ('right', 'left') if (g := group.get(f'{s}_bbox')))
    g_ev = group.get('evidence') or {}
    g_core = g_ev.get('core_y')
    gb = group.get('group_bbox') or box
    gx1, gx2 = int(gb[0]), int(gb[2])
    seam_x = g_ev.get('valley_x')
    for other in following:
        title = other.get('main_bbox')
        if not title:
            continue
        tx1, ty1, tx2, ty2 = title
        if (ty1 <= y1 + 2 * H or ty1 >= y2 - 0.5 * H or ty2 - ty1 < H
                or min(x2, tx2) - max(x1, tx1) < 0.55 * (x2 - x1)):
            continue
        if other.get('joint_evidence'):
            # 跨槽标题可能只挡住一侧。这里按原图取紧邻标题的完整留白。
            probe_start = max(y1 + int(2 * H), ty1 - int(1.6 * H))
            probe = labels[probe_start:ty1, x1:x2]
            min_gap, prefer_last = 2, True
        else:
            if (other['region_id'] != group.get('region_id')
                    or (other.get('evidence') or {}).get('windows', 0) < 8):
                continue
            o_core = (other.get('evidence') or {}).get('core_y')
            disjoint_core = bool(
                g_core and o_core and seam_x is not None
                and g_core[1] <= o_core[0] + H
                and g_core[1] < y2 - 1.5 * H
                and other['group_bbox'][1] > g_core[1]
            )
            if ty1 >= mate_end + H:
                probe_start = max(y1 + int(2 * H), mate_end - int(0.25 * H))
                probe = labels[probe_start:ty1, x1:x2]
                min_gap, prefer_last = max(3, round(0.65 * H)), False
                edge_rules = []
                for i in np.unique(probe):
                    if i:
                        x, _, w, h, _ = map(int, stats[i])
                        if w <= 0.25 * H and h >= 2.0 * H and (
                                x + w <= x1 + 0.3 * H or x >= x2 - 0.3 * H):
                            edge_rules.append(i)
                if edge_rules:
                    probe = np.where(np.isin(probe, edge_rules), 0, probe)
            elif disjoint_core:
                # 上下两组确认区段分离且中间夹有跨中缝药名大字时，在药名首字前的完整留白处停止上组。
                outer_all = (_outer_edge_rule_ids(labels, stats, [gx1, y1, gx2, y2], H, 'right')
                             | _outer_edge_rule_ids(labels, stats, [gx1, y1, gx2, y2], H, 'left'))
                search_ya = max(y1 + int(2 * H), int(g_core[1]) - round(0.8 * H))
                search_yb = min(y2 - int(H), int(other['group_bbox'][1]) + round(0.5 * H))
                title_tops = []
                for cid in _ink_ids(labels, gx1, search_ya, gx2, search_yb):
                    if cid in outer_all:
                        continue
                    cx, cy, cw, ch, _ = map(int, stats[cid])
                    if (cw >= 0.55 * H and ch >= 0.75 * H
                            and cx + 1 <= seam_x <= cx + cw - 1
                            and cy >= search_ya):
                        title_tops.append(cy)
                if not title_tops:
                    continue
                title_y1 = min(title_tops)
                probe_start = max(y1 + int(2 * H), title_y1 - round(1.5 * H))
                if title_y1 <= probe_start + 3:
                    continue
                probe = labels[probe_start:title_y1, gx1:gx2]
                ignore_ids = set(outer_all)
                for cid in np.unique(probe):
                    if cid and int(stats[cid][4]) <= 2:
                        ignore_ids.add(int(cid))
                if ignore_ids:
                    probe = np.where(np.isin(probe, list(ignore_ids)), 0, probe)
                blank = ~np.any(probe > 0, axis=1)
                edges = np.flatnonzero(np.diff(np.r_[False, blank, False]))
                gaps = [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if b - a >= 2 and a > 0]
                if gaps:
                    a, b = gaps[-1]
                    gap_start = probe_start + a
                    gap_end = probe_start + b
                    stop_y = gap_start + (1 if gap_end - gap_start >= 3 else 0)
                    return {'kind': 'following_title_stop', 'from': y2, 'to': stop_y,
                            'gap': [gap_start, gap_end],
                            'following_group_id': other['group_id'],
                            'following_title': [gx1, title_y1, gx2, int(other['group_bbox'][1])],
                            'reason': 'disjoint_core_y_title_break'}
                continue
            else:
                continue
        blank = ~np.any(probe, axis=1)
        edges = np.flatnonzero(np.diff(np.r_[False, blank, False]))
        gaps = [(a, b) for a, b in edges.reshape(-1, 2) if b - a >= min_gap and a > 0]
        if gaps:
            a, b = gaps[-1] if prefer_last else gaps[0]
            return {'kind': 'following_title_stop', 'from': y2, 'to': probe_start + int(a),
                    'gap': [probe_start + int(a), probe_start + int(b)],
                    'following_group_id': other['group_id'], 'following_title': list(title)}
    return None


def _stage8_outer_rule(box: Sequence[int], side: str, labels: np.ndarray,
                       stats: np.ndarray, H: float) -> Optional[Dict[str, Any]]:
    """分离轨框外侧的高置信独立长栏线、共线断续栏线或双边版框，前提是不横切任何字形。"""
    import numpy as np
    x1, y1, x2, y2 = box
    if y2 - y1 < 2 * H:
        return None
    pad = max(1, round(0.3 * H))
    outer = (x2 - pad, x2) if side == 'right' else (x1, x1 + pad)
    ids = _ink_ids(labels, max(x1, outer[0]), y1, min(x2, outer[1]), y2)
    cuts = []
    for i in sorted(ids):
        x, y, w, h, area = map(int, stats[i])
        if (w > 0.25 * H or h < 2.4 * H
                or area < 0.7 * h or min(y2, y + h) - max(y1, y) < H):
            continue
        lo, hi = (x1, x) if side == 'right' else (x + w, x2)
        if hi - lo < max(0.5 * H, 0.65 * (x2 - x1)) or min(lo - x1, x2 - hi) < 0:
            continue
        amount = lo - x1 + x2 - hi
        if not 0 < amount <= pad:
            continue
        strip = (hi, x2) if side == 'right' else (x1, lo)
        cut_x = hi if side == 'right' else lo
        if any(stats[j][0] < cut_x < stats[j][0] + stats[j][2]
               for j in _ink_ids(labels, strip[0], y1, strip[1], y2)):
            continue
        cuts.append((amount, {'kind': 'outer_rule_exclusion',
                               'rule_bbox': [x, y, x + w, y + h],
                               'from': [x1, x2], 'to': [lo, hi]}))
    if cuts:
        return min(cuts, key=lambda cut: cut[0])[1]

    # 共线断续纵栏线 / 与顶底版框相连成 T 形的纵栏线 / 双边外版框：
    # 按轨内局部行宽核验（避免顶底横版框把连通域全局宽度放大），且切线处须有 >=2px 纵向空白隔离带
    max_pad = max(2, round(0.5 * H))
    max_w = max(3, round(0.28 * H))
    ya = max(y1 + 3, 0)
    yb = min(y2 - 3, labels.shape[0])
    if yb - ya < round(1.8 * H):
        return None
    bin_interior = (labels[ya:yb, :] > 0)
    for d in range(max_pad, 1, -1):
        lo, hi = (x1, x2 - d) if side == 'right' else (x1 + d, x2)
        if hi - lo < max(0.55 * H, 0.65 * (x2 - x1)):
            continue
        strip = (hi, x2) if side == 'right' else (x1, lo)
        cut_x = hi if side == 'right' else lo
        gap_col = hi - 1 if side == 'right' else lo
        if not (x1 <= gap_col < x2):
            continue
        # 切线内侧紧邻列须几乎全空白，且无字形横向跨过切线 cut_x
        if int(bin_interior[:, gap_col].sum()) > max(3, round(0.015 * (yb - ya))):
            continue
        cross_x = (set(labels[ya:yb, cut_x - 1].tolist())
                   & set(labels[ya:yb, cut_x].tolist())) - {0}
        if cross_x:
            continue
        strip_bin = bin_interior[:, strip[0]:strip[1]]
        row_widths = strip_bin.sum(axis=1)
        covered_rows = int((row_widths > 0).sum())
        if covered_rows < max(round(2.0 * H), round(0.25 * (yb - ya))):
            continue
        # 局部绝大部分有墨行的宽度须符合细栏线尺度（允许双边外版框两根细线合计 <= 2*max_w）
        limit_w = max_w * 2 if d >= round(0.4 * H) else max_w
        if int((row_widths > limit_w).sum()) > max(2, round(0.03 * (yb - ya))):
            continue
        xs_on = np.flatnonzero(strip_bin.any(axis=0))
        rx1 = strip[0] + int(xs_on[0])
        rx2 = strip[0] + int(xs_on[-1]) + 1
        return {'kind': 'outer_rule_exclusion',
                'rule_bbox': [rx1, y1, rx2, y2],
                'from': [x1, x2], 'to': [lo, hi]}
    return None


def _stage8_frame_pullback(box: Sequence[int], ink: np.ndarray,
                           labels: np.ndarray, stats: np.ndarray, H: float
                           ) -> Optional[Dict[str, Any]]:
    """当轨尾压入横版框上沿 1–2px 且上方存在干净空白行时，退回到版框前的空白行。"""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    if y2 - y1 < 2.5 * H or y2 <= 4 or y2 >= ink.shape[0]:
        return None
    pad = int(round(1.5 * H))
    xa, xb = max(0, x1 - pad), min(ink.shape[1], x2 + pad)
    ya, yb = max(0, y2 - 3), min(ink.shape[0], y2 + 5)
    rule_w = max(3, int(round(3 * H)))
    if xb - xa < rule_w or yb <= ya:
        return None
    local = ink[ya:yb, xa:xb].astype(np.uint8)
    h_rule = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((1, rule_w), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    own = h_rule[:, x1 - xa:x2 - xa].sum(axis=1)
    if not np.any((h_rule.sum(axis=1) >= rule_w) & (own >= 0.65 * (x2 - x1))):
        return None
    # 从 y2-3 到 y2-1 寻找完全空白行，且空白行与 y2 之间不能有独立小字连通域
    for y_blank in range(y2 - 3, y2):
        if y_blank <= y1 + int(2 * H):
            continue
        if int(ink[y_blank, x1:x2].sum()) == 0:
            mid_ids = _ink_ids(labels, x1, y_blank + 1, x2, y2)
            # 若中间仅有连向底版框的宽墨迹或极小噪点，则安全退回至 y_blank + 1（半开上界=空白行下沿）
            if all(stats[i][2] >= 2.0 * H or stats[i][4] < max(4, round(0.02 * H * H))
                   for i in mid_ids):
                return {'kind': 'frame_top_pullback', 'from': y2, 'to': y_blank + 1,
                        'blank_row': y_blank}
    return None


def _stage8_review_lane(b: Sequence[int], ink: np.ndarray,
                        nb: Dict[str, Any], H: float, *,
                        title: Optional[Sequence[int]] = None,
                        seam_x: Optional[float] = None,
                        segs: Sequence[Sequence[int]] = (),
                        components: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        title_stop: Optional[Dict[str, Any]] = None,
                        side: Optional[str] = None,
                        core_y: Optional[Sequence[int]] = None,
                        group_x: Optional[Sequence[int]] = None,
                        max_complete_px: Optional[int] = None,
                        max_head_ratio: float = 0.5
                        ) -> Tuple[List[int], Dict[str, Any]]:
    """只在药名字形归属明确或边界能包全原有笔画时改框。"""
    x1, y1, x2, y2 = [int(v) for v in b]
    out = [x1, y1, x2, y2]
    review: Dict[str, Any] = {'before': list(b), 'after': list(b),
                              'actions': [], 'unresolved': [],
                              'head_bound': int(nb.get('head_y') or 0),
                              'tail_bound': int(nb.get('tail_y') or 0)}
    if not _bbox_ok(out, ink.shape[1], ink.shape[0])[0]:
        return out, review
    labels, stats = components if components is not None else _stage8_ink_components(ink)
    head_bound = review['head_bound']
    tail_bound = review['tail_bound'] or ink.shape[0]
    if title_stop:
        out[3] = tail_bound = y2 = title_stop['to']
        review['tail_bound'] = tail_bound
        review['actions'].append(title_stop)
    if side:
        outer_rule = _stage8_outer_rule(out, side, labels, stats, H)
        if outer_rule:
            x1, x2 = outer_rule['to']
            out[0], out[2] = x1, x2
            review['actions'].append(outer_rule)
    pullback = _stage8_frame_pullback(out, ink, labels, stats, H)
    if pullback:
        out[3] = y2 = tail_bound = pullback['to']
        review['actions'].append(pullback)
    title_members = {}
    outer_rules_for_title = _outer_edge_rule_ids(labels, stats, out, H, side)
    if seam_x is not None or title:
        eff_seam = seam_x if seam_x is not None else (title[0] + title[2]) / 2
        title_members = _title_components(labels, stats, title, out, H, eff_seam, core_y=core_y, side=side)
    if title_members:
        title_end = max(b[3] for b in title_members.values())
        removed = _ink_ids(labels, x1, y1, x2, title_end)
        non_title_ok = all(
            (i in title_members) or (i in outer_rules_for_title)
            or (stats[i][2] < 0.45 * H and stats[i][3] < 0.45 * H)
            for i in removed
        )
        if (title_end - y1 <= 1.1 * H and title_end < y2 - H
                and title_end >= head_bound and non_title_ok):
            out[1] = head_bound = title_end
            review['actions'].append({'kind': 'title_trim', 'from': y1, 'to': title_end,
                                      'components': list(title_members.values())})
        elif bool(title and title[1] < y1 < title[3] + 0.35 * H):
            review['unresolved'].append('title_conflict')
    elif title and seam_x is not None and group_x is not None:
        trail = _trailing_title_char_trim(
            labels, stats, title, out, group_x, H, seam_x, side,
        )
        if trail and trail['to'] >= head_bound and trail['to'] < y2 - H:
            out[1] = head_bound = trail['to']
            review['actions'].append(trail)

    step = int(max_complete_px) if max_complete_px is not None else max(1, min(4, int(round(0.2 * H))))
    min_area = max(4, round(0.02 * H * H))
    for edge, index in (('head', 1), ('tail', 3)):
        y = out[index]
        if y in (0, ink.shape[0]):
            continue
        crossing = _crossing_ink_ids(labels, x1, x2, y)
        if not crossing:
            continue
        if any(stats[i][4] < min_area for i in crossing):
            review['unresolved'].append(f'cut_{edge}')
            continue
        if edge == 'tail':
            completed = _stage8_tail_before_frame(out, ink, labels, stats, crossing, H,
                                                   tail_bound, side, seam_x)
            if completed is not None:
                complete_box, boxes = completed
                left, _, right, end = complete_box
                if not any(sx1 < right and left < sx2 and sy1 < end and y < sy2
                           for sx1, sy1, sx2, sy2 in segs):
                    out = complete_box
                    review['actions'].append({
                        'kind': 'tail_complete_before_frame', 'from': y, 'to': end,
                        'px': end - y, 'components': boxes,
                        'x_before': [x1, x2], 'x_after': [left, right]})
                    continue
        boxes = []
        for i in sorted(crossing):
            x, cy, w, h, _ = map(int, stats[i])
            boxes.append([x, cy, x + w, cy + h])
        outward = min(d[1] for d in boxes) if edge == 'head' else max(d[3] for d in boxes)
        low, high = sorted((outward, y))
        allowed_step = step
        if (edge == 'head' and title and outward >= title[3]
                and all(bx1 >= x1 and bx2 <= x2 and 0.4 * H <= by2 - by1 <= 1.15 * H
                        and 0.3 * H <= bx2 - bx1 <= 1.15 * H
                        for bx1, by1, bx2, by2 in boxes)):
            # 边界外的笔画与框内字形相连，且整块属于本轨与标题下方。
            # 只有框内还有另一块同尺度字形时才允许超过基础补笔余量。
            body_ids = _ink_ids(labels, x1, y, x2, min(y2, y + round(2 * H))) - crossing
            if any(0.3 * H <= stats[i][2] <= 1.15 * H
                   and 0.4 * H <= stats[i][3] <= 1.15 * H
                   and stats[i][0] >= x1 and stats[i][0] + stats[i][2] <= x2
                   for i in body_ids):
                allowed_step = max(step, round(max_head_ratio * H))
        added = _ink_ids(labels, x1, low, x2, high) if high - low <= allowed_step else set()
        if (high - low > allowed_step or not head_bound <= outward <= tail_bound
                or added - crossing or any(
                    bx1 < x1 or bx2 > x2 or by2 - by1 > 1.15 * H
                    for bx1, by1, bx2, by2 in boxes)
                or any(sx1 < x2 and x1 < sx2 and sy1 < high and low < sy2
                       for sx1, sy1, sx2, sy2 in segs)):
            review['unresolved'].append(f'cut_{edge}')
            continue
        out[index] = outward
        review['actions'].append({'kind': f'{edge}_complete', 'px': high - low,
                                  'from': y, 'to': outward, 'components': boxes})
    review['after'] = list(out)
    return out, review


def _stage8_record(group: Dict[str, Any], region_box: Sequence[int],
                   ink: Optional[np.ndarray] = None,
                   segs: Optional[Sequence[Any]] = None,
                   nb: Optional[Dict[str, Any]] = None,
                   components: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   following: Sequence[Dict[str, Any]] = ()
                   ) -> Dict[str, Any]:
    """逐侧复核检测框，并记录原框、复核框及其复识资格。"""
    H = float((group.get('scale_used') or {}).get('H') or 21)
    if components is None and ink is not None and ink.size:
        components = _stage8_ink_components(ink)
    sides: Dict[str, Any] = {}
    pending_all: List[str] = []
    for side in ('right', 'left'):
        b = group.get(f'{side}_bbox')
        if not b:
            continue
        review: Dict[str, Any] = {'before': list(b), 'after': list(b),
                                  'actions': [], 'unresolved': []}
        b_final = list(b)
        if components is None:
            fv = {'ok': False, 'status': 'missing_boundary_evidence',
                  'reason': 'missing_boundary_evidence', 'fallback_to_tentative': True}
        else:
            title_stop = (_stage8_following_title_stop(group, b, following, *components, H)
                          if following else None)
            detection = group.get('evidence') or {}
            frame_y = detection.get('bottom_frame_y')
            bounds = dict(nb or {})
            if frame_y is not None:
                bounds['tail_y'] = min(frame_y, bounds.get('tail_y') or frame_y)
            gb_orig = group.get('group_bbox')
            b_final, review = _stage8_review_lane(
                b, ink, bounds, H, title=group.get('main_bbox'),
                seam_x=detection.get('valley_x'),
                segs=segs or (), components=components, title_stop=title_stop, side=side,
                core_y=detection.get('core_y'),
                group_x=[gb_orig[0], gb_orig[2]] if gb_orig else None)
            if frame_y is not None:
                review['bottom_frame_y'] = frame_y
            fv = _stage8_validate_lane(b_final, ink, segs or [], bounds, H)
            fv['boundary_review'] = review
            fv['review_reasons'] = list(review['unresolved'])
            fv.setdefault('checks', {}).update(
                cut_head='cut_head' not in review['unresolved'],
                cut_tail='cut_tail' not in review['unresolved'],
                title_conflict='title_conflict' not in review['unresolved'])
            if review['unresolved']:
                if fv['ok']:
                    fv['status'] = 'pending_cut_review'
                fv['ok'] = False
                existing = [reason for reason in (fv.get('reason') or '').split(';') if reason]
                fv['reason'] = ';'.join(dict.fromkeys(existing + review['unresolved']))
        joint = group.get('joint_evidence') or {}
        if joint:
            # 原样带出 joint_propose 的完整证据（槽位、标题带、区段、字形高、
            # 左右墨迹外接），报告要逐项写触发依据，不能只留缝的坐标。
            fv['joint_source'] = dict(joint)
        side_pending: List[str] = []
        if not fv['ok']:
            side_pending.append('final_validation:' + str(fv.get('reason')))
            pending_all.extend(side_pending)
        sides[side] = {'bbox_before': list(b), 'bbox_after': b_final,
                       'boundary_review': review,
                       'flags': [], 'pending_reasons': side_pending,
                       'final_validation': fv}
    gb = group.get('group_bbox')
    after_r = (sides.get('right') or {}).get('bbox_after')
    after_l = (sides.get('left') or {}).get('bbox_after')
    if after_r and after_l:
        gb = [min(after_l[0], after_r[0]), min(after_l[1], after_r[1]),
              max(after_l[2], after_r[2]), max(after_l[3], after_r[3])]
    return {'group_id': group['group_id'], 'page_id': group['page_id'],
            'region_id': group.get('region_id'),
            'region_source': group.get('region_source') or 'slot_split',
            'low_confidence': any(not s['final_validation']['ok'] for s in sides.values()),
            'stage5_pending': False,
            'before': {'right_bbox': (sides.get('right') or {}).get('bbox_before'),
                       'left_bbox': (sides.get('left') or {}).get('bbox_before'),
                   'group_bbox': group.get('group_bbox')},
            'after': {'right_bbox': (sides.get('right') or {}).get('bbox_after'),
                      'left_bbox': (sides.get('left') or {}).get('bbox_after'),
                      'group_bbox': list(gb) if gb else None},
            'sides': sides, 'flags': [], 'pending_reasons': pending_all,
            'region_bbox': list(region_box), 'neighbor_bounds': dict(nb or {}),
            'scale_used': group.get('scale_used'), 'refine': 'validated_stage8_v4'}


def run_dual_track(image_path: str, page_id: str, *, blocks: Sequence[Any] = (),
                   mode: Optional[str] = None, is_demo: bool = False,
                   engine: str = 'rapidocr', width: int = 0, height: int = 0,
                   recognizer: Optional[Callable[..., Any]] = None,
                   crop_dir: Optional[Path] = None, zoom: float = 1.0,
                   max_groups: Optional[int] = None,
                   cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """整页 → 双轨证据。**永不抛异常**，失败一律降级为 status=failed/skipped。

    返回 {'status', 'evidence', 'attempted', 'audit', 'groups', 'timing_s', ...}
    """
    t0 = time.time()
    out: Dict[str, Any] = {'status': 'not_run', 'evidence': None, 'attempted': False,
                           'audit': [], 'groups': 0, 'sides': {}, 'timing_s': 0.0,
                           'plan': None, 'error': ''}
    cfg = cfg or {}
    mode = (mode or cfg.get('mode') or 'auto').strip().lower()

    def done(status: str, evidence=None, **kw) -> Dict[str, Any]:
        out.update(status=status, evidence=evidence, **kw)
        out['timing_s'] = round(time.time() - t0, 3)
        return out

    if mode == 'off':
        out['audit'].append({'kind': 'skip', 'reason': 'mode_off',
                             'detail': 'TCM_OCR_DUAL_TRACK_MODE=off'})
        return done('not_run')
    if is_demo or engine == 'stub':
        out['audit'].append({'kind': 'skip', 'reason': 'demo_or_stub_page',
                             'detail': f'engine={engine}, is_demo={is_demo}'})
        return done('skipped')
    if not image_path or not Path(image_path).is_file():
        out['audit'].append({'kind': 'skip', 'reason': 'image_missing',
                             'detail': image_path or '(empty)'})
        return done('skipped')

    try:
        from ocr.vertical import imread_unicode
        image = imread_unicode(image_path)
    except Exception as exc:
        return done('failed', error=f'{type(exc).__name__}: image_read')
    if image is None:
        return done('failed', error='image_unreadable')
    h, w = image.shape[:2]
    width, height = width or w, height or h
    out['attempted'] = True          # 从这里往后才算「真的跑过」（失败也算跑过）

    try:
        ctx = (_frozen_context(page_id, image) if mode == 'replay'
               else _live_context(image, image_path, page_id))
    except Exception as exc:
        out['audit'].append({'kind': 'stage4_failed', 'reason': type(exc).__name__,
                             'detail': str(exc)[:200]})
        return done('failed', error=f'{type(exc).__name__}: stage4_plan')
    out['plan'] = {'source': ctx['plan_source'], 'status': ctx.get('status'),
                   'regions': len(ctx['regions']), 'reference': ctx.get('reference')}

    groups: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    group_errors: List[Dict[str, Any]] = []
    try:
        from dual_track_track import (DEFAULT_CONFIG as S5_CFG, PageInput,
                                      detect_page, page_ink, rule_segments_for_region)
        contract = _contract()
        page = PageInput(page_id=page_id, image_path=image_path,
                         image_sha256=_sha256_file(Path(image_path)),
                         image_size=[w, h], halves=ctx['halves'],
                         regions=ctx['regions'], page_scale=ctx['page_scale'],
                         rule_lines=ctx['rule_lines'], audit={})
        ink = page_ink(image, contract)
        detector = str((cfg.get('detector') or 'stage5')).strip().lower()
        out['plan']['detector'] = detector
        if detector in ('stage8', 'stage8-v1', 'slot'):
            # 阶段八修正版：列槽周期切分 + 子框裁到槽内 + 按墨迹收紧 x
            # 列内检测仍复用 ocr.small_columns.propose_in_column；
            # 参数一律来自 dual_track_fix.settings（与跑批同一份来源）。
            import cv2
            from dual_track_fix.detect import (
                _overlap_1d,
                _superseded_windows,
                _uncovered_side_ink,
                detect as detect_stage8,
            )
            from dual_track_fix.settings import detect_options, effective_options
            H8 = float((ctx.get('page_scale') or {}).get('H') or 20)
            eff = effective_options({**(cfg.get('stage8') or {}), 'detector': detector})
            out['plan']['effective'] = eff
            gray8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            d8 = detect_stage8(image, page_id, H8, options=detect_options(eff), gray=gray8)
            ink_components = _stage8_ink_components(ink)
            separated_ink = None
            min_support = max(8, round(0.5 * H8))
            # 先对全部原始提案执行双侧字形支持校验，防止未过检的伪长框吞掉合格候选
            proposals = d8.get('all_proposals') if 'all_proposals' in d8 else d8['groups']
            pair_valid: List[Dict[str, Any]] = []
            for group in proposals:
                support = _stage8_pair_support(group, *ink_components, H8)
                if min(support.values()) < min_support:
                    if separated_ink is None:
                        separated_ink = _stage8_support_without_rules(ink, H8)
                    separated_support = _stage8_pair_support(group, *separated_ink, H8)
                    if min(separated_support.values()) >= min_support:
                        out['audit'].append({'kind': 'rule_attached_pair_recovered',
                                             'group_id': group['group_id'],
                                             'raw_side_owned_pixels': support,
                                             'separated_side_owned_pixels': separated_support})
                        pair_valid.append(group)
                        continue
                    out['audit'].append({'kind': 'insufficient_pair_shape_support',
                                         'group_id': group['group_id'],
                                         'region_id': group['region_id'],
                                         'side_owned_pixels': support,
                                         'separated_side_owned_pixels': separated_support})
                    continue
                pair_valid.append(group)

            # 仅在双侧校验通过的候选之间执行同槽短窗/嵌套去重与跨槽联合去重
            groups, sup_audit = _superseded_windows(pair_valid, H8, gray=gray8)
            joint_audit: List[Dict[str, Any]] = []
            confirmed_joints = [g for g in groups if g.get('region_source') == 'joint_auto']
            if confirmed_joints:
                drop_joint_ids: set[str] = set()
                for jg in confirmed_joints:
                    jgb, jmb = jg['group_bbox'], jg.get('main_bbox')
                    j_lanes = [jg.get('right_bbox'), jg.get('left_bbox'), jmb]
                    for sg in groups:
                        if sg is jg or sg.get('region_source') == 'joint_auto':
                            continue
                        if sg['group_id'] in drop_joint_ids:
                            continue
                        sgb = sg['group_bbox']
                        y_ov = _overlap_1d(sgb[1::2], jgb[1::2])
                        x_ov = _overlap_1d([sgb[0], sgb[2]], [jgb[0], jgb[2]])
                        if (y_ov < 0.75 * max(1, sgb[3] - sgb[1])
                                or x_ov < 0.5 * max(1, sgb[2] - sgb[0])):
                            continue
                        # 按实际左右轨框+标题框（而非联合外接矩形 group_bbox）计算未覆盖墨迹，
                        # 防止联合外接框跨越中间空隙误吞独立双轨候选
                        uncov_r = _uncovered_side_ink(ink, sg['right_bbox'], j_lanes)
                        uncov_l = _uncovered_side_ink(ink, sg['left_bbox'], j_lanes)
                        if min(uncov_r, uncov_l) < min_support:
                            drop_joint_ids.add(sg['group_id'])
                            joint_audit.append({
                                'kind': 'cross_slot_joint_superseded',
                                'group_id': sg['group_id'],
                                'superseded_by': jg['group_id'],
                                'y_overlap': y_ov,
                                'x_overlap': x_ov,
                                'uncovered_side_ink': {'right': uncov_r, 'left': uncov_l},
                                'reason': 'no_independent_dual_track_outside_joint',
                            })
                if drop_joint_ids:
                    groups = [g for g in groups if g['group_id'] not in drop_joint_ids]

            # 传递闭包归并 superseded_by，确保每条去重记录指向最终保留的有效组 ID
            sup_target: Dict[str, str] = {}
            for row in sup_audit:
                sup_target[row['group_id']] = row['superseded_by']
            for row in joint_audit:
                sup_target[row['group_id']] = row['superseded_by']

            def _final_keeper(gid: str) -> str:
                seen: set[str] = set()
                cur = gid
                while cur in sup_target and cur not in seen:
                    seen.add(cur)
                    cur = sup_target[cur]
                return cur

            for row in sup_audit:
                row = dict(row)
                orig_sup = row['superseded_by']
                final_sup = _final_keeper(orig_sup)
                if final_sup != orig_sup:
                    row['via_superseded_by'] = orig_sup
                    row['superseded_by'] = final_sup
                out['audit'].append({'kind': 'window_superseded', **row})
            for row in joint_audit:
                row = dict(row)
                orig_sup = row['superseded_by']
                final_sup = _final_keeper(orig_sup)
                if final_sup != orig_sup:
                    row['via_superseded_by'] = orig_sup
                    row['superseded_by'] = final_sup
                out['audit'].append(row)

            groups.sort(key=lambda g: (g['group_bbox'][1], g['group_bbox'][0]))
            for k, g in enumerate(groups):
                g['index'] = k
            # 阶段 6 精修只认 `scale_used`（它按 H 判字号），这里补上页面尺度；
            # 阶段 5 的 pending / low_confidence 语义在修正版里不产生，置 False。
            for _g in groups:
                _g.setdefault('scale_used', ctx.get('page_scale') or {'H': H8})
                _g.setdefault('pending', False)
                _g.setdefault('low_confidence', False)
            # 列槽即区域，追加到 ctx['regions'] 供阶段 6 查 region bbox
            ctx['regions'] = list(d8['regions']) + list(ctx['regions'])
            out['groups'] = len(groups)
            out['plan']['stage5_status'] = 'stage8_v1'
            out['plan']['slot_audit'] = [
                {'action': a.get('action'), 'pitch': a.get('pitch'),
                 'n_cut': a.get('n_cut')} for a in (d8.get('slot_audit') or [])]
        else:
            det = detect_page(page, ink, S5_CFG, contract)
            groups = det['groups']
            out['groups'] = len(groups)
            out['plan']['stage5_status'] = det.get('status')
    except Exception as exc:
        out['audit'].append({'kind': 'stage5_failed', 'reason': type(exc).__name__,
                             'detail': str(exc)[:200]})
        return done('failed', error=f'{type(exc).__name__}: stage5_detect')

    if max_groups is not None and len(groups) > max_groups:
        out['audit'].append({'kind': 'truncated', 'kept': int(max_groups),
                             'dropped': len(groups) - int(max_groups)})
        groups = groups[:int(max_groups)]

    try:
        from dual_track_refine import DEFAULT_CONFIG as S6_CFG, refine_group
        from dual_track_track.inputs import Bundle  # noqa: F401  (保持导入路径一致)
        s6 = cfg.get('refine') or S6_CFG
        region_by_id = {r.get('region_id'): r for r in ctx['regions']}
        refine_fn = refine_group
        if detector in ('stage8', 'stage8-v1', 'slot'):
            by_group_top = sorted(groups, key=lambda g: g['group_bbox'][1])

            def refine_fn(group, region_box, ink, segments, _cfg, bounds):
                rules = [(x1, y1, x2, y2) for x1, x2, y1, y2 in segments]
                following = [g for g in by_group_top
                             if g['group_bbox'][1] > group['group_bbox'][1]]
                return _stage8_record(group, region_box, ink, rules, bounds, ink_components,
                                      following)
        for g in sorted(groups, key=lambda x: (x['group_bbox'] or [0, 0, 0, 0])[1]):
            reg = region_by_id.get(g['region_id'])
            if reg is None:
                group_errors.append({'group_id': g['group_id'],
                                     'reason': 'region_missing'})
                continue
            try:
                segs = rule_segments_for_region(ctx['rule_lines'], reg['bbox'])
                nb = _neighbor_bounds(g, groups, reg['bbox'])
                rec = refine_fn(g, reg['bbox'], ink, segs, s6, nb)
                rec['neighbor_bounds'] = nb
                rec['region_bbox'] = list(reg['bbox'])
                rec['scale_used'] = g.get('scale_used')
                rec['confirmed_span'] = g.get('confirmed_span')
                records.append(rec)
            except Exception as exc:
                group_errors.append({'group_id': g['group_id'],
                                     'reason': f'{type(exc).__name__}',
                                     'region_id': reg.get('region_id')})
    except Exception as exc:
        out['audit'].append({'kind': 'stage6_failed', 'reason': type(exc).__name__,
                             'detail': str(exc)[:200]})
        return done('failed', error=f'{type(exc).__name__}: stage6_refine')

    rule_mask = None
    if detector in ('stage8', 'stage8-v1', 'slot') and any(r['low_confidence'] for r in records):
        from dual_track_fix.ruling import ruling_mask

        rule_mask = ruling_mask(ink, H8, *ink_components)
    built = build_evidence(records, image=image, page_id=page_id, width=width,
                           height=height, recognizer=recognizer, crop_dir=crop_dir,
                           zoom=zoom, blocks=blocks, rule_mask=rule_mask)
    evidence = built['evidence']
    _fill_model(evidence, engine)
    audit = out['audit'] + built['audit'] + [
        {'kind': 'group_failed', 'reason': e.get('reason'),
         'group_id': e.get('group_id'), 'region_id': e.get('region_id')}
        for e in group_errors]
    out['sides'] = built['side_counts']
    out['group_errors'] = group_errors

    # ---- 状态按**实际结果**判定（stage7-v2） --------------------------------
    # 候选（组）一个都没有：正常完成，列表为空，不是失败。
    # 有候选但一条精修记录都没有：所有候选都因处理异常失败 → failed + None。
    # 有精修记录但没有任何合格侧、且确实发生过异常（精修或复识）：同样 failed + None。
    # 除此之外只要有异常 → partial + 已成功证据；无异常 → completed + 全部证据。
    ok_sides = sum(1 for e in evidence if e['side_status'] == 'ok')
    side_errors = int(built['side_counts'].get('recognize_error', 0))
    had_exception = bool(group_errors) or side_errors > 0

    if not groups:
        return done('completed', [], audit=audit)
    if not records:
        out['error'] = 'all_groups_refine_failed'
        return done('failed', None, audit=audit)
    if had_exception and ok_sides == 0:
        out['error'] = 'no_usable_side_after_exception'
        return done('failed', None, audit=audit)
    if had_exception:
        return done('partial', evidence, audit=audit)
    return done('completed', evidence, audit=audit)
