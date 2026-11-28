"""逐轨复核 A4 的原图、栏线掩膜和去栏线裁图（含像素一致性与掩膜触字检查）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.dual_track import _crossing_ink_ids
from ocr.vertical import imread_unicode
from tools.dual_track_stage8_accept import IMG, edge_ink, ink_mask, online_evidence


def _read_decoded(path, flags):
    if not path:
        return None, 'missing_attachment'
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None, 'missing_attachment'
    try:
        encoded = np.fromfile(p, dtype=np.uint8)
    except OSError:
        return None, 'missing_attachment'
    if not encoded.size:
        return None, 'missing_attachment'
    decoded = cv2.imdecode(encoded, flags)
    if decoded is None:
        return None, 'decode_failed'
    return decoded, 'ok'


def _verify_crop(image, evidence):
    rule = evidence.get('rule_mask_review') or {}
    bounds = rule.get('crop_bbox')
    if not bounds:
        return None, None
    raw, st_raw = _read_decoded(evidence.get('image_path'), cv2.IMREAD_COLOR)
    cleared, st_clr = _read_decoded(rule.get('masked_crop_path'), cv2.IMREAD_COLOR)
    marked, st_msk = _read_decoded(rule.get('mask_path'), cv2.IMREAD_GRAYSCALE)
    for st in (st_raw, st_clr, st_msk):
        if st != 'ok':
            return st, None
    x1, y1, x2, y2 = bounds
    original = image[y1:y2, x1:x2]
    if marked.shape != original.shape[:2] or cleared.shape != original.shape or raw.shape != original.shape:
        return 'crop_dimensions_mismatch', None
    pixels = marked > 0
    if (int(pixels.sum()) != rule['marked_pixels']
            or not np.array_equal(raw, original)
            or not np.array_equal(cleared[~pixels], original[~pixels])):
        return 'source_pixels_mismatch', None
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = pixels
    return 'verified', mask


def _mask_touches_glyph(ink: np.ndarray, ruled: np.ndarray, box, y_edge: int, H: float = 21.0) -> bool:
    """检查端点附近的栏线掩膜是否与【框内】真实字形粘连/相邻（严格排除仅在框外的未掩膜版框残段）。"""
    if ruled is None:
        return False
    x1, y1, x2, y2 = box
    ya, yb = max(0, y_edge - 3), min(ink.shape[0], y_edge + 3)
    if yb <= ya or x2 <= x1:
        return False
    mask_band = ruled[ya:yb, x1:x2]
    if not mask_band.any():
        return False
    # 只检查严格位于框内端点邻域 [in_ya, in_yb) 内的去栏线墨迹是否与掩膜相邻
    in_ya = y1 if y_edge == y1 else max(y1, y2 - 3)
    in_yb = min(y2, y1 + 3) if y_edge == y1 else y2
    if in_yb <= in_ya:
        return False
    dilated_full = np.zeros_like(ruled, dtype=bool)
    dilated_full[ya:yb, x1:x2] = cv2.dilate(
        mask_band.astype(np.uint8), np.ones((3, 3), np.uint8)
    ) > 0
    cleaned = ink & ~ruled
    _, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    in_mask_hit = dilated_full[in_ya:in_yb, x1:x2]
    touching_ids = set(np.unique(c_labels[in_ya:in_yb, x1:x2][in_mask_hit])) - {0}
    for cid in touching_ids:
        cx, cy, cw, ch, area = map(int, c_stats[cid])
        if cw >= max(4, round(0.28 * H)) or area >= max(10, round(0.05 * H * H)):
            return True
    return False


def review(pred_path: Path, a4_path: Path, output: Path) -> dict:
    source = imread_unicode(str(IMG))
    ink = ink_mask(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    H = 21.0
    evidence = {e['evidence_id']: e for e in online_evidence(pred_path)}
    raw = json.loads(a4_path.read_text(encoding='utf-8'))
    rows = []
    for item in raw['checked_items']:
        eid, box = item['evidence_id'], item['bbox']
        ev = evidence[eid]
        if ev['bbox'] != box:
            raise ValueError(f'A4 与预测坐标不一致 {eid}')
        unresolved = set(ev['basis']['boundary_review'].get('unresolved') or ())
        flagged = {'top': 'cut_head', 'bottom': 'cut_tail'}
        edges = {edge for edge, why in flagged.items()
                 if item[edge] >= .5 or why in unresolved}
        if not edges:
            continue
        rule_evidence = ev.get('review_evidence') or {}
        file_verdict, ruled = _verify_crop(source, rule_evidence)
        x1, y1, x2, y2 = box
        cleaned = (ink & ~ruled) if ruled is not None else ink
        _, ids = cv2.connectedComponents(cleaned.astype(np.uint8), connectivity=8)
        fraction = edge_ink(cleaned, box)
        endpoints = {}
        for edge, y in (('top', y1), ('bottom', y2)):
            if edge not in edges:
                continue
            cross = _crossing_ink_ids(ids, x1, x2, y) if 0 < y < len(ids) else set()
            marked = int(ruled[max(0, y - 2):y + 2, x1:x2].sum()) if ruled is not None else 0
            touches_glyph = _mask_touches_glyph(ink, ruled, box, y, H)
            if file_verdict not in (None, 'verified'):
                verdict = f'attachment_invalid:{file_verdict}'
            elif touches_glyph:
                verdict = 'glyph_adheres_to_rule_pending'
            elif cross or fraction[edge] >= .5:
                verdict = 'residual_ink_at_boundary'
            elif marked:
                verdict = 'rule_explains_trigger'
            else:
                verdict = 'needs_review'
            endpoints[edge] = {
                'raw_overlap': item[edge],
                'clean_overlap': fraction[edge],
                'remaining_crossing_components': len(cross),
                'marked_pixels_at_boundary': marked,
                'mask_touches_glyph': touches_glyph,
                'verdict': verdict,
            }
        rows.append({
            'evidence_id': eid,
            'bbox': box,
            'files': file_verdict,
            'original_a4_trigger': max(item['top'], item['bottom']) >= .5,
            'endpoints': endpoints,
            'evidence': rule_evidence,
        })
    n_explained = sum(
        row['original_a4_trigger']
        and all(r['verdict'] == 'rule_explains_trigger' for r in row['endpoints'].values())
        for row in rows
    )
    n_with_residue = sum(
        any(r['verdict'] != 'rule_explains_trigger' for r in row['endpoints'].values())
        for row in rows
    )
    n_touches_glyph = sum(
        any(r.get('mask_touches_glyph') for r in row['endpoints'].values())
        for row in rows
    )
    n_file_failures = sum(
        row['files'] not in (None, 'verified') for row in rows
    )
    report = {
        'prediction_source': str(pred_path),
        'prediction_sha256': hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        'n_checked_original': raw['n_checked'],
        'n_raw_triggers': raw['n_cut'],
        'n_lanes_for_review': len(rows),
        'n_trigger_lanes_explained_by_rule': n_explained,
        'n_lanes_requiring_glyph_review': n_with_residue,
        'n_lanes_mask_touches_glyph': n_touches_glyph,
        'n_attachment_failures': n_file_failures,
        'lanes': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--a4', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = review(args.pred, args.a4, args.out)
    print(f"原图触发 {report['n_raw_triggers']} 轨，"
          f"掩膜可解释 {report['n_trigger_lanes_explained_by_rule']} 轨，"
          f"触字待核 {report['n_lanes_mask_touches_glyph']} 轨，"
          f"需核实字形 {report['n_lanes_requiring_glyph_review']} 轨")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
