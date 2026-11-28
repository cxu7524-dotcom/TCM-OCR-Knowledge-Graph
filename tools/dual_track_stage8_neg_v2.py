# -*- coding: utf-8 -*-
"""第八阶段补修③：按复核结论生成负例标注 v2（不改原冻结版）。

原版 output/dual_tracks/stage2/negative_regions.json 是冻结契约的一部分，
一律不动；本脚本只**另存**一版，把复核确认有误的 NEG-DF-SINGLE-01 从
single_track/confirmed 改成 retracted，并附上测量证据。

为什么改 status 而不是删记录：
  评测器只把 status=='confirmed' 的区域当确定负例（regions.py:136），
  改成 retracted 即退出负例池，但原始标签、原始 basis 和证据仍留在文件里，
  可追溯「改了什么、凭什么改」。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
TARGET = 'NEG-DF-SINGLE-01'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    src = S2 / 'negative_regions.json'
    doc = json.loads(src.read_text(encoding='utf-8'))
    review = json.loads((S8 / 'negative_review_df_single.json').read_text(encoding='utf-8'))

    seg = next((s for s in review['segments'] if s['verdict'] == 'dual'), None)
    if seg is None:
        print('复核没有给出双列区段，拒绝生成 v2')
        return 1

    out = json.loads(json.dumps(doc))
    for r in out['regions']:
        if r['negative_region_id'] != TARGET:
            continue
        r['status'] = 'retracted'
        r['review'] = {
            'reviewed_at': datetime.now(TZ).isoformat(timespec='seconds'),
            'reviewed_by': 'stage8-v2 补修③',
            'conclusion': '标签有误：区内是两列并排小字，不是单列',
            'measurements': {
                'window_profile': 'output/dual_tracks/stage8/negative_review_df_single.json',
                'probe_script': 'tools/dual_track_stage8_negprobe.py',
                'window_px': review['window'],
                'verdict_counts': review['verdict_counts'],
                'dual_segment': {'y0': seg['y0'], 'y1': seg['y1'],
                                 'seam_x': seg['seam_x'], 'seam_width': seg['seam_width'],
                                 'left_col': seg['left_col'], 'right_col': seg['right_col']},
                'stable_seam': review['stable_gap'],
                'method': ('用 ocr.vertical._binarize 同一套二值化；在 160px 长窗口内统计每列'
                           '被墨覆盖的行占比；缝要在一段连续行上持续接近零、两侧列宽与均覆盖率'
                           '都达标才算两列。40px 短带只作抽查，不作判定。'),
            },
            'zoom_images': review['zoom_images'],
            'detector_cross_check': {
                'note': '检测器在这块给出的组框与实测两列一致，说明是标签错、不是检测错',
                'group_bbox_545_594': 'stage8 组框 x 545–594，恰好覆盖左列 545–561 与右列 569–585',
                'valley_x': 564,
                'measured_seam': seg['seam_x'],
            },
            'retract_reason': ('原 basis 记「单一列，未分两轨」，但实测缝 x 561–569 在 '
                               'y330–815 连续成立，左右两列各约 16–17px；'
                               'stage7-v2/stage8-v1/stage8-v2 在该区的检测框因此被记为确定负例误检。'),
        }
    out['stage'] = 'stage8-v2-review'
    out['derived_from'] = {
        'path': 'output/dual_tracks/stage2/negative_regions.json',
        'sha256': sha(src),
        'rule': '只改 NEG-DF-SINGLE-01 的 status/basis 解释，其余区域与计数规则不动',
    }
    conf = sum(1 for r in out['regions'] if r['status'] == 'confirmed')
    out['counts'] = dict(out.get('counts') or {})
    out['counts']['regions'] = len(out['regions'])
    by_status: dict = {}
    for r in out['regions']:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    out['counts']['by_status'] = dict(sorted(by_status.items()))
    out['counts']['by_category'] = dict(sorted(
        {c: sum(1 for r in out['regions'] if r['category'] == c and r['status'] == 'confirmed')
         for c in {r['category'] for r in out['regions']}}.items()))
    out.setdefault('notes', []).append(
        'v2 口径：NEG-DF-SINGLE-01 复核为两列并排，状态改 retracted 退出确定负例池；'
        '原冻结版 output/dual_tracks/stage2/negative_regions.json 保持不变，'
        f'冻结口径仍按 {conf + 1} 个确定负例计分。')

    dst = S8 / 'negative_regions_v2.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'确定负例 {conf}（原 {conf + 1}）  状态分布 {out["counts"]["by_status"]}')
    print(f'→ {dst.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
