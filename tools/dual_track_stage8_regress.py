# -*- coding: utf-8 -*-
"""stage8 同进程「关 → 开 → 关」回归（p12 真实页）。

要证明的三件事
--------------
1. **首次关闭与末次关闭结果一致**：同一进程里先跑一次 off，中间跑 stage8，
   再跑一次 off，两次的 status / evidence / audit / groups / plan 必须逐字节相同。
   （v1 的跨度上限是进程级 monkeypatch，只要装了没还原，「关」也会带着 patch 跑——
   这条回归就是专门抓这类泄漏的。）
2. **默认路径不被污染**：stage8 之前和之后各跑一次 stage5，两次结果必须相同。
3. **重复检测确定**：同一页连续两次 stage8 检测，分组与框完全相同。

用法：
  python tools/dual_track_stage8_regress.py
输出：
  output/dual_tracks/stage8/acceptance/regress_switch.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'output' / 'dual_tracks' / 'stage8' / 'acceptance'
IMG = ROOT / '古籍评测集' / '医家类' / '丹方彚编' / 'page_12.png'
PID = '丹方彚编/page_12'


def _norm(ev):
    """证据 → 可比较的规范化结构（去掉时间戳等易变字段）。"""
    if ev is None:
        return None
    out = []
    for e in ev:
        out.append({k: e.get(k) for k in
                    ('evidence_id', 'group_id', 'side', 'bbox', 'bbox_source',
                     'side_status', 'status_reason', 'group_bbox', 'region_bbox')})
    return out


def _sig(r: Dict[str, Any]) -> Dict[str, Any]:
    return {'status': r.get('status'), 'attempted': r.get('attempted'),
            'groups': r.get('groups'), 'sides': r.get('sides'),
            'evidence': _norm(r.get('evidence')),
            'audit': r.get('audit'),
            'plan': {k: v for k, v in (r.get('plan') or {}).items()
                     if k not in ('slot_audit',)}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import app.config as C
    from importlib import reload
    reload(C)
    import ocr.dual_track as DT
    reload(DT)

    common = dict(width=0, height=0, engine='rapidocr')
    s5_cfg = {'detector': 'stage5'}
    s8_cfg = {'detector': 'stage8'}

    stage5_before = DT.run_dual_track(str(IMG), PID, mode='auto', cfg=s5_cfg, **common)
    off1 = DT.run_dual_track(str(IMG), PID, mode='off', cfg=s8_cfg, **common)
    on = DT.run_dual_track(str(IMG), PID, mode='auto', cfg=s8_cfg, **common)
    off2 = DT.run_dual_track(str(IMG), PID, mode='off', cfg=s8_cfg, **common)
    stage5_after = DT.run_dual_track(str(IMG), PID, mode='auto', cfg=s5_cfg, **common)

    # 重复检测确定性（不动入口，直接两次 detect）
    import cv2
    from dual_track_fix.detect import detect as detect_stage8
    from dual_track_fix.settings import detect_options, effective_options
    from ocr.vertical import imread_unicode
    image = imread_unicode(str(IMG))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    eff = effective_options({'detector': 'stage8'})
    d1 = detect_stage8(image, PID, 21.0, options=detect_options(eff), gray=gray)
    d2 = detect_stage8(image, PID, 21.0, options=detect_options(eff), gray=gray)
    det_same = ([g['group_bbox'] for g in d1['groups']]
                == [g['group_bbox'] for g in d2['groups']])

    s_off1, s_off2 = _sig(off1), _sig(off2)
    s5a, s5b = _sig(stage5_before), _sig(stage5_after)
    checks = [
        {'id': 'R1', 'name': '首次关闭 == 末次关闭',
         'pass': s_off1 == s_off2,
         'detail': {'status': [off1.get('status'), off2.get('status')],
                    'evidence': [off1.get('evidence'), off2.get('evidence')]}},
        {'id': 'R2', 'name': 'stage8 运行不污染默认 stage5 路径',
         'pass': s5a == s5b,
         'detail': {'stage5_groups': [stage5_before.get('groups'),
                                      stage5_after.get('groups')],
                    'stage5_status': [stage5_before.get('status'),
                                      stage5_after.get('status')]}},
        {'id': 'R3', 'name': '中间那次开启确实跑的是 stage8',
         'pass': (on.get('status') == 'completed' and (on.get('groups') or 0) > 0
                  and (on.get('plan') or {}).get('detector') == 'stage8'),
         'detail': {'status': on.get('status'), 'groups': on.get('groups'),
                    'detector': (on.get('plan') or {}).get('detector'),
                    'n_evidence': len(on.get('evidence') or [])}},
        {'id': 'R4', 'name': '同一页连续两次 stage8 检测结果相同',
         'pass': det_same,
         'detail': {'n_groups': [len(d1['groups']), len(d2['groups'])]}},
        {'id': 'R5', 'name': '关闭时不进入检测（attempted=false）',
         'pass': off1.get('attempted') is False and off2.get('attempted') is False,
         'detail': {'attempted': [off1.get('attempted'), off2.get('attempted')]}},
    ]
    doc = {'page_id': PID, 'checks': checks,
           'pass': all(c['pass'] for c in checks),
           'effective': eff,
           'stage8_summary': {'status': on.get('status'), 'groups': on.get('groups'),
                              'n_evidence': len(on.get('evidence') or [])},
           'off_signature': s_off1}
    (OUT / 'regress_switch.json').write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for c in checks:
        print(f"{c['id']} {'PASS' if c['pass'] else 'FAIL'}  {c['name']}")
    print('all', doc['pass'])
    return 0 if doc['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
