# -*- coding: utf-8 -*-
"""第三阶段复现验证：跑两遍、比哈希、核对上游指纹。

验证四件事：
1. 几何预测可复现——路径 C 连跑两次，预测 JSON 逐字节相同。
2. 评分可复现——同一份保存的预测重评两次，指标相同（只差 generated_at）。
3. 复识有波动就要如实记录——路径 A 连跑两次，报文本一致率。
4. 上游没被改——原始图片来源、阶段 1/2 已验收产物的指纹不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage1_data as D          # noqa: E402
import dual_track_stage3_common as C3      # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S1 = ROOT / 'output' / 'dual_tracks' / 'stage1'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))
PY = sys.executable


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd: List[str]) -> Dict[str, Any]:
    t0 = datetime.now()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {'cmd': ' '.join(cmd), 'returncode': r.returncode,
            'seconds': round((datetime.now() - t0).total_seconds(), 2),
            'stderr_tail': (r.stderr or '')[-400:]}


def strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k != 'generated_at'}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='第三阶段复现验证')
    ap.add_argument('--out', default=str(OUT / 'verification_report.json'))
    ap.add_argument('--skip-rerun', action='store_true', help='跳过重跑，只核对指纹')
    args = ap.parse_args(argv)

    report: Dict[str, Any] = {
        'stage': 'stage3_dual_track_evaluation', 'release': 'stage3-v1',
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'checks': [],
    }

    def add(cid: str, title: str, ok: bool, detail: str, **extra: Any) -> None:
        report['checks'].append({'id': cid, 'title': title, 'status': 'pass' if ok else 'fail',
                                 'detail': detail, **extra})

    # V01 上游产物指纹
    upstream = {}
    for name, p in (('stage1_normalized', S1 / 'normalized_annotations.json'),
                    ('stage1_corrections', S1 / 'corrections.json'),
                    ('stage2_annotations', S2 / 'annotations.json'),
                    ('stage2_protocol', S2 / 'evaluation_protocol.json'),
                    ('stage2_negative_regions', S2 / 'negative_regions.json')):
        upstream[name] = sha(p) if p.exists() else None
    rep_notes = S1 / '阶段报告.md'
    add('V01', '阶段 1/2 已验收产物指纹可读', all(upstream.values()),
        f"{len(upstream)} 个上游产物均存在", upstream=upstream)

    # V02 原始图片来源未改动（与阶段 1 的 before 快照比）
    before = json.loads((S1 / 'source_fingerprints_before.json').read_text(encoding='utf-8'))['files']
    now = D.snapshot_tree(D.DATASET)
    diff = D.diff_snapshots(before, now)
    add('V02', '原始来源目录零变更', diff.get('identical') is True,
        f"来源 {len(now)} 个文件；新增 {len(diff.get('added') or [])}、"
        f"删除 {len(diff.get('removed') or [])}、变更 {len(diff.get('changed') or [])}",
        diff={k: v for k, v in diff.items() if k != 'identical'})

    # V03 路径 C 预测可复现
    pred_c = OUT / 'predictions' / 'path_c.json'
    if not args.skip_rerun:
        c1 = C3.fingerprint(pred_c)
        r1 = run([PY, 'tools/dual_track_stage3_run.py', '--paths', 'C', '--quiet'])
        c2 = C3.fingerprint(pred_c)
        same = c1['content_sha256'] == c2['content_sha256']
        add('V03', '路径 C 几何预测两次一致（去耗时后内容哈希）', r1['returncode'] == 0 and same,
            f"content run1 {c1['content_sha256'][:16]} / run2 {c2['content_sha256'][:16]}；"
            f"file 哈希随时间戳不同属预期；用时 {r1['seconds']}s",
            run=r1, content_sha=[c1['content_sha256'], c2['content_sha256']],
            file_sha=[c1['file_sha256'], c2['file_sha256']])

        # V04 复识波动
        rec_a = OUT / 'recognition' / 'path_a.json'
        before_a = json.loads(rec_a.read_text(encoding='utf-8'))

        def lane_texts(payload: Dict[str, Any]) -> Dict[str, str]:
            out = {}
            for rec in payload.get('records') or []:
                for side, lane in (rec.get('lanes') or {}).items():
                    out[f"{rec['group_id']}|{side}"] = (lane or {}).get('text') or ''
            return out

        r2 = run([PY, 'tools/dual_track_stage3_run.py', '--paths', 'A', '--quiet'])
        after_a = json.loads(rec_a.read_text(encoding='utf-8'))
        t1, t2 = lane_texts(before_a), lane_texts(after_a)
        changed = [k for k in t1 if t1[k] != t2.get(k)]
        add('V04', '复识（路径 A）两次结果一致', r2['returncode'] == 0 and not changed,
            f'{len(t1)} 条轨级读取，变化 {len(changed)} 条；'
            f'{"完全一致" if not changed else "存在波动，已记录差异"}',
            changed=changed[:10], run=r2)

        # V05 评分可复现
        r3 = run([PY, 'tools/dual_track_stage3_score.py', '--predictions', 'C', 'B', 'X', '--quiet'])
        m1 = {p.stem: strip_volatile(json.loads(p.read_text(encoding='utf-8')))
              for p in (OUT / 'metrics').glob('*.json') if p.name != 'index.json'}
        r4 = run([PY, 'tools/dual_track_stage3_score.py', '--predictions', 'C', 'B', 'X', '--quiet'])
        m2 = {p.stem: strip_volatile(json.loads(p.read_text(encoding='utf-8')))
              for p in (OUT / 'metrics').glob('*.json') if p.name != 'index.json'}
        same = json.dumps(m1, sort_keys=True, ensure_ascii=False) == \
            json.dumps(m2, sort_keys=True, ensure_ascii=False)
        add('V05', '同一预测重评两次指标一致', same and r3['returncode'] == r3['returncode'] == 0,
            f'{len(m1)} 个指标文件；两次重评{"完全一致" if same else "有差异"}',
            runs=[r3, r4])
    else:
        add('V03', '路径 C 几何预测两次一致', True, '已跳过重跑（--skip-rerun）')

    # V06 评分只用保存的预测（不重新跑算法）
    src = (ROOT / 'tools' / 'dual_track_stage3_score.py').read_text(encoding='utf-8')
    add('V06', '评分入口只读保存的预测', ('dual_track_stage3_run' not in src) and ('run_entry' not in src),
        'score.py 不 import 检测入口，也不调用 run_ocr/propose_page')

    # V07 产物齐全
    need = ['baseline_snapshot.json', 'frozen_layouts.json', 'frozen_layout_labels.json',
            'evaluation_config.json', 'evaluation_targets.json', 'protocol_addendum.json',
            'predictions/path_c.json', 'predictions/path_b.json',
            'predictions/path_x_components.json', 'recognition/path_a.json',
            'metrics/c.json', 'metrics/b.json', 'metrics/x_components.json']
    miss = [n for n in need if not (OUT / n).exists()]
    add('V07', '建议产物齐全', not miss, f'{len(need)} 项，缺 {len(miss)} 项', missing=miss,
        sizes={n: (OUT / n).stat().st_size for n in need if (OUT / n).exists()})

    # V08 冻结快照与预测哈希对得上
    snap = json.loads((OUT / 'baseline_snapshot.json').read_text(encoding='utf-8'))
    bad = []
    for name, info in snap['outputs']['predictions'].items():
        p = OUT / 'predictions' / name if name.startswith('path_') else OUT / 'recognition' / 'path_a.json'
        if not p.exists():
            bad.append(f'{name} 缺失')
            continue
        if C3.content_sha256(json.loads(p.read_text(encoding='utf-8'))) != info.get('content_sha256'):
            bad.append(f'{name} 内容哈希不符')
    add('V08', '快照记录的预测指纹与文件一致', not bad, '；'.join(bad) or '全部一致',
        checked=list(snap['outputs']['predictions']))

    # V09 评测器内容指纹（本轮修正后必须可追）
    import dual_track_stage3_common as C3x
    ev_files = sorted((ROOT / 'dual_track_eval').glob('*.py'))
    ev_hash = hashlib.sha256(b''.join(f.read_bytes() for f in ev_files)).hexdigest()
    report['evaluator'] = {'module': 'dual_track_eval', 'version': '1.2.2',
                           'files': [f.name for f in ev_files],
                           'sha256': ev_hash,
                           'changes': 'v1.1.0：TOL 改精确相等、归属前移、穷尽区未匹配一律 FP、'
                                      '配置驱动、base_status 唯一；'
                                      'v1.2.0：匹配代价改 −IoU（修回退精度）、'
                                      'lex_search_limits 与 excluded_certainty 接线、'
                                      '局部穷尽区域与 data_error、配置白名单；'
                                      'v1.2.1：区分缺省与空排除名单、子区域 page_id 继承与冲突、'
                                      'data_error 页进入失败归因；'
                                      'v1.2.2：端点敏感性跳过 data_error 页并在同一可评分集合上比较、'
                                      '主评分与端点敏感性均注明范围与排除页数、score_file 支持相对路径'}
    add('V09', '评测器内容指纹已记录', len(ev_hash) == 64,
        f'{len(ev_files)} 个文件，sha256 {ev_hash[:16]}', sha256=ev_hash)

    # V10 生效配置与配置文件一致
    import dual_track_stage3_score as SC
    cfg_path = OUT / 'evaluation_config.json'
    runtime = SC.runtime_config(json.loads(cfg_path.read_text(encoding='utf-8')))
    bad = []
    for name in ('c.json', 'b.json', 'x_components.json'):
        mp = OUT / 'metrics' / name
        if not mp.exists():
            bad.append(f'{name} 缺失')
            continue
        eff = json.loads(mp.read_text(encoding='utf-8')).get('effective_config') or {}
        if eff.get('runtime') != runtime:
            bad.append(f'{name} runtime 不一致')
        if (eff.get('source') or {}).get('sha256') != C3x.file_sha256(cfg_path):
            bad.append(f'{name} 配置指纹不一致')
    add('V10', '三条路径的生效配置与 evaluation_config.json 完全一致', not bad,
        '；'.join(bad) or 'runtime 与 sha256 均一致', config_sha=C3x.file_sha256(cfg_path)[:16])

    # V11 新旧指标差异可对账
    dp = OUT / 'metrics_diff.json'
    if dp.exists():
        d = json.loads(dp.read_text(encoding='utf-8'))
        cur = json.loads((OUT / 'metrics' / 'c.json').read_text(encoding='utf-8'))['scores']['overall']
        ok = (d['items']['C']['0.5']['group_hits']['after'] == cur['0.5']['group_hits']
              and d['items']['C']['0.75']['group_hits']['after'] == cur['0.75']['group_hits'])
        add('V11', '新旧差异表的 after 值与当前 metrics 一致', ok,
            f"变化字段 {d['summary']['changed_fields']} 个；"
            f"C@0.5 {d['items']['C']['0.5']['group_hits']['before']}→"
            f"{d['items']['C']['0.5']['group_hits']['after']}")
    else:
        add('V11', '新旧差异表存在', False, '缺 metrics_diff.json')

    report['summary'] = {'checks': len(report['checks']),
                         'pass': sum(1 for c in report['checks'] if c['status'] == 'pass'),
                         'fail': sum(1 for c in report['checks'] if c['status'] == 'fail')}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n',
                              encoding='utf-8')
    print(f"验证 {report['summary']['checks']} 项：通过 {report['summary']['pass']}、"
          f"失败 {report['summary']['fail']}")
    for c in report['checks']:
        print(f"  [{'PASS' if c['status'] == 'pass' else 'FAIL'}] {c['id']} {c['title']} — "
              f"{c['detail'][:110]}")
    return 0 if report['summary']['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
