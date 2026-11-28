# -*- coding: utf-8 -*-
"""stage8 三方快照：C / stage7-v2 / 修正版（stage8-v1）各自独立存档。

每个快照目录含：
  manifest.json      代码/配置/模型/图像指纹 + 预测内容 sha256 + 生成方式
  predictions.json   该版本在 19 页上产出的全部预测（冻结格式）
  metrics.json       用 stage3-v1 冻结协议算出的指标（若已评）

为什么要指纹：三份结果要能各自复跑；只留数字而不留「当时跑的是哪份代码/配置/权重」，
事后无法判断差异来自改动还是来自环境。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
S3 = ROOT / 'output' / 'dual_tracks' / 'stage3'
S6 = ROOT / 'output' / 'dual_tracks' / 'stage6'
S8 = ROOT / 'output' / 'dual_tracks' / 'stage8'
TZ = timezone(timedelta(hours=8))

CODE_FILES = [
    'ocr/small_columns.py', 'ocr/vertical.py', 'ocr/dual_track.py', 'ocr/pipeline.py',
    'app/config.py',
    'dual_track_track/config.py', 'dual_track_track/api.py',
    'dual_track_refine/api.py', 'dual_track_refine/config.py',
    'dual_track_fix/slots.py', 'dual_track_fix/detect.py', 'dual_track_fix/relax.py',
    'dual_track_fix/config.py',
    'dual_track_eval/scorer.py', 'dual_track_eval/lane.py', 'dual_track_eval/matching.py',
]
CONFIG_FILES = [
    'output/dual_tracks/stage3/evaluation_config.json',
    'output/dual_tracks/stage3/evaluation_targets.json',
    'output/dual_tracks/stage4/search_regions.json',
    'output/dual_tracks/stage4/rule_masks.json',
    'output/dual_tracks/stage2/annotations.json',
    'output/dual_tracks/stage2/negative_regions.json',
    'output/dual_tracks/stage2/dataset_splits.json',
]


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon_sha(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def models() -> List[Dict[str, Any]]:
    out = []
    for d in (ROOT / '模型权重').rglob('*'):
        if d.is_file() and d.stat().st_size > 1024:
            out.append({'path': str(d.relative_to(ROOT)).replace('\\', '/'),
                        'bytes': d.stat().st_size, 'sha256': sha(d)})
    return sorted(out, key=lambda x: x['path'])


def images(ann) -> List[Dict[str, Any]]:
    seen = {}
    for g in ann['groups']:
        seen[g['image_path']] = g['image_sha256']
    return [{'image_path': k, 'sha256': v} for k, v in sorted(seen.items())]


def build(name: str, note: str, predictions: List[Dict[str, Any]],
          entry: str, extra: Dict[str, Any], ann) -> Dict[str, Any]:
    d = S8 / 'snapshots' / name
    d.mkdir(parents=True, exist_ok=True)
    doc = {'path': name, 'entry': entry, 'predictions': predictions, 'page_runs': []}
    (d / 'predictions.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n',
                                        encoding='utf-8')
    man = {
        'snapshot': name, 'note': note, 'entry': entry,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'n_predictions': len(predictions),
        'predictions_content_sha256': canon_sha(predictions),
        'predictions_file_sha256': sha(d / 'predictions.json'),
        'code': {f: sha(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()},
        'config': {f: sha(ROOT / f) for f in CONFIG_FILES if (ROOT / f).exists()},
        'models': models(),
        'images': images(ann),
        **extra,
    }
    (d / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1) + '\n',
                                     encoding='utf-8')
    return man


def main() -> int:
    from dual_track_refine import refined_predictions

    ann = load(S2 / 'annotations.json')
    idx = []

    # ---- A：旧算法 C（阶段三冻结基线快照，原样归档，不重跑）
    c = load(S3 / 'predictions' / 'path_c.json')
    idx.append(build('C', '旧算法 C：page_layout + ocr.small_columns.propose_page（阶段三冻结）',
                     c['predictions'], c.get('entry', 'layout_propose_page'),
                     {'source_file': 'output/dual_tracks/stage3/predictions/path_c.json',
                      'snapshot_id': 'baseline-848bb88d3e30',
                      'reproduce': 'python tools/dual_track_stage3_run.py（原始产出脚本）'}, ann))

    # ---- B：当前 stage7-v2 链路（阶段六精修后的最终框，阶段七入口用的就是这批）
    s6 = load(S6 / 'groups.json')
    preds_b = refined_predictions(s6, use_before=False)
    idx.append(build('stage7v2', '当前链路：阶段五逐窗状态机 + 阶段六精修（stage7-v2 入口同源）',
                     preds_b, 'stage5_detect+stage6_refine',
                     {'source_file': 'output/dual_tracks/stage6/groups.json',
                      'stage6_release': s6.get('release'),
                      'reproduce': ('python -c "import json;from dual_track_refine import '
                                    'refined_predictions as r;'
                                    'print(len(r(json.load(open(\'output/dual_tracks/stage6/'
                                    'groups.json\',encoding=\'utf-8\')))))"')}, ann))

    fix = load(S8 / 'predictions' / 'path_stage8v1.json')
    idx.append(build('fix', '修正版 stage8-v1：列槽周期切分 + 子框裁到槽内 + 按墨迹收紧 x',
                     fix['predictions'], fix.get('entry', ''),
                     {'pipeline_switch': "OCR_DUAL_TRACK_DETECTOR=stage8",
                      'source_file': 'output/dual_tracks/stage8/predictions/path_stage8v1.json',
                      'isolated_factors': ['slots', 'clip', 'relax_span', 'tighten_lanes'],
                      'reproduce': ('python tools/dual_track_stage8_run.py --tag stage8v1 '
                                    '--source FIX --clip --relax-span --tighten-lanes --no-dedupe')},
                     ann))

    (S8 / 'snapshot_index.json').write_text(
        json.dumps({'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
                    'items': [{'snapshot': m['snapshot'], 'n_predictions': m['n_predictions'],
                               'predictions_content_sha256': m['predictions_content_sha256'],
                               'entry': m['entry']} for m in idx]},
                   ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for m in idx:
        print(f"{m['snapshot']:10s} n={m['n_predictions']:3d} "
              f"content={m['predictions_content_sha256'][:16]} entry={m['entry']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
