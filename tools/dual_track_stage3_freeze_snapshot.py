# -*- coding: utf-8 -*-
"""冻结旧算法基线快照（baseline_snapshot.json）。

冻结什么
--------
代码、配置、模型、环境、输入图像、原始布局、原始预测与统一预测、逐页状态与耗时。
不冻什么：任何凭据（配置里出现 key/token/secret/password 的字段一律记成 ***）。

为什么要冻结
------------
后续阶段要比较「新算法 vs 旧算法」。如果旧算法的源码/模型/参数/输入有任何一项
没固定，后面的差异就说不清是谁造成的。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import dual_track_stage3_common as C3      # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))

# 参与运行的源码分三层：决定预测的、决定指标的、只产出其他产物的。
# 上一版只列了一层半，漏了 common.py / score.py 的配置解析等，本轮补齐并分层。
CODE_LAYERS = {
    'prediction_runtime': [
        'ocr/vertical.py',                    # page_layout / split_columns / imread_unicode
        'ocr/small_columns.py',               # propose_page / propose_in_column
        'ocr/small_columns_components.py',    # propose_components_page（辅助基线 X）
        'ocr/small_columns_attach.py',        # 生产挂接（_read 的 native/double 两读）
        'ocr/recognize.py',                   # recognize_crop / 模型解析
        'ocr/pipeline.py',                    # run_ocr（X 的 OCR 前置）
        'ocr/normalize.py',
        'app/config.py',
        'tools/dual_track_stage3_adapters.py',  # 旧入口 → 统一预测格式
        'tools/dual_track_stage3_run.py',       # 三路径执行
    ],
    'evaluation_runtime': [
        'dual_track_eval/__init__.py',
        'dual_track_eval/geometry.py',
        'dual_track_eval/matching.py',
        'dual_track_eval/lane.py',
        'dual_track_eval/diagnostics.py',
        'dual_track_eval/regions.py',
        'dual_track_eval/scorer.py',
        'tools/dual_track_stage3_score.py',     # 配置解析 + 评分入口
        'tools/dual_track_stage3_common.py',    # 内容哈希（去耗时）
    ],
    'artifact_generation': [
        'tools/dual_track_stage3_freeze_layouts.py',
        'tools/dual_track_stage3_labels.py',
        'tools/dual_track_stage3_build.py',
        'tools/dual_track_stage3_verify.py',
        'tools/dual_track_stage3_evidence.py',
        'tools/dual_track_stage3_freeze_snapshot.py',
    ],
}
CODE_FILES = [f for files in CODE_LAYERS.values() for f in files]

SENSITIVE = ('key', 'token', 'secret', 'password', 'credential')

ENTRIES = {
    'primary': {
        'name': 'C / full_page_propose_page',
        'entry': 'ocr.vertical.page_layout → ocr.small_columns.propose_page',
        'role': 'primary_baseline',
        'why': '纯几何、无 OCR 依赖、无外部服务，给定同一张图必定给出同一结果；'
               '可直接复现、可直接对比，后续新算法就与它比',
        'extra_ocr': '无。page_layout 只做二值化/切列，propose_page 只在列内找中缝与轨道',
        'config_gate': '不读 CONFIG.ocr.small_columns_enabled（此开关只控制生产挂接；'
                       '本入口是函数级调用）',
    },
    'auxiliary': [
        {
            'name': 'X / full_page_components',
            'entry': 'run_ocr → page_layout → small_columns_components.propose_components_page',
            'role': 'auxiliary_diagnostic',
            'why': '生产实际挂接的就是它（attach_small_columns 内部调用），但它要求 OCR 块支持'
                   '与跨字端闭合，链路更长、依赖更多，作为诊断而不是主基线',
            'extra_ocr': '有。整页 run_ocr（含低置信过滤、方向判别、繁简归一化）',
        },
        {
            'name': 'B / ctx_propose_page',
            'entry': '参考组上下文裁块 → page_layout → propose_page',
            'role': 'auxiliary_diagnostic',
            'why': '观察「给定局部范围」时的检测与边界问题；裁块依赖参考位置，不能当主基线',
            'extra_ocr': '无',
        },
        {
            'name': 'A / recog_ref_lane_crop',
            'entry': '参考轨框裁图 → recognize_crop(rot90, scale 1/2)',
            'role': 'auxiliary_text_only',
            'why': '只评复识，不产生框；用于观察参考裁框条件下的文本表现',
            'extra_ocr': '有。单块识别模型直推（跳过检测）',
        },
    ],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:                                     # noqa: BLE001
            return ''
    return {'head': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty_tracked_files': len([x for x in run('status', '--short').splitlines()
                                        if not x.startswith('??')]),
            'untracked_files': len([x for x in run('status', '--short').splitlines()
                                    if x.startswith('??')])}


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ('***' if any(s in k.lower() for s in SENSITIVE) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def ocr_config() -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    from app.config import CONFIG
    oc = CONFIG.ocr
    d = asdict(oc) if is_dataclass(oc) else dict(oc)
    return redact(d)


def model_info() -> Dict[str, Any]:
    from ocr.recognize import rec_model_name, resolve_rec_model
    path, name, source = resolve_rec_model()
    p = Path(path)
    return {'name': name, 'path': str(p.relative_to(ROOT)).replace('\\', '/') if p.is_relative_to(ROOT) else str(p),
            'sha256': sha256_of(p) if p.exists() else None,
            'source': source, 'runtime': 'onnxruntime CPU（rapidocr-onnxruntime 包内检测模型）'}


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {'python': sys.version.split()[0], 'platform': platform.platform(),
                            'machine': platform.machine()}
    for mod in ('numpy', 'cv2', 'onnxruntime', 'rapidocr_onnxruntime'):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, '__version__', 'unknown')
        except Exception:                                     # noqa: BLE001
            info[mod] = None
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='冻结旧算法基线快照')
    ap.add_argument('--out', default=str(OUT / 'baseline_snapshot.json'))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out_dir = out.parent

    code = {}
    for rel in CODE_FILES:
        p = ROOT / rel
        code[rel] = sha256_of(p) if p.exists() else None
    missing = [k for k, v in code.items() if v is None]

    layouts_path = out_dir / 'frozen_layouts.json'
    layouts = json.loads(layouts_path.read_text(encoding='utf-8'))
    pred_files = {}
    for name in ('path_c.json', 'path_b.json', 'path_x_components.json'):
        p = out_dir / 'predictions' / name
        if p.exists():
            fp = C3.fingerprint(p)
            pred_files[name] = {**fp,
                                'sha256': fp['file_sha256'],
                                'n_predictions': len(json.loads(p.read_text(encoding='utf-8'))
                                                     .get('predictions') or [])}
    recog = out_dir / 'recognition' / 'path_a.json'
    if recog.exists():
        fp = C3.fingerprint(recog)
        pred_files['recognition/path_a.json'] = {
            **fp, 'sha256': fp['file_sha256'],
            'n_records': len(json.loads(recog.read_text(encoding='utf-8')).get('records') or [])}

    inputs = [{'page_id': r['page_id'], 'image_path': r['image_path'],
               'sha256': r['sha256'], 'image_size': r.get('image_size')}
              for r in layouts['pages']]

    per_page: Dict[str, Dict[str, Any]] = {}
    for name, key in (('path_c.json', 'C'), ('path_b.json', 'B'),
                      ('path_x_components.json', 'X')):
        p = out_dir / 'predictions' / name
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding='utf-8'))
        for run in payload.get('page_runs') or []:
            per_page.setdefault(run['page_id'], {})[key] = {
                'status': run.get('status'), 'n_predictions': run.get('n_predictions', 0),
                'n_columns_detected': run.get('n_columns_detected'),
                'error': run.get('error'), 'timing_s': run.get('timing_s')}

    digest_src = json.dumps({'code': code, 'config': ocr_config(),
                             'model': model_info(), 'inputs': inputs},
                            ensure_ascii=False, sort_keys=True).encode('utf-8')
    snapshot_id = 'baseline-' + hashlib.sha256(digest_src).hexdigest()[:12]

    history_dir = out_dir / 'history'
    supersedes = None
    if out.exists():
        prev = json.loads(out.read_text(encoding='utf-8'))
        if prev.get('snapshot_id') and prev['snapshot_id'] != snapshot_id:
            supersedes = prev['snapshot_id']
            history_dir.mkdir(parents=True, exist_ok=True)
            (history_dir / f"baseline_snapshot.{prev['snapshot_id']}.json").write_text(
                json.dumps(prev, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    payload = {
        'stage': 'stage3_dual_track_evaluation',
        'release': 'stage3-v1',
        'snapshot_id': snapshot_id,
        'supersedes': supersedes,
        'history_policy': '每次源码/配置/模型/输入变动都新开 snapshot_id；旧版本归档到 history/，不覆盖',
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'generator': 'tools/dual_track_stage3_freeze_snapshot.py',
        'entries': ENTRIES,
        'selection_policy': {
            'primary_declared': 'C / full_page_propose_page',
            'basis': '先按「无外部依赖、可完全复现」选定主基线，再看评分；'
                     '首轮评分后未改选，也未改参数',
            'auxiliary_use': 'X 用于诊断「加 OCR 支持与跨字端闭合后是否更好」；'
                             'B 用于诊断「给定局部范围」；A 只评文本',
        },
        'code': {'files': code, 'missing': missing, 'git': git_info(),
                 'layers': {k: {f: code.get(f) for f in v} for k, v in CODE_LAYERS.items()},
                 'note': '分三层登记：prediction_runtime 决定预测、evaluation_runtime 决定指标、'
                         'artifact_generation 只产出周边产物。改任何一层都要新开快照版本。'},
        'config': {'ocr': ocr_config(),
                   'runner_params': {
                       'group_iou_thresholds': [0.5, 0.75],
                       'lane_child_thresholds': {'right_iou': 0.5, 'left_iou': 0.5},
                       'path_b_crop': {'column_width_multiplier': 4.4, 'up_ratio': 0.9,
                                       'down_ratio': 0.5, 'min_height': 380, 'min_width': 60},
                       'path_a_recog': {'rotation': 'rot90_ccw', 'scales': [1, 2], 'min_conf': 0.15},
                   },
                   'not_applicable': [
                       '新算法（分栏修复/状态机/框精修）的中缝阈值、尺度参数、补笔参数——'
                       '旧算法不存在这些参数，记 not_applicable',
                   ]},
        'model': model_info(),
        'environment': env_info(),
        'inputs': {'n_pages': len(inputs), 'pages': inputs,
                   'layouts_ref': {'file': 'output/dual_tracks/stage3/frozen_layouts.json',
                                   'sha256': sha256_of(layouts_path),
                                   'layout_snapshot_id': layouts.get('layout_snapshot_id')}},
        'outputs': {'predictions': pred_files,
                    'hash_rule': 'file_sha256 含耗时（每次运行不同）；'
                                 'content_sha256 去掉 timing_s/generated_at 后计算，用于判断同一算法两次运行是否一致',
                    'raw_layouts': 'output/dual_tracks/stage3/frozen_layouts.json',
                    'unified_prediction_format': {
                        'page_id': 'str', 'prediction_id': 'str',
                        'group_bbox': 'union(right_bbox, left_bbox)，不含主字框',
                        'right_bbox': '[x1,y1,x2,y2) 或 null', 'left_bbox': '[x1,y1,x2,y2) 或 null',
                        'parent_bbox': '旧布局列框（检测依据）',
                        'source': 'C / B / X', 'run_status': 'ok | degraded | failed',
                        'error': '空或问题列表', 'raw_ref': '指向原始输出的键'}},
        'per_page_runs': per_page,
        'notes': [
            '配置只保留运行所需项；凭据类字段一律 *** 或不出现',
            '旧算法没有「中缝阈值 / 尺度对象 / 补笔」这套参数，相关项标 not_applicable',
            'CONFIG.ocr.small_columns_enabled=false：生产默认关闭小字挂接，本基线是函数级调用',
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not args.quiet:
        print(f'snapshot_id = {snapshot_id}')
        print(f'代码文件 {len(code)} 个（缺失 {len(missing)}）  git HEAD {payload["code"]["git"]["head"][:12]}')
        print(f'模型 {payload["model"]["name"]}  sha256 {str(payload["model"]["sha256"])[:16]}')
        print(f'输入 {len(inputs)} 页；预测文件 {list(pred_files)}')
        print(f'→ {out.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
