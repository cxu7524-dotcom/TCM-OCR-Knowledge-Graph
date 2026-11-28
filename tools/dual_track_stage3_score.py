# -*- coding: utf-8 -*-
"""第三阶段评分 CLI：从**已保存的预测** + **冻结配置** 重算指标。

用法：
  python tools/dual_track_stage3_score.py --predictions C B X
  python tools/dual_track_stage3_score.py --config output/dual_tracks/stage3/evaluation_config.json
  python tools/dual_track_stage3_score.py --override group_matching.thresholds='[0.5,0.6]'

为什么必须有 --config：
  上一版把阈值写死在脚本里，改了 evaluation_config.json 也不影响结果——那样
  "配置驱动"只是说法。现在所有阈值（组匹配、子框、区域、诊断、端点位移）
  都从配置文件读，并把实际生效值连同文件指纹写进 metrics。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dual_track_eval import RegionTable, endpoint_sensitivity, score_predictions  # noqa: E402
from dual_track_eval.regions import PROVISIONAL                                  # noqa: E402

OUT = ROOT / 'output' / 'dual_tracks' / 'stage3'
S2 = ROOT / 'output' / 'dual_tracks' / 'stage2'
TZ = timezone(timedelta(hours=8))

FILES = {'B': 'path_b.json', 'C': 'path_c.json', 'X': 'path_x_components.json'}
DEFAULT_CONFIG = OUT / 'evaluation_config.json'


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_sha256(path: Path) -> str:
    import dual_track_stage3_common as C3
    return C3.content_sha256(load(path))


# ---------------------------------------------------------------- 配置解析

def _set_path(doc: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    cur = doc
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def runtime_config(doc: Dict[str, Any]) -> Dict[str, Any]:
    """把 evaluation_config 的嵌套文档翻成 scorer 认识的扁平参数。

    这是唯一的映射点：任何参数要生效必须经过这里，报告引用的 effective runtime
    也就是它的返回值。
    """
    gm = doc.get('group_matching') or {}
    ls = doc.get('lane_scoring') or {}
    rg = doc.get('regions') or {}
    dg = doc.get('diagnostics') or {}
    ep = doc.get('endpoint_sensitivity') or {}
    num = gm.get('float_policy') or {}
    frag = dg.get('fragment') or {}
    merge = dg.get('merge') or {}
    return {
        'group_iou_thresholds': list(gm.get('thresholds') or [0.5, 0.75]),
        'lane_child_thresholds': dict(ls.get('child_thresholds')
                                      or {'right_iou': 0.5, 'left_iou': 0.5}),
        # 缺省（字段不存在）用默认名单；显式 [] 表示「不排除任何 certainty」。
        # 不能写成 `ls.get(...) or DEFAULT`——空列表是 falsy，会被悄悄换回默认。
        'lane_eligible_exclude_certainty': (
            list(ls['excluded_certainty']) if 'excluded_certainty' in ls
            else ['ambiguous_lane_split', 'ambiguous_lane_extent']),
        'endpoint_pending_inclusive_shift_px': int(ep.get('shift_px') or 1),
        'region_thresholds': dict(rg.get('thresholds') or {}),
        'diagnostics': {'fragment_union_coverage': frag.get('union_coverage_of_gt_min', 0.5),
                        'merge_containment': merge.get('containment_of_gt_min', 0.6)},
        'numeric': {'tie_break_tolerance': num.get('tie_break_tolerance', 0.0),
                    'rounding_in_optimization': num.get('rounding_in_optimization', False)},
        'matching': {'solver': (gm.get('solver') or {}).get('primary'),
                     'tie_break': (gm.get('solver') or {}).get('tie_break'),
                     'lex_search_limits': (gm.get('solver') or {}).get('lex_search_limits')},
    }


SUPPORTED_NUMERIC = {'tie_break_tolerance': 0.0, 'rounding_in_optimization': False}


class ConfigError(SystemExit):
    """配置里出现不支持的取值时直接拒绝，而不是悄悄按默认跑。"""


def validate_runtime(runtime: Dict[str, Any]) -> List[str]:
    """只支持固定值的数值策略：值不对就报错。

    这两项是「实现约定」而非可调参数：
      - tie_break_tolerance 必须为 0（并列要用浮点精确相等，任何正容差都会
        把不同总分当并列，从而选中较低分解）；
      - rounding_in_optimization 必须为 False（优化中用舍入会人为制造并列）。
    所以只接受这两个值，其余一律拒绝。
    """
    problems = []
    num = runtime.get('numeric') or {}
    for k, must in SUPPORTED_NUMERIC.items():
        got = num.get(k, must)
        if got != must:
            problems.append(f'numeric.{k} 只支持 {must!r}，收到 {got!r}')
    thr = runtime.get('group_iou_thresholds') or []
    if not thr or not all(isinstance(x, (int, float)) and 0 < x <= 1 for x in thr):
        problems.append(f'group_matching.thresholds 必须是 (0,1] 内的数，收到 {thr!r}')
    lane = runtime.get('lane_child_thresholds') or {}
    for k in ('right_iou', 'left_iou'):
        v = lane.get(k)
        if not isinstance(v, (int, float)) or not 0 < v <= 1:
            problems.append(f'lane_scoring.child_thresholds.{k} 必须是 (0,1]，收到 {v!r}')
    lim = (runtime.get('matching') or {}).get('lex_search_limits') or {}
    for k in ('max_nodes', 'max_k'):
        if k in lim and (not isinstance(lim[k], int) or lim[k] < 1):
            problems.append(f'lex_search_limits.{k} 必须是正整数，收到 {lim[k]!r}')
    if 'time_budget_s' in lim and (not isinstance(lim['time_budget_s'], (int, float))
                                   or lim['time_budget_s'] <= 0):
        problems.append(f"lex_search_limits.time_budget_s 必须 > 0，收到 {lim['time_budget_s']!r}")
    return problems


def region_thresholds(doc_or_runtime: Dict[str, Any], from_runtime: bool = True) -> Dict[str, float]:
    """只取数值项给 RegionTable；status/basis 等说明字段不进去。"""
    t = (doc_or_runtime.get('region_thresholds') if from_runtime
         else (doc_or_runtime.get('regions') or {}).get('thresholds')) or {}
    out = {}
    for k in ('boundary_conflict_max', 'suspected_fp_min', 'confirmed_fp_min'):
        if isinstance(t.get(k), (int, float)):
            out[k] = float(t[k])
    return out


def resolve_config(path: Path, overrides: Optional[List[str]]) -> Dict[str, Any]:
    base = load(path)
    doc = copy.deepcopy(base)
    applied = []
    for spec in overrides or []:
        if '=' not in spec:
            raise SystemExit(f'--override 需要 key=value 形式：{spec!r}')
        key, raw = spec.split('=', 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_path(doc, key.strip(), value)
        note = ''
        if key.strip() == 'lane_scoring.excluded_certainty' and value == []:
            note = '空名单：所有已匹配组都具轨级评测资格（不是「没配」）'
        applied.append({'key': key.strip(), 'value': value, **({'note': note} if note else {})})
    version = base.get('release', 'unknown')
    if applied:
        version = f'{version}+override{len(applied)}'
    return {'doc': doc, 'overrides': applied, 'version': version,
            'source': {'file': str(path.relative_to(ROOT)).replace('\\', '/'),
                       'sha256': sha256_of(path),
                       'base_release': base.get('release')}}


# ---------------------------------------------------------------- 评分

def build_tables(ann: Dict[str, Any], neg: Dict[str, Any], extra_pages: List[str],
                 thresholds: Optional[Dict[str, float]] = None) -> Dict[str, RegionTable]:
    """coverage 口径：有正例标注的页用其 annotation_coverage；
    只划了负例、没有正例标注的页标 negative_only。"""
    coverage = {p['page_id']: p.get('annotation_coverage', 'partial_positions')
                for p in ann['pages']}
    # 局部穷尽区域：页记录里若给了 evaluation_regions（带原图 bbox），原样传给区域表
    regions_by_page = {p['page_id']: p.get('evaluation_regions') for p in ann['pages']}
    neg_pages = {r['page_id'] for r in (neg.get('regions') or [])}
    ambiguous = neg.get('ambiguous_regions') or []
    ignores = neg.get('ignore_regions') or []
    tables: Dict[str, RegionTable] = {}
    for pid in set(coverage) | set(extra_pages) | neg_pages:
        cov = coverage.get(pid) or ('negative_only' if pid in neg_pages else 'partial_positions')
        tables[pid] = RegionTable(pid, neg.get('regions') or [], ambiguous, ignores,
                                  coverage=cov, thresholds=thresholds,
                                  evaluation_regions=regions_by_page.get(pid))
    return tables


def page_split_map(splits: Dict[str, Any]) -> Dict[str, str]:
    return {a['page_id']: a['split'] for a in splits['assignments']}


def page_sizes_map(ann: Dict[str, Any]) -> Dict[str, List[int]]:
    return {p['page_id']: p['image_size'] for p in ann['pages'] if p.get('image_size')}


def score_file(path: Path, ann: Dict[str, Any], neg: Dict[str, Any],
               splits: Dict[str, str], runtime: Dict[str, Any],
               config_meta: Dict[str, Any]) -> Dict[str, Any]:
    # 相对路径也要能用：先解析成绝对路径，否则下面 relative_to(ROOT) 会抛 ValueError
    path = Path(path).resolve()
    payload = load(path)
    preds = payload.get('predictions') or []
    pages = sorted({p['page_id'] for p in preds} |
                   {g['page_id'] for g in ann['groups']} |
                   {r['page_id'] for r in (payload.get('page_runs') or [])})
    tables = build_tables(ann, neg, pages, region_thresholds(runtime))
    sizes = page_sizes_map(ann)
    res = score_predictions(preds, ann['groups'], tables, runtime, pages=pages,
                            page_split=splits, page_sizes=sizes)
    sens = endpoint_sensitivity(preds, ann['groups'], tables, runtime, pages=pages,
                                shift_px=runtime['endpoint_pending_inclusive_shift_px'],
                                page_sizes=sizes)
    return {
        'path': payload.get('path'), 'entry': payload.get('entry'),
        'predictions_file': str(path.relative_to(ROOT)).replace('\\', '/'),
        'predictions_file_sha256': sha256_of(path),
        'predictions_content_sha256': content_sha256(path),
        'n_predictions': len(preds),
        'effective_config': config_meta,
        'page_runs': payload.get('page_runs') or [],
        'scores': res['summary'],
        'data_errors': res.get('data_errors') or [],
        'invalid': res['invalid'],
        'metric_limits': res['metric_limits'],
        'pages': [{k: v for k, v in p.items() if k != 'page_geometry'} for p in res['pages']],
        'endpoint_sensitivity': sens,
    }


def failure_attribution(payload: Dict[str, Any], scored: Dict[str, Any]) -> Dict[str, Any]:
    """把每页的失败归到：区域定位 / 分组 / 框边界 / 复识 / 输入条件。

    注意：「检出了列」不能排除区域定位错误——列数正常但列框跨栏时，
    双轨组落在错误的列框里，归因仍是 grouping。
    """
    per_page = {p['page_id']: p for p in scored['pages']}
    out = []
    for run in payload.get('page_runs') or []:
        pid = run['page_id']
        s = per_page.get(pid)
        if not s:
            out.append({'page_id': pid, 'status': run.get('status'), 'stage': 'input',
                        'reason': run.get('error') or 'no_scoring_record'})
            continue
        if s.get('status') == 'data_error':
            # 这类页没有 thresholds，不能往下走；但页号、状态、原因必须进报告
            out.append({'page_id': pid, 'status': 'data_error', 'stage': 'data_error',
                        'reason': s.get('data_error') or '评测区域缺坐标或 page_id 冲突',
                        'n_columns_detected': run.get('n_columns_detected'),
                        'timing_s': run.get('timing_s')})
            continue
        if not s.get('thresholds'):
            out.append({'page_id': pid, 'status': run.get('status'), 'stage': 'data_error',
                        'reason': '该页没有任何阈值级结果（thresholds 为空）'})
            continue
        t = s['thresholds'].get('0.5') or next(iter(s['thresholds'].values()))
        stage, reason = None, ''
        if run.get('status') in ('failed', 'skipped'):
            stage = 'input'
            reason = run.get('error') or run.get('skip_reason') or run.get('status')
        elif t['annotations_total'] == 0:
            stage = 'no_gt_page'
            reason = (f"无正例标注（只划负例）；预测 {run.get('n_predictions', 0)} 个，"
                      f"落在确定负例区内 {t['negative_confirmed_fp'] + t['negative_suspected_fp']} 个")
        elif run.get('n_columns_detected') == 0:
            stage, reason = 'region_localization', '未检出任何列'
        elif t['group_hits'] == 0:
            stage = 'grouping'
            reason = (f"有 {run.get('n_predictions', 0)} 个预测但 0 组命中（列检出 "
                      f"{run.get('n_columns_detected')}）；列数正常不等于列框正确，"
                      f"需看核对图确认是否跨栏合并")
        elif t['tp'] > 0 and t['lane']['counts'].get('child_localization_failure', 0) > 0:
            stage, reason = 'box_boundary', '组命中但子框未达阈值'
        elif t['tp'] == 0 and t['new_candidates'] + t['negative_confirmed_fp'] + \
                t['negative_suspected_fp'] > 0:
            stage, reason = 'grouping', '预测落在别处（分组/定位偏移）'
        out.append({'page_id': pid, 'status': run.get('status'), 'stage': stage or 'none',
                    'reason': reason, 'timing_s': run.get('timing_s'),
                    'n_columns_detected': run.get('n_columns_detected')})
    return {'per_page': out,
            'stage_counts': _count([r['stage'] for r in out]),
            'data_error_pages': [r['page_id'] for r in out if r['stage'] == 'data_error'],
            'rule': '每页只记一个主因，按 data_error > input > region_localization > '
                    'grouping > box_boundary 顺序判定；data_error 页不出分数'}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _count(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='从保存的预测 + 冻结配置重算指标')
    ap.add_argument('--predictions', nargs='*', default=['C'],
                    help='路径代号：A 只出文本；B/C/X 出几何指标')
    ap.add_argument('--file', nargs='*', default=None, help='直接给预测文件路径')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG),
                    help='评测配置（默认 stage3/evaluation_config.json）')
    ap.add_argument('--override', nargs='*', default=None,
                    help="临时覆盖，形如 group_matching.thresholds='[0.5,0.6]'；会生成新配置版本")
    ap.add_argument('--ann', default=None,
                    help='正例标注文件（默认 stage2/annotations.json；换口径时指向新版）')
    ap.add_argument('--neg', default=None,
                    help='负例标注文件（默认 stage2/negative_regions.json；换口径时指向新版）')
    ap.add_argument('--out-dir', default=str(OUT / 'metrics'))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        print(f'配置不存在：{cfg_path}')
        return 2
    resolved = resolve_config(cfg_path, args.override or [])
    runtime = runtime_config(resolved['doc'])
    problems = validate_runtime(runtime)
    if problems:
        print('配置不可用：')
        for p in problems:
            print('  -', p)
        return 3
    # 标注口径可替换：默认仍是冻结版；换口径时必须显式传 --ann/--neg，
    # 免得「忘了传」被当成「用旧口径重算」，那会让两种口径的数字混在一张表里。
    ann_path = Path(args.ann) if args.ann else (S2 / 'annotations.json')
    neg_path = Path(args.neg) if args.neg else (S2 / 'negative_regions.json')
    if not ann_path.is_absolute():
        ann_path = ROOT / ann_path
    if not neg_path.is_absolute():
        neg_path = ROOT / neg_path

    config_meta = {
        'version': resolved['version'],
        'overrides': resolved['overrides'],
        'source': resolved['source'],
        'runtime': runtime,
        'note': 'runtime 是实际传给评测器的参数；source.sha256 是配置文件指纹',
        'annotations': {'path': str(ann_path.relative_to(ROOT)).replace('\\', '/'),
                        'sha256': _sha256_file(ann_path)},
        'negatives': {'path': str(neg_path.relative_to(ROOT)).replace('\\', '/'),
                      'sha256': _sha256_file(neg_path)},
    }

    ann = load(ann_path)
    neg = load(neg_path)
    sp = load(S2 / 'dataset_splits.json')
    split_map = page_split_map(sp)

    files: List[Path] = []
    if args.file:
        files += [Path(f).resolve() for f in args.file]
    for code in args.predictions or []:
        code = code.upper()
        if code == 'A':
            continue
        f = OUT / 'predictions' / FILES[code]
        if f.exists():
            files.append(f)
    if not files:
        print('没有可评的预测文件')
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for f in files:
        scored = score_file(f, ann, neg, split_map, runtime, config_meta)
        scored['failure_attribution'] = failure_attribution(load(f), scored)
        scored['generated_at'] = datetime.now(TZ).isoformat(timespec='seconds')
        name = f.stem.replace('path_', '') + '.json'
        (out_dir / name).write_text(json.dumps(scored, ensure_ascii=False, indent=1) + '\n',
                                    encoding='utf-8')
        o = scored['scores']['overall']
        index.append({'path': scored['path'], 'file': name,
                      'predictions_content_sha256': scored['predictions_content_sha256'][:16],
                      'n_predictions': scored['n_predictions'],
                      'hit_rate_0.5': o.get('0.5', {}).get('hit_rate_on_known'),
                      'hit_rate_0.75': o.get('0.75', {}).get('hit_rate_on_known'),
                      'group_hits_0.5': o.get('0.5', {}).get('group_hits'),
                      'annotations': o.get('0.5', {}).get('annotations_total')})
        if not args.quiet:
            print(f"\n=== {scored['path']} ({name})  配置 {config_meta['version']} ===")
            for tk in o:
                x = o[tk]
                print(f"  IoU {tk}: 命中 {x['group_hits']}/{x['annotations_total']} "
                      f"(hit_rate {x['hit_rate_on_known']}) | TP {x['tp']} FP {x['fp']} FN {x['fn']} | "
                      f"ignored {x['ignored']} ambiguous {x['pending_ambiguous']} 候选 {x['new_candidates']} | "
                      f"重复FP {x['duplicate_extra_fp']} | 负例误检 {x['negative_confirmed_fp']}/"
                      f"{x['negative_suspected_fp']} | 非法 {x['invalid_predictions']} 降级 {x['degraded_predictions']}")
                print(f"          轨级分母 {x['lane_denominator']}（排除 {x['lane_excluded']}）"
                      f" 分类 {x['lane_counts']} | 状态划分 {x['status_partition']}")
            if scored.get('data_errors'):
                print(f"  数据错误 {len(scored['data_errors'])} 页（该页不出分数）：")
                for d in scored['data_errors']:
                    print(f"    {d['page_id']:22s} {d['error'][:80]}")
            print('  失败归因', scored['failure_attribution']['stage_counts'])
            for r in scored['failure_attribution']['per_page']:
                if r['stage'] not in ('none',):
                    print(f"    {r['page_id']:22s} {r['stage']:20s} {r['reason'][:75]}")
    (out_dir / 'index.json').write_text(
        json.dumps({'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
                    'config': config_meta, 'items': index},
                   ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not args.quiet:
        print('\n→', out_dir.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
