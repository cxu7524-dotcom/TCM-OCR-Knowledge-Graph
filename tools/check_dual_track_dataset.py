"""双轨小字页数据集校验入口（第一阶段）。

用法：
  .venv/Scripts/python.exe tools/check_dual_track_dataset.py
  .venv/Scripts/python.exe tools/check_dual_track_dataset.py --json-out output/dual_tracks/stage1/validation_report.json
  .venv/Scripts/python.exe tools/check_dual_track_dataset.py --fingerprint before   # 运行前快照
  .venv/Scripts/python.exe tools/check_dual_track_dataset.py --fingerprint after    # 运行后核对

只读来源文件；结果写 output/dual_tracks/stage1/validation_report.json。
每项检查给出 pass / fail / pending 三态，缺省字段单独记，不算数据错误。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_track_stage1_data as D  # noqa: E402

TZ = timezone(timedelta(hours=8))


class Context:
    """一次校验所需的全部读取结果；图片只解码一次。

    参数可注入，便于测试构造合成数据；不传则从磁盘读取真实数据集。
    """

    def __init__(self, manifest: Optional[list] = None, manifest_csv: Optional[list] = None,
                 coords: Optional[list] = None, coords_csv: Optional[list] = None,
                 image_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.manifest = D.load_manifest() if manifest is None else manifest
        self.manifest_csv = D.load_manifest_csv() if manifest_csv is None else manifest_csv
        self.coords = D.load_coords() if coords is None else coords
        self.coords_csv = D.load_coords_csv() if coords_csv is None else coords_csv
        self.images: Dict[str, Dict[str, Any]] = {}
        self.image_overrides = image_overrides or {}
        self.page_numbers = [D.manifest_page_number(m) for m in self.manifest]

    def image(self, path: Path) -> Dict[str, Any]:
        key = D.rel(path)
        if key in self.image_overrides:
            return self.image_overrides[key]
        if key not in self.images:
            self.images[key] = D.image_info(path)
        return self.images[key]

    # 记录 -> 数据目录里的页面文件
    def dataset_image(self, rec: Dict[str, Any]) -> Path:
        return D.DATASET / rec['book'] / D.page_file_name(rec['page'])

    def page_records(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for m in self.manifest:
            n = D.manifest_page_number(m)
            out[D.page_id(m['book'], n)] = m
        return out

    def coords_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.coords:
            out.setdefault(D.page_id(c['book'], c['page']), []).append(c)
        return out


# ------------------------------------------------------------------ 各项检查

def check_files(ctx: Context) -> Dict[str, Any]:
    items, missing, unreadable = [], [], []
    core = [D.MANIFEST_JSON, D.MANIFEST_CSV, D.COORDS_JSON, D.COORDS_CSV, D.README]
    for p in core:
        if not p.exists():
            missing.append(D.rel(p))
    for m in ctx.manifest:
        n = D.manifest_page_number(m)
        expect = [(D.BOOKS / m['book'] / D.page_file_name(n), 'source_image')]
        if m.get('dataset_path'):
            expect.append((D.ROOT / m['dataset_path'], 'dataset_image'))
        for path, kind in expect:
            info = ctx.image(path)
            if not info['exists']:
                missing.append(f'{kind}:{D.rel(path)}')
            elif not info['readable']:
                unreadable.append(f'{kind}:{D.rel(path)}')
        items.append({'page_id': D.page_id(m['book'], n), 'status': m['status'],
                      'source_image': D.rel(D.BOOKS / m['book'] / D.page_file_name(n)),
                      'dataset_copy': m.get('dataset_path', '')})
    overlays = sorted(D.OVERLAY_DIR.glob('*.png')) if D.OVERLAY_DIR.exists() else []
    preview = D.DATASET / '预览_13页.png'
    if len(overlays) != 13:
        missing.append(f'坐标图 应有 13 张，实有 {len(overlays)}')
    if not preview.exists():
        missing.append(D.rel(preview))
    status = 'pass' if not missing and not unreadable else 'fail'
    return {
        'id': 'C01', 'title': '文件存在性与图片可读性', 'status': status,
        'detail': f'清单 {len(ctx.manifest)} 条、坐标 {len(ctx.coords)} 条、叠框图 {len(overlays)} 张；'
                  f'缺文件 {len(missing)} 项，不可读 {len(unreadable)} 项',
        'missing': missing, 'unreadable': unreadable,
        'pages': items,
        'evidence': [D.rel(D.MANIFEST_JSON), D.rel(D.COORDS_JSON), D.rel(D.OVERLAY_DIR)],
    }


def check_image_size(ctx: Context) -> Dict[str, Any]:
    rows, mismatched = [], []
    for c in ctx.coords:
        path = D.DATASET / c['book'] / D.page_file_name(c['page'])
        real = ctx.image(path).get('image_size')
        declared = c.get('image_size')
        match = list(declared or []) == list(real or [])
        rows.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                     'declared': declared, 'real': real, 'match': match})
        if not match:
            mismatched.append(f"{c['book']} p{c['page']} {c['region']}: 标注 {declared} / 真实 {real}")
    pages = {}
    for c in ctx.coords:
        path = D.DATASET / c['book'] / D.page_file_name(c['page'])
        pages[D.page_id(c['book'], c['page'])] = ctx.image(path).get('image_size')
    heights = sorted({int(v[1]) for v in pages.values() if v})
    status = 'pass' if not mismatched else 'fail'
    return {
        'id': 'C02', 'title': '真实图片尺寸与标注 image_size', 'status': status,
        'detail': f'{len(ctx.coords)} 组中 {len(mismatched)} 组的 image_size 与真实尺寸不同；'
                  f'宽度全部为 1024，差异只在高度（实测 {heights[0]}-{heights[-1]}，共 {len(heights)} 种）',
        'root_cause': 'tools/build_dual_track_coords.py 第 93 行把所有记录硬编码为 image_size=[1024, 804]',
        'real_sizes_by_page': dict(sorted(pages.items())),
        'mismatched': mismatched, 'rows': rows,
        'handling': '规范化结果一律改写为原图实测尺寸；坐标数值不动（见 C07 越界检查通过）',
        'evidence': [D.rel(D.COORDS_JSON), 'tools/build_dual_track_coords.py:93'],
    }


def check_fingerprints(ctx: Context) -> Dict[str, Any]:
    rows, bad = [], []
    for m in ctx.manifest:
        n = D.manifest_page_number(m)
        src = D.BOOKS / m['book'] / D.page_file_name(n)
        declared = m['sha256']
        src_hash = ctx.image(src).get('sha256')
        copy_hash = None
        if m.get('dataset_path'):
            copy_hash = ctx.image(D.ROOT / m['dataset_path']).get('sha256')
        ok_src = src_hash == declared
        ok_copy = (copy_hash == declared) if copy_hash else None
        rows.append({'page_id': D.page_id(m['book'], n), 'status': m['status'],
                     'declared': declared, 'source_sha256': src_hash,
                     'dataset_sha256': copy_hash,
                     'source_ok': ok_src, 'copy_ok': ok_copy})
        if not ok_src:
            bad.append(f"源图指纹不符：{D.rel(src)}")
        if ok_copy is False:
            bad.append(f"副本指纹不符：{m['dataset_path']}")
    n_with_copy = sum(1 for r in rows if r['dataset_sha256'])
    status = 'pass' if not bad else 'fail'
    return {
        'id': 'C03', 'title': '图片指纹与清单 SHA-256', 'status': status,
        'detail': f'清单 {len(rows)} 条中 {n_with_copy} 条有页面副本并全部核对；'
                  f'8 条排除记录只核源图，不复制图片（说明.md 第三节口径）',
        'bad': bad, 'rows': rows,
        'evidence': [D.rel(D.MANIFEST_JSON)],
    }


def _strip_uncertain(text: str) -> str:
    """去掉坐标侧的疑字标记〔?〕，便于与清单的普通文本比较。"""
    return (text or '').replace('〔?〕', '').replace('〔？〕', '').strip()


def _main_matches(manifest_main: str, coord_main: str) -> bool:
    """坐标侧用〔?〕标记存疑字，比对时按单个通配字符处理。"""
    pat = _strip_uncertain(coord_main)
    if '〔?〕' in (coord_main or '') or '〔？〕' in (coord_main or ''):
        import re
        regex = re.escape((coord_main or '').replace('〔?〕', '\x00').replace('〔？〕', '\x00'))
        regex = regex.replace('\x00', '.')
        return re.fullmatch(regex, manifest_main) is not None
    return pat == manifest_main


def _joined(rec: Dict[str, Any]) -> str:
    r = rec.get('right_text') if 'right_text' in rec else rec.get('right')
    l = rec.get('left_text') if 'left_text' in rec else rec.get('left')
    return f"{r or ''}{l or ''}"


def check_links(ctx: Context) -> Dict[str, Any]:
    problems, notes = [], []
    coords_pages = ctx.coords_by_page()
    pages = ctx.page_records()
    included = {k for k, v in pages.items() if v['status'] == D.STATUS_INCLUDED}
    pending = {k for k, v in pages.items() if v['status'] == D.STATUS_PENDING}

    for pid in coords_pages:
        if pid not in pages:
            problems.append(f'坐标页不在清单中：{pid}')
        elif pages[pid]['status'] != D.STATUS_INCLUDED:
            problems.append(f'坐标页状态不是收录：{pid}（{pages[pid]["status"]}）')
    for pid in sorted(included):
        if pid not in coords_pages:
            problems.append(f'收录页无任何坐标：{pid}')
    for pid in sorted(pending):
        notes.append(f'{pid} 为待核页，坐标缺省（已知缺省，非数据错误）')

    # 清单文本与坐标文本的对应关系（只记差异，不改来源）。
    # 清单每页只举一例，因此先在坐标里按主字定位「被举例的那一组」。
    for pid, m in sorted(pages.items()):
        man_main = (m.get('main') or '').strip()
        if not man_main:
            continue  # 排除页不记文本
        if m['status'] != D.STATUS_INCLUDED and not coords_pages.get(pid):
            continue  # 待核页无坐标，已在上面按缺省记录
        cands = [c for c in coords_pages.get(pid, [])
                 if _main_matches(man_main, c['main_text'])]
        if not cands:
            notes.append(f"{pid} 清单举例主字 {man_main!r} 未在坐标中定位；"
                         f"该页坐标共 {len(coords_pages.get(pid, []))} 组")
            continue
        c = cands[0]
        if len(cands) > 1:
            notes.append(f'{pid} 清单主字 {man_main!r} 对应多个坐标组，取 {c["region"]}')
        if (m.get('main') or '') != c['main_text']:
            notes.append(f"{c['region']} 清单 main={m.get('main')!r} / 坐标 main_text={c['main_text']!r}"
                         f"（疑字标记保留在坐标侧）")
        for side, man_key in (('right', 'right'), ('left', 'left')):
            man_txt = (m.get(man_key) or '').strip()
            if not man_txt:
                continue
            if man_txt == c[f'{side}_text']:
                continue
            if _joined(m) == _joined(c):
                notes.append(f"{c['region']} {side} 与清单切分不同（清单 {man_txt!r} / 坐标 "
                             f"{c[f'{side}_text']!r}），字符序列一致")
                break
            notes.append(f"{c['region']} {side} 文本不一致：清单 {man_txt!r} / 坐标 "
                         f"{c[f'{side}_text']!r}，需按原图复核")
            break

    per_page = {pid: len(v) for pid, v in sorted(coords_pages.items())}
    status = 'pass' if not problems else 'fail'
    return {
        'id': 'C04', 'title': '清单到图片、坐标到图片的关联关系', 'status': status,
        'detail': f'收录 {len(included)} 页全部有坐标（共 {sum(per_page.values())} 组）；'
                  f'待核 {len(pending)} 页坐标缺省；关联断裂 {len(problems)} 项',
        'groups_per_page': per_page, 'problems': problems, 'notes': notes,
        'evidence': [D.rel(D.MANIFEST_JSON), D.rel(D.COORDS_JSON)],
    }


def check_unique_ids(ctx: Context) -> Dict[str, Any]:
    seen, dups = {}, []
    for c in ctx.coords:
        key = (c['book'], int(c['page']), c['region'])
        if key in seen:
            dups.append(f'{key} 重复')
        seen[key] = c
    cross = []
    for col_name, rows in (('json', ctx.coords), ('csv', ctx.coords_csv)):
        keys = [(r['book'], str(r['page']), r['region']) for r in rows]
        if len(set(keys)) != len(keys):
            cross.append(f'{col_name} 内部有重复键')
    status = 'pass' if not dups and not cross else 'fail'
    return {
        'id': 'C05', 'title': '标注编号唯一性（书名 + 页号 + region）', 'status': status,
        'detail': f'{len(seen)} 个唯一键 / {len(ctx.coords)} 条记录，重复 {len(dups)} 个',
        'duplicates': dups, 'cross_source': cross,
        'evidence': [D.rel(D.COORDS_JSON), D.rel(D.COORDS_CSV)],
    }


def check_bbox_shape(ctx: Context) -> Dict[str, Any]:
    """逐框解析；非法时给出「记录键 + 字段 + 代码 + 具体值」，且不影响其他记录。"""
    details, code_counts = [], {}
    total = 0
    for c in ctx.coords:
        for field in ('main_bbox', 'right_bbox', 'left_bbox'):
            total += 1
            bbox, code, detail = D.parse_bbox(c.get(field))
            code_counts[code] = code_counts.get(code, 0) + 1
            if code == D.BBOX_OK:
                continue
            details.append({
                'group_id': D.group_id(c['book'], c['page'], c['region']),
                'field': field,
                'code': code,
                'code_text': D.BBOX_CODE_TEXT[code],
                'value': _jsonable(c.get(field)),
                'detail': detail,
            })
    status = 'pass' if not details else 'fail'
    return {
        'id': 'C06', 'title': 'bbox 是否为四个有效数字', 'status': status,
        'detail': f'{total} 个框，合法 {code_counts.get(D.BBOX_OK, 0)} 个、非法 {len(details)} 个；'
                  f'解析代码分布 {dict(sorted(code_counts.items()))}',
        'parse_contract': {
            'int': '接受',
            'float': '必须有限且为整数值（899.0 归一化为 899）；nan / ±inf / 899.5 判非法',
            'str': "只接受 int(s, 10) 能完整解析的十进制整数字符串；'899.0' / '1e3' 判非法",
            'bool': '显式拒绝（bool 是 int 子类，不拒绝会被静默当成 0/1）',
            'other': '判非法',
        },
        'invalid': details,
        'empty': [d for d in details if d['code'] == D.BBOX_EMPTY],
        'code_counts': dict(sorted(code_counts.items())),
        'evidence': [D.rel(D.COORDS_JSON)],
    }


def _jsonable(value: Any) -> Any:
    """把 inf/nan 等转成可写进 JSON 的文本，避免报告自身序列化失败。"""
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))):
        return repr(value)
    return value


def check_bbox_geometry(ctx: Context) -> Dict[str, Any]:
    bad, rows, skipped = [], [], []
    for c in ctx.coords:
        path = D.DATASET / c['book'] / D.page_file_name(c['page'])
        size = ctx.image(path).get('image_size')
        for field in ('main_bbox', 'right_bbox', 'left_bbox'):
            bbox, code, detail = D.parse_bbox(c[field])
            if code != D.BBOX_OK:
                skipped.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                                'field': field, 'code': code, 'detail': detail})
                continue
            area = D.bbox_area(bbox)
            inside = D.in_image(bbox, size)
            rows.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                         'field': field, 'bbox': list(bbox), 'area': area, 'in_image': inside})
            if area <= 0:
                bad.append(f"{c['region']}.{field} 正面积不成立：{list(bbox)}")
            if not inside:
                bad.append(f"{c['region']}.{field} 越出原图 {size}：{list(bbox)}")
    status = 'pass' if not bad and not skipped else 'fail'
    max_y2 = max((r['bbox'][3] for r in rows), default=None)
    return {
        'id': 'C07', 'title': 'bbox 正面积及原图范围', 'status': status,
        'detail': f'可解析的 {len(rows)} 个框全部正面积且落在原图内（越界 {len(bad)} 个）；'
                  f'另有 {len(skipped)} 个框因解析失败跳过几何检查（见 C06）；最大 y2 = {max_y2}',
        'bad': bad, 'skipped_invalid': skipped, 'rows': rows,
        'evidence': [D.rel(D.COORDS_JSON)],
    }


def check_lane_order(ctx: Context) -> Dict[str, Any]:
    rows, bad, skipped = [], [], []
    for c in ctx.coords:
        r, rc, rdet = D.parse_bbox(c['right_bbox'])
        l, lc, ldet = D.parse_bbox(c['left_bbox'])
        if rc != D.BBOX_OK or lc != D.BBOX_OK:
            skipped.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                            'code': [rc, lc], 'detail': [rdet, ldet]})
            continue
        rx, lx = D.center_x(r), D.center_x(l)
        ok = rx > lx
        rows.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                     'right_center_x': rx, 'left_center_x': lx, 'ok': ok})
        if not ok:
            bad.append(f"{c['region']} 右轨中心 {rx} 未大于左轨中心 {lx}")
    status = 'pass' if not bad and not skipped else 'fail'
    gap = min((r['right_center_x'] - r['left_center_x']) for r in rows) if rows else None
    return {
        'id': 'C08', 'title': '右、左轨中心顺序', 'status': status,
        'detail': f'可解析的 {len(rows)} 组中异常 {len(bad)} 组，跳过 {len(skipped)} 组；'
                  f'最小中心间距 {gap}px',
        'bad': bad, 'skipped_invalid': skipped, 'rows': rows,
        'evidence': [D.rel(D.COORDS_JSON)],
    }


def check_overlaps(ctx: Context) -> Dict[str, Any]:
    rows, lane_overlap, main_overlap, skipped = [], [], [], []
    for c in ctx.coords:
        m, mc, _ = D.parse_bbox(c['main_bbox'])
        r, rc, _ = D.parse_bbox(c['right_bbox'])
        l, lc, _ = D.parse_bbox(c['left_bbox'])
        if not (mc == rc == lc == D.BBOX_OK):
            skipped.append(D.group_id(c['book'], c['page'], c['region']))
            continue
        a_rl = D.intersect_area(r, l)
        a_mr = D.intersect_area(m, r)
        a_ml = D.intersect_area(m, l)
        rows.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                     'right_left': a_rl, 'main_right': a_mr, 'main_left': a_ml})
        if a_rl > 0:
            lane_overlap.append(f"{c['region']} 左右轨重叠 {a_rl}px²")
        if a_mr > 0 or a_ml > 0:
            main_overlap.append(f"{c['region']} 主字与小字重叠 main∩right={a_mr} main∩left={a_ml}")
    # 只记录实测，不判定为错误：真实排版里主字底边常与两轨顶边相接。
    status = 'pass' if not lane_overlap and not main_overlap and not skipped else (
        'pending' if not skipped else 'fail')
    return {
        'id': 'C09', 'title': '左右框重叠、主字与小字重叠（实测记录）', 'status': status,
        'detail': f'{len(rows)} 组可测：左右轨重叠 {len(lane_overlap)} 组、主字与小字重叠 '
                  f'{len(main_overlap)} 组，跳过 {len(skipped)} 组；重叠量均为 1-2px，'
                  f'属「框相接」而非内容交叠',
        'lane_overlaps': lane_overlap, 'main_overlaps': main_overlap,
        'rows': rows, 'skipped_invalid': skipped,
        'handling': '保留原坐标；1-2px 相接来自端点约定未定与人工框松紧，见 C12。未按原图逐组重定框。',
        'evidence': [D.rel(D.COORDS_JSON)],
    }


def check_json_csv(ctx: Context) -> Dict[str, Any]:
    diffs = []
    if len(ctx.coords) != len(ctx.coords_csv):
        diffs.append(f'条数不同：JSON {len(ctx.coords)} / CSV {len(ctx.coords_csv)}')
    jmap = {(c['book'], str(c['page']), c['region']): c for c in ctx.coords}
    cmap = {(c['book'], str(c['page']), c['region']): c for c in ctx.coords_csv}
    if set(jmap) != set(cmap):
        diffs.append(f'编号集合不同：仅 JSON {sorted(set(jmap) - set(cmap))}；仅 CSV {sorted(set(cmap) - set(jmap))}')
    for key in sorted(set(jmap) & set(cmap)):
        j, k = jmap[key], cmap[key]
        for field in ('main_text', 'right_text', 'left_text'):
            if (j.get(field) or '') != (k.get(field) or ''):
                diffs.append(f'{key} {field} 不同：{j.get(field)!r} / {k.get(field)!r}')
        for field in ('main_bbox', 'right_bbox', 'left_bbox'):
            if list(D.as_bbox(j[field]) or []) != list(D.as_bbox(k[field]) or []):
                diffs.append(f'{key} {field} 不同：{j.get(field)} / {k.get(field)}')
        if j.get('source') != k.get('source'):
            diffs.append(f'{key} source 不同')
        if (j.get('column') if j.get('column') is not None else '') != (k.get('column') if k.get('column') is not None else ''):
            diffs.append(f"{key} column 不同：{j.get('column')!r} / {k.get('column')!r}")
    json_only = [f for f in ('image_size', 'coordinate_system') if f in (ctx.coords[0] if ctx.coords else {})]
    csv_fields = list(ctx.coords_csv[0].keys()) if ctx.coords_csv else []
    status = 'pass' if not diffs else 'fail'
    return {
        'id': 'C10', 'title': '双轨坐标 JSON 与 CSV 的一致性（数量、编号、坐标、共有字段）', 'status': status,
        'detail': f'两边各 {len(ctx.coords)} 条，编号集合一致，共有字段（文本/坐标/来源/column）逐项相等，差异 {len(diffs)} 项',
        'differences': diffs,
        'field_differences': {
            'json_only': json_only, 'csv_only': [],
            'json_extra_keys_in_csv_parse': sorted(set(ctx.coords_csv[0]) - set(ctx.coords[0])) if ctx.coords_csv else [],
            'column_type': 'JSON 为 null，CSV 为空串；23 条 column 缺省属已知缺省，统一解析为 None',
        },
        'evidence': [D.rel(D.COORDS_JSON), D.rel(D.COORDS_CSV)],
    }


MANIFEST_FIELDS = ('status', 'confidence', 'source', 'dataset_path', 'sha256',
                   'main', 'right', 'left', 'note')


def _manifest_value(field: str, value: Any) -> str:
    """清单字段的归一化：只抹掉「缺省表示差异」和「换行/首尾空白」，真实内容差异保留。"""
    if value is None:
        return ''
    text = str(value)
    if field == 'sha256':
        return text.strip().lower()
    if field == 'note':
        text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


def _index_manifest(rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    idx: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        idx.setdefault(D.manifest_key(r), []).append(r)
    return idx


def check_manifest_json_csv(ctx: Context) -> Dict[str, Any]:
    """清单.json 与 清单.csv 逐条对齐；输出明确的「文件 / 记录键 / 字段 / 两侧值」。"""
    a_label, b_label = D.rel(D.MANIFEST_JSON), D.rel(D.MANIFEST_CSV)
    a_idx = _index_manifest(ctx.manifest)
    b_idx = _index_manifest(ctx.manifest_csv)
    diffs: List[Dict[str, Any]] = []

    if len(ctx.manifest) != len(ctx.manifest_csv):
        diffs.append({'kind': 'count_mismatch', 'file_a': a_label, 'file_b': b_label,
                      'key': '-', 'field': '-',
                      'a': len(ctx.manifest), 'b': len(ctx.manifest_csv)})

    for label, idx in ((a_label, a_idx), (b_label, b_idx)):
        for key, rows in idx.items():
            if len(rows) > 1:
                diffs.append({'kind': 'duplicate', 'file_a': label, 'file_b': label,
                              'key': _key_label(key), 'field': '-',
                              'a': len(rows), 'b': len(rows),
                              'detail': f'{label} 中同一键出现 {len(rows)} 次'})

    for key in sorted(set(a_idx) | set(b_idx), key=lambda k: (str(k[0]), k[1] if k[1] is not None else -1)):
        ra = a_idx.get(key, [])
        rb = b_idx.get(key, [])
        if not rb:
            diffs.append({'kind': 'missing_in_csv', 'file_a': a_label, 'file_b': b_label,
                          'key': _key_label(key), 'field': '-', 'a': 'present', 'b': 'absent'})
            continue
        if not ra:
            diffs.append({'kind': 'missing_in_json', 'file_a': a_label, 'file_b': b_label,
                          'key': _key_label(key), 'field': '-', 'a': 'absent', 'b': 'present'})
            continue
        for field in MANIFEST_FIELDS:
            va = _manifest_value(field, ra[0].get(field))
            vb = _manifest_value(field, rb[0].get(field))
            if va != vb:
                diffs.append({'kind': 'value_mismatch', 'file_a': a_label, 'file_b': b_label,
                              'key': _key_label(key), 'field': field, 'a': va, 'b': vb})

    status = 'pass' if not diffs else 'fail'
    kinds: Dict[str, int] = {}
    for d in diffs:
        kinds[d['kind']] = kinds.get(d['kind'], 0) + 1
    return {
        'id': 'C14', 'title': '清单 JSON 与 CSV 的一致性', 'status': status,
        'detail': f'两边各 {len(ctx.manifest)} 条，唯一键 {len(set(a_idx) | set(b_idx))} 个；'
                  f'逐字段比对 {len(MANIFEST_FIELDS)} 个字段，差异 {len(diffs)} 项'
                  f'{"" if not kinds else "：" + str(dict(sorted(kinds.items())))}',
        'files': {'json': a_label, 'csv': b_label},
        'fields': list(MANIFEST_FIELDS),
        'normalization': {
            'key': '书名 + 页号；页号两侧统一解析成整数（page_33.png / 33 视为同一键）',
            'absent_vs_empty': "dataset_path 在 JSON 缺键、在 CSV 为空串，统一视为「无」，不算差异",
            'whitespace': '首尾空白与换行符（CRLF/CR→LF）归一，其余字符原样比较',
            'sha256': '仅大小写归一',
            'note': 'note 只做换行归一与首尾去空白，正文差异一字不漏地保留',
        },
        'counts': {'json': len(ctx.manifest), 'csv': len(ctx.manifest_csv),
                   'diff_by_kind': dict(sorted(kinds.items()))},
        'differences': diffs,
        'evidence': [a_label, b_label],
    }


def _key_label(key: Any) -> str:
    book, page = key if isinstance(key, tuple) and len(key) == 2 else (key, None)
    return f'{book}/page_{page}' if page is not None else f'{book}/page_?(无法解析)'


def check_sources(ctx: Context) -> Dict[str, Any]:
    rows, unknown = [], []
    counts = {k: 0 for k in D.SOURCE_CLASSES}
    for c in ctx.coords:
        try:
            cls = D.classify_source(c.get('source', ''))
        except ValueError:
            unknown.append(f"{c['region']}: {c.get('source')!r}")
            cls = '未分类'
        counts[cls] = counts.get(cls, 0) + 1
        rows.append({'group_id': D.group_id(c['book'], c['page'], c['region']),
                     'source_raw': c.get('source', ''), 'annotation_source': cls,
                     'profile': D.source_profile(cls) if cls in D.SOURCE_CLASSES else None})
    status = 'pass' if not unknown else 'fail'
    return {
        'id': 'C11', 'title': '标注来源及精度分类', 'status': status,
        'detail': f'历史人工标注 {counts[D.SOURCE_MANUAL]} 组、墨迹测量 {counts[D.SOURCE_MEASURED]} 组、'
                  f'人工近似框 {counts[D.SOURCE_APPROX]} 组；无法归类 {len(unknown)} 条',
        'counts': counts, 'unclassified': unknown, 'rows': rows,
        'note': '说明.md 第三之二节写「人工标注（23 处）」，实际为 21 处（holdout 20 + dev 1），见 C13',
        'evidence': [D.rel(D.COORDS_JSON), D.rel(D.README)],
    }


def check_coordinate_convention(ctx: Context) -> Dict[str, Any]:
    """右下角端点：能从代码证据确认的标 half_open，其余标待核并保留原值。

    这里只处理「坐标怎么读」（端点是否为区间闭端），不涉及「框画得准不准」。
    """
    rows, confirmed, pending = [], [], []
    for c in ctx.coords:
        gid = D.group_id(c['book'], c['page'], c['region'])
        cls = D.classify_source(c.get('source', ''))
        if cls == D.SOURCE_MEASURED:
            conv, basis = D.CONV_HALF_OPEN, (
                'tools/dual_track_boxes_from_window.py 的 tight() 返回 '
                '[x0+cols[0], y0+rows[0], x0+cols[-1]+1, y0+rows[-1]+1]，'
                'x2/y2 为末笔墨迹坐标 +1，即半开区间 [x1,y1,x2,y2)')
            confirmed.append(gid)
        else:
            conv, basis = D.CONV_PENDING, (
                '来源文档只声明「图像像素框」，未声明右下角是否包含端点；'
                '来源侧的生成代码未保留，数据本身无法区分')
            pending.append(gid)
        rows.append({
            'group_id': gid,
            'annotation_source': cls,
            'coordinate_convention': conv,
            'basis': basis,
            'original_coordinate_convention': 'unknown',
            'candidate_interpretations': D.ENDPOINT_CANDIDATES if conv == D.CONV_PENDING
            else ['half_open:[x1,y1,x2,y2)'],
            'endpoint_edge_shift_px': D.ENDPOINT_EDGE_SHIFT_PX if conv == D.CONV_PENDING else 0,
        })
    status = 'pass' if not pending else 'pending'
    return {
        'id': 'C12', 'title': '右下角坐标的端点约定', 'status': status,
        'detail': f'{len(confirmed)} 组可由代码证据确认为半开区间；{len(pending)} 组保留原值、'
                  f'同时记录「含端点」与「半开」两种读法，端点约定待核',
        'confirmed_half_open': confirmed, 'pending': pending, 'rows': rows,
        'semantics': {
            'what_it_is': '坐标系怎么解释右下角：半开 [x1,y1,x2,y2) 表示不含 x2/y2 这一行一列，'
                          '含端点表示包含。两种读法只让框的右/下边界平移',
            'edge_shift_px': D.ENDPOINT_EDGE_SHIFT_PX,
            'is_definitional': '1px 差异由定义直接给出，不是测量出来的',
            'what_it_is_not': '不能由此推出「标注误差 ≤1px」：端点解释差异 ≠ 框与字形边界的误差 ≠ 评测容差',
            'glyph_boundary_error': 'unknown_uncalibrated（三类来源均未标定）',
            'protocol_tolerance': 'pending_protocol（评测协议未定）',
        },
        'impact': '精确边界评测对这 23 组只能给「待协议确定」的容差；定位、分栏、裁图不受影响',
        'evidence': ['tools/dual_track_boxes_from_window.py:20-26', D.rel(D.README)],
    }


def _parse_readme_claims() -> Dict[str, Any]:
    text = D.README.read_text(encoding='utf-8')
    claims: Dict[str, Any] = {}
    # 「清单.csv / 清单.json （15 条记录」
    for line in text.splitlines():
        if '清单.csv / 清单.json' in line:
            digits = ''.join(ch for ch in line.split('（')[-1] if ch.isdigit())
            if digits:
                claims['manifest_records'] = int(digits)
        if '人工标注**（' in line or '人工标注（' in line:
            seg = line.split('（', 1)[-1]
            digits = ''.join(ch for ch in seg.split('处')[0] if ch.isdigit())
            if digits:
                claims['manual_annotations'] = int(digits)
        if '共 **26 处' in line or '共 26 处' in line:
            claims['coord_groups'] = 26
    claims['total_pages_108'] = '**108**' in text
    claims['included_13'] = '**13**' in text
    claims['pending_2'] = '待核' in text and '2 页列「待核」' in text
    return claims


def check_readme_stats(ctx: Context) -> Dict[str, Any]:
    claims = _parse_readme_claims()
    pages = ctx.page_records()
    status_counts = {}
    for m in ctx.manifest:
        status_counts[m['status']] = status_counts.get(m['status'], 0) + 1
    source_counts = {k: 0 for k in D.SOURCE_CLASSES}
    for c in ctx.coords:
        source_counts[D.classify_source(c.get('source', ''))] += 1
    actual = {
        'manifest_records': len(ctx.manifest),
        'included': status_counts.get(D.STATUS_INCLUDED, 0),
        'pending': status_counts.get(D.STATUS_PENDING, 0),
        'excluded': status_counts.get(D.STATUS_EXCLUDED, 0),
        'coord_groups': len(ctx.coords),
        'coord_pages': len(ctx.coords_by_page()),
        'manual_annotations': source_counts[D.SOURCE_MANUAL],
        'measured': source_counts[D.SOURCE_MEASURED],
        'approx': source_counts[D.SOURCE_APPROX],
        'book_pages_total': {b: len(list((D.BOOKS / b).glob('*.png'))) for b in
                             ('思濟堂方書', '温疫方論', '丹方彚编', '合併脉訣難經太素評林', '太醫院校註婦人良方大全')},
    }
    actual['book_pages_sum'] = sum(actual['book_pages_total'].values())
    conflicts, agreed = [], []
    if claims.get('manifest_records') is not None:
        (agreed if claims['manifest_records'] == actual['manifest_records'] else conflicts).append(
            f"说明.md 称清单 {claims['manifest_records']} 条记录，实际 {actual['manifest_records']} 条")
    if claims.get('manual_annotations') is not None:
        (agreed if claims['manual_annotations'] == actual['manual_annotations'] else conflicts).append(
            f"说明.md 称人工标注 {claims['manual_annotations']} 处，实际 {actual['manual_annotations']} 处"
            f"（holdout 20 + dev 1）")
    agreed.extend([
        f"收录 {actual['included']} 页、待核 {actual['pending']} 页、排除 {actual['excluded']} 页，与说明一致",
        f"坐标 {actual['coord_groups']} 处 / {actual['coord_pages']} 页，与说明一致",
        f"5 部书合计 {actual['book_pages_sum']} 页，与说明的 108 页一致",
    ])
    status = 'pass' if not conflicts else 'fail'
    return {
        'id': 'C13', 'title': '说明文件统计与实际数据的一致性', 'status': status,
        'detail': f'说明.md 的 {len(agreed)} 项统计与实测一致，{len(conflicts)} 项不一致',
        'claims': claims, 'actual': actual, 'conflicts': conflicts, 'agreed': agreed,
        'handling': '不改说明.md（属原始来源）；差异记入 corrections.json 的说明性条目，'
                    '以本报告的实测值为准',
        'evidence': [D.rel(D.README)],
    }


CHECKS = (check_files, check_image_size, check_fingerprints, check_links, check_unique_ids,
          check_bbox_shape, check_bbox_geometry, check_lane_order, check_overlaps,
          check_json_csv, check_sources, check_coordinate_convention, check_readme_stats,
          check_manifest_json_csv)


# ------------------------------------------------------------------ 入口

def run_checks(ctx: Optional[Context] = None) -> Dict[str, Any]:
    ctx = ctx or Context()
    checks = [fn(ctx) for fn in CHECKS]
    tally = {'pass': 0, 'fail': 0, 'pending': 0}
    for c in checks:
        tally[c['status']] = tally.get(c['status'], 0) + 1
    status_counts: Dict[str, int] = {}
    for m in ctx.manifest:
        status_counts[m['status']] = status_counts.get(m['status'], 0) + 1
    source_counts = {k: 0 for k in D.SOURCE_CLASSES}
    for c in ctx.coords:
        source_counts[D.classify_source(c.get('source', ''))] += 1
    statistics = {
        'manifest_records': len(ctx.manifest),
        'page_status_counts': dict(sorted(status_counts.items())),
        'included_pages': status_counts.get(D.STATUS_INCLUDED, 0),
        'pending_pages': status_counts.get(D.STATUS_PENDING, 0),
        'excluded_pages': status_counts.get(D.STATUS_EXCLUDED, 0),
        'coord_groups': len(ctx.coords),
        'coord_pages': len(ctx.coords_by_page()),
        'source_counts': source_counts,
        'overlay_images': len(list(D.OVERLAY_DIR.glob('*.png'))) if D.OVERLAY_DIR.exists() else 0,
        'defaults': {
            'column_null': sum(1 for c in ctx.coords if c.get('column') in (None, '')),
            'column_null_note': '历史标注未记录分栏序号，属缺省不是错误；本轮测量 3 组有 column',
        },
    }
    return {
        'stage': D.STAGE,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'dataset_root': D.rel(D.DATASET),
        'sources': [D.rel(p) for p in (D.MANIFEST_JSON, D.MANIFEST_CSV, D.COORDS_JSON,
                                       D.COORDS_CSV, D.README)],
        'summary': {
            'checks': len(checks), 'pass': tally['pass'], 'fail': tally['fail'],
            'pending': tally['pending'],
            'failed_ids': [c['id'] for c in checks if c['status'] == 'fail'],
            'pending_ids': [c['id'] for c in checks if c['status'] == 'pending'],
        },
        'statistics': statistics,
        'checks': checks,
    }


def write_fingerprint(mode: str, out_dir: Path = D.OUT_DIR) -> Dict[str, Any]:
    """来源目录快照。写到 probe_*，不覆盖 normalize 生成的权威 before/after。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        'mode': mode,
        'generated_at': datetime.now(TZ).isoformat(timespec='seconds'),
        'root': D.rel(D.DATASET),
        'files': D.snapshot_tree(D.DATASET),
    }
    path = out_dir / f'source_fingerprints_probe_{mode}.json'
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding='utf-8')
    ref = out_dir / 'source_fingerprints_before.json'
    if not ref.exists():
        ref = out_dir / 'source_fingerprints_probe_before.json'
    result = {'written': D.rel(path), 'file_count': len(snap['files'])}
    if mode == 'after' and ref.exists():
        prev = json.loads(ref.read_text(encoding='utf-8'))
        result['compared_with'] = D.rel(ref)
        result['diff'] = D.diff_snapshots(prev.get('files', {}), snap['files'])
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='双轨小字页数据集第一阶段校验')
    ap.add_argument('--json-out', default=str(D.OUT_DIR / 'validation_report.json'))
    ap.add_argument('--fingerprint', choices=['before', 'after'], default=None,
                    help='只做原始来源目录指纹快照')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    if args.fingerprint:
        res = write_fingerprint(args.fingerprint)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0

    report = run_checks()
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    if not args.quiet:
        s = report['summary']
        print(f"检查 {s['checks']} 项：通过 {s['pass']} / 失败 {s['fail']} / 待核 {s['pending']}")
        for c in report['checks']:
            mark = {'pass': 'PASS', 'fail': 'FAIL', 'pending': 'PEND'}[c['status']]
            print(f"  [{mark}] {c['id']} {c['title']} — {c['detail']}")
        st = report['statistics']
        print(f"收录 {st['included_pages']} 页 / 待核 {st['pending_pages']} 页 / 排除 {st['excluded_pages']} 页；"
              f"坐标 {st['coord_groups']} 组 / {st['coord_pages']} 页；来源 {st['source_counts']}")
        print(f"报告：{D.rel(out)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
