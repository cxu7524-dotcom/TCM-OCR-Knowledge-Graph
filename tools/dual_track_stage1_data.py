"""双轨小字页数据集第一阶段的共享数据层。

职责：
  - 定位数据目录与来源文件（只读）
  - 读取清单 / 坐标 JSON / 坐标 CSV，并把 CSV 的字符串形态统一成 JSON 的形态
  - 图片真实尺寸、SHA-256、目录指纹
  - 坐标几何（面积、相交、并集、中心）与来源分类

本模块不写任何来源文件。规范化结果一律写到 output/dual_tracks/stage1/。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / '古籍评测集' / '双轨小字页'
BOOKS = ROOT / '古籍评测集' / '医家类'
OUT_DIR = ROOT / 'output' / 'dual_tracks' / 'stage1'

MANIFEST_JSON = DATASET / '清单.json'
MANIFEST_CSV = DATASET / '清单.csv'
COORDS_JSON = DATASET / '双轨坐标.json'
COORDS_CSV = DATASET / '双轨坐标.csv'
README = DATASET / '说明.md'
OVERLAY_DIR = DATASET / '坐标图'
PENDING_DIR = DATASET / '待核'

STAGE = 'stage1_dual_track_annotation_normalization'

# 三类来源：精度不同，不可混用（说明.md 第三之二节）。
SOURCE_MANUAL = '历史人工标注'
SOURCE_MEASURED = '墨迹测量'
SOURCE_APPROX = '人工近似框'
SOURCE_CLASSES = (SOURCE_MANUAL, SOURCE_MEASURED, SOURCE_APPROX)

# 端点约定取值。
CONV_HALF_OPEN = 'half_open'          # [x1,y1,x2,y2)
CONV_PENDING = 'pending_verification'  # 保留原值，待核

COVERAGE_PARTIAL = 'partial_positions'

STATUS_INCLUDED = '收录'
STATUS_PENDING = '待核'
STATUS_EXCLUDED = '排除'


# ---------------------------------------------------------------- 基础读写

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def snapshot_tree(root: Path, suffixes: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """目录内文件的相对路径 -> {sha256, size}，用于运行前后指纹核对。"""
    out: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        out[rel(p)] = {'sha256': sha256_file(p), 'size': p.stat().st_size}
    return out


def diff_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, List[str]]:
    b, a = set(before), set(after)
    changed = sorted(k for k in b & a if before[k].get('sha256') != after[k].get('sha256'))
    return {
        'added': sorted(a - b),
        'removed': sorted(b - a),
        'changed': changed,
        'identical': not (a - b) and not (b - a) and not changed,
    }


# ---------------------------------------------------------------- 图片

def image_info(path: Path) -> Dict[str, Any]:
    """真实尺寸 + 可读性。用 cv2 的 unicode 兜底读取，避免中文路径失败。"""
    info: Dict[str, Any] = {'path': rel(path), 'exists': path.exists(), 'readable': False}
    if not path.exists():
        return info
    info['sha256'] = sha256_file(path)
    try:
        from ocr.vertical import imread_unicode
        im = imread_unicode(str(path))
        if im is not None:
            info['readable'] = True
            info['image_size'] = [int(im.shape[1]), int(im.shape[0])]
            return info
    except Exception as exc:  # pragma: no cover - 依赖缺失时退回 PIL
        info['cv2_error'] = repr(exc)
    try:
        from PIL import Image
        with Image.open(path) as im2:
            info['readable'] = True
            info['image_size'] = [int(im2.size[0]), int(im2.size[1])]
            info['decoder'] = 'PIL'
    except Exception as exc:
        info['pil_error'] = repr(exc)
    return info


# ---------------------------------------------------------------- 清单 / 坐标

def page_file_name(page: Any) -> str:
    return f'page_{int(page)}.png'


def page_id(book: str, page: Any) -> str:
    return f'{book}/page_{int(page)}'


def group_id(book: str, page: Any, region: str) -> str:
    """唯一键：书名 + 页号 + region。"""
    return f'{book}/p{int(page)}/{region}'


def load_manifest() -> List[Dict[str, Any]]:
    return read_json(MANIFEST_JSON)


def load_manifest_csv() -> List[Dict[str, str]]:
    return read_csv_rows(MANIFEST_CSV)


def load_coords() -> List[Dict[str, Any]]:
    return read_json(COORDS_JSON)


def _parse_bbox_cell(cell: str) -> Any:
    cell = (cell or '').strip()
    if not cell:
        return None
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        return cell


def load_coords_csv() -> List[Dict[str, Any]]:
    """CSV -> 与 JSON 同形态的记录；page 转 int，bbox 转 list，column 空串转 None。"""
    rows = []
    for r in read_csv_rows(COORDS_CSV):
        page_raw = (r.get('page') or '').strip()
        col_raw = (r.get('column') or '').strip()
        rows.append({
            'book': (r.get('book') or '').strip(),
            'page': int(page_raw) if page_raw.lstrip('-').isdigit() else page_raw,
            'page_raw': page_raw,
            'region': (r.get('region') or '').strip(),
            'main_text': r.get('main_text') or '',
            'right_text': r.get('right_text') or '',
            'left_text': r.get('left_text') or '',
            'main_bbox': _parse_bbox_cell(r.get('main_bbox', '')),
            'right_bbox': _parse_bbox_cell(r.get('right_bbox', '')),
            'left_bbox': _parse_bbox_cell(r.get('left_bbox', '')),
            'source': (r.get('source') or '').strip(),
            'column': int(col_raw) if col_raw.isdigit() else None,
            'column_raw': col_raw,
        })
    return rows


# ---------------------------------------------------------------- 分类

def classify_source(source: str) -> str:
    """把来源字符串归到三类之一。无法归类时抛错，避免静默混用。"""
    s = source or ''
    if '本轮测量' in s or '墨迹' in s:
        return SOURCE_MEASURED
    if '近似框' in s:
        return SOURCE_APPROX
    if 'annotations.json' in s or 'annotation.json' in s or '人工标注' in s:
        return SOURCE_MANUAL
    raise ValueError(f'无法归类的标注来源：{source!r}')


def source_profile(cls: str) -> Dict[str, Any]:
    """三类来源的已知精度与待核事项（说明.md 第三之二节口径）。

    注意：这里只写「框是怎么来的」和「已知缺什么」，不编造像素误差。
    真实字形边界误差三类来源都没有标定测量，一律 unknown。
    """
    common = {
        'glyph_boundary_error': {
            'status': 'unknown_uncalibrated',
            'value_px': None,
            'calibration': 'none',
            'note': '本项目未做过「同一组框 vs 逐字人工边界」的对照测量，'
                    '任何具体像素误差都只是猜测，不能用于误差预算',
        },
    }
    if cls == SOURCE_MANUAL:
        return {
            **common,
            'precision': '人工逐框，双列逐框转录，已对原图与叠框检查过',
            'known_limits': ['端点约定（含/不含右下像素）未在来源文档中声明',
                             '尚无第二人独立复核',
                             '未做字形级边界标定'],
            'upstream': 'output/small_columns_step1_holdout/annotations.json 与 '
                        'output/small_columns_step1_dev/annotation.json',
        }
    if cls == SOURCE_MEASURED:
        return {
            **common,
            'precision': '栏内墨迹外接框，6 倍叠框核对过；字间留白未计入，比人工框略紧',
            'known_limits': ['只覆盖墨迹，不含字距',
                             '边界由 adaptiveThreshold(31,14) 决定，换参数结果会变',
                             '未做字形级边界标定'],
            'upstream': 'tools/build_dual_track_coords.py 中的本轮测量记录（经 '
                        'tools/dual_track_boxes_from_window.py 的 tight() 生成）',
        }
    return {
        **common,
        'precision': '人工近似框，来源文档自述「并非字形级精确分割」',
        'known_limits': ['非字形级精确框', '来源文档已声明近似', '不宜用于精确边界评测',
                         '未做字形级边界标定'],
        'upstream': 'docs/双列小字_步骤3有限覆盖准入与新书页.md 人工近似框',
    }


# ---------------------------------------------------------------- 几何

# 坐标解析约定（写进契约，避免后续各阶段各写一套）：
#   - int            接受
#   - float          必须是有限值且等于其整数部分（如 899.0），归一化为 int；
#                    nan / ±inf / 899.5 一律判非法
#   - str            只接受能被 int(s, 10) 完整解析的十进制整数字符串（允许首尾空白与正负号），
#                    '899.0' / '1e3' / '0x10' 一律判非法
#   - bool           显式拒绝（bool 是 int 子类，不拒绝会被当成 0/1 静默通过）
#   - 其它类型       判非法
# 解析永不抛异常，返回 (bbox|None, code, detail)，便于逐条出失败明细。
BBOX_EMPTY = 'empty'
BBOX_NOT_SEQUENCE = 'not_sequence'
BBOX_LENGTH = 'length'
BBOX_BOOL = 'bool'
BBOX_NON_FINITE = 'non_finite'
BBOX_NON_INTEGRAL = 'non_integral'
BBOX_NOT_NUMERIC = 'not_numeric'
BBOX_OK = 'ok'

BBOX_CODE_TEXT = {
    BBOX_EMPTY: '空值',
    BBOX_NOT_SEQUENCE: '不是数组/元组',
    BBOX_LENGTH: '元素个数不是 4',
    BBOX_BOOL: '含布尔值',
    BBOX_NON_FINITE: '含正负无穷或 NaN',
    BBOX_NON_INTEGRAL: '含非整数数值',
    BBOX_NOT_NUMERIC: '无法解析为整数',
    BBOX_OK: '合法',
}


def _parse_one_number(value: Any) -> Tuple[Optional[int], str, str]:
    if isinstance(value, bool):
        return None, BBOX_BOOL, f'布尔值 {value!r}'
    if isinstance(value, int):
        return value, BBOX_OK, ''
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, BBOX_NON_FINITE, f'非有限数 {value!r}'
        if value != int(value):
            return None, BBOX_NON_INTEGRAL, f'非整数 {value!r}'
        return int(value), BBOX_OK, ''
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10), BBOX_OK, ''
        except ValueError:
            return None, BBOX_NOT_NUMERIC, f'字符串 {value!r} 不是十进制整数'
    return None, BBOX_NOT_NUMERIC, f'类型 {type(value).__name__} 不可解析为整数'


def parse_bbox(value: Any) -> Tuple[Optional[Tuple[int, int, int, int]], str, str]:
    """解析坐标框为四个整数；永不抛异常。返回 (bbox, code, detail)。"""
    if value is None:
        return None, BBOX_EMPTY, '值为 None'
    if isinstance(value, (str, bytes)):
        return None, BBOX_NOT_SEQUENCE, f'整个值不是数组：{value!r}'
    if not isinstance(value, (list, tuple)):
        return None, BBOX_NOT_SEQUENCE, f'类型 {type(value).__name__} 不是数组'
    if len(value) != 4:
        return None, BBOX_LENGTH, f'元素个数 {len(value)}'
    nums: List[int] = []
    for i, item in enumerate(value):
        num, code, detail = _parse_one_number(item)
        if code != BBOX_OK:
            return None, code, f'第 {i} 个元素：{detail}'
        nums.append(num)
    return (nums[0], nums[1], nums[2], nums[3]), BBOX_OK, ''


def as_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    """只取解析结果，非法时返回 None（不抛异常）。需要失败原因时用 parse_bbox。"""
    bbox, code, _ = parse_bbox(value)
    return bbox if code == BBOX_OK else None


def bbox_area(b: Sequence[int]) -> int:
    return max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1]))


def intersect_area(a: Sequence[int], b: Sequence[int]) -> int:
    w = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    h = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1]))
    if w <= 0 or h <= 0:
        return 0
    return w * h


def union_bbox(boxes: Iterable[Sequence[int]]) -> Optional[List[int]]:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [min(int(b[0]) for b in boxes), min(int(b[1]) for b in boxes),
            max(int(b[2]) for b in boxes), max(int(b[3]) for b in boxes)]


def center_x(b: Sequence[int]) -> float:
    return (float(b[0]) + float(b[2])) / 2.0


def in_image(b: Sequence[int], size: Sequence[int]) -> bool:
    return 0 <= int(b[0]) and 0 <= int(b[1]) and int(b[2]) <= int(size[0]) and int(b[3]) <= int(size[1])


# ---------------------------------------------------------------- 统计

def count_by(items: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(items).items()))

def page_status_map(manifest: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {page_id(m['book'], Path(m['page']).stem.split('_')[-1]): m for m in manifest}


def manifest_page_number(record: Dict[str, Any]) -> int:
    return int(str(record['page']).replace('page_', '').replace('.png', ''))


def parse_page_number(raw: Any) -> Optional[int]:
    """'page_33.png' / 33 / '33' -> 33；解析不出返回 None（不抛异常）。"""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw if raw is not None else '').strip()
    if text.lower().endswith('.png'):
        text = text[:-4]
    if text.lower().startswith('page_'):
        text = text[5:]
    try:
        return int(text, 10)
    except (TypeError, ValueError):
        return None


def manifest_key(record: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """清单唯一键：书名 + 页号（页号统一成整数，兼容 'page_33.png' 与 33）。"""
    return (str(record.get('book', '')).strip(), parse_page_number(record.get('page')))


# 端点解释：两种读法，差异是定义性的，恰好让右下边界平移 1px。
# 这是「坐标怎么读」的歧义，不是「框画得准不准」的误差，也不是「评测允许多少偏差」。
ENDPOINT_CANDIDATES = ['half_open:[x1,y1,x2,y2)', 'inclusive:[x1,y1,x2,y2]']
ENDPOINT_EDGE_SHIFT_PX = 1


# ---------------------------------------------------------------- 规范化产物读取
# 后续阶段固定从这三个入口读，不要回头读原始 双轨坐标.json。

NORMALIZED = OUT_DIR / 'normalized_annotations.json'
CORRECTIONS = OUT_DIR / 'corrections.json'


def load_normalized(path: Optional[Path] = None) -> Dict[str, Any]:
    return read_json(Path(path) if path else NORMALIZED)


def load_groups(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return load_normalized(path)['groups']


def load_page_records(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return load_normalized(path)['pages']


def load_corrections(path: Optional[Path] = None) -> Dict[str, Any]:
    return read_json(Path(path) if path else CORRECTIONS)


def groups_for_page(book: str, page: int, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    pid = page_id(book, page)
    return [g for g in load_groups(path) if g['page_id'] == pid]
