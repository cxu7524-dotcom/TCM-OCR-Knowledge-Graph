"""Assemble per-region coordinates of the dual-track examples in the curated pages.

Sources, in order of authority:
  1. existing manual annotations (holdout 20 positives + dev page 35)
  2. the W12 manual boxes recorded in docs/双列小字_步骤3有限覆盖准入与新书页.md
  3. this round's measurement for the three pages without annotations (p33/p36/p42),
     boxes taken from ink extents inside the detected column, checked on 6x overlays
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / '古籍评测集/双轨小字页'
BOOKS = ROOT / '古籍评测集/医家类'
OVER = DST / '坐标图'
OVER.mkdir(parents=True, exist_ok=True)

PAGES = {('思濟堂方書', 33), ('思濟堂方書', 34), ('思濟堂方書', 35), ('思濟堂方書', 36),
         ('思濟堂方書', 37), ('思濟堂方書', 39), ('思濟堂方書', 42), ('思濟堂方書', 43),
         ('思濟堂方書', 45), ('思濟堂方書', 46), ('思濟堂方書', 49), ('思濟堂方書', 50),
         ('温疫方論', 12)}

MANUAL = [
    {'book': '思濟堂方書', 'page': 33, 'region': 'S33-01', 'main_text': '川大黃', 'right_text': '二錢半',
     'left_text': '酒洗', 'main_bbox': [899, 236, 933, 316], 'right_bbox': [917, 317, 933, 398],
     'left_bbox': [899, 317, 914, 375], 'source': '本轮测量（栏内墨迹外接框，6倍叠框核对）', 'column': 3},
    {'book': '思濟堂方書', 'page': 36, 'region': 'S36-01', 'main_text': '正嫩烏藥', 'right_text': '二錢',
     'left_text': '打碎', 'main_bbox': [189, 245, 220, 362], 'right_bbox': [205, 372, 221, 414],
     'left_bbox': [187, 362, 203, 414], 'source': '本轮测量（栏内墨迹外接框，6倍叠框核对）', 'column': 16},
    {'book': '思濟堂方書', 'page': 42, 'region': 'S42-01', 'main_text': '麥冬', 'right_text': '各三錢',
     'left_text': '去心', 'main_bbox': [950, 408, 979, 466], 'right_bbox': [966, 466, 981, 543],
     'left_bbox': [950, 466, 963, 514], 'source': '本轮测量（栏内墨迹外接框，6倍叠框核对）', 'column': 2},
    # W12: boxes as recorded in docs/双列小字_步骤3有限覆盖准入与新书页.md (image pixels, top-left origin)
    {'book': '温疫方論', 'page': 12, 'region': 'W12-01', 'main_text': '澤瀉', 'right_text': '一錢',
     'left_text': '五分', 'main_bbox': [250, 353, 285, 420], 'right_bbox': [265, 418, 287, 478],
     'left_bbox': [244, 418, 265, 478], 'source': 'docs/双列小字_步骤3有限覆盖准入与新书页.md 人工近似框',
     'column': None},
    {'book': '温疫方論', 'page': 12, 'region': 'W12-02', 'main_text': '豬苓', 'right_text': '一錢',
     'left_text': '五分', 'main_bbox': [250, 478, 285, 531], 'right_bbox': [265, 529, 287, 590],
     'left_bbox': [244, 529, 265, 589], 'source': 'docs/双列小字_步骤3有限覆盖准入与新书页.md 人工近似框',
     'column': None},
]


def from_annotations():
    out = []
    hold = json.loads((ROOT / 'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'))
    for s in hold['samples']:
        if s['label'] != 'positive':
            continue
        page = int(s['page'])
        if ('思濟堂方書', page) not in PAGES:
            continue
        cols = {c['side']: c for c in s['columns']}
        out.append({'book': '思濟堂方書', 'page': page, 'region': s['id'], 'main_text': s.get('main_text', ''),
                    'right_text': cols.get('right', {}).get('text', ''),
                    'left_text': cols.get('left', {}).get('text', ''),
                    'main_bbox': s['main_bbox'], 'right_bbox': cols['right']['bbox'],
                    'left_bbox': cols['left']['bbox'],
                    'source': 'output/small_columns_step1_holdout/annotations.json（人工标注）', 'column': None})
    dev = json.loads((ROOT / 'output/small_columns_step1_dev/annotation.json').read_text(encoding='utf8'))['sample']
    cols = {c['side']: c for c in dev['columns']}
    out.append({'book': '思濟堂方書', 'page': 35, 'region': dev['id'], 'main_text': dev.get('main_text', ''),
                'right_text': cols['right']['text'], 'left_text': cols['left']['text'],
                'main_bbox': dev['main_bbox'], 'right_bbox': cols['right']['bbox'], 'left_bbox': cols['left']['bbox'],
                'source': 'output/small_columns_step1_dev/annotation.json（人工标注）', 'column': None})
    return out


def draw_page(book, page, regions):
    path = BOOKS / book / f'page_{page}.png'
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    vis = im.copy()
    for r in regions:
        for key, color in (('main_bbox', (0, 210, 255)), ('right_bbox', (0, 0, 255)), ('left_bbox', (0, 150, 0))):
            b = r[key]
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), color, 1)
        b = r['main_bbox']
        cv2.putText(vis, r['region'], (b[0], max(10, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    out = OVER / f"{book}_{page}_双轨框.png"
    out.write_bytes(cv2.imencode('.png', vis)[1].tobytes())
    return out


def main():
    rows = from_annotations() + MANUAL
    rows.sort(key=lambda r: (r['book'] != '思濟堂方書', r['page'], r['region']))
    for r in rows:
        r['image_size'] = [1024, 804]
        r['coordinate_system'] = '原扫描图像素，左上角原点，x右y下'
    with open(DST / '双轨坐标.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['book', 'page', 'region', 'main_text', 'right_text', 'left_text',
                                          'main_bbox', 'right_bbox', 'left_bbox', 'source', 'column'])
        w.writeheader()
        for r in rows:
            row = dict(r)
            for k in ('main_bbox', 'right_bbox', 'left_bbox'):
                row[k] = json.dumps(row[k], ensure_ascii=False)
            w.writerow({k: row.get(k, '') for k in w.fieldnames})
    (DST / '双轨坐标.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf8')
    # per-page overlay images
    by_page = {}
    for r in rows:
        by_page.setdefault((r['book'], r['page']), []).append(r)
    for (book, page), regs in sorted(by_page.items()):
        print(draw_page(book, page, regs))
    print('regions', len(rows), 'pages', len(by_page))


if __name__ == '__main__':
    main()
