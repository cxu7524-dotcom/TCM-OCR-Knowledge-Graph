"""Scan every page of 医家类 for geometric dual-track small-character candidates.

Uses the existing conservative proposer (`ocr.small_columns.propose_page`) on the
automatic column layout. No OCR, no ground truth. Output is a candidate list only;
each candidate must be checked against the page image before use.
"""
import json, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.vertical import imread_unicode, page_layout
from ocr.small_columns import propose_page

BOOKS = ROOT / '古籍评测集/医家类'
OUT = ROOT / 'output/双轨小字页扫描'
OUT.mkdir(parents=True, exist_ok=True)
CROPS = OUT / 'candidates'
CROPS.mkdir(exist_ok=True)

MIN_LANE_H = 14       # a lane must span at least ~1 small glyph
MIN_LANE_W = 7
MIN_MAIN_H = 18


def keep(q):
    lanes = sorted(q['columns'], key=lambda c: c['bbox'][0])
    h = [c['bbox'][3] - c['bbox'][1] for c in lanes]
    w = [c['bbox'][2] - c['bbox'][0] for c in lanes]
    mh = q['main_bbox'][3] - q['main_bbox'][1]
    if min(h) < MIN_LANE_H or min(w) < MIN_LANE_W or mh < MIN_MAIN_H:
        return None
    return {'short_side_h': min(h), 'long_side_h': max(h), 'lane_w': w, 'main_h': mh}


def main():
    pages = sorted((b, p) for b in BOOKS.iterdir() if b.is_dir() for p in b.glob('page_*.png'))
    scan = {'method': 'ocr.small_columns.propose_page on ocr.vertical.page_layout columns; geometric candidates only',
            'filters': {'MIN_LANE_H': MIN_LANE_H, 'MIN_LANE_W': MIN_LANE_W, 'MIN_MAIN_H': MIN_MAIN_H},
            'books': [b.name for b in BOOKS.iterdir() if b.is_dir()], 'pages': {}}
    t0 = time.time()
    for book, path in pages:
        key = f'{book.name}/{path.stem}'
        lay = page_layout(str(path))
        im = imread_unicode(str(path))
        raw = propose_page(im, lay['cols'])
        cands = []
        for i, q in enumerate(raw):
            m = keep(q)
            if not m:
                continue
            x1 = min(c['bbox'][0] for c in q['columns']) - 6
            x2 = max(c['bbox'][2] for c in q['columns']) + 6
            y1 = min(q['main_bbox'][1], min(c['bbox'][1] for c in q['columns'])) - 8
            y2 = max(c['bbox'][3] for c in q['columns']) + 8
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
            crop = im[y1:y2, x1:x2]
            name = ''
            if crop.size:
                name = f'candidates/{book.name}_{path.stem}_{i:02d}.png'
                up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                (OUT / name).write_bytes(cv2.imencode('.png', up)[1].tobytes())
            cands.append({'index': i, 'parent_bbox': q['parent_bbox'], 'main_bbox': q['main_bbox'],
                          'columns': {c['side']: c['bbox'] for c in q['columns']},
                          'crop_bbox': [x1, y1, x2, y2], 'crop': name,
                          'windows': q['evidence'].get('windows'),
                          'min_valley_ratio': q['evidence'].get('min_valley_ratio'), **m})
        scan['pages'][key] = {'source': str(path.relative_to(ROOT)).replace('\\', '/'),
                              'image_size': [im.shape[1], im.shape[0]],
                              'auto_columns': len(lay['cols']),
                              'raw_proposals': len(raw), 'candidates': cands}
        print(key, 'raw', len(raw), 'kept', len(cands), flush=True)
    scan['elapsed_seconds'] = round(time.time() - t0, 1)
    (OUT / 'scan.json').write_text(json.dumps(scan, ensure_ascii=False, indent=1), encoding='utf8')
    hit = {k: len(v['candidates']) for k, v in scan['pages'].items() if v['candidates']}
    (OUT / 'summary.json').write_text(json.dumps({'pages_total': len(scan['pages']), 'pages_with_candidates': len(hit),
                                                  'per_page': hit, 'elapsed_seconds': scan['elapsed_seconds']},
                                                 ensure_ascii=False, indent=1), encoding='utf8')
    print('pages', len(scan['pages']), 'with candidates', len(hit), 'elapsed', scan['elapsed_seconds'])


if __name__ == '__main__':
    main()
