"""Build labelled contact sheets of dual-track candidates for visual review."""
import json, sys, math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / 'output/双轨小字页扫描'
scan = json.loads((SCAN / 'scan.json').read_text(encoding='utf8'))

def sheet(keys, out, per_row=6, cell_h=210, cell_w=170):
    items = [(k, c) for k in keys for c in scan['pages'][k]['candidates']]
    rows = math.ceil(len(items) / per_row)
    canvas = np.full((rows * cell_h, per_row * cell_w, 3), 255, np.uint8)
    for i, (k, c) in enumerate(items):
        fp = SCAN / c['crop']
        im = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR) if fp.exists() else None
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min(cell_w / w, (cell_h - 24) / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        r, col = divmod(i, per_row)
        y0, x0 = r * cell_h + 22, col * cell_w
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.putText(canvas, f"{k.split('/')[0][:3]}{k.split('/')[1]}-{c['index']}", (x0 + 2, r * cell_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    Path(out).write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    print(out, len(items))

if __name__ == '__main__':
    keys = sys.argv[1].split(',')
    sheet(keys, ROOT / f'output/双轨小字页扫描/sheet_{sys.argv[2]}.png')
