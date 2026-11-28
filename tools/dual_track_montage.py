"""Build labelled contact sheets of whole pages for visual dual-track screening."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / '古籍评测集/医家类'
OUT = ROOT / 'output/双轨小字页检测/审阅'
OUT.mkdir(parents=True, exist_ok=True)


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def page_paths(book, stems):
    return [BOOKS / book / f'page_{s}.png' for s in stems]


def montage(paths, out, width=560, gap=6, head=24):
    ims = [rd(p) for p in paths]
    hs = max(int(im.shape[0] * width / im.shape[1]) for im in ims)
    canvas = np.full((hs + head, len(ims) * (width + gap), 3), 245, np.uint8)
    for i, im in enumerate(ims):
        im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
        x = i * (width + gap)
        canvas[head:head + im.shape[0], x:x + im.shape[1]] = im
        label = f"{paths[i].parent.name}_{paths[i].stem}"
        cv2.putText(canvas, label, (x + 3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    Path(out).write_bytes(cv2.imencode('.png', canvas)[1].tobytes())
    return out


if __name__ == '__main__':
    book = sys.argv[1]
    stems = sys.argv[2].split(',')
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 560
    out = OUT / f'{book}_{stems[0]}-{stems[-1]}.png'
    print(montage(page_paths(book, stems), out, width))
