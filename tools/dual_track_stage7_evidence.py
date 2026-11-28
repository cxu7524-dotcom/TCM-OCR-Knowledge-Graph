# -*- coding: utf-8 -*-
"""stage7 证据图：p12 原图叠框（逐侧状态）、两组放大、旋转裁图拼版。

输出 output/dual_tracks/stage7/evidence/
  p12_overlay.jpg          整页：右轨橙 / 左轨青 / 待核红虚线 / 标注绿
  p12_df12_03_zoom.jpg     雲母（右轨冲突待核）
  p12_df12_11_zoom.jpg     陽起石（双侧合格，各自复识）
  crops_panel.jpg          四张旋转裁图 + 读数与置信度
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ocr.vertical import imread_unicode  # noqa: E402

S7 = ROOT / 'output' / 'dual_tracks' / 'stage7'
E = S7 / 'evidence'
COL = {'right': (0, 120, 255), 'left': (210, 200, 0), 'pending': (0, 0, 230),
       'gt': (0, 170, 0), 'text': (30, 30, 30)}


def _write(path: Path, img: Any, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        path.write_bytes(buf.tobytes())


def _imread(path: Path) -> Any:
    """项目路径含中文：cv2.imread 在 Windows 上读不了，改用 fromfile+imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, ValueError):
        return None


def _rect(canvas, b, color, th=1, dash=False, off=(0, 0)) -> None:
    if not b:
        return
    x0, y0 = max(0, int(b[0]) - off[0]), max(0, int(b[1]) - off[1])
    x1 = min(canvas.shape[1] - 1, int(b[2]) - off[0])
    y1 = min(canvas.shape[0] - 1, int(b[3]) - off[1])
    if x1 <= x0 or y1 <= y0:
        return
    if dash:
        for k in range(x0, x1, 6):
            cv2.line(canvas, (k, y0), (min(k + 3, x1), y0), color, th)
            cv2.line(canvas, (k, y1), (min(k + 3, x1), y1), color, th)
        for k in range(y0, y1, 6):
            cv2.line(canvas, (x0, k), (x0, min(k + 3, y1)), color, th)
            cv2.line(canvas, (x1, k), (x1, min(k + 3, y1)), color, th)
    else:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, th)


def _label(canvas, text: str, xy: Tuple[int, int], color, scale=0.34, th=1) -> None:
    x, y = xy
    cv2.putText(canvas, text, (max(0, x), max(10, y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (max(0, x), max(10, y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, th, cv2.LINE_AA)


def overlay_page(ocr: Dict[str, Any], image_path: str, annot: Dict[str, Any]) -> Path:
    img = imread_unicode(image_path)
    canvas = cv2.resize(img, None, fx=1.0, fy=1.0)
    ev = ocr.get('dual_track_evidence') or []
    for e in ev:
        side = e['side']
        color = COL['pending'] if e['side_status'] != 'ok' else COL[side]
        _rect(canvas, e.get('bbox'), color, 1, dash=e['side_status'] != 'ok')
        if e['side_status'] == 'ok':
            _label(canvas, e['text'][:6] if e['text'] else '（空）',
                   (e['bbox'][0] - 6, e['bbox'][1] - 3), color, 0.28)
    for g in annot.get('groups') or []:
        _rect(canvas, g['group_bbox'], COL['gt'], 1)
        _label(canvas, str(g.get('main_text') or '')[:4],
               (g['group_bbox'][0], g['group_bbox'][3] + 10), COL['gt'], 0.3)
    return canvas


def zoom_group(ocr: Dict[str, Any], image_path: str, group_id: str, pad: int = 26,
               zoom: float = 3.0) -> Optional[Any]:
    img = imread_unicode(image_path)
    ev = [e for e in (ocr.get('dual_track_evidence') or []) if e['group_id'] == group_id]
    if not ev:
        return None
    xs = [e['bbox'] for e in ev if e.get('bbox')]
    x0 = min(b[0] for b in xs) - pad
    y0 = min(b[1] for b in xs) - pad
    x1 = max(b[2] for b in xs) + pad
    y1 = max(b[3] for b in xs) + pad
    x0, y0 = max(0, x0), max(0, y0)
    crop = img[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    for e in ev:
        b = [int(v) for v in e['bbox']]
        bb = [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]
        bb = [int(v * zoom) for v in bb]
        ok = e['side_status'] == 'ok'
        color = COL[e['side']] if ok else COL['pending']
        _rect(big, bb, color, 2, dash=not ok)
        tag = (f"{e['side']} ok conf={e['conf']} {e['text']}" if ok
               else f"{e['side']} 待核 {e['status_reason']}")
        _label(big, tag, (bb[0] + 2, bb[1] - 6 if bb[1] > 18 else bb[3] + 14), color, 0.45, 1)
    return big


def crops_panel(ocr: Dict[str, Any], group_ids: List[str]) -> Any:
    """四张旋转裁图的拼版：每格下方写 组 / 侧 / 状态 / 读数 / 置信度。"""
    tiles = []
    for gid in group_ids:
        for e in ocr.get('dual_track_evidence') or []:
            if e['group_id'] != gid or not e.get('crop_path'):
                continue
            p = ROOT / e['crop_path']
            if not p.exists():
                continue
            im = _imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            scale = 300 / max(1, w)
            im = cv2.resize(im, (300, max(24, int(h * scale))), interpolation=cv2.INTER_CUBIC)
            pad = np.full((40, 300, 3), 255, np.uint8)
            canvas = np.vstack([im, pad])
            line1 = f"{gid.split('#')[-1]} {e['side']} {e['side_status']} conf={e['conf']}"
            line2 = (e['text'] or '（空读数）')[:14]
            _label(canvas, line1, (4, im.shape[0] + 14), COL['text'], 0.36, 1)
            _label(canvas, line2, (4, im.shape[0] + 32), COL['text'], 0.42, 1)
            tiles.append(canvas)
    if not tiles:
        return None
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((hmax - t.shape[0], t.shape[1], 3), 255, np.uint8)])
             for t in tiles]
    return np.hstack(tiles)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='stage7 证据图')
    ap.add_argument('--run', default=str(S7 / 'run' / '丹方彚编__page_12_on.json'))
    ap.add_argument('--out-dir', default=str(E))
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.run).read_text(encoding='utf-8'))
    ocr = payload['ocr']
    image_path = str(ROOT / ocr['image_path']) if not Path(ocr['image_path']).is_absolute() \
        else ocr['image_path']
    annot = json.loads((ROOT / 'output' / 'dual_tracks' / 'stage2' /
                        'annotations.json').read_text(encoding='utf-8'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write(out / 'p12_overlay.jpg', overlay_page(ocr, image_path, annot))
    z1 = zoom_group(ocr, image_path, '丹方彚编/page_12#g006')
    if z1 is not None:
        _write(out / 'p12_df12_03_zoom.jpg', z1)
    z2 = zoom_group(ocr, image_path, '丹方彚编/page_12#g036')
    if z2 is not None:
        _write(out / 'p12_df12_11_zoom.jpg', z2)
    panel = crops_panel(ocr, ['丹方彚编/page_12#g006', '丹方彚编/page_12#g036'])
    if panel is not None:
        _write(out / 'crops_panel.jpg', panel)
    print('evidence ->', out)
    for p in sorted(out.glob('*.jpg')):
        print(' ', p.name, p.stat().st_size, 'bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
