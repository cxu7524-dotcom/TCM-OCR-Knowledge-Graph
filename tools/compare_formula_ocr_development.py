"""Single-variable model comparison on annotated development-page title crops.

33/35 are development; 32/36 are held out. Same source bbox and rotation for
all models. This probes recognizer substitutions only, not full-page detection.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.normalize import normalize_text
from ocr.recognize import _engine
from ocr.vertical import imread_unicode

ann = json.loads((ROOT / 'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
pages = [p for p in ann['pages'] if p.get('split') == '开发' and
         p.get('book') == '思濟堂方書' and p['page_no'] in (32, 33, 35, 36)]
assert len(pages) == 4
models = {'current_v5': str(ROOT / '模型权重/ch_PP-OCRv5_rec_mobile.onnx'),
          'traditional_v3': str(ROOT / '模型权重/chinese_cht_PP-OCRv3_rec_mobile.onnx'),
          'bundled_v3': ''}

def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]

out = {'scope': '步骤1开发页题名标注框；仅比较同框识别，不代表端到端结果',
       'tuning_pages': [33, 35], 'heldout_pages': [32, 36], 'samples': []}
for p in pages:
    image = imread_unicode(str(ROOT / '古籍评测集/医家类/思濟堂方書' / f"page_{p['page_no']}.png"))
    for instance in p['instances']:
        if not instance.get('name_verbatim') or not instance.get('name_bbox'):
            continue
        x1, y1, x2, y2 = instance['name_bbox']
        crop = image[max(0,y1-5):min(image.shape[0],y2+5), max(0,x1-5):min(image.shape[1],x2+5)]
        rotated = np.ascontiguousarray(np.rot90(crop))
        expected = normalize_text(instance['name_verbatim'])
        row = {'page': p['page_no'], 'split': 'heldout' if p['page_no'] in (32,36) else 'tuning',
               'bbox': instance['name_bbox'], 'expected': instance['name_verbatim'], 'models': {}}
        for name, path in models.items():
            engine = _engine(path)
            recognizer = getattr(engine, 'text_recognizer', None) or engine.text_rec
            result, _ = recognizer([rotated])
            text, confidence = result[0]
            row['models'][name] = {'text': text, 'confidence': float(confidence),
                                   'edit_distance': distance(normalize_text(text), expected)}
        out['samples'].append(row)
        print('page', p['page_no'], {k:v['edit_distance'] for k,v in row['models'].items()}, flush=True)
for split in ('tuning', 'heldout'):
    out[split + '_distance'] = {name: sum(s['models'][name]['edit_distance'] for s in out['samples'] if s['split'] == split) for name in models}
out['decision'] = '仅凭题名裁剪对照不自动替换整页识别；还须独立页端到端验证。'
dest = ROOT / 'output/formula_stage6/开发页OCR单变量对照.json'
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
print('saved', dest, flush=True)
