import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
from ocr.recognize import resolve_rec_model
from ocr.vertical import imread_unicode, page_layout
from ocr.small_columns import propose_page
from ocr.small_columns_hybrid import propose_hybrid_page

O = ROOT/'output/small_columns_step3_hybrid'
target = O/'holdout_fourth_predictions.json'
assert not target.exists(), 'Refusing to replace the fourth run'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
f = json.loads((O/'freeze_v4.json').read_text(encoding='utf8'))
for path, digest in f['files_sha256'].items(): assert sha(ROOT/path) == digest, path
model_path, _, _ = resolve_rec_model()
assert sha(model_path) == f['model']['sha256']
labels = json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'))
object.__setattr__(CONFIG.paths, 'reports_dir', O/'holdout_reports')
pages = {}
for n, source in labels['pages'].items():
    path = ROOT/source['path']
    assert sha(path) == source['sha256']
    r = run_ocr(str(path), page_id=f'HYBRID-HOLDOUT-{n}')
    assert r.model_loaded and not r.is_demo, f'Normal OCR unavailable on page {n}'
    image = imread_unicode(str(path))
    cols = page_layout(str(path))['cols']
    geometric = propose_page(image, cols)
    hybrid = propose_hybrid_page(image, cols, r.blocks)
    pages[n] = {'source': source['path'], 'image_sha256': source['sha256'], 'normal_ocr': r.to_dict(), 'automatic_columns': cols, 'geometry_predictions': geometric, 'predictions': hybrid['predictions'], 'audit': hybrid['audit']}
    print(n, 'normal OCR blocks', len(r.blocks), 'geometry', len(geometric), 'hybrid', len(hybrid['predictions']), flush=True)
target.write_text(json.dumps({'run_number': 4, 'independent_blind_holdout': False, 'freeze_sha256': sha(O/'freeze_v4.json'), 'pages': pages}, ensure_ascii=False, indent=2), encoding='utf8')
