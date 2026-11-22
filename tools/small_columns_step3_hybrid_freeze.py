import hashlib, importlib.metadata, json, shutil, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr.recognize import resolve_rec_model

O = ROOT / 'output/small_columns_step3_hybrid'
target = O / 'freeze_v4.json'
assert not target.exists(), 'Refusing to replace frozen manifest'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old = json.loads((ROOT/'output/small_columns_step3_main/freeze_v3.json').read_text(encoding='utf8'))
assert sha(ROOT/'ocr/small_columns.py') == old['algorithm_sha256']
assert sha(ROOT/'output/small_columns_step1_holdout/annotations.json') == old['holdout_label_sha256']
evidence = json.loads((O/'development_evidence.json').read_text(encoding='utf8'))
assert len(evidence['samples']) == 11
model_path, model_name, model_source = resolve_rec_model()
assert model_path and 'PP-OCRv5' in model_name
files = ['ocr/small_columns.py', 'ocr/small_columns_hybrid.py', 'ocr/recognize.py', 'ocr/pipeline.py', 'ocr/vertical.py', 'app/config.py', 'output/small_columns_step1_holdout/annotations.json', 'output/small_columns_step3_hybrid/development_evidence.json', 'output/small_columns_step3_hybrid/development_full_comparison.json', 'output/small_columns_step3_hybrid/development_full_ocr.json']
shutil.copyfile(ROOT/'ocr/small_columns_hybrid.py', O/'frozen_hybrid.py')
manifest = {'frozen_at': datetime.now().astimezone().isoformat(), 'prior_holdout_runs': 3, 'independent_blind_holdout': False, 'files_sha256': {p: sha(ROOT/p) for p in files}, 'model': {'path': model_path, 'name': model_name, 'source': model_source, 'sha256': sha(model_path), 'rapidocr_onnxruntime_version': importlib.metadata.version('rapidocr-onnxruntime')}, 'development_pages': [31,32,33,35,36,38,40,42,48], 'inference_inputs': 'full page image, automatic layout columns, normal full-page OCR blocks and local crop recognition; no annotation boxes'}
target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
