"""One-shot full-page frozen-v5 run on the independent 温疫方論 page."""
import hashlib,importlib.metadata,json,shutil,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
from ocr.recognize import resolve_rec_model
from ocr.vertical import imread_unicode,page_layout
from ocr.small_columns import propose_page
from ocr.small_columns_components import propose_components_page

O=ROOT/'output/small_columns_step3_newbook';O.mkdir(exist_ok=True)
target=O/'wenyi_page12_v5_full_predictions.json';manifest=O/'pre_inference_manifest.json'
assert not target.exists() and not manifest.exists(), 'one-shot evidence already exists'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=ROOT/'output/small_columns_step3_detector/freeze_v5.json';f=json.loads(freeze.read_text(encoding='utf8'))
for path,digest in f['files_sha256'].items():assert sha(ROOT/path)==digest,path
v4=json.loads((ROOT/'output/small_columns_step3_hybrid/freeze_v4.json').read_text(encoding='utf8'))
for path in ('ocr/pipeline.py','ocr/recognize.py','ocr/vertical.py','app/config.py'):assert sha(ROOT/path)==v4['files_sha256'][path],path
model,name,source=resolve_rec_model();assert sha(model)==f['model']['sha256']
image_path=ROOT/'古籍评测集/医家类/温疫方論/page_12.png';image_hash=sha(image_path)
labels=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));assert image_hash not in {v['sha256'] for v in labels['pages'].values()}
shutil.copyfile(ROOT/'ocr/small_columns_components.py',O/'frozen_v5_components.py')
manifest.write_text(json.dumps({'created_at':datetime.now().astimezone().isoformat(),'source':str(image_path.relative_to(ROOT)),'source_sha256':image_hash,'v5_freeze_sha256':sha(freeze),'v5_algorithm_sha256':sha(ROOT/'ocr/small_columns_components.py'),'frozen_snapshot_sha256':sha(O/'frozen_v5_components.py'),'model_name':name,'model_sha256':sha(model),'rapidocr_version':importlib.metadata.version('rapidocr-onnxruntime'),'v4_frozen_pipeline_sha256':sha(ROOT/'ocr/pipeline.py'),'inputs':'entire original image, automatic columns, actual normal full-page OCR blocks; no manual label coordinates'},ensure_ascii=False,indent=2),encoding='utf8')
object.__setattr__(CONFIG.paths,'reports_dir',O/'ocr_reports')
r=run_ocr(str(image_path),page_id='NEWBOOK-WENYI-P12');assert r.model_loaded and not r.is_demo,'normal full-page OCR unavailable'
im=imread_unicode(str(image_path));layout=page_layout(str(image_path));cols=layout['cols'];geom=propose_page(im,cols);out=propose_components_page(im,cols,r.blocks)
result={'source':str(image_path.relative_to(ROOT)),'source_sha256':image_hash,'pre_inference_manifest_sha256':sha(manifest),'normal_ocr':r.to_dict(),'automatic_columns':cols,'geometry_predictions':geom,'predictions':out['predictions'],'audit':out['audit']}
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('image_sha256',image_hash,'ocr blocks',len(r.blocks),'auto columns',len(cols),'geometry',len(geom),'v5',len(out['predictions']),'saved_sha256',sha(target),flush=True)
