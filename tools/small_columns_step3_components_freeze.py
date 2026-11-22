import hashlib,importlib.metadata,json,shutil,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.recognize import resolve_rec_model
O=ROOT/'output/small_columns_step3_detector';target=O/'freeze_v5.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v4=ROOT/'output/small_columns_step3_hybrid/freeze_v4.json';f=json.loads(v4.read_text(encoding='utf8'))
for p,h in f['files_sha256'].items():assert sha(ROOT/p)==h,p
prior=ROOT/'output/small_columns_step3_hybrid/holdout_fourth_predictions.json';a=json.loads(prior.read_text(encoding='utf8'));assert a['freeze_sha256']==sha(v4)
dev=json.loads((O/'development_comparison.json').read_text(encoding='utf8'))
positive=[r for r in dev['samples'] if r['sample']['label']=='positive'];negative=[r for r in dev['samples'] if r['sample']['label']=='negative']
assert len(positive)==7 and len(negative)==4
for r in positive:
 assert r['combined']['prediction'] is not None
 assert all(r['combined']['side_ink_coverage'][s]>=r['geometry']['side_ink_coverage'][s] for s in ('right','left'))
assert all(r['combined']['candidate_count']==0 for r in negative)
s36=next(r for r in positive if r['sample']['id']=='S36');assert s36['combined']['side_ink_coverage']['right']>s36['geometry']['side_ink_coverage']['right']
for n,x in [(31,751),(35,654)]:assert any(r['page']==n and r['parent_bbox'][0]==x and r['combined'] is None for r in dev['automatic_candidate_evidence'])
model,name,source=resolve_rec_model();assert sha(model)==f['model']['sha256']
files=['ocr/small_columns_components.py','ocr/small_columns.py','ocr/small_columns_hybrid.py','ocr/recognize.py','ocr/vertical.py','app/config.py','output/small_columns_step1_holdout/annotations.json','output/small_columns_step3_detector/development_detector_raw.json','output/small_columns_step3_detector/development_comparison.json','output/small_columns_step3_hybrid/holdout_fourth_predictions.json','output/small_columns_step3_hybrid/freeze_v4.json']
shutil.copyfile(ROOT/'ocr/small_columns_components.py',O/'frozen_components.py')
data={'frozen_at':datetime.now().astimezone().isoformat(),'previous_holdout_runs':4,'independent_blind_holdout':False,'files_sha256':{p:sha(ROOT/p) for p in files},'model':{'path':model,'name':name,'sha256':sha(model),'rapidocr_onnxruntime_version':importlib.metadata.version('rapidocr-onnxruntime')},'candidate':'v3 geometry with component crossing evidence and v4 nonlocal main guard; local recognition status is diagnostic only; detector boxes retained in development but not selected','holdout_inputs':'full page and previously saved full-page automatic columns and normal OCR blocks from frozen fourth run'}
target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print('frozen',sha(target))
