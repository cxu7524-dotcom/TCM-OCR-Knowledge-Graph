import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
from ocr.recognize import resolve_rec_model
from ocr.small_columns_components import propose_components_page
O=ROOT/'output/small_columns_step3_detector';target=O/'holdout_fifth_predictions.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((O/'freeze_v5.json').read_text(encoding='utf8'))
for p,h in freeze['files_sha256'].items():assert sha(ROOT/p)==h,p
assert sha(resolve_rec_model()[0])==freeze['model']['sha256']
fourth=json.loads((ROOT/'output/small_columns_step3_hybrid/holdout_fourth_predictions.json').read_text(encoding='utf8'))
pages={}
for n,old in fourth['pages'].items():
 path=ROOT/old['source'];assert sha(path)==old['image_sha256'];assert old['normal_ocr']['model_loaded'] and not old['normal_ocr']['is_demo']
 result=propose_components_page(imread_unicode(str(path)),old['automatic_columns'],old['normal_ocr']['blocks'])
 pages[n]={'source':old['source'],'image_sha256':old['image_sha256'],'normal_ocr':old['normal_ocr'],'automatic_columns':old['automatic_columns'],'geometry_predictions':old['geometry_predictions'],'predictions':result['predictions'],'audit':result['audit']}
 print(n,'geometric',len(old['geometry_predictions']),'components',len(result['predictions']),flush=True)
target.write_text(json.dumps({'run_number':5,'independent_blind_holdout':False,'freeze_sha256':sha(O/'freeze_v5.json'),'uses_prior_automatic_ocr_and_columns':True,'pages':pages},ensure_ascii=False,indent=2),encoding='utf8')
