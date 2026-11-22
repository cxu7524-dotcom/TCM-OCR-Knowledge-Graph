import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode,page_layout;from ocr.small_columns import propose_page
O=ROOT/'output/small_columns_step3_main';dest=O/'holdout_third_predictions.json'
if dest.exists():raise SystemExit('Third evaluation exists; refusing overwrite')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();f=json.loads((O/'freeze_v3.json').read_text(encoding='utf8'));label=ROOT/'output/small_columns_step1_holdout/annotations.json'
assert sha(ROOT/'ocr/small_columns.py')==f['algorithm_sha256'];assert sha(label)==f['holdout_label_sha256'];a=json.loads(label.read_text(encoding='utf8'));out={'run_number':3,'independent_blind_holdout':False,'freeze_sha256':sha(O/'freeze_v3.json'),'algorithm_sha256':f['algorithm_sha256'],'label_sha256':sha(label),'input':'automatic full-page column detector and pixels only','pages':{}}
for n,meta in a['pages'].items():
 p=ROOT/meta['path'];assert sha(p)==meta['sha256'];image=imread_unicode(str(p));columns=page_layout(str(p))['cols'];pred=propose_page(image,columns);out['pages'][n]={'source':meta['path'],'image_sha256':meta['sha256'],'automatic_columns':columns,'predictions':pred};print(n,'auto_cols',len(columns),'proposals',len(pred),flush=True)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
