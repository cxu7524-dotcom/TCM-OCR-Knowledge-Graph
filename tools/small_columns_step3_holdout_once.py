import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode,page_layout
from ocr.small_columns import propose_page
out=ROOT/'output/small_columns_step3';dst=out/'holdout_first_predictions.json'
if dst.exists():raise SystemExit('First holdout predictions already exist: refusing to overwrite.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((out/'freeze.json').read_text(encoding='utf8'));label=ROOT/'output/small_columns_step1_holdout/annotations.json'
assert sha(ROOT/'ocr/small_columns.py')==f['algorithm_sha256'];assert sha(label)==f['holdout_annotation_sha256']
a=json.loads(label.read_text(encoding='utf8'));result={'freeze_sha256':sha(out/'freeze.json'),'algorithm_sha256':f['algorithm_sha256'],'label_sha256':sha(label),'run_number':1,'input':'automatic page_layout columns only; no annotated crop used for proposals','pages':{}}
for n,meta in a['pages'].items():
 p=ROOT/meta['path'];assert sha(p)==meta['sha256'];im=imread_unicode(str(p));layout=page_layout(str(p));pred=propose_page(im,layout['cols']);result['pages'][n]={'source':meta['path'],'image_sha256':meta['sha256'],'automatic_columns':layout['cols'],'predictions':pred}
 print(n,'columns',len(layout['cols']),'proposals',len(pred),flush=True)
dst.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
