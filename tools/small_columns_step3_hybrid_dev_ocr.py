import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
O=ROOT/'output/small_columns_step3_hybrid';O.mkdir(exist_ok=True)
object.__setattr__(CONFIG.paths,'reports_dir',O/'dev_reports')
label=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));forbidden={x['sha256'] for x in label['pages'].values()}
result={}
for n in (31,32,33,35,36,38,40,42,48):
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h not in forbidden
 r=run_ocr(str(p),page_id=f'HYBRID-DEV-{n}');result[str(n)]=r.to_dict();result[str(n)]['image_sha256']=h;print(n,r.engine,r.model_loaded,r.is_demo,len(r.blocks),flush=True)
(O/'development_full_ocr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
