import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
if len(sys.argv)!=2 or sys.argv[1] not in ('before','after'): raise SystemExit('specify before or after')
phase=sys.argv[1]; out=ROOT/'output/small_columns_step2';out.mkdir(exist_ok=True)
object.__setattr__(CONFIG.paths,'reports_dir',out/(phase+'_reports'))
paths={str(n):ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png' for n in (32,35,42)}
paths.update({k:ROOT/'data/samples/synthetic/pages'/f'SYN-2026-001__{k}.png' for k in ('p0011','p0015')})
holdout=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf-8')); forbidden={x['sha256'] for x in holdout['pages'].values()}
rows={}
for key,p in paths.items():
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h in forbidden: raise RuntimeError('holdout image used')
 r=run_ocr(str(p),page_id='STEP2-'+key); rows[key]=r.to_dict();rows[key]['image_sha256']=h
 print(phase,key,r.engine,r.model_loaded,r.is_demo,len(r.blocks),len(r.tables),flush=True)
(out/(phase+'_page_results.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
