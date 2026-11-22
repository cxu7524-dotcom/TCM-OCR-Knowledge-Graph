import hashlib,json,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr, _recognize_page
from ocr.normalize import normalize_text
OUT=ROOT/'output/small_columns_step1_holdout'; a=json.loads((OUT/'annotations.json').read_text(encoding='utf-8'))
object.__setattr__(CONFIG.paths,'reports_dir',OUT/'reports')
model=ROOT/CONFIG.ocr.rec_model if CONFIG.ocr.rec_model else None
all_matches=[]
for n,meta in a['pages'].items():
 p=ROOT/meta['path']; pid='SMALL-HOLDOUT-'+n
 selected=_recognize_page(str(p),pid,None)
 raw_selected={'page_id':pid,'image':meta['path'],'image_sha256':meta['sha256'],'engine':selected.engine,'model_loaded':selected.model_loaded,'is_demo':selected.is_demo,'width':selected.width,'height':selected.height,'notes':selected.notes,'blocks':[b.to_dict() for b in selected.blocks],'scope':'selected recognition before low-confidence filtering and layout postprocessing'}
 (OUT/f'page_{n}_selected_raw.json').write_text(json.dumps(raw_selected,ensure_ascii=False,indent=2),encoding='utf-8')
 result=run_ocr(str(p),page_id=pid)
 raw=result.to_dict();raw['baseline_metadata']={'ocr_config':asdict(CONFIG.ocr),'image_sha256':meta['sha256'],'recognizer_sha256':hashlib.sha256(model.read_bytes()).hexdigest() if model and model.is_file() else None,'method':'unchanged run_ocr; report directory isolated'}
 (OUT/f'page_{n}_full_ocr.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
 for s in (s for s in a['samples'] if str(s['page'])==n):
  q=s['outer_bbox']
  blocks=[b.to_dict() for b in result.blocks if min(b.bbox[2],q[2])>max(b.bbox[0],q[0]) and min(b.bbox[3],q[3])>max(b.bbox[1],q[1])]
  dose=(s.get('dose') or {}).get('verbatim')
  all_matches.append({'id':s['id'],'page':s['page'],'blocks':blocks,'dose_fragment':dose,'applies_to':(s.get('dose') or {}).get('applies_to',[]),'exact_hit':any(dose in b['text'] for b in blocks) if dose else None,'normalized_hit':any(normalize_text(dose) in normalize_text(b['text']) for b in blocks) if dose else None})
 print(n,result.engine,result.model_loaded,result.is_demo,len(result.blocks),flush=True)
(OUT/'sample_matches.json').write_text(json.dumps({'method':'bbox intersects outer; literal or normalized fragment within one output block; no cross-block concatenation','samples':all_matches},ensure_ascii=False,indent=2),encoding='utf-8')
