import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
from ocr.vertical import imread_unicode,page_layout
from ocr.small_columns_components import propose_components_page
O=ROOT/'output/small_columns_step3_independent_tracks';object.__setattr__(CONFIG.paths,'reports_dir',O/'dev_ocr_reports')
data=json.loads((O/'development_independent_tracks.json').read_text(encoding='utf8'));cached=json.loads((ROOT/'output/small_columns_step3_hybrid/development_full_ocr.json').read_text(encoding='utf8'));pages={}
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
for row in data['samples']:
 s=row['sample'];path=s['path'].replace('\\','/')
 if path in pages:continue
 p=ROOT/path;im=imread_unicode(str(p));n=Path(path).stem.split('_')[-1];use_cache='思濟堂方書' in path and n in cached
 if use_cache:
  result=cached[n];assert hashlib.sha256(p.read_bytes()).hexdigest()==result['image_sha256'];blocks=result['blocks'];status={'mode':'reused_saved_normal_ocr','loaded':result['model_loaded'],'demo':result['is_demo']}
 else:
  r=run_ocr(str(p),page_id='INDEPENDENT-DEV-'+n);blocks=r.to_dict()['blocks'];status={'mode':'new_normal_ocr','loaded':r.model_loaded,'demo':r.is_demo}
 assert status['loaded'] and not status['demo']
 cols=page_layout(str(p))['cols'];pred=propose_components_page(im,cols,blocks);pages[path]={'ocr_status':status,'predictions':pred['predictions'],'audit':pred['audit']}
 print(path,'v5 proposals',len(pred['predictions']),flush=True)
for r in data['samples']:
 s=r['sample'];mx=(s['outer_bbox'][0]+s['outer_bbox'][2])/2;opts=[p for p in pages[s['path'].replace('\\','/')]['predictions'] if p['parent_bbox'][0]-7<=mx<=p['parent_bbox'][2]+7 and any(area(c['bbox'],s['outer_bbox']) for c in p['columns'])]
 if s['label']=='positive':
  gt={c['side']:c['bbox'] for c in s['columns']};opts=[q for q in opts if all(area(c['bbox'],gt[c['side']]) for c in q['columns'])]
 r['fifth_round_baseline']={'candidate_count':len(opts),'predictions':opts}
 print(s['id'],'v5',len(opts),'independent',r['independent_tracks'].get('candidate_count',int(r['independent_tracks'].get('prediction') is not None)),flush=True)
target=O/'development_v5_comparison.json';assert not target.exists();target.write_text(json.dumps({'method':'fifth-round frozen components model; actual non-demo normal OCR; labels used only after inference; new OCR reports isolated','samples':data['samples'],'pages':pages},ensure_ascii=False,indent=2),encoding='utf8')
