"""Development-only RapidOCR detector experiment; no holdout inference."""
import json,sys,time,hashlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.recognize import _engine
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_detector';O.mkdir(exist_ok=True)
assert not (O/'development_detector_raw.json').exists()
prior=json.loads((ROOT/'output/small_columns_step3_hybrid/development_full_comparison.json').read_text(encoding='utf8'))['pages']
samples=json.loads((ROOT/'output/small_columns_step3_main/development_samples.json').read_text(encoding='utf8'))['samples']
holdout=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'))
forbidden={v['sha256'] for v in holdout['pages'].values()}
detector=_engine().text_det
def detect(im,roi,scale=2):
 x1,y1,x2,y2=map(int,roi);x1=max(0,x1);y1=max(0,y1);x2=min(im.shape[1],x2);y2=min(im.shape[0],y2)
 crop=im[y1:y2,x1:x2];up=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);rot=np.ascontiguousarray(np.rot90(up));boxes,elapsed=detector(rot);out=[]
 if boxes is not None:
  for b in boxes:
   pts=[[round(x1+(up.shape[1]-1-float(v))/scale,2),round(y1+float(u)/scale,2)] for u,v in b]
   out.append({'polygon_original':pts,'bbox_original':[round(min(p[0] for p in pts),2),round(min(p[1] for p in pts),2),round(max(p[0] for p in pts),2),round(max(p[1] for p in pts),2)]})
 return {'roi_original':[x1,y1,x2,y2],'scale':scale,'rotation':'90 degrees counterclockwise','detector_seconds':elapsed,'boxes':out}
def intersects(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))>0
tasks=[]
for s in samples:
 n=str(s['page']);page=prior[n];outer=s['outer_bbox'];mid=(outer[0]+outer[2])/2
 cols=[c for c in page['auto_columns'] if c['bbox'][0]-7<=mid<=c['bbox'][2]+7 and intersects(c['bbox'],outer)]
 for col in cols:tasks.append((n,col['bbox'],s['id']))
for n,x in [('31',751),('35',654)]:
 col=next(c for c in prior[n]['auto_columns'] if c['bbox'][0]==x);tasks.append((n,col['bbox'],'wrong_link_negative'))
seen=set();results=[]
for n,parent,tag in tasks:
 key=(n,tuple(parent))
 if key in seen:continue
 seen.add(key);page=prior[n];path=ROOT/page['source'];digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==page['image_sha256'] and digest not in forbidden
 im=imread_unicode(str(path));props=[p for p in page['geometry_predictions'] if p['parent_bbox']==parent]
 event={'page':int(n),'parent_bbox':parent,'source_sha256':digest,'selection_tag':tag,'proposals':[]}
 # Full parent detection supplies independent evidence, including negatives with no geometric proposal.
 event['full_parent']=detect(im,[parent[0]-8,parent[1],parent[2]+8,parent[3]])
 for p in props:
  h=p['evidence']['page_typography']['height'];left,right=sorted(p['columns'],key=lambda c:c['bbox'][0]);b0=left['bbox'];b1=right['bbox'];y0=max(parent[1],int(min(b0[1],b1[1])-h));y1=min(parent[3],int(max(b0[3],b1[3])+h))
  inputs={'both':[min(b0[0],b1[0]),y0,max(b0[2],b1[2]),y1], 'left':[b0[0],y0,b0[2],y1], 'right':[b1[0],y0,b1[2],y1]}
  event['proposals'].append({'geometry':p,'detections':{side:detect(im,roi) for side,roi in inputs.items()}})
 results.append(event);print('page',n,'parent',parent,'geometry',len(props),'tag',tag,flush=True)
(O/'development_detector_raw.json').write_text(json.dumps({'method':'existing local RapidOCR text detector; 2x bicubic; 90-degree CCW; mapped back to original page; selected automatic parents in development only; annotations never enter detector or candidate geometry','events':results},ensure_ascii=False,indent=2),encoding='utf8')
