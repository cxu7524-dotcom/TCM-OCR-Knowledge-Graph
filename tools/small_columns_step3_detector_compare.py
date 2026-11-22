"""Compare detector-only and geometry+connected-ink candidates on development data."""
import json,sys,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
from ocr.small_columns_hybrid import _main_link_evidence
O=ROOT/'output/small_columns_step3_detector'
raw=json.loads((O/'development_detector_raw.json').read_text(encoding='utf8'))
old=json.loads((ROOT/'output/small_columns_step3_hybrid/development_full_comparison.json').read_text(encoding='utf8'))['pages']
samples=json.loads((ROOT/'output/small_columns_step3_main/development_samples.json').read_text(encoding='utf8'))['samples']
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def cover(mask,gt,b):
 x,y,X,Y=map(int,gt);a,b0,c,d=map(int,b);part=mask[y:Y,x:X];den=part.sum();return round(float(mask[max(y,b0):min(Y,d),max(x,a):min(X,c)].sum())/max(1,int(den)),4) if min(Y,d)>max(y,b0) and min(X,c)>max(x,a) else 0
def component_end(mask,box):
 x,y,X,Y=map(int,box);bottom=min(mask.shape[0],Y+9);top=max(y,Y-20);tile=mask[top:bottom,x:X].astype('uint8')
 if tile.size==0:return box,[]
 _,_,stats,_=cv2.connectedComponentsWithStats(tile,8);cross=[]
 for a,b,w,h,n in stats[1:]:
  start=int(top+b);end=int(top+b+h)
  if start<Y and Y<end<=Y+8 and n>=5 and end<bottom:cross.append({'bbox':[int(x+a),start,int(x+a+w),end],'pixels':int(n)})
 end=max([Y]+[c['bbox'][3] for c in cross]);return [x,y,X,end],cross
def detector_candidate(p,ev):
 # A detector box must be a narrow lane-local segment in the existing child interval.
 # No ground-truth coordinates enter this selection.
 selected=[];diagnostic={}
 for side in p['columns']:
  lane=side['bbox'];mx=(lane[0]+lane[2])/2;boxes=[]
  for b in ev['detections']['both']['boxes']:
   x,y,X,Y=b['bbox_original'];center=(x+X)/2
   if not (lane[0]-2<=center<=lane[2]+2 and X-x<=1.3*(lane[2]-lane[0]) and y>=lane[1]-2 and Y<=lane[3]+8 and area([x,y,X,Y],lane)>0):continue
   boxes.append([max(lane[0],math.floor(x)),max(lane[1]-2,math.floor(y)),min(lane[2],math.ceil(X)),math.ceil(Y)])
  diagnostic[side['side']]=boxes
  if not boxes:return None,diagnostic
  selected.append({'side':side['side'],'bbox':[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]})
 q=dict(p);q['columns']=selected;return q,diagnostic
events=[];proposals={}
for e in raw['events']:
 n=str(e['page']);im=imread_unicode(str(ROOT/old[n]['source']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);blur=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16);mask=(blur-gray.astype(np.int16))>=22
 for ev in e['proposals']:
  p=ev['geometry'];link=_main_link_evidence(im,p);det,details=detector_candidate(p,ev)
  sides=[];trace=[]
  for c in p['columns']:
   b,components=component_end(mask,c['bbox']);sides.append({'side':c['side'],'bbox':b});trace.append({'side':c['side'],'crossing_components':components})
  q=dict(p);q['columns']=sides
  if not link['valid']:q=None;det=None
  cache=next((a for a in old[n]['audit'] if a['parent_bbox']==p['parent_bbox'] and a['geometric_main']==p['main_bbox'] and a['geometric_columns']==p['columns']),None)
  recognition=[{'side':a['side'],'original_status':a['candidates'][1]['status']} for a in cache.get('side_comparisons',[])] if cache else []
  item={'page':int(n),'parent_bbox':p['parent_bbox'],'geometry':p,'detector_only':det,'combined':q,'link':link,'component_trace':trace,'detector_lane_segments':details,'recognition_status':recognition,'recognition_status_is_not_geometry_gate':True}
  events.append(item);proposals.setdefault(n,[]).append(item)
rows=[];positives=[s for s in samples if s['label']=='positive'];sheet=Image.new('RGB',(1800,((len(positives)+1)//2)*700),'white');draw=ImageDraw.Draw(sheet)
for s in samples:
 n=str(s['page']);im=imread_unicode(str(ROOT/s['path']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);blur=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16);mask=(blur-gray.astype(np.int16))>=30;outer=s['outer_bbox'];mx=(outer[0]+outer[2])/2
 choices=[v for v in proposals.get(n,[]) if v['parent_bbox'][0]-7<=mx<=v['parent_bbox'][2]+7 and any(area(c['bbox'],outer) for c in v['geometry']['columns'])]
 row={'sample':s}
 for name in ('geometry','detector_only','combined'):
  cand=[v[name] for v in choices if v[name] is not None]
  if s['label']=='negative':row[name]={'candidate_count':len(cand)};continue
  gt={c['side']:c['bbox'] for c in s['columns']};cand=[q for q in cand if all(area(c['bbox'],gt[c['side']]) for c in q['columns'])]
  q=max(cand,key=lambda q:cover(mask,s['main_bbox'],q['main_bbox'])+sum(cover(mask,gt[c['side']],c['bbox']) for c in q['columns'])) if cand else None
  row[name]={'prediction':q,'main_ink_coverage':cover(mask,s['main_bbox'],q['main_bbox']) if q else 0,'side_ink_coverage':{c['side']:cover(mask,gt[c['side']],c['bbox']) for c in q['columns']} if q else None}
 rows.append(row)
 if s['label']=='positive':
  i=positives.index(s);xx=(i%2)*900;yy=(i//2)*700
  for k,name in enumerate(('geometry','detector_only','combined')):
   q=row[name]['prediction'];x,y,X,Y=outer;box=(max(0,x-23),max(0,y-24),min(im.shape[1],X+23),min(im.shape[0],Y+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,275/crop.width,600/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
   def rect(b,color):d.rectangle(tuple((b[z]-box[z%2])*scale for z in range(4)),outline=color,width=2)
   rect(s['main_bbox'],'yellow')
   for c in s['columns']:rect(c['bbox'],'lime')
   if q:
    rect(q['main_bbox'],'cyan')
    for c in q['columns']:rect(c['bbox'],'red')
   sheet.paste(crop,(xx+k*300,yy+55));draw.text((xx+k*300,yy+3),s['id']+' '+name,fill='black');draw.text((xx+k*300,yy+20),str(row[name]['side_ink_coverage']),fill='black')
sheet.save(O/'development_comparison.png')
(O/'development_comparison.json').write_text(json.dumps({'method':'development annotations used only after generation for association and coverage; connected components and detector outputs use automatic geometry alone','samples':rows,'automatic_candidate_evidence':events},ensure_ascii=False,indent=2),encoding='utf8')
for r in rows:
 print(r['sample']['id'],[(k,r[k].get('side_ink_coverage',r[k].get('candidate_count'))) for k in ('geometry','detector_only','combined')])
