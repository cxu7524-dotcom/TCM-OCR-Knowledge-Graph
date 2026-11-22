"""Offline proposal-seeded independent-lane continuation feasibility study."""
import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ocr.vertical import imread_unicode,page_layout
from ocr.small_columns import propose_page
from ocr.small_columns_hybrid import _main_link_evidence
from small_columns_structured_core import structured
O=ROOT/'output/small_columns_step3_independent_tracks'
old=json.loads((ROOT/'output/small_columns_step3_main/development_samples.json').read_text(encoding='utf8'))['samples']
new=json.loads((O/'new_development_annotations.json').read_text(encoding='utf8'))['samples']
samples=old+new;paths=sorted({s['path'].replace('\\','/') for s in samples});events=[]
def continuation(im,p):
 link=_main_link_evidence(im,p);q,trace=structured(im,p);audit={'main_link':link,'structured_reason':trace['reason'],'state_runs':trace['state_runs']}
 if not link['valid']:return None,dict(audit,reason='rejected_main_link')
 if q is None:return None,dict(audit,reason='no_confirmed_dual_state')
 seed=trace['selected_dual'];audit['confirmed_dual']=[seed['start'],seed['end']];h=p['evidence']['page_typography']['height'];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);delta=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)-gray.astype(np.int16);mask=(delta>=22).astype('uint8');sides=[];tracks=[]
 for c in p['columns']:
  x,y,X,Y=c['bbox'];tile=mask[y:Y,x:X];tile=cv2.morphologyEx(tile,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));_,_,stats,_=cv2.connectedComponentsWithStats(tile,8)
  groups=[]
  for a,b,w,hh,n in stats[1:]:
   if n<5 or w>1.2*(X-x) or hh>h*2.2:continue
   groups.append({'bbox':[int(x+a),int(y+b),int(x+a+w),int(y+b+hh)],'pixels':int(n)})
  groups.sort(key=lambda g:g['bbox'][1]);start=max(y,seed['start']-4);end=min(Y,seed['end']+4)
  active=[g for g in groups if g['bbox'][3]>start and g['bbox'][1]<end]
  if not active:return None,dict(audit,reason='no_lane_group',missing_side=c['side'])
  # A lane may independently continue after the simultaneous state, while its
  # existing automatic geometry box is a hard limit against later main glyphs.
  accepted=list(active);last=max(g['bbox'][3] for g in accepted)
  for g in groups:
   if g in accepted or g['bbox'][3]<=last:continue
   if g['bbox'][1]-last>max(4,int(h*.8)):break
   accepted.append(g);last=max(last,g['bbox'][3])
  box=[x,y,X,min(Y,last+1)];sides.append({'side':c['side'],'bbox':box});tracks.append({'side':c['side'],'seed_range':[start,end],'first_group_y':min(g['bbox'][1] for g in accepted),'last_group_y':last,'continuation_after_dual':max(0,last-seed['end']),'geometry_limit':Y,'groups':accepted})
 z=dict(p);z['columns']=sides;z['evidence']=dict(p['evidence'],independent_track_experiment=True)
 return z,dict(audit,reason='candidate',tracks=tracks)
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def cover(mask,gt,b):
 x,y,X,Y=map(int,gt);a,c,d,e=map(int,b);den=int(mask[y:Y,x:X].sum());return round(int(mask[max(y,c):min(Y,e),max(x,a):min(X,d)].sum())/max(1,den),4) if min(Y,e)>max(y,c) and min(X,d)>max(x,a) else 0
for path in paths:
 im=imread_unicode(str(ROOT/path));cols=page_layout(str(ROOT/path))['cols'];props=propose_page(im,cols)
 for p in props:
  prev,prevtrace=structured(im,p);newq,ev=continuation(im,p)
  events.append({'path':path,'geometry':p,'three_state':prev,'independent_tracks':newq,'audit':ev})
 print(path,'automatic proposals',len(props),flush=True)
rows=[]
for s in samples:
 path=s['path'].replace('\\','/');im=imread_unicode(str(ROOT/path));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=(cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)-gray.astype(np.int16))>=30;outer=s['outer_bbox'];mx=(outer[0]+outer[2])/2;choices=[e for e in events if e['path']==path and e['geometry']['parent_bbox'][0]-7<=mx<=e['geometry']['parent_bbox'][2]+7 and any(area(c['bbox'],outer) for c in e['geometry']['columns'])];row={'sample':s}
 for name in ('geometry','three_state','independent_tracks'):
  candidates=[e[name] for e in choices if e[name]]
  if s['label']=='negative':row[name]={'candidate_count':len(candidates)};continue
  gt={c['side']:c['bbox'] for c in s['columns']};candidates=[q for q in candidates if all(area(c['bbox'],gt[c['side']]) for c in q['columns'])]
  q=max(candidates,key=lambda q:cover(mask,s['main_bbox'],q['main_bbox'])+sum(cover(mask,gt[c['side']],c['bbox']) for c in q['columns'])) if candidates else None
  row[name]={'prediction':q,'main_ink_coverage':cover(mask,s['main_bbox'],q['main_bbox']) if q else 0,'side_ink_coverage':{c['side']:cover(mask,gt[c['side']],c['bbox']) for c in q['columns']} if q else None}
 rows.append(row)
target=O/'development_independent_tracks.json';assert not target.exists();target.write_text(json.dumps({'scope':'full automatic proposals on selected development pages; manual labels used only for evaluation','samples':rows,'events':events},ensure_ascii=False,indent=2),encoding='utf8')
for row in rows:
 print(row['sample']['id'],[(n,row[n].get('side_ink_coverage',row[n].get('candidate_count'))) for n in ('geometry','three_state','independent_tracks')],flush=True)
