"""Development-only multi-scale glyph-cluster / three-state sequence experiment.

Samples select which automatically detected parents are inspected; no manual
annotation coordinates are used by the state decoder or candidate generator.
"""
import json,math,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
from ocr.small_columns_hybrid import _main_link_evidence
O=ROOT/'output/small_columns_step3_structured';O.mkdir(exist_ok=True)
old=json.loads((ROOT/'output/small_columns_step3_detector/development_comparison.json').read_text(encoding='utf8'))
dev=json.loads((ROOT/'output/small_columns_step3_hybrid/development_full_comparison.json').read_text(encoding='utf8'))['pages']
samples=json.loads((ROOT/'output/small_columns_step3_main/development_samples.json').read_text(encoding='utf8'))['samples']
def intersects(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def groups(mask,x0,y0):
 # Closes small gaps inside character-sized ink groups; positions remain on the source page.
 m=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 _,_,stats,_=cv2.connectedComponentsWithStats(m,8);return [{'bbox':[int(x+x0),int(y+y0),int(x+x0+w),int(y+y0+h)],'pixels':int(n)} for x,y,w,h,n in stats[1:] if n>=5]
def structured(im,p):
 main=p['main_bbox'];child=p['columns'];crop=p['crop_bbox'];typ=p['evidence']['page_typography'];h=typ['height'];step=4
 left=next(v['bbox'] for v in child if v['side']=='left');right=next(v['bbox'] for v in child if v['side']=='right');seam=(left[2]+right[0])/2
 x0=max(crop[0],min(left[0],right[0]));x1=min(crop[2],max(left[2],right[2]));y0=max(crop[1],int(min(main[1],left[1],right[1])-h));y1=min(crop[3],int(max(left[3],right[3])+h*1.2))
 gray=cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY);delta=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)-gray.astype(np.int16)
 masks={str(t):delta>=t for t in (14,22,30)};allgroups={t:groups(m,x0,y0) for t,m in masks.items()}
 centers=list(range(y0+step//2,y1,step));features=[];emissions=[]
 for cy in centers:
  # The smaller and normal character scales contribute independently.
  scales=[]
  for radius in (max(5,int(h*.27)),max(8,int(h*.45))):
   ya=max(0,cy-y0-radius);yb=min(y1-y0,cy-y0+radius);tile=masks['22'][ya:yb];k=max(1,min(tile.shape[1]-1,int(seam-x0)));s=max(0,k-2);t=min(tile.shape[1],k+2)
   l=int(tile[:,:s].sum());r=int(tile[:,t:].sum());middle=int(tile[:,s:t].sum());total=l+r+middle
   gl=gr=gc=0
   for g in allgroups['22']:
    a,b,c,d=g['bbox'];cx=(a+c)/2
    if b>=cy+radius or d<=cy-radius:continue
    if a<seam-2 and c>seam+2:gc+=1
    elif cx<seam:gl+=1
    else:gr+=1
   balance=2*min(l,r)/max(1,l+r);mid=middle/max(1,total);strength=min(1,total/max(18.,(yb-ya)*2.4));independent=min(1,gl)*min(1,gr);cross=min(2,gc)/2
   scales.append({'left':l,'right':r,'center':middle,'groups_left':gl,'groups_right':gr,'groups_cross':gc,'balance':round(balance,3),'center_fraction':round(mid,3),'strength':round(strength,3)})
  b=sum(s['balance'] for s in scales)/2;m=sum(s['center_fraction'] for s in scales)/2;strength=sum(s['strength'] for s in scales)/2
  separate=sum(min(1,s['groups_left'])*min(1,s['groups_right']) for s in scales)/2;cross=sum(min(2,s['groups_cross'])/2 for s in scales)/2
  # Background, centered main glyphs, and dual independent tracks are competing states.
  emissions.append([2.2*(1-strength)-.55, .6*strength+1.5*cross+2.2*m-.35, .8*strength+1.5*separate+1.3*b-1.8*cross-1.6*m-.8]);features.append({'y':cy,'scales':scales})
 # Viterbi: persistence encodes vertical continuity; main->dual is cheaper than background->dual.
 transition=np.array([[0,-.8,-2.0],[-.7,0,-.45],[-.7,-.55,0]],dtype=float);score=np.full((len(centers),3),-1e9);prev=np.zeros((len(centers),3),dtype=int);score[0]=emissions[0]
 for i in range(1,len(centers)):
  for state in range(3):
   cand=score[i-1]+transition[:,state];prev[i,state]=int(np.argmax(cand));score[i,state]=cand[prev[i,state]]+emissions[i][state]
 state=int(np.argmax(score[-1]));states=[state]
 for i in range(len(centers)-1,0,-1):state=int(prev[i,state]);states.append(state)
 states=states[::-1];runs=[]
 for i,state in enumerate(states):
  if runs and runs[-1]['state']==state:runs[-1]['end']=min(y1,centers[i]+step//2)
  else:runs.append({'state':state,'start':max(y0,centers[i]-step//2),'end':min(y1,centers[i]+step//2)})
 dual=[r for r in runs if r['state']==2 and intersects([x0,r['start'],x1,r['end']],[x0,min(left[1],right[1]),x1,max(left[3],right[3])])]
 trace={'roi':[x0,y0,x1,y1],'groups_by_contrast':allgroups,'features':features,'state_runs':[dict(r,state=('background','main','dual')[r['state']]) for r in runs]}
 if not dual:return None,dict(trace,reason='no_dual_state')
 best=max(dual,key=lambda r:min(r['end'],max(left[3],right[3]))-max(r['start'],min(left[1],right[1])))
 prior=[r for r in runs if r['state']==1 and r['end']<=best['start']+step]
 if not prior:return None,dict(trace,reason='no_preceding_main_state')
 mg=prior[-1]
 if best['start']-mg['end']>h*.75:return None,dict(trace,reason='nonlocal_main_state')
 # Gather each track's glyph groups from the decoded dual interval independently.
 sides=[]
 for side,lane in [('right',right),('left',left)]:
  candidates=[]
  for g in allgroups['22']:
   a,b,c,d=g['bbox'];cx=(a+c)/2
   if not (lane[0]<=cx<lane[2] and d>best['start']-step and b<best['end']+step):continue
   if c-a>1.25*(lane[2]-lane[0]):continue
   candidates.append(g)
  if not candidates:return None,dict(trace,reason='missing_lane_groups',missing_side=side)
  start=max(int(best['start']-step),min(g['bbox'][1] for g in candidates));end=min(int(best['end']+step),max(g['bbox'][3] for g in candidates));sides.append({'side':side,'bbox':[lane[0],start,lane[2],end]})
 q={'parent_bbox':p['parent_bbox'],'crop_bbox':p['crop_bbox'],'main_bbox':[main[0],mg['start'],main[2],mg['end']],'columns':sides,'reading_order':['main','right','left'],'evidence':{'structured_state_model':True}}
 return q,dict(trace,reason='candidate',selected_dual=best,selected_main=mg)
def coverage(mask,gt,b):
 x,y,X,Y=map(int,gt);a,b0,c,d=map(int,b);den=int(mask[y:Y,x:X].sum());inter=mask[max(y,b0):min(Y,d),max(x,a):min(X,c)];return round(int(inter.sum())/max(1,den),4) if min(Y,d)>max(y,b0) and min(X,c)>max(x,a) else 0
events=[]
for e in old['automatic_candidate_evidence']:
 im=imread_unicode(str(ROOT/dev[str(e['page'])]['source']));q,trace=structured(im,e['geometry']);events.append({'page':e['page'],'parent_bbox':e['parent_bbox'],'geometry':e['geometry'],'baseline_components':e['combined'],'structured_candidate':q,'trace':trace})
rows=[];positives=[s for s in samples if s['label']=='positive'];sheet=Image.new('RGB',(1200,((len(positives)+1)//2)*700),'white');draw=ImageDraw.Draw(sheet)
for s in samples:
 im=imread_unicode(str(ROOT/s['path']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=(cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)-gray.astype(np.int16))>=30;outer=s['outer_bbox'];mx=(outer[0]+outer[2])/2
 choices=[v for v in events if v['page']==s['page'] and v['parent_bbox'][0]-7<=mx<=v['parent_bbox'][2]+7 and any(intersects(c['bbox'],outer) for c in v['geometry']['columns'])];row={'sample':s}
 for name in ('baseline_components','structured_candidate'):
  cand=[e[name] for e in choices if e[name]]
  if s['label']=='negative':row[name]={'candidate_count':len(cand)};continue
  gt={c['side']:c['bbox'] for c in s['columns']};cand=[q for q in cand if all(intersects(c['bbox'],gt[c['side']]) for c in q['columns'])]
  q=max(cand,key=lambda q:coverage(mask,s['main_bbox'],q['main_bbox'])+sum(coverage(mask,gt[c['side']],c['bbox']) for c in q['columns'])) if cand else None
  row[name]={'prediction':q,'main_ink_coverage':coverage(mask,s['main_bbox'],q['main_bbox']) if q else 0,'side_ink_coverage':{c['side']:coverage(mask,gt[c['side']],c['bbox']) for c in q['columns']} if q else None}
 rows.append(row)
 if s['label']=='positive':
  i=positives.index(s);xx=i%2*600;yy=i//2*700
  for k,name in enumerate(('baseline_components','structured_candidate')):
   q=row[name]['prediction'];x,y,X,Y=outer;box=(max(0,x-23),max(0,y-24),min(im.shape[1],X+23),min(im.shape[0],Y+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,275/crop.width,600/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
   def rect(b,color):d.rectangle(tuple((b[z]-box[z%2])*scale for z in range(4)),outline=color,width=2)
   rect(s['main_bbox'],'yellow')
   for c in s['columns']:rect(c['bbox'],'lime')
   if q:
    rect(q['main_bbox'],'cyan')
    for c in q['columns']:rect(c['bbox'],'red')
   sheet.paste(crop,(xx+k*300,yy+55));draw.text((xx+k*300,yy+3),s['id']+' '+name,fill='black');draw.text((xx+k*300,yy+20),str(row[name]['side_ink_coverage']),fill='black')
sheet.save(O/'development_comparison.png')
target=O/'development_structured.json';assert not target.exists();target.write_text(json.dumps({'method':'three-state Viterbi over multiscale connected ink groups; automatic parent and geometric proposal only; annotation used for selection and post-inference scoring','samples':rows,'events':events},ensure_ascii=False,indent=2),encoding='utf8')
for row in rows:print(row['sample']['id'],[(n,row[n].get('side_ink_coverage',row[n].get('candidate_count'))) for n in ('baseline_components','structured_candidate')],flush=True)
