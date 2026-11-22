import json,sys,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_main';a=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));p=json.loads((O/'holdout_third_predictions.json').read_text(encoding='utf8'))
def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def coverage(mask,target,box):
 x1,y1,x2,y2=target; area=mask[y1:y2,x1:x2];total=int(area.sum());x3,y3,x4,y4=box
 x5,y5,x6,y6=max(x1,x3),max(y1,y3),min(x2,x4),min(y2,y4);covered=int(mask[y5:y6,x5:x6].sum()) if x6>x5 and y6>y5 else 0
 return round(covered/max(total,1),4),total
def contam(mask,pred,gt):
 x1,y1,x2,y2=pred;pixels=mask[y1:y2,x1:x2];total=int(pixels.sum());x3,y3,x4,y4=gt;inside=mask[max(y1,y3-2):min(y2,y4+2),max(x1,x3-2):min(x2,x4+2)];covered=int(inside.sum()) if inside.size else 0
 return round(max(0,total-covered)/max(total,1),4)
rows=[];all_sheets={}
for label in ('positive','negative'):
 subset=[s for s in a['samples'] if s['label']==label];sheet=Image.new('RGB',(1500,((len(subset)+4)//5)*700),'white');draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(subset):
  n=str(s['page']);im=imread_unicode(str(ROOT/a['pages'][n]['path']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,14)>0
  outer=s['outer_bbox'];auto=p['pages'][n]['automatic_columns'];pred=p['pages'][n]['predictions'];mx=(outer[0]+outer[2])/2
  col_options=[c for c in auto if c['bbox'][0]-7<=mx<=c['bbox'][2]+7 and intersection(c['bbox'],outer)>0];associated=[v for v in pred if v['parent_bbox'] in [c['bbox'] for c in col_options]]
  row={'id':s['id'],'label':label,'page':s['page'],'target_outer':outer,'target_main':s.get('main_bbox'),'target_columns':s['columns'],'matching_auto_columns':col_options,'matching_proposals':associated,'selected_proposal':None}
  if label=='positive':
   gt={x['side']:x['bbox'] for x in s['columns']};candidates=[]
   for q in associated:
    qb={x['side']:x['bbox'] for x in q['columns']}
    if not all(intersection(qb[k],gt[k])>0 for k in ('right','left')):continue
    cr={k:coverage(mask,gt[k],qb[k])[0] for k in ('right','left')};extra={k:contam(mask,qb[k],gt[k]) for k in ('right','left')};mc=coverage(mask,s['main_bbox'],q['main_bbox'])[0] if s.get('main_bbox') else 0
    score=min(cr.values())-.3*max(extra.values())+.1*mc;candidates.append((score,q,cr,extra,mc))
   if candidates:
    _,q,cr,extra,mc=max(candidates,key=lambda z:z[0]);row.update({'selected_proposal':q,'target_ink_coverage':cr,'extra_ink_fraction':extra,'main_ink_coverage':mc,'screen':'manual_review_required'})
   else:row['screen']='miss_no_matched_split' if col_options else 'miss_no_automatic_outer_column'
  else:
   # Only emitted dual pairs intersecting the fixed negative region count as candidate false splits.
   overlapping=[q for q in pred if sum(intersection(c['bbox'],outer) for c in q['columns'])>0]
   row['matching_proposals']=overlapping;row['screen']='candidate_false_split_review' if overlapping else 'no_split';row['selected_proposal']=overlapping[0] if overlapping else None
  rows.append(row)
  x1,y1,x2,y2=outer;box=(max(0,x1-25),max(0,y1-20),min(im.shape[1],x2+25),min(im.shape[0],y2+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3.0,280/crop.width,620/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
  def rect(b,color,width=2):
   x,y,z,w=b;d.rectangle(((x-box[0])*scale,(y-box[1])*scale,(z-box[0])*scale,(w-box[1])*scale),outline=color,width=width)
  if s.get('main_bbox'):rect(s['main_bbox'],'yellow')
  for c in s['columns']:rect(c['bbox'],'lime')
  if row['selected_proposal']:
   q=row['selected_proposal'];rect(q['main_bbox'],'cyan')
   for c in q['columns']:rect(c['bbox'],'red')
  x=(i%5)*300;y=(i//5)*700;sheet.paste(crop,(x,y+45));draw.text((x+3,y+4),s['id']+' '+row['screen'],fill='black')
  if 'target_ink_coverage' in row:draw.text((x+3,y+22),str(row['target_ink_coverage'])+' extra '+str(row['extra_ink_fraction']),fill='black')
 sheet.save(O/f'holdout_{label}_review.png')
(O/'holdout_third_review_screen.json').write_text(json.dumps({'method':'predictions generated independently; labels only for mapping and ink-screen; visual review required','samples':rows},ensure_ascii=False,indent=2),encoding='utf8')
for t in ('positive','negative'):
 from collections import Counter
 print(t,dict(Counter(x['screen'] for x in rows if x['label']==t)))
