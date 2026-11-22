import json,sys,importlib.util
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode,page_layout;from ocr.small_columns import propose_page,estimate_page_typography
O=ROOT/'output/small_columns_step3_main';src=O.parent/'small_columns_step3_retry/frozen_retry.py';spec=importlib.util.spec_from_file_location('frozen_two',src);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
a=json.loads((O/'development_samples.json').read_text(encoding='utf8'));samples=a['samples'];pages={}
for n in sorted({s['page'] for s in samples}):
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';im=imread_unicode(str(p));cols=page_layout(str(p))['cols'];pages[str(n)]={'automatic_columns':cols,'page_typography':estimate_page_typography(im,cols),'v2':old.propose_page(im,cols),'v3':propose_page(im,cols)}
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def coverage(mask,gt,b):
 x1,y1,x2,y2=gt;x3,y3,x4,y4=b;total=int(mask[y1:y2,x1:x2].sum());xa=max(x1,x3);xb=min(x2,x4);ya=max(y1,y3);yb=min(y2,y4);return round(int(mask[ya:yb,xa:xb].sum())/max(1,total),4) if xa<xb and ya<yb else 0.0
rows=[];pos=[s for s in samples if s['label']=='positive'];sheet=Image.new('RGB',(1200,((len(pos)+1)//2)*700),'white');d=ImageDraw.Draw(sheet)
for s in samples:
 im=imread_unicode(str(ROOT/s['path']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);back=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16);ink=(back-gray.astype(np.int16))>=30;o=s['outer_bbox'];mid=(o[0]+o[2])/2;r={'sample':s}
 for v in ('v2','v3'):
  candidates=[q for q in pages[str(s['page'])][v] if q['parent_bbox'][0]-7<=mid<=q['parent_bbox'][2]+7 and any(area(c['bbox'],o)>0 for c in q['columns'])]
  if s['label']=='positive':
   gt={c['side']:c['bbox'] for c in s['columns']};candidates=[q for q in candidates if all(area(c['bbox'],gt[c['side']])>0 for c in q['columns'])]
   q=max(candidates,key=lambda q:coverage(ink,s['main_bbox'],q['main_bbox'])+sum(coverage(ink,gt[c['side']],c['bbox']) for c in q['columns'])) if candidates else None
   r[v]={'prediction':q,'main_ink_coverage':coverage(ink,s['main_bbox'],q['main_bbox']) if q else 0,'side_ink_coverage':{c['side']:coverage(ink,gt[c['side']],c['bbox']) for c in q['columns']} if q else None}
  else:r[v]={'predictions':candidates}
 rows.append(r)
 if s['label']=='positive':
  i=pos.index(s);x=(i%2)*600;y=(i//2)*700
  for j,v in enumerate(('v2','v3')):
   q=r[v]['prediction'];x1,y1,x2,y2=o;box=(max(0,x1-23),max(0,y1-24),min(im.shape[1],x2+23),min(im.shape[0],y2+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,275/crop.width,600/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));dd=ImageDraw.Draw(crop)
   def rect(b,color):dd.rectangle(tuple((b[k]-box[k%2])*scale for k in range(4)),outline=color,width=2)
   rect(s['main_bbox'],'yellow')
   for c in s['columns']:rect(c['bbox'],'lime')
   if q:
    rect(q['main_bbox'],'cyan')
    for c in q['columns']:rect(c['bbox'],'red')
   sheet.paste(crop,(x+j*300,y+55));d.text((x+j*300,y+3),s['id']+' '+v,fill='black');d.text((x+j*300,y+20),'main ink '+str(r[v]['main_ink_coverage']),fill='black')
sheet.save(O/'development_before_after_overlay.png');(O/'development_comparison.json').write_text(json.dumps({'pages':pages,'samples':rows},ensure_ascii=False,indent=2),encoding='utf8')
for r in rows:
 print(r['sample']['id'], 'v2',r['v2'].get('main_ink_coverage',len(r['v2'].get('predictions',[]))),'v3',r['v3'].get('main_ink_coverage',len(r['v3'].get('predictions',[]))), 'notes',r['v3'].get('side_ink_coverage'))
