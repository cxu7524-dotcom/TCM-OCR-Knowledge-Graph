import json,sys,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_hybrid';pages=json.loads((O/'development_full_comparison.json').read_text(encoding='utf8'))['pages'];samples=json.loads((ROOT/'output/small_columns_step3_main/development_samples.json').read_text(encoding='utf8'))['samples']
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def coverage(mask,gt,b):
 x1,y1,x2,y2=gt;x3,y3,x4,y4=b;total=int(mask[y1:y2,x1:x2].sum());xa=max(x1,x3);xb=min(x2,x4);ya=max(y1,y3);yb=min(y2,y4);return round(int(mask[ya:yb,xa:xb].sum())/max(1,total),4) if xa<xb and ya<yb else 0.
results=[];pos=[x for x in samples if x['label']=='positive'];sheet=Image.new('RGB',(1200,((len(pos)+1)//2)*700),'white');draw=ImageDraw.Draw(sheet)
for s in samples:
 im=imread_unicode(str(ROOT/s['path']));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);bg=cv2.GaussianBlur(g,(31,31),0).astype('int16');mask=(bg-g.astype('int16'))>=30;o=s['outer_bbox'];mid=(o[0]+o[2])/2;page=pages[str(s['page'])];r={'sample':s}
 for name,key in [('geometry','geometry_predictions'),('hybrid','hybrid_predictions')]:
  cands=[p for p in page[key] if p['parent_bbox'][0]-7<=mid<=p['parent_bbox'][2]+7 and any(area(c['bbox'],o)>0 for c in p['columns'])]
  if s['label']=='positive':
   gt={c['side']:c['bbox'] for c in s['columns']};cands=[p for p in cands if all(area(c['bbox'],gt[c['side']])>0 for c in p['columns'])];q=max(cands,key=lambda p:coverage(mask,s['main_bbox'],p['main_bbox'])+sum(coverage(mask,gt[c['side']],c['bbox']) for c in p['columns'])) if cands else None
   r[name]={'prediction':q,'main_ink_coverage':coverage(mask,s['main_bbox'],q['main_bbox']) if q else 0,'side_ink_coverage':{c['side']:coverage(mask,gt[c['side']],c['bbox']) for c in q['columns']} if q else None}
  else:r[name]={'candidate_count':len(cands)}
 results.append(r)
 if s['label']=='positive':
  i=pos.index(s);x=i%2*600;y=i//2*700
  for k,name in enumerate(('geometry','hybrid')):
   q=r[name]['prediction'];x1,y1,x2,y2=o;box=(max(0,x1-23),max(0,y1-24),min(im.shape[1],x2+23),min(im.shape[0],y2+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,275/crop.width,600/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));dd=ImageDraw.Draw(crop)
   def rect(b,color):dd.rectangle(tuple((b[z]-box[z%2])*scale for z in range(4)),outline=color,width=2)
   rect(s['main_bbox'],'yellow')
   for c in s['columns']:rect(c['bbox'],'lime')
   if q:
    rect(q['main_bbox'],'cyan')
    for c in q['columns']:rect(c['bbox'],'red')
   sheet.paste(crop,(x+k*300,y+55));draw.text((x+k*300,y+3),s['id']+' '+name,fill='black');draw.text((x+k*300,y+20),str(r[name]['side_ink_coverage']),fill='black')
sheet.save(O/'development_geometry_vs_hybrid.png')
wrong=[]
for n,parent,start in [(31,751,613),(35,654,597)]:
 event=next(e for e in pages[str(n)]['audit'] if e['parent_bbox'][0]==parent and abs(e['geometric_columns'][0]['bbox'][1]-start)<=10);wrong.append({'page':n,'source_parent_x':parent,'review':'原圖中主字候選和雙軌之間有獨立大字；雙軌實為普通大字','event':event})
result={'scope':'development examples, post-inference comparisons; pixel coverage is auxiliary','samples':results,'wrong_main_link_cases':wrong}
(O/'development_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for r in results:
 if r['sample']['label']=='positive':print(r['sample']['id'],'geo',r['geometry']['side_ink_coverage'],'hybrid',r['hybrid']['side_ink_coverage'])
 else:print(r['sample']['id'],'geo',r['geometry']['candidate_count'],'hybrid',r['hybrid']['candidate_count'])
for r in wrong:print('wrong-main',r['page'],r['event']['decision'],r['event']['reason'])
