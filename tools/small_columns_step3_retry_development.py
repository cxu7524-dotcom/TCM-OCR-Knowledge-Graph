import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode,page_layout;from ocr.small_columns import propose_page
O=ROOT/'output/small_columns_step3_retry';samples=json.loads((O/'development_samples.json').read_text(encoding='utf8'))['samples'];pages={}
for n in sorted({s['page'] for s in samples}):
 path=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';im=imread_unicode(str(path));layout=page_layout(str(path));pages[str(n)]={'auto_columns':layout['cols'],'predictions':propose_page(im,layout['cols'])}
def area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
out=[];sheet=Image.new('RGB',(1500,1400),'white');draw=ImageDraw.Draw(sheet)
for i,s in enumerate(samples):
 im=imread_unicode(str(ROOT/s['path']));o=s['outer_bbox'];pred=pages[str(s['page'])]['predictions'];mid=(o[0]+o[2])/2
 near=[p for p in pred if p['parent_bbox'][0]-7<=mid<=p['parent_bbox'][2]+7 and any(area(c['bbox'],o)>0 for c in p['columns'])]
 r={'sample':s,'predictions':near};out.append(r)
 x1,y1,x2,y2=o;box=(max(0,x1-25),max(0,y1-20),min(im.shape[1],x2+25),min(im.shape[0],y2+20));crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,280/crop.width,620/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
 def rect(b,color):d.rectangle(tuple((b[z]-box[z%2])*scale for z in range(4)),outline=color,width=2)
 for c in s.get('columns',[]):rect(c['bbox'],'lime')
 for p in near:
  rect(p['main_bbox'],'cyan')
  for c in p['columns']:rect(c['bbox'],'red')
 x=(i%5)*300;y=(i//5)*700;sheet.paste(crop,(x,y+35));draw.text((x,y),s['id']+'  candidates='+str(len(near)),fill='black')
sheet.save(O/'development_review.png');(O/'development_predictions.json').write_text(json.dumps({'pages':pages,'samples':out},ensure_ascii=False,indent=2),encoding='utf8')
for n,r in pages.items():print('page',n,'predictions',len(r['predictions']))
for r in out:print(r['sample']['id'],len(r['predictions']), [[c['bbox'] for c in p['columns']] for p in r['predictions']])
