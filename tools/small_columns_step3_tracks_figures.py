import json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_independent_tracks'
data=json.loads((O/'development_v5_comparison.json').read_text(encoding='utf8'));events=json.loads((O/'development_independent_tracks.json').read_text(encoding='utf8'))['events'];ids=['S35','S42b','S36','A44-dual','A33-single','A40-single'];sheet=Image.new('RGB',(1200,len(ids)*600),'white');draw=ImageDraw.Draw(sheet)
for i,key in enumerate(ids):
 row=next(x for x in data['samples'] if x['sample']['id']==key);s=row['sample'];im=imread_unicode(str(ROOT/s['path']));x,y,X,Y=s['outer_bbox'];box=[max(0,x-28),max(0,y-35),min(im.shape[1],X+28),min(im.shape[0],Y+25)]
 for k,name in enumerate(('fifth_round_baseline','three_state','independent_tracks')):
  q=(row[name]['predictions'][0] if row[name]['predictions'] else None) if name=='fifth_round_baseline' else row[name].get('prediction')
  if s['label']=='negative' and name!='fifth_round_baseline':
   path=s['path'].replace('\\','/');cand=[e[name] for e in events if e['path']==path and e[name] and e['geometry']['parent_bbox'][0]-7<=(x+X)/2<=e['geometry']['parent_bbox'][2]+7];q=cand[0] if cand else None
  crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(3,360/crop.width,520/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
  def rect(b,color):d.rectangle(tuple((b[j]-box[j%2])*scale for j in range(4)),outline=color,width=2)
  rect(s['main_bbox'],'yellow')
  for c in s['columns']:rect(c['bbox'],'lime')
  if q:
   rect(q['main_bbox'],'cyan')
   for c in q['columns']:rect(c['bbox'],'red')
  sheet.paste(crop,(k*400,i*600+50));draw.text((k*400+5,i*600+8),key+' '+name,fill='black')
sheet.save(O/'development_tracks_comparison.png')
