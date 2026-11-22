"""Collect unlabelled automatic proposal contact sheets from other scanned books."""
import hashlib,json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode,page_layout
from ocr.small_columns import propose_page
O=ROOT/'output/small_columns_step3_independent_tracks';O.mkdir(exist_ok=True)
labels=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));forbidden={x['sha256'] for x in labels['pages'].values()}
books=['丹方彚编','合併脉訣難經太素評林'];events=[]
for book in books:
 for path in sorted((ROOT/'古籍评测集/医家类'/book).glob('page_*.png')):
  digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest not in forbidden
  im=imread_unicode(str(path));cols=page_layout(str(path))['cols'];props=propose_page(im,cols)
  for q in props:
   x,y,X,Y=q['main_bbox'];end=max(c['bbox'][3] for c in q['columns']);top=max(0,y-18);bottom=min(im.shape[0],end+30);left=max(0,min(c['bbox'][0] for c in q['columns'])-12);right=min(im.shape[1],max(c['bbox'][2] for c in q['columns'])+12)
   events.append({'id':f'C{len(events)+1:03d}','book':book,'path':str(path.relative_to(ROOT)),'sha256':digest,'crop_bbox':[left,top,right,bottom],'proposal':q})
  print(book,path.name,'candidates',len(props),flush=True)
for start in range(0,len(events),30):
 subset=events[start:start+30];sheet=Image.new('RGB',(1800,((len(subset)+9)//10)*500),'white');d=ImageDraw.Draw(sheet)
 for i,e in enumerate(subset):
  im=imread_unicode(str(ROOT/e['path']));x,y,X,Y=e['crop_bbox'];crop=Image.fromarray(cv2.cvtColor(im[y:Y,x:X],cv2.COLOR_BGR2RGB));scale=min(2.5,170/crop.width,450/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));xx=i%10*180;yy=i//10*500;sheet.paste(crop,(xx,yy+40));d.text((xx+2,yy+3),e['id']+' '+Path(e['path']).stem,fill='black')
 sheet.save(O/f'other_books_contact_{start//30+1:02d}.png')
(O/'other_books_automatic_candidates.json').write_text(json.dumps({'method':'unlabelled automated candidates; manual review required','scanned_books':books,'events':events},ensure_ascii=False,indent=2),encoding='utf8');print('total candidates',len(events),flush=True)
