"""Manually reviewed, non-holdout development fragments (not model input)."""
import hashlib,json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_independent_tracks'
base='古籍评测集/医家类/思濟堂方書/'
items=[
 {'id':'A44-dual','path':base+'page_44.png','label':'positive','types':['unequal_tracks','adjacent_large'],'outer_bbox':[137,315,184,492],'main_bbox':[140,315,179,381],'columns':[{'side':'right','bbox':[158,379,180,489],'text':'二方俱見'},{'side':'left','bbox':[139,379,159,456],'text':'中暑門'}],'certainty':'双轨实印明确；主字上界及整句层级有歧义','adjacent_large':'下方 y约490 开始大字内'},
 {'id':'A16-dual','path':'古籍评测集/医家类/丹方彚编/page_16.png','label':'positive','types':['adjacent_large','paired_tracks'],'outer_bbox':[450,250,494,539],'main_bbox':[459,250,490,312],'columns':[{'side':'right','bbox':[473,345,490,536],'text':'未逐字转录'},{'side':'left','bbox':[453,345,473,536],'text':'未逐字转录'}],'certainty':'目视两条实印轨明确；字轨末尾及两个段落分界有歧义','adjacent_large':'上方痞滿为大字标题；下方另有正文'},
 {'id':'A32-lone','path':base+'page_32.png','label':'negative','types':['single_character','single_track_faint'],'outer_bbox':[128,390,179,501],'main_bbox':[134,398,172,457],'columns':[{'side':'right','bbox':[151,456,171,484],'text':'煅'}],'certainty':'右侧单字实印明确；左侧淡影不标为实字','adjacent_large':'上方石膏贴邻，后方淡影为疑似透印'},
 {'id':'A33-single','path':base+'page_33.png','label':'negative','types':['single_track_faint','adjacent_large'],'outer_bbox':[841,232,890,365],'main_bbox':[850,237,887,293],'columns':[{'side':'right','bbox':[870,290,888,351],'text':'土炒'}],'certainty':'右侧单列实印明确；左侧淡影为疑似透印','adjacent_large':'上方白术，下方淡影'},
 {'id':'A40-single','path':base+'page_40.png','label':'negative','types':['single_track_faint','adjacent_large'],'outer_bbox':[894,235,945,581],'main_bbox':[899,240,939,326],'columns':[{'side':'right','bbox':[916,324,939,577],'text':'自二錢可以加至四錢'}],'certainty':'右侧单轨实印明确；左侧淡字疑似背面透印','adjacent_large':'上方牡丹皮'},
 {'id':'A47-single','path':base+'page_47.png','label':'negative','types':['single_track_faint','adjacent_large'],'outer_bbox':[428,217,484,454],'main_bbox':[437,217,475,276],'columns':[{'side':'right','bbox':[450,274,476,447],'text':'辨治方藥大暑'}],'certainty':'右侧小字单轨明确；左侧淡影非确定实印','adjacent_large':'上方傷寒'},
]
labels=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));forbidden={v['sha256'] for v in labels['pages'].values()}
sheet=Image.new('RGB',(1800,1000),'white');draw=ImageDraw.Draw(sheet)
for i,s in enumerate(items):
 p=ROOT/s['path'];s['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();assert s['sha256'] not in forbidden
 im=imread_unicode(str(p));x,y,X,Y=s['outer_bbox'];box=[max(0,x-22),max(0,y-22),min(im.shape[1],X+22),min(im.shape[0],Y+22)];s['review_crop_bbox']=box
 crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));scale=min(2.5,280/crop.width,430/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));d=ImageDraw.Draw(crop)
 def rect(b,color):d.rectangle(tuple((b[k]-box[k%2])*scale for k in range(4)),outline=color,width=2)
 rect(s['main_bbox'],'yellow')
 for c in s['columns']:rect(c['bbox'],'lime')
 xx=i%6*300;sheet.paste(crop,(xx,45));draw.text((xx+4,5),s['id'],fill='black')
sheet.save(O/'new_development_reviewed.png')
target=O/'new_development_annotations.json';assert not target.exists();target.write_text(json.dumps({'scope':'manual original-image inspection; independent from 8 fixed pages; provisional boxes and uncertainties explicitly recorded','samples':items},ensure_ascii=False,indent=2),encoding='utf8');print('reviewed fragments',len(items))
