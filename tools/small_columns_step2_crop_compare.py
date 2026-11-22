import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.recognize import recognize_crop,resolve_rec_model,_engine
from ocr.normalize import normalize_text
from ocr.vertical import imread_unicode
OUT=ROOT/'output/small_columns_step2';OUT.mkdir(exist_ok=True)
ref=json.loads((ROOT/'output/formula_page35_small_columns_ab.json').read_text(encoding='utf8'))
a=ref['records']; samples=[{'id':'S35','page':35,'outer':a[0]['bbox'],'right':{'bbox':a[3]['bbox'],'expected':a[3]['expected_groups'][0]},'left':{'bbox':a[4]['bbox'],'expected':a[4]['expected_groups'][0]}},
 {'id':'S32','page':32,'outer':[133,240,172,376],'right':{'bbox':[151,320,172,376],'expected':'三錢'},'left':{'bbox':[133,320,151,376],'expected':'飛淨'}},
 {'id':'S42a','page':42,'outer':[944,400,983,543],'right':{'bbox':[964,458,983,543],'expected':'各三錢'},'left':{'bbox':[944,458,963,516],'expected':'去心'}},
 {'id':'S42b','page':42,'outer':[944,565,983,710],'right':{'bbox':[964,649,983,710],'expected':'一錢'},'left':{'bbox':[944,649,963,680],'expected':'研'}}]
holdpath=ROOT/'output/small_columns_step1_holdout/annotations.json'; frozen_bytes=holdpath.read_bytes();hold=json.loads(frozen_bytes);forbidden={v['sha256'] for v in hold['pages'].values()}
model,name,source=resolve_rec_model(); engine=_engine(); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert name=='ch_PP-OCRv5_rec_mobile.onnx' and model and hasattr(engine,'text_rec'),(name,source)
assert samples[0]['right']['bbox']==a[3]['bbox'] and samples[0]['left']['bbox']==a[4]['bbox']
def read(img,box,scale):
 x1,y1,x2,y2=box; crop=img[y1:y2,x1:x2]
 if scale==2:crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
 return recognize_crop(np.ascontiguousarray(np.rot90(crop)),min_conf=0.15)
rows=[]
for s in samples:
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f"page_{s['page']}.png"; im=imread_unicode(str(p)); h=sha(p)
 assert h not in forbidden and im is not None
 o=s['outer'];hgt,wdt=im.shape[:2];assert 0<=o[0]<o[2]<=wdt and 0<=o[1]<o[3]<=hgt
 row={'sample':s,'image':str(p.relative_to(ROOT)),'image_sha256':h,'rotation':'90 deg counterclockwise','inputs':{}}
 for label,box,scale in [('outer',o,1),('outer_2x',o,2),('right_native',s['right']['bbox'],1),('right_2x',s['right']['bbox'],2),('left_native',s['left']['bbox'],1),('left_2x',s['left']['bbox'],2)]:
  assert o[0]<=box[0]<box[2]<=o[2] and o[1]<=box[1]<box[3]<=o[3]
  runs=[read(im,box,scale) for _ in range(3)]; row['inputs'][label]={'bbox':box,'scale':scale,'runs':[{'text':t,'confidence':c} for t,c in runs],'stable':len(set(runs))==1}
 row['selection']={'right':'right_2x','left':'left_native','policy':'fixed using prior page35 manual comparison; applied unchanged to all development examples'}
 for side in ('right','left'):
  chosen=row['inputs'][row['selection'][side]]; expected=s[side]['expected']; chosen['expected']=expected
  chosen['exact_success']=all(v['text']==expected for v in chosen['runs']);chosen['normalized_success']=all(normalize_text(v['text'])==normalize_text(expected) for v in chosen['runs']);chosen['status']='success' if chosen['exact_success'] else 'recognition_failure'
 rows.append(row);print(s['id'],[(side,row['inputs'][row['selection'][side]]['exact_success']) for side in ('right','left')],flush=True)
assert hashlib.sha256(holdpath.read_bytes()).digest()==hashlib.sha256(frozen_bytes).digest()
result={'model':name,'model_source':source,'model_sha256':sha(Path(model)),'dependency_entry':'text_rec','production_entry':'ocr.recognize.recognize_crop','original_manual_boxes_equal':True,'holdout_annotation_sha256_before_after':sha(holdpath),'holdout_pages_used':False,'repeat_count':3,'samples':rows}
(OUT/'development_crop_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
