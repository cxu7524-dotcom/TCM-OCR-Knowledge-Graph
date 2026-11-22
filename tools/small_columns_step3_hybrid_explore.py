import json,sys,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode;from ocr.recognize import recognize_crop
O=ROOT/'output/small_columns_step3_hybrid';dev=json.loads((ROOT/'output/small_columns_step3_main/development_comparison.json').read_text(encoding='utf8'));full=json.loads((O/'development_full_ocr.json').read_text(encoding='utf8'));out=[]
for row in dev['samples']:
 s=row['sample'];q=row['v3'].get('prediction')
 if s['label']!='positive' or q is None:continue
 im=imread_unicode(str(ROOT/s['path']));rec={'id':s['id'],'page':s['page'],'auto_parent':q['parent_bbox'],'main_bbox':q['main_bbox'],'ocr_blocks':[{'line_id':b['line_id'],'bbox':b['bbox'],'text':b['text'],'confidence':b['conf']} for b in full[str(s['page'])]['blocks'] if min(b['bbox'][2],q['parent_bbox'][2])>max(b['bbox'][0],q['parent_bbox'][0]) and min(b['bbox'][3],max(c['bbox'][3] for c in q['columns']))>max(b['bbox'][1],q['main_bbox'][1])],'sides':{}}
 for side in q['columns']:
  x1,y1,x2,y2=side['bbox'];opts=[]
  for delta in (-8,-4,0,4,8,12):
   end=min(im.shape[0],max(y1+4,y2+delta));crop=im[y1:end,x1:x2]
   if side['side']=='right':crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
   text,conf=recognize_crop(np.ascontiguousarray(np.rot90(crop)),min_conf=.15)
   opts.append({'delta_bottom':delta,'bbox':[x1,y1,x2,end],'text':text,'confidence':conf})
  rec['sides'][side['side']]=opts
 out.append(rec)
 print(s['id'],json.dumps({k:[(z['delta_bottom'],z['text'],z['confidence']) for z in v] for k,v in rec['sides'].items()},ensure_ascii=True),flush=True)
(O/'development_boundary_ocr_exploration.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
