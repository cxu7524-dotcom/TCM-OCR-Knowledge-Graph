"""Read-only replay of selective server fallback under existing stage-4 gates."""
import copy,json,sys
from pathlib import Path
import cv2,numpy as np
from rapidocr_onnxruntime import RapidOCR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.schemas import OCRResult
from extract.formula_rows import extract_formula_rows,_DOSE
from ocr.normalize import normalize_text
import ocr.small_columns_attach as module
O=ROOT/'output/small_columns_step5';target=O/'selective_replay.json';assert not target.exists()
model=RapidOCR(rec_model_path=str((O/'official_server_inference.onnx').resolve()),rec_keys_path=str((O/'official_server_keys.txt').resolve())).text_rec
def read_server(im,box,scale):
 x,y,X,Y=box;crop=im[y:Y,x:X]
 if scale==2:crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
 out,_=model([np.ascontiguousarray(np.rot90(crop))]);text,conf=out[0] if out else ('',0)
 return (str(text).strip(),round(float(conf),4)) if float(conf)>=.15 else ('',0)
base=ROOT/'output/small_columns_step1_holdout';labels=json.loads((base/'annotations.json').read_text(encoding='utf8'));fifth=json.loads((ROOT/'output/small_columns_step3_detector/holdout_fifth_predictions.json').read_text(encoding='utf8'));orig=module._read;pages={}
for n,meta in labels['pages'].items():
 ocr=OCRResult.from_dict(json.loads((base/f'page_{n}_full_ocr.json').read_text(encoding='utf8')));ocr.image_path=str(ROOT/meta['path']);right_boxes={tuple(c['bbox']) for p in fifth['pages'][n]['predictions'] for c in p['columns'] if c['side']=='right'};cache={};selected=[]
 def choose(im,box,scale):
  key=tuple(box)
  if key not in right_boxes:return orig(im,box,scale)
  if key not in cache:
   m=[orig(im,box,k) for k in (1,2)];right=normalize_text(m[1][0]);match=_DOSE.match(right)
   if not match or min(m[0][1],m[1][1])<.8:
    s=[read_server(im,box,k) for k in (1,2)];n0=normalize_text(s[0][0]);n1=normalize_text(s[1][0]);sm=_DOSE.match(n1)
    if sm and n0==n1 and min(s[0][1],s[1][1])>=.8:
     cache[key]=s;selected.append({'bbox':list(box),'mobile':m,'server':s})
    else:cache[key]=m
   else:cache[key]=m
  return cache[key][scale-1]
 module._read=choose
 try:
  original=extract_formula_rows(ocr.blocks);evidence,enhanced,audit=module.attach_small_columns(ocr,original)
 finally:module._read=orig
 pages[n]={'evidence':[b.to_dict() for b in evidence],'formula':enhanced,'audit':audit,'server_selections':selected}
 print(n,'server-selected',len(selected),'attached',audit['accepted'],flush=True)
target.write_text(json.dumps({'method':'fixed v5 auto geometry and step4 existing gates; server only when mobile dose does not parse or either mobile scale conf <0.8; server 1x/2x must match and both >=0.8; original mobile retained otherwise','pages':pages},ensure_ascii=False,indent=2),encoding='utf8')
