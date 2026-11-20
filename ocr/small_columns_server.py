"""Lazy optional official PP-OCRv5-server recognition for local crop fallback."""
from functools import lru_cache
import hashlib
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'模型权重/PP-OCRv5_server_rec_official.onnx'
KEYS=ROOT/'模型权重/PP-OCRv5_server_rec_official.keys.txt'
MODEL_SHA256='d9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752'
KEYS_SHA256='1ea29636956177e400af712d9782e7693f3fb25f98617bed10479d2965a836fd'

@lru_cache(maxsize=1)
def _recognizer():
 from rapidocr_onnxruntime import RapidOCR
 for path,expected in ((MODEL,MODEL_SHA256),(KEYS,KEYS_SHA256)):
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
   raise ValueError('official server model or dictionary missing or changed')
 return RapidOCR(rec_model_path=str(MODEL),rec_keys_path=str(KEYS)).text_rec

def read_pair(image,box):
 """Use the same native and 2x rotated crops as mobile; exceptions propagate."""
 x,y,X,Y=map(int,box);crop=image[y:Y,x:X]
 if not crop.size:raise ValueError('empty crop')
 rec=_recognizer();out=[]
 for scale in (1,2):
  image=crop if scale==1 else cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
  rows,_=rec([np.ascontiguousarray(np.rot90(image))]);text,conf=rows[0] if rows else ('',0.)
  out.append((str(text).strip(),round(float(conf),4)) if float(conf)>=.15 else ('',0.))
 return out
