"""Compare official recognition models on frozen identical local crops."""
import hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort,yaml
from rapidocr_onnxruntime import RapidOCR
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step5';manifest=O/'crop_manifest.json';target=O/'recognition_raw.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads(manifest.read_text(encoding='utf8'));mobile=ROOT/'模型权重/ch_PP-OCRv5_rec_mobile.onnx';server=O/'official_server_inference.onnx';keys=O/'official_server_keys.txt';yml=O/'official_server_inference.yml'
f=json.loads((ROOT/'output/small_columns_step3_detector/freeze_v5.json').read_text(encoding='utf8'));assert sha(mobile)==f['model']['sha256'];assert sha(server)=='d9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752'
chars=yaml.safe_load(yml.read_text(encoding='utf8'))['PostProcess']['character_dict'];assert chars==keys.read_text(encoding='utf8').splitlines();session=ort.InferenceSession(str(mobile),providers=['CPUExecutionProvider']);assert chars==session.get_modelmeta().custom_metadata_map['character'].splitlines();del session
models={};loads={}
for name,path in [('mobile',mobile),('server',server)]:
 start=time.perf_counter();kw={'rec_model_path':str(path.resolve())};
 if name=='server':kw['rec_keys_path']=str(keys.resolve())
 models[name]=RapidOCR(**kw).text_rec;loads[name]=round(time.perf_counter()-start,4)
assert models['mobile'].rec_image_shape==models['server'].rec_image_shape==[3,48,320]
warm=np.full((48,100,3),255,np.uint8)
for model in models.values():model([warm])
rows=[]
for i,c in enumerate(cfg['crops']):
 path=ROOT/c['crop'];assert sha(path)==c['crop_sha256'];crop=imread_unicode(str(path));readings={}
 for scale in (1,2):
  image=crop if scale==1 else cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC);image=np.ascontiguousarray(np.rot90(image))
  for name,model in models.items():
   start=time.perf_counter();raw,_=model([image]);elapsed=time.perf_counter()-start;text,conf=(raw[0] if raw else ('',0));text=str(text).strip();conf=float(conf)
   readings[f'{name}_{scale}x']={'text':text if conf>=.15 else '','raw_text':text,'confidence':round(conf,4),'elapsed_ms':round(elapsed*1000,3)}
 rows.append({'crop':c,'readings':readings})
 if (i+1)%12==0:print('processed',i+1,'/',len(cfg['crops']),flush=True)
result={'runtime':{'sys_prefix':sys.prefix,'python':sys.version,'onnxruntime':ort.__version__,'providers':['CPUExecutionProvider'],'rec_image_shape':[3,48,320],'shared_preprocess':'lossless same source crop; optional bicubic 2x; 90 degrees CCW; same RapidOCR 1.4.4 recognizer, dict and conf floor','init_seconds':loads,'warmup':'one blank crop per model'},'model_sources':{'mobile':{'path':str(mobile.relative_to(ROOT)),'sha256':sha(mobile)},'server':{'path':str(server.relative_to(ROOT)),'sha256':sha(server),'dictionary_sha256':sha(keys),'source':'https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx/tree/main','dictionary_identical_to_mobile':True}},'crop_manifest_sha256':sha(manifest),'rows':rows}
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('saved raw',sha(target),flush=True)
