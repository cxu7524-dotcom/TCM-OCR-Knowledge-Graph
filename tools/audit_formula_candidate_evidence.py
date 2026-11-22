"""Audit frozen acceptance inputs using the predeclared local candidate set.

This is measurement only; no production selection or rule changes depend on it.
It does not claim an upper bound on OCR systems or on combining adjacent boxes.
"""
import json,sys,re
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ocr.normalize import normalize_text
from ocr.recognize import _engine
from ocr.vertical import imread_unicode,_binarize,_cut_points
frozen=ROOT/'output/formula_stage6/run-v6az4xah'
ann=json.loads((frozen/'正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
browser=json.loads((frozen/'正文方剂表_步骤6网页原始结果.json').read_text(encoding='utf8'))
models={'v5':str(ROOT/'模型权重/ch_PP-OCRv5_rec_mobile.onnx'),
        'cht_v3':str(ROOT/'模型权重/chinese_cht_PP-OCRv3_rec_mobile.onnx'), 'bundled_v3':''}
engines={k:_engine(v) for k,v in models.items()}
def read(img,model):
    rec=getattr(engines[model],'text_recognizer',None) or engines[model].text_rec
    val,_=rec([np.ascontiguousarray(np.rot90(img))]);return str(val[0][0])
def coverage(box,gt):
    x1,y1,x2,y2=box;a,b,c,d=gt
    return max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/max(1,(c-a)*(d-b))
def norm(v):return normalize_text(str(v or '')).replace(' ','')
result={'scope':'仅测量第一次复测冻结的自动框，局部候选使用与开发页同一预设方案；不进行方案选择。',
        'limits':'只检查单一自动框及其预设复识文本，不证明更强模型、跨框组合或端到端抽取无法恢复。',
        'pages':{}}
for page in ann['pages']:
 if page.get('split')!='验收':continue
 n=page['page_no']; blocks=browser['pages'][str(n)]['api']['blocks']
 image=imread_unicode(str(ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png'));height,width=image.shape[:2]
 cache={}; ledger=[]
 def evaluate(field,value,box):
    if not box:return {'field':field,'value':value,'status':'无标注框'}
    choices=[(coverage(b['bbox'],box),i) for i,b in enumerate(blocks) if b['bbox'][2]-b['bbox'][0]<=100]
    ratio,idx=max(choices,default=(0,-1))
    if ratio<.65:return {'field':field,'value':value,'status':'无充分覆盖的自动框','coverage':round(ratio,3)}
    if idx not in cache:
      b=blocks[idx];x1,y1,x2,y2=b['bbox']; orig=image[y1:y2,x1:x2]
      px1,py1,px2,py2=max(0,x1-6),max(0,y1-2),min(width,x2+6),min(height,y2+2)
      pad=image[py1:py2,px1:px2]
      proj=_binarize(cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)).sum(axis=1)/255
      cuts=_cut_points(proj,220)
      cache[idx]={'bbox':b['bbox'],'block_ref':b['block_ref'],'baseline':b['text'],
                  'same_box_v5':read(orig,'v5'),'pad6_v5':read(pad,'v5'),
                  'same_box_cht_v3':read(orig,'cht_v3'),'same_box_bundled_v3':read(orig,'bundled_v3'),
                  'split220_v5':''.join(read(orig[a:z],'v5') for a,z in zip(cuts[:-1],cuts[1:]) if z-a>=2),
                  'pad_bbox':[px1,py1,px2,py2], 'split_bboxes':[[x1,y1+a,x2,y1+z] for a,z in zip(cuts[:-1],cuts[1:])]}
    b=cache[idx]; modes=('baseline','same_box_v5','pad6_v5','same_box_cht_v3','same_box_bundled_v3','split220_v5')
    matches=[m for m in modes if norm(value) in norm(b[m])]
    score_value=re.sub(r'^各','',norm(value)) if field=='dose' else norm(value)
    score_matches=[m for m in modes if score_value in norm(b[m])]
    return {'field':field,'value':value,'block_ref':b['block_ref'],'bbox':b['bbox'],'coverage':round(ratio,3),
            'matching_candidates':matches,'matching_score_normalized_candidates':score_matches,
            'status':'候选有原字' if matches else '候选均缺少完整原字'}
 for instance in page['instances']:
  if instance.get('display_status')!='应出表':continue
  for index,row in enumerate(instance['ingredients'],1):
   if row.get('display_status')!='应出表':continue
   fields=[evaluate('name',instance['name_verbatim'],instance['name_bbox']),
           evaluate('herb',row['herb_verbatim'],row['herb_bbox']),
           evaluate('dose',row['dose_verbatim'],row['dose_bbox'])]
   ledger.append({'id':instance['id']+':'+str(index),'fields':fields,
                  'all_fields_in_candidates':all(f['status']=='候选有原字' for f in fields),
                  'all_fields_score_normalized_in_candidates':all(bool(f.get('matching_score_normalized_candidates')) for f in fields)})
 result['pages'][str(n)]={'rows':ledger,'automatic_box_candidates':cache}
 print('page',n,'rows',len(ledger),'all fields covered',sum(r['all_fields_in_candidates'] for r in ledger),flush=True)
rows=[r for p in result['pages'].values() for r in p['rows']]
result['summary']={'rows':len(rows),'all_fields_in_candidates':sum(r['all_fields_in_candidates'] for r in rows),
                   'shortfall_to_20':max(0,20-sum(r['all_fields_in_candidates'] for r in rows)),
                   'all_fields_score_normalized_in_candidates':sum(r['all_fields_score_normalized_in_candidates'] for r in rows),
                   'score_normalized_shortfall_to_20':max(0,20-sum(r['all_fields_score_normalized_in_candidates'] for r in rows)),
                   'field_status_counts':{f:{s:sum(x['status']==s for r in rows for x in r['fields'] if x['field']==f)
                                              for s in ('候选有原字','候选均缺少完整原字','无充分覆盖的自动框','无标注框')}
                                          for f in ('name','herb','dose')}}
(ROOT/'output/formula_stage6/验收页自动框候选输入证据审计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
