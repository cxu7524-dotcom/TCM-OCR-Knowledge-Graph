"""Compare local OCR candidates on *automatic* detection boxes from development pages.

Ground-truth rectangles only assign evaluation targets to existing boxes; they
never define a crop or a production candidate. 33/35 tune, 32/36 hold out.
The JSON keeps original and candidate crop coordinates for independent audit.
"""
import json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ocr.recognize import _engine
from ocr.normalize import normalize_text
from ocr.vertical import imread_unicode,_binarize,_cut_points

annotations=json.loads((ROOT/'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
previous=json.loads((ROOT/'output/formula_stage6/开发页竖排路径对照.json').read_text(encoding='utf8'))
models={'v5':str(ROOT/'模型权重/ch_PP-OCRv5_rec_mobile.onnx'),
        'cht_v3':str(ROOT/'模型权重/chinese_cht_PP-OCRv3_rec_mobile.onnx'), 'bundled_v3':''}
engines={k:_engine(v) for k,v in models.items()}
def recognize(img,model):
    rec=getattr(engines[model],'text_recognizer',None) or engines[model].text_rec
    pieces, _=rec([np.ascontiguousarray(np.rot90(img))])
    return {'text':str(pieces[0][0]),'confidence':float(pieces[0][1])}
def coverage(box,target):
    x1,y1,x2,y2=box; a,b,c,d=target
    return max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/max(1,(c-a)*(d-b))
def contains(candidate, expected):
    return normalize_text(expected).replace(' ','') in normalize_text(candidate).replace(' ','')

result={'description':'检测框由同一页面 OCR 自动产生；标注框只用于分配核验字段；识别候选不读标注文字或坐标。',
        'tuning_pages':[33,35],'holdout_pages':[32,36], 'pages':{}}
for page in annotations['pages']:
    n=page.get('page_no')
    if page.get('split')!='开发' or page.get('book')!='思濟堂方書' or n not in (32,33,35,36):continue
    image=imread_unicode(str(ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png'))
    blocks=previous['pages'][str(n)]['as_installed']['blocks']
    targets=[]
    for instance in page['instances']:
        if instance.get('name_bbox') and instance.get('name_verbatim'):
            targets.append({'field':'name','value':instance['name_verbatim'],'bbox':instance['name_bbox']})
        for row in instance['ingredients']:
            if row.get('role')!='基本组成':continue
            for field in ('herb','dose'):
                value=row.get(field+'_verbatim'); bbox=row.get(field+'_bbox')
                if value and bbox:targets.append({'field':field,'value':value,'bbox':bbox})
    # Shared dose belongs to several ingredients; count its evidence once.
    uniq={ (t['field'],t['value'],tuple(t['bbox'])):t for t in targets}
    targets=list(uniq.values())
    box_targets={}; unmatched=[]
    for target in targets:
        choices=[(coverage(b['bbox'],target['bbox']),i) for i,b in enumerate(blocks)
                 if b['bbox'][2]-b['bbox'][0] <= 100]
        score,idx=max(choices,default=(0,-1))
        item=dict(target,coverage=round(score,3))
        if score<.65:unmatched.append(item)
        else:box_targets.setdefault(idx,[]).append(item)
    records=[]
    for idx,ts in sorted(box_targets.items()):
        block=blocks[idx]; x1,y1,x2,y2=block['bbox']; h,w=image.shape[:2]
        def crop(pad):
            x0=max(0,x1-pad); xe=min(w,x2+pad); y0=max(0,y1-2 if pad else y1); ye=min(h,y2+2 if pad else y2)
            return image[y0:ye,x0:xe],[x0,y0,xe,ye]
        orig,box0=crop(0); padded,box6=crop(6)
        candidates={'baseline':{'text':block['text'],'confidence':block['confidence'],'crop_bbox':box0},
                    'same_box_v5':dict(recognize(orig,'v5'),crop_bbox=box0),
                    'pad6_v5':dict(recognize(padded,'v5'),crop_bbox=box6),
                    'same_box_cht_v3':dict(recognize(orig,'cht_v3'),crop_bbox=box0),
                    'same_box_bundled_v3':dict(recognize(orig,'bundled_v3'),crop_bbox=box0)}
        rowproj=_binarize(cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)).sum(axis=1)/255.
        cuts=_cut_points(rowproj,220);parts=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a<2:continue
            parts.append(dict(recognize(orig[a:b],'v5'),bbox=[x1,y1+a,x2,y1+b]))
        candidates['split220_v5']={'text':''.join(p['text'] for p in parts),'pieces':parts,'crop_bbox':box0}
        for target in ts:
            target['hits']={mode:contains(v['text'],target['value']) for mode,v in candidates.items()}
        records.append({'automatic_block_index':idx,'automatic_bbox':block['bbox'],'targets':ts,'candidates':candidates})
    result['pages'][str(n)]={'split':'holdout' if n in (32,36) else 'tuning',
                             'uncovered_targets':unmatched,'boxes':records}
    print('page',n,'automatic boxes',len(records),'covered fields',sum(len(x['targets']) for x in records),'uncovered',len(unmatched),flush=True)
    (ROOT/'output/formula_stage6/开发页自动框复识对照.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')

for split in ('tuning','holdout'):
    result[split+'_summary']={}
    for field in ('name','herb','dose'):
        pages=[p for p in result['pages'].values() if p['split']==split]
        targets=[t for p in pages for b in p['boxes'] for t in b['targets'] if t['field']==field]
        uncovered=sum(sum(t['field']==field for t in p['uncovered_targets']) for p in pages)
        modes=('baseline','same_box_v5','pad6_v5','same_box_cht_v3','same_box_bundled_v3','split220_v5')
        result[split+'_summary'][field]={'covered':len(targets),'uncovered':uncovered,
              'hits':{m:sum(t['hits'][m] for t in targets) for m in modes},
              'oracle_union':sum(any(t['hits'].values()) for t in targets)}
(ROOT/'output/formula_stage6/开发页自动框复识对照.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k.endswith('_summary')},ensure_ascii=False),flush=True)
