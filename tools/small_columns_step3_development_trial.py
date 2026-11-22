import json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));O=ROOT/'output/small_columns_step3'
from ocr.vertical import page_layout,imread_unicode
v1=(O/'frozen_v1.py').read_text(encoding='utf8');v2=v1.replace('MIN_WINDOWS = 5','MIN_WINDOWS = 2').replace('        if len(group)<MIN_WINDOWS:continue','        if len(group)<MIN_WINDOWS or (len(group)<5 and min(v[2] for v in group)>0.01):continue').replace('np.ones((5,1),np.uint8)).ravel()\n        main_end','np.ones((11,1),np.uint8)).ravel()\n        main_end')
p=O/'development_trial_v2.py';p.write_text(v2,encoding='utf8');spec=importlib.util.spec_from_file_location('step3_trial_v2',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
samples=json.loads((ROOT/'output/small_columns_step2/development_crop_comparison.json').read_text(encoding='utf8'))['samples'];result={}
for n in [32,35,42]:
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';pred=module.propose_page(imread_unicode(str(p)),page_layout(str(p))['cols']);items={}
 for s in samples:
  if s['sample']['page']!=n:continue
  q=s['sample']['outer'];mid=(q[0]+q[2])/2
  items[s['sample']['id']]=[x for x in pred if x['parent_bbox'][0]-7<=mid<=x['parent_bbox'][2]+7 and all(max(c['bbox'][1],q[1])<min(c['bbox'][3],q[3]) for c in x['columns'])]
 result[str(n)]={'proposal_count':len(pred),'samples':items,'predictions':pred}
(O/'development_trial_v2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print({n:{'count':v['proposal_count'],'sample_counts':{k:len(x) for k,x in v['samples'].items()}} for n,v in result.items()})
