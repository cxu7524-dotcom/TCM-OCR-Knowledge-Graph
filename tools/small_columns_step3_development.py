import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import page_layout,imread_unicode
from ocr.small_columns import propose_page
j=json.loads((ROOT/'output/small_columns_step2/development_crop_comparison.json').read_text(encoding='utf8'));out={}
for n in [32,35,42]:
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';image=imread_unicode(str(p));cols=page_layout(str(p))['cols'];pred=propose_page(image,cols);out[str(n)]={'auto_columns':cols,'predictions':pred}
 print('PAGE',n,'predictions',len(pred))
 for s in j['samples']:
  if s['sample']['page']!=n:continue
  o=s['sample']['outer'];near=[v for v in pred if min(v['parent_bbox'][2],o[2])>max(v['parent_bbox'][0],o[0]) and any(min(c['bbox'][3],o[3])>max(c['bbox'][1],o[1]) for c in v['columns'])]
  print(s['sample']['id'],[(x['parent_bbox'],x['columns']) for x in near])
(ROOT/'output/small_columns_step3/development_predictions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
