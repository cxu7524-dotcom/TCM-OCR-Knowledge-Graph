import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from ocr.vertical import imread_unicode,page_layout;from ocr.small_columns import propose_page;from ocr.small_columns_hybrid import propose_hybrid_page
O=ROOT/'output/small_columns_step3_hybrid';ocr=json.loads((O/'development_full_ocr.json').read_text(encoding='utf8'));results={}
for n in (31,32,33,35,36,38,40,42,48):
 p=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png';im=imread_unicode(str(p));cols=page_layout(str(p))['cols'];raw=ocr[str(n)];assert raw['model_loaded'] and not raw['is_demo'];assert hashlib.sha256(p.read_bytes()).hexdigest()==raw['image_sha256']
 geom=propose_page(im,cols);mixed=propose_hybrid_page(im,cols,raw['blocks']);results[str(n)]={'source':str(p.relative_to(ROOT)),'image_sha256':raw['image_sha256'],'auto_columns':cols,'auto_ocr_blocks':raw['blocks'],'geometry_predictions':geom,'hybrid_predictions':mixed['predictions'],'audit':mixed['audit']}
 print(n,'geometry',len(geom),'hybrid',len(mixed['predictions']),'fallback',sum(x['decision']=='fallback_outer' for x in mixed['audit']),flush=True)
for n,x,y in [(31,751,613),(35,654,597),(36,187,355)]:
 print('EXAMPLE',n,x)
 for ev in results[str(n)]['audit']:
  if ev['parent_bbox'][0]==x and abs(ev['geometric_columns'][0]['bbox'][1]-y)<=10:print(json.dumps(ev,ensure_ascii=True))
(O/'development_full_comparison.json').write_text(json.dumps({'scope':'same source image, automatic columns and normal OCR; no reference box used in inference','pages':results},ensure_ascii=False,indent=2),encoding='utf8')
