"""Compare unchanged page routing with a compatibility-enabled vertical route.

Runs both routes on the same development images; reports name and ingredient
coverage, not acceptance-page performance. No files under the production OCR
package are modified. 32/36 are the held-out pages.
"""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ocr.pipeline import _recognize_page
from ocr.normalize import normalize_text
from ocr.recognize import _engine

ann=json.loads((ROOT/'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
pages=[p for p in ann['pages'] if p.get('split')=='开发' and p.get('book')=='思濟堂方書' and p['page_no'] in (32,33,35,36)]
engine=_engine()
out={'development_pages':[33,35], 'heldout_pages':[32,36], 'pages':{}}
for p in pages:
 n=p['page_no']; image=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png'; out['pages'][str(n)]={}
 for mode in ('as_installed','recognizer_compatibility'):
  if mode=='as_installed' and hasattr(engine,'text_recognizer'):
   delattr(engine,'text_recognizer')
  if mode=='recognizer_compatibility':
   engine.text_recognizer=engine.text_rec
  result=_recognize_page(str(image),f'DEV-{n}',None)
  body=''.join(normalize_text(x.text) for x in result.blocks)
  names=[i['name_verbatim'] for i in p['instances'] if i.get('name_verbatim')]
  ingredients=[h['herb_verbatim'] for i in p['instances'] for h in i['ingredients'] if h.get('role')=='基本组成']
  item={'engine':result.engine,'notes':result.notes,'block_count':len(result.blocks),
        'name_hits':sum(normalize_text(x) in body for x in names),'name_total':len(names),
        'herb_hits':sum(normalize_text(x) in body for x in ingredients),'herb_total':len(ingredients),
        'blocks':[{'text':b.text,'bbox':b.bbox,'confidence':b.conf} for b in result.blocks]}
  out['pages'][str(n)][mode]=item
  print(n,mode,item['engine'],item['name_hits'],item['name_total'],item['herb_hits'],item['herb_total'],flush=True)
  (ROOT/'output/formula_stage6/开发页竖排路径对照.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
