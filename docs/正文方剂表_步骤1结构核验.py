import json, hashlib
from pathlib import Path
from collections import Counter
from PIL import Image
root=Path(__file__).resolve().parents[1]
d=json.loads((root/'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
ids=set(); total=Counter(); report=[]
for p in d['pages']:
 img=root/p['image']; assert img.is_file(),p['id']
 assert hashlib.sha256(img.read_bytes()).hexdigest()==p['sha256'],p['id']
 assert list(Image.open(img).size)==p['image_size'],p['id']
 w,h=p['image_size']
 def rect(b):
  if b is None:return
  assert isinstance(b,list) and len(b)==4 and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h,(p['id'],b)
 for region in p.get('coverage_regions',[]):rect(region['bbox']);assert region['kind'] and region['details']
 for n in p['negative_spans']:rect(n['bbox']);assert n['reason']
 for i in p['instances']:
  assert i['id'] not in ids;ids.add(i['id']);rect(i['name_bbox']);rect(i['quote_bbox'])
  for a in i['ingredients']:
   rect(a['herb_bbox']);rect(a['dose_bbox']);assert a['herb_bbox']
   if a['dose_status']=='原文有载':assert a['dose_verbatim'] and a['dose_bbox']
   if a['dose_status']=='原文未载':assert a['dose_verbatim'] is None and a['dose_bbox'] is None
 if p['split']=='验收':
  r=p['full_page_review']; assert r['status']=='全页正文已检查' and not p['unreviewed_regions'] and p['coverage_regions']
  rows=[a for i in p['instances'] for a in i['ingredients']]
  readable=sum(a.get('observation_status')=='可计分' for a in rows);pending=sum(a.get('observation_status')=='待核' for a in rows)
  assert readable+pending==len(rows) and readable==r['observation_readable_rows'] and pending==r['observation_pending_rows']
  assert readable==r['base_rows']+r['conditional_rows']+r.get('guide_rows',0)
  legal={'应出表','疑似续页提示','留在原文的加减／转引','制服法','待核'}
  assert all(a.get('display_status') in legal for a in rows)
  assert all(i.get('display_status') in legal for i in p['instances'])
  output=sum(a['display_status']=='应出表' for a in rows)
  assert output==r['output_table_rows']
  total.update({'readable_observations':readable,'pending':pending,'output_table_rows':output,'instances':len(p['instances']),'pages':1})
  report.append({'page_id':p['id'],'readable_observations':readable,'pending':pending,'output_table_rows':output,'instances':len(p['instances']),'regions':len(p['coverage_regions'])})
expected=[['藥名','劑量','炮製'],['桂枝','三兩','去皮'],['芍藥','三兩','切'],['甘草','二兩','炙'],['生薑','三兩','切'],['大棗','十二枚','擘']]
for table in json.loads((root/'docs/正文方剂表_规整表本轮基线.json').read_text(encoding='utf8')):
 assert table['rows']==expected and table['model_loaded'] and not table['is_demo']
 assert sum(len(row) for row in table['cells'])==18
 for rr,cc in zip(table['rows'],table['cells']):
  for text,cell in zip(rr,cc):assert text==cell['text'] and cell['ref'] and len(cell['bbox'])==4
 assert table['cells'][1][1]['bbox']==[342,323,405,359]
assert total['pages']==5 and total['readable_observations']==53 and total['pending']==1 and total['output_table_rows']==25
print(json.dumps({'验收页':report,'合计':dict(total),'合成表基线':'2页，36格存在同文引用与坐标'},ensure_ascii=False,indent=2))
