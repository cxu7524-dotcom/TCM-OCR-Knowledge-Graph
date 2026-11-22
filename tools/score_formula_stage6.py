"""冻结步骤1分母，按方名、药名、剂量及各自出处一对一计算步骤6指标。"""
import json,re,sys,hashlib,argparse
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.normalize import normalize_text
parser=argparse.ArgumentParser()
parser.add_argument('--input-dir',type=Path,default=ROOT/'docs')
parser.add_argument('--output',type=Path,default=ROOT/'docs/正文方剂表_步骤6逐行评测.json')
args=parser.parse_args()
ann_path=args.input_dir/'正文方剂表_步骤1标注.json'
ann=json.loads(ann_path.read_text(encoding='utf8'));browser=json.loads((args.input_dir/'正文方剂表_步骤6网页原始结果.json').read_text(encoding='utf8'));manual=json.loads((args.input_dir/'正文方剂表_步骤6人工转写结果.json').read_text(encoding='utf8'))
pages=[p for p in ann['pages'] if p.get('split')=='验收']; assert [p['page_no'] for p in pages]==[37,39,40,41,50]
def norm(s):return re.sub(r'\s+','',normalize_text(str(s or '')))
def dose_norm(s):return re.sub(r'^各','',norm(s))
def overlap(a,b):
 if not a or not b or len(a)!=4 or len(b)!=4:return False
 return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
def field_sources(field,byref):
 src=(field.get('source') or {}).get('sources') or [];out=[]
 for s in src:
  block=byref.get(s.get('block_ref'));span=s.get('block_raw_range')
  if not block or not isinstance(span,list) or len(span)!=2:return []
  a,b=span
  if not (isinstance(a,int) and isinstance(b,int) and 0<=a<=b<=len(block['text'])):return []
  if block['text'][a:b]!=s.get('text') or block['bbox']!=s.get('bbox'):return []
  out.append(s)
 return out
def pass_match(gt,pred,byref,strict=True):
 for key in ('name','herb','dose'):
  f=pred[key]
  if f.get('status')!='已识别':return False
  if (dose_norm if key=='dose' else norm)(f.get('value')) != (dose_norm if key=='dose' else norm)(gt[key]):return False
  if (strict or key=='name') and not any(overlap(src['bbox'],gt[key+'_bbox']) for src in field_sources(f,byref)):return False
 return True
def get_gt(page):
 out=[]
 for i in page['instances']:
  if i.get('display_status')!='应出表':continue
  for k,h in enumerate(i['ingredients']):
   if h.get('display_status')!='应出表':continue
   out.append({'id':f"{i['id']}:{k+1}",'instance':i['id'],'name':i['name_verbatim'],'name_bbox':i['name_bbox'],
               'herb':h['herb_verbatim'],'herb_bbox':h['herb_bbox'],'dose':h['dose_verbatim'],'dose_bbox':h['dose_bbox']})
 return out
def predictions_from_manual(data):
 return [{'name':g['name_field'],'herb':r['herb'],'dose':r['dose'],'instance_id':f'manual-{i+1}','row':j+1,'context':g['composition']['text']}
         for i,g in enumerate(data['extraction']['formulas']) for j,r in enumerate(g['rows'])]
def predictions_from_browser(entry):
 out=[]
 for t in entry.get('api',{}).get('tables',[]):
  if t.get('source_type')!='正文抽取':continue
  for j,fields in enumerate((t.get('field_details') or [])[1:],1):
   if len(fields)>=3:out.append({'name':fields[0],'herb':fields[1],'dose':fields[2],'instance_id':t['table_id'],'row':j+1,'context':(fields[1].get('source') or {}).get('text','')})
 return out
def score(gt,preds,blocks,strict=True):
 byref={b['block_ref']:b for b in blocks};edges=[[j for j,p in enumerate(preds) if pass_match(g,p,byref,strict)] for g in gt]
 # 最大二分匹配；重复输出只能匹配一次，其余均计误出。
 owner={}
 def visit(i,seen):
  for k in edges[i]:
   if k in seen:continue
   seen.add(k)
   if k not in owner or visit(owner[k],seen):owner[k]=i;return True
  return False
 for i in range(len(gt)):visit(i,set())
 matched={i:k for k,i in owner.items()}
 ledger=[dict(g,matched=i in matched,prediction_index=matched.get(i)) for i,g in enumerate(gt)]
 fp=[dict(p,prediction_index=k) for k,p in enumerate(preds) if k not in owner]
 return {'gt_count':len(gt),'prediction_count':len(preds),'hits':len(owner),'false_positive_count':len(fp),
         'false_negative_count':len(gt)-len(owner),'ledger':ledger,'false_positives':fp,'predictions':preds}
result={'annotation_sha256':hashlib.sha256(ann_path.read_bytes()).hexdigest(),'scope':'步骤1固定五页、display_status=应出表；正文抽取的所有数据行均计入出表分母。',
        'matching':'方名、药材、剂量三字段状态已识别且归一化后相符，各字段有效原文块 bbox 与对应标注 bbox 相交；最大一对一匹配。',
        'pages':{},'notes':'原书表格类型单列记录，非正文药材数据行；固定五页实际未输出该类型。'}
for page in pages:
 n=page['page_no'];gt=get_gt(page); entry=browser['pages'][str(n)]
 if entry.get('upload')!='成功':
  m=score(gt,[],[]);o=score(gt,[],[])
 else:
  d=manual[str(n)];m=score(gt,predictions_from_manual(d),d['blocks']);o=score(gt,predictions_from_browser(entry),entry['api']['blocks'])
 mp=score(gt,predictions_from_manual(manual[str(n)]),manual[str(n)]['blocks'],False)
 op=score(gt,predictions_from_browser(entry),entry.get('api',{}).get('blocks',[]),False)
 # 主因先按人工转写能否匹配分类；相当于对OCR影响的对照，不等同严格因果实验。
 for k,record in enumerate(o['ledger']):
  if record['matched']: record['primary_reason']='命中';continue
  if m['ledger'][k]['matched']:record['primary_reason']='真实OCR或阅读顺序引入的变化'
  else:
   name=record['name']; manual_predictions=predictions_from_manual(manual[str(n)])
   present=any(norm(p['name'].get('value'))==norm(name) for p in manual_predictions)
   record['primary_reason']='人工转写已有药量抽取或出处匹配失败' if present else '人工转写已有方剂切段或方名失败'
 result['pages'][str(n)]={'ground_truth':gt,'manual':m,'ocr':o,'manual_primary':mp,'ocr_primary':op,
                          'printed_tables':sum(t.get('source_type')=='原书表格' for t in entry.get('api',{}).get('tables',[])),
                          'click_count':len(entry.get('clicks',[])),
                          'click_located':sum(bool(c.get('blocks')) for c in entry.get('clicks',[])),
                          'hints':len(entry.get('api',{}).get('formula_continuation_hints',[]))}
for mode in ('manual','ocr','manual_primary','ocr_primary'):
 total=Counter()
 for p in result['pages'].values():
  for k in ('gt_count','prediction_count','hits','false_positive_count','false_negative_count'):total[k]+=p[mode][k]
 total['precision']=total['hits']/total['prediction_count'] if total['prediction_count'] else None
 total['recall']=total['hits']/total['gt_count'] if total['gt_count'] else None
 result[mode+'_total']=dict(total)
result['ocr_missing_primary_reasons']=dict(Counter(x['primary_reason'] for p in result['pages'].values() for x in p['ocr']['ledger'] if not x['matched']))
result['browser_source_checks'] = {'checked': 0, 'matched': 0, 'mismatches': []}
for page_no, entry in browser['pages'].items():
 for click in entry.get('clicks', []):
  if click.get('error'):
   result['browser_source_checks']['mismatches'].append({'page': page_no, 'error': click['error']})
   continue
  table = entry['api']['tables'][click['table']]
  row = 1 if click['field'] == '方名' else click['row']
  col = 0 if click['field'] == '方名' else click['column']
  expected = set(table['cell_refs_multi'][row][col])
  actual = {part['ref'] for part in click['blocks']}
  result['browser_source_checks']['checked'] += 1
  if expected == actual:
   result['browser_source_checks']['matched'] += 1
  else:
   result['browser_source_checks']['mismatches'].append({'page': page_no, 'field': click['field'],
                                                          'expected': sorted(expected), 'actual': sorted(actual)})
result['printed_regression'] = {k: {'clicked_R006': v.get('pass', False),
                                    'upload': v.get('upload')} for k, v in browser['printed'].items()}
assert result['ocr_total']['gt_count']==25
result['primary_matching']='方名出处有效且与同一标注实例方名框相交，方名、药名、剂量均已识别且归一化相符；最大一对一匹配。'
result['acceptance']={'precision_threshold':0.90,'recall_threshold':0.80,'passed':bool(result['ocr_primary_total']['precision'] is not None and result['ocr_primary_total']['precision']>=.9 and result['ocr_primary_total']['recall']>=.8)}
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for mode in ('manual','ocr'):print(mode,result[mode+'_total'])
print('primary reasons',result['ocr_missing_primary_reasons'],'pass',result['acceptance']['passed'])
for n,p in result['pages'].items(): print(n,'gt',p['ocr']['gt_count'],'pred',p['ocr']['prediction_count'],'hits',p['ocr']['hits'])
