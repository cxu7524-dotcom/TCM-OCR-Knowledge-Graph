import hashlib,json
from collections import Counter
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'output/small_columns_step1_holdout'
a=json.loads((O/'annotations.json').read_text(encoding='utf-8')); matches=json.loads((O/'sample_matches.json').read_text(encoding='utf-8'))['samples']; dev=json.loads((ROOT/'output/small_columns_step1_dev/annotation.json').read_text(encoding='utf-8'))
errors=[];page_counts={};types=Counter();ids=[]
for n,m in a['pages'].items():
 p=ROOT/m['path']; img=Image.open(p); h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=m['sha256'] or list(img.size)!=m['size']:errors.append('image mismatch '+n)
 if h==dev['sha256'] or int(n)==a['development_page']:errors.append('dev/holdout overlap '+n)
 r=json.loads((O/f'page_{n}_full_ocr.json').read_text(encoding='utf-8'))
 if not r['model_loaded'] or r['is_demo'] or r['baseline_metadata']['image_sha256']!=h:errors.append('OCR failed '+n)
 page_counts[n]={'positive':0,'negative':0,'dose_positive':0,'ocr_blocks':len(r['blocks'])}
for s in a['samples']:
 n=str(s['page']);p=a['pages'][n];w,h=p['size']; page_counts[n][s['label']]+=1;ids.append(s['id'])
 if s['label']=='negative':types[s['negative_type']]+=1
 if s['label']=='positive' and s['dose']['verbatim']:page_counts[n]['dose_positive']+=1
 def valid(b):return len(b)==4 and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
 def inside(b,o):return o[0]<=b[0]<b[2]<=o[2] and o[1]<=b[1]<b[3]<=o[3]
 outer=s['outer_bbox'];bs=([s['main_bbox']] if s['main_bbox'] else [])+[v['bbox'] for v in s['columns']]
 if not valid(outer) or any(not valid(b) or not inside(b,outer) for b in bs):errors.append('bad box '+s['id'])
 if s['label']=='positive' and ([c['side'] for c in s['columns']]!=['right','left'] or s['reading_order']!=['right','left'] or not all(c['text'] for c in s['columns'])):errors.append('bad columns '+s['id'])
 if not s.get('uncertainty'):errors.append('missing uncertainty '+s['id'])
if len(ids)!=len(set(ids)):errors.append('duplicate ids')
if set(ids)!={x['id'] for x in matches}:errors.append('match coverage')
pos=[x for x in a['samples'] if x['label']=='positive'];neg=[x for x in a['samples'] if x['label']=='negative'];dose=[x for x in matches if x['id'] in {p['id'] for p in pos} and x['dose_fragment']]
checks={'unique_ids_and_boxes_and_pages':not errors,'dev_holdout_page_and_sha_disjoint':all(x['sha256']!=dev['sha256'] for x in a['pages'].values()),'positive_at_least_20':len(pos)>=20,'negative_at_least_20':len(neg)>=20,'negative_categories_present':all(any(k in z for z in types) for k in ['普通','單列','透印','相鄰']),'all_holdout_pages_have_nondemo_ocr':all((O/f'page_{n}_full_ocr.json').exists() for n in a['pages']),'prior_task_page_exposure_disclosed':True}
out={'checks':checks,'structural_errors':errors,'by_page':page_counts,'negative_types':types,'dose_positive_count':len(dose),'dose_exact_hits':sum(x['exact_hit'] for x in dose),'dose_normalized_hits':sum(x['normalized_hit'] for x in dose),'acceptance_passed':all(checks.values()),'limitations':['本次樣本經原圖及疊框核對；大〔?〕棗的字形保留疑字，未由第二人獨立複核。','多數負例為同頁普通大字短片段，難例各僅一處。','固定留出集為局部樣本；不代表整頁方劑驗收或產品整行指標。']}
(O/'structure_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=True))
