import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step3_detector';H=ROOT/'output/small_columns_step3_hybrid'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((O/'freeze_v5.json').read_text(encoding='utf8'))
for path,digest in f['files_sha256'].items():assert sha(ROOT/path)==digest,path
assert sha(f['model']['path'])==f['model']['sha256']
p=json.loads((O/'holdout_fifth_predictions.json').read_text(encoding='utf8'));assert p['freeze_sha256']==sha(O/'freeze_v5.json')
for page in p['pages'].values():assert sha(ROOT/page['source'])==page['image_sha256']
rows=json.loads((O/'holdout_fifth_review_screen.json').read_text(encoding='utf8'))['samples'];old=json.loads((H/'holdout_fourth_adjudicated.json').read_text(encoding='utf8'));prior={r['id']:r for r in old['samples']}
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
for r in rows:
 prev=prior[r['id']];a=prev['selected_proposal'];b=r['selected_proposal']
 if r['id']=='P46-06':
  assert a is None and b is not None
  r['review']={'correct':False,'failure_types':['clipped','local_ocr_unreadable'],'reason':'恢復薑／製兩列候選，右側原始識別仍為空；為保留夏字而延後子框起點，兩列頂端筆畫被截，沿用第三輪嚴格判定。','reading_order':b['reading_order'],'method':'目視原圖與叠框；識別可讀性單獨記錄，不作幾何存在性判斷。'}
  r['comparison_to_prior']='第四輪因識別為空回退；第五輪有候選但兩側頂部裁字，仍不合格。'
 elif r['label']=='positive':
  assert (a is None and b is None) or (a is not None and b is not None and a['main_bbox']==b['main_bbox'] and a['columns']==b['columns']),r['id']
  r['review']=prev['review'];r['comparison_to_prior']='第五輪主字及兩列邊界與第四輪一致，保留原目視判定；含標注疑點不改判。'
 else:
  assert prev['screen']==r['screen'],r['id'];r['review']=prev['review'];r['comparison_to_prior']='第四輪與第五輪對該負例的輸出狀態一致。'
 if r['label']=='positive':
  gt={c['side']:c['bbox'] for c in r['target_columns']};relevant=[]
  for ev in p['pages'][str(r['page'])]['audit']:
   if 'geometric_columns' not in ev:continue
   g={c['side']:c['bbox'] for c in ev['geometric_columns']}
   if all(overlap(g[k],gt[k])>0 for k in ('right','left')):relevant.append(ev)
  r['target_matching_candidate_audit']=relevant
pos=[r for r in rows if r['label']=='positive'];neg=[r for r in rows if r['label']=='negative'];cnt=Counter(k for r in rows for k in r['review']['failure_types']);cnt['wrong_reading_order']=sum(r['selected_proposal'] is not None and r['selected_proposal']['reading_order']!=['main','right','left'] for r in pos);correct=sum(r['review']['correct'] for r in pos);fp=sum(r['review']['false_split'] for r in neg)
out={'run_number':5,'independent_blind_holdout':False,'freeze_sha256':sha(O/'freeze_v5.json'),'predictions_sha256':sha(O/'holdout_fifth_predictions.json'),'uses_prior_automatic_ocr_and_columns':True,'method':'依原圖目視主字、左右筆畫及污染；相同邊界保留第四輪原判和標注疑點，新增 P46-06 沿用第三輪裁字判定；分類可重疊。','positive_correct':correct,'positive_total':len(pos),'positive_with_matched_candidate':sum(r['selected_proposal'] is not None for r in pos),'negative_false_split':fp,'negative_total':len(neg),'failure_counts':dict(cnt),'acceptance_passed':correct/len(pos)>=.9 and fp/len(neg)<=.05,'prior_runs':[{'run':1,'positive_correct':3,'negative_false_split':1},{'run':2,'positive_correct':7,'negative_false_split':1},{'run':3,'positive_correct':11,'negative_false_split':1},{'run':4,'positive_correct':11,'negative_false_split':1}],'samples':rows}
target=O/'holdout_fifth_adjudicated.json';assert not target.exists();target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='samples'},ensure_ascii=True,indent=2))
