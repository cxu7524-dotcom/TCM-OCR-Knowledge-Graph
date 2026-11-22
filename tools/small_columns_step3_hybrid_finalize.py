import hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'output/small_columns_step3_hybrid'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((O/'freeze_v4.json').read_text(encoding='utf8'))
for path,digest in f['files_sha256'].items(): assert sha(ROOT/path)==digest,path
assert sha(f['model']['path'])==f['model']['sha256']
p=json.loads((O/'holdout_fourth_predictions.json').read_text(encoding='utf8'))
assert p['freeze_sha256']==sha(O/'freeze_v4.json')
for x in p['pages'].values(): assert sha(ROOT/x['source'])==x['image_sha256']
rows=json.loads((O/'holdout_fourth_review_screen.json').read_text(encoding='utf8'))['samples']
# The judgment is based on inspecting the original image with the reference and prediction overlays.
reviews={
'P37-01':(True,[], '主字及右二錢半、左酒炒均完整；順序主、右、左。'),
'P39-02':(True,[], '川附子及右各二錢、左製可辨完整；順序正確。'),
'P39-03':(True,[], '主字及右各三錢、左酒炒可辨完整；順序正確。'),
'P39-04':(True,[], '白术及右四錢黃土、左拌炒可辨完整；順序正確。'),
'P43-05':(True,[], '巴豆及右五錢去皮、左心研如脂可辨完整；順序正確。'),
'P46-06':(False,['missing','local_ocr_unreadable'], '右側原始裁塊識別為空，觸發回退；薑／製未輸出，不能驗收讀序。'),
'P46-07':(True,[], '蒼术及右各一錢、左炒可辨完整；順序正確。'),
'P46-08':(False,['main_association'], '右八分若頭不、左疼無汗不用均有候選；主字黃的底部筆畫落框外，讀序字段正確。'),
'P50-09':(False,['main_association'], '右各一錢半、左酒炒均有候選；主字芩的底端局部落框外，讀序字段正確。'),
'P34-10':(False,['missing','main_association'], '大膠棗下的兩列未輸出；幾何候選主字漏棗，間隔26像素觸發回退，不能驗收讀序。'),
'P34-11':(True,[], '羚羊角及右一錢二分、左用尖銼可辨完整；順序正確。'),
'P49-12':(True,[], '杏仁及右七枚去皮、左尖研可辨完整；順序正確。'),
'P49-13':(False,['missing'], '目標兩列均未輸出；沒有重疊目標的幾何候選，不能驗收主字和讀序。'),
'P34-14':(True,[], '杏仁及右去皮尖、左別研可辨完整；順序正確。'),
'P43-15':(False,['clipped'], '主字可關聯，左列可辨；右列左緣裁掉方見中字形部分筆畫，讀序字段正確。'),
'P43-16':(True,[], '送下及右方見痰、左門可辨完整；同句上文在主字框中，讀序正確。'),
'P45-17':(False,['main_association','clipped','neighbor_text'], '主字漏二；左右子列上緣裁字，右列帶入後續以字；讀序字段正確。'),
'P45-18':(True,[], '法及右方見、左暑門完整，主字框含同句相鄰文字；讀序正確。'),
'P45-19':(False,['clipped'], '主字及右方見可辨；左列風字右下被中縫截斷，讀序字段正確。'),
'P49-20':(False,['missing','main_association'], '目標兩列未輸出；原候選跳過80像素正文而被回退，不能驗收讀序。')}
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
for r in rows:
 if r['label']=='positive':
  ok,cats,why=reviews[r['id']]
  r['review']={'correct':ok,'failure_types':cats,'reason':why,'reading_order':r['selected_proposal']['reading_order'] if r['selected_proposal'] else None,'method':'原圖目視標注和候選叠框；像素覆蓋僅作輔助，疑點從嚴'}
  gt={c['side']:c['bbox'] for c in r['target_columns']}
  relevant=[]
  for ev in p['pages'][str(r['page'])]['audit']:
   g={c['side']:c['bbox'] for c in ev['geometric_columns']}
   if all(overlap(g[k],gt[k])>0 for k in ('right','left')): relevant.append(ev)
  r['target_matching_hybrid_audit']=relevant
 else:
  bad=r['screen']=='candidate_false_split_review'
  assert not bad or r['id']=='N46-19'
  r['review']={'false_split':bad,'failure_types':['false_split'] if bad else [],'reason':'白芷八分單列及旁側淡墨被當作雙列；左側僅極窄碎片。' if bad else '標注負例區域未輸出雙列候選；原圖確認無誤拆。'}
pos=[r for r in rows if r['label']=='positive'];neg=[r for r in rows if r['label']=='negative']
counts=Counter(k for r in rows for k in r['review']['failure_types'])
counts['wrong_reading_order']=sum(r['selected_proposal'] is not None and r['selected_proposal']['reading_order']!=['main','right','left'] for r in pos)
counts['neighbor_column_ink']=0
correct=sum(r['review']['correct'] for r in pos);fp=sum(r['review']['false_split'] for r in neg)
result={'run_number':4,'independent_blind_holdout':False,'freeze_sha256':sha(O/'freeze_v4.json'),'predictions_sha256':sha(O/'holdout_fourth_predictions.json'),'method':'逐項目視原圖和叠框：主字歸屬、雙列完整性、鄰字污染和讀序；無候選計漏拆；類別可重疊。','positive_correct':correct,'positive_total':len(pos),'positive_with_matched_candidate':sum(r['selected_proposal'] is not None for r in pos),'negative_false_split':fp,'negative_total':len(neg),'failure_counts':dict(counts),'acceptance_passed':correct/len(pos)>=0.9 and fp/len(neg)<=0.05,'previous_run':{'positive_correct':11,'positive_total':20,'negative_false_split':1,'negative_total':22},'samples':rows}
target=O/'holdout_fourth_adjudicated.json'
assert not target.exists()
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False,indent=2))
