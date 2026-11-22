import hashlib,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step3_retry';a=json.loads((O/'holdout_retest_review_screen.json').read_text(encoding='utf8'));f=json.loads((O/'freeze_retry.json').read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'ocr/small_columns.py')==f['algorithm_sha256'];assert sha(ROOT/'output/small_columns_step1_holdout/annotations.json')==f['holdout_label_sha256']
# Independent labels are permitted for post-inference review only. Categories overlap.
reviews={
'P37-01':(False,['main_association'],'主字末端少量筆畫落入子框，主字框提前結束'),
'P39-02':(True,[],'兩列字形完整；右框邊緣有少量灰點但目視未見目標筆畫缺失'),
'P39-03':(False,['main_association'],'主字芍的下部被截進兩個小字框'),
'P39-04':(False,['main_association','neighbor_text'],'主字框上方帶入上一味的錢字；右框底部的灰線位於頁框附近'),
'P43-05':(False,['main_association'],'巴字頂緣有少量深色筆畫落在主字框之外，存疑從嚴計失敗'),
'P46-06':(False,['main_association'],'夏字底部筆畫被主字框截斷並落入子框頂部'),
'P46-07':(True,[],'主字和左右小字均完整，同屬自動列且順序正確'),
'P46-08':(False,['main_association'],'黃字末筆落入小字框頂部，主字框未完整保留'),
'P50-09':(True,[],'主字及兩軌完整，右側邊緣的零星點未見構成缺字'),
'P34-10':(False,['main_association','clipped'],'棗字底部被切入小字框，右左兩列末字邊緣亦有裁切疑點'),
'P34-11':(False,['main_association','neighbor_text'],'主字框包含上一項的張且羚羊角底部與子框交疊'),
'P49-12':(False,['main_association'],'杏字頂緣有少量深色筆畫在主字框之外，存疑從嚴計失敗'),
'P49-13':(False,['missing'],'有橫向相應自動列但未提出此雙列'),
'P34-14':(True,[],'主字和兩列可辨完整，與首輪人工審閱口徑一致'),
'P43-15':(False,['main_association','clipped'],'服字底部與子框交疊；右列中有可見邊緣裁切'),
'P43-16':(True,[],'主字及方見痰／門完整，附帶同列前文不影響主字歸屬'),
'P45-17':(False,['main_association','neighbor_text'],'主字漏二且右列向下帶入後續大字以'),
'P45-18':(True,[],'法及方見／暑門可辨完整，讀序正確'),
'P45-19':(False,['clipped'],'左列風字右下部筆畫被分界截斷'),
'P49-20':(True,[],'主字及方見中／暑門可辨完整，讀序正確')}
for r in a['samples']:
 if r['label']=='positive':
  ok,cats,why=reviews[r['id']];r['review']={'correct':ok,'failure_types':cats,'reason':why,'method':'目視原圖疊框，像素覆蓋與高對比缺失筆畫僅供輔助；存疑從嚴'}
 else:
  bad=r['id']=='N46-19';r['review']={'false_split':bad,'failure_types':['false_split'] if bad else [],'reason':'白芷八分為單列，旁邊淡墨被誤判為第二列' if bad else '反例範圍內沒有雙列預測'}
pos=[x for x in a['samples'] if x['label']=='positive'];neg=[x for x in a['samples'] if x['label']=='negative'];counts=Counter(z for x in a['samples'] for z in x['review']['failure_types']);success=sum(x['review']['correct'] for x in pos);fp=sum(x['review']['false_split'] for x in neg)
order_wrong=sum(x['selected_proposal'] is not None and x['selected_proposal']['reading_order']!=['main','right','left'] for x in pos);counts['wrong_reading_order']=order_wrong;counts['neighbor_column_ink']=0
result={'run_number':2,'independent_blind_holdout':False,'scope':'20 positive and 22 negative fixed local samples; categories overlap','adjudication':'逐圖人工審閱原圖、目標與預測疊框；字形無法確認完整時從嚴計失敗，不把空白裁框重疊率視為成功','positive_correct':success,'positive_total':len(pos),'negative_false_split':fp,'negative_total':len(neg),'failure_counts':dict(counts),'emitted_positive_with_order':sum(x['selected_proposal'] is not None for x in pos),'acceptance_passed':success/len(pos)>=.9 and fp/len(neg)<=.05,'first_run_positive':'3/20','first_run_false_positive':'1/22','holdout_pages':[34,37,39,43,45,46,49,50],'samples':a['samples']}
(O/'holdout_retest_adjudicated.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=True))
