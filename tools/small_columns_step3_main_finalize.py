import hashlib,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step3_main';a=json.loads((O/'holdout_third_review_screen.json').read_text(encoding='utf8'));f=json.loads((O/'freeze_v3.json').read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'ocr/small_columns.py')==f['algorithm_sha256'];assert sha(ROOT/'output/small_columns_step1_holdout/annotations.json')==f['holdout_label_sha256']
reviews={
'P37-01':(True,[],'主字完整，兩列目標筆畫完整且同父列，讀序正確'),
'P39-02':(True,[],'主字及左右小字目視完整；右上少量差異為主字邊緣與目標框重疊'),
'P39-03':(True,[],'主字芍的下部已保留，兩列目標可見完整'),
'P39-04':(True,[],'主字已排除上一味錢字，土字與頁框可區分'),
'P43-05':(True,[],'巴豆完整，兩側目標可見完整，未見後文入框'),
'P46-06':(False,['clipped'],'為保留夏字而推遲子框起點，薑／製頂端筆畫被截去'),
'P46-07':(True,[],'主字及兩列可辨完整'),
'P46-08':(False,['main_association'],'黃字底端一條筆畫落在主字框外'),
'P50-09':(False,['main_association'],'芩字底端局部筆畫落在主字框外'),
'P34-10':(False,['main_association','clipped'],'主字框只保留前兩字，棗字漏出；末字邊緣亦有裁切疑點'),
'P34-11':(True,[],'羚羊角主字完整且已排除上一項張字，兩列可辨完整'),
'P49-12':(True,[],'杏仁和兩軌可辨完整'),
'P49-13':(False,['missing'],'自動外列存在但目標處無雙列候選'),
'P34-14':(True,[],'杏仁和兩軌完整'),
'P43-15':(False,['clipped'],'右列左緣裁掉方見中字形的部分筆畫'),
'P43-16':(True,[],'送下及兩軌完整；包含同句上文不妨礙關聯'),
'P45-17':(False,['main_association','clipped','neighbor_text'],'主字漏二，兩軌上緣裁字，右框向下帶入後續以字'),
'P45-18':(True,[],'法及方見／暑門完整；主字框包含同句相鄰文字'),
'P45-19':(False,['clipped'],'左列風字右下部被中縫截斷'),
'P49-20':(False,['main_association','neighbor_text'],'主字框跳到上方無關文字，未包含目標之字')}
for r in a['samples']:
 if r['label']=='positive':
  ok,cats,why=reviews[r['id']];r['review']={'correct':ok,'failure_types':cats,'reason':why,'method':'目視原圖及叠框；高對比像素只作輔助，含弱筆畫疑點從嚴'}
 else:
  bad=r['id']=='N46-19';r['review']={'false_split':bad,'failure_types':['false_split'] if bad else [],'reason':'白芷八分單列旁淡墨被誤判第二軌' if bad else '標注區域未見雙列預測'}
pos=[x for x in a['samples'] if x['label']=='positive'];neg=[x for x in a['samples'] if x['label']=='negative'];count=Counter(z for x in a['samples'] for z in x['review']['failure_types']);count['wrong_reading_order']=sum(x['selected_proposal'] is not None and x['selected_proposal']['reading_order']!=['main','right','left'] for x in pos);count['neighbor_column_ink']=0;correct=sum(x['review']['correct'] for x in pos);fp=sum(x['review']['false_split'] for x in neg)
result={'run_number':3,'independent_blind_holdout':False,'scope':'20 positive and 22 negative fixed local samples; categories may overlap','adjudication':'對原圖、目標與預測叠框逐項目視；先看主字歸屬、目標字完整、是否帶入鄰字和讀序，不以框交疊直接判對','positive_correct':correct,'positive_total':len(pos),'negative_false_split':fp,'negative_total':len(neg),'failure_counts':dict(count),'emitted_positive_with_order':sum(x['selected_proposal'] is not None for x in pos),'acceptance_passed':correct/len(pos)>=.9 and fp/len(neg)<=.05,'prior_results':[{'run':1,'positive_correct':3,'positive_total':20,'false_split':1,'negative_total':22},{'run':2,'positive_correct':7,'positive_total':20,'false_split':1,'negative_total':22}],'holdout_pages':[34,37,39,43,45,46,49,50],'samples':a['samples']}
(O/'holdout_third_adjudicated.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=True))
