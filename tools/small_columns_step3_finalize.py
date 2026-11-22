import json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step3';data=json.loads((O/'holdout_first_review_screen.json').read_text(encoding='utf8'));freeze=json.loads((O/'freeze.json').read_text(encoding='utf8'))
assert hashlib.sha256((ROOT/'ocr/small_columns.py').read_bytes()).hexdigest()==freeze['algorithm_sha256']
judgments={
'P37-01':(True,'兩個子框字形完整，未見鄰列字，主字可追溯，順序正確'),
'P39-02':(False,'右列截斷：各二錢中的錢未包含'),
'P39-03':(False,'主字框截斷芍字且子框頂部侵入主字'),
'P39-04':(False,'自動外層列存在，目標位置未提出雙列'),
'P43-05':(False,'主字僅包含豆，右小字框向下帶入後續大字'),
'P46-06':(False,'主字框邊緣裁切；不能確認完整保留半夏'),
'P46-07':(False,'自動外層列存在，目標位置未提出雙列'),
'P46-08':(False,'主字框裁切麻黃邊緣，子框頂部與主字交疊'),
'P50-09':(False,'右列只覆蓋各一，漏掉錢半'),
'P34-10':(False,'右列框向下帶入下一大字，主字邊界亦不完整'),
'P34-11':(False,'主字僅有角字一部分，未保留羚羊角'),
'P49-12':(False,'右列漏掉皮字；主字只保留仁'),
'P49-13':(False,'自動外層列存在，未提出雙列'),
'P34-14':(True,'兩個子框目標字完整、未見鄰列字，主字可追溯且順序正確'),
'P43-15':(False,'主字只保留服，右列末字中有裁切'),
'P43-16':(False,'主字只保留下，右列末字痰有裁切'),
'P45-17':(False,'主字只保留錢，左右子框頂部筆畫也有裁切'),
'P45-18':(False,'自動外層列存在，未提出雙列'),
'P45-19':(False,'自動外層列存在，目標位置未提出雙列'),
'P49-20':(True,'兩個子框目標字完整、未見鄰列字，主字可追溯且順序正確')}
for r in data['samples']:
 if r['label']=='positive':
  passed,reason=judgments[r['id']];r['review']={'correct':passed,'reason':reason,'method':'人工查看原圖目標/預測叠框，逐項檢查內容完整、外層歸屬、鄰字及讀序；筆畫覆蓋只作輔助'}
 else:
  fp=r['id']=='N46-20';r['review']={'false_split':fp,'reason':'單列五分旁的透印被視為另一列' if fp else '標注反例範圍未見輸出分欄候選'}
pos=[r for r in data['samples'] if r['label']=='positive'];neg=[r for r in data['samples'] if r['label']=='negative'];pc=sum(r['review']['correct'] for r in pos);nf=sum(r['review']['false_split'] for r in neg)
report={'review_method':'原圖叠框人工逐項審閱；只讀取首次凍結預測。主字框裁切在本輪從屬關係要求下從嚴計失敗。','run_number':1,'positive_correct':pc,'positive_total':len(pos),'positive_threshold':'>=90%','negative_false_split':nf,'negative_total':len(neg),'negative_threshold':'<=5%','positive_no_proposal':sum(r['screen'].startswith('miss_') for r in pos),'acceptance_passed':pc/len(pos)>=.9 and nf/len(neg)<=.05,'holdout_retests':0,'after_first_review_development_only_trial':'放寬短窗口及主字行連通；開發頁誤拆增加且短夾注仍漏，已恢復首次凍結版，未作留出復測。','seen_holdout_pages':sorted({r['page'] for r in data['samples']}),'samples':data['samples']}
(O/'holdout_first_adjudicated.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=True))
