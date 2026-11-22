"""步骤 6 人工转写输入；不改变系统或标注。"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.schemas import Block
from ocr.normalize import normalize_text
from extract.formula_segments import extract_formula_candidates
from extract.formula_rows import extract_formula_rows

j=json.loads((ROOT/'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
P={p['page_no']:p for p in j['pages'] if p.get('split')=='验收'}
I={i['id']:i for p in P.values() for i in p['instances']}
# 基于步骤1已核对原文及所在列构造输入块；仅使用本页文字，不把邻页题名搬入。
S={
37:[(I['S37-0']['quote'],[690,220,990,740]),('四物湯',[650,212,687,310]),
 ('當歸五錢揀芍二錢半大熟地三錢',[595,220,645,720]),('川芎一錢半',[545,220,595,400]),('不加引水煎溫服',[425,230,480,735])],
39:[('人參川附子製各二錢白朮四錢黃土拌炒',[945,225,990,735]),('白茯苓揀白芍酒炒各三錢',[895,225,940,735]),('水四鍾煎一鍾大溫服',[840,225,890,735]),
 ('六味地黃湯',[375,215,415,355]),('大熟地自六錢可以加至一二三兩',[130,220,180,740]),('淮山藥自三錢可以加至五七九錢',[80,220,130,740]),('山萸肉自二錢半可以加至五七九錢',[30,220,80,740])],
40:[('正白茯苓自一錢八分可以加至五七錢',[945,220,990,735]),('牡丹皮自二錢可以加至四錢',[895,220,940,735]),('澤瀉一錢二分',[845,220,890,735]),('不加引水煎溫服',[790,220,840,735]),('景岳張氏曰腎虛不能藏精',[740,220,790,735])],
41:[('理中湯',[850,212,885,305]),('治中氣不運腹中不實口失滋味',[795,230,835,735]),('人參白朮土炒干薑各二錢',[695,220,745,735]),('炙草一錢半',[645,220,695,390]),('水煎熱服',[595,225,645,735]),
 ('附子理中湯',[550,212,590,355]),('即前方加川附子製一錢半',[435,240,480,555]),('生脈飲',[135,212,170,305]),('治熱傷元氣氣短倦怠口乾出汗脈虛小無力者',[35,240,135,735])],
50:[('桂枝湯',[945,212,985,305]),('桂枝芍藥生薑各三錢',[695,220,745,735]),('甘草二錢大棗二枚',[645,220,695,540]),('右五味水煎溫服',[595,225,645,735]),
 ('九味羌活湯',[435,212,475,355]),('羌活三錢防風黃芩各一錢半',[330,220,380,755]),('白芷川芎各一錢蒼朮',[280,220,330,650]),('生地各一錢半細辛甘草各五分',[230,220,280,720]),('引加生薑二錢蔥白二寸水煎熱服',[175,225,225,735]),
 ('衝和化滯湯',[135,212,175,355]),('治感寒傷食者效',[75,210,125,735]),('即九味羌活湯加神麴麥芽山查各二錢',[25,225,75,735])]
}
out={}
for n,entries in S.items():
 pid=f'MANUAL-S{n}'
 blocks=[Block(page_id=pid,line_id=f'M{k:03}',block_ref=f'{pid}:M{k:03}',reading_order=k,col_id=k,text=t,text_norm=normalize_text(t),bbox=bb,conf=1.) for k,(t,bb) in enumerate(entries,1)]
 seg=extract_formula_candidates(blocks);rows=extract_formula_rows(blocks,seg['candidates'],verified_complete=True)
 out[str(n)]={'input_type':'步骤1原图人工转写，按列构造Block，非OCR','blocks':[b.to_dict() for b in blocks], 'segments':seg,'extraction':rows}
 print(n,[(g['name_field']['value'],[(r['herb']['value'],r['dose']['value']) for r in g['rows']]) for g in rows['formulas']], 'hints',len(seg['continuation_hints']))
(ROOT/'docs/正文方剂表_步骤6人工转写结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
