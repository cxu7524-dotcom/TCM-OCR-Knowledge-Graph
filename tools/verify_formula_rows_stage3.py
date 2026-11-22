"""步骤 3 开发样例复核；输入含人工转写和既有 OCR，绝不读取验收页。"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.schemas import Block
from ocr.normalize import normalize_text
from extract.formula_segments import _Stream, extract_formula_candidates
from extract.formula_rows import extract_formula_rows


def manual(page, entries):
    return [Block(page_id=page,line_id=f'M{n:02}',block_ref=f'{page}:M{n:02}',reading_order=n,
                  text=text,text_norm=normalize_text(text),bbox=box,conf=1.0)
            for n,(text,box) in enumerate(entries,1)]

def verify_sources(blocks, result):
    byref={b.block_ref:b for b in blocks}; count=0
    for formula in result['formulas']:
        fields=[formula['name_field']['source']]
        fields.extend(x[k]['source'] for x in formula['rows'] for k in ('herb','dose'))
        for row in formula['rows']:
            dose=row['dose']
            fields.append(dose.get('evidence_context'))
            proof=dose.get('absence_evidence') or {}
            fields.extend([proof.get('start_cue'),proof.get('terminator')])
        for field in fields:
            if field is None:continue
            for src in field['sources']:
                b=byref[src['block_ref']];a,z=src['block_raw_range']
                assert b.text[a:z]==src['text'] and list(b.bbox)==src['bbox']
                count+=1
    return count

ann=json.loads((ROOT/'docs/正文方剂表_步骤1标注.json').read_text(encoding='utf8'))
entries=[('三化湯',[795,213,835,311]),('羌活枳實厚朴',[650,240,687,635]),
         ('川大黃各三錢芒硝一錢半',[600,236,645,552]),('水三鍾煎八分溫服',[552,238,592,735]),
         ('六君子湯',[327,213,365,320]),('人參二錢白朮五錢正白苓二錢',[135,240,175,735]),
         ('甘草廣皮製半夏各一錢',[85,240,125,735]),('引加生薑一錢半紅棗肉三枚水煎溫服',[35,240,75,735])]
b35=manual('思濟堂方書-0035',entries)
r35=extract_formula_rows(b35,verified_complete=True)
pairs=lambda g:[(r['herb']['value'],r['dose']['value']) for r in g['rows']]
assert [g['name_field']['value'] for g in r35['formulas']]==['三化湯','六君子湯']
assert pairs(r35['formulas'][0])==[('羌活','三錢'),('枳實','三錢'),('厚朴','三錢'),('川大黃','三錢'),('芒硝','一錢半')]
assert pairs(r35['formulas'][1])==[('人參','二錢'),('白朮','五錢'),('正白苓','二錢'),('甘草','一錢'),('廣皮','一錢'),('製半夏','一錢')]
for x in r35['formulas'][0]['rows'][:4]:
 assert x['dose']['source']['text']=='各三錢' and x['dose']['source']['sources'][0]['block_ref'].endswith('M03')
assert r35['formulas'][0]['rows'][0]['herb']['source']['sources'][0]['block_ref'].endswith('M02')
b33=manual('思濟堂方書-0033',[('稀涎散',[235,215,275,307]),('白礬生用牙皂角',[180,240,230,500]),('共為細末每服二錢',[130,240,180,735])])
r33=extract_formula_rows(b33,verified_complete=True)
r33_default=extract_formula_rows(b33)
assert pairs(r33_default['formulas'][0])==[('白礬','原文未载'),('牙皂角','原文未载')]
assert all(r['dose']['absence_basis']=='自动结构判定' for r in r33_default['formulas'][0]['rows'])
assert pairs(r33['formulas'][0])==[('白礬','原文未载'),('牙皂角','原文未载')]
pg=next(p for p in ann['pages'] if p['id']=='丹方彚编-0023');item=next(i for i in pg['instances'] if i['id']=='D23-1')
b23=manual(pg['id'],[(item['name_verbatim'],item['name_bbox']),(item['quote'],item['quote_bbox']),('右為細末',[300,200,350,750])])
st=_Stream(b23);a=st.parts[1]['start'];b=st.parts[1]['end']
c={'page_id':pg['id'],'name_status':'有名方','name':st.source(0,a),'start_cue':st.source(0,a),'composition':st.source(a,b),'passage':st.source(0,b),'end_reason':'制法服法','continuation':'未见续页迹象'}
r23=extract_formula_rows(b23,[c],verified_complete=True)
assert pairs(r23['formulas'][0])==[('真鴉片','待核'),('石草霜','九錢'),('西牛黃','待核'),('麝香','待核')]
ocr=[];o_blocks=[]
for n in (33,35):
 d=json.loads((ROOT/f'output/qa_verify3/pages/思濟堂方書__page_{n}.json').read_text(encoding='utf8'))
 bs=[Block.from_dict(x) for x in d['blocks']];o_blocks.append(bs);ocr.append(extract_formula_rows(bs))
assert all(r['dose']['value']=='待核' for r in ocr[0]['formulas'][0]['rows'])
source_count=sum(verify_sources(bs,r) for bs,r in [(b35,r35),(b33,r33),(b33,r33_default),(b23,r23),*zip(o_blocks,ocr)])
out={'manual_35':r35,'manual_33':r33,'manual_33_default':r33_default,'manual_23_seeded_candidate':r23,'existing_ocr':ocr,'source_spans_checked':source_count,
     'notes':'人工转写经图片和步骤1标注核对，块位置为原图区域；丹方23因步骤2未识别题名，单独使用人工划定配伍范围；OCR为既有记录。'}
(ROOT/'docs/正文方剂表_步骤3开发样例结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
for label,g in [('manual35',r35),('manual33',r33),('manual33_default',r33_default),('manual23',r23),('ocr33',ocr[0]),('ocr35',ocr[1])]:
 print(label,[(f['name_field']['value'],pairs(f)) for f in g['formulas']])
print('source_spans_checked',source_count)
