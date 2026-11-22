import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step5';x=json.loads((O/'api_server_ab_v2.json').read_text(encoding='utf8'));a=json.loads((O/'analysis.json').read_text(encoding='utf8'));checks={}
for name in ('p35','p39','p34','p31','newbook','p0011','p0015'):
 left=x['0_'+name]['read'];right=x['1_'+name]['read'];checks[name+'_same_formula_rows']=[t['rows'] for t in left['tables']]==[t['rows'] for t in right['tables']]
 clean=lambda z:sorted((b['bbox'],b['text'],b['kind']) for b in z['blocks'] if b['kind']!='small_column_evidence')
 checks[name+'_same_original_ocr']=clean(left)==clean(right)
for name in ('p0011','p0015'):
 tables=x['1_'+name]['read']['tables'];checks[name+'_table_structure']=len(tables)==1 and len(tables[0]['rows'])==6 and all(len(r)==3 for r in tables[0]['rows']) and sum(bool(v) for row in tables[0]['cell_refs'] for v in row)==18
def find(model,page,box):
 return [b for b in x[model+'_'+page]['read']['blocks'] if b['bbox']==box and b['kind']=='small_column_evidence' and b['evidence_role']=='right']
for page,box,text in [('p39',[964,489,981,573],'各二錢'),('p34',[58,630,82,738],'一錢二分')]:
 v=find('1',page,box);checks[page+'_server_crop_traceable']=len(v)==1 and v[0]['text']==text and v[0]['evidence_engine']=='PP-OCRv5_server_rec' and v[0]['evidence_status']=='待核' and bool(v[0]['parent_block_ref'])
new=x['1_newbook']['read'];checks['newbook_target_still_fallback']=not [b for b in new['blocks'] if b['kind']=='small_column_evidence' and b['bbox'][0]<294 and b['bbox'][2]>240 and b['bbox'][1]<590 and b['bbox'][3]>353]
checks['fixed_no_new_confirmation']=a['mobile_attached_dose_rows']==a['selective_server_attached_dose_rows']==0
checks['fragment_improves_without_loss']=a['selective_server_hits']>a['selective_mobile_hits'] and all(not v['step4_mobile_fragment_hit'] or v['selective_server_fragment_hit'] for v in a['samples'])
result={'checks':checks,'all_pass':all(checks.values()),'fixed_11_mobile_fragment':a['selective_mobile_hits'],'fixed_11_server_fallback_fragment':a['selective_server_hits'],'fixed_11_strict_correct_formula_rows':0,'newbook_automatic_ze_xie':False,'newbook_automatic_zhu_ling':False,'model_source':'https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx/tree/main'}
(O/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=True,indent=2));assert result['all_pass']
