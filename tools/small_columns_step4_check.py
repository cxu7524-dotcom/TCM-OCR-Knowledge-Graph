"""Validate captured real HTTP A/B records without rerunning inference."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step4'
x=json.loads((O/'api_ab_results_v3.json').read_text(encoding='utf8'));ab=json.loads((O/'holdout_ab_v2.json').read_text(encoding='utf8'))
assert (ab['baseline_hits'],ab['enabled_hits'],ab['denominator'])==(0,6,11)
def main_blocks(read):return sorted((b['bbox'],b['text'],b['kind'],b['conf']) for b in read['blocks'] if b['kind']!='small_column_evidence')
checks={}
for name in ('p35','newbook','p0011','p0015','p31'):
 off=x['0_'+name]['read'];on=x['1_'+name]['read'];checks[name+'_original_blocks_preserved']=main_blocks(off)==main_blocks(on)
on=x['1_p35']['read'];off=x['0_p35']['read'];ons=[t for t in on['tables'] if t['source_type']=='正文抽取'];offs=[t for t in off['tables'] if t['source_type']=='正文抽取'];assert len(ons)==len(offs)==2
checks['p35_no_extra_or_removed_rows']=all(len(a['rows'])==len(b['rows']) for a,b in zip(ons,offs))
found=[(t,i,r) for t in ons for i,r in enumerate(t['rows']) if r[1:]==['白术','五錢']];assert len(found)==1;t,i,row=found[0];dose=t['field_details'][i][2];source=dose['source']['sources'][0];byref={b['block_ref']:b for b in on['blocks']};child=byref[source['block_ref']]
checks['p35_dose_right_crop_resolves']=(source['bbox']==[151,460,177,570] and child['bbox']==source['bbox'] and child['evidence_role']=='right' and child['text'].startswith('五錢') and bool(child['parent_block_ref']))
ctx=dose['processing_context'];checks['p35_processing_preserved_pending']=ctx['status']=='待核' and [(v['text'],v['status']) for v in ctx['fragments']]==[('黃土','待核'),('排炒','待核')]
checks['p35_other_rows_unchanged']=(ons[0]['rows']==offs[0]['rows'] and all(a==b for j,(a,b) in enumerate(zip(ons[1]['rows'],offs[1]['rows'])) if j!=i))
for name in ('p0011','p0015'):
 a=x['0_'+name]['read']['tables'];b=x['1_'+name]['read']['tables'];checks[name+'_same_cells_and_locatable']=len(a)==len(b)==1 and a[0]['rows']==b[0]['rows'] and len(b[0]['rows'])==6 and all(len(r)==3 for r in b[0]['rows']) and sum(bool(ref) for r in b[0]['cell_refs'] for ref in r)==18
for name in ('newbook','p31'):
 a=x['0_'+name]['read'];b=x['1_'+name]['read'];checks[name+'_formula_rows_unchanged']=[t['rows'] for t in a['tables']]==[t['rows'] for t in b['tables']]
new=x['1_newbook']['read'];target=[b for b in new['blocks'] if b['kind']=='small_column_evidence' and b['bbox'][0]<294 and b['bbox'][2]>240 and b['bbox'][1]<590 and b['bbox'][3]>353]
checks['newbook_two_target_regions_fallback']=not target and any(b['line_id']=='R021' for b in new['blocks'])
checks['repeat_same_page']=x['repeat_same_page']['equal_tables'];checks['sqlite_restore']=x['sqlite_restore']['equal_tables'] and any(b['block_ref']==child['block_ref'] and b['evidence_role']=='right' for b in x['sqlite_restore']['blocks'])
checks['fragment_improvement_at_least_15_points']=ab['absolute_gain_percentage_points']>=15
summary={'checks':checks,'all_checks_pass':all(checks.values()),'fixed_11_fragment_baseline':ab['baseline_hits'],'fixed_11_fragment_enabled':ab['enabled_hits'],'fixed_11_strict_correct_formula_rows':ab['strict_formula_row_correct'],'p35_api_row':row,'p35_dose_bbox':source['bbox'],'p35_processing':ctx,'newbook_ze_xie_detected':False,'newbook_zhu_ling_detected':False,'newbook_scope':'target parent has no frozen-v5 proposal; original OCR preserved; OCR itself has errors'}
(O/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=True,indent=2));assert all(checks.values())
