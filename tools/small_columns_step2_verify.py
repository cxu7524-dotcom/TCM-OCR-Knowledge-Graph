import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.schemas import Block
from ocr.table import fill_missing_cells
O=ROOT/'output/small_columns_step2'; before=json.loads((O/'before_page_results.json').read_text(encoding='utf8'));after=json.loads((O/'after_page_results.json').read_text(encoding='utf8')); crops=json.loads((O/'development_crop_comparison.json').read_text(encoding='utf8'))
hold=ROOT/'output/small_columns_step1_holdout/annotations.json';hold_hash=hashlib.sha256(hold.read_bytes()).hexdigest();anno=json.loads(hold.read_text(encoding='utf8')); forbidden={p['sha256'] for p in anno['pages'].values()}
checks={}; pages={}
for k in before:
 b,a=before[k],after[k]; sig=lambda r:[(x['bbox'],x['text'],x['conf']) for x in r['blocks']]
 pages[k]={'same_blocks':sig(b)==sig(a),'same_tables':[x['rows'] for x in b['tables']]==[x['rows'] for x in a['tables']],'same_image':b['image_sha256']==a['image_sha256'],'actual_real_model':a['model_loaded'] and not a['is_demo']}
checks['page_and_table_regression']=all(all(v.values()) for v in pages.values());checks['development_images_not_holdout']=all(x['image_sha256'] not in forbidden for x in crops['samples']) and all(v['image_sha256'] not in forbidden for v in before.values());checks['holdout_annotations_unchanged']=hold_hash==crops['holdout_annotation_sha256_before_after'];checks['page35_manual_boxes_and_stable_exact']=(crops['original_manual_boxes_equal'] and all(crops['samples'][0]['inputs'][crops['samples'][0]['selection'][s]]['exact_success'] and crops['samples'][0]['inputs'][crops['samples'][0]['selection'][s]]['stable'] for s in ('right','left')))
r=before['p0011'];removed=next(b for b in r['blocks'] if b['line_id']=='R006');blocks=[Block.from_dict(b) for b in r['blocks'] if b['line_id']!='R006'];img=ROOT/'data/samples/synthetic/pages/SYN-2026-001__p0011.png';filled,trace=fill_missing_cells(str(img),blocks,'STEP2-p0011');added=[v.to_dict() for v in filled if v.line_id.startswith('C')];checks['real_table_missing_cell_refill']=any(b['text']=='三兩' and b['bbox'][0]<=342+5 and b['bbox'][2]>=405 for b in added)
refill={'removed':removed,'added':added,'trace':trace};(O/'table_refill_regression.json').write_text(json.dumps(refill,ensure_ascii=False,indent=2),encoding='utf8')
summary={'checks':checks,'page_comparison':pages,'selected_development_results':{r['sample']['id']:{s:{'expected':r['sample'][s]['expected'],'selection':r['selection'][s],**r['inputs'][r['selection'][s]]} for s in ('right','left')} for r in crops['samples']},'all_checks_passed':all(checks.values())}
(O/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'checks':checks,'all_checks_passed':summary['all_checks_passed']},ensure_ascii=True));assert summary['all_checks_passed']
