"""A/B fixed 11 fragment score and stricter correct formula-row score."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.schemas import OCRResult
from extract.formula_rows import extract_formula_rows
from ocr.normalize import normalize_text
from ocr.small_columns_attach import attach_small_columns
O=ROOT/'output/small_columns_step4';O.mkdir(exist_ok=True);target=O/'holdout_ab_v3.json';assert not target.exists()
base=ROOT/'output/small_columns_step1_holdout';labels=json.loads((base/'annotations.json').read_text(encoding='utf8'));old=json.loads((base/'sample_matches.json').read_text(encoding='utf8'))['samples'];old={s['id']:s for s in old};manual=json.loads((ROOT/'output/small_columns_step3_detector/holdout_fifth_adjudicated.json').read_text(encoding='utf8'))['samples'];verdict={s['id']:s['review'].get('correct',False) for s in manual}
def overlaps(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
pages={}
for n,meta in labels['pages'].items():
 p=ROOT/meta['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 raw=json.loads((base/f'page_{n}_full_ocr.json').read_text(encoding='utf8'));ocr=OCRResult.from_dict(raw);ocr.image_path=str(p);assert ocr.model_loaded and not ocr.is_demo
 baseline=extract_formula_rows(ocr.blocks);extra,enhanced,audit=attach_small_columns(ocr,baseline)
 pages[n]={'original_blocks':[b.to_dict() for b in ocr.blocks],'evidence_blocks':[b.to_dict() for b in extra],'baseline_formula':baseline,'enhanced_formula':enhanced,'audit':audit}
 print(n,'original blocks',len(ocr.blocks),'evidence',len(extra),'attached rows',audit['accepted'],flush=True)
rows=[]
for s in labels['samples']:
 if s['label']!='positive' or not s.get('dose',{}).get('verbatim'):continue
 page=pages[str(s['page'])];target_dose=s['dose']['verbatim'];outer=s['outer_bbox']
 def hits(blocks):return [b for b in blocks if overlaps(b['bbox'],outer) and normalize_text(target_dose) in normalize_text(b['text'])]
 bh=hits(page['original_blocks']);eh=hits(page['original_blocks']+page['evidence_blocks']);assert bool(bh)==old[s['id']]['normalized_hit']
 linked=[]
 for f in page['enhanced_formula']['formulas']:
  for r in f['rows']:
   src=(r['dose'].get('source') or {}).get('sources') or []
   herb=normalize_text(r['herb'].get('value',''));expected=normalize_text(s['main_text']);actual=normalize_text(r['dose'].get('value',''));want=normalize_text(target_dose).removeprefix('各')
   if r['dose']['status']=='已识别' and actual==want and herb==expected and src and any(overlaps(x['bbox'],s['columns'][0]['bbox']) for x in src) and verdict[s['id']]:linked.append({'herb':r['herb'],'dose':r['dose']})
 rows.append({'id':s['id'],'dose_fragment':target_dose,'baseline_hit':bool(bh),'enabled_hit':bool(eh),'matched_evidence':[b for b in eh if b['kind']=='small_column_evidence'],'strict_formula_row_correct':bool(linked),'strict_formula_rows':linked,'fifth_round_visual_box_correct':verdict[s['id']]})
assert len(rows)==11
result={'scope':'same step1 one-block normalized exact-fragment match, fixed 11 readable dose samples; GT used only after inference','baseline_hits':sum(x['baseline_hit'] for x in rows),'enabled_hits':sum(x['enabled_hit'] for x in rows),'denominator':len(rows),'absolute_gain_percentage_points':round(100*(sum(x['enabled_hit'] for x in rows)-sum(x['baseline_hit'] for x in rows))/len(rows),2),'required_gain_percentage_points':15,'strict_formula_row_correct':sum(x['strict_formula_row_correct'] for x in rows),'strict_formula_row_method':'same 11; existing formula row, recognized herb equals label, full dose equals label, right-side crop source intersects target, fifth-round visual box adjudicated correct; shared doses are not automatically assigned','samples':rows,'pages':pages}
target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','pages')},ensure_ascii=True),flush=True)
