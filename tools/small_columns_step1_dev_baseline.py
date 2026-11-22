import hashlib, json, os, sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import CONFIG
from ocr.pipeline import run_ocr
from ocr.normalize import normalize_text
OUT = ROOT / 'output' / 'small_columns_step1_dev'
OUT.mkdir(parents=True, exist_ok=True)
object.__setattr__(CONFIG.paths, 'reports_dir', OUT / 'reports')
source = json.loads((ROOT / 'output/formula_page35_small_columns_ab.json').read_text(encoding='utf-8'))
page = Path(source['page'])
if not page.exists():
    raise FileNotFoundError(page)
records = {x['region']: x for x in source['records']}
whole, main, right, left = source['records'][0], source['records'][2], source['records'][3], source['records'][4]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
annotation = {'status':'development_only_incomplete', 'page':str(page.relative_to(ROOT)), 'sha256':sha(page), 'image_size':[1024,804], 'sample': {'id':'dev-35-001', 'label':'positive', 'outer_bbox':whole['bbox'], 'main_bbox':main['bbox'], 'main_text':main['expected_groups'][0], 'reading_order':['right','left'], 'columns':[{'side':'right','bbox':right['bbox'],'text':right['expected_groups'][0]}, {'side':'left','bbox':left['bbox'],'text':left['expected_groups'][0]}], 'dose':{'verbatim':'\u4e94\u9322','applies_to':[main['expected_groups'][0]]}, 'uncertainty':'none_recorded_in_existing_manual_AB', 'evidence':'output/formula_page35_small_columns_ab.json'}, 'holdout':{'positive':[], 'negative':[]}, 'note':'Partial development example. No holdout or acceptance claim.'}
(OUT/'annotation.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2),encoding='utf-8')
result = run_ocr(str(page),page_id='SMALL-COLUMNS-DEV-35')
obj = result.to_dict()
model = ROOT / CONFIG.ocr.rec_model if CONFIG.ocr.rec_model else None
obj['baseline_metadata']={'ocr_config':asdict(CONFIG.ocr),'image_sha256':sha(page),'recognizer_sha256':sha(model) if model and model.is_file() else None,'environment_TCM_keys':[k for k in os.environ if k.startswith('TCM_')], 'method':'unchanged ocr.pipeline.run_ocr; reports redirected to output folder'}
(OUT/'page35_full_ocr.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
box=whole['bbox']
def overlap(b):
    a=b.bbox
    return min(a[2],box[2])>max(a[0],box[0]) and min(a[3],box[3])>max(a[1],box[1])
blocks=[b for b in result.blocks if overlap(b)]
dose=annotation['sample']['dose']['verbatim']
match={'sample_id':'dev-35-001','matching':'positive area intersection; dose substring within a single intersecting block, exact and normalized', 'blocks':[{'line_id':b.line_id,'bbox':b.bbox,'text':b.text,'text_norm':b.text_norm,'confidence':b.conf} for b in blocks], 'dose':dose, 'applies_to':annotation['sample']['dose']['applies_to'], 'exact_hit':any(dose in b.text for b in blocks),'normalized_hit':any(normalize_text(dose) in normalize_text(b.text) for b in blocks)}
(OUT/'page35_dose_match.json').write_text(json.dumps(match,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'engine':result.engine,'model_loaded':result.model_loaded,'is_demo':result.is_demo,'block_count':len(result.blocks),'overlap':len(blocks),'exact_hit':match['exact_hit'],'normalized_hit':match['normalized_hit']},ensure_ascii=True))

