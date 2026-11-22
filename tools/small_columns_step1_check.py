import hashlib, json
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/small_columns_step1_dev'
a = json.loads((OUT/'annotation.json').read_text(encoding='utf-8'))
r = json.loads((OUT/'page35_full_ocr.json').read_text(encoding='utf-8'))
p = ROOT/a['page']
w,h = Image.open(p).size
s=a['sample']; outer=s['outer_bbox']
def within(b): return len(b)==4 and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
def contained(b): return outer[0]<=b[0]<b[2]<=outer[2] and outer[1]<=b[1]<b[3]<=outer[3]
checks={'source_sha256_matches':hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'], 'image_size_matches':a['image_size']==[w,h], 'all_boxes_in_image':all(map(within,[outer,s['main_bbox']]+[c['bbox'] for c in s['columns']])), 'subboxes_in_outer':all(map(contained,[s['main_bbox']]+[c['bbox'] for c in s['columns']])), 'reading_order_and_columns':s['reading_order']==[c['side'] for c in s['columns']]==['right','left'], 'real_ocr':r['model_loaded'] and not r['is_demo'], 'positive_holdout_at_least_20':len(a['holdout']['positive'])>=20, 'negative_holdout_at_least_20':len(a['holdout']['negative'])>=20}
out={'scope':'development example only; visual transcription and semantics cannot be validated structurally', 'checks':checks, 'counts':{'development_positive':1,'holdout_positive':len(a['holdout']['positive']),'holdout_negative':len(a['holdout']['negative'])},'acceptance_passed':all(checks.values()), 'page_split_check':'not assessable: no holdout pages'}
(OUT/'structure_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=True))
