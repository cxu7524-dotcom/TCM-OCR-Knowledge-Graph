"""Freeze identical automatic and human-reference crop inputs before model comparison."""
import hashlib,json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step5';target=O/'crop_manifest.json';assert not target.exists();(O/'crops').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
labels=json.loads((ROOT/'output/small_columns_step1_holdout/annotations.json').read_text(encoding='utf8'));screen=json.loads((ROOT/'output/small_columns_step3_detector/holdout_fifth_review_screen.json').read_text(encoding='utf8'))['samples'];screen={s['id']:s for s in screen};adjudicated=json.loads((ROOT/'output/small_columns_step3_detector/holdout_fifth_adjudicated.json').read_text(encoding='utf8'))['samples'];adjudicated={s['id']:s for s in adjudicated}
samples=[s for s in labels['samples'] if s['label']=='positive' and s.get('dose',{}).get('verbatim')];assert len(samples)==11
items=[]
def add(identifier,mode,path,digest,boxes,truth,metadata):
 path=ROOT/path;assert sha(path)==digest;im=imread_unicode(str(path))
 for role,box in boxes.items():
  x,y,X,Y=map(int,box);assert 0<=x<X<=im.shape[1] and 0<=y<Y<=im.shape[0]
  crop=im[y:Y,x:X];out=O/'crops'/f'{identifier}_{mode}_{role}.png';assert not out.exists();cv2.imencode('.png',crop)[1].tofile(str(out))
  items.append({'sample_id':identifier,'mode':mode,'role':role,'source':str(path.relative_to(ROOT)),'source_sha256':digest,'bbox':list(box),'crop':str(out.relative_to(ROOT)),'crop_sha256':sha(out),'expected_text':truth.get(role),'expected_dose':truth.get('dose') if role=='right' else None,**metadata})
for s in samples:
 meta=labels['pages'][str(s['page'])];gt={c['side']:c for c in s['columns']};truth={'main':s['main_text'],'right':gt['right']['text'],'left':gt['left']['text'],'dose':s['dose']['verbatim']};q=screen[s['id']]['selected_proposal'];info={'fifth_round_full_geometry_correct':adjudicated[s['id']]['review']['correct'],'expected_applies_to':s['dose']['applies_to']}
 if q:
  boxes={'main':q['main_bbox'],**{c['side']:c['bbox'] for c in q['columns']}};add(s['id'],'auto_v5',meta['path'],meta['sha256'],boxes,truth,info)
 boxes={'main':s['main_bbox'],'right':gt['right']['bbox'],'left':gt['left']['bbox']};add(s['id'],'human_reference',meta['path'],meta['sha256'],boxes,truth,info)
new=json.loads((ROOT/'output/small_columns_step3_newbook/wenyi_page12_manual_review.json').read_text(encoding='utf8'))
for s in new['regions']:
 gt={c['side']:c for c in s['columns']};add(s['id'],'human_reference',new['source'],new['source_sha256'],{'main':s['main']['bbox'],'right':gt['right']['bbox'],'left':gt['left']['bbox']},{'main':s['main']['text'],'right':gt['right']['text'],'left':gt['left']['text'],'dose':gt['right']['text']},{'automatic_geometry_missing':True,'manual_only_not_automatic_score':True})
dev=json.loads((ROOT/'output/small_columns_step3_detector/development_comparison.json').read_text(encoding='utf8'));s=next(x for x in dev['samples'] if x['sample']['id']=='S35');d=s['sample'];q=s['combined']['prediction'];gt={c['side']:c for c in d['columns']};add('DEV35','auto_v5',d['path'],d['sha256'],{'main':q['main_bbox'],**{c['side']:c['bbox'] for c in q['columns']}},{'main':'白朮','right':gt['right']['text'],'left':gt['left']['text'],'dose':'五錢'},{'development_case':True})
data={'method':'same lossless original pixel crop for both models; 90-degree CCW; scale 1 and 2; GT only for post-inference scoring; human_reference excluded from auto metrics','fixed_11_total':11,'fixed_11_with_auto':sum(screen[s['id']]['selected_proposal'] is not None for s in samples),'crops':items}
target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print('frozen',len(items),'crops; auto targets',data['fixed_11_with_auto'],'manifest sha',sha(target),flush=True)
