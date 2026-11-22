"""Post-inference human visual record and non-integrated fallback demonstration."""
import hashlib,json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.vertical import imread_unicode
O=ROOT/'output/small_columns_step3_newbook';pred=O/'wenyi_page12_v5_full_predictions.json';manifest=O/'pre_inference_manifest.json';target=O/'wenyi_page12_manual_review.json'
assert pred.exists() and not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(pred.read_text(encoding='utf8'));m=json.loads(manifest.read_text(encoding='utf8'));assert p['source_sha256']==m['source_sha256']==sha(ROOT/p['source']);assert p['pre_inference_manifest_sha256']==sha(manifest)
im=imread_unicode(str(ROOT/p['source']));parent=[163,9,293,801];assert any(c['bbox']==parent for c in p['automatic_columns']);assert not any(q['parent_bbox']==parent for q in p['geometry_predictions']);assert not any(q['parent_bbox']==parent for q in p['predictions'])
block=next(b for b in p['normal_ocr']['blocks'] if b['line_id']=='R021')
regions=[
 {'id':'W12-01','main':{'text':'澤瀉','bbox':[250,353,285,420]},'columns':[{'side':'right','bbox':[265,418,287,478],'text':'一錢'},{'side':'left','bbox':[244,418,265,478],'text':'五分'}],'human_reading_order':['澤瀉','一錢','五分'],'reading':'澤瀉一錢五分','certainty':'主要字形和錢、分可辨；框為人工目測近似值','adjacent_text':'上方茯苓二錢；下方豬苓'},
 {'id':'W12-02','main':{'text':'豬苓','bbox':[250,478,285,531]},'columns':[{'side':'right','bbox':[265,529,287,590],'text':'一錢'},{'side':'left','bbox':[244,529,265,589],'text':'五分'}],'human_reading_order':['豬苓','一錢','五分'],'reading':'豬苓一錢五分','certainty':'主要字形和錢、分可辨；框為人工目測近似值','adjacent_text':'上方澤瀉；下方陳皮一錢'}]
review={'source':p['source'],'source_sha256':p['source_sha256'],'prediction_sha256':sha(pred),'method':'先封存整頁自動推理，後目視原圖和放大裁圖；座標左上原點、右下為近似排他邊界；人工轉錄不進 OCR 或劑量表','title':{'text':'四苓湯','bbox':[300,184,323,274]},'full_visual_reading':'四苓湯：茯苓二錢，澤瀉一錢五分，豬苓一錢五分，陳皮一錢。','regions':regions,'automatic_finding':{'automatic_outer_parent_bbox':parent,'normal_ocr_block':block,'geometry_candidates_in_parent':0,'v5_candidates_in_parent':0,'diagnosis':'版面自動父列寬130像素，包住多個相鄰印刷列；其內未形成 v3 幾何候選，因此 v5 沒有局部分欄輸出。無局部復識結果可報；整頁 R021 存在漏字錯字。'},'fallback_demonstration':{'type':'offline caller contract demonstration, not integrated into production','when':'無可用分欄候選','preserve_original_parent_bbox':parent,'preserved_normal_ocr_line_ids':[block['line_id']],'normal_ocr_block':block,'warning':'保留可追溯原圖與原始 OCR，不能保證 OCR 文字和劑量正確；實際集成分支和下游行為仍須後續測試。'}}
target.write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
full=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(full)
for q in p['predictions']:
 draw.rectangle(q['main_bbox'],outline='cyan',width=2)
 for c in q['columns']:draw.rectangle(c['bbox'],outline='red',width=2)
draw.rectangle(parent,outline='yellow',width=2);full.save(O/'wenyi_page12_full_automatic.png')
box=[230,220,294,705];crop=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB)).resize(((box[2]-box[0])*4,(box[3]-box[1])*4));d=ImageDraw.Draw(crop)
def rect(b,color):d.rectangle(tuple((b[i]-box[i%2])*4 for i in range(4)),outline=color,width=3)
for r in regions:
 rect(r['main']['bbox'],'yellow')
 for c in r['columns']:rect(c['bbox'],'lime' if c['side']=='left' else 'red')
crop.save(O/'wenyi_page12_manual_regions.png')
integrity={'manifest_sha256':sha(manifest),'full_predictions_sha256':sha(pred),'manual_review_sha256':sha(target),'source_sha256':sha(ROOT/p['source']),'scope':'single new page; no holdout sample count changes'}
(O/'evidence_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'regions':len(regions),'full_ocr_text':block['text'],'outer_parent':parent,'geometry':0,'v5':0,'fallback_ocr_block_preserved':True,'integrity':integrity},ensure_ascii=True))
