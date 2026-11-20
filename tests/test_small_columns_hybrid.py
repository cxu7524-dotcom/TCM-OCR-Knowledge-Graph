import json,unittest
from pathlib import Path
import numpy as np
from ocr.small_columns_hybrid import _main_link_evidence,propose_hybrid_page
from ocr.vertical import imread_unicode,page_layout
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'output/small_columns_step3_hybrid/development_full_ocr.json'
class HybridSmallColumnsTests(unittest.TestCase):
 def test_nonlocal_main_and_missing_ocr_fall_back(self):
  image=np.full((250,100,3),255,dtype=np.uint8)
  p={'parent_bbox':[20,0,60,250],'main_bbox':[20,20,60,60],'columns':[{'side':'right','bbox':[40,110,60,140]},{'side':'left','bbox':[20,110,40,140]}],'evidence':{'page_typography':{'height':24,'ink_contrast':50}}}
  ev=_main_link_evidence(image,p)
  self.assertFalse(ev['valid']);self.assertEqual(ev['reason'],'nonlocal_main_gap')
  result=propose_hybrid_page(image,[],[]);self.assertEqual(result['predictions'],[]);self.assertEqual(result['audit'][0]['reason'],'missing_auto_ocr_fallback')
 def test_real_development_extension_and_main_fallback(self):
  weight=ROOT/'模型权重/ch_PP-OCRv5_rec_mobile.onnx'
  if not CACHE.exists() or not weight.exists():self.skipTest('local model or recorded development OCR unavailable')
  pages=json.loads(CACHE.read_text(encoding='utf8'))
  for n in (31,36):
   path=ROOT/'古籍评测集/医家类/思濟堂方書'/f'page_{n}.png'
   if not path.exists():self.skipTest('development image unavailable')
   image=imread_unicode(str(path));r=propose_hybrid_page(image,page_layout(str(path))['cols'],pages[str(n)]['blocks'])
   if n==31:
    event=next(e for e in r['audit'] if e['parent_bbox'][0]==751 and e['geometric_columns'][0]['bbox'][1]==613)
    self.assertEqual(event['reason'],'nonlocal_main_gap');self.assertEqual(event['decision'],'fallback_outer');self.assertTrue(event['auto_ocr_support'])
   else:
    event=next(e for e in r['audit'] if e['parent_bbox'][0]==187 and e['geometric_columns'][0]['bbox'][1]==355)
    self.assertEqual(event['decision'],'proposed');right=event['side_comparisons'][0]
    self.assertEqual(right['selection'],'extended');self.assertEqual(right['candidates'][1]['bbox'][3],404);self.assertEqual(right['candidates'][2]['bbox'][3],408)
    self.assertEqual(right['candidates'][2]['text'],'二錢');self.assertGreater(right['candidates'][2]['confidence'],right['candidates'][1]['confidence'])
if __name__=='__main__':unittest.main()
