import copy,unittest
from unittest.mock import patch
import numpy as np
from app.schemas import Block,OCRResult
from ocr.small_columns_attach import attach_small_columns
from extract.formula_segments import _Stream

class SmallColumnAttachTest(unittest.TestCase):
 def fixture(self):
  parent=Block(page_id='P',bbox=[10,0,50,100],line_id='R1',text='白术',text_norm='白术',conf=.6,source_hash='parent')
  ocr=OCRResult(page_id='P',image_path='image.png',blocks=[parent],model_loaded=True,is_demo=False)
  row={'herb':{'value':'待核','status':'待核','source':{'text':'白术','sources':[{'block_ref':parent.block_ref}]}},'dose':{'value':'待核','status':'待核','source':None}}
  formulas={'formulas':[{'rows':[row]}]}
  proposal={'parent_bbox':[10,0,50,100],'main_bbox':[10,10,50,30],'columns':[{'side':'right','bbox':[30,30,50,70]},{'side':'left','bbox':[10,30,30,50]}],'evidence':{'page_typography':{'ink_contrast':50}}}
  return ocr,formulas,proposal
 def run_case(self,right='五錢黃土',second=None,formula=None):
  ocr,base,p=self.fixture();base=formula or base;second=right if second is None else second
  reads=iter([('白术',.98),('白术',.98),(right,.94),(second,.94),('排炒',.7),('排炒',.7)])
  with patch('ocr.small_columns_attach.imread_unicode',return_value=np.zeros((100,80,3),dtype=np.uint8)),patch('ocr.small_columns_attach.page_layout',return_value={'cols':[]}),patch('ocr.small_columns_attach.propose_components_page',return_value={'predictions':[p]}),patch('ocr.small_columns_attach._read',side_effect=lambda *a:next(reads)),patch('ocr.small_columns_attach._ink',return_value=True):
   return ocr,base,attach_small_columns(ocr,base)
 def test_attaches_only_existing_row_and_preserves_source_and_processing(self):
  ocr,base,(blocks,result,audit)=self.run_case();self.assertEqual(audit['accepted'],1);self.assertEqual(base['formulas'][0]['rows'][0]['dose']['value'],'待核')
  row=result['formulas'][0]['rows'][0];self.assertEqual(row['dose']['value'],'五錢');self.assertEqual(row['dose']['source']['sources'][0]['bbox'],[30,30,50,70]);self.assertEqual([x['text'] for x in row['dose']['processing_context']['fragments']],['黃土','排炒']);self.assertTrue(all(x['status']=='待核' for x in row['dose']['processing_context']['fragments']))
  self.assertEqual(len(ocr.blocks),1);self.assertEqual(len(blocks),3);self.assertEqual(_Stream(ocr.blocks+blocks).norm,'白术')
 def test_inconsistent_or_shared_dose_does_not_attach(self):
  for right,second in [('五錢','六錢'),('各三錢','各三錢'),('二錢手','二錢手'),('一錢二','一錢二')]:
   _,base,(blocks,result,audit)=self.run_case(right,second);self.assertEqual(audit['accepted'],0);self.assertEqual(result,base)
 def test_existing_dose_not_overwritten(self):
  _,base,_=self.fixture();base['formulas'][0]['rows'][0]['dose']={'value':'二錢','status':'已识别'}
  _,_,(_,result,audit)=self.run_case(formula=base);self.assertEqual(audit['accepted'],0);self.assertEqual(result,base)
 def test_printed_table_falls_back(self):
  ocr,base,_=self.fixture();ocr.tables=[object()];blocks,result,audit=attach_small_columns(ocr,base);self.assertFalse(blocks);self.assertEqual(result,base);self.assertTrue(audit['fallback'])
 def test_optional_server_fallback_retains_existing_gates(self):
  ocr,base,p=self.fixture();reads=iter([('白术',.98),('白术',.98),('五线',.7),('五線',.7),('排炒',.7),('排炒',.7)])
  with patch('ocr.small_columns_attach.imread_unicode',return_value=np.zeros((100,80,3),dtype=np.uint8)),patch('ocr.small_columns_attach.page_layout',return_value={'cols':[]}),patch('ocr.small_columns_attach.propose_components_page',return_value={'predictions':[p]}),patch('ocr.small_columns_attach._read',side_effect=lambda *a:next(reads)),patch('ocr.small_columns_attach._ink',return_value=True),patch('ocr.small_columns_server.read_pair',return_value=[('五錢黃土',.9),('五錢黃土',.9)]):
   blocks,result,audit=attach_small_columns(ocr,base,server_fallback=True)
  self.assertEqual(audit['accepted'],1);self.assertEqual(result['formulas'][0]['rows'][0]['dose']['value'],'五錢');self.assertEqual(audit['candidates'][0]['recognition']['right']['recognizer'],'official_server_fallback');right=next(b for b in blocks if b.evidence_role=='right');self.assertEqual(right.bbox,[30,30,50,70]);self.assertEqual(right.evidence_engine,'PP-OCRv5_server_rec')
 def test_server_failure_preserves_mobile_fallback(self):
  ocr,base,p=self.fixture();reads=iter([('白术',.98),('白术',.98),('五线',.7),('五線',.7),('排炒',.7),('排炒',.7)])
  with patch('ocr.small_columns_attach.imread_unicode',return_value=np.zeros((100,80,3),dtype=np.uint8)),patch('ocr.small_columns_attach.page_layout',return_value={'cols':[]}),patch('ocr.small_columns_attach.propose_components_page',return_value={'predictions':[p]}),patch('ocr.small_columns_attach._read',side_effect=lambda *a:next(reads)),patch('ocr.small_columns_attach._ink',return_value=True),patch('ocr.small_columns_server.read_pair',side_effect=ValueError('missing model')):
   _,result,audit=attach_small_columns(ocr,base,server_fallback=True)
  self.assertEqual(result,base);self.assertEqual(audit['accepted'],0);self.assertEqual(audit['candidates'][0]['recognition']['right']['server_status'],'unavailable_ValueError')

if __name__=='__main__':unittest.main()
