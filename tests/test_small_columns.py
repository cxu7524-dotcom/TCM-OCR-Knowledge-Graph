import unittest
from pathlib import Path
import numpy as np
from ocr.small_columns import propose_in_column,propose_page
from ocr.vertical import imread_unicode,page_layout
ROOT=Path(__file__).resolve().parents[1]
class SmallColumnsTests(unittest.TestCase):
 def test_blank_or_empty_falls_back(self):
  self.assertEqual(propose_in_column(np.full((500,100,3),255,dtype=np.uint8),[25,0,65,500]),[])
  self.assertEqual(propose_in_column(None,[1,2,3,4]),[])
 def test_auto_column_outputs_source_coordinates_and_order(self):
  path=ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png'
  if not path.exists():self.skipTest('development page not present')
  img=imread_unicode(str(path));cols=page_layout(str(path))['cols'];pred=propose_page(img,cols)
  matches=[p for p in pred if p['parent_bbox'][0]==136 and 455<=p['columns'][0]['bbox'][1]<=465]
  self.assertEqual(len(matches),1);p=matches[0]
  self.assertEqual(p['reading_order'],['main','right','left'])
  self.assertGreaterEqual(p['columns'][0]['bbox'][0],p['columns'][1]['bbox'][2])
  self.assertLess(p['main_bbox'][3],p['columns'][0]['bbox'][3])
  self.assertIn(p['parent_bbox'],[c['bbox'] for c in cols])
 def test_development_long_right_and_short_left_have_independent_ends(self):
  path=ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png'
  if not path.exists():self.skipTest('development page not present')
  image=imread_unicode(str(path));pred=propose_page(image,page_layout(str(path))['cols'])
  q=[p for p in pred if p['parent_bbox'][0]==136 and 450<=p['columns'][0]['bbox'][1]<=470]
  self.assertEqual(len(q),1)
  right,left=[x['bbox'] for x in q[0]['columns']]
  self.assertGreaterEqual(right[3],560)
  self.assertLess(left[3],right[3]-30)
  self.assertLessEqual(q[0]['main_bbox'][1],410)
 def test_development_short_note_and_single_column(self):
  path=ROOT/'古籍评测集/医家类/思濟堂方書/page_42.png'
  if not path.exists():self.skipTest('development page not present')
  image=imread_unicode(str(path));pred=propose_page(image,page_layout(str(path))['cols'])
  q=[p for p in pred if p['parent_bbox'][0]==948 and 650<=p['columns'][0]['bbox'][1]<=675]
  self.assertEqual(len(q),1)
  self.assertGreater(q[0]['columns'][0]['bbox'][3],700)
  self.assertLess(q[0]['columns'][1]['bbox'][3],690)
  self.assertFalse(any(p['parent_bbox'][0]==948 and p['columns'][0]['bbox'][1]<375 for p in pred))

 def test_page_typography_and_three_character_main_band(self):
  from ocr.small_columns import estimate_page_typography
  path=ROOT/'古籍评测集/医家类/思濟堂方書/page_33.png'
  if not path.exists():self.skipTest('development page not present')
  image=imread_unicode(str(path));columns=page_layout(str(path))['cols'];scale=estimate_page_typography(image,columns)
  self.assertIsNotNone(scale);self.assertGreaterEqual(scale['sample_bands'],10)
  pred=propose_page(image,columns)
  found=[p for p in pred if p['parent_bbox'][0]==899 and 315<=p['columns'][0]['bbox'][1]<=325]
  self.assertEqual(len(found),1);main=found[0]['main_bbox']
  self.assertLessEqual(main[1],240);self.assertGreaterEqual(main[3],320)
  self.assertLessEqual(main[3],found[0]['columns'][0]['bbox'][1])
 def test_joined_long_main_text_is_not_reduced_to_one_band(self):
  path=ROOT/'古籍评测集/医家类/思濟堂方書/page_36.png'
  if not path.exists():self.skipTest('development page not present')
  image=imread_unicode(str(path));pred=propose_page(image,page_layout(str(path))['cols'])
  found=[p for p in pred if p['parent_bbox'][0]==187 and 345<=p['columns'][0]['bbox'][1]<=365]
  self.assertEqual(len(found),1);main=found[0]['main_bbox']
  self.assertLessEqual(main[1],246);self.assertGreaterEqual(main[3],350)

if __name__=='__main__':unittest.main()
