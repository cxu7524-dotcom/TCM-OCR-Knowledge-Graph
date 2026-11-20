import unittest
from unittest.mock import patch
import numpy as np
from ocr.small_columns_components import _crossing_end,propose_components_page

class ComponentEvidenceTest(unittest.TestCase):
    def test_extends_only_crossing_component(self):
        mask=np.zeros((40,20),dtype=bool);mask[15:23,5:8]=True;mask[27:31,5:8]=True
        box,trace=_crossing_end(mask,[4,10,10,20])
        self.assertEqual(box,[4,10,10,23]);self.assertEqual(len(trace),1)

    def test_empty_recognition_does_not_erase_candidate(self):
        image=np.full((100,80,3),255,np.uint8)
        p={'parent_bbox':[10,0,50,100],'main_bbox':[10,10,50,30],'columns':[{'side':'right','bbox':[30,30,50,50]},{'side':'left','bbox':[10,30,30,50]}],'evidence':{}}
        with patch('ocr.small_columns_components.propose_page',return_value=[p]),patch('ocr.small_columns_components._supported_blocks',return_value=[{'line_id':'R1'}]),patch('ocr.small_columns_components._main_link_evidence',return_value={'valid':True}),patch('ocr.small_columns_components._recognize',return_value=('',0)):
            result=propose_components_page(image,[],[{'bbox':[10,10,50,50]}])
        self.assertEqual(len(result['predictions']),1)
        self.assertEqual([s['original_recognition']['status'] for s in result['audit'][0]['side_evidence']],['unreadable','unreadable'])

if __name__=='__main__':unittest.main()
