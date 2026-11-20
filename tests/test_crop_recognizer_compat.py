import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ocr import recognize

class CropRecognizerCompatibilityTests(unittest.TestCase):
    def test_current_text_rec_entry(self):
        engine=SimpleNamespace(text_rec=lambda images: ([('甲',0.91)],0.01))
        with patch.object(recognize,'_engine',return_value=engine):
            self.assertEqual(recognize.recognize_crop(object()),('甲',0.91))
    def test_legacy_entry_and_confidence_filter(self):
        engine=SimpleNamespace(text_recognizer=lambda images: ([('乙',0.12)],0.01))
        with patch.object(recognize,'_engine',return_value=engine):
            self.assertEqual(recognize.recognize_crop(object()),('',0.0))
            self.assertEqual(recognize.recognize_crop(object(),min_conf=0.1),('乙',0.12))
    def test_failure_logs_and_returns_compatible_empty_result(self):
        with patch.object(recognize,'_engine',return_value=SimpleNamespace()):
            with self.assertLogs('ocr.recognize',level='WARNING') as log:
                self.assertEqual(recognize.recognize_crop(object()),('',0.0))
            self.assertIn('AttributeError',log.output[0])
    def test_charset_uses_same_current_entry(self):
        engine=SimpleNamespace(text_rec=SimpleNamespace(postprocess_op=SimpleNamespace(character=['blank','甲',' '])) )
        with patch.object(recognize,'_engine',return_value=engine):
            self.assertEqual(recognize.model_charset(),{'甲'})

if __name__=='__main__':unittest.main()
