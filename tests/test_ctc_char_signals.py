import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ocr import recognize as ocr


class CtcSignalTests(unittest.TestCase):
    def setUp(self):
        self.charset = ['blank', '甲', '乙', '丙']
        self.probs = np.array([[.1, .8, .05, .05], [.1, .7, .1, .1],
                               [.9, .05, .03, .02], [.1, .2, .6, .1],
                               [.1, .7, .1, .1]])

    def test_collapse_and_statistics(self):
        result = ocr.extract_ctc_chars(self.probs, self.charset)
        self.assertEqual(result.text, '甲乙甲')
        self.assertEqual([c.timestep for c in result.chars], [0, 3, 4])
        self.assertAlmostEqual(result.conf_mean, .7)
        self.assertAlmostEqual(result.conf_var, (.01 + .01 + 0) / 3)
        self.assertAlmostEqual(result.conf_min, .6)
        self.assertAlmostEqual(result.conf, .7)
        self.assertEqual(result.total_steps, 5)

    def test_empty_invalid_and_out_of_charset(self):
        for matrix in ([], np.empty((0, 4)), np.ones((2,)), np.empty((2, 0))):
            with self.subTest(shape=np.shape(matrix)):
                result = ocr.extract_ctc_chars(matrix, self.charset)
                self.assertEqual(result.chars, [])
                self.assertEqual((result.conf_mean, result.conf_var, result.conf_min), (0, 0, 0))
        result = ocr.extract_ctc_chars(np.array([[0, 0, 1]]), ['blank', '甲'])
        self.assertEqual(result.chars, [])
        result = ocr.extract_ctc_chars(self.probs, self.charset, content_steps=100)
        self.assertEqual(result.content_steps, 5)
        result = ocr.extract_ctc_chars(self.probs, self.charset, content_steps=.1)
        self.assertEqual(result.content_steps, 1)
        self.assertTrue(all(0 <= c.rel_pos <= c.rel_end <= 1 for c in result.chars))

    def test_padding(self):
        self.assertAlmostEqual(ocr.compute_content_steps((48, 80, 3), (3, 48, 320), 40), 10)
        self.assertEqual(ocr.compute_content_steps((48, 640, 3), (3, 48, 320), 40), 40)
        self.assertEqual(ocr.compute_content_steps((0, 80), (3, 48, 320), 40), 0)
        self.assertEqual(ocr.compute_content_steps((48, 80), (3, 48, 320), 0), 0)
        self.assertEqual(ocr.compute_content_steps((), (), -1), 0)

    def test_spans(self):
        chars = ocr.extract_ctc_chars(self.probs, self.charset, content_steps=4).chars
        spans = ocr.map_chars_to_y_spans(chars, 10, 110)
        self.assertEqual([s[0] for s in spans], ['甲', '乙', '甲'])
        self.assertTrue(all(10 <= a <= b <= 110 for _, _, a, b in spans))
        self.assertEqual([s[2] for s in spans], sorted(s[2] for s in spans))
        self.assertEqual(spans[0][2], 10)
        self.assertEqual(spans[-1][3], 110)

    def test_override(self):
        result = ocr.extract_ctc_chars(self.probs, self.charset, text_override=' 甲丙甲 ', conf_override=.91)
        self.assertEqual(result.text, '甲丙甲')
        self.assertEqual([c.char for c in result.chars], list('甲丙甲'))
        self.assertAlmostEqual(result.chars[1].prob, .1)
        self.assertEqual([c.timestep for c in result.chars], [0, 3, 4])
        self.assertEqual(result.conf, .91)
        result = ocr.extract_ctc_chars(self.probs, self.charset, text_override='不同长度文字')
        self.assertEqual(result.text, '不同长度文字')
        self.assertEqual(''.join(c.char for c in result.chars), '甲乙甲')
        self.assertEqual(ocr.extract_ctc_chars(self.probs, self.charset, text_override='  ').text, '甲乙甲')

    def make_engine(self, fail=False):
        probs = self.probs
        class Postprocess:
            character = self.charset
            def __call__(self, preds):
                if fail:
                    raise RuntimeError('failure')
                return [('甲乙甲', .87654)]
        class Rec:
            rec_image_shape = (3, 48, 320)
            def __init__(self):
                self.postprocess_op = Postprocess()
            def __call__(self, images):
                return self.postprocess_op(np.array([probs])), None
        rec = Rec()
        return SimpleNamespace(text_rec=rec), rec.postprocess_op

    def test_inference_with_and_without_rerank(self):
        image = np.zeros((48, 80, 3), dtype=np.uint8)
        for enabled in (False, True):
            with self.subTest(rerank=enabled):
                engine, orig = self.make_engine()
                with patch.object(ocr, '_engine', return_value=engine) as get_engine, \
                     patch.object(ocr, 'CONFIG', SimpleNamespace(ocr=SimpleNamespace(ctc_rerank=enabled))), \
                     patch('ocr.ctc_rerank.ClassicalTCMLM.from_default_corpus', return_value=object()), \
                     patch('ocr.ctc_rerank.rerank_sequence_from_probs', return_value=('甲丙甲', [])):
                    result = ocr.recognize_crop_with_ctc(image, rec_model='model')
                    get_engine.assert_called_once_with('model')
                self.assertIs(engine.text_rec.postprocess_op, orig)
                self.assertEqual(result.text, '甲丙甲' if enabled else '甲乙甲')
                self.assertEqual([c.timestep for c in result.chars], [0, 3, 4])
                self.assertEqual(result.conf, .87654)
                self.assertEqual(result.content_steps, 1.25)

    def test_restore_on_failure_and_threshold(self):
        image = np.zeros((48, 80, 3), dtype=np.uint8)
        for enabled in (False, True):
            engine, orig = self.make_engine(fail=True)
            with patch.object(ocr, '_engine', return_value=engine), \
                 patch.object(ocr, 'CONFIG', SimpleNamespace(ocr=SimpleNamespace(ctc_rerank=enabled))), \
                 patch('ocr.ctc_rerank.ClassicalTCMLM.from_default_corpus', return_value=object()):
                self.assertEqual(ocr.recognize_crop_with_ctc(image), ocr.CropCTCResult())
            self.assertIs(engine.text_rec.postprocess_op, orig)
        engine, orig = self.make_engine()
        with patch.object(ocr, '_engine', return_value=engine), \
             patch.object(ocr, 'CONFIG', SimpleNamespace(ocr=SimpleNamespace(ctc_rerank=False))):
            self.assertEqual(ocr.recognize_crop_with_ctc(image, .9), ocr.CropCTCResult())
            self.assertEqual(ocr.recognize_crop(image), ('甲乙甲', .8765))
            self.assertEqual(ocr.recognize_crop(image, .9), ('', 0.0))
        self.assertIs(engine.text_rec.postprocess_op, orig)


if __name__ == '__main__':
    unittest.main()
