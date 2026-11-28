import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from app.config import CONFIG
from app.schemas import Block, LayoutResult
from ocr.column_segments import ColumnSegment, ColumnSegmentAnalysis
from ocr.column_jiazhu import (apply_column_seg_jiazhu, evaluate_column_gate,
                               prepare_track_crop, has_dosage_or_jiazhu_cue,
                               has_lexicon_anomaly, load_tcm_lexicon)
from ocr.recognize import CharCTC, CropCTCResult, RecognizeResult


def ctc(mean=.9, var=.01, minimum=.8, chars=None):
    return CropCTCResult(conf_mean=mean, conf_var=var, conf_min=minimum, chars=chars or [])


class TestGate(unittest.TestCase):
    def setUp(self):
        self.image = np.full((180, 30, 3), 245, np.uint8)
        self.block = Block(text='天地玄黃宇宙', conf=.9)

    def test_defaults_and_snapshot(self):
        self.assertTrue(CONFIG.ocr.col_seg_jiazhu)
        self.assertIs(CONFIG.summary()["ocr_col_seg_jiazhu"], True)
        self.assertEqual(CONFIG.summary()["ocr_col_seg_jiazhu_thin"], "none")
        self.assertEqual(CONFIG.ocr.col_seg_jiazhu_thin, "none")
        b = Block.from_dict({'text': '茯苓'})
        self.assertEqual(b.segments, [])
        self.assertEqual(Block.from_dict(b.to_dict()).segments, [])
        self.assertFalse(any('segments' in p for p in b.validate()))
        b.segments = [1]
        self.assertTrue(any('segments' in p for p in b.validate()))
        b.segments = {}
        self.assertTrue(any('segments' in p for p in b.validate()))

    def test_gate_signals(self):
        self.assertFalse(evaluate_column_gate(self.block, self.image, ctc()).trigger)
        cases = [
            (replace(self.block, conf=.7), ctc(), 'low_conf'),
            (replace(self.block, conf=.83), ctc(.8, .06, .4), 'high_var_valley'),
            (replace(self.block, conf=.83, text='天地'), ctc(), 'density_under'),
            (replace(self.block, text='天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏'), ctc(), 'density_over'),
            (replace(self.block, conf=.83, text='玄黃宇宙洪荒'), ctc(.8, .01, .49), 'lexicon_anomaly'),
        ]
        for block, result, reason in cases:
            with self.subTest(reason=reason):
                self.assertIn(reason, evaluate_column_gate(block, self.image, result).reasons)
        char = CharCTC('天', .9, 0, 10, 10, 0, .5)
        self.assertIn('density_under', evaluate_column_gate(replace(self.block, conf=.83), self.image, ctc(chars=[char])).reasons)
        self.assertFalse(evaluate_column_gate(replace(self.block, conf=.95, text='天地'), self.image, ctc()).trigger)
        fn = Mock()
        self.assertFalse(evaluate_column_gate(replace(self.block, kind='title'), self.image, ctc_rec_fn=fn).trigger)
        fn.assert_not_called()

    def test_lexicon(self):
        self.assertTrue(has_dosage_or_jiazhu_cue('二錢'))
        self.assertFalse(has_lexicon_anomaly('茯苓'))
        self.assertTrue(has_lexicon_anomaly('龘靐齉'))
        self.assertTrue(load_tcm_lexicon())
        from pathlib import Path
        with patch('ocr.column_jiazhu.PROJECT_ROOT', Path('missing-lexicon-root')):
            load_tcm_lexicon.cache_clear()
            self.assertTrue(has_dosage_or_jiazhu_cue('二錢'))
            self.assertFalse(has_lexicon_anomaly('二錢'))
        load_tcm_lexicon.cache_clear()


class TestCrops(unittest.TestCase):
    def test_resize_polarity_and_padding(self):
        img = np.full((30, 12, 3), 240, np.uint8)
        img[5:25, 4:8] = 0
        none = prepare_track_crop(img, is_small=True, thin_mode='none')
        mild = prepare_track_crop(img, is_small=True, thin_mode='mild')
        erode = prepare_track_crop(img, is_small=True, thin_mode='erode')
        self.assertEqual(none.shape, (36, 72, 3))
        self.assertGreater(none[0, 0, 0], 200)
        self.assertLess(np.mean(mild < 128), np.mean(none < 128))
        self.assertLess(np.mean(erode < 128), np.mean(none < 128))
        self.assertEqual(prepare_track_crop(img, is_small=False).shape, (20, 38, 3))


class TestReassembly(unittest.TestCase):
    def test_wide_column_thresholds_and_default_thin_mode(self):
        image = np.full((300, 300, 3), 245, np.uint8)
        widths_and_text = [(100, ''), (100, ''), (178, ''), (185, '漢' * 24),
                           (186, '漢' * 23), (186, '漢' * 24),
                           (204, ''), (206, '')]
        blocks = [Block(line_id=str(i), bbox=[0, 0, width, 300], text=text)
                  for i, (width, text) in enumerate(widths_and_text)]
        # Supply enough ordinary columns to keep the page median at 100 px.
        blocks.extend(Block(line_id=f'base{i}', bbox=[0, 0, 100, 300]) for i in range(6))
        with patch('ocr.column_jiazhu.reassemble_column_block', return_value=(None, None)) as run:
            apply_column_seg_jiazhu(image, blocks)
        called = [call.args[1].line_id for call in run.call_args_list]
        self.assertEqual(called[:6], ['0', '1', '2', '3', '4', '6'])
        self.assertNotIn('5', called)
        self.assertNotIn('7', called)
        self.assertTrue(all(call.kwargs['thin_mode'] == 'none' for call in run.call_args_list))

    def test_dual_cleanliness_and_radical_guards(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        seg = ColumnSegment('dual_track', 0, 100, 2, 38,
                            (22, 0, 38, 100), (2, 0, 18, 100), 20)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        cases = [
            ('お力口；K', '五分', .9, False), ('2口', '五分', .9, False),
            ('#', '五分', .9, False), ('良干', '佳王', .8, False),
            ('月', '重', .8, False), ('一錢', '五分', .8, True),
            ('二錢半', '酒洗', .8, True), ('神旺去之或', '易沙参亦可', .8, True),
            ('良干', '佳王', .9, False), ('薑', '製', .9, True),
            ('一兩', '酒製', .9, True), ('虚證散在諸篇', '此不再贅', .8, True),
        ]
        for right, left, confidence, accepted in cases:
            with self.subTest(right=right, left=left):
                block = Block(line_id='r', bbox=[10, 10, 50, 130], text='原', conf=.6)
                rec = Mock(side_effect=[(right, confidence), (left, confidence), ('原', .9)])
                _, traces = apply_column_seg_jiazhu(image, [block], rec_fn=rec,
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(bool(traces), accepted)
                self.assertEqual(rec.call_count, 2)

    def test_high_overlap_requires_two_strong_plausible_tracks(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        seg = ColumnSegment('dual_track', 0, 100, 2, 38,
                            (22, 0, 38, 100), (2, 0, 18, 100), 20)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        chars = [CharCTC('原', .8, 0, 2, 2, 0, .6)]
        for right, left, confidence, accepted in [('月', '重', .9, False),
                                                  ('一錢', '五分', .71, False),
                                                  ('一錢', '五分', .8, True),
                                                  ('神旺去之或', '易沙参亦可', .8, True)]:
            with self.subTest(right=right, confidence=confidence):
                b = Block(bbox=[10, 10, 50, 130], text='原', conf=.6)
                rec = Mock(side_effect=[(right, confidence), (left, confidence), ('原', .9)])
                _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=rec,
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2, chars)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(bool(traces), accepted)

    def test_single_track_garbage_falls_back_to_large(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        seg = ColumnSegment('single_track', 0, 100, 20, 36, alignment='right')
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        for text in ['お力口；K', '2口', '#']:
            with self.subTest(text=text):
                b = Block(bbox=[10, 10, 50, 130], text='原', conf=.6)
                rec = Mock(side_effect=[(text, .9), ('原', .9)])
                _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=rec,
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(traces, [])
                self.assertEqual(rec.call_count, 1)

    def test_short_split_without_cue_rejected_even_with_high_confidence(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        seg = ColumnSegment('dual_track', 0, 100, 2, 38,
                            (22, 0, 38, 100), (2, 0, 18, 100), 20)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        for probability, confidence, accepted in [(.0, .99, False), (.54, .85, False), (.55, .9, False), (.54, .84, False)]:
            with self.subTest(probability=probability, confidence=confidence):
                chars = [CharCTC('原', probability, 0, 2, 2, 0, .6)]
                b = Block(bbox=[10, 10, 50, 130], text='原', conf=.6)
                rec = Mock(side_effect=[('良干', confidence), ('佳王', confidence), ('原', .9)])
                _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=rec,
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2, chars)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(bool(traces), accepted)

    def test_high_confidence_reassembly_cannot_degrade(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        seg = ColumnSegment('single_track', 0, 100, 20, 36, alignment='right')
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        for confidence, accepted in [(.7, False), (.92, False), (.93, True)]:
            with self.subTest(confidence=confidence):
                b = Block(bbox=[10, 10, 50, 130], text='原', conf=.95)
                _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=Mock(return_value=('二錢', confidence)),
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(bool(traces), accepted)

    def test_single_track_upward_expansion_is_limited_by_gap(self):
        image = np.full((160, 70, 3), 245, np.uint8)
        b = Block(bbox=[10, 10, 50, 150], text='原', conf=.6)
        segs = [ColumnSegment('large', 0, 30, 2, 38),
                ColumnSegment('single_track', 50, 90, 20, 36, alignment='right')]
        analysis = ColumnSegmentAnalysis(40, 140, 2, 38, 40, segs)
        _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=Mock(side_effect=[('二錢', .9), ('原', .9)]),
            ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
        self.assertTrue(traces)
        self.assertEqual(b.segments[1]['bbox'][1], 46)
        self.assertGreaterEqual(b.segments[1]['bbox'][1], 10 + segs[0].y2)

    def test_first_single_track_upward_expansion_uses_full_gap(self):
        image = np.full((160, 70, 3), 245, np.uint8)
        b = Block(bbox=[10, 10, 50, 150], text='原', conf=.6)
        seg = ColumnSegment('single_track', 8, 90, 20, 36, alignment='right')
        analysis = ColumnSegmentAnalysis(40, 140, 2, 38, 40, [seg])
        _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=Mock(return_value=('二錢', .9)),
            ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
        self.assertTrue(traces)
        self.assertEqual(b.segments[0]['bbox'][1], 10)

    def test_order_and_anchors(self):
        image = np.full((230, 100, 3), 245, np.uint8)
        b = Block(page_id='p', line_id='r', bbox=[20, 10, 60, 210], text='茯苓澤瀉', conf=.60)
        segs = [ColumnSegment('large', 5, 45, 3, 37),
                ColumnSegment('single_track', 50, 85, 23, 36, alignment='right'),
                ColumnSegment('large', 90, 130, 3, 37),
                ColumnSegment('dual_track', 135, 190, 3, 37,
                              right_box=(22, 135, 37, 190), left_box=(3, 135, 18, 190), split_x=20)]
        analysis = ColumnSegmentAnalysis(40, 200, 3, 37, 40, segs)
        rec = Mock(side_effect=[('二錢', .8), ('一錢', .8), ('五分', .8), ('茯苓', .9), ('澤瀉', .9)])
        blocks, traces = apply_column_seg_jiazhu(image, [b], rec_fn=rec,
            ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
        self.assertEqual(blocks[0].text, '茯苓二錢澤瀉一錢五分')
        self.assertEqual([s['seg_type'] for s in b.segments], ['large', 'single_track', 'large', 'dual_right', 'dual_left'])
        self.assertEqual([s['bbox'] for s in b.segments], [[23, 13, 57, 57], [38, 58, 57, 97], [23, 98, 57, 142], [40, 143, 57, 205], [23, 143, 40, 205]])
        self.assertEqual(b.segments[3]['group_id'], b.segments[4]['group_id'])
        self.assertEqual(len(traces), 1)
        self.assertTrue(all(s['conf'] > 0 for s in b.segments))

    def test_no_small_and_failed_acceptance(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        original = Block(line_id='r', bbox=[10, 10, 50, 130], text='茯苓澤瀉', conf=.7)
        large = ColumnSegment('large', 0, 110, 2, 38)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [large])
        rec = Mock()
        _, traces = apply_column_seg_jiazhu(image, [original], rec_fn=rec,
            ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
        rec.assert_not_called()
        self.assertEqual(traces, [])
        small = ColumnSegment('single_track', 0, 110, 20, 36, alignment='right')
        analysis.segments = [small]
        for result in [('亂碼', .4), ('a!?', .9), ('茯苓', .9)]:
            with self.subTest(result=result):
                block = replace(original)
                _, traces = apply_column_seg_jiazhu(image, [block], rec_fn=Mock(return_value=result),
                    ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)), seg_fn=Mock(return_value=analysis))
                self.assertEqual(traces, [])
                self.assertEqual(block.text, original.text)

    def test_empty_dual_side_is_not_inserted(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        block = Block(line_id='r', bbox=[10, 10, 50, 130], text='茯苓', conf=.6)
        dual = ColumnSegment('dual_track', 0, 100, 2, 38,
                             (22, 0, 38, 100), (2, 0, 18, 100), 20)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [dual])
        _, traces = apply_column_seg_jiazhu(image, [block],
            rec_fn=Mock(side_effect=[('茯苓二錢', .8), ('亂碼', .2)]),
            ctc_rec_fn=Mock(return_value=ctc(.6, .1, .2)),
            seg_fn=Mock(return_value=analysis))
        self.assertEqual(block.text, '茯苓二錢')
        self.assertEqual(len(traces), 1)
        self.assertEqual([s['seg_type'] for s in block.segments], ['dual_right'])

    def test_radical_split_rejected(self):
        image = np.full((150, 70, 3), 245, np.uint8)
        b = Block(line_id='r', bbox=[10, 10, 50, 130], text='諸脹', conf=.7)
        chars = [CharCTC('諸', .9, 0, 2, 2, 0, .3), CharCTC('脹', .9, 1, 2, 2, .3, .6)]
        seg = ColumnSegment('dual_track', 0, 60, 2, 38, (22, 0, 38, 60), (2, 0, 18, 60), 20)
        analysis = ColumnSegmentAnalysis(40, 120, 2, 38, 40, [seg])
        rec = Mock(side_effect=[('复長', .8), ('月月', .8), ('諸脹', .9)])
        _, traces = apply_column_seg_jiazhu(image, [b], rec_fn=rec,
            ctc_rec_fn=Mock(return_value=ctc(.8, .06, .4, chars)), seg_fn=Mock(return_value=analysis))
        self.assertEqual(rec.call_count, 2)
        self.assertEqual(b.text, '諸脹')
        self.assertEqual(traces, [])


class TestClosingConstraints(unittest.TestCase):
    def setUp(self):
        from ocr.column_jiazhu import GateDecision
        self.image = np.full((220, 50, 3), 245, np.uint8)
        self.block = Block(bbox=[0, 0, 40, 200], text='原', conf=.6)
        self.single = ColumnSegment('single_track', 50, 90, 20, 36, alignment='right')
        self.large = ColumnSegment('large', 0, 40, 2, 38)
        self.dual = ColumnSegment('dual_track', 120, 150, 2, 38,
                                  (22, 120, 38, 150), (2, 120, 18, 150), 20)
        self.gate = GateDecision(True, ['low_conf'], ctc(.6, .1, .2), 1, 0, 0)

    def assemble(self, segs, answers, block=None, gate=None, **kwargs):
        from ocr.column_jiazhu import reassemble_column_block
        analysis = ColumnSegmentAnalysis(40, 200, 2, 38, 40, segs)
        rec = Mock(side_effect=answers)
        with patch('ocr.column_jiazhu.evaluate_column_gate', return_value=gate or self.gate):
            result, _ = reassemble_column_block(self.image, block or self.block,
                                                 seg_fn=Mock(return_value=analysis), rec_fn=rec, **kwargs)
        return result, rec

    def test_helpers_and_normalization(self):
        from ocr.column_jiazhu import (count_strict_dosage_cues, count_distinct_herbs,
                                       has_herb_entity, load_tcm_herbs)
        for text in ['二錢', '一兩半', '酒製', '土炒', '生用']:
            self.assertGreaterEqual(count_strict_dosage_cues(text), 1)
        self.assertEqual(count_strict_dosage_cues('分心可見'), 0)
        self.assertTrue(has_dosage_or_jiazhu_cue('生用'))
        self.assertTrue(has_herb_entity('白礬茯苓'))
        self.assertGreaterEqual(count_distinct_herbs('白礬茯苓白礬'), 2)
        from pathlib import Path
        with patch('ocr.column_jiazhu.PROJECT_ROOT', Path('missing-lexicon-root')):
            load_tcm_herbs.cache_clear()
            self.assertEqual(load_tcm_herbs(), frozenset())
        load_tcm_herbs.cache_clear()

    def test_single_gate_confidence_and_exceptions(self):
        from ocr.column_jiazhu import GateDecision
        block = replace(self.block, conf=.76)
        for reasons, minimum, accepted in [(['low_conf'], .45, False), (['low_conf'], .44, True),
                                           (['density_under'], .8, True)]:
            gate = GateDecision(True, reasons, ctc(minimum=minimum), 1, 0, 0)
            result, rec = self.assemble([self.single], [('二錢', .9)], block, gate)
            self.assertEqual(result is not None, accepted)
            self.assertEqual(rec.call_count, int(accepted))

    def test_confidence_guard_and_no_cue_guard(self):
        for old, new in [(.79, .69), (.75, .67)]:
            result, _ = self.assemble([self.single], [('二錢', new)], replace(self.block, conf=old))
            self.assertIsNone(result)
        for new, accepted in [(.70, False), (.71, True)]:
            result, _ = self.assemble([self.large, self.single], [('天地', .8), ('茯苓', 2 * new - .8)],
                                      replace(self.block, conf=.70))
            self.assertEqual(result is not None, accepted)

    def test_single_semantics_and_herb_anchor(self):
        for cue in ['二錢', '生用']:
            result, _ = self.assemble([self.single], [(cue, .8)])
            self.assertIsNotNone(result)
        for align, confidence, anchor, accepted in [('right', .8, '白礬', True),
                ('right', .8, '天地', False), ('left', .8, '白礬', False),
                ('right', .74, '白礬', False), ('right', .8, '茯苓', True)]:
            single = replace(self.single, alignment=align)
            result, _ = self.assemble([self.large, single], [('天地', confidence), (anchor, .9)])
            self.assertEqual(result is not None, accepted)

    def test_rejected_runs_are_merged_and_empty_has_no_fallback(self):
        rejected = replace(self.single, alignment='left')
        result, rec = self.assemble([self.large, rejected], [('天地', .8)])
        self.assertIsNone(result)
        self.assertEqual(rec.call_count, 1)
        segs = [self.large, rejected, ColumnSegment('large', 95, 115, 2, 38), self.dual,
                ColumnSegment('large', 160, 190, 2, 38)]
        result, rec = self.assemble(segs, [('天地', .8), ('二錢', .9), ('酒洗', .9),
                                           ('合併正文', .9), ('尾文', .9)])
        self.assertIsNotNone(result)
        self.assertEqual(rec.call_count, 5)
        self.assertEqual([e['seg_type'] for e in result['segments']],
                         ['large', 'dual_right', 'dual_left', 'large'])
        self.assertEqual(result['segments'][0]['bbox'][1:4:2], [0, 117])

    def test_herb_anchor_large_read_is_reused(self):
        result, rec = self.assemble([self.large, self.single], [('天地', .9), ('茯苓', .9)])
        self.assertIsNotNone(result)
        self.assertEqual(rec.call_count, 2)
        self.assertEqual(result['segments'][0]['text'], '茯苓')

    def test_cursive_dual_requires_strict_cue_and_confidence(self):
        for right, left, confidence, accepted in [('二錢', '天地', .85, True),
                ('二錢', '天地', .84, False), ('可見', '天地', .95, False)]:
            result, _ = self.assemble([self.dual], [(right, confidence), (left, confidence)],
                                      cursive_strict=True)
            self.assertEqual(result is not None, accepted)

    def test_strict_modes_and_gain(self):
        for align, text, conf, old, accepted in [
                ('left', '二錢', .95, .6, False), ('right', '二錢', .84, .6, False),
                ('right', '二錢', .85, .81, False), ('right', '二錢', .90, .81, True)]:
            result, _ = self.assemble([replace(self.single, alignment=align)], [(text, conf)],
                                      replace(self.block, conf=old), strict_single_track=True)
            self.assertEqual(result is not None, accepted)
        for text, conf, old, accepted in [('生用', .84, .6, False), ('可見', .95, .6, False),
                                          ('生用', .85, .80, False), ('生用', .90, .80, True)]:
            result, _ = self.assemble([self.single], [(text, conf)], replace(self.block, conf=old), cursive_strict=True)
            self.assertEqual(result is not None, accepted)

    def test_page_prior_and_upgrade(self):
        from ocr.column_jiazhu import evaluate_page_prior
        def blocks(text='天地', confidence=.75):
            return [Block(line_id=str(i), bbox=[0, 0, 40, 200], text=text, conf=confidence) for i in range(6)]
        plain = blocks()
        prior = evaluate_page_prior(plain)
        self.assertFalse(prior.has_text_prior)
        self.assertFalse(prior.low_quality_cursive)
        self.assertTrue(evaluate_page_prior(blocks('茯苓白礬甘草')).has_text_prior)
        self.assertTrue(evaluate_page_prior(blocks('二錢')).has_text_prior)
        calls = []
        def fake(image, block, **kwargs):
            calls.append((block.line_id, kwargs.get('strict_single_track', False)))
            if block.line_id == '0' and kwargs.get('strict_single_track'):
                return ({'text': '二錢', 'conf': .9, 'segments': [{'seg_type': 'dual_right', 'conf': .9}], 'trace': {}}, None)
            return None, None
        with patch('ocr.column_jiazhu.reassemble_column_block', side_effect=fake):
            apply_column_seg_jiazhu(self.image, plain)
        self.assertEqual(len(calls), 11)
        self.assertEqual(calls[6:], [(str(i), False) for i in range(1, 6)])
        with patch('ocr.column_jiazhu.reassemble_column_block', return_value=(None, None)) as run:
            apply_column_seg_jiazhu(self.image, blocks('茯苓白礬甘草'))
        self.assertTrue(all('strict_single_track' not in call.kwargs for call in run.call_args_list))

    def test_low_quality_page_fuse_and_strict(self):
        from ocr.column_jiazhu import evaluate_page_prior
        blocks = [Block(bbox=[0, 0, 40, 200], text='天地', conf=.55) for _ in range(6)]
        self.assertTrue(evaluate_page_prior(blocks).low_quality_cursive)
        with patch('ocr.column_jiazhu.reassemble_column_block') as run:
            self.assertEqual(apply_column_seg_jiazhu(self.image, blocks)[1], [])
        run.assert_not_called()
        blocks[0].text = '二錢'
        with patch('ocr.column_jiazhu.reassemble_column_block', return_value=(None, None)) as run:
            apply_column_seg_jiazhu(self.image, blocks)
        self.assertEqual(run.call_count, 6)
        self.assertTrue(all(call.kwargs['cursive_strict'] for call in run.call_args_list))


class TestPipeline(unittest.TestCase):
    def run_case(self, enabled, apply):
        from ocr import pipeline
        cfg = replace(CONFIG, ocr=replace(CONFIG.ocr, col_seg_jiazhu=enabled, merge_jiazhu=False,
                                          correct_enabled=True, normalize_enabled=True))
        blocks = [Block(line_id='a', text='原文'), Block(line_id='b', text='正文')]
        rec = RecognizeResult(blocks=blocks, width=50, height=100, is_demo=False)
        with patch.object(pipeline, 'CONFIG', cfg), patch.object(pipeline, '_recognize_page', return_value=rec), \
             patch.object(pipeline, '_postprocess', return_value=(blocks, LayoutResult(), [])), \
             patch.object(pipeline, '_attach_dual_track', return_value={}), \
             patch('ocr.vertical.imread_unicode', return_value=np.ones((100, 50, 3), np.uint8)), \
             patch('ocr.column_jiazhu.apply_column_seg_jiazhu', apply), \
             patch.object(pipeline.correct, 'correct_blocks', side_effect=lambda bs: (bs, [])) as correct, \
             patch.object(pipeline.normalize, 'normalize_blocks', side_effect=lambda bs: bs) as norm:
            result = pipeline.run_ocr('image.png', 'p')
        return result, correct, norm

    def test_disabled(self):
        apply = Mock()
        self.run_case(False, apply)
        apply.assert_not_called()

    def test_enabled_and_exception(self):
        def update(image, blocks, **kwargs):
            blocks[0].text = '新文'
            blocks[0].segments = [{'text': '錢'}]
            return blocks, [{'line_id': 'a'}]
        result, correct, norm = self.run_case(True, Mock(side_effect=update))
        self.assertEqual([b.line_id for b in correct.call_args.args[0]], ['a'])
        self.assertEqual([b.line_id for b in norm.call_args.args[0]], ['a'])
        from ocr.normalize import normalize_text
        self.assertEqual(result.blocks[0].segments[0]['text_norm'], normalize_text('錢'))
        self.assertTrue(any('列内分段夹注重组：1 块' in n for n in result.notes))
        result, _, _ = self.run_case(True, Mock(side_effect=RuntimeError()))
        self.assertEqual(result.blocks[0].text, '原文')
        self.assertTrue(any('异常已隔离' in n for n in result.notes))
        self.assertFalse(result.is_demo)


if __name__ == '__main__':
    unittest.main()
