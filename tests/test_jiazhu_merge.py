import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from app.config import CONFIG
from app.schemas import Block, LayoutResult
from ocr.jiazhu_merge import select_verified_jiazhu_groups, splice_jiazhu_into_blocks
from ocr.recognize import RecognizeResult


def evidence(gid, right, left):
    return [{'group_id': gid, 'side': 'right', 'bbox': right},
            {'group_id': gid, 'side': 'left', 'bbox': left}]


class TestJiazhuMerge(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((160, 120, 3), dtype=np.uint8)
        self.pair = evidence('a', [44, 50, 60, 80], [28, 50, 44, 80])
        self.group = {'group_id': 'a', 'bbox': [28, 50, 60, 80],
                      'right_text': '右注', 'left_text': '左注', 'conf': .8}

    def test_verify_geometry_and_both_readings(self):
        narrow = evidence('bad', [50, 50, 64, 80], [28, 50, 48, 80])
        rec = Mock(side_effect=[('右注', .9), ('左注', .8)])
        groups = select_verified_jiazhu_groups(self.image, narrow + self.pair, rec)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['text'], '右注左注')
        self.assertEqual(groups[0]['conf'], .8)
        self.assertEqual(rec.call_count, 2)
        self.assertEqual(rec.call_args_list[0].args[0].shape, (32, 60, 3))
        for readings in ((('右注', .9), ('', .9)), (('右注', .9), ('左注', .54)),
                         (('右注', .54), ('左注', .9)), (('abc', .9), ('左注', .9))):
            with self.subTest(readings=readings):
                self.assertEqual(select_verified_jiazhu_groups(self.image, self.pair,
                                                               Mock(side_effect=readings)), [])

    def test_splice_and_trace(self):
        block = Block(line_id='line', bbox=[25, 10, 63, 120], text='原文')
        rec = Mock(side_effect=[('前文', .9), ('後文', .9)])
        blocks, traces = splice_jiazhu_into_blocks(self.image, [block], [self.group], rec)
        self.assertEqual(blocks[0].text, '前文右注左注後文')
        self.assertEqual(block.quote, block.text)
        self.assertEqual(traces, [{'line_id': 'line', 'before': '原文', 'after': block.text, 'groups': ['a']}])
        self.assertEqual([c.args[0].shape for c in rec.call_args_list], [(40, 38, 3), (40, 38, 3)])

    def test_guard_does_not_replace_equal_length(self):
        block = Block(line_id='line', bbox=[25, 10, 63, 120], text='一二三四五六七八')
        _, traces = splice_jiazhu_into_blocks(self.image, [block], [self.group],
                                              Mock(side_effect=[('前文', .9), ('後文', .9)]))
        self.assertEqual(block.text, '一二三四五六七八')
        self.assertEqual(block.quote, block.text)
        self.assertEqual(traces, [])

    def test_overlap_dedup_and_non_paragraph(self):
        weaker = dict(self.group, group_id='weaker', bbox=[28, 55, 60, 82], conf=.7)
        block = Block(line_id='line', bbox=[25, 10, 63, 120], text='原')
        title = Block(line_id='title', bbox=[25, 10, 63, 120], text='标题', kind='title')
        _, traces = splice_jiazhu_into_blocks(self.image, [block, title], [weaker, self.group],
                                              Mock(side_effect=[('前文', .9), ('後文', .9)]))
        self.assertEqual(traces[0]['groups'], ['a'])
        self.assertEqual(title.text, '标题')

    def test_enabled_pipeline_reprocesses_only_changed_blocks(self):
        from ocr import pipeline
        config = replace(CONFIG, ocr=replace(CONFIG.ocr, merge_jiazhu=True,
                                             correct_enabled=True, normalize_enabled=True))
        block = Block(line_id='line', bbox=[25, 10, 63, 120], text='原文')
        other = Block(line_id='other', bbox=[70, 10, 100, 120], text='正文')
        rec = RecognizeResult(blocks=[block, other], width=120, height=160, is_demo=False)
        dual = {'dual_track_status': 'completed', 'dual_track_evidence': self.pair,
                'dual_track_audit': [], 'dual_track_run': True}
        def splice(image, blocks, groups):
            blocks[0].text = blocks[0].quote = '回插文字'
            return blocks, [{'line_id': 'line', 'groups': ['a']}]
        with patch.object(pipeline, 'CONFIG', config), \
             patch.object(pipeline, '_recognize_page', return_value=rec), \
             patch.object(pipeline, '_postprocess', return_value=([block, other], LayoutResult(), [])), \
             patch.object(pipeline, '_attach_dual_track', return_value=dual), \
             patch('ocr.vertical.imread_unicode', return_value=self.image), \
             patch('ocr.jiazhu_merge.select_verified_jiazhu_groups', return_value=[self.group]), \
             patch('ocr.jiazhu_merge.splice_jiazhu_into_blocks', side_effect=splice), \
             patch.object(pipeline.correct, 'correct_blocks', side_effect=lambda bs: (bs, [])) as correct, \
             patch.object(pipeline.normalize, 'normalize_blocks', side_effect=lambda bs: bs) as normalize:
            result = pipeline.run_ocr('image.png', 'P')
        self.assertEqual(result.blocks[0].text, '回插文字')
        self.assertIs(result.blocks[1], other)
        self.assertEqual([b.line_id for b in correct.call_args.args[0]], ['line'])
        self.assertEqual([b.line_id for b in normalize.call_args.args[0]], ['line'])
        self.assertTrue(any(n.startswith('双行夹注回插：') for n in result.notes))

    def test_default_disabled_in_pipeline(self):
        from ocr import pipeline
        self.assertFalse(CONFIG.ocr.merge_jiazhu)
        config = replace(CONFIG, ocr=replace(CONFIG.ocr, merge_jiazhu=False))
        block = Block(line_id='line', bbox=[25, 10, 63, 120], text='原文')
        rec = RecognizeResult(blocks=[block], width=120, height=160, is_demo=False)
        dual = {'dual_track_status': 'completed', 'dual_track_evidence': self.pair,
                'dual_track_audit': [], 'dual_track_run': True}
        with patch.object(pipeline, 'CONFIG', config), \
             patch.object(pipeline, '_recognize_page', return_value=rec), \
             patch.object(pipeline, '_postprocess', return_value=([block], LayoutResult(), [])), \
             patch.object(pipeline, '_attach_dual_track', return_value=dual), \
             patch('ocr.jiazhu_merge.select_verified_jiazhu_groups') as select, \
             patch('ocr.jiazhu_merge.splice_jiazhu_into_blocks') as splice:
            result = pipeline.run_ocr('image.png', 'P')
        self.assertFalse(result.is_demo)
        select.assert_not_called()
        splice.assert_not_called()


if __name__ == '__main__':
    unittest.main()
