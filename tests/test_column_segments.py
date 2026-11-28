import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ocr.column_segments import (RuleSegmentClassifier, SegmentFeatures, render_column_segments_debug,
                                 sauvola_binarize, segment_column, strip_column_borders)


def rect(im, x1, y1, x2, y2, value=25):
    im[y1:y2, x1:x2] = value


def large(im, y):
    rect(im, 12, y, 34, y+70)
    rect(im, 47, y, 70, y+70)


def dual(im, y, n=2):
    for i in range(n):
        rect(im, 12, y+i*34, 34, y+i*34+26)
    rect(im, 49, y, 71, y+26)


class ColumnSegmentsTests(unittest.TestCase):
    def test_sauvola_gradient_and_showthrough(self):
        image = np.tile(np.linspace(200, 245, 100, dtype=np.uint8), (180, 1))
        rect(image, 25, 35, 70, 80, 30)
        rect(image, 35, 120, 65, 145, 175)
        bw = sauvola_binarize(image)
        self.assertGreater(np.mean(bw[35:80,25:70] == 255), .95)
        self.assertLess(np.mean(bw[120:145,35:65] == 255), .05)
        self.assertEqual(set(np.unique(bw)).difference({0,255}), set())

    def test_borders(self):
        bw = np.zeros((250, 100), np.uint8)
        bw[:, 3:6] = 255; bw[:, 94:97] = 255
        bw[0:2, 5:95] = 255; bw[-2:, 5:95] = 255
        bw[50:90, 25:75] = 255
        original = bw.copy()
        clean, x1, x2 = strip_column_borders(bw)
        self.assertEqual((x1,x2), (25,75))
        self.assertFalse(clean[:, :x1].any()); self.assertFalse(clean[:, x2:].any())
        self.assertTrue(np.array_equal(bw, original))
        self.assertFalse(clean[0].any())

    def test_connected_large_to_small_tracks(self):
        for is_dual in (False, True):
            with self.subTest(dual=is_dual):
                im = np.full((220, 80), 245, np.uint8)
                large(im, 15)
                if is_dual:
                    dual(im, 96)
                    rect(im, 52, 85, 54, 96)
                else:
                    rect(im, 52, 96, 70, 121)
                    rect(im, 52, 130, 70, 155)
                    rect(im, 53, 85, 55, 96)
                analysis = segment_column(im)
                self.assertGreaterEqual(len(analysis.segments), 2)
                self.assertEqual(analysis.segments[0].seg_type, 'large')
                self.assertEqual(analysis.segments[1].seg_type, 'dual_track' if is_dual else 'single_track')
                self.assertLess(abs(analysis.segments[0].y2-90), 12)

    def test_mixed_column_and_unequal_tracks(self):
        im = np.full((560, 80), 245, np.uint8)
        large(im, 10)
        rect(im, 52, 90, 70, 115); rect(im, 52, 122, 70, 147)
        dual(im, 160)
        large(im, 245)
        rect(im, 29, 330, 50, 355); rect(im, 29, 363, 50, 388)
        a = segment_column(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), keep_bw=True)
        self.assertEqual([s.seg_type for s in a.segments], ['large','single_track','dual_track','large','single_track'])
        self.assertEqual(a.segments[1].alignment, 'right')
        self.assertEqual(a.segments[-1].alignment, 'center')
        d = a.segments[2]
        self.assertLess(d.left_box[0], d.split_x); self.assertGreaterEqual(d.right_box[0], d.split_x)
        self.assertGreater(d.left_box[3], d.right_box[3])
        self.assertEqual(len(d.features.h_profile_32), 32)

    def test_large_radicals_are_not_dual(self):
        im = np.full((300, 80), 245, np.uint8)
        for y in (10, 95, 180): large(im, y)
        a = segment_column(im)
        self.assertEqual([s.seg_type for s in a.segments], ['large'])
        self.assertEqual((a.segments[0].y1,a.segments[0].y2), (10,250))

    def test_classifier_injection(self):
        class Custom:
            def __init__(self): self.calls = []
            def classify(self, features, *, adjacent_dual=False):
                self.calls.append((features, adjacent_dual))
                return 'single_track', .93, 'left'
        im = np.full((120, 80), 245, np.uint8); large(im, 10)
        c = Custom(); a = segment_column(im, classifier=c)
        self.assertTrue(c.calls); self.assertEqual(a.segments[0].seg_type, 'single_track')
        self.assertAlmostEqual(a.segments[0].score, .93)
        self.assertEqual(a.segments[0].alignment, 'left')

    def test_debug_png(self):
        im = np.full((120, 80, 3), 245, np.uint8); im[10:70, 15:60] = 10
        a = segment_column(im)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '切片.png'
            result = render_column_segments_debug(im, a, path)
            self.assertEqual(result.shape, (120,240,3)); self.assertTrue(path.is_file())
            self.assertIsNotNone(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR))

    def test_slanted_borders_remove_outer_marks(self):
        im = np.full((300, 100), 245, np.uint8)
        for y in range(8, 292):
            if y % 95 not in (0, 1):
                x = 7 + (y // 80) % 2
                im[y, x:x+2] = 20
                x = 91 + (y // 80) % 2
                im[y, x:x+2] = 20
        rect(im, 2, 125, 5, 132)
        rect(im, 95, 150, 98, 157)
        rect(im, 25, 30, 75, 75)
        rect(im, 57, 120, 74, 140)
        rect(im, 57, 150, 74, 170)
        a = segment_column(im, keep_bw=True)
        self.assertFalse(a.clean_bw[:, :10].any())
        self.assertFalse(a.clean_bw[:, 91:].any())
        self.assertEqual((a.inner_x1, a.inner_x2), (25, 75))
        self.assertEqual([(s.seg_type, s.alignment) for s in a.segments],
                         [('large', 'center'), ('single_track', 'right')])

    def test_wide_blank_margins_use_activity_width(self):
        im = np.full((200, 80), 245, np.uint8)
        for y in (10, 60):
            rect(im, 20, y, 60, y+38)
        rect(im, 43, 115, 59, 132)
        rect(im, 43, 140, 59, 157)
        a = segment_column(im)
        self.assertEqual((a.inner_x1, a.inner_x2), (20, 60))
        self.assertEqual([s.seg_type for s in a.segments], ['large', 'single_track'])
        self.assertEqual(a.segments[1].alignment, 'right')

    def test_stroke_gaps_do_not_count_as_characters(self):
        from ocr.column_segments import _features
        bw = np.zeros((42, 40), np.uint8)
        for y in range(0, 40, 5):
            bw[y:y+4, 2:17] = 255
            bw[y:y+4, 23:38] = 255
        f = _features(bw, 40)
        self.assertEqual((f.left_sub_blobs, f.right_sub_blobs), (1, 1))
        self.assertEqual(RuleSegmentClassifier().classify(f)[0], 'large')

    def test_sparse_dual_and_one_sided_tail(self):
        im = np.full((200, 80), 245, np.uint8)
        rect(im, 10, 5, 70, 58)
        rect(im, 12, 80, 32, 101)
        rect(im, 49, 90, 69, 92)
        rect(im, 12, 110, 32, 131)
        rect(im, 49, 110, 69, 131)
        rect(im, 49, 140, 69, 160)
        a = segment_column(im)
        self.assertEqual([s.seg_type for s in a.segments], ['large', 'dual_track'])
        d = a.segments[1]
        self.assertEqual(d.y1, 80)
        self.assertEqual(d.right_box[3], 160)

    def test_closing_keeps_nearby_track_characters_separate(self):
        from ocr.column_segments import _count_track_chars
        part = np.zeros((40, 20), np.uint8)
        part[2:15, :] = 255
        part[19:32, :] = 255
        self.assertEqual(_count_track_chars(part, 60), 2)

    def test_tiny_noise_is_discarded(self):
        im = np.full((180, 80), 245, np.uint8)
        rect(im, 15, 10, 65, 55)
        rect(im, 40, 95, 41, 97)
        rect(im, 32, 120, 35, 123)
        a = segment_column(im)
        self.assertEqual(len(a.segments), 1)
        self.assertEqual(a.segments[0].seg_type, 'large')

    def test_dashed_borders_and_outer_marks_are_removed(self):
        bw = np.zeros((300, 100), np.uint8)
        for y in range(0, 300, 10):
            bw[y:y+6, 7:9] = 255
            bw[y:y+6, 91:93] = 255
        bw[120:125, 2:4] = 255
        bw[150:155, 96:98] = 255
        bw[60:110, 25:75] = 255
        clean, x1, x2 = strip_column_borders(bw)
        self.assertEqual((x1, x2), (25, 75))
        self.assertFalse(clean[:, :25].any())
        self.assertFalse(clean[:, 75:].any())
        self.assertTrue(clean[60:110, 25:75].all())

    def test_tall_right_track_and_thin_horizontal_stroke(self):
        im = np.full((200, 100), 245, np.uint8)
        rect(im, 20, 10, 80, 70)
        rect(im, 51, 90, 80, 133)
        rect(im, 55, 150, 78, 153)
        a = segment_column(im)
        self.assertEqual([(s.seg_type, s.alignment) for s in a.segments],
                         [('large', 'center'), ('single_track', 'right')])
        self.assertEqual(a.segments[1].y2, 153)
        self.assertAlmostEqual(a.segments[1].features.height_ratio, 63/60)

    def test_tall_narrow_large_characters_merge_on_both_sides(self):
        for x1, x2 in ((20, 42), (58, 80)):
            with self.subTest(x1=x1):
                im = np.full((200, 100), 245, np.uint8)
                rect(im, 20, 10, 80, 70)
                rect(im, x1, 80, x2, 132)
                a = segment_column(im, keep_bw=True)
                self.assertEqual(a.large_char_h, 60)
                from ocr.column_segments import _features
                f = _features(a.clean_bw[80:132, a.inner_x1:a.inner_x2], 60)
                self.assertLess(f.width_ratio, .62)
                self.assertEqual(RuleSegmentClassifier().classify(f)[0], 'single_track')
                self.assertEqual([(s.seg_type, s.y1, s.y2) for s in a.segments],
                                 [('large', 10, 132)])

    def test_three_right_track_characters_survive_total_height(self):
        im = np.full((250, 100), 245, np.uint8)
        rect(im, 20, 10, 80, 70)
        for y in (90, 128, 166):
            rect(im, 58, y, 80, y+30)
        a = segment_column(im)
        self.assertEqual(a.large_char_h, 60)
        self.assertEqual([(s.seg_type, s.alignment, s.y1, s.y2) for s in a.segments],
                         [('large', 'center', 10, 70),
                          ('single_track', 'right', 90, 196)])

    def test_single_track_group_height_review(self):
        for heights, expected in (((4, 31), 'single_track'),
                                  ((4, 33), 'large'),
                                  ((24, 24), 'large')):
            with self.subTest(heights=heights):
                im = np.full((180, 80), 245, np.uint8)
                rect(im, 20, 10, 60, 38)
                y = 65
                for height in heights:
                    rect(im, 48, y, 60, y + height)
                    y += height + 4
                a = segment_column(im)
                self.assertEqual(a.large_char_h, 28)
                self.assertEqual([(s.seg_type, s.alignment, s.y1, s.y2)
                                  for s in a.segments],
                                 [('large', 'center', 10, 38),
                                  (expected, 'right' if expected == 'single_track' else 'center',
                                   65, y - 4)])

    def test_isolated_nonright_small_character_is_rejected(self):
        # A centered candidate is recognized through its neighboring dual row.
        for x1, x2 in ((20, 42), (40, 60)):
            with self.subTest(x1=x1):
                im = np.full((240, 100), 245, np.uint8)
                rect(im, 20, 10, 80, 70)
                rect(im, x1, 80, x2, 110)
                rect(im, 21, 130, 42, 150)
                rect(im, 58, 130, 79, 150)
                a = segment_column(im, keep_bw=True)
                from ocr.column_segments import _features
                f = _features(a.clean_bw[80:110, a.inner_x1:a.inner_x2], 60)
                self.assertEqual(RuleSegmentClassifier().classify(f, adjacent_dual=True)[0],
                                 'single_track')
                self.assertEqual([(s.seg_type, s.y1, s.y2) for s in a.segments],
                                 [('large', 10, 110), ('dual_track', 130, 150)])

    def test_large_height_review_prevents_clean_seam_split(self):
        im = np.full((180, 100), 245, np.uint8)
        rect(im, 20, 10, 80, 60)
        rect(im, 21, 90, 42, 128)
        rect(im, 58, 90, 79, 128)
        a = segment_column(im, keep_bw=True)
        self.assertEqual(a.large_char_h, 50)
        # This band initially satisfies the classifier's small-pair rule.
        from ocr.column_segments import _features
        f = _features(a.clean_bw[90:128, a.inner_x1:a.inner_x2], 60)
        self.assertEqual(RuleSegmentClassifier().classify(f)[0], 'dual_track')
        self.assertEqual([s.seg_type for s in a.segments], ['large', 'large'])
        self.assertEqual(a.segments[1].y2, 128)

    def test_dual_merge_precedes_large_review_and_single_track_merge(self):
        im = np.full((500, 100), 245, np.uint8)
        rect(im, 20, 10, 80, 60)  # Full-width large character: height 50.
        for y, height in ((100, 20), (130, 43)):
            rect(im, 21, y, 42, y+height)
            rect(im, 58, y, 79, y+height)
        # An isolated clean-seam character, 0.82 times the large height.
        rect(im, 21, 220, 42, 261)
        rect(im, 58, 220, 79, 261)
        for y in (320, 376, 432):
            rect(im, 58, y, 79, y+20)

        a = segment_column(im)
        self.assertEqual(a.large_char_h, 50)
        self.assertEqual([(s.seg_type, s.alignment) for s in a.segments],
                         [('large', 'center'), ('dual_track', 'center'),
                          ('large', 'center'), ('single_track', 'right')])
        dual_segment = a.segments[1]
        self.assertEqual((dual_segment.y1, dual_segment.y2), (100, 173))
        self.assertGreaterEqual(max(dual_segment.features.left_sub_blobs,
                                    dual_segment.features.right_sub_blobs), 2)
        self.assertEqual((a.segments[2].y1, a.segments[2].y2), (220, 261))
        self.assertEqual((a.segments[3].y1, a.segments[3].y2), (320, 452))

    def test_empty_and_tiny(self):
        for shape in ((0,0), (0,5), (5,0), (1,1), (2,2)):
            im = np.full(shape, 255, np.uint8)
            a = segment_column(im, keep_bw=True)
            self.assertEqual(a.segments, [])
            self.assertEqual(a.clean_bw.shape, shape)
            self.assertEqual(render_column_segments_debug(im,a).shape, (shape[0],shape[1]*3,3))


if __name__ == '__main__':
    unittest.main()

