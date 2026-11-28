import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ocr.clean_column import _long_lines, clean_body_crop, remove_vertical_grid_lines, strip_side_kunten
from ocr.frame import recognize_with_frame_isolation


class CleanColumnTests(unittest.TestCase):
    def test_long_lines_detects_broken_ink(self):
        binary = np.zeros((200, 100), np.uint8)
        binary[10:95, 40] = 255
        binary[100:190, 40] = 255
        intervals, lines = _long_lines(binary)
        self.assertTrue(any(a <= 40 < b for a, b in intervals))
        self.assertEqual(np.count_nonzero(lines[:, 40]), 175)

    def test_bordered_columns_with_narrow_gaps_and_kunten(self):
        image = np.full((180, 128, 3), (210, 230, 245), np.uint8)
        for x in (10, 46, 82, 118):
            image[8:85, x] = 0
            image[90:172, x] = 0
        for x in (10, 46, 82):
            for y in (20, 55, 90, 125):
                image[y:y + 20, x + 3:x + 23] = 0
                image[y + 4:y + 8, x + 30:x + 33] = 0
        before = image.copy()
        cleaned, stats = strip_side_kunten(image)
        self.assertTrue(stats['triggered'])
        self.assertEqual(stats['columns'], 3)
        self.assertEqual(stats['components_removed'], 12)
        for x in (10, 46, 82):
            for y in (20, 55, 90, 125):
                np.testing.assert_array_equal(cleaned[y:y + 20, x + 3:x + 23],
                                              before[y:y + 20, x + 3:x + 23])
                np.testing.assert_array_equal(cleaned[y + 4:y + 8, x + 30:x + 33],
                                              np.full((4, 3, 3), (210, 230, 245), np.uint8))
        np.testing.assert_array_equal(image, before)

    def test_grid_preserves_crossing_color_and_input(self):
        image = np.full((180, 140, 3), (210, 230, 245), np.uint8)
        image[10:170, 70] = (0, 0, 0)
        image[80:83, 55:86] = (0, 0, 0)
        before = image.copy()
        cleaned, stats = remove_vertical_grid_lines(image)
        self.assertGreaterEqual(stats['lines_removed'], 1)
        self.assertGreater(stats['pixels_filled'], 0)
        np.testing.assert_array_equal(cleaned[40, 70], [210, 230, 245])
        np.testing.assert_array_equal(cleaned[81, 55:86], before[81, 55:86])
        np.testing.assert_array_equal(image, before)
        self.assertEqual(cleaned.shape, image.shape)
        self.assertGreater(len(np.unique(cleaned)), 2)

    @staticmethod
    def columns(with_marks):
        image = np.full((180, 240, 3), (210, 230, 245), np.uint8)
        for x in (25, 95, 165):
            for y in (20, 55, 90, 125):
                image[y:y + 20, x:x + 20] = 0
                if with_marks:
                    image[y + 4:y + 8, x + 23:x + 26] = 0
        return image

    def test_kunten_trigger_and_preserve_main(self):
        image = self.columns(True)
        before = image.copy()
        cleaned, stats = strip_side_kunten(image)
        self.assertTrue(stats['triggered'])
        self.assertEqual(stats['components_removed'], 12)
        for x in (25, 95, 165):
            for y in (20, 55, 90, 125):
                np.testing.assert_array_equal(cleaned[y:y + 20, x:x + 20], before[y:y + 20, x:x + 20])
                np.testing.assert_array_equal(cleaned[y + 4, x + 24], [210, 230, 245])
        np.testing.assert_array_equal(image, before)

    def test_no_kunten_and_disabled(self):
        image = self.columns(False)
        cleaned, stats = strip_side_kunten(image)
        self.assertFalse(stats['triggered'])
        np.testing.assert_array_equal(cleaned, image)
        cleaned, stats = clean_body_crop(image)
        self.assertIs(cleaned, image)
        self.assertEqual(stats, {})

    def test_empty_and_grayscale(self):
        empty = np.empty((0, 0), np.uint8)
        self.assertIs(remove_vertical_grid_lines(empty)[0], empty)
        self.assertFalse(strip_side_kunten(empty)[1]['triggered'])
        gray = cv2.cvtColor(self.columns(True), cv2.COLOR_BGR2GRAY)
        cleaned, stats = strip_side_kunten(gray)
        self.assertTrue(stats['triggered'])
        self.assertEqual(cleaned.ndim, 2)

    def test_frame_only_cleans_body(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        frame = {'bbox': [10, 40, 190, 190], 'top_y': 40, 'bot_y': 190}
        config = SimpleNamespace(ocr=SimpleNamespace(remove_grid=True, strip_kunten=False))
        seen = []
        def infer(crop):
            seen.append(crop.copy())
            return [([[1, 1], [5, 1], [5, 5], [1, 5]], 'x', .9)]
        def clean(crop, **kwargs):
            self.assertTrue(kwargs['remove_grid'])
            return np.zeros_like(crop), {'remove_grid': {'lines_removed': 1}}
        with patch('ocr.frame.imread_unicode', return_value=image), \
             patch('ocr.frame.detect_body_frames', return_value=[frame]), \
             patch('ocr.frame.CONFIG', config), \
             patch('ocr.frame.clean_body_crop', side_effect=clean) as call:
            result = recognize_with_frame_isolation('x', 'p', infer)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(np.all(seen[0] == 0))
        self.assertTrue(np.all(seen[1] == 255))
        self.assertTrue(any('lines_removed' in note for note in result.notes))


if __name__ == '__main__':
    unittest.main()
