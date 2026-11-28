import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.schemas import Block, source_hash_of
from ocr import layout
from ocr.frame import detect_body_frames, recognize_with_frame_isolation


def make_block(name, bbox, kind="paragraph"):
    return Block(page_id="P", bbox=bbox, line_id=name, quote=name,
                 source_hash=source_hash_of("test", name), text=name, text_norm=name,
                 kind=kind, conf=.9, col_id=0, block_ref="P:" + name)


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class TestFrameIsolation(unittest.TestCase):
    def setUp(self):
        self.img = np.full((600, 1000, 3), 255, np.uint8)
        for x1, x2 in ((2, 480), (520, 998)):
            cv2.rectangle(self.img, (x1, 100), (x2, 540), (0, 0, 0), 1)
        cv2.putText(self.img, "NOTE", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        cv2.rectangle(self.img, (8, 150), (16, 200), (0, 0, 0), -1)

    def test_detect_two_frames(self):
        frames = detect_body_frames(self.img)
        self.assertEqual([f["half"] for f in frames], ["R", "L"])
        for frame, expected in zip(frames, ([520, 100, 998, 540], [2, 100, 480, 540])):
            for actual, target in zip(frame["bbox"], expected):
                self.assertLessEqual(abs(actual - target), 5)

    def test_no_frame(self):
        img = np.full((600, 500, 3), 255, np.uint8)
        cv2.putText(img, "ordinary text", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        self.assertEqual(detect_body_frames(img), [])

    def test_spread_with_only_one_frame_is_rejected(self):
        img = np.full((600, 1000, 3), 255, np.uint8)
        cv2.rectangle(img, (520, 100), (980, 540), (0, 0, 0), 1)
        self.assertEqual(detect_body_frames(img), [])

    def test_nearly_square_spread_detects_both_frames(self):
        img = np.full((1000, 1050, 3), 255, np.uint8)
        for x1, x2 in ((20, 500), (550, 1030)):
            cv2.rectangle(img, (x1, 150), (x2, 900), (0, 0, 0), 1)
        self.assertEqual([f["half"] for f in detect_body_frames(img)], ["R", "L"])

    def test_spread_with_misaligned_frames_is_rejected(self):
        img = np.full((1000, 1400, 3), 255, np.uint8)
        cv2.rectangle(img, (30, 80), (670, 900), (0, 0, 0), 1)
        cv2.rectangle(img, (730, 250), (1370, 900), (0, 0, 0), 1)
        self.assertEqual(detect_body_frames(img), [])

    def test_portrait_table_grid_is_rejected(self):
        img = np.full((1000, 700, 3), 255, np.uint8)
        cv2.line(img, (100, 80), (600, 80), (0, 0, 0), 1)
        cv2.rectangle(img, (50, 180), (650, 900), (0, 0, 0), 1)
        cv2.line(img, (50, 500), (650, 500), (0, 0, 0), 1)
        self.assertEqual(detect_body_frames(img), [])

    def test_portrait_outer_border_with_header_underline_is_rejected(self):
        img = np.full((1200, 900, 3), 255, np.uint8)
        cv2.rectangle(img, (30, 30), (870, 1170), (0, 0, 0), 1)
        cv2.line(img, (30, 118), (870, 118), (0, 0, 0), 1)
        self.assertEqual(detect_body_frames(img), [])

    def test_portrait_frame_requires_both_vertical_borders(self):
        img = np.full((1000, 700, 3), 255, np.uint8)
        cv2.rectangle(img, (50, 150), (650, 900), (0, 0, 0), 1)
        self.assertEqual([f["half"] for f in detect_body_frames(img)], ["ALL"])
        cv2.line(img, (650, 151), (650, 899), (255, 255, 255), 3)
        self.assertEqual(detect_body_frames(img), [])

    def test_recognize_and_map(self):
        frames = detect_body_frames(self.img)
        def infer(crop):
            if crop.shape[0] < 100:
                return [(box(50, 10, 80, 30), "note", .8)]
            if infer.calls == 0:
                infer.calls += 1
                return [(box(20, 10, 40, 90), "body", .9)]
            return [(box(4, 20, 12, 80), "spine", .7)]
        infer.calls = 0
        with patch("ocr.frame.imread_unicode", return_value=self.img):
            result = recognize_with_frame_isolation("image", "P", infer)
        self.assertEqual([b.kind for b in result.blocks], ["paragraph", "spine", "margin_note"])
        right, left = [f["bbox"] for f in frames]
        self.assertEqual(result.blocks[0].bbox, [right[0]+22, right[1]+12, right[0]+42, right[1]+92])
        self.assertEqual(result.blocks[1].bbox, [left[0]+6, left[1]+22, left[0]+14, left[1]+82])
        self.assertEqual(result.blocks[2].bbox, [50, 10, 80, 30])

    def test_layout_keeps_auxiliary_separate(self):
        def body():
            return [make_block("left", [100, 100, 120, 300]), make_block("right", [300, 100, 320, 300])]
        baseline, _ = layout.analyze_layout(body(), "P", direction="vertical-rtl")
        items, lay = layout.analyze_layout(body() + [make_block("note", [100, 10, 320, 40], "margin_note"),
                                                      make_block("spine", [1, 100, 12, 300], "spine")],
                                           "P", direction="vertical-rtl")
        self.assertEqual([(b.text, b.col_id, b.reading_order) for b in items[:2]],
                         [(b.text, b.col_id, b.reading_order) for b in baseline])
        self.assertEqual([(b.kind, b.col_id, b.reading_order) for b in items[2:]],
                         [("margin_note", -2, 3), ("spine", -3, 4)])
        self.assertEqual(lay.block_order, [b.block_ref for b in items])

    def test_pipeline_uses_isolation_and_falls_back(self):
        from types import SimpleNamespace
        from ocr import pipeline
        from ocr.recognize import RecognizeResult
        adapter = SimpleNamespace(name="rapidocr", infer=lambda image: [])
        isolated = RecognizeResult(blocks=[make_block("body", [10, 20, 30, 40])])
        fallback = RecognizeResult()
        config = SimpleNamespace(ocr=SimpleNamespace(frame_isolate=True, vertical_split=False, pick="auto"))
        with patch.object(pipeline, "CONFIG", config), \
             patch("ocr.recognize.get_adapter", return_value=(adapter, [])), \
             patch("ocr.frame.recognize_with_frame_isolation", return_value=isolated), \
             patch.object(pipeline, "recognize", return_value=fallback) as whole:
            self.assertIs(pipeline._recognize_page("image", "P", None), isolated)
            whole.assert_not_called()
        with patch.object(pipeline, "CONFIG", config), \
             patch("ocr.recognize.get_adapter", return_value=(adapter, [])), \
             patch("ocr.frame.recognize_with_frame_isolation", return_value=None), \
             patch.object(pipeline, "recognize", return_value=fallback) as whole:
            self.assertIs(pipeline._recognize_page("image", "P", None), fallback)
            whole.assert_called_once()

    def test_no_body_results_fall_back(self):
        with patch("ocr.frame.imread_unicode", return_value=self.img):
            self.assertIsNone(recognize_with_frame_isolation("image", "P", lambda crop: []))


if __name__ == "__main__":
    unittest.main()
