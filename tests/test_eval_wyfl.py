import unittest

from tools.eval_wyfl import clean_text, evaluate_records, is_ts_equiv, levenshtein_ops


class TextTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text(" 溫温\t〔夾注〕□，。！\n熱热（中）/\\-"), "溫温夾注熱热中")

    def test_edit_operations(self):
        cases = [("abc", "abc", (0, 0, 0, 0)), ("", "", (0, 0, 0, 0)),
                 ("abc", "axc", (1, 1, 0, 0)), ("abc", "ac", (1, 0, 1, 0)),
                 ("ac", "abc", (1, 0, 0, 1)), ("abc", "", (3, 0, 3, 0)),
                 ("", "abc", (3, 0, 0, 3)), ("abcd", "axce", (2, 2, 0, 0)),
                 ("abcd", "axcde", (2, 1, 0, 1))]
        for gt, pred, expected in cases:
            with self.subTest(gt=gt, pred=pred):
                result = levenshtein_ops(gt, pred)
                self.assertEqual(result, expected)
                self.assertEqual(result[0], sum(result[1:]))

    def test_equivalent_characters(self):
        for a, b in (("爲", "為"), ("熱", "热"), ("衛", "衞"), ("雞", "鷄")):
            with self.subTest(a=a, b=b):
                self.assertTrue(is_ts_equiv(a, b))
                self.assertEqual(levenshtein_ops(a, b, is_ts_equiv), (0, 0, 0, 0))
        self.assertFalse(is_ts_equiv("熱", "冷"))


class EvaluationTests(unittest.TestCase):
    def test_body_filter_and_aggregation(self):
        records = [
            {"page": "p4", "gt": "甲乙", "engine": "one", "direction": "rtl", "blocks": [
                {"reading_order": 3, "kind": "spine", "text": "脊"},
                {"reading_order": 2, "kind": "paragraph", "text": "丙"},
                {"reading_order": 0, "kind": "margin_note", "text": "注"},
                {"reading_order": 1, "kind": "paragraph", "text": "甲"}]},
            {"page": "p5", "gt": "天地玄黃", "engine": "one", "direction": "rtl", "blocks": [
                {"reading_order": 1, "kind": "paragraph", "text": "天地玄黃"}]},
        ]
        report = evaluate_records(records, body_only=True)
        self.assertEqual(report["pages"][0]["pred_chars"], 2)
        self.assertEqual(report["pages"][0]["cer"], 0.5)
        self.assertEqual(report["summary"]["cer"], 1 / 6)
        self.assertEqual(report["summary"]["median_cer"], 0.25)
        self.assertEqual(report["summary"]["cer_ts"], 1 / 6)
        self.assertEqual(report["summary"]["engine_distribution"], {"one": 2})
        self.assertEqual(evaluate_records(records)["pages"][0]["pred_chars"], 4)

    def test_empty_reference_is_undefined(self):
        result = evaluate_records([{"page": "p", "gt": "", "blocks": [
            {"reading_order": 1, "kind": "paragraph", "text": "甲"}]}])
        self.assertIsNone(result["pages"][0]["cer"])
        self.assertIsNone(result["summary"]["cer"])
        self.assertEqual(result["summary"]["I"], 1)


if __name__ == "__main__":
    unittest.main()
