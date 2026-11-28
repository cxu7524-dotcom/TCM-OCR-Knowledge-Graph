import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from app import config
from ocr import recognize


class OcrConfigDefaultsTests(unittest.TestCase):
    def test_model_candidates_and_fallback(self):
        self.assertEqual(config.REC_MODEL_CANDIDATES, (
            "模型权重/PP-OCRv5_server_rec_official.onnx",
            "模型权重/ch_PP-OCRv5_rec_mobile.onnx",
        ))
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "PROJECT_ROOT", Path(directory)), \
                patch.dict(os.environ, {"TCM_OCR_REC_MODEL": ""}):
            server, mobile = (Path(directory) / path for path in config.REC_MODEL_CANDIDATES)
            server.parent.mkdir()
            self.assertEqual(config._resolve_rec_model(), ("", "bundled"))
            mobile.touch()
            self.assertEqual(config._resolve_rec_model(), (config.REC_MODEL_CANDIDATES[1], "auto"))
            server.touch()
            self.assertEqual(config._resolve_rec_model(), (config.REC_MODEL_CANDIDATES[0], "auto"))

    def test_flag_defaults_and_rerank_opt_out(self):
        env = os.environ.copy()
        for key in ("CTC_RERANK", "FRAME_ISOLATE", "REMOVE_GRID", "STRIP_KUNTEN", "MERGE_JIAZHU", "COL_SEG_JIAZHU"):
            env.pop("TCM_OCR_" + key, None)
        code = (
            "from app.config import CONFIG; "
            "o = CONFIG.ocr; "
            "print(int(o.ctc_rerank), int(o.frame_isolate), int(o.remove_grid), "
            "int(o.strip_kunten), int(o.merge_jiazhu), int(o.col_seg_jiazhu), "
            "int(CONFIG.summary()['ocr_col_seg_jiazhu']))"
        )
        self.assertEqual(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip(), "1 0 0 0 0 1 1")
        env["TCM_OCR_CTC_RERANK"] = "0"
        env["TCM_OCR_COL_SEG_JIAZHU"] = "0"
        self.assertEqual(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip(), "0 0 0 0 0 0 0")


class RecognizeKeysTests(unittest.TestCase):
    def setUp(self):
        recognize._ENGINES.clear()
        self.addCleanup(recognize._ENGINES.clear)
        self.env = patch.dict(os.environ, {"TCM_OCR_REC_KEYS": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "model.onnx"
        self.model.touch()
        self.keys = self.model.with_suffix(".keys.txt")
        rapid_patcher = patch("rapidocr_onnxruntime.RapidOCR")
        self.rapid = rapid_patcher.start()
        self.addCleanup(rapid_patcher.stop)

    def test_companion_keys(self):
        self.keys.touch()
        self.assertEqual(recognize._resolve_rec_keys(str(self.model)), str(self.keys.resolve()))
        recognize._engine(str(self.model))
        self.rapid.assert_called_once_with(rec_model_path=str(self.model), rec_keys_path=str(self.keys.resolve()))

    def test_without_keys(self):
        self.assertEqual(recognize._resolve_rec_keys(str(self.model)), "")
        recognize._engine(str(self.model))
        self.rapid.assert_called_once_with(rec_model_path=str(self.model))

    def test_environment_takes_precedence_and_cache_tracks_keys(self):
        self.keys.touch()
        other = Path(self.temp.name) / "other.txt"
        other.touch()
        recognize._engine(str(self.model))
        with patch.dict(os.environ, {"TCM_OCR_REC_KEYS": str(other)}):
            self.assertEqual(recognize._resolve_rec_keys(str(self.model)), str(other.resolve()))
            recognize._engine(str(self.model))
        self.assertEqual(self.rapid.call_count, 2)
        self.rapid.assert_called_with(rec_model_path=str(self.model), rec_keys_path=str(other.resolve()))

    def test_relative_environment_and_missing_override(self):
        root = Path(recognize.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            keys = Path(directory) / "keys.txt"
            keys.touch()
            with patch.dict(os.environ, {"TCM_OCR_REC_KEYS": str(keys.relative_to(root))}):
                self.assertEqual(recognize._resolve_rec_keys(str(self.model)), str(keys.resolve()))
        self.keys.touch()
        with patch.dict(os.environ, {"TCM_OCR_REC_KEYS": "missing-keys.txt"}):
            self.assertEqual(recognize._resolve_rec_keys(str(self.model)), str(self.keys.resolve()))

    def test_empty_model_uses_default_arguments(self):
        with patch.dict(os.environ, {"TCM_OCR_REC_KEYS": str(self.model)}):
            self.assertEqual(recognize._resolve_rec_keys(""), "")
            recognize._engine("")
        self.rapid.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
