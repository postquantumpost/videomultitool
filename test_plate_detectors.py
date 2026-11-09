import contextlib
import importlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import filter_in_to_out as filtering
import procfun_fastanpr as fastanpr
import procfun_retinaface as retinaface
from procfun_util import LazyModel
import videomultitool as cli


class PlateDetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 96, 3), dtype=np.uint8)
        self.info = filtering.FrameInfo(0, 0.0, 96, 64)

    def test_fastanpr_boxes_confidence_and_empty_results(self):
        detector = Mock()
        detector.run.return_value = [[
            SimpleNamespace(box=[10.2, 20.4, 40.1, 35.3], conf=0.9),
            SimpleNamespace(box=[1, 2, 3, 4], conf=0.1),
        ]]
        with (patch.object(fastanpr._model, "get", return_value=detector),
              patch.object(fastanpr, "FASTANPR_MIN_CONFIDENCE", 0.5)):
            self.assertEqual(fastanpr.procfun_fastanpr(self.frame, self.info),
                             [(10, 20, 30, 15)])
            self.assertIs(detector.run.call_args.args[0][0], self.frame)
            detector.run.return_value = [[]]
            self.assertEqual(fastanpr.procfun_fastanpr(self.frame, self.info), [])

    def test_fastanpr_loader_uses_bundled_detection_model_only(self):
        package = SimpleNamespace(__file__="/models/fastanpr/__init__.py")
        detector = Mock()
        with patch.dict("sys.modules", {
            "fastanpr": package,
            "fastanpr.detection": SimpleNamespace(Detector=detector),
        }):
            self.assertIs(fastanpr._load(), detector.return_value)
        detector.assert_called_once_with(
            detection_model=Path("/models/fastanpr/best.pt"), device="cpu",
        )

    def test_loading_failure_warns_once_and_skips(self):
        loader = Mock(side_effect=ImportError("test missing package"))
        model = LazyModel("test detector", loader)
        with (patch.object(fastanpr, "_model", model),
              contextlib.redirect_stderr(io.StringIO()) as errors):
            self.assertEqual(fastanpr.procfun_fastanpr(self.frame, self.info), [])
            self.assertEqual(fastanpr.procfun_fastanpr(self.frame, self.info), [])
            loader.assert_called_once_with()
            self.assertEqual(errors.getvalue().count("Warning:"), 1)

    def test_cli_wires_each_selection_and_resets_defaults(self):
        for name in ("fastanpr", "retinaface"):
            with (self.subTest(name=name),
                  patch.object(cli, "filter_in_to_out") as pipeline,
                  contextlib.redirect_stdout(io.StringIO())):
                cli.main(["--alacarte", name, "--input", "in", "--output", "out"])
                for candidate in cli.PROCESSING_NAMES[:-1]:
                    self.assertEqual(pipeline.call_args.kwargs[candidate], candidate == name)
                cli.main(["--blur", "--input", "in", "--output", "out"])
                for candidate in cli.PROCESSING_NAMES[:-1]:
                    self.assertTrue(pipeline.call_args.kwargs[candidate])

    def test_pipeline_combines_boxes_in_fixed_order(self):
        detector = Mock()
        detector.detect_faces.return_value = {"face": {"facial_area": [8, 9, 18, 20]}}
        with contextlib.ExitStack() as stack:
            for name in cli.PROCESSING_NAMES[:-1]:
                module = importlib.import_module(f"procfun_{name}")
                stack.enter_context(patch.object(module._model, "get", return_value=None))
            stack.enter_context(patch.object(
                fastanpr._model, "get", return_value=SimpleNamespace(
                    run=lambda images: [[SimpleNamespace(box=[1, 2, 5, 6], conf=1.0)]],
                ),
            ))
            stack.enter_context(patch.object(
                retinaface._model, "get", return_value=(detector, object()),
            ))
            filtering.analyze_frame(self.frame, self.info)
        self.assertEqual(self.info.boxes, [(8, 9, 10, 11), (1, 2, 4, 4)])
        self.assertIs(detector.detect_faces.call_args.args[0], self.frame)
        self.assertFalse(self.frame.any())


if __name__ == "__main__":
    unittest.main()
