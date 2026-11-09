import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

import filter_in_to_out as filtering
import videomultitool as cli


ROOT = Path(__file__).resolve().parent


class AlacarteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def run_cli(self, *args):
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            return cli.main(args)

    def test_defaults_unchanged(self):
        args = cli.parse_args(["--blur"])
        self.assertIsNone(args.alacarte)
        self.assertFalse(args.debug2)
        for name in cli.PROCESSING_NAMES[:-1]:
            self.assertTrue(getattr(args, name))
        self.assertTrue(cli.parse_args(["--debug2"]).debug2)

    def test_selection_booleans(self):
        for name in cli.PROCESSING_NAMES:
            with self.subTest(name=name):
                args = cli.parse_args(["--alacarte", name, "--debug2"])
                for candidate in cli.PROCESSING_NAMES:
                    self.assertEqual(getattr(args, candidate), candidate == name)

    def test_whitespace_duplicates_and_empty_list(self):
        args = cli.parse_args(["--alacarte", " deepface, retinaface,deepface "])
        self.assertTrue(args.deepface)
        self.assertTrue(args.retinaface)
        self.assertFalse(args.nudenet)
        for value in ("", "  "):
            args = cli.parse_args(["--alacarte", value, "--debug2"])
            self.assertEqual(args.alacarte, ())
            for name in cli.PROCESSING_NAMES:
                self.assertFalse(getattr(args, name))

    def test_invalid_arguments(self):
        for args in (
            ["--alacarte", "unknown"],
            ["--alacarte", "deepface,"],
            ["--alacarte", ",deepface"],
            ["--alacarte", "deepface,,retinaface"],
            ["--alacarte"],
            ["--alacarte", "", "--blankless"],
            ["--alacarte", "prp"],
            ["--alacarte", "simplelpr"],
            ["--slowok"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.parse_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_standalone_selection_and_reset(self):
        with patch.object(cli, "filter_in_to_out") as filter_video:
            self.run_cli("--alacarte", "deepface,retinaface",
                         "--input", "in.mp4", "--output", "out.mp4")
            options = filter_video.call_args.kwargs
            self.assertTrue(options["retinaface"])
            self.assertTrue(options["deepface"])
            self.assertFalse(options["nudenet"])
            self.assertFalse(options["hyperlpr"])
            self.assertFalse(options["debug2"])
            self.run_cli("--blur", "--input", "in.mp4", "--output", "out.mp4")
            options = filter_video.call_args.kwargs
            self.assertTrue(options["retinaface"])
            self.assertTrue(options["nudenet"])
            self.assertTrue(options["hyperlpr"])

    def test_empty_selection_copies_exact_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "in.mp4"
            target = Path(directory) / "out.mp4"
            payload = bytes(range(256)) * 8
            source.write_bytes(payload)
            with patch.object(cli, "filter_in_to_out") as filter_video:
                self.run_cli("--alacarte", "", "--blur", "--debug", "--debug2",
                             "--fill", "--input", str(source),
                             "--output", str(target))
                filter_video.assert_not_called()
            self.assertEqual(target.read_bytes(), payload)

    def test_disabled_functions_do_not_load_models(self):
        info = filtering.FrameInfo(0, 0.0, 64, 64,
                                   retinaface=False, deepface=False,
                                   nudenet=False, hyperlpr=False,
                                   fastanpr=False)
        for name in cli.PROCESSING_NAMES[:-1]:
            module = importlib.import_module(f"procfun_{name}")
            with self.subTest(name=name), patch.object(module._model, "get") as load:
                self.assertEqual(getattr(module, f"procfun_{name}")(None, info), [])
                load.assert_not_called()
        with patch("procfun_debug2.random.randint") as random_number:
            self.assertEqual(filtering.procfun_debug2(None, info), [])
            random_number.assert_not_called()

    def test_enabled_functions_reach_models(self):
        info = filtering.FrameInfo(0, 0.0, 64, 64)
        for name in cli.PROCESSING_NAMES[:-1]:
            module = importlib.import_module(f"procfun_{name}")
            with self.subTest(name=name), patch.object(
                module._model, "get", return_value=None,
            ) as load:
                self.assertEqual(getattr(module, f"procfun_{name}")(None, info), [])
                load.assert_called_once_with()

    def test_debug2_video_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "in.mp4"
            target = Path(directory) / "out.mp4"
            rng = np.random.default_rng(42)
            with av.open(str(source), "w") as container:
                stream = container.add_stream("libx264", rate=24)
                stream.width = stream.height = 64
                stream.pix_fmt = "yuv420p"
                for _ in range(2):
                    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    container.mux(stream.encode(frame))
                container.mux(stream.encode())
            with contextlib.ExitStack() as stack:
                for name in cli.PROCESSING_NAMES[:-1]:
                    module = importlib.import_module(f"procfun_{name}")
                    stack.enter_context(patch.object(
                        module._model, "get",
                        side_effect=AssertionError("disabled detector loaded"),
                    ))
                render_frame = stack.enter_context(patch.object(
                    filtering, "render_frame", wraps=filtering.render_frame,
                ))
                self.run_cli("--alacarte", "debug2", "--input", str(source),
                             "--output", str(target))
                self.assertEqual(render_frame.call_count, 2)
                for call in render_frame.call_args_list:
                    self.assertEqual(len(call.args[1].boxes), 20)
            with av.open(str(target)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 2)
                self.assertEqual((frames[0].width, frames[0].height), (64, 64))


if __name__ == "__main__":
    unittest.main()
