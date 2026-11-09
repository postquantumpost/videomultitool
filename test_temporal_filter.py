import contextlib
from fractions import Fraction
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

import filter_in_to_out as filtering


ROOT = Path(__file__).resolve().parent


class TemporalFilterTests(unittest.TestCase):
    def run_pipeline(self, times_ms, window_seconds=0.15, fill=False,
                     blur_edge=False, empty=False):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "in.mkv"
            target = Path(directory) / "out.mkv"
            rng = np.random.default_rng(42)
            with av.open(str(source), "w") as container:
                stream = container.add_stream("ffv1", rate=20)
                stream.width = stream.height = 64
                stream.pix_fmt = "yuv444p"
                audio = container.add_stream("pcm_s16le", rate=8000)
                audio.layout = "mono"
                for timestamp_ms in times_ms:
                    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    frame.pts = timestamp_ms
                    frame.time_base = Fraction(1, 1000)
                    container.mux(stream.encode(frame))
                container.mux(stream.encode())
                samples = np.arange(8000, dtype=np.int16).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                frame.sample_rate = 8000
                frame.pts = 0
                frame.time_base = Fraction(1, 8000)
                container.mux(audio.encode(frame))
                container.mux(audio.encode())

            with av.open(str(source)) as container:
                originals = list(container.decode(video=0))
            timestamps = [frame.pts * frame.time_base for frame in originals]
            self.assertEqual(timestamps, [Fraction(t, 1000) for t in times_ms])
            boxes_by_frame = [
                [] if empty else [(8 + i * 2, 8, 24, 24)]
                for i in range(len(times_ms))
            ]
            analyzed = []
            rendered = []
            real_render = filtering.render_frame

            def detect(image, info):
                analyzed.append(info.number)
                np.testing.assert_array_equal(
                    image, originals[info.number].to_ndarray(format="bgr24"),
                )
                return boxes_by_frame[info.number]

            def render(image, info):
                timestamp = timestamps[info.number]
                window = Fraction(str(window_seconds))
                expected_boxes = [
                    box
                    for detected_at, boxes in zip(timestamps, boxes_by_frame)
                    if abs(detected_at - timestamp) <= window
                    for box in boxes
                ]
                self.assertEqual(info.boxes, expected_boxes)
                expected = image.copy()
                expected_info = filtering.FrameInfo(
                    info.number, info.timestamp, 64, 64, fill=fill,
                    blur_edge=blur_edge, boxes=expected_boxes,
                )
                filtering.render_blur(expected, image, expected_info)
                actual = real_render(image, info)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    image, originals[info.number].to_ndarray(format="bgr24"),
                )
                rendered.append((info.number, len(analyzed)))
                return actual

            with (patch.object(filtering, "half_window_seconds", window_seconds),
                  patch.object(filtering, "PROCESSING_FUNCTIONS", [detect]),
                  patch.object(filtering, "RENDERING_FUNCTIONS", [filtering.render_blur]),
                  patch.object(filtering, "render_frame", side_effect=render),
                  contextlib.redirect_stdout(io.StringIO())):
                filtering.filter_in_to_out(
                    str(source), str(target), fill=fill, blur_edge=blur_edge,
                )

            self.assertEqual([number for number, _ in rendered], list(range(len(times_ms))))
            if window_seconds and len(times_ms) > 1:
                self.assertGreater(rendered[0][1], 1)
            with av.open(str(target)) as container:
                output = list(container.decode(video=0))
            self.assertEqual([frame.pts * frame.time_base for frame in output], timestamps)
            self.assertEqual(len(output), len(originals))

            def audio_packets(path):
                with av.open(str(path)) as container:
                    return [
                        (packet.pts * packet.time_base,
                         packet.duration * packet.time_base, bytes(packet))
                        for packet in container.demux(audio=0)
                        if packet.dts is not None
                    ]

            self.assertEqual(audio_packets(source), audio_packets(target))

    def test_default_window(self):
        self.assertEqual(filtering.half_window_seconds, 0.15)

    def test_variable_timestamps_and_overlaps(self):
        for fill in (False, True):
            for blur_edge in (False, True):
                with self.subTest(fill=fill, blur_edge=blur_edge):
                    self.run_pipeline(
                        [1000, 1050, 1150, 1200, 1300, 1450, 1650],
                        fill=fill, blur_edge=blur_edge,
                    )

    def test_zero_window(self):
        self.run_pipeline([0, 50, 150, 200], window_seconds=0)

    def test_short_video_and_end_flush(self):
        self.run_pipeline([0, 50])
        self.run_pipeline([0])

    def test_empty_detections(self):
        self.run_pipeline([0, 50, 200, 400], empty=True)

    def test_invalid_window(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), patch.object(
                filtering, "half_window_seconds", value,
            ):
                with self.assertRaisesRegex(ValueError, "half_window_seconds"):
                    filtering.filter_in_to_out("unused", "unused")


if __name__ == "__main__":
    unittest.main()
