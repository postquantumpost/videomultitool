from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from nudenet.nudenet import __labels as labels

import procfun_nudenet as nudenet


class NudeNetTests(unittest.TestCase):
    def detect(self, rows, width=320, height=320):
        predictions = np.zeros((1, 4 + len(labels), len(rows)), np.float32)
        for i, (box, label, score) in enumerate(rows):
            predictions[0, :4, i] = box
            predictions[0, 4 + labels.index(label), i] = score
        detector = SimpleNamespace(
            input_width=320, input_height=320, input_name="images",
            onnx_session=Mock(),
        )
        detector.onnx_session.run.return_value = [predictions]
        frame = np.zeros((height, width, 3), np.uint8)
        original = frame.copy()
        with patch.object(nudenet._model, "get", return_value=detector):
            boxes = nudenet.procfun_nudenet(
                frame, SimpleNamespace(nudenet=True),
            )
        np.testing.assert_array_equal(frame, original)
        return boxes

    def test_both_face_classes_are_blurred(self):
        self.assertEqual(self.detect([
            ([50, 50, 40, 40], "FACE_FEMALE", 0.9),
            ([200, 200, 40, 40], "FACE_MALE", 0.8),
        ]), [(30, 30, 40, 40), (180, 180, 40, 40)])

    def test_overlap_above_old_threshold_is_retained(self):
        boxes = [[50, 50, 100, 100], [70, 50, 100, 100]]
        self.assertEqual(len(cv2.dnn.NMSBoxes(
            boxes, [0.9, 0.8], 0.25, 0.45,
        )), 1)
        self.assertEqual(self.detect([
            ([100, 100, 100, 100], "FACE_FEMALE", 0.9),
            ([120, 100, 100, 100], "FACE_MALE", 0.8),
        ]), [(50, 50, 100, 100), (70, 50, 100, 100)])

    def test_near_duplicates_are_still_suppressed(self):
        self.assertEqual(self.detect([
            ([100, 100, 100, 100], "FACE_MALE", 0.9),
            ([101, 100, 100, 100], "FACE_MALE", 0.8),
        ]), [(50, 50, 100, 100)])

    def test_confidence_and_class_filtering_are_preserved(self):
        self.assertEqual(self.detect([
            ([50, 50, 40, 40], "FACE_MALE", 0.24),
            ([100, 100, 40, 40], "FEET_EXPOSED", 0.9),
            ([200, 200, 40, 40], "BUTTOCKS_EXPOSED", 0.8),
        ]), [(180, 180, 40, 40)])

    def test_scaling_padding_and_clipping(self):
        self.assertEqual(self.detect([
            ([310, 150, 40, 40], "FACE_MALE", 0.9),
        ], width=640, height=320), [(580, 260, 60, 60)])

    def test_no_candidates(self):
        self.assertEqual(self.detect([
            ([100, 100, 40, 40], "FACE_MALE", 0.1),
        ]), [])


if __name__ == "__main__":
    unittest.main()
