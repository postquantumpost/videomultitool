"""Processing function: face and nudity detection with NudeNet."""

import cv2
import numpy as np

from procfun_util import LazyModel

# NudeNet scores are often low (0.25-0.4) even for clear detections, especially
# when the subject is small in the frame; err on the side of blurring.
NUDENET_MIN_SCORE = 0.2
NUDENET_NMS_SCORE = 0.25
NUDENET_NMS_OVERLAP = 0.9
NUDENET_BLUR_CLASSES = {
    "FACE_FEMALE",
    "FACE_MALE",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def _load():
    from nudenet import NudeDetector

    return NudeDetector()


_model = LazyModel("NudeNet", _load)


def _detect(frame, detector):
    # NudeNet's detect() hard-codes NMS thresholds; reuse its preprocessing
    # and model but apply our own suppression before any boxes are discarded.
    from nudenet.nudenet import __labels as labels, _read_image

    blob, _xr, _yr, x_pad, y_pad, width, height = _read_image(
        frame, detector.input_width,
    )
    output = detector.onnx_session.run(None, {detector.input_name: blob})
    predictions = output[0][0].T
    boxes, scores, classes = [], [], []
    for prediction in predictions:
        class_id = int(np.argmax(prediction[4:]))
        score = float(prediction[4 + class_id])
        if score < NUDENET_MIN_SCORE:
            continue
        cx, cy, w, h = prediction[:4]
        x_scale = (width + x_pad) / detector.input_width
        y_scale = (height + y_pad) / detector.input_height
        x = max(0, min(float((cx - w / 2) * x_scale), width))
        y = max(0, min(float((cy - h / 2) * y_scale), height))
        w = min(float(w * x_scale), width - x)
        h = min(float(h * y_scale), height - y)
        boxes.append([x, y, w, h])
        scores.append(score)
        classes.append(labels[class_id])
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, NUDENET_NMS_SCORE, NUDENET_NMS_OVERLAP,
    )
    return [
        {"class": classes[i], "score": scores[i],
         "box": [int(v) for v in boxes[i]]}
        for i in indices
    ]


def procfun_nudenet(frame, info):
    """Detect faces and nudity with NudeNet and return their bounding boxes.

    frame: BGR image as a numpy array (height, width, 3); must not be modified.
    info: FrameInfo for this frame.
    Returns a list of (x, y, width, height) boxes to blur.
    """
    if not info.nudenet:
        return []
    detector = _model.get()
    if detector is None:
        return []
    return [
        tuple(d["box"])
        for d in _detect(frame, detector)
        if d["class"] in NUDENET_BLUR_CLASSES and d["score"] >= NUDENET_MIN_SCORE
    ]
