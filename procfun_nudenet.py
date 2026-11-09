"""Processing function: nudity detection with NudeNet."""

from procfun_util import LazyModel

# NudeNet scores are often low (0.25-0.4) even for clear detections, especially
# when the subject is small in the frame; err on the side of blurring.
NUDENET_MIN_SCORE = 0.2
NUDENET_BLUR_CLASSES = {
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


def procfun_nudenet(frame, info):
    """Detect nudity with NudeNet and return bounding boxes of exposed regions.

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
        for d in detector.detect(frame)
        if d["class"] in NUDENET_BLUR_CLASSES and d["score"] >= NUDENET_MIN_SCORE
    ]
