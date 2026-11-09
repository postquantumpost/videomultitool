"""Processing function: face detection with RetinaFace."""

import cv2

from procfun_util import LazyModel, corners_to_box

RETINAFACE_THRESHOLD = 0.9
# Frames are shrunk so their longest side is at most this many pixels before
# detection (RetinaFace is very slow on CPU at full HD). None = no limit.
RETINAFACE_MAX_SIZE = 960


def _load():
    from retinaface import RetinaFace

    return RetinaFace, RetinaFace.build_model()


_model = LazyModel("RetinaFace", _load)


def procfun_retinaface(frame, info):
    """Detect faces with RetinaFace and return their bounding boxes.

    frame: BGR image as a numpy array (height, width, 3); must not be modified.
    info: FrameInfo for this frame.
    Returns a list of (x, y, width, height) boxes to blur.
    """
    if not info.retinaface:
        return []
    loaded = _model.get()
    if loaded is None:
        return []
    retinaface, model = loaded
    scale = 1.0
    image = frame
    longest = max(frame.shape[:2])
    if RETINAFACE_MAX_SIZE and longest > RETINAFACE_MAX_SIZE:
        scale = RETINAFACE_MAX_SIZE / longest
        image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    faces = retinaface.detect_faces(
        image, threshold=RETINAFACE_THRESHOLD, model=model, allow_upscaling=False
    )
    # Older retina-face versions return a tuple instead of {} when nothing is found.
    if not isinstance(faces, dict):
        return []
    return [
        corners_to_box(*(v / scale for v in face["facial_area"]))
        for face in faces.values()
    ]
