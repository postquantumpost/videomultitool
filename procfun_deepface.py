"""Processing function: face detection with deepface."""

from procfun_util import LazyModel

# deepface detector backend; see deepface docs for options
# (opencv, ssd, mtcnn, retinaface, yunet, yolov8, centerface, ...).
DEEPFACE_DETECTOR = "yunet"
DEEPFACE_MIN_CONFIDENCE = 0.5


def _load():
    from deepface import DeepFace

    return DeepFace


_model = LazyModel("deepface", _load)


def procfun_deepface(frame, info):
    """Detect faces with deepface and return their bounding boxes.

    frame: BGR image as a numpy array (height, width, 3); must not be modified.
    info: FrameInfo for this frame.
    Returns a list of (x, y, width, height) boxes to blur.
    """
    deepface = _model.get()
    if deepface is None:
        return []
    faces = deepface.extract_faces(
        img_path=frame,
        detector_backend=DEEPFACE_DETECTOR,
        enforce_detection=False,
        align=False,
    )
    boxes = []
    for face in faces:
        # With enforce_detection=False and no faces found, deepface returns the
        # whole frame with confidence 0, so the confidence filter removes it.
        if (face.get("confidence") or 0) < DEEPFACE_MIN_CONFIDENCE:
            continue
        area = face["facial_area"]
        boxes.append((area["x"], area["y"], area["w"], area["h"]))
    return boxes
