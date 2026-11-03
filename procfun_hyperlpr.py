"""Processing function: license plate detection with HyperLPR3."""

from procfun_util import LazyModel, corners_to_box

HYPERLPR_MIN_CONFIDENCE = 0.0


def _load():
    import hyperlpr3

    return hyperlpr3.LicensePlateCatcher(detect_level=hyperlpr3.DETECT_LEVEL_HIGH)


_model = LazyModel("HyperLPR", _load)


def procfun_hyperlpr(frame, info):
    """Detect license plates with HyperLPR and return their bounding boxes.

    frame: BGR image as a numpy array (height, width, 3); must not be modified.
    info: FrameInfo for this frame.
    Returns a list of (x, y, width, height) boxes to blur.
    """
    catcher = _model.get()
    if catcher is None:
        return []
    # Each result is [plate_code, confidence, plate_type, [x0, y0, x1, y1]].
    return [
        corners_to_box(*bbox)
        for _code, confidence, _type, bbox in catcher(frame)
        if confidence >= HYPERLPR_MIN_CONFIDENCE
    ]
