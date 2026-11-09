"""Processing function: license plate detection with FastANPR (without OCR)."""

from pathlib import Path

from procfun_util import LazyModel, corners_to_box

FASTANPR_MIN_CONFIDENCE = 0.0
FASTANPR_DEVICE = "cpu"


def _load():
    import fastanpr
    from fastanpr.detection import Detector

    return Detector(
        detection_model=Path(fastanpr.__file__).parent / "best.pt",
        device=FASTANPR_DEVICE,
    )


_model = LazyModel("FastANPR", _load)


def procfun_fastanpr(frame, info):
    if not info.fastanpr:
        return []
    detector = _model.get()
    if detector is None:
        return []
    return [
        corners_to_box(*detection.box)
        for detection in detector.run([frame])[0]
        if detection.conf >= FASTANPR_MIN_CONFIDENCE
    ]
