"""Processing function: debug2 (random bounding boxes for testing)."""

import random

NUM_BOXES = 10


def procfun_debug2(frame, info):
    """If debug2 is enabled, return ten random boxes that fit inside the frame.

    frame: BGR image as a numpy array (height, width, 3); must not be modified.
    info: FrameInfo for this frame.
    Returns a list of (x, y, width, height) boxes to blur.
    """
    if not info.debug2:
        return []
    boxes = []
    for _ in range(NUM_BOXES):
        w = random.randint(max(1, info.width // 20), max(1, info.width // 4))
        h = random.randint(max(1, info.height // 20), max(1, info.height // 4))
        x = random.randint(0, info.width - w)
        y = random.randint(0, info.height - h)
        boxes.append((x, y, w, h))
    return boxes
