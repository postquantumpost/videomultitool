"""Frame-by-frame video filtering: decode, analyze, render, encode.

Each decoded video frame is passed through PROCESSING_FUNCTIONS, which return
bounding boxes (x, y, width, height) of regions to blur. RENDERING_FUNCTIONS
then modify a copy of the frame (blur, debug overlays) before it is encoded.
Audio streams are copied to the output without re-encoding.
"""

from dataclasses import dataclass, field

import av
import cv2
import numpy as np

from procfun_debug2 import procfun_debug2
from procfun_deepface import procfun_deepface
from procfun_hyperlpr import procfun_hyperlpr
from procfun_nudenet import procfun_nudenet
from procfun_retinaface import procfun_retinaface
from video_util import (
    Progress,
    add_audio_copies,
    add_video_encoder,
    frame_from_planes,
    frame_planes,
    mux_audio_packet,
)


@dataclass
class FrameInfo:
    number: int
    timestamp: float
    width: int
    height: int
    debug: bool = False
    debug2: bool = False
    slowok: bool = False
    boxes: list = field(default_factory=list)


# Processing functions: func(frame: np.ndarray, info: FrameInfo) -> list of
# (x, y, width, height) boxes. Called in order; results are appended to
# info.boxes. The input frame must not be modified.
PROCESSING_FUNCTIONS = [
    procfun_retinaface,
    procfun_deepface,
    procfun_nudenet,
    procfun_hyperlpr,
    procfun_debug2,
]


def clip_box(box, width, height):
    """Clamp a (x, y, w, h) box to the frame; return (x0, y0, x1, y1) or None."""
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def blur_kernel_size(width, height):
    k = max(15, max(width, height) // 40)
    return k if k % 2 else k + 1


def render_blur(out_frame, in_frame, info):
    """Blur the union of all boxes so overlapping pixels are blurred once."""
    rects = [r for r in (clip_box(b, info.width, info.height) for b in info.boxes) if r]
    if not rects:
        return
    mask = np.zeros((info.height, info.width), dtype=bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True

    # Blur only the area covering all boxes, with margin so the kernel sees
    # real neighbouring pixels at region edges.
    k = blur_kernel_size(info.width, info.height)
    pad = k // 2
    ux0 = max(0, min(r[0] for r in rects) - pad)
    uy0 = max(0, min(r[1] for r in rects) - pad)
    ux1 = min(info.width, max(r[2] for r in rects) + pad)
    uy1 = min(info.height, max(r[3] for r in rects) + pad)

    blurred = cv2.GaussianBlur(in_frame[uy0:uy1, ux0:ux1], (k, k), 0)
    region_mask = mask[uy0:uy1, ux0:ux1]
    out_region = out_frame[uy0:uy1, ux0:ux1]
    out_region[region_mask] = blurred[region_mask]


def render_debug_boxes(out_frame, in_frame, info):
    if not info.debug:
        return
    for box in info.boxes:
        rect = clip_box(box, info.width, info.height)
        if rect:
            x0, y0, x1, y1 = rect
            cv2.rectangle(out_frame, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)


def format_timestamp(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_debug_timestamp(out_frame, in_frame, info):
    if not info.debug:
        return
    text = f"{format_timestamp(info.timestamp)}  frame {info.number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, info.height / 720)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = int(10 * scale)
    x, y = margin, info.height - margin - baseline
    cv2.rectangle(out_frame, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(out_frame, text, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


# Rendering functions: func(out_frame, in_frame, info) -> None. Called in
# order; they modify out_frame in place.
RENDERING_FUNCTIONS = [
    render_blur,
    render_debug_boxes,
    render_debug_timestamp,
]


def process_frame(in_frame, info):
    for func in PROCESSING_FUNCTIONS:
        boxes = func(in_frame, info)
        if boxes:
            info.boxes.extend(boxes)
    out_frame = in_frame.copy()
    for func in RENDERING_FUNCTIONS:
        func(out_frame, in_frame, info)
    return out_frame


def downsample_mask(mask, shape):
    """Shrink a boolean mask to shape; a cell is set if any covered pixel is set."""
    ph, pw = shape
    sy, sx = -(-mask.shape[0] // ph), -(-mask.shape[1] // pw)
    padded = np.zeros((ph * sy, pw * sx), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    return padded.reshape(ph, sy, pw, sx).any(axis=(1, 3))


def build_output_frame(in_frame, in_image, out_image, pix_fmt):
    """Create the frame to encode, keeping unmodified pixels bit-exact.

    Converting YUV -> BGR -> YUV loses detail (chroma resampling, rounding),
    so only pixels changed by the renderers are taken from out_image; all
    other pixels come straight from the decoded input planes.
    """
    width, height = in_frame.width, in_frame.height
    if in_frame.format.name != pix_fmt:
        in_frame = in_frame.reformat(format=pix_fmt)
    planes = frame_planes(in_frame)

    changed = np.any(out_image != in_image, axis=2)
    if changed.any():
        new = av.VideoFrame.from_ndarray(out_image, format="bgr24").reformat(format=pix_fmt)
        for orig, repl in zip(planes, frame_planes(new)):
            mask = changed if orig.shape == changed.shape else downsample_mask(changed, orig.shape)
            orig[mask] = repl[mask]
    return frame_from_planes(planes, pix_fmt, width, height)


def filter_in_to_out(input_file, output_file, debug=False, debug2=False, slowok=False):
    with av.open(input_file) as in_container, av.open(output_file, "w") as out_container:
        if not in_container.streams.video:
            raise ValueError(f"{input_file}: no video stream found")
        in_video = in_container.streams.video[0]
        in_video.thread_type = "AUTO"
        width = in_video.codec_context.width
        height = in_video.codec_context.height

        out_video = add_video_encoder(out_container, in_video, width, height)
        audio_map = add_audio_copies(out_container, in_container)
        progress = Progress(in_container, in_video)

        if debug:
            print(f"Debug: input {width}x{height} @ {in_video.average_rate} fps, "
                  f"{len(audio_map)} audio stream(s)")

        frame_number = 0
        for packet in in_container.demux(in_video, *in_container.streams.audio):
            if mux_audio_packet(out_container, packet, audio_map):
                continue

            for frame in packet.decode():
                image = frame.to_ndarray(format="bgr24")
                timestamp = float(frame.time) if frame.time is not None else 0.0
                info = FrameInfo(frame_number, timestamp, width, height,
                                 debug, debug2, slowok)
                out_image = process_frame(image, info)

                out_frame = build_output_frame(frame, image, out_image, out_video.pix_fmt)
                out_frame.pts = frame.pts
                out_frame.time_base = frame.time_base
                out_container.mux(out_video.encode(out_frame))
                frame_number += 1
                progress.update(timestamp)

        out_container.mux(out_video.encode())
        progress.finish()

        if debug:
            print(f"Debug: wrote {frame_number} frames to {output_file}")
