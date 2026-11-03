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
    blur_edge: bool = True
    fill: bool = False
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


# Boxes up to this size (max of width, height) use the base blur; larger boxes
# get proportionally more (twice the size -> twice the blur).
BLUR_BASE_BOX_SIZE = 100
# With blur_edge, the blur extends this fraction of the box width/height
# beyond each side, fading out to no blur.
BLUR_EDGE_FRAC = 0.10
# Kernels larger than this are applied on a downscaled image for speed.
BLUR_MAX_DIRECT_KERNEL = 61


def odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def blur_kernel_size(width, height):
    """Base blur kernel for a frame of this size."""
    return odd(max(15, max(width, height) // 40))


def box_kernel_size(base_k, box_w, box_h):
    """Scale the base kernel with box size (linear above BLUR_BASE_BOX_SIZE)."""
    return odd(base_k * max(1.0, max(box_w, box_h) / BLUR_BASE_BOX_SIZE))


def gaussian_blur(image, k):
    """Gaussian blur with kernel k; very large kernels use a downscaled copy."""
    if k <= BLUR_MAX_DIRECT_KERNEL:
        return cv2.GaussianBlur(image, (k, k), 0)
    h, w = image.shape[:2]
    f = -(-k // BLUR_MAX_DIRECT_KERNEL)
    small = cv2.resize(image, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (odd(k / f), odd(k / f)), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def fade_profile(lo, hi, margin, start, stop):
    """1 inside [lo, hi), fading linearly to 0 over `margin` pixels outside it,
    sampled for pixel positions start..stop-1."""
    pos = np.arange(start, stop, dtype=np.float32) + 0.5
    dist = np.maximum(np.maximum(lo - pos, pos - hi), 0)
    if margin <= 0:
        return (dist == 0).astype(np.float32)
    return np.clip(1 - dist / (margin + 1), 0, 1)


def render_blur(out_frame, in_frame, info):
    """Blur (or fill) every box, blending boxes so each pixel is changed once.

    Each box gets a weight map (1 inside, fading to 0 over the edge margin when
    blur_edge is set) and a replacement image (the input blurred with a kernel
    scaled to the box size, or the box's average colour with fill). Where boxes
    overlap, the replacements are averaged by weight, and the result is blended
    with the input by the strongest weight, always starting from the input.
    """
    rects = [r for r in (clip_box(b, info.width, info.height) for b in info.boxes) if r]
    if not rects:
        return
    base_k = blur_kernel_size(info.width, info.height)

    regions = []
    for x0, y0, x1, y1 in rects:
        bw, bh = x1 - x0, y1 - y0
        mx = int(round(bw * BLUR_EDGE_FRAC)) if info.blur_edge else 0
        my = int(round(bh * BLUR_EDGE_FRAC)) if info.blur_edge else 0
        ex0, ey0 = max(0, x0 - mx), max(0, y0 - my)
        ex1, ey1 = min(info.width, x1 + mx), min(info.height, y1 + my)
        weight = np.outer(fade_profile(y0, y1, my, ey0, ey1),
                          fade_profile(x0, x1, mx, ex0, ex1))
        if info.fill:
            # The fill colour is mixed in by weight twice (into the blurred edge,
            # then over the input), so take the root to keep its fade linear.
            weight = np.sqrt(weight)
        regions.append(((x0, y0, x1, y1), (ex0, ey0, ex1, ey1), weight))

    ux0 = min(r[1][0] for r in regions)
    uy0 = min(r[1][1] for r in regions)
    ux1 = max(r[1][2] for r in regions)
    uy1 = max(r[1][3] for r in regions)
    acc = np.zeros((uy1 - uy0, ux1 - ux0, 3), dtype=np.float32)
    wsum = np.zeros((uy1 - uy0, ux1 - ux0), dtype=np.float32)
    wmax = np.zeros_like(wsum)

    for (x0, y0, x1, y1), (ex0, ey0, ex1, ey1), weight in regions:
        inner = weight >= 1.0
        need_blur = not (info.fill and inner.all())
        if need_blur:
            k = box_kernel_size(base_k, x1 - x0, y1 - y0)
            # Blur with surrounding context so the kernel sees real neighbours.
            pad = k // 2
            px0, py0 = max(0, ex0 - pad), max(0, ey0 - pad)
            px1, py1 = min(info.width, ex1 + pad), min(info.height, ey1 + pad)
            blurred = gaussian_blur(in_frame[py0:py1, px0:px1], k)
            repl = blurred[ey0 - py0:ey1 - py0, ex0 - px0:ex1 - px0].astype(np.float32)
        if info.fill:
            color = in_frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0).astype(np.float32)
            if need_blur:
                # Fade the fill colour into the blurred edge.
                repl = repl + (color - repl) * weight[..., None]
            else:
                repl = np.broadcast_to(color, weight.shape + (3,))

        sl = (slice(ey0 - uy0, ey1 - uy0), slice(ex0 - ux0, ex1 - ux0))
        acc[sl] += repl * weight[..., None]
        wsum[sl] += weight
        np.maximum(wmax[sl], weight, out=wmax[sl])

    touched = wmax > 0
    if not touched.any():
        return
    orig = in_frame[uy0:uy1, ux0:ux1].astype(np.float32)
    repl = acc[touched] / wsum[touched][:, None]
    a = wmax[touched][:, None]
    result = orig[touched] * (1 - a) + repl * a
    out_region = out_frame[uy0:uy1, ux0:ux1]
    out_region[touched] = np.clip(np.rint(result), 0, 255).astype(np.uint8)


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


def filter_in_to_out(input_file, output_file, debug=False, debug2=False, slowok=False,
                     blur_edge=True, fill=False):
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
                                 debug, debug2, slowok, blur_edge, fill)
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
