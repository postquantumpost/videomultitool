"""Remove letterbox/pillarbox borders: detect the real video region and crop to it.

The region is detected once for the whole file from frames sampled across its
duration. Border lines are those that are dark nearly all the time (black bars,
possibly with text such as subtitles or UI menus drawn on them) or completely
static (filler such as gradients or a frame). A border is only cropped where a
strong persistent straight edge marks the boundary and the bar is almost
entirely static, so dark or static parts of a real full-frame scene are kept.

Cropping is done on the decoded YUV planes (no colour conversion) and the
result is re-encoded with x264 in constant-quality mode. Audio is copied.
"""

import av
import numpy as np

from video_util import (
    X264_PIX_FMTS,
    Progress,
    add_audio_copies,
    add_video_encoder,
    frame_from_planes,
    frame_planes,
    mux_audio_packet,
    output_pix_fmt,
    stream_duration,
)

BLANKLESS_SAMPLES = 40
# Luma at or below this counts as dark (limited-range black is 16).
DARK_Y = 40
# A line is a border if dark in at least this fraction of its pixel samples.
BAR_DARK_FRAC = 0.7
# A pixel is static if its luma std-dev over the samples is below this...
STATIC_STD = 2.0
# ...and a line is static filler if at least this fraction of its pixels are.
BAR_STATIC_FRAC = 0.97
# A cropped bar must have at least this fraction of static pixels (bars may
# carry changing text such as subtitles, but real video moves almost everywhere).
BAR_REGION_STATIC_FRAC = 0.8
# Gaps of border-like lines shorter than this fraction are bridged.
GAP_FRAC = 0.02
# Boundaries are snapped to a persistent edge within this fraction of the size.
EDGE_SEARCH_FRAC = 0.03
# An edge must be this many times stronger than typical nearby edges...
EDGE_PEAK_RATIO = 2.5
# ...and at least this strong (mean absolute luma step).
EDGE_MIN_STRENGTH = 6.0
# Extra pixels trimmed inside each cropped boundary to drop blended edge pixels.
EDGE_MARGIN = 2
# Never crop to less than this fraction of either dimension.
MIN_REGION_FRAC = 0.2


def luma_plane(frame):
    if frame.format.name not in X264_PIX_FMTS:
        frame = frame.reformat(format="yuv420p")
    return frame_planes(frame)[0]


def sample_luma(container, stream, count):
    """Decode about `count` frames spread evenly over the stream; return luma planes."""
    start, duration = stream_duration(container, stream)
    planes = []
    if duration:
        for i in range(count):
            target = start + duration * (i + 0.5) / count
            try:
                container.seek(int(target / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    if frame.time is None or frame.time >= target - 1e-3:
                        planes.append(luma_plane(frame))
                        break
            except (av.FFmpegError, StopIteration):
                break
    if len(planes) < count // 2:
        # Seeking failed or duration unknown: decode from the start instead.
        container.seek(0)
        planes = [luma_plane(f) for _, f in zip(range(count * 10), container.decode(stream))]
        planes = planes[:: max(1, len(planes) // count)]
    if not planes:
        raise ValueError("could not decode any video frames")
    return np.stack(planes)


class RegionStats:
    def __init__(self, luma):
        self.luma = luma
        lf = luma.astype(np.float32)
        self.dark = (luma <= DARK_Y).mean(axis=0)
        self.static = (lf.std(axis=0) < STATIC_STD).astype(np.float32)

    def line_profiles(self, axis, lo, hi):
        """Per-line dark and static fractions; axis 1 = columns, 0 = rows."""
        if axis == 1:
            return self.dark[lo:hi].mean(axis=0), self.static[lo:hi].mean(axis=0)
        return self.dark[:, lo:hi].mean(axis=1), self.static[:, lo:hi].mean(axis=1)

    def static_fraction(self, axis, lo, hi, a, b):
        """Fraction of static pixels in lines [a, b) limited to [lo, hi) across."""
        if axis == 1:
            return float(self.static[lo:hi, a:b].mean())
        return float(self.static[a:b, lo:hi].mean())

    def edge_profile(self, axis, lo, hi):
        """Strength of the boundary before each line, persistent across samples.

        Entry i measures the step between line i-1 and line i (entry 0 is 0),
        averaged along the line and then taking the median across samples.
        """
        if axis == 1:
            part = self.luma[:, lo:hi, :].astype(np.int16)
            steps = np.abs(np.diff(part, axis=2)).mean(axis=1)
        else:
            part = self.luma[:, :, lo:hi].astype(np.int16)
            steps = np.abs(np.diff(part, axis=1)).mean(axis=2)
        profile = np.median(steps, axis=0)
        return np.concatenate([[0.0], profile])


def content_span(border, gap):
    """Largest run of non-border lines, bridging border gaps up to `gap` long."""
    n = len(border)
    runs = []
    i = 0
    while i < n:
        if border[i]:
            i += 1
            continue
        j = i
        while j < n and not border[j]:
            j += 1
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j
    if not runs:
        return None
    return tuple(max(runs, key=lambda r: r[1] - r[0]))


def snap_to_edge(edges, boundary, search):
    """Return the persistent edge position near `boundary`, or None if there is none."""
    n = len(edges)
    lo, hi = max(1, boundary - search), min(n - 1, boundary + search)
    if hi <= lo:
        return None
    peak = lo + int(np.argmax(edges[lo:hi + 1]))
    ctx = edges[max(1, peak - 4 * search):min(n, peak + 4 * search + 1)]
    strength = edges[peak]
    if strength >= EDGE_MIN_STRENGTH and strength >= EDGE_PEAK_RATIO * np.median(ctx):
        return peak
    return None


def find_span(stats, axis, lo, hi, size, debug=False):
    """Find the [start, end) span of real video along one axis."""
    name = "columns" if axis == 1 else "rows"
    dark, static = stats.line_profiles(axis, lo, hi)
    gap = max(2, int(size * GAP_FRAC))
    dark_border = dark >= BAR_DARK_FRAC
    dark_span = content_span(dark_border, gap) or (0, size)
    full_span = content_span(dark_border | (static >= BAR_STATIC_FRAC), gap) or dark_span

    edges = stats.edge_profile(axis, lo, hi)
    search = max(4, int(size * EDGE_SEARCH_FRAC))
    result = []
    for side in (0, 1):
        label = ("start", "end")[side]
        chosen, reason = None, "no border"
        for kind, boundary in (("static", full_span[side]), ("dark", dark_span[side])):
            if boundary in (0, size):
                continue
            edge = snap_to_edge(edges, boundary, search)
            if edge is None:
                reason = f"{kind} border at {boundary} has no edge"
                continue
            bar = (0, edge) if side == 0 else (edge, size)
            frac = stats.static_fraction(axis, lo, hi, *bar)
            if frac < BAR_REGION_STATIC_FRAC:
                reason = f"{kind} border at {edge} is not static ({frac:.2f})"
                continue
            chosen = (kind, edge)
            break
        boundary = chosen[1] if chosen else (0, size)[side]
        if debug:
            if chosen:
                print(f"Debug: {name} {label}: {chosen[0]} border, edge at {boundary}")
            else:
                print(f"Debug: {name} {label}: not cropped ({reason})")
        result.append(boundary)
    return tuple(result)


def align_span(start, end, size, step):
    """Trim blended edge pixels and align a cropped span to `step` pixels."""
    if start > 0:
        start += EDGE_MARGIN
    if end < size:
        end -= EDGE_MARGIN
    start = -(-start // step) * step
    end = end // step * step
    return start, end


def detect_video_region(container, stream, debug=False):
    """Return (x, y, width, height) of the real video region."""
    width = stream.codec_context.width
    height = stream.codec_context.height
    stats = RegionStats(sample_luma(container, stream, BLANKLESS_SAMPLES))
    if debug:
        print(f"Debug: sampled {len(stats.luma)} frames for border detection")

    x0, x1 = find_span(stats, 1, 0, height, width, debug)
    y0, y1 = find_span(stats, 0, x0, x1, height, debug)
    x0, x1 = find_span(stats, 1, y0, y1, width, debug)

    x0, x1 = align_span(x0, x1, width, 2)
    y0, y1 = align_span(y0, y1, height, 2)
    if x1 - x0 < width * MIN_REGION_FRAC or y1 - y0 < height * MIN_REGION_FRAC:
        print("Warning: detected video region is implausibly small; not cropping")
        return 0, 0, width, height
    return x0, y0, x1 - x0, y1 - y0


def crop_frame(frame, pix_fmt, x, y, width, height):
    """Crop a frame on its YUV planes, without colour conversion."""
    if frame.format.name != pix_fmt:
        frame = frame.reformat(format=pix_fmt)
    cropped = []
    for plane in frame_planes(frame):
        sx = frame.width // plane.shape[1]
        sy = frame.height // plane.shape[0]
        cropped.append(plane[y // sy:(y + height) // sy, x // sx:(x + width) // sx])
    return frame_from_planes(cropped, pix_fmt, width, height)


def blankless_in_to_out(input_file, output_file, debug=False):
    with av.open(input_file) as in_container:
        if not in_container.streams.video:
            raise ValueError(f"{input_file}: no video stream found")
        in_video = in_container.streams.video[0]
        in_video.thread_type = "AUTO"
        in_width = in_video.codec_context.width
        in_height = in_video.codec_context.height
        x, y, width, height = detect_video_region(in_container, in_video, debug)

    if (width, height) == (in_width, in_height):
        print(f"Video region: full frame {width}x{height} (no borders found)")
    else:
        print(f"Video region: {width}x{height} at ({x}, {y}) "
              f"in {in_width}x{in_height} input")

    with av.open(input_file) as in_container, av.open(output_file, "w") as out_container:
        in_video = in_container.streams.video[0]
        in_video.thread_type = "AUTO"
        out_video = add_video_encoder(out_container, in_video, width, height)
        pix_fmt = output_pix_fmt(in_video)
        audio_map = add_audio_copies(out_container, in_container)
        progress = Progress(in_container, in_video)

        for packet in in_container.demux(in_video, *in_container.streams.audio):
            if mux_audio_packet(out_container, packet, audio_map):
                continue
            for frame in packet.decode():
                out_frame = crop_frame(frame, pix_fmt, x, y, width, height)
                out_frame.pts = frame.pts
                out_frame.time_base = frame.time_base
                out_container.mux(out_video.encode(out_frame))
                progress.update(float(frame.time) if frame.time is not None else 0.0)

        out_container.mux(out_video.encode())
        progress.finish()

        if debug:
            print(f"Debug: wrote {progress.frames} frames to {output_file}")
