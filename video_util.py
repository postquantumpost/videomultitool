"""Shared video I/O helpers: encoder setup, raw plane access, audio copy, progress."""

import av
import numpy as np

# Output encoding: x264 constant-quality mode. CRF 18 is visually lossless for
# most content; lower values give higher quality and larger files.
X264_CRF = 18
X264_PRESET = "medium"
X264_PIX_FMTS = {
    "yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p",
}


def output_pix_fmt(in_video):
    in_pix_fmt = in_video.codec_context.pix_fmt
    return in_pix_fmt if in_pix_fmt in X264_PIX_FMTS else "yuv420p"


def add_video_encoder(out_container, in_video, width, height):
    """Add an x264 constant-quality output stream matching in_video's timing."""
    rate = in_video.average_rate or in_video.guessed_rate or 30
    out_video = out_container.add_stream("libx264", rate=rate)
    out_video.width = width
    out_video.height = height
    out_video.pix_fmt = output_pix_fmt(in_video)
    out_video.time_base = in_video.time_base
    # PyAV defaults new streams to a 1 Mbps bitrate; clear it so CRF
    # (constant quality) rate control is used instead.
    out_video.codec_context.bit_rate = 0
    out_video.options = {"crf": str(X264_CRF), "preset": X264_PRESET}
    return out_video


def add_stream_copy(out_container, in_stream):
    """Add an output stream that copies in_stream's packets unchanged."""
    # PyAV >= 13 replaced add_stream(template=...) with add_stream_from_template.
    if hasattr(out_container, "add_stream_from_template"):
        return out_container.add_stream_from_template(in_stream)
    return out_container.add_stream(template=in_stream)


def add_audio_copies(out_container, in_container):
    """Add pass-through copies of all audio streams; return {input index: output stream}."""
    return {
        in_audio.index: add_stream_copy(out_container, in_audio)
        for in_audio in in_container.streams.audio
    }


def mux_audio_packet(out_container, packet, audio_map):
    """Mux packet if it belongs to a copied audio stream; return True if handled."""
    if packet.stream.index not in audio_map:
        return False
    if packet.dts is not None:
        packet.stream = audio_map[packet.stream.index]
        out_container.mux(packet)
    return True


def stream_duration(container, stream):
    """Return (start_time, duration) of the stream in seconds; duration may be None."""
    start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
    if stream.duration:
        return start, float(stream.duration * stream.time_base)
    if container.duration:
        return start, container.duration / av.time_base
    return start, None


class Progress:
    """Print a line each time another 10% of the input has been processed."""

    def __init__(self, container, stream):
        self.start, self.duration = stream_duration(container, stream)
        self.next_percent = 10
        self.frames = 0

    def update(self, timestamp):
        self.frames += 1
        if not self.duration:
            return
        percent = 100 * (timestamp - self.start) / self.duration
        while self.next_percent < 100 and percent >= self.next_percent:
            print(f"Progress: {self.next_percent}% ({self.frames} frames)")
            self.next_percent += 10

    def finish(self):
        print(f"Progress: 100% ({self.frames} frames)")


def frame_planes(frame):
    """Return copies of a planar 8-bit frame's planes as 2-D uint8 arrays."""
    planes = []
    for plane in frame.planes:
        buf = np.frombuffer(plane, dtype=np.uint8)
        buf = buf[: plane.height * plane.line_size].reshape(plane.height, plane.line_size)
        planes.append(buf[:, : plane.width].copy())
    return planes


def frame_from_planes(planes, pix_fmt, width, height):
    frame = av.VideoFrame(width, height, pix_fmt)
    for plane, data in zip(frame.planes, planes):
        buf = np.zeros((plane.height, plane.line_size), dtype=np.uint8)
        buf[:, : plane.width] = data
        plane.update(buf)
    return frame
