"""Reading video: metadata, and sampling frames at a chosen rate in Hz.

This is the only module in the pipeline that knows what a frame index is.
Everything downstream works in **seconds**, because the hidden videos may not
share this one's frame rate, and a rule written in frames would silently mean
something different on a 25 fps clip.

Two details that matter more than they look:

* **Every time comes from the file.** Each frame is stored with its own
  timestamp (its presentation time). We read it, with the picture, from the same
  decoder (PyAV), and never compute ``index / fps``. That formula assumes evenly
  spaced frames; screen recordings are not: Untitled3.mov records 34 frames/s for
  its first 10 s and 24 after, so ``index / fps`` ran 3.4 s ahead of the video.
  OpenCV is not used to read video: its timestamp is sometimes exact, sometimes
  0 depending on the build, and the old code silently fell back to
  ``index / fps`` when it read 0. A frame with no timestamp is an error, never a
  guess.

* **Convert only the frames we keep.** Every frame is decoded (a video cannot
  be skipped through without decoding), but only the sampled ones are turned
  into images.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    """What we know about an input file before processing it."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    # "constant" | "variable": whether the stored frame timestamps are evenly
    # spaced. Reported only -- every time is read from the file either way.
    timeline: str = "constant"

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def stride_for(self, rate_hz: float) -> int:
        """How many source frames to advance to sample at roughly ``rate_hz``.

        This is the single place where a rate in Hz becomes a number of frames.
        Never at least 1, so a slow-frame-rate video is sampled every frame
        rather than being skipped past.
        """
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        return max(1, round(self.fps / rate_hz))

    def summary(self) -> str:
        source = {
            "constant": "timestamps stored in the file  (constant frame rate)",
            "variable": "timestamps stored in the file  (VARIABLE frame rate)",
        }[self.timeline]
        return (
            f"{self.path.name}\n"
            f"  resolution : {self.resolution}\n"
            f"  frame rate : {self.fps:.6f} fps\n"
            f"  frames     : {self.frame_count}\n"
            f"  duration   : {self.duration_seconds:.2f} s\n"
            f"  time from  : {source}"
        )


@dataclass
class Sample:
    """One processed moment in the video.

    ``time_seconds`` is the authoritative reference for everything downstream;
    ``frame_index`` exists only for rendering overlays back onto the source video.
    """

    frame_index: int
    time_seconds: float
    image: np.ndarray  # BGR, as OpenCV's image functions expect


def decode(path: str | Path) -> Iterator[tuple[int, float, av.VideoFrame]]:
    """Every frame of the video, in display order: ``(index, seconds, frame)``.

    ``seconds`` is the frame's own stored timestamp, measured from the first
    frame. This is the ONLY place in the pipeline a frame gets a time; the index
    is its position in this sequence, and the renderer uses the same sequence,
    so an index means the same frame everywhere.

    Raises if a frame has no timestamp: guessing one from the frame rate is the
    bug this replaces."""
    path = Path(path)
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise RuntimeError(f"could not open video: {path}: {exc}") from None
    with container:
        stream = container.streams.video[0]
        start = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise RuntimeError(
                    f"{path.name}: frame {index} has no timestamp, so its time in "
                    "seconds is unknown; refusing to guess it from the frame rate"
                )
            if start is None:
                start = frame.time
            yield index, float(frame.time - start), frame


def frame_times(path: str | Path) -> np.ndarray:
    """Every frame's stored time in seconds, from the first frame. Decodes the
    whole video (no images are made), so call it once and keep the result."""
    return np.array([seconds for _, seconds, _ in decode(path)], dtype=float)


def classify_timeline(times: np.ndarray, tolerance: float = 0.002) -> str:
    """ "constant" when the stored frame intervals are even, else "variable".

    Reported, never used to choose a clock: every time is read from the file."""
    gaps = np.diff(np.asarray(times, dtype=float))
    if len(gaps) < 2:
        return "constant"
    median = float(np.median(gaps))
    if median <= 0:
        return "variable"
    return "constant" if float(np.ptp(gaps)) / median <= tolerance else "variable"


def probe(path: str | Path, verify: bool = False) -> VideoInfo:
    """Read a video's metadata.

    Args:
        path: the video file.
        verify: count the frames that actually decode instead of trusting the
            container's header. Costs one demux pass (no pixel decoding), and it
            is worth it: the task video's header claims 1102 frames when only 886
            exist, which would overstate its duration by 20%. Any quantity
            derived from duration would inherit that error silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise RuntimeError(f"could not open video: {path}: {exc}") from None
    with container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        frame_count = int(stream.frames or 0)
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)

    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError(f"video reports an unusable frame rate ({fps})")

    timeline = "constant"
    duration = frame_count / fps if frame_count > 0 else 0.0
    if verify:
        times = frame_times(path)
        counted = len(times)
        if counted != frame_count:
            log.warning(
                "%s: header claims %d frames, %d decode; using the decoded count",
                path.name,
                frame_count,
                counted,
            )
            frame_count = counted
        timeline = classify_timeline(times)
        if counted:
            # the last frame is shown for one typical frame interval
            step = float(np.median(np.diff(times))) if counted > 1 else 1.0 / fps
            duration = float(times[-1]) + step
            fps = counted / duration
        if timeline == "variable":
            log.warning(
                "%s: VARIABLE frame rate -- every time is the frame's stored "
                "timestamp; `fps` is only the average (%.3f)",
                path.name,
                fps,
            )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        timeline=timeline,
    )


def iter_samples(
    path: str | Path,
    rate_hz: float,
    max_edge: int | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Iterator[Sample]:
    """Yield frames at approximately ``rate_hz``, with real timestamps.

    Args:
        path: video file.
        rate_hz: how many samples per second to yield.
        max_edge: if set, downscale so the longest side is at most this many
            pixels. Speeds up inference; keep an eye on it, since a bucket that
            becomes a handful of pixels is a bucket whose angle is noise.
        start_seconds, end_seconds: optional trim, useful for quick experiments.

    Yields:
        Samples in time order. Frames are decoded lazily -- the whole video is
        never held in memory, which matters on the 8 GB GPU node.
    """
    info = probe(path)
    stride = info.stride_for(rate_hz)

    for index, time_seconds, frame in decode(path):
        if index % stride:
            continue
        if time_seconds < start_seconds:
            continue
        if end_seconds is not None and time_seconds > end_seconds:
            break
        image = frame.to_ndarray(format="bgr24")
        if max_edge is not None:
            image = _downscale(image, max_edge)
        yield Sample(frame_index=index, time_seconds=time_seconds, image=image)


def _downscale(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Shrink so the longest side is ``max_edge``; never upscale."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
