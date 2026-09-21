"""Reading video: metadata, and sampling frames at a chosen rate in Hz.

This is the only module in the pipeline that knows what a frame index is.
Everything downstream works in **seconds**, because the hidden videos may not
share this one's frame rate, and a rule written in frames would silently mean
something different on a 25 fps clip.

Two details that matter more than they look:

* **Real timestamps.** We ask the decoder where a frame actually sits in time
  rather than computing ``index / fps``. A file that reports 30 fps but is truly
  29.97 drifts about 0.1% -- small, until you remember the whole task is graded
  at +/-0.6 s.

* **Grab, then retrieve.** Decoding a frame is far more expensive than skipping
  one. When sampling at 10 Hz from 30 fps video we only fully decode every third
  frame and cheaply skip the rest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """What we know about an input file before processing it."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

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
        return (
            f"{self.path.name}\n"
            f"  resolution : {self.resolution}\n"
            f"  frame rate : {self.fps:.3f} fps\n"
            f"  frames     : {self.frame_count}\n"
            f"  duration   : {self.duration_seconds:.2f} s"
        )


@dataclass
class Sample:
    """One processed moment in the video.

    ``time_seconds`` is the authoritative reference for everything downstream;
    ``frame_index`` exists only for rendering overlays back onto the source video.
    """

    frame_index: int
    time_seconds: float
    image: np.ndarray  # BGR, as OpenCV decodes it


def probe(path: str | Path) -> VideoInfo:
    """Read a video's metadata without decoding its contents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError(
            f"video reports an unusable frame rate ({fps}); cannot convert to seconds"
        )

    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
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

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {info.path}")

    try:
        index = 0
        while True:
            # `grab` advances without decoding pixels; `retrieve` does the
            # expensive part, and we only pay it on frames we actually want.
            grabbed = capture.grab()
            if not grabbed:
                break

            if index % stride == 0:
                time_seconds = _timestamp_seconds(capture, index, info.fps)

                if time_seconds < start_seconds:
                    index += 1
                    continue
                if end_seconds is not None and time_seconds > end_seconds:
                    break

                ok, image = capture.retrieve()
                if ok:
                    if max_edge is not None:
                        image = _downscale(image, max_edge)
                    yield Sample(
                        frame_index=index,
                        time_seconds=time_seconds,
                        image=image,
                    )

            index += 1
    finally:
        capture.release()


def _timestamp_seconds(capture: cv2.VideoCapture, index: int, fps: float) -> float:
    """The frame's real position in time, falling back to index/fps.

    Some containers do not report per-frame timestamps. The fallback keeps the
    pipeline running, but it is worth knowing which one you got, because only
    the first is immune to a 29.97-vs-30 mismatch.
    """
    milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
    if milliseconds and milliseconds > 0:
        return float(milliseconds) / 1000.0
    return index / fps


def _downscale(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Shrink so the longest side is ``max_edge``; never upscale."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def sample_times(info: VideoInfo, count: int) -> list[float]:
    """``count`` timestamps spread evenly across the video.

    Used by the detector spike: we want frames from throughout the clip, so the
    sample covers every phase, not just whichever one the video opens in.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [info.duration_seconds / 2]
    # Avoid the very first and last frames, which are often black or torn.
    margin = info.duration_seconds * 0.02
    span = info.duration_seconds - 2 * margin
    return [margin + span * i / (count - 1) for i in range(count)]
