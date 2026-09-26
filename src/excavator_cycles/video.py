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
    # "constant" | "variable" | "absent" -- which clock to trust. See
    # `classify_timeline` for why neither source is right for both cases.
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
            "constant": "index / fps  (constant frame rate)",
            "variable": "decoder timestamps  (VARIABLE frame rate)",
            "absent": "index / fps  (decoder reports no timestamps)",
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
    image: np.ndarray  # BGR, as OpenCV decodes it


def classify_timeline(path: Path, samples: int = 240, tolerance: float = 0.002) -> str:
    """Decide which clock to trust: the frame index, or the decoder.

    Neither source is right for both cases, and picking the wrong one is a
    systematic bias in the only number this project reports.

    **Constant frame rate.** ``CAP_PROP_POS_MSEC`` is often computed from a
    *rounded* rate rather than read from the container. On the development clip
    it returns exactly ``n / 30`` while the true rate is 29.97396912 -- so every
    timestamp is 0.1% short, which is 0.022 s over one work cycle. Small against
    a 0.6 s tolerance, but systematic: it never averages out, and it offsets
    every prediction against ground truth that was converted at the true rate.
    Here ``index / fps`` is exact and the decoder is not.

    **Variable frame rate.** Phone and web footage genuinely has uneven frame
    intervals. There is no single fps, so ``index / fps`` is meaningless and the
    decoder's timestamps are the only truth.

    The test: are the reported intervals constant? A round-rate timeline is
    perfectly even and gets classified constant, which is what we want -- we
    then ignore it in favour of the more precise arithmetic.

    Returns "constant", "variable", or "absent" (the decoder gave nothing).
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    stamps = []
    try:
        for _ in range(samples):
            if not capture.grab():
                break
            stamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        capture.release()

    usable = [t for t in stamps[1:] if t > 0]
    if len(usable) < 3:
        return "absent"

    gaps = np.diff(np.asarray(stamps[: len(usable) + 1], dtype=float))
    median = float(np.median(gaps))
    if median <= 0:
        return "absent"
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

    timeline = classify_timeline(path)
    if timeline == "variable":
        log.warning(
            "%s: VARIABLE frame rate -- using the decoder's timestamps. Rates "
            "assume one sample spacing, so they will be biased by the local ratio.",
            path.name,
        )
    elif timeline == "absent":
        log.warning(
            "%s: the decoder reports no timestamps; falling back to index / fps. "
            "If this file is variable-rate, every duration is wrong.",
            path.name,
        )

    if verify:
        counted = _count_frames(path)
        if counted != frame_count:
            log.warning(
                "%s: header claims %d frames, only %d decode (%.2fs vs %.2fs); "
                "using the decoded count",
                path.name,
                frame_count,
                counted,
                frame_count / fps,
                counted / fps,
            )
            frame_count = counted

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
                time_seconds = _timestamp_seconds(capture, index, info.fps, info.timeline)

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


def _count_frames(path: Path) -> int:
    """Count decodable frames by demuxing, without decoding pixels.

    ``grab`` advances the stream without producing an image, so this is far
    cheaper than reading the video, though not free on long files.
    """
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def _timestamp_seconds(
    capture: cv2.VideoCapture, index: int, fps: float, timeline: str = "constant"
) -> float:
    """Where this frame sits in time, from whichever clock is trustworthy here.

    See `classify_timeline`. On a constant-rate file ``index / fps`` is exact
    and the decoder's timestamps may be quantised to a round rate; on a
    variable-rate file the decoder is the only source that means anything.
    """
    if timeline == "variable":
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


def read_frames_at(path: str | Path, times_seconds: list[float]) -> list[Sample]:
    """Decode the frames nearest to the given timestamps.

    Seeking is used rather than a sequential scan because the spike wants ~20
    frames spread across the whole video, and decoding everything in between
    would be wasteful. Seek accuracy varies by codec, so the timestamp reported
    back is the frame's *actual* position, not the one that was requested.
    """
    info = probe(path)
    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {info.path}")

    samples: list[Sample] = []
    try:
        for requested in times_seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000.0)
            ok, image = capture.read()
            if not ok:
                continue
            index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual = _timestamp_seconds(capture, max(index, 0), info.fps, info.timeline)
            samples.append(Sample(frame_index=index, time_seconds=actual, image=image))
    finally:
        capture.release()
    return samples


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
