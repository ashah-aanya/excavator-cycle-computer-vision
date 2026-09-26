"""Stage 2: turn a video into a mask of the excavator for every sample.

The pipeline's measurements all come from these masks, so this stage decides how
good everything downstream can be. It runs the two models -- Grounding DINO to
find the machine, SAM 2 to outline and follow it -- and writes masks, prompts and
quality metrics to a cache that later stages read without a GPU.

Three decisions here were made from measurements on real footage, recorded in
docs/stages/02-tracking-findings.md:

1. **Detection runs at 1 Hz, tracking at 10 Hz.** On this footage the detector
   merges the excavator and the dump truck into one box in about half the frames,
   so it is *less* reliable than the tracker. It seeds and audits; it does not
   drive.

2. **Prompts carry negative points inside the truck's box.** "Not this" makes a
   merged box harmless: mask area drops from 13.8% of the frame (both machines)
   to ~7% (the excavator alone). The truck box is taken from frames where the two
   are cleanly separated, since a parked truck does not move.

3. **One seed, propagated both ways.** SAM 2 can propagate in reverse, so the
   seed does not have to be the first frame -- it is the best-scoring frame that
   is not merged.

Two objects, two sessions
-------------------------
Three of the four phase onsets need the bucket separated from the rest of the
machine, and no detector can box it -- measured across 19 detector x prompt
combinations on two architectures (docs/stages/06-bucket-mask.md). The bucket is
therefore found by geometry (``seeding.py``), which needs the excavator's masks
to exist first. So the stage runs **two sessions, one object each**:

1. the excavator: seeded from the detector's best box at ``choose_seed``'s
   frame, propagated forward and backward over the whole clip;
2. the bucket: seeded from *all* of those masks by ``seeding.choose_seed``, on
   whichever frame that ranking picks, and propagated both ways from there.

One object per session rather than two objects in one, because SAM 2's forward
loop couples objects to a shared frame cursor. An object flagged as having new
inputs is treated as being on its conditioning frame, and its prompt is then
looked up *for the frame being processed*; an object prompted elsewhere finds
``point_inputs=None, mask_inputs=None`` there and stores a memory built from
nothing, silently (``modeling_sam2_video.py``, the ``has_new_inputs`` branch).
Sharing a session would therefore force both prompts onto the same frame -- the
detector's frame, chosen for detection confidence, which on the development
video was sample 290 of 296 and bypassed the bucket ranking entirely.

The price is that the video is encoded twice: roughly 2x the SAM time, with the
detector unchanged at ~5% of the total. It cannot be avoided by resetting one
session instead, because the session caches vision features for
``max_vision_features_cache_size`` frames and that is **1** by default -- there
is no whole-video encoding to keep. See docs/stages/06-bucket-mask.md.

Nothing here is specific to a particular video: every threshold is either a
dimensionless constant from the config or a statistic of the video being
analysed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import masks as mask_io
from . import seeding
from .config import Config
from .detect import Detection, build_detector, iou
from .devices import resolve_device
from .geometry import arm_reach, farthest_point, rotation_centre
from .kinematics import body_core, boom_base
from .logging_setup import get_logger
from .provenance import config_digest, file_digest, run_record, set_seeds
from .seeding import BucketSeed
from .video import Sample, VideoInfo, iter_samples, probe

log = get_logger(__name__)

# SAM 2 object ids. Any integers would do; these are fixed so that a mask read
# back from disk can be attributed without consulting the run that wrote it.
EXCAVATOR_OBJECT_ID = 1
BUCKET_OBJECT_ID = 2

BUCKET_PROMPT_FORMS = ("mask", "points", "box")


@dataclass
class FrameRecord:
    """What is known about one sampled frame.

    Every field after the first two carries a default, because ``load_result``
    rebuilds these straight from JSON and several completed runs predate the
    bucket. A field without a default would make every one of those files
    unreadable.
    """

    frame_index: int
    time_seconds: float
    mask_area_fraction: float = 0.0
    sam_confidence: float = 0.0
    has_mask: bool = False
    # Present only on frames where the detector ran (1 Hz).
    detection_box: list[float] | None = None
    detection_score: float | None = None
    truck_box: list[float] | None = None
    detection_truck_iou: float | None = None
    detection_mask_iou: float | None = None
    # The second tracked object. Absent from runs made before it existed.
    bucket_area_fraction: float = 0.0
    bucket_confidence: float = 0.0
    has_bucket_mask: bool = False


@dataclass
class TrackResult:
    """Everything stage 3 needs, plus the evidence that it can be trusted."""

    video: str
    video_sha256: str
    config_digest: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    rate_hz: float
    seed: dict[str, Any]
    frames: list[FrameRecord]
    qa: dict[str, Any]
    # Where the bucket was pointed at, and in what form. None when no frame
    # yielded a usable band, which is a legitimate outcome: the run then carries
    # excavator masks only, exactly as it did before there was a second object.
    bucket_seed: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["frames"] = [asdict(f) if not isinstance(f, dict) else f for f in self.frames]
        return data


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


@dataclass
class DetectionPass:
    """Detector output at 1 Hz, before any tracking."""

    sample_positions: list[int] = field(default_factory=list)
    excavator: dict[int, Detection] = field(default_factory=dict)
    truck: dict[int, Detection] = field(default_factory=dict)


def detect_anchors(
    samples: list[Sample], detector, config: Config, info: VideoInfo
) -> DetectionPass:
    """Run the detector on a sparse, evenly spaced subset of the samples.

    Sparse because detection is expensive and, on this kind of footage,
    unreliable: its role is to seed the tracker and to provide an opinion that is
    independent of it, not to be believed frame by frame.
    """
    every = max(1, round(config.sampling.rate_hz / config.sampling.anchor_rate_hz))
    result = DetectionPass(sample_positions=list(range(0, len(samples), every)))

    log.info(
        "detecting on %d of %d samples (%.1f Hz)",
        len(result.sample_positions),
        len(samples),
        config.sampling.anchor_rate_hz,
    )
    for position in result.sample_positions:
        image = samples[position].image
        for prompt in config.detection.excavator_prompts:
            found = detector.detect(image, prompt)
            if found.found:
                result.excavator[position] = found
                break  # prompts are ordered by preference; first hit wins
        for prompt in config.detection.truck_prompts:
            found = detector.detect(image, prompt)
            if found.found:
                result.truck[position] = found
                break
    return result


def estimate_truck_box(detections: DetectionPass, config: Config) -> np.ndarray | None:
    """Where the truck is, taken only from frames where it is clearly separate.

    On frames where the detector merges the two machines, the "truck" box is the
    merged box and says nothing. Frames where the two boxes barely overlap are
    the ones that carry information, and a parked truck keeps that position while
    the excavator swings over it.

    Returns None when no truck is ever cleanly detected, which is a legitimate
    outcome: a video with no truck cannot suffer the merge at all.
    """
    separated = []
    for position, truck in detections.truck.items():
        excavator = detections.excavator.get(position)
        if excavator is None:
            continue
        overlap = iou(excavator.best()[0], truck.best()[0])
        if overlap < config.track.max_clean_truck_iou:
            separated.append(truck.best()[0])

    if not separated:
        log.warning("no frame had a cleanly separated truck; prompting without negatives")
        return None

    box = np.median(np.stack(separated), axis=0)
    log.info(
        "truck box from %d separated frame(s): %s",
        len(separated),
        [round(float(v)) for v in box],
    )
    return box


def negative_points(truck_box: np.ndarray | None, config: Config) -> list[list[float]]:
    """Points to mark as "not the object", spread inside the truck's box.

    Biased to the lower part of the box on purpose. The bucket empties *into* the
    bed, so it occupies the upper region; the wheels and chassis are where the
    arm never goes. A negative point landing on the bucket cancels the exclusion
    entirely -- measured, see the findings doc -- so placement is not incidental.
    """
    if truck_box is None:
        return []
    x1, y1, x2, y2 = truck_box
    width, height = x2 - x1, y2 - y1
    xs = np.linspace(
        x1 + width * config.track.negative_x_margin,
        x2 - width * config.track.negative_x_margin,
        config.track.negative_columns,
    )
    ys = np.linspace(
        y1 + height * config.track.negative_y_low,
        y1 + height * config.track.negative_y_high,
        config.track.negative_rows,
    )
    return [[float(x), float(y)] for y in ys for x in xs]


def choose_seed(detections: DetectionPass, config: Config) -> tuple[int, np.ndarray]:
    """Pick the frame to prompt on.

    Ranked by detection confidence among frames whose box is *not* merged with
    the truck. Negative points make even a merged frame usable, but an unmerged
    one needs no such help, so prefer it when available.

    Raises if the detector never found the machine -- that is a stage-1 failure
    and must be loud rather than papered over.
    """
    if not detections.excavator:
        raise RuntimeError(
            "the detector never found the excavator; stage 1 must pass before tracking"
        )

    def merged(position: int) -> bool:
        truck = detections.truck.get(position)
        if truck is None:
            return False
        return (
            iou(detections.excavator[position].best()[0], truck.best()[0])
            >= config.track.max_clean_truck_iou
        )

    clean = [p for p in detections.excavator if not merged(p)]
    candidates = clean or list(detections.excavator)
    if not clean:
        log.warning(
            "every detected frame is merged with the truck; "
            "seeding on a merged frame and relying on negative points"
        )

    best = max(candidates, key=lambda p: detections.excavator[p].best()[1])
    box, score = detections.excavator[best].best()
    log.info(
        "seeding at sample %d (score %.2f, %s)",
        best,
        score,
        "clean" if best in clean else "merged",
    )
    return best, box


# --------------------------------------------------------------------------
# The second object: pointing SAM 2 at the bucket
# --------------------------------------------------------------------------


def derive_bucket_seed(
    excavator_masks: dict[int, np.ndarray],
    sample_count: int,
    truck_box: np.ndarray | None,
    config: Config,
) -> BucketSeed | None:
    """Turn the excavator's *whole* mask set into a prompt for the bucket.

    The geodesic rule in ``seeding.py`` needs three things the masks themselves
    supply, and nothing else:

    * the **body core** -- which pixels are machine in almost every frame. It is
      subtracted before the arm is traced, and that subtraction is the reason
      the far end of the geodesic is the bucket rather than the undercarriage.
      It needs several frames of arm motion to exist at all: the persistent core
      of one mask is that whole mask, which would leave the body in and let the
      tracks compete with the bucket for "farthest along the metal";
    * the **pivot** -- the top of that core, where the boom is anchored, which is
      where the geodesic wavefront starts;
    * the **scale** -- how far the arm reaches in this video's pixels, used only
      to compare candidate frames.

    All three are computed from every mask the excavator's session produced, and
    every one of those masks is then a candidate frame. That is the whole reason
    the bucket gets a session of its own: sharing one would pin it to the
    detector's frame, the core would have to come from a short window around it,
    and ``seeding.choose_seed``'s ranking -- reach, compactness, clearance from
    the truck, mid-clip-ness -- would never get to choose anything.

    ``excavator_masks`` is keyed by **sample ordinal**, so the candidate list is
    rebuilt at full length with ``None`` in the gaps. ``choose_seed`` skips
    those, which keeps ``BucketSeed.sample`` a true sample ordinal rather than
    an index into whichever frames happened to be masked.

    Returns ``None`` when no band can be derived; the caller then tracks the
    excavator alone rather than prompting SAM 2 with a guess.
    """
    usable = [mask for mask in excavator_masks.values() if mask.any()]
    if not usable:
        log.warning("the excavator pass produced no usable mask; cannot seed the bucket")
        return None

    core = body_core(usable, config.geometry.occupancy_quantile)
    if not core.any():
        log.warning("no persistent body core in the excavator masks; cannot place a pivot")
        return None
    if len(usable) == 1:
        log.warning(
            "body core taken from a single mask: the persistent body of one "
            "mask is that whole mask, so the band will be empty or will include "
            "the undercarriage"
        )

    pivot = boom_base(core)
    centre = rotation_centre(usable, config.geometry.occupancy_quantile)
    tips = [farthest_point(mask, centre) for mask in usable]
    scale = arm_reach(tips, centre, config.geometry.reach_percentile)

    candidates: list[np.ndarray | None] = [None] * sample_count
    for position, mask in excavator_masks.items():
        if 0 <= position < sample_count:
            candidates[position] = mask

    return seeding.choose_seed(
        candidates,
        core,
        pivot,
        scale,
        truck_box=None if truck_box is None else tuple(float(v) for v in truck_box),
        point_count=config.track.bucket_point_count,
        negative_count=config.track.bucket_negative_count,
    )


def bucket_prompt_payload(seed: BucketSeed, form: str) -> dict[str, Any]:
    """The prompt keyword arguments SAM 2's processor expects, in one of three forms.

    The seed carries all three because the model's own ablation prices them
    apart -- a box is worth 72.9 J&F, five clicks 75.4, a mask 77.6 -- and the
    difference between the cheapest and the dearest is larger than most of the
    tuning available anywhere else in this stage.

    Nesting is ``[batch][object][...]``, which is why every value here looks
    over-bracketed: one batch, one object, then the prompt itself.
    """
    if form == "mask":
        # The band IS a mask, already at the video's resolution, so the
        # strongest prompt form costs nothing extra to produce.
        return {"input_masks": [seed.band.astype(bool)]}
    if form == "box":
        return {"input_boxes": [[[float(v) for v in seed.box]]]}
    if form == "points":
        points = [[float(x), float(y)] for x, y in seed.points]
        points += [[float(x), float(y)] for x, y in seed.negatives]
        labels = [1] * len(seed.points) + [0] * len(seed.negatives)
        return {"input_points": [[points]], "input_labels": [[labels]]}
    raise ValueError(
        f"unknown bucket prompt form {form!r}; expected one of {BUCKET_PROMPT_FORMS}"
    )


def register_prompt(
    processor, session, frame_idx: int, obj_id: int, prompt: dict[str, Any]
) -> None:
    """Attach one object's prompt to one frame of one session.

    Deliberately singular. When two objects shared a session this had to restore
    the union of ``session.obj_with_new_inputs`` afterwards, because
    ``add_inputs_to_inference_session`` *assigns* that list rather than appending
    to it, so registering the second object cleared the first's pending flag and
    the first was then tracked from a memory bank it never built. With one object
    per session there is no second registration and nothing to clobber -- the
    workaround is gone, and the reason it existed is recorded in
    docs/stages/06-bucket-mask.md so it is not rediscovered the hard way.
    """
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=frame_idx, obj_ids=obj_id, **prompt
    )


def track_object(
    model,
    processor,
    session,
    obj_id: int,
    frame_idx: int,
    prompt: dict[str, Any],
    unpack,
    keep_empty: bool = True,
    label: str = "object",
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Prompt a session with a single object and propagate it over the whole clip.

    One seed, both directions. SAM 2 propagates in reverse as well as forward, so
    the conditioning frame does not have to be the first one -- it can be the
    best one, which is the entire reason the bucket gets its own session.

    ``unpack`` turns one model output into ``{object id: (mask, confidence)}``;
    it is injected rather than built here so that the post-processing work
    (which needs the processor and the frame size) stays in one place and so
    that this loop can be exercised without a model. The session is expected to
    report exactly ``obj_id`` and nothing else: anything else means the caller
    has reused a session that still remembers a previous object, which is the
    failure this two-session arrangement exists to make impossible.

    ``keep_empty`` distinguishes the two objects. The excavator's masks are kept
    whatever they contain, because a sample with no machine in it is still a
    sample the QA gate must see. The bucket's are not: SAM returns a prediction
    on every frame whether or not it still believes the object is there, so
    storing the empty ones would make bucket coverage 100% by construction and
    say nothing. Confidences are recorded for every frame either way.
    """
    register_prompt(processor, session, frame_idx, obj_id, prompt)

    masks: dict[int, np.ndarray] = {}
    confidences: dict[int, float] = {}

    def store(output) -> None:
        found = unpack(output)
        if sorted(found) != [obj_id]:
            raise RuntimeError(
                f"the session reported objects {sorted(found)} while tracking "
                f"{obj_id} alone; each session must carry exactly one object"
            )
        mask, confidence = found[obj_id]
        confidences[output.frame_idx] = confidence
        if keep_empty or mask.any():
            masks[output.frame_idx] = mask

    store(model(inference_session=session, frame_idx=frame_idx))
    log.info("%s: propagating forward from sample %d", label, frame_idx)
    for output in model.propagate_in_video_iterator(
        inference_session=session, start_frame_idx=frame_idx
    ):
        store(output)
    if frame_idx > 0:
        log.info("%s: propagating backward from sample %d", label, frame_idx)
        for output in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=frame_idx, reverse=True
        ):
            store(output)

    log.info("%s: %d mask(s) over %d sample(s)", label, len(masks), len(confidences))
    return masks, confidences


def split_objects(
    object_ids,
    processed_masks,
    object_score_logits,
    min_fractions: dict[int, float],
) -> dict[int, tuple[np.ndarray, float]]:
    """Route one multi-object SAM 2 output into a mask and a confidence per object.

    The rows of ``processed_masks`` are in the order of ``object_ids``, not in
    the order of the ids themselves. With a single object the two coincide,
    which is why reading row 0 and calling it "the excavator" worked right up
    until a second object was added -- at which point the bucket would have been
    dropped without a word. Indexing by the reported ids removes the coincidence.

    Each session now carries one object, so this routing has one row to route.
    It is kept, and kept tested, because ``processed[0, 0]`` is exactly the kind
    of indexing that looks correct forever and then is not: anyone who merges
    the two sessions back into one lands on the bug again, and this is the guard
    that stops them.

    Each object gets its own speck threshold: the same relative cut means very
    different absolute sizes on a 9,000-pixel machine and a 1,100-pixel bucket.
    """
    masks = _to_numpy(processed_masks)
    logits = (
        None if object_score_logits is None else _to_numpy(object_score_logits).reshape(-1)
    )

    out: dict[int, tuple[np.ndarray, float]] = {}
    for row, obj_id in enumerate(object_ids):
        obj_id = int(obj_id)
        mask = np.asarray(masks[row, 0]) > 0
        cleaned = drop_small_components(mask, min_fractions.get(obj_id, 0.0))
        # `object_score_logits` is how sure the model is that the object is
        # present at all -- not mask quality. It drops under occlusion, which is
        # exactly when a sample should count as MISSING rather than as evidence.
        confidence = float("nan") if logits is None else _sigmoid(float(logits[row]))
        out[obj_id] = (cleaned, confidence)
    return out


def _to_numpy(value) -> np.ndarray:
    """Torch tensor or array -> array, without importing torch.

    Torch is an optional dependency here (the `models` extra), so the parts of
    this module that only shuffle numbers must not reach for it.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sigmoid(x: float) -> float:
    """Logit -> probability, written so a large negative logit cannot overflow."""
    return float(0.5 * (1.0 + np.tanh(0.5 * x)))


# --------------------------------------------------------------------------
# The stage itself
# --------------------------------------------------------------------------


def track(
    video_path: str | Path,
    config: Config,
    output_dir: str | Path,
    device: str | None = None,
    detector_name: str = "grounding_dino",
) -> TrackResult:
    """Produce a mask per sample, and the evidence that they are trustworthy.

    Writes ``masks.npz``, ``track.json`` and ``run.json`` into ``output_dir``.
    Later stages read those and never touch a model again.
    """
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    set_seeds()
    # Checked before anything expensive happens: a typo here would otherwise
    # surface after the detector pass and a model download.
    if config.track.bucket_prompt not in BUCKET_PROMPT_FORMS:
        raise ValueError(
            f"track.bucket_prompt is {config.track.bucket_prompt!r}; "
            f"expected one of {BUCKET_PROMPT_FORMS}"
        )
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    info = probe(video_path, verify=True)

    with run_record("track", output_dir, config.to_dict()) as record:
        record.add_input("video", video_path)

        samples = list(
            iter_samples(
                video_path,
                rate_hz=config.sampling.rate_hz,
                max_edge=config.sampling.inference_max_edge,
            )
        )
        log.info(
            "%d samples at %.1f Hz from %.1f s of video",
            len(samples),
            config.sampling.rate_hz,
            info.duration_seconds,
        )

        detector = build_detector(detector_name, config.detection, device=device)
        detections = detect_anchors(samples, detector, config, info)
        truck_box = estimate_truck_box(detections, config)
        negatives = negative_points(truck_box, config)
        seed_position, seed_box = choose_seed(detections, config)

        resolved_device = resolve_device(device)
        log.info("loading %s on %s", config.track.model_id, resolved_device)
        processor = Sam2VideoProcessor.from_pretrained(config.track.model_id)
        model = (
            Sam2VideoModel.from_pretrained(config.track.model_id).to(resolved_device).eval()
        )

        frames_rgb = [cv2.cvtColor(s.image, cv2.COLOR_BGR2RGB) for s in samples]
        height, width = samples[0].image.shape[:2]

        def new_session():
            # Frames stay on the CPU; only the model's working tensors go to the
            # accelerator. Pushing the whole decoded video onto the device costs
            # memory the machine may not have, and a swapping run is
            # indistinguishable from a hung one until you check CPU time.
            return processor.init_video_session(
                video=frames_rgb,
                inference_device=resolved_device,
                video_storage_device="cpu",
                dtype=torch.float32,
            )

        excavator_prompt: dict[str, Any] = {"input_boxes": [[seed_box.tolist()]]}
        if negatives:
            excavator_prompt["input_points"] = [[negatives]]
            excavator_prompt["input_labels"] = [[[0] * len(negatives)]]

        min_fractions = {
            EXCAVATOR_OBJECT_ID: config.track.min_component_fraction,
            BUCKET_OBJECT_ID: config.track.bucket_min_component_fraction,
        }

        def unpack(output) -> dict[int, tuple[np.ndarray, float]]:
            # Masks are left overlapping on purpose: the bucket is *part of* the
            # excavator, so the non-overlapping constraint SAM 2 can apply would
            # carve one out of the other.
            processed = processor.post_process_masks(
                [output.pred_masks], [[height, width]], binarize=True
            )[0]
            return split_objects(
                output.object_ids,
                processed,
                getattr(output, "object_score_logits", None),
                min_fractions,
            )

        # --- session A: the excavator, alone ---------------------------------
        log.info("tracking the excavator from sample %d of %d", seed_position, len(samples))
        excavator_session = new_session()
        with torch.inference_mode():
            masks, confidences = track_object(
                model,
                processor,
                excavator_session,
                EXCAVATOR_OBJECT_ID,
                seed_position,
                excavator_prompt,
                unpack,
                keep_empty=True,
                label="excavator",
            )
        # A session remembers its object in its memory bank, so the bucket gets
        # a new one rather than this one reset: `reset_inference_session` clears
        # the vision-feature cache anyway, and that cache holds one frame, so
        # there is nothing to save by reusing it.
        del excavator_session

        # --- session B: the bucket, alone, on its own best frame --------------
        bucket_masks: dict[int, np.ndarray] = {}
        bucket_confidences: dict[int, float] = {}
        bucket_seed = derive_bucket_seed(masks, len(samples), truck_box, config)

        if bucket_seed is None:
            log.warning("no bucket seed; the run carries excavator masks only")
        else:
            log.info(
                "tracking the bucket as object %d, prompted by %s at sample %d "
                "(the excavator was seeded at %d)",
                BUCKET_OBJECT_ID,
                config.track.bucket_prompt,
                bucket_seed.sample,
                seed_position,
            )
            bucket_session = new_session()
            with torch.inference_mode():
                bucket_masks, bucket_confidences = track_object(
                    model,
                    processor,
                    bucket_session,
                    BUCKET_OBJECT_ID,
                    bucket_seed.sample,
                    bucket_prompt_payload(bucket_seed, config.track.bucket_prompt),
                    unpack,
                    keep_empty=False,
                    label="bucket",
                )
            del bucket_session

        records = _build_records(
            samples,
            masks,
            confidences,
            bucket_masks,
            bucket_confidences,
            detections,
            truck_box,
            height,
            width,
        )
        qa = evaluate_quality(records, config)
        qa["bucket"] = evaluate_bucket_quality(masks, bucket_masks, len(samples), config)

        result = TrackResult(
            video=str(video_path),
            video_sha256=file_digest(video_path),
            config_digest=config_digest(config.to_dict()),
            width=width,
            height=height,
            fps=info.fps,
            duration_seconds=info.duration_seconds,
            rate_hz=config.sampling.rate_hz,
            seed={
                "sample_position": seed_position,
                "frame_index": samples[seed_position].frame_index,
                "time_seconds": samples[seed_position].time_seconds,
                "box": [float(v) for v in seed_box],
                "negative_points": negatives,
                "truck_box": None if truck_box is None else [float(v) for v in truck_box],
            },
            frames=records,
            qa=qa,
            bucket_seed=(
                None if bucket_seed is None else _seed_record(bucket_seed, samples, config)
            ),
        )

        mask_io.save_objects(
            output_dir / "masks.npz",
            {"excavator": masks, "bucket": bucket_masks},
            (height, width),
        )
        (output_dir / "track.json").write_text(json.dumps(result.to_dict(), indent=2))
        record.outputs.extend(["masks.npz", "track.json"])
        record.metrics = {
            "qa_status": qa["status"],
            "masked_samples": len(masks),
            "bucket_masked_samples": len(bucket_masks),
        }

    log.info(
        "tracking complete: %d masks, %d bucket masks, QA %s",
        len(masks),
        len(bucket_masks),
        qa["status"],
    )
    return result


def _seed_record(seed: BucketSeed, samples: list[Sample], config: Config) -> dict[str, Any]:
    """The bucket seed as plain JSON, so a run can be audited without rerunning it."""
    sample = samples[seed.sample]
    return {
        "sample_position": seed.sample,
        "frame_index": sample.frame_index,
        "time_seconds": sample.time_seconds,
        "prompt": config.track.bucket_prompt,
        "points": [[float(x), float(y)] for x, y in seed.points],
        "negative_points": [[float(x), float(y)] for x, y in seed.negatives],
        "box": [float(v) for v in seed.box],
        "band_pixels": int(seed.band.sum()),
        "score": float(seed.score),
        "reach": float(seed.reach),
    }


def drop_small_components(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    """Remove stray specks, keeping any component of a meaningful size.

    SAM sometimes leaves a few dozen pixels on the soil or in a shadow. Those
    pixels are harmless to the area statistics and fatal to the geometry: the
    bucket tip is defined as the point of the mask farthest from the machine's
    centre, so one speck on the far side of the frame relocates the bucket.

    Small components are dropped rather than keeping only the largest, because
    an occlusion can legitimately split the arm from the body.
    """
    if not mask.any() or min_fraction <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2:  # background plus at most one component
        return mask
    total = float(mask.sum())
    keep = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] / total >= min_fraction:
            keep |= labels == index
    return keep if keep.any() else mask


def _build_records(
    samples,
    masks,
    confidences,
    bucket_masks,
    bucket_confidences,
    detections,
    truck_box,
    height,
    width,
):
    """Per-sample facts, including the detector's independent opinion where it ran.

    Both mask dictionaries are keyed by **sample ordinal**, which is what
    ``position`` is here -- never by the source frame index, which is a
    different number stored alongside it.
    """
    frame_area = float(height * width)
    records: list[FrameRecord] = []
    for position, sample in enumerate(samples):
        mask = masks.get(position)
        bucket = bucket_masks.get(position)
        record = FrameRecord(
            frame_index=sample.frame_index,
            time_seconds=sample.time_seconds,
            has_mask=mask is not None,
            mask_area_fraction=float(mask.sum() / frame_area) if mask is not None else 0.0,
            sam_confidence=confidences.get(position, 0.0),
            has_bucket_mask=bucket is not None,
            bucket_area_fraction=(
                float(bucket.sum() / frame_area) if bucket is not None else 0.0
            ),
            bucket_confidence=bucket_confidences.get(position, 0.0),
        )
        detection = detections.excavator.get(position)
        if detection is not None:
            box, score = detection.best()
            record.detection_box = [float(v) for v in box]
            record.detection_score = float(score)
            truck = detections.truck.get(position)
            if truck is not None:
                record.detection_truck_iou = float(iou(box, truck.best()[0]))
            if mask is not None and mask.any():
                record.detection_mask_iou = float(iou(box, _mask_bbox(mask)))
        if truck_box is not None:
            record.truck_box = [float(v) for v in truck_box]
        records.append(record)
    return records


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def evaluate_quality(records: list[FrameRecord], config: Config) -> dict[str, Any]:
    """Judge the masks without ground truth.

    Every threshold is relative to this video's own statistics. A mask that
    suddenly covers twice the usual area has probably absorbed the truck; one
    that collapses has lost the machine. Neither needs an absolute size to detect.
    """
    areas = np.array([r.mask_area_fraction for r in records if r.has_mask])
    coverage = float(len(areas) / len(records)) if records else 0.0
    failures: list[str] = []

    if coverage < config.qa.min_excavator_coverage:
        failures.append(f"only {coverage:.1%} of samples have a mask")

    median_area = float(np.median(areas)) if areas.size else 0.0
    swollen = shrunken = 0
    if median_area > 0:
        ratios = areas / median_area
        swollen = int((ratios > config.track.max_area_ratio).sum())
        shrunken = int((ratios < config.track.min_area_ratio).sum())
        if swollen / max(areas.size, 1) > config.track.max_bad_area_fraction:
            failures.append(
                f"{swollen} samples have a mask far above the median area "
                "(the tracker may have absorbed another object)"
            )
        if shrunken / max(areas.size, 1) > config.track.max_bad_area_fraction:
            failures.append(
                f"{shrunken} samples have a mask far below the median area "
                "(the tracker may have lost the machine)"
            )

    agreements = np.array(
        [
            r.detection_mask_iou
            for r in records
            if r.detection_mask_iou is not None
            and (r.detection_truck_iou or 0.0) < config.track.max_clean_truck_iou
        ]
    )
    median_agreement = float(np.median(agreements)) if agreements.size else float("nan")
    if agreements.size and median_agreement < config.qa.min_anchor_agreement:
        failures.append(
            f"mask disagrees with independent detections "
            f"(median IoU {median_agreement:.2f}) on unmerged frames"
        )

    confidences = np.array([r.sam_confidence for r in records if r.has_mask])
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "coverage": round(coverage, 4),
        "mask_area": {
            "median": round(median_area, 5),
            "min": round(float(areas.min()), 5) if areas.size else None,
            "max": round(float(areas.max()), 5) if areas.size else None,
            "samples_far_above_median": swollen,
            "samples_far_below_median": shrunken,
        },
        "detection_agreement_median_iou": None
        if np.isnan(median_agreement)
        else round(median_agreement, 4),
        "sam_confidence_median": round(float(np.median(confidences)), 4)
        if confidences.size
        else None,
    }


def evaluate_bucket_quality(
    masks: dict[int, np.ndarray],
    bucket_masks: dict[int, np.ndarray],
    sample_count: int,
    config: Config,
) -> dict[str, Any]:
    """Judge the bucket masks on their own terms, with no ground truth.

    Three numbers, none of which needs a label and none of which is a pixel
    count:

    * **coverage** -- the share of samples that got a bucket mask at all. The
      open question this whole stage turns on is whether SAM 2.1-tiny can hold a
      ~25 px object for hundreds of samples, and this is where losing it shows.
    * **containment** -- the median share of bucket pixels that lie inside the
      excavator mask. The bucket *is* part of the machine, so a bucket that has
      slid onto the truck bed or the spoil pile is visible here and nowhere
      else: its area would still look reasonable and its confidence high.
    * **area stability** -- p90 over p10 of the bucket's area across the clip. A
      tracker letting the bucket creep up the arm swells; one losing it
      collapses. A ratio rather than a level, so it carries nothing about this
      video's scale. Some variation is real -- the bucket foreshortens as the
      machine slews -- so this flags, it does not condemn.

    These are reported, **not** folded into the excavator's pass/fail. The
    excavator gate is calibrated against measurements; nothing about the bucket
    has been measured yet, and a gate invented before its first measurement
    would only be a guess with a threshold attached.
    """
    coverage = float(len(bucket_masks) / sample_count) if sample_count else 0.0

    containments = [
        float(np.logical_and(bucket, masks[position]).sum() / bucket.sum())
        for position, bucket in bucket_masks.items()
        if position in masks and bucket.any()
    ]
    containment = float(np.median(containments)) if containments else None

    areas = np.array([float(mask.sum()) for mask in bucket_masks.values()])
    spread = None
    if areas.size:
        low = float(np.percentile(areas, 10))
        spread = float(np.percentile(areas, 90) / low) if low > 0 else float("inf")

    notes: list[str] = []
    if coverage < config.qa.min_bucket_coverage:
        notes.append(f"only {coverage:.1%} of samples have a bucket mask")
    if containment is not None and containment < config.qa.min_bucket_containment:
        notes.append(
            f"median {1 - containment:.1%} of the bucket mask lies outside the machine "
            "(the bucket may have drifted onto another object)"
        )
    if spread is not None and spread > config.qa.max_bucket_area_ratio:
        notes.append(
            f"bucket area varies by {spread:.1f}x across the clip "
            "(p90/p10), which is more than foreshortening explains"
        )

    return {
        "coverage": round(coverage, 4),
        "samples_with_mask": len(bucket_masks),
        "containment_median": None if containment is None else round(containment, 4),
        "area_p90_over_p10": None if spread is None else round(spread, 3),
        "area_median_pixels": round(float(np.median(areas)), 1) if areas.size else None,
        "notes": notes,
    }


def load_result(output_dir: str | Path) -> tuple[TrackResult, dict[int, np.ndarray]]:
    """Read a completed tracking run back from disk.

    This is the boundary that lets everything after perception run on a laptop:
    the rendering and analysis stages call this, never a model.
    """
    output_dir = Path(output_dir)
    data = json.loads((output_dir / "track.json").read_text())
    data["frames"] = [FrameRecord(**f) for f in data["frames"]]
    result = TrackResult(**data)
    loaded, _ = mask_io.load(output_dir / "masks.npz")
    return result, loaded


def load_bucket_masks(output_dir: str | Path) -> dict[int, np.ndarray]:
    """The bucket masks from a completed run, keyed by sample ordinal.

    Separate from ``load_result`` rather than bolted onto its return value,
    because every stage downstream already unpacks that two-tuple and only the
    stages that need the bucket should pay to decode it. Returns an empty
    mapping for a run made before the bucket was tracked.
    """
    objects, _ = mask_io.load_objects(Path(output_dir) / "masks.npz")
    return objects.get("bucket", {})
