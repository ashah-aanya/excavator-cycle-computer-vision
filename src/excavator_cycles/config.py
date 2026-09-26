"""Every tunable number in the pipeline, in one place.

Why this file exists
--------------------
The task forbids "video-specific values" in the pipeline. The way we honour that
is a discipline, not a promise:

1. **Nothing here is measured in pixels or frames.** Distances are expressed as
   fractions of the machine's own reach (``L``), times in seconds. A hidden video
   shot from further away, or at 25 fps instead of 30, then needs no special case.

2. **Thresholds on signals are multipliers, not levels.** A rule never says "curl
   rate above 0.31 rad/s". It says "curl rate above ``alpha`` times the 95th
   percentile of curl rate *in this video*". The multiplier is fixed across all
   videos; the level adapts to each one. That is the line between globally tuned
   (allowed) and fitted to one video (banned).

3. **A value of ``None`` means "derive it from the data".** It is not a missing
   setting -- it is a deliberate statement that no constant belongs there.

Override any of these from a YAML file (see ``configs/default.yaml``) without
touching code::

    cfg = Config.load("configs/default.yaml")
    cfg = Config.load(overrides={"sampling": {"rate_hz": 15.0}})
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class SamplingConfig:
    """How often each stage looks at the video.

    These are deliberately decoupled. Decoding is cheap, detection is expensive,
    and the state machine needs neither every frame nor the source frame rate.
    """

    # Rate at which masks, features and the state machine are computed.
    # 10 Hz is chosen from the *fastest event* we must resolve -- the bucket's
    # uncurl at the start of dumping, a few tenths of a second -- not from the
    # shortest phase. Boundary precision does not depend on this rate, because
    # pass 2 interpolates crossings between samples.
    rate_hz: float = 10.0

    # Rate at which the detector re-runs to re-anchor the tracker. Also doubles
    # as an independent second opinion for the perception QA gate.
    anchor_rate_hz: float = 1.0

    # Longest edge the frame is resized to before model inference. None = native.
    # Smaller is faster; too small and the bucket becomes a few pixels.
    inference_max_edge: int | None = 1024


@dataclass(frozen=True)
class DetectionConfig:
    """What we ask the detector for, and how confident it must be.

    Note what is *absent*: no prompt for "bucket", "arm" or "dirt pile". Parts and
    amorphous terrain are where open-vocabulary detectors fail; all of those are
    derived geometrically instead (see ``geometry.py``).
    """

    # Text prompts, one detection pass per group (scores are not comparable
    # across phrases, so they must not be mixed in a single prompt string).
    #
    # "digger." was measured against "excavator." on real frames from this
    # footage (docs/stages/01-perception-findings.md): it scored lower (0.64 vs
    # 0.75 median) and also fired on the dump truck at 0.35-0.45. Always below
    # its score on the excavator, so the best box was still correct -- but it is
    # avoidable risk on unseen footage. Kept here as a documented fallback.
    excavator_prompts: tuple[str, ...] = ("excavator.",)
    truck_prompts: tuple[str, ...] = ("dump truck.", "truck.")

    # Hugging Face model id. Pinning `revision` makes a run reproducible even if
    # the repo is updated upstream; None = current main (fill in once measured).
    model_id: str = "IDEA-Research/grounding-dino-base"
    revision: str | None = None

    # Detection confidence floors. Scores from open-vocabulary detectors are not
    # calibrated across prompts, so these are starting points to be checked in
    # the spike, not trusted constants.
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    # Alternative detector evaluated alongside the default in the spike.
    alt_model_id: str = "google/owlv2-base-patch16-ensemble"



@dataclass(frozen=True)
class TrackConfig:
    """Stage 2: turning boxes into masks that follow the machine.

    The constants here encode findings measured on real footage
    (docs/stages/02-tracking-findings.md), not guesses.
    """

    # SAM 2 checkpoint. Ungated and Apache-2.0, so a reviewer can reproduce the
    # run without requesting access to anything.
    model_id: str = "facebook/sam2.1-hiera-tiny"

    # Above this overlap between the excavator box and the truck box, the
    # detector has merged the two machines into one region rather than finding
    # the excavator. Such frames are unusable as a seed and their "truck" box
    # carries no information.
    max_clean_truck_iou: float = 0.30

    # Negative points ("not this") are spread inside the truck's box. The y band
    # is deliberately low: the bucket empties INTO the bed, so it occupies the
    # upper region, and a negative landing on the bucket cancels the exclusion
    # entirely -- measured, mask area went from 7.1% back to 13.8%.
    negative_rows: int = 2
    negative_columns: int = 3
    negative_y_low: float = 0.55  # fraction of the truck box height
    negative_y_high: float = 0.90
    negative_x_margin: float = 0.15

    # Connected components smaller than this fraction of the mask are dropped.
    # SAM occasionally leaves a speck on the soil or a shadow; a few dozen stray
    # pixels would move the "farthest point from the rotation centre" -- the
    # bucket tip -- somewhere the machine is not. Kept as a fraction rather than
    # a pixel count so it holds at any resolution. Not 0, because the arm can be
    # legitimately split by an occlusion.
    min_component_fraction: float = 0.05

    # Mask-area sanity, relative to this video's own median. A mask that doubles
    # has probably absorbed another object; one that collapses has lost the
    # machine. Neither needs an absolute size.
    max_area_ratio: float = 1.8
    min_area_ratio: float = 0.4
    max_bad_area_fraction: float = 0.10  # share of samples allowed to be off

    # --- the second tracked object: the bucket ----------------------------
    # Which of the three prompt forms the bucket seed is handed to SAM 2 as.
    # The model's own ablation (paper Table 4, zero-shot VOS over 17 datasets)
    # prices them 64.3 J&F for one click, 72.9 for a box, 75.4 for five clicks
    # and 77.6 for a mask -- and the seed already computes the mask, so the
    # strongest form is also the free one. "points" and "box" are kept because
    # the same ablation says a box is the robust choice when the mask itself is
    # suspect, and because a prompt form is the first thing worth sweeping if
    # the bucket cannot be held.
    bucket_prompt: str = "mask"

    # The speck cut is a fraction of the MASK's own area, not the frame's, so it
    # is scale-free in principle. In practice it is not: SAM's strays are a few
    # dozen pixels whatever the object's size, and the bucket's mask is an order
    # of magnitude smaller than the machine's (~1,100 px against ~9,000 on the
    # development footage). At 0.05 the cut would erase a genuinely detached
    # 50-pixel fragment of bucket while leaving a 50-pixel speck untouched, so
    # the bucket gets a much smaller fraction of its own.
    bucket_min_component_fraction: float = 0.01

    # Prompt sizes when `bucket_prompt` is "points": five positives is the knee
    # of the ablation curve above, and the negatives sit on the stick to stop
    # SAM claiming the whole arm as "the bucket".
    bucket_point_count: int = 5
    bucket_negative_count: int = 3


@dataclass(frozen=True)
class GeometryConfig:
    """Constants for deriving scene structure from the excavator mask.

    All of these are *dimensionless ratios* -- fractions of the machine's reach,
    or quantiles of a distribution -- so none of them encodes anything about a
    particular video.
    """

    # Pixels belonging to the excavator in at least this fraction of frames are
    # treated as its persistent core (body + undercarriage). The arm sweeps, so
    # it is only intermittently present and drops out of the core.
    occupancy_quantile: float = 0.90

    # `L`, the scale unit, is this percentile of the centre-to-bucket distance.
    # A percentile rather than the max, so one bad frame cannot set the scale.
    reach_percentile: float = 95.0

    # The bucket region is the outermost part of the arm, as a fraction of the
    # machine's CURRENT radial reach -- not a disk around the tip. A circular
    # crop makes any region circular: measured that way the bucket's elongation
    # is 1.25 (apparently shapeless), measured as a radial band it is 2.13.
    bucket_band_low: float = 0.85

    # Size of the optical-flow patch below the bucket, in units of L.
    bucket_radius_frac: float = 0.22


    # Temporal filter on the bucket's position (design doc section 4.3). All in
    # units of the machine's reach L and in seconds, so one set of values serves
    # any video at any scale or frame rate.
    #
    # process_noise is how much the bucket's velocity may change per second: it
    # sets how much the filter trusts its own motion model against the next
    # measurement. Too low and a real swing gets rejected; too high and the
    # filter believes everything and does nothing.
    tip_process_noise: float = 2.0  # L per second squared
    tip_measurement_noise: float = 0.02  # L, expected error of one pose fit
    tip_gate_sigma: float = 3.0  # reject beyond this many sigmas


    # Samples slower than this quantile of speed count as "dwelling", and are
    # what the dig/dump location clustering is run on.
    dwell_speed_quantile: float = 0.25

    # The two derived locations must be at least this far apart (in units of `L`)
    # to be believed. Closer than this and the clustering is reported as failed
    # rather than silently returning two overlapping blobs.
    min_zone_separation: float = 0.40


@dataclass(frozen=True)
class FeatureConfig:
    """Smoothing and the threshold convention."""

    # Width of the symmetric smoothing window, in seconds. Symmetric (zero-phase)
    # is not a preference: a one-sided filter delays signals by an amount that
    # depends on their shape, which is exactly the systematic bias the +/-0.6 s
    # tolerance cannot absorb.
    smoothing_window_seconds: float = 0.5
    smoothing_polyorder: int = 2

    # Box smoothing (design diagram stage 1.5). Measured on the development
    # video, counting transitions inside the +/-0.6 s tolerance: 0.3 s -> 4/4,
    # 0.5 s -> 4/4, 0.9 s -> 1/4, 1.5 s -> 1/4. The diagram's "[i, i+10]" is a
    # 1.0 s window at 10 Hz, which is in the collapsed region -- the mechanism is
    # right, the length is not.
    box_window_seconds: float = 0.5
    # trailing | centred | leading. The diagram draws a LEADING window. All three
    # give byte-identical durations, because a uniform time shift cancels in every
    # difference and durations are what the task grades. "trailing" is the default
    # only because it is what was used when all four transitions were verified.
    box_alignment: str = "trailing"


    # Samples below this confidence are treated as MISSING, not as evidence
    # against a transition. A gap in perception is not a statement about physics.
    min_sample_confidence: float = 0.35



@dataclass(frozen=True)
class FSMConfig:
    """State machine: how long evidence must persist, and how strong it must be.

    The alphas are the multipliers described in ``FeatureConfig.threshold_percentile``.
    Every one of them is a candidate for the sensitivity sweep: we want a plateau
    (answers stable across a wide range), not a knife-edge optimum.
    """

    # Pass 1: how long a transition's evidence must hold before it is confirmed.
    # The SAME value for all four transitions, on purpose -- any residual
    # confirmation lag is then common-mode and cancels in phase durations.
    hold_seconds: float = 0.40

    # Gaps in perception shorter than this are bridged rather than breaking a run
    # of sustained evidence.
    max_evidence_gap_seconds: float = 0.30

    # Pass 2 searches this far back from the confirmation point for the true
    # onset (the extremum or level crossing).
    max_lookback_seconds: float = 2.0

    # Evidence strength multipliers, per transition. Named for what they gate.
    alpha_contact_speed: float = 0.35  # T1: bucket decelerating into material
    alpha_clearance_margin: float = 0.05  # T2: margin above surface, in units of L
    alpha_uncurl_rate: float = 0.50  # T3: how fast the bucket must open
    alpha_swing_rate: float = 0.40  # T4: how fast the machine must rotate back

    # Minimum phase duration. None means: derive it from this video's own measured
    # phase durations and use it to FLAG a suspicious phase, never to block a
    # transition from firing. A hard lockout delays triggers in one direction
    # only, and can swallow a genuinely short dumping phase.
    min_phase_seconds: float | None = None


@dataclass(frozen=True)
class CycleConfig:
    """What counts as a complete cycle."""

    # A cycle whose duration falls outside this band, relative to the median
    # cycle duration in the same video, is flagged as an outlier.
    duration_band: tuple[float, float] = (0.4, 2.5)

    # Mean per-sample confidence a cycle needs to contribute to the averages.
    min_mean_confidence: float = 0.50


@dataclass(frozen=True)
class QAConfig:
    """Pass/fail bands for the perception quality gate.

    The gate never aborts a run -- the hidden videos must always produce an
    answer -- it records status and reasons.
    """

    min_excavator_coverage: float = 0.97  # fraction of samples with a valid mask
    min_anchor_agreement: float = 0.70  # median IoU, fresh detection vs track

    # The bucket's own bands. These are ADVISORY: they produce notes in the
    # report, not a pass/fail, because whether a ~25 px object can be held for a
    # whole clip is still an open question and a gate calibrated before the
    # first measurement would be a guess dressed as a standard.
    min_bucket_coverage: float = 0.90  # samples that got a bucket mask at all
    min_bucket_containment: float = 0.80  # bucket pixels inside the machine's mask
    max_bucket_area_ratio: float = 4.0  # p90/p10 of bucket area across the clip


@dataclass(frozen=True)
class Config:
    """Top-level configuration. Load once, pass down."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    cycles: CycleConfig = field(default_factory=CycleConfig)
    qa: QAConfig = field(default_factory=QAConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Build a config from defaults, optionally layered with YAML and overrides.

        Later sources win: defaults < YAML file < explicit overrides.
        """
        data: dict[str, Any] = {}
        if path is not None:
            loaded = yaml.safe_load(Path(path).read_text()) or {}
            data = _deep_merge(data, loaded)
        if overrides:
            data = _deep_merge(data, overrides)
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for writing into run metadata so any result is traceable."""
        return dataclasses.asdict(self)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dicts, with `extra` winning."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Construct a (possibly nested) dataclass from a plain dict.

    Unknown keys raise rather than being ignored: a typo in a config file should
    be loud, not silently leave the default in place.
    """
    # `from __future__ import annotations` makes field types strings, so resolve
    # them before asking whether a field is itself a dataclass.
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        field_type = hints[name]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(field_type, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)  # YAML lists -> tuples, so config stays frozen
        else:
            kwargs[name] = value

    return cls(**kwargs)
