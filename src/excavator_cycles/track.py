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
from .config import Config
from .detect import Detection, build_detector, iou
from .devices import resolve_device
from .logging_setup import get_logger
from .provenance import config_digest, file_digest, run_record, set_seeds
from .video import Sample, VideoInfo, iter_samples, probe

log = get_logger(__name__)


@dataclass
class FrameRecord:
    """What is known about one sampled frame."""

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
        # Frames stay on the CPU; only the model's working tensors go to the
        # accelerator. Pushing the whole decoded video onto the device costs
        # memory the machine may not have, and a swapping run is indistinguishable
        # from a hung one until you check CPU time.
        session = processor.init_video_session(
            video=frames_rgb,
            inference_device=resolved_device,
            video_storage_device="cpu",
            dtype=torch.float32,
        )

        prompt: dict[str, Any] = {"input_boxes": [[seed_box.tolist()]]}
        if negatives:
            prompt["input_points"] = [[negatives]]
            prompt["input_labels"] = [[[0] * len(negatives)]]
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=seed_position, obj_ids=1, **prompt
        )

        height, width = samples[0].image.shape[:2]
        masks: dict[int, np.ndarray] = {}
        confidences: dict[int, float] = {}

        progress_every = max(1, len(samples) // 10)

        def store(output) -> None:
            processed = processor.post_process_masks(
                [output.pred_masks], [[height, width]], binarize=True
            )[0]
            mask = processed[0, 0].cpu().numpy() > 0
            masks[output.frame_idx] = drop_small_components(
                mask, config.track.min_component_fraction
            )
            if len(masks) % progress_every == 0:
                log.info("  %d/%d masks", len(masks), len(samples))
            # The video model reports `object_score_logits` -- how confident it is
            # that the tracked object is present at all -- rather than the image
            # model's mask-quality score. That is the more useful signal here:
            # it drops when the machine is occluded, which is exactly when a
            # sample should count as MISSING rather than as evidence.
            logits = getattr(output, "object_score_logits", None)
            confidences[output.frame_idx] = (
                float(torch.sigmoid(logits.flatten()[0]))
                if logits is not None
                else float("nan")
            )

        with torch.inference_mode():
            store(model(inference_session=session, frame_idx=seed_position))
            log.info("propagating forward from sample %d", seed_position)
            for output in model.propagate_in_video_iterator(
                inference_session=session, start_frame_idx=seed_position
            ):
                store(output)
            if seed_position > 0:
                log.info("propagating backward from sample %d", seed_position)
                for output in model.propagate_in_video_iterator(
                    inference_session=session, start_frame_idx=seed_position, reverse=True
                ):
                    store(output)

        records = _build_records(
            samples, masks, confidences, detections, truck_box, height, width
        )
        qa = evaluate_quality(records, config)

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
        )

        mask_io.save(output_dir / "masks.npz", masks, (height, width))
        (output_dir / "track.json").write_text(json.dumps(result.to_dict(), indent=2))
        record.outputs.extend(["masks.npz", "track.json"])
        record.metrics = {"qa_status": qa["status"], "masked_samples": len(masks)}

    log.info("tracking complete: %d masks, QA %s", len(masks), qa["status"])
    return result


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


def _build_records(samples, masks, confidences, detections, truck_box, height, width):
    """Per-sample facts, including the detector's independent opinion where it ran."""
    frame_area = float(height * width)
    records: list[FrameRecord] = []
    for position, sample in enumerate(samples):
        mask = masks.get(position)
        record = FrameRecord(
            frame_index=sample.frame_index,
            time_seconds=sample.time_seconds,
            has_mask=mask is not None,
            mask_area_fraction=float(mask.sum() / frame_area) if mask is not None else 0.0,
            sam_confidence=confidences.get(position, 0.0),
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
