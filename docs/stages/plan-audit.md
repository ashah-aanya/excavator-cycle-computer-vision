# Plan audit — what the design doc specifies vs what exists

**Date:** 2026-09-22. Checked against `docs/pipeline-design.md` section by section.

Kept current from here on: **every build step references the plan section it
implements, and any deviation is recorded here with its reason.** Several
deviations below were discovered by accident rather than decided, which is the
problem this file exists to stop.

## Built as specified

| § | Item | Notes |
|---|---|---|
| 2 | Stage split with a cache boundary | `track` (GPU, once) then `render`/`features` (CPU, repeatable) |
| 2 | Perception not fused with the FSM | cache keyed by video hash + config digest |
| 3 | Grounding DINO anchors + SAM 2 propagation | both Apache-2.0, ungated |
| 3.4 | No bucket prompt, no pile prompt | derived geometrically instead |
| 3.6 | Prompts limited to excavator / dump truck | `"digger."` dropped on measured evidence |
| 4.1 | Machine pivot from a pixel occupancy map | |
| 4.2 | Scale `L` as a percentile of reach | every distance divided by it |
| 4.3 | Tip measured **along the mask**, not straight-line | now via the fitted chain |
| 4.3 | **Constant-velocity filter with outlier gating** | built; extended with a backward pass, see deviations |
| 4.4 | Dig/dump zones by clustering dwell positions | plus the derived return direction |
| 4.5 | Surface height by Otsu inside the dig zone | |
| 5 | Feature table: bearing, slew, extension, elevation, speed, curl, area, zones, confidence | |
| 5.2 | Zero-phase smoothing only | Savitzky-Golay, symmetric |
| 6 | Decoupled rates: 1 Hz detection, 10 Hz tracking, source-rate render | |
| 9 | Some QA: coverage, mask-area stability, anchor agreement, SAM confidence | |
| 10 | Annotated video from the pipeline's own output | masks, boxes, chain, readout, signal strip |
| 13 | A100 target; frames streamed, masks compressed | |

## Missing

Ordered by how much each would change the answer.

| § | Item | Consequence of its absence |
|---|---|---|
| 7 | **The state machine** — both passes, the four transitions, debounce, onset back-dating, revert rule, initialisation, stage-plausibility audit | no phases, so no answer |
| 8 | **Cycle assembly, statistics, `answer.json`** | the deliverable does not exist |
| 8 | Periodicity cross-check on cycle count | no independent check on the field that matters most |
| 7 | Anomaly log | failures are invisible rather than recorded |
| 9 | QA metrics: tip-jump rate, frame-to-frame mask IoU, identity switches, background motion, cyclicity | drift and camera motion would go unnoticed |
| 9.3 | Surface-height sensitivity, `d(answer)/d(h_surf)` | T1 and T2 share that scalar and move in opposite directions; the size of that effect is unmeasured |
| 11 | Compliance tests: no hardcoded video-specific values, `answer.json` never read | the rules are claimed, not enforced |
| 11 | Model weights pinned by revision; a prefetch command | reproduction depends on upstream not moving |
| 11 | uv PyTorch index and locked platform environments | a reviewer on Linux may resolve different wheels |
| 12.2 | Threshold sensitivity sweep; synthetic hidden videos (flip, rescale, re-fps) | **no evidence the pipeline generalises** — the closest available proxy for the hidden videos |
| 5 | Truck-overlap feature; cab-rotation fallback from optical flow | thinner corroboration, no fallback when the arm is occluded |
| 4.3 | Motion gating of the bucket region | mask leakage into soil is unmitigated |
| 4.5 | Cross-check of the surface height against the background terrain edge | a single estimator with no second opinion |

## Deviations — built differently from the plan

| § | Plan | Built | Why |
|---|---|---|---|
| 4.3 | bucket region = disk around the tip | radial band along the arm | a disk crop makes any region round: elongation 1.25 vs 2.13 measured on the same masks. **The plan was wrong here** |
| 4.1 | pivot = centroid of the occupancy core | top of the core (the boom's anchor) | the centroid sits among the tracks, so arm paths started by climbing through the body |
| 5 | curl = bucket axis vs forearm axis | joint angle between the fitted bucket and stick links | a link has a direction where an axis has a 180-degree ambiguity; the ambiguity was a large part of the earlier noise |
| 9 | features stored as parquet | `.npz` plus a readable `.csv` | avoids a pyarrow dependency for no loss |
| 5 | falling-material flow as corroboration for dumping | built, then measured as non-discriminating | it reads highest while the bucket descends to dig and lowest over the truck |
| 4.3 | a forward filter | forward filter **plus a backward smoothing pass**, and a cap on coasting | forward-only trades spikes for excursions: it coasts at constant velocity through rejected runs and extrapolates into empty space. Offline the whole track is available, so later evidence can pull those back. Worst frame-to-frame jump: 86 px raw, 268 px with a naive backward pass that swept across filter restarts, 23.5 px once restarts are respected |

## Beyond the plan

| Item | Why it was added |
|---|---|
| Negative points from the truck's box when prompting | detection merges the two machines in ~51% of frames; SAM 2 then masks both. The plan did not anticipate this |
| `kinematics.py` — fitting a three-link chain | the plan named the tip and the curl but not how to recover them robustly |
| Mask speck removal | strays relocate the tip across the frame |
| Device fallback instead of crashing | the plan said "fall back loudly"; it did not |
| Frame-count verification | this video's header overstates its length by 20% |

## The pattern worth naming

Three of the deviations above were found by looking at pictures, not by
consulting the plan: the disk crop, the pivot placement, and the straight-line
tip — which the plan explicitly warned against and which was implemented anyway.
The missing temporal filter is the same kind of omission, and is the cause of
the jitter currently visible on the video.

**Next action, per §4.3:** implement the temporal filter the plan already
specifies, before considering the 3D kinematic model, which would be an addition
to the plan rather than a completion of it.
