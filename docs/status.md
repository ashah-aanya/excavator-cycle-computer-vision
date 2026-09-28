# Status — what works, measured

**Date:** 2026-09-26 · 458 tests, 1 strict xfail · videos: the 29.6 s dev clip (1 cycle) and an 83 s clip (3 cycles)

> **Updated for the shape cues** (`docs/stages/08-shape-cues.md`). The summary, the
> Cues and Answer rows, and the section after the table are current; the rest of
> this file predates the state machine and is partly stale.

## Honest summary

The pipeline runs end to end: `run.py run <video>` writes `answer.json`, and a fresh
run from the raw video reproduces the cached one's features **byte for byte**. On the
dev clip the answer passes **4 of 6** graded fields: `cycle_count`, the cycle average,
digging and swinging. Hauling and dumping each miss by 1.5 s, both because the dump
onset is found 1.8 s late.

| Stage | Status | Measured |
|---|---|---|
| Detection | works | excavator found in 100% of sampled frames, median score 0.73 |
| Tracking | works | 296/296 masks; area steady at 7.2% of frame (4.2–7.8%); 0.91 IoU against independent detections |
| Arm pose | mostly works | tip on the bucket in every pose inspected; 10% of frames rejected as physically implausible |
| Calibration | works | all three levels separable at 0.81–0.87; robust to a single glitched frame |
| **Cues** | **2 of 4 solid** | dev clip −0.17 / −0.23 / +1.77 / −0.23 / −0.13 s against ±0.6 s (was −0.77 / +1.97 / −5.14 / +2.67 / −1.23); 83 s clip 10/13 within ±0.6 s (was 0/13) |
| State machine | works | sequencing, non-overlap and `fired_at`-in-window hold over 6000 fuzzed runs |
| Answer | **4 of 6 fields** | dev clip: `cycle_count`, cycle, digging, swinging PASS; hauling +1.50 s, dumping −1.50 s FAIL |

## Where the cue error comes from

The old cues were levels ("the bucket is down and still"), true for a whole phase,
so poor at marking when one starts. They are replaced by shape cues -- a trend that
changes at the onset -- documented with every decision and measurement in
`docs/stages/08-shape-cues.md`. What remains: the dump cue fires 1.8 s late on the dev
clip, and the haul cue fires ~1.5 s early in two of the 83 s clip's three cycles.

## The number that matters

The task grades durations to ±0.6 s, so measurement noise matters only through
the timing error it causes. Elevation noise is 0.032 L (≈7 px, ≈28 cm on a 9 m
machine). Converting that to time depends on how fast elevation is moving when
it crosses the material surface:

| Surface crossing | Rate of change | Timing uncertainty |
|---|---|---|
| t = 3.9 s | 0.117 L/s | 0.27 s |
| **t = 10.0 s** | **0.059 L/s** | **0.53 s** |
| t = 28.9 s | 0.138 L/s | 0.23 s |

**0.53 s of the 0.6 s budget consumed by noise alone**, at a crossing where the
bucket leaves the material slowly. Nothing systematic has been added yet — no
detection bias, no onset-definition error. The margin is thinner than the
tracking metrics suggest, and a shallow crossing is where this pipeline will
fail first.

## Jitter, in interpretable terms

Per-frame pose fitting produced movements that no excavator can make. Assuming
a ~9 m reach (1 px ≈ 4 cm):

| | movement per 0.1 s | implied speed |
|---|---|---|
| Worst raw jump | 3.6 m | 36 m/s (130 km/h) |
| Worst after smoothing | 1.0 m | 9.7 m/s |
| Typical after smoothing | 10 cm | 1 m/s |

The temporal filter (§4.3) rejects the impossible ones: 28 of 283 measurements,
about 10%.

## What each component does

| Module | Question it answers | In | Out |
|---|---|---|---|
| `detect/` | where is the excavator? | frame + text prompt | a box |
| `track.py` | which pixels are it? | box | a mask per sample, cached |
| `kinematics.py` | where is the arm? | mask | boom base, two joints, bucket tip |
| `filtering.py` | is that physically possible? | tip track | smoothed track, implausible jumps rejected |
| `geometry.py` | what is the scene? | masks over time | pivot, scale, dig/dump zones, surface height |
| `features.py` | what is it doing? | pose + scene | elevation, swing rate, bucket angle, speed |
| `render.py` | show me | cache | the annotated video |

Every stage is inspectable, which is how each bug so far was found — by looking
at pictures, not at metrics.

## Artifacts

| What | Where |
|---|---|
| **Annotated video** | `outputs/track/real/physics.mp4` — masks, arm chain, landmarks, live values, signal strip |
| Signal plots | `outputs/track/real/signals.png` |
| Derived scene | `outputs/track/real/scene.png` |
| Evidence for every claim above | `docs/evidence/` (25 files) |
| Per-sample numbers | `outputs/track/real/features.csv` |

## What is missing, in order

1. **§7 state machine** — no phases without it
2. **§8 cycles and `answer.json`** — the actual deliverable
3. **Hand-labelled boundaries** — the only way to know whether any cue fires at
   the right instant. Everything about cue quality is currently unverified.
4. **§12.2 robustness checks** — flip, rescale, re-fps. No evidence yet that any
   of this generalises to the hidden videos.
5. **§11 compliance tests** — no-hardcoding and no-answer-read are claimed but
   not enforced.

## Two open facts about the video

* It contains about **one complete cycle**, so every reported average will rest
  on a single measurement, with no averaging to reduce error.
* Its container overstates its length by 20% (1102 frames claimed, 886 decode).
  Handled, and worth knowing if the hidden videos share the defect.
