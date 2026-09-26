# Status — what works, measured

**Date:** 2026-09-26 · 385 tests, 1 strict xfail · video: 480×272, 29.6 s, 1 complete cycle

## Honest summary

The pipeline runs end to end: `run.py run <video>` writes `answer.json`, and a fresh
run from the raw video reproduces the cached one's features **byte for byte**.
`cycle_count` is correct. **The four phase averages are 0.000, and that is the honest
output rather than a bug** — all four coarse cues fire outside the window containing
the transition they look for, so pass 2 has nothing to refine and refuses to invent a
number.

| Stage | Status | Measured |
|---|---|---|
| Detection | works | excavator found in 100% of sampled frames, median score 0.73 |
| Tracking | works | 296/296 masks; area steady at 7.2% of frame (4.2–7.8%); 0.91 IoU against independent detections |
| Arm pose | mostly works | tip on the bucket in every pose inspected; 10% of frames rejected as physically implausible |
| Calibration | works | all three levels separable at 0.81–0.87; robust to a single glitched frame |
| **Cues** | **wrong** | every window misses its onset: −0.77 / +1.97 / −5.14 / +2.67 / −1.23 s against ±0.6 s |
| State machine | works | sequencing, non-overlap and `fired_at`-in-window hold over 6000 fuzzed runs |
| Answer | **1 of 6 fields** | `cycle_count` 1 = 1 PASS; the five duration fields are 0.000 |

## Where the cue error comes from

Not four independent problems. `low_height.threshold` is the Otsu valley between the
dig mode (−0.11 `L`) and the carry mode (+0.23 `L`), landing at 0.0942 `L` — about 1.3
bucket-heights above where the bucket sits while digging. That one misplacement makes
digging run long and hauling fire late. Dumping is separate and worse: it tests
POSITION where the spec requires TIPPING, so it fires 5.14 s early on the bucket
passing over the bed while still hauling — the exact case the spec's anti-spillage
clause legislates against. `aspect_ratio` is computed, labelled "Carries T3", and read
by no trigger.

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
