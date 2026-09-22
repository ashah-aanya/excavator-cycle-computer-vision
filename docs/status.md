# Status — what works, measured

**Date:** 2026-09-22 · 26 commits · 108 tests · video: 480×272, 29.6 s, ~1 complete cycle

## Honest summary

Perception works. Measurement is adequate but has less margin than it looks.
**The deliverable does not exist yet** — there is no `answer.json`, because the
state machine (design doc §7) and cycle statistics (§8) are not built.

| Stage | Status | Measured |
|---|---|---|
| Detection | works | excavator found in 100% of sampled frames, median score 0.73 |
| Tracking | works | 296/296 masks; area steady at 7.2% of frame (4.2–7.8%); 0.91 IoU against independent detections |
| Arm pose | mostly works | tip on the bucket in every pose inspected; 10% of frames rejected as physically implausible |
| Cues | unproven | cannot be validated without hand-labelled boundaries |
| Answer | **missing** | §7 and §8 not built |

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
