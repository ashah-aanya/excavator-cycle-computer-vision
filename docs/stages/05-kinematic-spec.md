# The kinematic feature set: four onsets, five quantities

**Date:** 2026-09-23. Supersedes the feature table in design doc §5 and constrains §7.
Ground truth: `eval/labels.json`. Scorer: `eval/score.py`.

## What changed

The task spec's phase definitions were read exactly, rather than paraphrased. They
are far more specific than the working assumption, and they settle several
questions that had been open for days:

> **Digging:** begins when the bucket first contacts the material and starts
> scooping.
> **Hauling:** begins when the entire loaded bucket clears the material surface.
> **Dumping:** begins when the bucket reaches the dumping location **and** starts
> tipping or uncurling to release its load. *Material that spills while the loaded
> bucket is still being lifted or transported remains part of hauling.*
> **Swinging:** begins when the excavator starts rotating back the emptied bucket
> toward the digging location.
>
> *(each phase ends immediately before the next phase begins)*

Three consequences, each of which changes the build.

**Only onsets are defined.** No phase has an end condition of its own; it ends
where the next begins. The pipeline must therefore locate **four events**, never
decide "which phase is this frame in."

**Dumping is a conjunction.** Location *and* tipping. The second sentence is the
anti-spillage guard written into the definition, not an optional refinement.

**Dumping is about the bucket tipping, not about material leaving.** This is much
easier to see than falling soil, and it revives a family of cues that had been
set aside.

## The four onsets and what each requires

| | onset | physical quantity | kind |
|---|---|---|---|
| **T1** | digging | bucket's lowest point **arrives at** the material surface | level, falling |
| **T2** | hauling | bucket's lowest point **departs above** the material surface | level, rising |
| **T3** | dumping | bucket rotates relative to the stick, **while at the dump location** | rate + gate |
| **T4** | swinging | house begins to slew back toward the dig side | rate |

T1 and T2 are the *same level crossed in opposite directions*. That symmetry is
worth exploiting and worth fearing: an error in the surface level pushes the two
apart, lengthening digging and shortening its neighbours. It is the textbook
differential bias.

## The minimal feature set

Five quantities. Everything previously in the feature table that is not listed
here is dropped.

| symbol | definition | serves |
|---|---|---|
| `h(t)` | bucket's lowest point, relative to the pivot, in units of `L` | T1, T2 |
| `h_surf` | material surface level | reference for T1, T2 |
| `theta(t)` | arm bearing about the slew pivot | location gate for T3 |
| `omega_house(t)` | slew rate of the house | T4 |
| `omega_rel(t)` | bucket's angular rate **relative to the stick** | T3 |

Dropped, with the measured reason:

| dropped | separability* | why |
|---|---|---|
| mask `area` | 0.93 | whole-machine area barely moves between phases |
| `falling` material | 0.85 | its strongest response is during **hauling** (spillage) |
| `v_radial` | 0.82 | no phase signal |
| chain `speed` | 1.25 | superseded by flow speed |
| fitted joints, `curl` from the chain | — | the chain is invalid (see below) |

\* spread of phase medians divided by typical within-phase scatter; above ~2 is
usable. Measured on the development video over phase windows read off the frames.

## Seven rules, each with the evidence that produced it

**R1. Detect onsets; never classify frames.** Phases tile, and two of them contain
long stretches where the machine is not doing the thing the phase is named after
— hauling includes **4.4 s** parked at the truck with a full bucket (f426–f559),
dumping includes **2.7 s** of repositioning after the bucket is empty. Any
per-frame classifier labels those wrongly by construction.

**R2. An onset is a departure from baseline, not a threshold crossing.** The
largest measured error on this project comes from breaking this rule. Truth for
T4 is f691; the threshold-based detector fired at f779, **3.1 s late**, because
the threshold is a fraction of `p95` — calibrated off the *peak* of the swing —
while the machine accelerates from rest:

| | house-region flow speed |
|---|---|
| parked (f645–f681) | 0.02 – 0.11 |
| f687 (true onset) | 0.23 |
| f780 (threshold trips) | 1.45 |

**R3. Two passes.** Pass 1 brackets coarsely and may lag; pass 2 walks back inside
the bracket to the departure point. Pass 1's lag never reaches the answer. This
was in the design from the start and was never built; R2 is what it was for.

**R4. Rates from pixel motion, levels from mask geometry.** Optical flow reduces
thousands of vectors with a median and never makes a hard choice, so it measures
rates well. It cannot measure a level. Levels must come from mask geometry — but
only **aggregate** geometry over a restricted region, never a fitted joint.

**R5. Normalise by radius before differencing two parts.** For a rigid body
`v = omega * r`, so the outer part always outruns the inner one and a raw velocity
difference is dominated by that gradient rather than by articulation. Measured, on
a swing where the bucket was not articulating at all:

| comparison | swing | dump plateau |
|---|---|---|
| raw velocity difference | 6.50 px | 0.97 px |
| radius-normalised (`omega`) | 0.360 | 0.035 |

The confound shrinks about twentyfold. Compare `omega`, never `v`.

**R6. Honour the conjunctions the spec states.** T3 requires the dump location
**and** the tipping motion. Either alone is wrong: at f444 the bucket visibly
tilts and sheds material *during transport*, 3.8 s before the real dump. The gate
is what separates them.

**R7. One criterion, applied identically at every instance.** A detector that
fires at first contact on one cycle and at full engagement on the next injects
differential bias directly into the phase averages. The ground-truth labels
themselves carry a known asymmetry of this kind (+9 frames at the cycle start,
+22 at the close, ≈0.43 s); see `eval/README.md`.

## Why the arm-pose chain is not part of this

The chain fitted three links to the silhouette and took the two breakpoints as
joints. The links are steel and their lengths are constant. The fit reports:

| link | median | p5 | p95 | spread |
|---|---|---|---|---|
| base → joint 1 | 52.4 px | 20.5 | 106.7 | **5.2x** |
| joint 1 → joint 2 | 49.3 px | 18.0 | 119.4 | **6.6x** |
| joint 2 → tip | 56.0 px | 13.1 | 107.0 | **8.2x** |

An eightfold variation in a rigid link is not a noisy estimate of the right
quantity; it is the wrong quantity. The joints also teleport up to 0.47 `L`
between samples 0.1 s apart, and differencing that produces slew rates of 2–3.5
rad/s, which no excavator can reach.

It is not repairable in a single view: the house slews, so the arm's projected
length changes with rotation, and the one constraint that would stabilise the fit
— constant link length — is therefore invalid in 2D. Stereo or a known camera
pose would fix it. Neither is available.

**`theta` and `h` survive this.** `theta` is `atan2` of the filtered tip about the
pivot and `h` is an extreme point of the mask; neither uses the two interior
breakpoints, which are the part that fails. They are different quantities with
different failure modes, and collapsing them was an error in earlier notes.

## State of each quantity

| quantity | status | evidence / what is missing |
|---|---|---|
| `theta` | **have** | best signal measured: separability 5.12, jitter 8.9% of phase span |
| `h_surf` | **have** | `h` pins to within ±0.0005 `L` for ~0.7 s while the bucket is buried; the pinned value *is* the surface |
| `omega_house` | **measured, not implemented** | median flow in the mask within 0.45·r_max of the pivot; baseline 0.05, 6x separation within 0.3 s of the true onset |
| `h` | **needs replacing** | currently from the raw chain and bypasses the temporal filter; replacement is the lowest mask point inside the dig zone — **untested**, and may read the tracks rather than the bucket |
| `omega_rel` | **open** | radial bands cannot separate stick from body: the body sits at small radius too, so the inner band reads ~0 in every phase and the difference degenerates to `omega_bucket` alone. Needs real part masks. |

Four of five are in hand or close. `omega_rel` is the one genuinely unsolved
problem, and it is the only thing standing between this design and a complete set
of onset detectors.

## What this does not answer

* **`omega_rel` has no implementation.** The proposed route is a bucket mask and a
  stick mask, seeded once by propagation from a frame where the arm is extended
  and well separated from the truck (the dig window qualifies), then `omega`
  differenced between them. Untried.
* **`h` from mask geometry is untested.** If the undercarriage sits lower in the
  frame than the bucket, the statistic reads the tracks and the cue is worthless.
  This is a short experiment and it gates T1 and T2.
* **Nothing here is validated on a second video.** Every number is from the
  development clip, which contains one complete cycle. The two hidden videos are
  untested in every respect.
* **The surface-level sensitivity is unquantified.** T1 and T2 cross `h_surf` in
  opposite directions, so the design doc's `d(answer)/d(h_surf)` sweep is now a
  requirement rather than a nicety.
