# Tracking the bucket: what points at it, and can SAM hold it

**Date:** 2026-09-24. Branch `kinematic-signals`.
Artifacts: `outputs/spike/bucket_prompts/` (contact sheet + `metrics.json`).

## Why this stage exists

Three of the four phase onsets need the bucket separated from the rest of the
machine:

| | onset | needs |
|---|---|---|
| T1 | digging | bucket height arrives at a plateau, scoop begins |
| T2 | hauling | bucket height departs that plateau |
| T3 | dumping | bucket rotates *relative to the arm*, at the dump location |

Flow aggregated over the whole mask cannot supply any of them, and Euclidean
radius banding fails because the body sits at small radius too — the inner band
read ≈0 in every phase, so a bucket-minus-arm difference degenerated into the
bucket's absolute rate.

SAM 2 can track a bucket if something points at it once. **This stage asks what
does the pointing.**

## The probe: can Grounding DINO box the bucket?

The design doc ruled out part-level open-vocabulary grounding, but that was an
argument from analogy — the 50% merge rate was measured on *whole machines* and a
bucket prompt was never tried. So it was tried.

**Criteria, fixed before running** (`L = 217.9 px`, frame 480x272 = 130,560 px):

1. a box on ≥ 60% of anchors
2. median box area in **[475, 7600] px²** — 0.36–5.8% of frame. Target `(0.2L)² ≈
   1900 px²`; reject 4x too large (that is the arm) or 4x too small (a fragment).
   The whole excavator *mask* is ~7% of frame, so anything above 5.8% is the
   machine, not the bucket.
3. box centre at radius > 0.5·L from the pivot (189, 150)
4. stable frame to frame where the arm is stationary

**Result: FAIL.** 24 frames, Grounding DINO.

| prompt | found | score | median area | | median jump |
|---|---|---|---|---|---|
| `"excavator bucket."` | 100% | 0.645 | **25,543 px² (19.6%)** | 13x too large | 0.06 |
| `"digger bucket."` | 83% | 0.430 | **11,186 px² (8.6%)** | 6x too large | **0.70** |
| `"bucket."` | **0%** | — | — | | — |

Criterion 2 fails on both prompts that detect anything, and the contact sheet says
why:

* `"excavator bucket."` boxes the **entire machine** — arm, cab and tracks. It has
  matched the word *excavator*, not *bucket*.
* `"digger bucket."` boxes the **dump truck** in most frames. It occasionally
  finds the real bucket (f131, f168), which is what the 0.70 median centre jump
  records: the box is moving between objects, not tracking one.
* `"bucket."` alone returns nothing — without a machine word there is no anchor.

**The spike harness's own gate printed PASS.** That gate's area thresholds are
calibrated for a whole excavator, so it is not a judgement about a bucket. This is
why the criteria above were written down first, and why the harness prints *"Look
at contact_sheet.jpg before trusting these numbers."*

## What follows

The design doc's assumption was right, and is now measured rather than assumed:
**part-level open-vocabulary grounding does not work on this footage.** Do not
re-litigate it with another prompt list.

The bucket seed therefore comes from **geometry**, not from a detector: on one
deliberately-chosen frame — arm near full extension, inside the dig window, truck
cleanly separated — the far end of the mask *is* the bucket, and
`geometry.farthest_point` finds it without any hard choice that could flip. This
is safe precisely because it happens **once**, unlike the arm-chain fit, which had
to be re-derived every frame and did flip.

That choice carries a cost. Because the bucket seed is derived from the excavator
*mask*, it cannot be known before the first SAM forward pass, and the processor
assigns rather than appends `obj_with_new_inputs` — so both objects must be
registered before any object has produced a conditioning output. Hence a
two-phase session: a throwaway pass to get the excavator mask, then a fresh
session carrying both prompts.

## Still open

* **Whether SAM 2.1-tiny can hold a ~25 px object for 296 samples.** This is the
  risk the whole stage turns on and it is not addressed by anything above.
  Measured by the QA gate below.
* If that gate cannot be passed, the fallback is geodesic banding —
  `geodesic_distance(arm_region(mask, body_core(masks)), pivot)`. Distance *along
  the metal* does not collapse when the arm folds, which is the specific failure
  Euclidean banding hit.
