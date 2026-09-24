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

### A wider sweep, because three prompts is a thin basis for "does not work"

Eight more prompts across **two** detectors (Grounding DINO and OWLv2), 16
combinations, same criteria. Only two landed inside the area band, both OWLv2:

| detector | prompt | found | median area | jump |
|---|---|---|---|---|
| owlv2 | `"digging attachment."` | 100% | **2,155 px²** (target 1900) | 0.75 |
| owlv2 | `"backhoe bucket."` | 68% | 2,560 px² | 0.96 |

Everything else was 4-16x too large or returned nothing. Cross-model agreement
between the two detectors was **median IoU 0.12** — they are not finding the same
object.

`owlv2 / "digging attachment."` was re-run alone so its contact sheet could be
read. Isolated, its centre jump is **0.04** — and that number is the tell. A box
genuinely on the bucket *must* move, because the bucket swings across the frame;
a box that barely moves is on something that does not. Looking confirms it: the
modal detection is the **tracks and undercarriage**.

Not uniformly, though. At f114 the box is squarely on the bucket at **0.47**, the
highest score anywhere in the sweep, and the same happens at f160 and f850. The
pattern is legible: OWLv2 finds the bucket when it is **low and visually
separated during digging**, and falls back to the undercarriage once the bucket
is airborne near the truck.

That is not a detector we can seed from. But it is a useful negative: the word
that works best is a *function* word (`"digging attachment"`), not a part name,
and it works exactly where the part is visually isolated — which is the same
condition that makes the geometric seed unambiguous.

**The spike harness's own gate printed PASS.** That gate's area thresholds are
calibrated for a whole excavator, so it is not a judgement about a bucket. This is
why the criteria above were written down first, and why the harness prints *"Look
at contact_sheet.jpg before trusting these numbers."*

## What follows

The design doc's assumption was right, and is now measured rather than assumed
across **19 detector x prompt combinations on two architectures**: part-level
open-vocabulary grounding does not work on this footage. Do not re-litigate it
with another prompt list.

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

## The geometric seed, drawn and checked

`farthest_point` was run against every cached mask and the proposed box drawn on
the frame (`docs/evidence/geometric-seed-across-clip.jpg`). Two things showed up
that no summary statistic would have surfaced.

**Where the farthest point actually lands:**

| arm attitude | box lands on |
|---|---|
| extended, in the dig window | the bucket |
| extended right over the truck | the bucket |
| **raised high (the dumping window)** | **the boom apex** |
| folded | the tracks |

This is the known failure of an extreme point and it is harmless, because the
seed is taken **once** on a frame we choose. It is also a reminder of why the
same operation could not be trusted per-frame in the arm chain.

**Two corrections to the seed rule, both found by looking:**

1. *Max extension alone picks the wrong frame.* It chose sample 295 — the last
   frame in the clip — so SAM would reverse-propagate the entire video from one
   end. Score candidates on reach x box-fill x mid-clip-ness instead.
2. *The box was centred on the bucket's outermost corner*, so it covered roughly
   half bucket and half background. `farthest_point` returns an extremity, not a
   centre. Step inward toward the pivot by half a box side.

With both fixed, the rule chooses **sample 53 (t = 5.3 s, frame 159)**: reach
0.74 L, tip 198 px clear of the truck box, and the seed box **83% machine
pixels** (`docs/evidence/geometric-seed-chosen.jpg`). **92 of 296 samples**
qualify as candidates, so the rule does not hang on one lucky frame.

The box still clips a little of the stick at its top-right corner. SAM segments
the dominant object inside a box, and the pivot is available as a negative point
if object 2 over-reaches up the arm.

## What was wired, and two things the API does that the plan did not say

The rule that shipped is the **fallback below, not `farthest_point`**: geodesic
banding (`seeding.choose_seed`), because it does not collapse when the arm folds
and it hands SAM all three prompt forms instead of only a box. The stage sends
the band as a **mask** by default — the strongest form in the ablation, and free,
since the band had to be computed anyway. `track.bucket_prompt` switches it to
`points` or `box` without touching code.

Reading `transformers/models/sam2_video` while wiring it turned up two things
that would each have failed silently on the GPU:

1. **Both objects must be prompted on the *same* frame.** `forward` treats any
   object flagged as having new inputs as being on its conditioning frame, then
   looks up that object's prompt *for the frame it is processing*. An object
   prompted on a different frame finds nothing, and is conditioned on nothing —
   no error. So the bucket is seeded on the excavator's own seed sample, and the
   throwaway pass exists to produce that one mask.

2. **Registering the second object erases the first's prompt.**
   `obj_with_new_inputs = obj_ids` is an assignment, so two registration calls
   before the first forward pass leave only the second object pending; the
   excavator would then be tracked from a memory bank it never built. Both calls
   are still needed — points for two objects in one call must have equal point
   counts — so `track.register_prompts` restores the union afterwards.

The throwaway pass propagates `track.bucket_seed_window_seconds` (4 s) rather
than a single frame, because the geodesic rule subtracts the machine's
persistent body before tracing the arm, and the persistent body of *one* mask is
that whole mask. With one frame the subtraction removes everything, the pivot
lands part way up the boom, and the band is empty — reproduced in
`tests/test_bucket_track.py`. Set it to 0.0 for a literal single forward pass.

## Still open

* **Whether SAM 2.1-tiny can hold a ~25 px object for 296 samples.** This is the
  risk the whole stage turns on and it is not addressed by anything above.
  Measured by the QA gate below.
* If that gate cannot be passed, the fallback is geodesic banding —
  `geodesic_distance(arm_region(mask, body_core(masks)), pivot)`. Distance *along
  the metal* does not collapse when the arm folds, which is the specific failure
  Euclidean banding hit.
