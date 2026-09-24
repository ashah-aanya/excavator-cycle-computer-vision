# Separating the bucket from the arm without a bucket mask

**Date:** 2026-09-24. Branch `kinematic-signals`.
Evidence: `docs/evidence/geodesic-bands.jpg`. Labels: `eval/labels.json`.

## The problem this solves

Three of the four phase onsets need the bucket separated from the rest of the
machine. Two routes were open: track a bucket mask with SAM 2 as a second object
(needs a seed, a GPU re-run, and a ~25 px object to survive 296 frames of
propagation), or partition the existing excavator mask geometrically.

Euclidean radius banding had already failed: the body sits at small radius too, so
the inner band was mostly cab, `omega_stick` read ≈0 in **every** phase, and the
bucket-minus-arm difference degenerated into the bucket's absolute rate.

**Geodesic distance does not have that problem.** Distance measured *along the
metal* cannot rank a midpoint of the arm above its end, however the arm is folded.

## The method

All four functions already existed; nothing new was written for the probe.

```python
core  = body_core(masks, quantile=0.9)      # pixels occupied most of the time
arm   = arm_region(mask, core)              # mask minus the dilated body
g     = geodesic_distance(arm, pivot)       # wavefront dilation, constrained
reach = g[g >= 0].max()

bucket = g >= 0.82 * reach
stick  = 0.50 * reach <= g < 0.75 * reach   # gap is deliberate: keep the joint out
```

Rotation per region reuses the existing decomposition, `median((r x F)/|r|^2)`.
The `|r|^2` is the radius normalisation: for a rigid body this is identical at
every radius, so a **difference** between two regions is a joint rate rather than
a velocity gradient. Without it, the bucket always outruns the arm in raw pixel
velocity even with the bucket cylinder locked.

## What works

**The bands land on the right parts.** Drawn on twelve frames spanning the clip,
the bucket band is on the bucket and the stick band on the stick in **10 of 12**.
The two failures are both with the arm folded low (f249, f870). Coverage is
**96%** of sample intervals.

This is the part that had never been demonstrated before, and it is the reason the
route is viable at all.

**`Δω` reverses sign at the dumping onset.** Zero-phase smoothed:

| t | Δω |
|---|---|
| 17.0–17.8 s | **+0.055** (sustained plateau) |
| 18.4 s | +0.003 |
| **18.5 s** | **−0.002  ← zero crossing** |
| 18.6 s | −0.008  (label f559 = 18.65 s) |
| 19.8 s | **−0.137** (sustained excursion) |

A crossing at ≈18.55 s against a label of 18.65 s: **0.1 s**, well inside the
±0.6 s tolerance. The shape is physically right — the bucket rotates one way
relative to the stick while carrying, then reverses when the operator uncurls.

## What does not work yet

**The crossing is not unique.** In the 16–23 s window alone `Δω` crosses zero
around 16.25, 18.55, 20.35 and 20.75 s. The one at 18.55 s is distinguished by
separating a *sustained* plateau from a *sustained* excursion, but four crossings
in seven seconds means a 0.1 s match could be coincidence. A gated detector has to
earn this; eyeballing it does not.

**The swing still dominates.** Largest smoothed |Δω| excursions in the clip:

| t | Δω | what it is |
|---|---|---|
| 26.4 s | **+1.779** | the return swing |
| 2.0 s | +0.757 | tail of the prior cycle |
| 18.6 s | ≈ −0.14 | **the actual dump** |

A detector that simply finds the largest excursion fires at 26.4 s. The location
gate is therefore not a refinement — it is what makes the cue usable, which is
exactly what the spec's conjunction (*"reaches the dumping location **and** starts
tipping"*) has been saying all along.

## A criterion that was wrong when it was written

C1 was pre-registered as *"during the return swing, `omega_stick` must track
`omega_house` within ~30%."* It cannot: the house rotates about a **vertical**
axis, which in projection is a small lateral shift, not image-plane rotation about
the pivot. `omega_house ≈ 0` is correct behaviour, not a failure. The reference
was wrong, not the measurement.

Recorded because this is the second pre-registered criterion on this project to be
mis-specified — the first assumed the arm chain was a usable yardstick while it
was the thing being replaced. Writing criteria in advance protects against
hindsight, not against being wrong about the physics.

## Why this route is preferred over a tracked bucket mask

* **No seed.** Every frame is computed from its own mask, independently. There is
  no single decision that propagates, so a bad frame is bad for that frame only.
  SAM 2 keeps its seed as a permanent memory anchor, so a bad seed there does not
  merely start the track wrong — it re-anchors every subsequent frame.
* **No 25 px propagation risk**, no GPU re-run, no gated weights, no new dependency.
* **Failure is visible.** The bands are drawable, and every failure caught on this
  project so far was caught by drawing something, never by a metric.

What is given up: the band edge is a fraction of reach rather than the true bucket
pin, so `h(t)` will carry some stick pixels, and there is no clean
`bucket ∩ truck_box` overlap — tip position relative to the truck box substitutes,
more weakly.

## Related work

The three papers the task spec cites do **not** localise the bucket from video.
Cheng et al. (2023) annotate a single box around the whole machine with the
activity as an attribute (visible in their CVAT screenshot, Fig 5) and train YOWO
on 20 hand-annotated cycles. Molaei et al. (2023) is not a vision paper at all —
four IMUs bolted to the bucket, arm, boom and cabin, with video used "only for
data annotation". Kim et al. (2018) track whole excavator and dump-truck boxes and
infer action from centroid displacement plus normalised image differencing inside
the box, which detects *that* the machine is articulating, never *where* the
bucket is.

Their reported accuracies are not comparable to a ±0.6 s per-boundary tolerance:
Cheng's 99.7% is an average cycle time after discarding low-confidence frames,
Kim's 5.4% is error on aggregate activity duration over 100 minutes, and only
Molaei reports seconds (<1.5 s) — from bolted sensors.

Two leads outside the spec's list, neither read: **Bao, Sadeghi &
Golparvar-Fard (2016)**, CRC 849–858, which per Kim's description *"determined
activity types by analyzing distance and elevation changes of the detected parts
(e.g., bucket, body, and joint)"* — the same approach as this document; and
**Chen et al. (2023)**, *Automation in Construction* 146:104702, zero-shot
activity recognition "without pre-training or fine-tuning".

Techniques from that literature worth adopting, all training-free: Kim's
**normalised image differencing** as an independent articulation cue, Kim's
**scale-invariant centroid ratio**, and Cheng's **confidence gating before
averaging**, which moved cycle-time accuracy from 81.59% to 99.7% in their setting
and whose principle — discard uncertain samples, then aggregate — survives outside
it.
