# Stage 2 findings — tracking, and the merged-box problem

**Date:** 2026-09-21
**Verdict:** SAM 2 **video propagation** tracks the excavator correctly, including
through the frames where detection cannot separate it from the dump truck.
Single-frame box prompting does **not**. Stage 3 (geometry) proceeds.

## The real video, at last

`construction_excavator_cycle_duration_1.mp4`

| | |
|---|---|
| Resolution | **480 x 272** |
| Frame rate | 29.989 fps |
| Duration | 86.8 s (2602 frames) |
| Machine | Caterpillar excavator + articulated dump truck |
| Estimated cycles | ~6-7 at roughly 13 s each |

Two consequences worth stating plainly:

* The screenshots used earlier are this video **upscaled about 4.7x** by the
  display. They are not lower quality than the source; they are a magnified view
  of it. The source is the limit.
* At 480x272 the excavator is roughly 220 px wide and the bucket perhaps 25 px.
  **Any angle measured from a 25 px object will be noisy**, which puts pressure on
  the curl-angle cue for the dumping boundary. See "Implications" below.

## Detection on the real video

19 sampled frames: **100% detection, median score 0.73**, exactly one box per frame,
every box on the excavator. Evidence:
[`../evidence/spike-real-video.jpg`](../evidence/spike-real-video.jpg).

But sampling 39 frames across the whole video and also detecting the truck shows
the real picture:

| | |
|---|---|
| Excavator missed | **0 of 39** |
| Box merged with the truck (IoU > 0.5) | **20 of 39 (51%)** |
| Box area, clean frames | 17-21% of frame |
| Box area, merged frames | 28-37% of frame |

**The merges cluster in time** -- 10-21 s, 36-47 s, 64-74 s -- which is the
arm-over-truck window: hauling through dumping. The failure is systematic with
phase, not random, and it lands on the phases that matter most.

## Box prompting fails; propagation works

Evidence: [`../evidence/sam2-box-prompt-fails.jpg`](../evidence/sam2-box-prompt-fails.jpg)
and [`../evidence/sam2-video-propagation-works.jpg`](../evidence/sam2-video-propagation-works.jpg).

| Approach | Mask area, merged frames | Mask area, clean frames |
|---|---|---|
| Single-frame box prompt | **13.8%** (covers both machines) | 6.9-7.5% |
| Video propagation, prompted once | **7.0-7.7%** | 7.3% |

Propagated over 0-25 s at 10 Hz from a single box on frame 0: median mask area
7.3%, range 4.3-8.4%, with no excursion during the merged window. Visually the
mask is the excavator alone in every sampled frame.

**Why.** A single-frame prompt gives SAM 2 only a box that misstates the object's
extent, and it grabs the most salient region consistent with it -- both machines.
In video mode it has already seen the excavator alone on earlier frames, so
"which pixels match the thing I am following" resolves correctly even when the
box would mislead it. The memory is doing the work that the box cannot.

## Implications for the design

1. **Never re-anchor blindly.** Re-anchoring on a merged box would poison the
   memory with the truck. The merge is detectable for free -- we already detect the
   truck, and IoU > 0.5 between the two boxes identifies it -- so re-anchoring must
   skip those frames. This was listed as a fallback; it is now required.
2. **Seed on a clean frame, not frame 0.** Choose the initial prompt where the
   boxes are separated and confidence is high.
3. **Mask area is a live QA signal.** Clean tracking sits near 7%; swallowing the
   truck roughly doubles it. That threshold is a per-video statistic, not a
   constant.
4. **The curl-angle cue is under pressure at this resolution.** A ~25 px bucket
   gives a noisy long-axis estimate. Stage 3 must measure how noisy before the
   dumping trigger is finalised; if it is too noisy, the fallback cues are the
   bucket's position over the dump zone and falling-material motion below it.

## Cost

Propagation ran at **0.7 fps on CPU** (251 frames in 337 s). The full video at
10 Hz is ~870 frames, so roughly 20 minutes on CPU. This is the stage that wants
the GPU; it should drop to a couple of minutes on the A100, and it only runs once
per video because the masks are cached.

## Caveats

* Tested over 0-25 s, one merged window of three. The remaining windows and the
  long-run drift behaviour are not yet verified.
* `sam2.1-hiera-tiny` was used for speed. Larger checkpoints may behave
  differently, better or worse.
* No re-anchoring was exercised -- this was a single prompt propagated. The
  interaction between re-anchoring and merged frames is exactly what implication 1
  above is about.

---

## A per-frame fix: negative points from the truck's own box

Evidence: [`../evidence/truck-negative-points-fix.jpg`](../evidence/truck-negative-points-fix.jpg)

SAM 2 accepts *negative* points -- "not this" -- alongside a box. Since Grounding
DINO already locates the truck, those negatives can be placed automatically.

The wrinkle: on a merged frame the truck box **is** the merged box, so points drawn
from it would land on the excavator. The resolution is that **the truck is parked**.
Its box measured on frames where the two are cleanly separated stays valid on the
frames where they are not.

```
DINO "excavator."  -> box                  -> SAM: the object is HERE
DINO "dump truck." -> box (clean frames,   -> SAM: and NOT here  (6 negative points)
                          median)
```

| frame | box only | box + 6 negatives |
|---|---|---|
| 10.3 s | 13.8% (both machines) | **7.3% — excavator only** |
| 14.5 s | 13.0% (both machines) | **5.8% — excavator only** |
| 35.9 s | 13.8% (both machines) | **7.1% — excavator only** |
| 63.7 s | 13.6% (both machines) | **7.5% — excavator only** |

Those match the ~7% that clean frames produce, and visual inspection confirms the
mask covers the excavator including the bucket over the truck bed, with the truck
excluded.

The truck box was derived from 6 of 16 sampled frames where excavator/truck IoU
was below 0.2, then taken as the median: `(235,134,377,211)`.

### What was tried first and did not work

Evidence: [`../evidence/prompt-combos-tested.jpg`](../evidence/prompt-combos-tested.jpg)

| prompt | mask area on merged frames |
|---|---|
| box only | 13.8% (both machines) |
| box + one negative point placed by motion | 14.3% (no improvement) |
| a single positive point, no box | 20-43% (worse -- grabs everything) |

A single negative point is too weak, and a bare point leaves the extent ambiguous.
Several negatives spread through the excluded object is what does the work.

### Why this matters more than the propagation result

Propagation fixes merged frames only because **memory** carries the identity forward,
which makes correctness depend on the seed being clean and on never re-anchoring.
Negative points fix a merged frame **on its own terms**, with no history. So:

1. A video that *opens* on a merged frame can still be seeded correctly.
2. Re-anchoring stops being dangerous -- a re-prompt that always carries truck
   negatives cannot poison the tracker's memory with the truck.
3. The elaborate anchor-scheduling scheme is unnecessary. The rule becomes: include
   truck negatives whenever prompting, and re-anchor on a fixed cadence if at all.

### Limits

* Assumes the truck is stationary for the stretch where its box is reused. If it
  departs and another arrives elsewhere, the box must be refreshed from the nearest
  clean frames rather than taken once globally.
* If no truck is detected anywhere, no negatives are needed, and no merge can occur.
* Six points, biased toward the lower part of the truck box where the arm rarely
  reaches, was the first configuration tried and it worked; the count and placement
  have not been swept.
