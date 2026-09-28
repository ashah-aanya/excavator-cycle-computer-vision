# 10 — Finding the bucket again after SAM 2 loses it

**Date:** 2026-09-27 · **Branch:** `track/bucket-reseed` · **Code:** `src/excavator_cycles/reseed.py`

## The problem

The bucket is prompted **once** (`seeding.py`, doc 06) and SAM 2 follows it from
there. The bucket disappears as a matter of routine: it is lowered into the truck
bed and buried by what it tips, and it goes into the pile to dig. SAM 2 recognises
its object by comparing each frame with a memory of recent frames, so after about
a second with nothing to see, its memory is mostly "nothing", and when the bucket
comes back out it is not claimed again.

Seen on `random.mp4` (an unseen clip, 2026-09-27): the bucket went into the bed at
34.3 s, shrank to a sliver, and from 34.7 s to the end of the clip -- 10.6 s --
SAM 2's confidence sat at ~0.08 while the bucket was plainly in view. Every feature
is measured from the bucket, so every graph stopped at 34.7 s.

## What was built

1. **Audit, after the bucket pass.** A sample is *missing* when it has no bucket
   mask or its confidence is below `features.min_sample_confidence` -- the floor
   the features stage already uses to treat a sample as a gap. Reusing it means a
   lost bucket means the same thing in both stages and there is no new threshold.
   A run of missing samples lasting `track.bucket_lost_seconds` (1 s) is a
   **lost span**.
2. **Choose where to reseed inside the span**, never at the frame where the loss
   was noticed. The first seed's ranking (reach, compactness, truck clearance,
   middle of the span) scores every frame in the span, with one extra
   requirement: **no pixel of the bucket band inside the truck box**. Inside the
   box is where the bucket hides; a bucket hovering above the truck, visible,
   is allowed.
3. **Re-track that span only.** A new SAM 2 session (one object per session, doc
   06) is prompted on that frame and propagated forward to the span's end and
   backward to its start (`max_frame_num_to_track`). Samples outside the span are
   untouched.
4. **Once per span.** A span overlapping one already attempted is not retried; a
   bucket that stays buried is recorded as a gap. `track.bucket_max_reseeds` (5)
   bounds the total.

Every attempt is written to `track.json` under `bucket_reseeds`: the span, the
reseed frame (or `null`), and missing samples before and after.

## Cost

The audit reads numbers SAM 2 already recorded, so it is free. A reseed ranks
only the frames in the span (the first seed ranks the whole clip -- ~27 s on
vid1's 32 s) and runs SAM 2 over the span only (the full bucket pass on vid1 was
~39 s). A clip that never loses the bucket pays nothing.

## Evidence, before any GPU run

The audit and the reseed *choice* need no model, so both were run on the saved
masks of every run available (the SAM 2 re-track itself has not run yet):

| Run | Tracked well? | Lost span (≥ 1 s) | Reseed choice |
|---|---|---|---|
| dev clip (`dual`) | yes | none | -- |
| vid1, vid2, downloads_clip, 14415851, 16499298 | yes* | none | -- |
| random | lost at 34.7 s | 34.9–45.3 s | 42.4 s, on the bucket, above the truck |
| 15107636 | lost for 17 s | 12.1–29.7 s | 19.4 s, on the bucket, in the dirt |
| 15107628 | lost at the end | 12.7–14.3 s | **none** -- every band touched the truck box |

\*16499298 tracked the *wrong object* (the tracks) at full confidence -- a failure
this audit cannot see, because nothing went missing. It came from the moving
camera, which is out of scope (static camera is a stated assumption).

On 15107628 the first version of the truck rule tested only the band's **centre**:
the best band was in two pieces, one on the bucket and one on the cab inside the
truck box, and its centre fell between them, clear of the truck. Testing every
pixel refuses it. `tests/test_reseed.py` pins that case, and fails on the
centre-only rule.

## What was not built, and why

**Flagging a partly hidden bucket.** The obvious signal is mask area falling well
below its own median. It was measured first, on runs where the bucket was tracked
correctly:

| Run | Samples under half the bucket's own median area |
|---|---|
| dev clip | 4 of 296 (minimum 0.16×) |
| vid2 (its full cycle was found) | 16 of 141 -- 1.6 s |

The visible area changes with the bucket's angle and its distance from the camera,
so a size rule would flag real tipping -- which is exactly what the dump onset
reads (the aspect-ratio drop). Partial hiding needs a signal that separates "covered"
from "tilted"; area alone does not.

**The arm check** (does the bucket sit at the far end of the arm?) would catch
confident tracking of the wrong object, as on 16499298. Deferred: that case came
from a moving camera, and it adds a per-frame geometric check to every run.

## To verify on the cluster

Rerun `random`, `vid1`, `vid2` and a static clip with dumping. Expected, not yet
observed: `random` has one entry in `bucket_reseeds` (seeded at 42.4 s) and a
bucket again from roughly when it leaves the bed (frames show it clear by 36 s),
with the samples before that still missing -- correctly; `vid1` and `vid2` have
an empty `bucket_reseeds`, so their bucket pass is the same code path as before.
