# Motion field: measuring what the machine does, not where its joints are

**Date:** 2026-09-22. Code: `src/excavator_cycles/motion.py`, `excavator-cycles motion`.
Artifacts: `outputs/track/real/motion.png`, `motion.mp4`, `motion.csv`.

## Why this was tried

Four attempts at recovering the arm's pose from the mask have failed, and they
failed the same way each time:

| attempt | what it estimated | why it broke |
|---|---|---|
| farthest point | one pixel, by argmax | argmax flips between the bucket and the boom apex |
| disk-cropped bucket | a region, by an imposed shape | the crop made an elongated region measure round |
| blob axis for curl | orientation of ~300 px | ill-conditioned when the region is near-round |
| three-link chain fit | two breakpoints along a path | more freedom than the evidence supports, refit every frame |

Each estimates **a few parameters from a few pixels by a hard choice**, and a
hard choice does not degrade — it flips. Then motion is recovered by
**differencing** those positions, which amplifies the flip.

The motion field inverts both halves: it measures motion directly, and reduces
thousands of flow vectors with a **median**.

## The test, and what it actually showed

A criterion was fixed **before** running: correlation between flow-derived swing
rate and the chain's slew rate above 0.8 passes, below 0.5 abandons.

**First run: correlation 0.050.** That is an abandon by the stated rule. It was
wrong twice over, and both are worth recording.

**Bug.** `masks.npz` is keyed by **sample ordinal** (0…295); `track.json` is
keyed by **source frame index** (0, 3, 6 … 885). The pipeline uses ordinals
throughout, so it was never affected — but the probe joined on frame index,
which silently matched 99 frames to masks from elsewhere in the video and
dropped the other 197. It was caught by drawing the mask on the frame and seeing
a boom outlined in empty sky, not by any number. `build_motion_field` now takes
two parallel lists and refuses mismatched lengths, and `test_motion.py` covers
it.

**A mis-specified criterion.** Corrected, the correlation is 0.29 — still below
the abandon line. But the reference in that test is the chain's slew rate, which
is the thing under replacement. A low correlation with an untrustworthy
reference says the two disagree, not which is wrong. The criterion assumed the
chain was a usable yardstick, and the data says it is not:

| | median frame-to-frame change | p95 |
|---|---|---|
| flow swing rate | 0.017 rad/s | 0.12 |
| chain slew rate | 0.031 rad/s | **0.69** |

So the deciding evidence is the other half of the criterion — whether the signal
matches motion that can be seen in the frames.

## Verified against visible motion

Phases read off the frames at 1 s intervals, then the measured signals at those
same times:

| | **return swing** 25–29 s | **haul** 10–14 s | **dig** 5–9 s |
|---|---|---|---|
| what the frames show | right→left, bucket descending | lifts out, left→right | bucket parked, barely moving |
| flow swing rate | +0.09 … +0.46 | −0.09 … −0.22 | ≈ 0 |
| flow bucket vertical | −0.04 … −0.17 (falling) | +0.05 … +0.10 (rising) | ≈ 0 |
| flow speed | 0.06–0.09 | 0.03–0.08 | **0.002–0.010** |
| chain slew rate | **+2.72, +2.28** | **−3.49, +1.22** | **+0.49, −0.20** |

Three things follow.

1. **The sign convention is right, and self-consistent.** The loaded haul and
   the empty return come out with opposite signs, as they must. Nothing was
   assumed: both were read off the frames first.
2. **Vertical rate tracks the bucket correctly** — positive climbing out of the
   cut, negative descending into it.
3. **Speed separates dig from swing by an order of magnitude**, which is a
   usable dwell gate with no threshold fitted to this video.

Meanwhile the chain reports ±2–3.5 rad/s — 150–200 °/s, a slew rate no excavator
can reach — and reports motion during the four seconds the machine is visibly
parked.

## Coverage

| | samples with a usable measurement |
|---|---|
| chain (`valid`) | 82 of 296 comparable, ~86% nominally valid |
| motion field | **295 of 295** intervals |

The motion field needs the mask and two frames. It never has to find anything.

## The aggregation choice, and why it is physical

Measured during the same swing:

| aggregate | swing rate |
|---|---|
| median over the whole mask | 0.005 rad/s |
| median over the outer half | **0.257 rad/s** |

Fifty times weaker, same motion. The tracks and body occupy much of the mask and
do not rotate about the pivot in projection, so they contribute zeros that drag
the median down. `radial_fraction` excludes them. This is a statement about the
machine, not a knob turned until a number improved — and `test_motion.py` pins
it with a synthetic body whose rotation is known to be zero.

## What this does not fix

* **Dumping is still the hard transition.** The bucket is small and its paint is
  flat; flow is weakest exactly there. Between 14.5 s and 18.5 s the speed trace
  shows only 0.03–0.045 L/s.
* **Flow needs texture.** A hidden video shot further away, or in flatter light,
  will carry less of it. Untested.
* **One raw sample reaches 1.0 rad/s at t≈26 s**, at the peak of the swing.
  Smoothing absorbs it, but no outlier rule has been written.
* **Nothing is validated against hand-labelled boundaries**, because there are
  none. Everything above is consistency with motion visible in the frames, which
  is weaker evidence than a label.

## Consequence for the plan

Design doc §5 lists features derived from arm pose. This adds a **fourth,
independent cue family** — pixel motion — alongside mask geometry, mask
statistics and scene context (`docs/stages/03-cue-plan.md` asks for exactly that
independence). It does not replace the chain: elevation relative to the material
surface is a *level*, and flow measures *rates*, so the level-crossing cue for
T1/T2 still needs a position estimate. The chain's role narrows to that.

## Related work

Checked after the fact, which is the wrong order but worth recording.
Golparvar-Fard, Heydarian & Niebles (2013), *Advanced Engineering Informatics*
27(4):652–663, classify earthmoving equipment actions from **HOG + histogram of
optical flow** inside the machine region, reporting 86.33% on excavator actions.
That is prior evidence that flow statistics inside the machine region carry the
phase signal. Their method needs a trained SVM and is not usable here; the
feature choice is the part that transfers.

Two papers checked and found **not applicable**: Jeong et al. (2016),
*Procedia Manufacturing* 5:1107–1118 — CNN classification of rotor vibration
orbit plots, no video. Wang, Li & Guan (2007), *Optics and Lasers in
Engineering* 45(11):1037–1048 — angular velocity from motion blur in a single
exposure, designed for objects at hundreds of RPM; an excavator slews at
roughly 10 °/s and produces no measurable blur.
