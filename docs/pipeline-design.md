# Pipeline Design — Excavator Work-Cycle Duration

Design document for the excavator cycle-duration task. Architecture and reasoning only: no
build order, no schedule, no code. Per-stage implementation specs are written just before each
stage is built, because each stage's details depend on what the previous stage actually produces.

**Status:** draft, pre-implementation. The input video has not yet been inspected; every number
quoted from prior work is context, never a value to encode.

---

## 1. Problem and grading model

### 1.1 The task

Find every **complete** work cycle in a video of a hydraulic excavator loading material, and report
the average duration of each phase plus the average cycle duration.

The four phase boundaries, quoted from the task spec, are the contract this pipeline is measured
against. Nothing else defines correctness:

| Phase | Begins when |
|---|---|
| **Digging** | the bucket first contacts the material and starts scooping; includes loading and lifting the bucket |
| **Hauling** | the **entire** loaded bucket clears the material surface, continuing as it moves toward the dumping location |
| **Dumping** | the bucket reaches the dumping location and **starts tipping or uncurling** to release its load. Material that spills while the loaded bucket is still being lifted or transported **remains part of hauling** |
| **Swinging** | the excavator starts rotating the emptied bucket back toward the digging location; ends immediately before the next digging phase begins |

Output (`answer.json`):

```json
{
  "cycle_count": 1,
  "average_cycle_duration_seconds": 15.200,
  "average_phase_duration_seconds": {
    "digging": 2.501, "hauling": 3.342, "dumping": 4.234, "swinging": 5.123
  }
}
```

Graded on the provided video **plus two hidden videos**, at **±0.6 s tolerance per field**
(~18 frames at 30 fps).

### 1.2 Constraints that shape the design

**Allowed:** object detectors, segmentation models, trackers, optical flow, motion estimation,
temporal smoothing, rule-based processing, GPU inference. Text-conditioned detection or
segmentation is allowed *when the model returns standard CV outputs such as boxes or masks*.

**Not allowed:** hardcoded answers, event frames, tracks, durations or other video-specific values;
reading `answer.json` from the pipeline; calling any vision-language or multimodal LLM to analyze
the videos or determine the answer.

Two consequences drive everything below:

1. **No trained phase classifier.** One unlabeled video is far too little to train on, and the
   obvious shortcut — having a VLM label frames — is banned. The approach is therefore
   *pretrained perception → measured kinematics → rule-based state machine*.
2. **The pipeline must run fully automatically on videos we will never see.** No per-video manual
   step, no clicking a first-frame prompt. Every threshold must be either dimensionless or derived
   from the video under analysis.

### 1.3 The error budget — where accuracy actually comes from

The four phases **tile** the cycle with no gaps. Two consequences follow, and they reorder the
engineering priorities:

**(a) The average cycle duration telescopes.** If cycles are contiguous,

```
average_cycle_duration = (last_dig_start − first_dig_start) / N
```

Every *interior* boundary error cancels exactly. This field is almost free — **provided
`cycle_count` is right.**

**(b) A uniform lag is invisible; only differential bias hurts.** If every boundary is late by the
same δ, all four phase durations are unchanged. What damages the answer is one transition type
lagging *differently* from another.

Rough magnitudes, with `N` complete cycles and per-boundary error σ<sub>b</sub>: a phase duration
uses two boundaries, so its error is √2·σ<sub>b</sub>, and averaging over N cycles divides by √N.
For N ≈ 5, staying inside ±0.6 s tolerates roughly **σ<sub>b</sub> ≈ 0.45 s of random error** — a
lot — but only about **0.35 s of systematic differential bias**.

**N is small here.** The provided video is about one minute long, so at roughly 13 s per cycle it
contains on the order of 4–5 complete cycles. Two consequences: averaging buys less noise
reduction than it would on longer footage, and **a single miscounted or wrongly excluded cycle
moves every field substantially** — one cycle out of five is a 20% error in the count and shifts
the averages with it. This sharpens the priority order above rather than changing it.

**Priority order, highest first:**

1. **`cycle_count` correctness.** One miscount corrupts every field.
2. **Differential bias between the four transition definitions.**
3. Random per-boundary noise — a distant third.

### 1.4 Internal consistency, enforced

Phase averages are computed over *exactly* the same set of complete cycles as the cycle average.
Then `average_cycle_duration == Σ average_phase_duration` holds to floating point, and the pipeline
asserts it. It costs nothing, and since the grader's reference must obey the same tiling, a
mutually consistent answer is strictly more likely to land in tolerance than five independently
estimated numbers.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ A. PERCEPTION                      expensive · GPU · once per video │
│    decode → detect (anchors) → track → per-frame masks              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                     ════ CACHE BOUNDARY ════   compressed masks + feature table
                              │
┌─────────────────────────────────────────────────────────────────────┐
│ B. GEOMETRY + FEATURES             cheap · CPU · seconds            │
│    masks → scene anchors (rotation centre, scale, dig/dump zones,   │
│            surface height, return direction) → feature table        │
├─────────────────────────────────────────────────────────────────────┤
│ C. FSM PASS 1 — coarse: did the state change?                       │
│    ordered transitions, sustained evidence                          │
│    → a bracketed window per boundary                                │
├─────────────────────────────────────────────────────────────────────┤
│ D. FSM PASS 2 — fine: exactly when did it start?                    │
│    inside each bracket, find the physical turning point             │
│    → exact boundary times                                           │
├─────────────────────────────────────────────────────────────────────┤
│ E. SMOOTHING SLOT (placeholder; see §7.6)                           │
├─────────────────────────────────────────────────────────────────────┤
│ F. CYCLES + STATS → answer.json                                     │
├─────────────────────────────────────────────────────────────────────┤
│ G. ANNOTATION RENDERER → annotated.mp4 (also the QA artifact)       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Why perception is not fused with the state machine

A single streaming loop (detect and decide phases frame by frame) looks tempting. It is the wrong
choice here:

| | Fused loop | Cached stages |
|---|---|---|
| Model inference | minutes | minutes — *identical work* |
| State machine | microseconds | microseconds |
| **First run** | about the same | about the same |
| **Re-run after tuning a rule** | **full re-run, minutes** | **sub-second** |
| Thresholds calibrated on the video's own statistics | impossible | yes |
| Dig/dump zones known | impossible | yes |

Fusing saves nothing on the first run, because the cost is essentially all GPU inference, and makes
every subsequent run roughly a thousand times slower — which is where development time actually
goes. It also breaks two things structurally:

- **The dig and dump zones do not exist until the whole trajectory has been seen** (§4.4). Ten
  seconds into a fused loop, there is no way to know where the dump location is.
- **Thresholds calibrated against the video's own distribution** (§5.2) require all the frames.
  Deciding on the fly would force fixed constants, which are either video-specific (banned) or too
  loose to be accurate.

A fused streaming design is the right one for live camera analysis. This task analyzes a file.

### 2.2 The two-pass state machine

| Pass | Question | Output | Property |
|---|---|---|---|
| **1 — coarse** | Did the state change? | a *bracketed window*: "hauling began somewhere in [4.1 s, 5.0 s]" | May be deliberately conservative — its lag never reaches the answer |
| **2 — fine** | Exactly when did it start? | a single time: "hauling began at 4.37 s" | Searches inside the bracket for the physical turning point |

This split is what lets pass 1 demand strong, sustained evidence without that caution turning into
a late bias: pass 2 goes back inside the window and finds the true onset regardless.

Both passes are cheap and read the same cached feature table, running back to back.

---

## 3. Detector choice

### 3.1 What the pipeline actually needs

| Concept | Needed for | Hardest requirement |
|---|---|---|
| Excavator body | rotation centre, scale unit, cab rotation | stable mask, no identity switches, near-total coverage |
| Bucket / end-effector | tip position, elevation, **curl angle** | orientation ⇒ needs a **mask**, not a box |
| Material surface | the elevation that defines two boundaries | only a scalar height, not a mask |
| Truck bed | *optional* corroboration for dumping | must be allowed to be absent |

That third column rules out one whole family: a rotating arm-and-bucket is a diagonal elongated
structure whose axis-aligned bounding box is mostly empty, and whose box centroid and aspect ratio
swing wildly with arm angle. **A box-only detector cannot give a usable curl angle**, which is the
primary dumping signal.

### 3.2 Options

| Option | Verdict |
|---|---|
| **COCO-pretrained YOLO segmentation** | COCO has `truck` but no excavator, no bucket, no terrain; it tends to fire `truck` on the excavator too. Useful only as a cheap truck prior. Also AGPL-3.0, which complicates the licensing story of a deliverable repo. |
| **Grounding DINO / OWLv2** (open-vocabulary detection) | Apache-2.0, ungated, first-class in HF `transformers`. Strong on whole objects ("excavator", "dump truck"); unreliable on parts ("excavator bucket") and useless on loose soil. **Boxes only.** → ideal *anchor*, not a per-frame workhorse. |
| **SAM 3** (promptable concept segmentation, text → masks) | Best zero-shot quality available, but **weights are gated behind a Meta approval request**, and the image predictor is reported at roughly 2.9 s/image on a high-end GPU with a 3.45 GB checkpoint. A reviewer running `uv run` cannot be blocked on an approval queue. → optional, never required. |
| **Open-vocab anchors + SAM 2 video propagation** | SAM 2.1 is Apache-2.0, ungated, ~160 MB, and reported around 44 fps for video propagation. Gives temporally coherent, identity-stable masks — exactly what a kinematic feature extractor needs. Weakness is drift over minutes, fixed by re-anchoring. |
| **Fine-tune a small detector on self-labeled frames** | Fast and tiny, but trained on one machine, one camera angle, one colour — precisely what fails on two unseen videos. Also adds a training artifact the reviewer must trust. |

### 3.3 Decision

> **Primary:** Grounding DINO anchors at **1 Hz** + SAM 2.1 mask propagation at **10 Hz**, with the
> end-effector derived **geometrically** from the excavator mask.
>
> **Optional:** SAM 3 behind `--detector sam3`, also used as the automatic retry when the
> perception QA gate fails.

Reasoning: the primary path is Apache-2.0 end to end, ungated, mask-producing, identity-stable, and
fast enough to iterate on. The optional path is higher quality but gated, so it can never be a
requirement. Agreement between the two on the development video is itself a useful QA artifact.

The anchor model is a swappable component (§11), and **OWLv2 is evaluated beside Grounding DINO in
the detector spike** — same library, near-identical interface, so the comparison costs minutes.
Whichever finds the excavator more reliably on real frames wins; see §3.5 for the alternatives that
were ruled out and why.

### 3.4 Two deliberate non-goals

**Do not detect "the bucket" as a concept.** A bucket is a *part*, and part-level open-vocabulary
grounding is the least reliable and least temporally stable regime for these models. Instead detect
the **whole excavator** — canonical and zero-shot-easy — and derive the end-effector from
articulated geometry: the arm tip is the mask point farthest (along the mask, not in a straight
line) from the rotation centre. This never "loses the class" and degrades gracefully when the bucket
is buried or partly occluded. A bucket detection, if one is reliable, becomes a *refiner* that must
agree with the geometric tip, never a dependency.

**Do not segment the material pile.** Amorphous terrain is the worst case for any segmenter. The
two scene anchors actually needed — the dig location and the dump location — fall out of the tip
trajectory itself (§4.4).

### 3.5 Alternatives considered

The open-vocabulary field is larger than the table in §3.2, so the near neighbours were checked
explicitly:

| Model | Open-vocab | Output | License | Verdict |
|---|---|---|---|---|
| **Grounding DINO** | yes | boxes | **Apache-2.0** | **Chosen.** In HF `transformers`; reported to hold up comparatively well on out-of-distribution objects |
| **OWLv2** | yes | boxes | **Apache-2.0** | **Evaluated alongside it in the spike** — same library, near-identical API, comparable published accuracy |
| **YOLO-World v2** | yes | boxes | GPL-3.0 | Much faster and smaller, but weaker on small objects and rare categories; copyleft |
| **YOLOE** (ICCV 2025) | yes | **masks** | AGPL-3.0 | Technically interesting — outputs masks directly, so it could replace detector *and* tracker — but copyleft, and its masks on a thin articulated arm are unlikely to match a dedicated video segmenter's, with no temporal memory. Held in reserve. |
| **Construction-specific detector** (e.g. ACID-trained) | no — fixed classes | boxes | **CC-BY-NC, non-commercial**, access by request form | Rejected on both counts: the non-commercial clause is a poor fit for a company deliverable, and an approval-gated download means the reviewer cannot reproduce it. Same category of problem as gated weights. |

**Why the YOLO family's main advantage does not apply here.** Their selling point is real-time
open-vocabulary detection. This pipeline runs detection **once per second**, purely to anchor the
tracker — a few hundred calls per video. The difference between 10 ms and 150 ms per call is
seconds over a multi-minute run. The speed is a feature we would not use, while the licensing cost
is real.

**What the detector is actually selected on:** reliability on an object class none of these models
were benchmarked against (construction machinery appears in no standard open-vocabulary benchmark),
and a permissive license for code a reviewer installs and runs. Published mAP numbers cannot settle
the first, which is why the spike compares two candidates on real frames rather than deciding from
documentation.

### 3.6 Prompts

Separate forward passes per concept group, since detection scores are not comparable across
phrases:

- `"excavator"` / `"digger"` → one merged machine instance (required)
- `"dump truck"` / `"truck"` → optional, supporting evidence only

No bucket prompt and no pile prompt in the required path.

---

## 4. Scene geometry, derived per video

Everything in this section is estimated **from the video under analysis**, in a first sweep over the
cached masks. No constants tied to any particular scene.

### 4.1 Rotation centre `C_house`

The camera is fixed and the machine mostly does not travel, so accumulate a per-pixel **occupancy
map** — the fraction of frames in which each pixel belongs to the excavator mask — and take the
centroid of the high-occupancy region. The persistent core of the mask is the undercarriage and
house; the arm sweeps and is therefore low-occupancy.

Recomputed in overlapping windows so a machine that repositions is handled; a large shift between
windows is logged as a `machine_moved` anomaly.

### 4.2 Scale unit `L`

```
L = 95th percentile over time of  ‖ p_tip(t) − C_house ‖      (pixels)
```

This is the machine's working reach in this video's pixels. **Every distance threshold downstream is
expressed in units of `L`, every speed in `L` per second.** That single normalization is the entire
scale-invariance story: a hidden video shot closer or farther produces a different `L` and identical
dimensionless features.

### 4.3 End-effector `p_tip` and bucket region `B`

`p_tip` is the point of the excavator mask farthest from `C_house` *measured along the mask*
(geodesic, not straight-line — it matters when the arm folds back over the house). A constant-velocity
filter with outlier gating rejects single-frame jumps to the counterweight rather than integrating
them.

`B`, the bucket region, is the part of the mask within a fixed fraction of `L` of the tip, with
**motion gating**: pixels that are static over a short window relative to a temporal-median
background are removed. That is the cheap cure for the mask leaking into the soil the bucket is
sitting in.

### 4.4 Dig and dump zones, and the return direction

Take every sample where the tip is moving slowly — the dwell samples — and cluster their positions
into two groups. Loading work is strongly bimodal: the bucket lingers where it digs and where it
dumps, and moves briskly in between. The lower-elevation cluster is the **dig** zone; the other is
the **dump** zone. Membership is soft, giving two values in [0, 1] per frame.

From the two cluster centres comes one more scalar: **the direction of the return swing**. This
replaces any assumption about which side the truck is on, and makes the pipeline **mirror-invariant**
at no cost — a hidden video with the opposite layout works unchanged.

### 4.5 Material surface height `h_surf`

Restrict to samples inside the dig zone and take the distribution of bucket-bottom elevation. It is
genuinely bimodal there — the bucket is either below the surface (digging) or above it (approaching
and lifting) — so a parameter-free threshold (Otsu's method) splits it automatically.

Cross-checked against a second, independent estimate from the terrain silhouette in a temporal-median
background image. The discrepancy between the two is logged, because this scalar is unusually
load-bearing (§9.3).

---

## 5. Feature table

One row per processed sample, all derived from masks and optical flow.

| Symbol | Meaning | Units | Role |
|---|---|---|---|
| `θ` | arm bearing: angle of `p_tip` about `C_house`, unwrapped | rad | the swing signal |
| `ω` | slew rate, `dθ/dt` | rad/s | **its sign separates hauling from swinging** |
| `r` | arm extension, `‖tip − C_house‖ / L` | — | reach |
| `h` | bucket-bottom elevation relative to `C_house`, divided by `L` | — | digging / hauling boundaries |
| `v` | tip speed | `L`/s | dwell vs transit |
| `φ` | **curl angle**: bucket major axis relative to the forearm | rad | **the primary dumping signal** |
| `ω_b` | curl rate, `dφ/dt` | rad/s | tipping / uncurling |
| `a_b` | bucket silhouette area / `L²` | — | drops when buried or when load leaves |
| `g_dig`, `g_dump` | soft zone membership | [0,1] | scene context |
| `f_down` | downward optical flow just below the bucket | — | falling material (corroborating only) |
| `s_truck` | bucket / truck-bed overlap; 0 when no truck | [0,1] | supporting only |
| `ω_cab` | house rotation from sparse optical flow | rad/s | independent swing estimate when the arm is occluded |
| `q` | per-sample confidence (mask validity, filter innovation, anchor agreement) | [0,1] | gates everything |

### 5.1 Two conventions that keep the pipeline legal

**Thresholds are formulas, never constants.** For any signal `s`, define a robust per-video scale —
for example the 95th percentile of `|s|` over that video's valid samples — and write every threshold
as a small dimensionless multiple of it. The multiplier is fixed across all videos and chosen once;
the scale adapts per video. This is precisely the line between *globally tuned* (allowed) and
*video-specific* (banned).

**Units are seconds and `L`, never frames and pixels.** Frame rate appears in exactly one place: the
conversion from the video's real timestamps to the processing stride. A hidden video at 25 fps then
needs no special handling.

### 5.2 Smoothing must be zero-phase

All derivatives come from a symmetric, zero-lag filter (Savitzky–Golay, or forward-backward
filtering). A one-sided moving average or exponential filter shifts signals later in time by an
amount that depends on the signal's shape — which injects exactly the differential bias the error
budget cannot absorb (§1.3). This constraint is worth a comment in the code so nobody later
"optimizes" it into a causal filter.

---

## 6. Frame rates

Four rates, deliberately decoupled:

| Rate | Value | Why |
|---|---|---|
| Decode | source fps | needed for rendering; cheap |
| Detection anchors | ~1 Hz | drift correction, plus a free independent QA signal |
| **Track + features + FSM** | **~10 Hz** | see below |
| Render | source fps, overlays interpolated | smooth output for the reviewer |

**Why 10 Hz.** The binding constraint is not the shortest *phase* but the fastest *event*: the
bucket's uncurl at the start of dumping is a transient of a few tenths of a second, and it triggers
the most fragile boundary. Five or more samples across it means roughly 10 Hz or better. At 5 Hz
that event gets two or three samples and its onset estimate degrades badly.

Boundary quantization is *not* the binding constraint, because pass 2 interpolates the crossing
between samples (§7.4), so effective resolution is well below one sample interval.

Meanwhile the shortest phase (dumping, on the order of 2 s in published work) still gets ~20
samples at 10 Hz, comfortable for both the sustained-evidence window and the smoothing kernel.

**Use the video's real presentation timestamps**, not `frame_index / 30`. A 29.97 vs 30 mismatch
drifts about 0.1%, which over several minutes is a meaningful fraction of the tolerance, purely from
arithmetic.

---

## 7. The state machine

A **finite state machine (FSM)** is a program that remembers which phase it is currently in and, at
each step, checks only for the one change that could legally come next.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   ┌─────────┐     ┌─────────┐     ┌─────────┐   ┌──────────┐
   │ DIGGING │────▶│ HAULING │────▶│ DUMPING │──▶│ SWINGING │
   └─────────┘     └─────────┘     └─────────┘   └──────────┘
        ▲                                              │
        └──────────────────────────────────────────────┘
```

**Why this rather than labeling each frame.** A per-frame classifier makes four-way guesses that
flicker at the boundaries; the published work on this problem then bolts sequence logic on top
(hidden Markov models, LSTMs, hand-built sequential patterns) to clean it up. Building the order in
from the start gets that for free: while hauling, three of the four possible answers are impossible
by construction, and the only question is "has dumping started?"

**A phase is an interval, not a label.** Digging is the span between the swing→dig boundary and the
dig→haul boundary. The FSM never classifies frames; it locates edges. That is deliberate: your grade
depends on *when* a phase started, and an edge — a reversal, a crossing — happens at an instant,
while region classification is inherently fuzzy exactly at the borders where accuracy is needed.

### 7.1 Transition predicates and onset definitions

`τ(s)` denotes a per-video threshold on signal `s` as defined in §5.1. Low-confidence samples
(`q` below a floor) are treated as **missing, not negative**: they neither fire nor block a
transition, and short gaps are bridged.

| # | Transition | Pass-1 evidence (all must hold, sustained) | Pass-2 onset = |
|---|---|---|---|
| **T1** | swing → dig | in the dig zone **and** bucket at or below `h_surf` **and** tip decelerating **and** (curling **or** bucket area collapsing) | the **downward crossing** of `h = h_surf` |
| **T2** | dig → haul | bucket bottom above `h_surf` by a margin **and** rising | the **upward crossing** of `h = h_surf` |
| **T3** | haul → dump | uncurl rate exceeds `τ(ω_b)` **and** *not* in the dig zone **and** above `h_surf` **and** (in the dump zone **or** over a truck **or** arm extended) | the **curl-angle extremum** immediately before the uncurl (zero crossing of `ω_b`) |
| **T4** | dump → swing | slew rate exceeds `τ(ω)` **and** its sign matches the return direction | the **bearing extremum** on the dump side (zero crossing of `ω`) |

Notes on the two that are easy to get wrong:

- **T2 uses the bucket's lowest point, not its centre.** The spec says the *entire* bucket clears
  the surface. (A common alternative — taking the deepest point of the scoop — is a different
  event, typically one to two seconds earlier, and would shorten every digging phase and lengthen
  every hauling phase.)
- **T3 carries an anti-spillage guard.** The spec states that material spilling during lift or
  transport is still hauling, so falling material can never trigger dumping on its own; the bucket's
  own rotation is required, and the tip must be out of the dig zone and above the surface. Truck
  overlap is a bonus term whose weight is zero when no truck track exists, **so dumping onto a spoil
  pile works identically.**

Three of the four onsets are extrema of a smooth signal and one is a level crossing. That is a
useful property: a symmetric smoothing kernel does not move an extremum, so these are bias-free
estimators.

### 7.2 Sustained evidence (pass 1)

A transition is accepted only after its evidence holds for a short window — on the order of 0.4 s,
a modest fraction of the shortest phase. **The same window is used for all four transitions**, so
any residual confirmation lag is common-mode and cancels in phase durations (§1.3). Pass 2 then
removes it anyway.

### 7.3 Minimum phase duration — deliberately not fixed here

No constant. The rule is derived from the video's own measured distribution of phase durations, and
it **flags** a suspiciously short phase in the anomaly log rather than **blocking** transition
detection.

Rationale: a hard lockout (say, ignoring all triggers for the first 1.5 s of a phase) can swallow a
genuinely short dumping phase — individual dumps of about 1.9 s appear in the published data — and
it delays triggers in one direction only, which is exactly the differential bias the error budget
cannot absorb. **Detect first, judge second.** The exact rule is fixed once real distributions exist.

### 7.4 Pass 2: locating the onset

Given a bracketed window from pass 1, extend it to a sensible search range (roughly from the middle
of the previous phase to the middle of the next), find the relevant turning point or level crossing
inside it, and interpolate between adjacent samples for sub-sample precision. Clamp the result so it
cannot cross the previous boundary.

Bracketed extremum search is preferred over walking backward to a baseline: it does not depend on
where a baseline sits, and it is robust when the signal never fully settles.

### 7.5 Cold start, re-digs, and the anomaly log

**Initialization.** Do not initialize at frame 0 and do not try to classify the opening phase.
Cycles are defined from one digging start to the next, and the spec says to ignore the partial cycle
at the beginning — so scan for the **first high-confidence dig contact** and start there. Whatever
phase the video opens in never needs to be known.

*Guard:* dig onsets should be roughly evenly spaced. A gap near twice the typical spacing means one
was missed, which is logged rather than silently accepted, since `cycle_count` is the field
everything else depends on.

**Re-dig revert.** If, during hauling and before any dumping, the bucket returns to the material,
the dig→haul boundary is withdrawn, the state returns to digging, and the two digging segments
merge — a bucket going back into the material was still digging. Without this, a double scoop would
produce a spurious extra phase and invalidate an otherwise good cycle.

**Anomaly log.** Every unusual event is recorded with a timestamp, the active state, the evidence
values, and what the pipeline did about it: evidence for an illegal transition (ignored), re-dig
applied, phase implausibly short, detection gap, incomplete cycle, cycle-count cross-check
mismatch, machine moved, camera motion, surface-estimate disagreement.

### 7.6 Two audits, and the upgrade path

**Stage-plausibility audit.** The FSM detects edges, not stages, so nothing in it verifies that the
frames between two boundaries actually look like the phase they were assigned. A cheap check closes
that gap: score each emitted segment against its own phase's feature signature — digging is in the
dig zone with low elevation and a static bearing; hauling has rising elevation, bearing moving toward
the dump side, curl closed; dumping is in the dump zone with the curl opening; swinging has the
bearing moving back with the curl closed. These are rules over features already computed, not a
trained model. The audit **never drives** the FSM. A poor score flags the cycle, turning a silent
mislabel into a visible one.

**Periodicity cross-check.** Loading cycles are strongly periodic, so the dominant period of the arm
bearing gives a cycle count that is completely independent of the FSM. A disagreement is logged
loudly. It is *not* used to silently override the FSM — a hidden arbitration rule would be both a
debugging nightmare and a reviewer red flag.

**Upgrade path.** Sustained evidence plus onset search is a greedy approximation to a hidden Markov
model over the four states with only cyclic transitions permitted. The smoothing slot (stage E) is
kept as a seam so a global decode — which can revise an early decision in light of later evidence,
and can accept a briefly-visible phase rather than stretching its neighbour — can replace the
debounce layer behind the same interface if the anomaly log shows frequent misses.

---

## 8. Cycles and statistics

```
cycle = one digging onset → the next digging onset

complete  ⟺  it contains digging, hauling, dumping and swinging, in that order,
             each with a plausible duration, at acceptable mean confidence
```

One rule, no special cases. Footage before the first digging onset and after the last complete cycle
falls outside every cycle and is ignored automatically — which is the spec's "ignore the incomplete
cycle at the beginning or end" without needing a separate code path for it. A mid-video cycle missing
a phase fails the same rule and is excluded from **both** the count and the averages, with a logged
reason.

Excluding a whole cycle is preferred to patching one. (A published approach deletes any detected
segment shorter than ~0.4 s before averaging, which silently removes *time* and leaves the cycle
short; excluding the cycle changes nothing silently and is reportable: "5 cycles found, 4 complete,
1 excluded — no dumping detected".)

Averages are computed over exactly the valid set, and the consistency assertion of §1.4 is checked
before writing `answer.json`.

---

## 9. Perception QA gate

Every stage after perception assumes the masks are right, so that assumption is measured explicitly.
The gate emits artifacts unconditionally and **never aborts the run** — the hidden videos must always
produce an `answer.json`.

### 9.1 Metrics

Detection coverage and longest gap · agreement between the independent 1 Hz anchors and the
propagated masks · implausible tip jumps · mask area stability and jump size · frame-to-frame mask
overlap · identity switches · background motion (camera drift) · **cyclicity**: the strength of the
dominant period in the arm bearing.

Two deserve emphasis. **Anchor agreement is the real drift detector**: because detection re-runs
independently every second, there is a free second opinion on the track once per second;
disagreement triggers re-anchoring, and persistent disagreement fails the gate. **Cyclicity is a
perception check disguised as a downstream one**: if the bearing has no dominant period, the masks
are wrong, and this catches failures that per-frame metrics miss.

Each metric has a pass / warn / fail band; on failure the default behaviour is to retry with the
alternate detector and keep whichever scores better, recording both.

### 9.2 Artifacts, always written

A QA report (metrics, status, model identities and revisions, package versions), a diagnostic
overlay video showing raw anchors against propagated masks, feature time-series plots with phase
bands and boundary markers, a zone-and-trajectory plot, the feature table, the anomaly log, and the
phase segmentation.

### 9.3 The surface-height sensitivity, quantified

T1 is a *downward* crossing of `h_surf` and T2 an *upward* crossing of the same level, so an error
in that one scalar moves the two boundaries in **opposite** directions: digging shrinks while hauling
grows, or vice versa. A modest misestimate can consume a large share of the tolerance in two fields
at once.

Rather than hoping, the pipeline **measures it**: re-run the (cached, sub-second) FSM with `h_surf`
perturbed slightly in each direction and report how far the answer moved. Knowing the number beats
assuming it is small.

---

## 10. Annotation renderer

The annotated video is both a graded deliverable and the main debugging instrument, so it is built
early and serves both audiences.

**Main canvas.** Excavator mask tinted by the current phase, so the phase is readable at a glance
without reading text; bucket region outlined; tip and rotation-centre markers; the arm line; a faint
circle of radius `L` so the scale unit is visible and auditable; the two zone ellipses labelled DIG
and DUMP; **the `h_surf` line drawn across the dig zone**, which lets a reviewer verify the surface
estimate by eye; the truck box in a dashed "supporting evidence only" style when present.

**HUD.** Current phase, phase elapsed time, cycle index, cycle elapsed time, completed-cycle count,
duration of the last completed cycle.

**QA badges.** Rolling detection coverage, anchor agreement and confidence, turning red when a gate
metric is in its fail band — this is what makes the deliverable double as the QA artifact.

**Signal strip.** A pre-rendered full-width plot of the key signals with phase bands and boundary
markers, blitted as a moving window under a playhead. Cheap, and it shows *why* each boundary landed
where it did.

**Timeline ribbon.** The whole video's segmentation compressed to the frame width with a playhead, so
the entire answer is visible in any single frame.

All overlays are generated from the pipeline's actual detections; nothing is corrected after
inference.

---

## 11. Repository layout, reproducibility and compliance

```
run.py                   thin CLI entry point
pyproject.toml  uv.lock  environment, locked
answer.json              result for the provided video
configs/default.yaml     every dimensionless constant, in one place
docs/pipeline-design.md  this document
docs/stages/             per-stage specs, written just before each stage is built
labels/                  hand-labeled boundaries — evaluation only, never used by run.py
src/
  io_video.py            timestamp-accurate frame iterator
  detect/                one uniform Detection type; model choice is a one-line swap
  geometry.py            rotation centre, scale, tip, bucket region, zones, surface height
  features.py            the feature table and the threshold convention
  fsm.py                 both passes, revert rule, anomaly log
  cycles.py              validity rule and statistics
  qa.py  render.py  cache.py
scripts/                 eval against hand labels, threshold sweep, robustness checks
tests/
```

**CLI contract.** `uv run run.py --video <path>`; everything else optional (output directory,
device, detector choice, processing rate, render scale, cache directory, config). Auto-detects frame
rate and resolution. Falls back to CPU with a loud warning rather than crashing. **Always writes
`answer.json`**; QA failure sets a status field and fills the anomaly log rather than raising.

**Caching.** Keyed by video content hash, the perception-relevant config, and the model identity and
revision. Masks are stored run-length encoded (megabytes, not gigabytes). This is what makes state
machine iteration sub-second.

**Reproducibility.** Model weights pinned by revision, not just name; a prefetch command for offline
runs; seeds fixed; full-precision inference by default, since reduced precision perturbs mask
boundaries by a pixel or two and mask boundaries feed the surface estimate. Run metadata (git
revision, config, versions, device, wall time) written alongside every result. Honest caveat for the
README: exact bit-for-bit reproduction across different GPU models is not guaranteed; run-to-run
determinism on one machine is, and the sensitivity sweep (§12) shows why a pixel of mask jitter does
not move the answer.

**Licensing.** The default path is Apache-2.0 end to end. AGPL-licensed and gated-weight options are
opt-in extras, stated plainly in the README.

**Compliance with the banned list**, as tests rather than assertions in prose:

- no video-specific values: the test suite greps the source for the development video's filename,
  its resolution and duration, and for any float suspiciously close to the known answers
- `answer.json` is never read: run the pipeline with it replaced by garbage and confirm identical
  output
- no vision-language model is a dependency at all — grep-checkable
- hand-labeled boundaries live in `labels/`, are used only by the evaluation script, and the README
  states exactly what was labeled, by whom, and why

---

## 12. Risks and mitigations

| Risk | Mechanism | Mitigation |
|---|---|---|
| **Bucket occluded by the cab mid-swing** | the farthest-point tip jumps to the counterweight | outlier-gated tracking filter; fall back to mask orientation and to cab rotation from optical flow; mark the sample low-confidence so it is *missing*, not *negative* |
| **Bucket buried in the material** | bucket region collapses, curl angle meaningless | only happens during digging, where the FSM is already committed and the exit test is elevation rising — observable on emergence. The area collapse is itself a positive digging cue |
| **Mask merges bucket with soil** | area spike, centroid shift | area-jump QA metric triggers re-anchoring; bucket region capped near the tip; motion gating removes static terrain pixels |
| **No truck (spoil-pile dump)** | truck overlap identically zero | by construction: dumping is triggered by the curl reversal; truck overlap is a zero-weight bonus; the dump zone comes from the trajectory, not from a truck |
| **Hidden-video differences** — zoom, angle, mirrored layout, frame rate, extra machines, camera drift | fixed pixel or frame constants break | every threshold in `L` units, seconds, or a per-video percentile; return direction derived per video ⇒ mirror-invariant; multiple machines resolved by picking the most periodic track and logging the choice; background stabilization when camera drift is detected |
| **Differential bias eating the ±0.6 s budget** | one transition lagging differently from another | uniform evidence window; onsets at extrema and level crossings; zero-phase smoothing; measure the **mean signed error per transition** against hand labels and fix any biased transition by changing its *onset definition globally* — **never** by adding a per-video offset |

### 12.1 Fragility ranking — where the effort goes

1. **Dumping.** The shortest phase, so ±0.6 s is the tightest relative tolerance, and it depends on
   the curl angle, the noisiest feature. Both of its boundaries are hard.
2. **Digging and hauling.** Jointly sensitive to the surface-height estimate, in opposite directions.
3. **Swinging.** Driven by bearing extrema, the most robust signal available.
4. **`cycle_count`.** Not fragile in a continuous sense, but catastrophic when wrong — guarded by the
   periodicity cross-check.

### 12.2 Two validation exercises worth more than any tuning

**Threshold sensitivity sweep.** Vary each dimensionless constant over a wide range and plot all five
output fields. The goal is a **plateau, not a knife-edge optimum**. A plateau is the only real
evidence that the constants transfer to videos we cannot see; a cliff means that predicate needs
redesigning, not retuning. This plot belongs in the README.

**Synthetic hidden videos.** Re-run the whole pipeline on the development video horizontally
flipped (tests the derived return direction), downscaled (tests `L` normalization), resampled to a
different frame rate (tests the seconds-not-frames discipline), cropped and zoomed, and
brightness-shifted. All five fields must stay within ±0.6 s of the unmodified run. This is the
closest available proxy to the actual grading condition.

---

## 13. Execution environment

NCSA Illinois Computes Research Notebooks (`jupyter.ncsa.illinois.edu`), VS Code environment.
Target profile: **A100 80 GB, 2 CPU, 8 GB RAM**.

The GPU is far beyond what this needs — the two models together use well under 10 GB of the 80. The
binding constraints are **2 CPUs and 8 GB of system RAM**, which shapes three design choices rather
than being a mere footnote:

- frames are **streamed, never accumulated in memory**
- masks are stored compressed
- the annotated render is CPU-bound H.264 encoding, likely the slowest stage on two cores, so a
  reduced render scale is supported for iteration and full resolution reserved for the deliverable

Estimated cold run: perception 2–4 min, everything downstream seconds, render 3–6 min, so roughly
**5–10 minutes end to end**, with cached state-machine iterations in seconds. *These are estimates,
not measurements; the first perception run replaces them.*

The H200 profile (10 CPU / 32 GB) is a fallback purely for its CPUs if encoding proves painful.

**To verify:** session idle and wall-clock limits, and whether the home directory persists between
sessions — the perception cache must survive a session ending, or be written somewhere that does.

The README will state that development happened on an A100 while the pipeline still runs on CPU with
a warning, since the reviewer's hardware is unknown.

---

## 14. Open questions

**Blocking**

- The input video has not been inspected: length, frame rate, resolution, cycle count, camera angle,
  and whether a truck is present are all unknown. Several design choices are contingent on it.
- **Is the open-vocabulary detector reliable on this footage?** Published zero-shot numbers are on
  general-object benchmarks, not construction machinery. This is the assumption most worth
  falsifying early and cheaply; if it fails, §3 changes.
- NCSA session limits and cache persistence (§13).

**For the task owner**

- Does their verified answer exclude **mid-video** incomplete cycles, or only those at the start and
  end? (This document assumes the former, since a cycle without all four phases is not complete.)
- Do **long idle waits** — waiting for a truck to be swapped — count inside swinging? The spec's
  wording implies yes, and a single long pause could shift the swinging average well past tolerance.
- **How will the pipeline be run for grading?** GPU available? Network access for model weight
  download, or must it work offline? This quietly determines whether large or gated weights are
  viable at all.

**Deferred by design, with a scheduled answer**

- the minimum-phase-duration rule → after the first measured duration distribution
- the dimensionless threshold constants → after the sensitivity sweep
- the processing rate (10 Hz proposed) → after observing the real uncurl transient
- which surface-height estimator wins → after comparing both on real data
- whether a bucket detector is needed to refine the geometric tip → only if the tip proves unreliable

**Unverified claims from the literature and model documentation**

- SAM 3's per-frame throughput in *video* mode is unpublished; only the image-predictor figure is
  documented.
- Reported detector and segmenter speeds come from vendor documentation on high-end hardware and
  have not been reproduced here.
- Phase durations referenced for sanity (dumping ≈ 2 s, cycle ≈ 13 s) come from Cheng et al. (2023)
  on different footage, under their own labeling conventions. They are context for judging
  plausibility, never values to encode.

---

## References

1. Cheng, M-Y., Cao, M-T., Nuralim, C. K. (2023). Computer vision-based deep learning for supervising
   excavator operations and measuring real-time earthwork productivity. *The Journal of
   Supercomputing*, 79(4), 4468–4492.
2. Molaei, A., Kolu, A., Lahtinen, K., Geimer, M. (2023). Automatic estimation of excavator actual and
   relative cycle times in loading operations. *Automation in Construction*, 156, 105080.
3. Kim, J., Chi, S., Seo, J. (2018). Interaction analysis for vision-based activity identification of
   earthmoving excavators and dump trucks. *Automation in Construction*, 87, 297–308.
4. Roberts, D., Golparvar-Fard, M. (2019). End-to-end vision-based detection, tracking and activity
   analysis of earthmoving equipment filmed at ground level. *Automation in Construction*, 105,
   102811.
5. Wu, et al. (2021). Construction of stretching-bending sequential pattern to recognize work cycles
   for earthmoving excavators from long video sequences. *Sensors*, 21(2).
