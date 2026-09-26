# Prior art: the box-physics detector that worked

**This is recovered exploratory code, not production code. Do not import it.**

## What it is

Scripts from the 22–25 Sep 2026 session that were never committed. They were
found in that session's scratchpad under `/private/tmp` on 25 Sep and copied here
before that directory could be purged. `boxphysics.py` was re-run on recovery and
**reproduces its result**, so this is verified working code, not a claim.

It matters because it is the first thing in this project to detect all four phase
transitions inside tolerance, and it does so **with no optical flow, from bounding
boxes** — the architecture the v2 design diagram asks for.

## The result

Re-run on 25 Sep 2026 (`PYTHONPATH=../src python boxphysics.py`):

```
digging     4.50 vs  4.17  +0.33  PASS
hauling    11.30 vs 10.74  +0.56  PASS
dumping    19.20 vs 18.65  +0.55  PASS
swinging   22.90 vs 23.05  -0.15  PASS
```

That is the 0.5 s trailing / 0.9 s Savitzky–Golay setting. The session also found
a better one — 0.3 s trailing / 0.3 s SG — reported as **6/6 graded fields**:

```
T1 +0.03   T2 -0.24   T3 +0.35   T4 -0.15
cycle 25.30 (+0.11) · dig 6.30 (-0.27) · haul 8.50 (+0.59)
dumping 3.90 (-0.50) · swinging 6.60 (+0.29)
```

## The cue set

| | Pass 1 — bracket (Otsu, lag-free) | Pass 2 — clock cue |
|---|---|---|
| T1 digging | longest interval where `h < otsu(h)`, ≥1.0 s | `dh/dt` **arrives** 0 |
| T2 hauling | `argmin h` within dig → truck start | `d²h/dt²` **arrives** 0 |
| T3 dumping | longest interval where `overlap > otsu(overlap)`, ≥0.5 s | **argmax** aspect ratio |
| T4 swinging | truck start → first `\|dx/dt\|` excursion after it | `\|dx/dt\|` **departs** 0 |

All four signals come from two bounding boxes: the bucket box (SAM object 2) and
the body box (excavator mask ∩ `body_core`), plus a static truck box.

## Why it cannot be used as-is

It is saturated with exactly what the task spec bans. Every one of these has to
be derived per video before any of this ships:

```python
F = 29.97396912419384                       # this video's frame rate
TRUCK_BOX = (234.5, 134.0, 376.5, 211.0)    # this video's truck, in pixels
n = 296                                     # this video's sample count
t = np.arange(n) * 0.1                      # assumes exactly 10 Hz
PIV, L  = scene["centre"], scene["scale"]   # read from a cached run
REPO = Path("/Users/aanyashah/Desktop/...") # absolute path
TRUTH = {"digging": 125 / F, ...}           # READS THE GROUND TRUTH
```

That last one is the important one: the script imports the answer to score
itself. Nothing in `src/` may do that — `eval/` exists precisely to keep the
labels out of the pipeline's reach.

## How strong is the result, honestly

- **n = 1.** One video, 1.2 cycles. Every average rests on a single measurement.
- **The margins are thin.** At 6/6, `hauling` had **0.01 s** of margin and
  `dumping` **0.10 s**, against a ±0.6 s tolerance.
- **The 6/6 cells were scattered, not contiguous**, across the smoothing grid —
  fields flipping across the tolerance boundary as parameters move. The session's
  own reading: *"we're at the tolerance boundary on two fields"* rather than solved.
- **T2 is fragile.** Across a sigma × smoothing sweep it ranged 10.30–14.80 s:
  *"T2 swings 4.5 seconds across parameter settings. It passes at the default by
  luck, not by robustness."*
- **T4 is the robust one.** Total spread 0.40 s across every region, sigma, hold
  and smoothing setting tried, and every one passed.

## Negative results worth not rediscovering

| Tried | Why it failed |
|---|---|
| `omega_house`, `delta_omega` (flow) | The house rotates about a **vertical** axis, which projects to a lateral shift, not image-plane rotation. Flat at ~0 in every phase. |
| Bucket mask **area** for T2 | Runs **backwards**: 1341 px buried vs 990 px carried. Dirt merges into the mask rather than occluding it. |
| Loaded-vs-empty bucket **appearance** | Fails backwards — empty reads *more* saturated and warmer. Side-on you see the bucket's shell, not its contents. |
| Truck-bed **brightness** | Dominated by the arm passing over and by shadows; biggest changes at 16.9 s and 24.7 s, not the dump. |
| **Falling material** | Peaks during *hauling* (spillage), which the spec says is still hauling. Separability 0.85. |
| `minAreaRect` **rotated angle** for T3 | Flips 90° whenever width and height cross over; the bucket is near-square. |
| Fitted 3-link **arm chain** | "Rigid" links varying 5.2–8.2× in length; joints jumping 0.466 L in 0.1 s. |
| `d(bearing)/dt` for T4 | +2.45 s. A rigid body rotating about a fixed axis has a stationary centroid. |

## Two cross-cutting lessons

1. **"Rest for a rate is zero, absolutely."** Estimating the rest level for a rate
   made the estimator settle on the *hauling* height instead of the dig plateau.
   Fixing rest at 0 and estimating only the band width took the project from 0/4
   to 3/4 in one change.
2. **Errors compound into durations, and signs matter more than magnitudes.**
   `hauling = T3 − T2 = (+0.55) − (+0.56) = −0.01` — same-signed, they cancel.
   `dumping = T4 − T3 = (−0.15) − (+0.55) = −0.70` — opposite-signed, they add.
   So `avg_dumping` failed *because T4 was too accurate*. The fix is to make all
   four estimators the same kind so their biases cancel in every difference — not
   to nudge one by a constant, which is fitting to one video.

## Files

`boxphysics.py` is the one that matters — features, graphs and a marked video from
box centres. `final_video.py` and `deliverable.py` are renderers. `boxfeat.npz` and
`sig4.npz` are cached feature arrays. The `flow_*`, `omega_rel`, `vrel` and `probe*`
files are the optical-flow investigations that produced the negative results above.

Media from the same session (179 MB of mp4/jpg) is **not** committed; it stays in
`recovered-scratchpad/`, which is gitignored, and is regenerable by re-running
these scripts.
