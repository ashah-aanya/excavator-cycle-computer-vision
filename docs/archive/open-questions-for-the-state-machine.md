# Open questions the state machine has to answer

Everything here is a **decision**, not a defect. The confirmed bugs from the
physics review are fixed (commit `fd1fce1`); these are the judgement calls it
surfaced, plus the two gaps the recovered detector never covered. They are
written down because each one changes what `fsm.py` should be, and deciding them
inside the implementation would bury them.

> ## STATUS, 2026-09-26: sections 1 and 2 are decided, in code
>
> Read this before the sections below, which are preserved as the record of how the
> decision was reached rather than as open questions.
>
> **1. Every cycle is measured**, not just the first. `cycles.assemble` splits on
> digging, `summarise` averages every measurable cycle, and multi-cycle needed no
> special case -- the loop simply keeps going.
>
> **2. Incomplete cycles** were resolved by a split this document did not propose:
> `cycle_count` counts cycles that OCCURRED and the averages come from cycles that
> could be MEASURED. Two populations, on purpose. A cue failing is a fact about the
> pipeline, not about the excavator, so a cycle with a missed onset is still counted
> and simply contributes no duration. `Cycle.reason` says which and why. The
> consequence -- that `cycle_count × average_cycle_duration` will not equal elapsed
> time -- is correct and is documented in `cycles.py`.
>
> A **mid-video re-dig** is handled by abandon-and-restart, not by the revert rule
> sketched in section 2: an out-of-sequence dig drops the cycle in progress and
> becomes a new boundary, and the abandoned span is then judged on its own per-span
> evidence like any other.
>
> **3 remains open**, and section 3's own answer turned out to be right in an
> unexpected way: `low_height.threshold` is NOT the material surface. It is the Otsu
> valley between the dig mode (-0.11 `L`) and the carry mode (+0.23 `L`), landing at
> 0.0942 `L` -- about 1.3 bucket-heights above where the bucket actually sits while
> digging. That single misplacement is why digging runs 1.9 s long and hauling fires
> 1.97 s late. It is the first thing to fix in the cue work.

---

## 1. Only the first cycle is measured  ★ the one that decides the answer

The recovered detector computes `T2`, `T3` and `T4` **once**. So `avg_digging`,
`avg_hauling`, `avg_dumping` and `avg_swinging` all come from cycle 1 alone.
Only `avg_cycle_duration` uses every episode, as `mean(diff(dig_starts))`.

The task asks for *"the average duration of each phase across all complete
cycles."* The development video has 1.2 cycles, so the difference is invisible
here and decisive on a hidden video with three or four.

It also extrapolates the next onset as `next_T1 = dig_start[1] + (T1 - dig_start[0])`,
which assumes the offset from a dig episode's start to its true onset is the same
in every cycle. Untestable at n = 2.

**To decide:** run all four detectors per cycle, or keep the single-cycle
shortcut and accept it. There is no reason to keep it.

## 2. Incomplete cycles

Two cases, and they land differently.

**Head and tail: already correct, for free.** A cycle is *one digging onset to
the next*, so `cycle_count = len(dig_episodes) - 1`. Footage before the first
onset and after the last falls outside every cycle with no special-case code.

**Mid-video: not handled.** A machine that digs, hauls, then re-digs without
dumping — a bucket that came up short — produces a dig→dig span that is not a
complete cycle, but `len(eps) - 1` counts it and its bogus durations enter the
averages. Re-digs are ordinary operator behaviour.

**To decide:** a cycle is complete only if all four onsets are found, in order,
within it; incomplete ones are excluded from **both** the count and every
average. `pipeline-design.md` §7.5 already specifies a re-dig revert rule.

## 3. What "the floor" is — and the answer may be "nothing"

The design diagram says *"vertical goes to 'floor' (need definition of floor)"*.

The recovered detector never defines one. Its dig bracket is `h < otsu(h)` —
fully data-driven — and T1/T2 are **departures from `h`'s own plateau** rather
than crossings of a calibrated level. That is probably the right answer, and it
removes a differential-bias source: an Otsu split of the elevation distribution
sits *between* the buried and above modes, i.e. below the true surface, which
would bias T1 late and T2 early, shortening every digging phase and lengthening
every hauling phase.

There is also a hard reason not to use an absolute level. Sweeping the occupancy
quantile 0.80–0.95 moves `pivot_y / L` by **0.098 L**, which is **22% of
`height`'s entire 0.45 L range**. So `height` must never be thresholded
absolutely; only its *derivatives* are offset-immune. Currently only `dh_dt` is
thresholded, which is correct but undocumented as a constraint.

**To decide:** confirm the plateau formulation, and write the constraint down.

## 4. `aspect_ratio` is a weak dumping cue

It is what T3 uses (`argmax` inside the truck bracket) and it scored +0.55 s
against a ±0.6 s tolerance — inside, with essentially no margin. The review
found three reasons it is soft:

- **It is symmetric about 90°.** For a rectangle at angle θ the axis-aligned
  box's aspect ratio is the same at 45° and 135°, so it cannot distinguish
  tip-forward from tip-back, and the bucket passes through `argmax` twice per
  rotation.
- **A quarter of its variance is not rotation.** On the real clip
  `corr(AR, radius) = −0.505`; the bucket box area ranges 486→2315 px across the
  clip, and an axis-aligned box's aspect ratio moves with apparent size.
- **Perspective scales it.** A 20% vertical foreshortening changes AR by +25% at
  every angle. A per-video constant, so a within-video `argmax` survives, but no
  cross-video threshold does.

**To decide:** keep `argmax(AR)` gated by the truck bracket, or find a cue with
more margin. Worth knowing that the truck-overlap gate is doing a lot of the
work here, and the overlap fix (`fd1fce1`) made it more accurate.

## 5. `d2h_dt2` is a chained first derivative

`derivative(derivative(height))` with the same window, rather than
`savgol_filter(deriv=2)` on `height` directly.

It is a **true** second derivative — exact on a quadratic, correct units, no
1/dt error. The cost is bandwidth: against a slope discontinuity, the chained
version's peak is **37% lower** and its response spans **1.30 s against 0.70 s**.
Peak *location* is preserved, because Savitzky-Golay is zero-phase, so a
symmetric kink is measured correctly — but two kinks closer than ~1.3 s would
merge, and an asymmetric one would shift.

This matters because `d2h_dt2` carries T2, which is already the most fragile
transition: across a sigma × smoothing sweep it ranged **10.30–14.80 s**, and the
prior session's own read was *"it passes at the default by luck, not by
robustness."*

**To decide:** switch to a direct `deriv=2` filter (one filter, half the edge
region, 33% less attenuation at 0.5 Hz) or keep the chain. Low risk either way,
and worth measuring rather than arguing.

## 6. Clip-edge lag, if a cycle starts near t = 0

The trailing moving average introduces a uniform 0.20 s delay at the default
window, which cancels in every duration — except in the first `width − 1`
samples, where the lag ramps 0 → 0.2 s. A T1 landing in the first 0.4 s of a clip
carries up to 0.2 s of *differential* error, a third of the tolerance budget.

On this video T1 is at 4.17 s, so it does not bite. A hidden clip that opens
mid-cycle would. Switching to `centred` alignment halves it; both are config.

## 7. Smaller items, recorded so they are not rediscovered

- **`cabin.stability()` measures the wrong box.** It reports on the core alone,
  while the feature layer uses `mask & core`. Its "use the right edge" advice
  does not transfer: re-measured on the quantity actually in use, the right edge
  moves **0.107 L** across the quantile sweep — the *worst* of the four, not the
  best. The cabin reference itself is sound (`centre_y` std 0.19 px, about 1% of
  the `rel_cabin_y` range it anchors); only the diagnostic's advice is off.
- **Three near-identical occupancy-core implementations** —
  `geometry.rotation_centre`, `cabin.stable_core`, `kinematics.body_core` —
  differing only in their zero floor. They will drift.
- **`pivot` is the body centroid, not the slew axis.** Harmless: a constant
  offset differentiates away, and `L` shares the same origin. The `radius`
  docstring has been corrected to say what it measures.

---

## What the physics review confirmed correct

Verified by running code, not by reading it: `height` and `rel_cabin_y` are
genuinely up-positive (`corr(−dy/dt, dh_dt) = 0.979`, sign agreement 98.1%);
`delta=` is passed to Savitzky-Golay properly, so rates are not scaled by 1/dt;
smoothing happens before differentiation; the time base is real decoder seconds
everywhere; windows are seconds and behave identically at 10 and 20 Hz; every
feature column except the two noted above is exactly scale-invariant at 1× and
2× pixels; `radius` has consistent units; and re-integrating `dh_dt` reproduces
`height` to 0.0126 L against a 0.45 L range.
