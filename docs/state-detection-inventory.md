# State detection: what exists today, and what survives dropping optical flow

**Written:** 2026-09-25, before any deletion, on branch `minimal-bbox-pipeline`.
**Purpose:** preserve the current state-detection logic so the no-optical-flow
rewrite disturbs as little of it as possible. Nothing here is a new design; it is
a record of what is already written down, and a judgement per trigger of whether
it survives.

---

## First, an honest statement of what "current" means

**No finite state machine exists.** There is no `fsm.py`, no cycle assembly, and
no `answer.json`. `cycle_count` appears only in `eval/score.py` and `tests/`,
never in `src/`. The CLI exposes `probe, spike, track, render, features, motion`
and none of them produces a phase.

So there is no running state detection to preserve. What exists is:

1. **Three written generations of trigger specification** (below), which disagree
   with each other;
2. **Unwired pass-2 primitives** in `onsets.py`, well tested, with no caller
   outside `tests/test_onsets.py`;
3. **Feature computations** in `features.py` that the triggers were written
   against.

Preserving (1) and (3) is the point of this document. (2) survives untouched —
it is signal-agnostic.

---

## The three generations, and why they disagree

| Gen | Lives in | Basis | Status |
|---|---|---|---|
| **v1** | `docs/pipeline-design.md` §7.1, `docs/stages/03-cue-plan.md`, `features.py` | Chain-fit geometry: curl angle, bearing, elevation | Written, features built, FSM never built |
| **v2** | `signals.py` header | Dense optical flow: `omega_house`, `omega_stick`, `omega_bucket`, `delta_omega` | Written, signals built, never wired to the CLI |
| **v3** | The design diagram of 2026-09-25 | Bounding-box position and velocity | This rebuild |

The audit read these as contradictions. They are not — they are **chronological**.
v2 replaced v1 because the chain fit was unreliable (`05-kinematic-spec.md`: link
lengths varying 5.2–8.2×, joints teleporting 0.47 L between 0.1 s samples), and
flow looked like the way out. v3 returns to geometry, but takes it from bounding
boxes rather than from a fitted chain — which sidesteps the reason v2 existed,
because a box needs no chain fit.

**Consequence for this rebuild: dropping flow means reverting to v1's trigger
shape, not inventing one.** v1 was already geometric.

---

## Trigger-by-trigger: what survives

`τ(s)` is a per-video threshold on signal `s` — a fixed multiplier of that
video's own 95th percentile, never an absolute level.

### T1 — swinging → digging

| | |
|---|---|
| **v1 pass 1** | in the dig zone **and** bucket at or below `h_surf` **and** tip decelerating **and** (curling **or** bucket area collapsing) |
| **v1 pass 2** | the downward crossing of `h = h_surf` |
| **v2** | `h` arrives at a plateau (no `h_surf` at all) |
| **v3 diagram** | no horizontal motion · vertical goes to "floor" · more significant y motion · moving toward vs. away from truck · bounds on x (can't go past the truck) |
| **Flow used?** | **No.** Every cue is position or a position derivative. |
| **Verdict** | **Survives intact.** Elevation and zone membership come from masks and boxes. |

### T2 — digging → hauling

| | |
|---|---|
| **v1 pass 1** | bucket bottom above `h_surf` by a margin **and** rising |
| **v1 pass 2** | the upward crossing of `h = h_surf` |
| **v2** | `h` departs the plateau |
| **v3 diagram** | vertical threshold crossing · enforce previous state was dig · both x and y strictly inc/dec · if trigger held long enough, dig started x ago |
| **Flow used?** | **No.** |
| **Verdict** | **Survives intact.** The strongest-specified trigger in the repo. |

**Preserve this detail:** T2 uses the bucket's **lowest** point, not its centre. *(Not yet true of the code -- `features.py` uses the box centre. See the note in `pipeline-design.md` 3.2 for why this is sequenced after the surface estimate.)*
The task spec says the *entire* bucket clears the surface. Using the deepest point
of the scoop instead is a different event, typically 1–2 s earlier, and would
shorten every digging phase while lengthening every hauling phase. Encoded at
`features.py:165-168`.

### T3 — hauling → dumping

| | |
|---|---|
| **v1 pass 1** | uncurl rate exceeds `τ(ω_b)` **and** not in the dig zone **and** above `h_surf` **and** (in the dump zone **or** over a truck **or** arm extended) |
| **v1 pass 2** | the curl-angle extremum immediately before the uncurl |
| **v2** | `delta_omega` departs zero **while over the truck** |
| **v3 diagram** | moving toward truck · cabin is to the left of the bucket · higher x and lower y than cabin · closest to the truck |
| **Flow used?** | **v2 only.** v1's `ω_b` came from the chain's curl angle; v2 replaced it with `delta_omega` from flow. |
| **Verdict** | **Rate cue lost, location gates survive.** |

The location gates — not in the dig zone, above the surface, over the truck — are
all geometric and carry straight over. `signals.truck_overlap` is the same
computation and is geometric.

**Preserve this detail — the anti-spillage guard.** The task spec states that
material spilling while the bucket is still being lifted or transported is *still
hauling*. So falling material must never trigger dumping on its own. v1 required
the bucket's own rotation plus position gates. **v3 drops the rotation term
entirely**, leaving only position — which means the guard now rests wholly on
"closest to the truck" and "higher x, lower y than cabin". Worth checking at the
FSM stage that those are sufficient, because this is the trigger the spec is
fussiest about.

### T4 — dumping → swinging

| | |
|---|---|
| **v1 pass 1** | slew rate exceeds `τ(ω)` **and** its sign matches the return direction |
| **v1 pass 2** | the bearing extremum on the dump side |
| **v2** | `omega_house` departs zero (**no sign condition** — a regression) |
| **v3 diagram** | opposite conditions to hauling |
| **Flow used?** | **v2 only.** v1's `ω` was `d(bearing)/dt`, geometric. |
| **Verdict** | **Replaced.** v2's version was measuring the wrong quantity anyway. |

`omega_house` correlated with the geometric slew rate at **r = +0.025, sign
agreement 0.50** — chance. The house sits near the rotation axis, so a
vertical-axis slew makes it *translate* horizontally rather than rotate in the
image; the in-plane cross-product estimator then returns a value set by where
pixels sit above or below the pivot row. Dropping it removes a defect.

**Preserve this detail — the direction requirement.** The task spec says swinging
begins when the machine starts rotating **back toward the digging location**. v1
required the sign to match `return_direction`; v2 dropped it, which would let T4
fire on repositioning at the truck. v3's "opposite conditions to hauling" restores
a directional sense positionally — keep it that way.

---

## What survives in code, unchanged

| Symbol | File | Why it survives |
|---|---|---|
| `motion_boundary` | `onsets.py:314` | Pass-2 onset finder. Signal-agnostic: takes any rate that rests at zero. **The design's preferred primitive** — it fixes the rest level at exactly zero rather than estimating it. |
| `noise_scale` | `onsets.py:275` | MAD of successive differences; trend-immune. Used to set the band around rest. |
| `smooth`, `derivative` | `onsets.py:423,446` | Symmetric Savitzky–Golay, window in seconds. Zero-phase, so they do not move an extremum in time. |
| `truck_overlap` | `signals.py:94` | Geometric: bucket mask ∩ truck box. T3's location gate. **Rescue this before deleting `signals.py`.** |
| `bucket_height` (`h`) | `signals.py:91` | Geometric: lowest bucket pixel relative to the pivot. Carries T1 and T2. **Also rescue.** |
| `rotation_centre` | `geometry.py:51` | The cabin reference. See `cabin.py` for why the centroid rather than a box. |
| `dwell_clusters`, `surface_height`, `return_direction` | `geometry.py:182,231,281` | Dig/dump zones, the "floor", and which way is back to the pile. All geometric. |

## What dies with optical flow

| Symbol | File | Consequence |
|---|---|---|
| `dense_flow`, `region_omega`, `build_motion_field`, `decompose` | `motion.py` (whole module) | T3's rate cue and T4 lose their v2 basis; v3 replaces both positionally |
| `omega_house`, `omega_stick`, `omega_bucket`, `delta_omega` | `signals.py:175-180` | See above |
| `falling_material` | `features.py:249` | Already measured as non-discriminating: separability 0.85, and its peak is during **hauling**, not dumping |

---

## Open question the diagram raises and does not answer

**"Vertical goes to 'floor' (need definition of floor)"** — the diagram says this
in the digging cues, in Aanya's own words. v1 defined it as `h_surf`, an Otsu
split of the bimodal elevation distribution inside the dig zone
(`geometry.surface_height:230`). That split sits *between* the "buried" and
"above" modes, i.e. below the true surface, which biases T1 late and T2 early —
shortening digging and lengthening hauling. `pipeline-design.md` §9.3 prescribes a
sensitivity sweep for exactly this and it was never run.

Left open deliberately. **Aanya directs this at the FSM stage.**

---

## Instruction of record

From Aanya, 2026-09-25: preserve how the current design uses kinematic features
for state detection, given the features are position and velocity; keep only
geometric features; **pause at the FSM stage for direction.**
