# Where things live

Written 2026-09-25, because the project's code was spread across four branches,
an untracked scratchpad and a purgeable temp directory, and the most valuable
part of it was in the least durable place.

**Read this before refactoring anything.**

## The naming scheme

Status is carried in the name, so nothing has to be remembered:

| Prefix | Means |
|---|---|
| `deprecated/…` (branch) | Superseded. Never build on it. Kept so deletions stay recoverable by name. |
| `reference/…` (directory) | Works, and is the thing to port **from**. Never imported by `src/`. |
| `src/…` | Live production code. If it is here, it runs. |
| gitignored | Regenerable. Losing it costs a re-run, nothing more. |

---

## The one-paragraph history

Perception was built first and works. Measurement was then attempted three times:
from a fitted arm chain (v1), from dense optical flow (v2), and from bounding
boxes (v3). Only v3 ever detected all four transitions inside tolerance — and it
was never committed, so the repository looks like a project with no working
measurement stage. It isn't; it is a project whose working measurement stage lived
in a scratchpad.

---

## Branches

| Branch | Holds | Status |
|---|---|---|
| **`minimal-bbox-pipeline`** | the rebuild | **Current work branch.** Everything happens here. |
| `main` | perception + v1 chain features + flow | Merge target. 16 commits behind; lacks `seeding.py`, `onsets.py`. Not the newest anything. |
| `deprecated/kinematic-signals` | geodesic seeding, SAM2 bucket-as-object-2, `signals.py`, `onsets.py` | The base this branch was cut from. Its content is already in the rebuild. |
| `deprecated/optical-flow` | flow + chain fit + YOLO-World, pinned | **Everything deleted during the rebuild is recoverable here by name.** |
| `deprecated/yoloworld-probe` | identical commit to `deprecated/optical-flow` | Redundant pointer; safe to delete. |

`origin/main` and `origin/kinematic-signals` are both behind their local
counterparts. Nothing has been pushed during the rebuild.

---

## The three generations of measurement

They are **chronological, not contradictory**. Each replaced the last for a stated
reason. Knowing which generation a file belongs to is the single most useful thing
when reading this repo.

| Gen | Basis | Lives in | Why it was replaced |
|---|---|---|---|
| **v1** | fitted 3-link arm chain — curl angle, bearing, elevation | `features.py`, `filtering.py`, `kinematics.py` chain fit, `pipeline-design.md` §7.1 | The chain fit is invalid: "rigid" links varying 5.2–8.2× in length, joints jumping 0.466 L in 0.1 s |
| **v2** | dense optical flow — `omega_house`, `delta_omega` | `motion.py`, `signals.py` | The house rotates about a **vertical** axis, so in projection it translates rather than rotates. `omega_house` correlated with the real slew at **r = 0.025** — chance |
| **v3** | bounding-box position and velocity | `reference/box-physics/`, **uncommitted until now** | Not replaced. **This is the one that works.** |

---

## Directories

| Path | Status |
|---|---|
| `src/excavator_cycles/` | Production. Mixed v1 and v2; the rebuild strips v2 and replaces v1. |
| `reference/box-physics/` | **Recovered v3. Verified working** — re-run on recovery. Port **from** it; never import it. Its README lists the hardcoding that blocks direct reuse. |
| `recovered-scratchpad/` | 179 MB of media from the same session. **Gitignored**, regenerable by re-running the scripts. |
| `eval/` | Ground-truth labels + scorer. **Walled off**: nothing in `src/` may import or read it, and `tests/test_eval_labels.py` enforces that by grepping the package. |
| `docs/stages/` | Findings per stage, including the negative results. Keep. |
| `docs/evidence/` | ~20 MB of screenshots — 83 % of the repo's tracked bytes. |
| `outputs/`, `CACHE/` | Cached runs. `CACHE/` is tracked and shouldn't be; no code hardcodes its path, so it can't corrupt a run. |

---

## Artifacts that no committed code can regenerate

This is the trap that made the project look further along than it was, and then
less far along than it was. Both readings were wrong for the same reason.

- `outputs/track/dual/*/signals.npz` — `signals.build_signals` has **no caller**
- `outputs/track/real/features.csv`, `motion.csv` — from CLI commands that exist
- `reference/box-physics/boxfeat.npz`, `sig4.npz` — now committed alongside the
  scripts that made them

**Rule going forward: if an artifact is worth keeping, the code that produced it
is committed in the same commit.**

---

## What is production, prior art, archive, and dead

**Production** (survives the rebuild): `track.py`, `seeding.py`, `masks.py`,
`video.py`, `geometry.py` (scene landmarks), `kinematics.body_core`/`boom_base`/
`geodesic_distance`, `onsets.motion_boundary`/`noise_scale`/`smooth`/`derivative`,
`render.py`, `config.py`, `cabin.py`, `detect/`.

**Prior art** (reference, not imported): `reference/box-physics/`.

**Archive** (deleted here, recoverable from `archive/optical-flow`): `motion.py`,
`signals.py`'s flow channels, `features.falling_material`, the chain fit in
`kinematics.py`, `filtering.py`, YOLO-World.

**Dead** (delete outright): the abandoned `onsets` path (`rest_boundary`,
`rest_level`, `strongest_interval`), ten zero-reference symbols, orphaned config
keys, `spike.py`.

Two things must be **rescued out of `signals.py` before it is deleted**, because
they are geometric and load-bearing: `truck_overlap` (T3's location gate) and
`bucket_height` (carries T1 and T2).

---

## The target layout

```
src/excavator_cycles/
  detect/        DINO boxes               (diagram §1, unchanged)
  seeding.py     geodesic bucket seed     (diagram §1, unchanged)
  track.py       SAM2 two objects         (diagram §1, unchanged)
  masks.py  video.py  config.py  devices.py  logging_setup.py  provenance.py
  geometry.py    scene landmarks: pivot, L, zones
  cabin.py       cabin reference          (centroid of the persistent core)
  boxes.py       NEW — boxes + [i,i+10] smoothing + centres   (diagram §1.5)
  features.py    REWRITTEN — dh/dt, dx/dt, trajectories,
                 relative position, bucket-truck overlap      (diagram §2)
  onsets.py      pass-2 primitives (trimmed)
  fsm.py         NEW — pass 1 windows, pass 2 frames           (diagram §3)
  cycles.py      NEW — cycle assembly + answer.json            (diagram §4)
  render.py      annotated video: phase, duration, cycle count (diagram §4)
  plots.py       feature graphs
```

No `motion.py`. No `signals.py`. No `filtering.py`. No chain fit. No ω.

---

## Ground truth has a known flaw

`eval/labels.json` was hand-made frame by frame. Commit `3097f50` records that the
cycle's two ends use **different onset offsets** (+9 vs +22 frames after first
touch), which lengthens swinging and the cycle average by about **0.43 s** —
most of the ±0.6 s tolerance on two of the five graded fields.

So tuning to hit these labels exactly would bake that error in. Relevant when the
FSM is tuned; not before.
