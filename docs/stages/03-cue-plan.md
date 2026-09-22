# Cue plan: what evidence marks each phase boundary

Status: **plan, not implemented.**

The state machine needs, for each of the four transitions, evidence that it
happened and a time at which it began. This document fixes which signals provide
that evidence, how they are combined, and how each will be validated.

## The governing principle

> **Several cues vote on whether a transition occurred. One cue says when.**

Combining cue *timings* — averaging them, or taking the first to fire — would
mix their different systematic offsets into the reported boundary. If the mask
area drops 0.4 s after the bucket starts tipping, blending the two puts the
boundary 0.2 s late in every cycle, and a consistent offset is exactly what the
±0.6 s tolerance cannot absorb.

So each transition has:

* a **clock cue** — the sharpest, whose crossing or extremum defines the time;
* **corroborating cues** — which raise or lower confidence and can veto, but
  never move the boundary.

## Independence matters more than count

Three cues that all derive from the same mask are not three opinions. If the
mask is wrong, they are wrong together, and their agreement is worthless.

| Family | Derived from | Fails when |
|---|---|---|
| **A. Mask geometry** | mask shape: elevation, bearing, extension, curl | the mask is wrong |
| **B. Mask statistics** | mask area | the mask is wrong (but differently — area survives shape errors) |
| **C. Pixel motion** | optical flow, independent of any mask | texture is poor, camera moves |
| **D. Scene context** | dig/dump zone membership | the zones are misplaced |

Only family C is genuinely independent of perception. A transition corroborated
by A and C is far better evidenced than one corroborated by two members of A.

---

## T1 — swinging → digging

*"begins when the bucket first contacts the material and starts scooping"*

| Cue | Family | Role | Rationale |
|---|---|---|---|
| **elevation crosses the surface downward** | A | **clock** | The literal definition. A level crossing of a smooth signal, so it is sharp and its position does not move under symmetric smoothing |
| tip decelerates sharply | A | corroborate | Contact with material stops the bucket |
| mask area drops | B | corroborate | The bucket enters the pile and is occluded by soil — a real physical signature, and it survives shape errors |
| in the dig zone | D | **gate** | Prevents any firing elsewhere in the scene |

Verdict: **well evidenced.** The clock cue is precise and the corroboration comes
from two different families.

## T2 — digging → hauling

*"begins when the entire loaded bucket clears the material surface"*

| Cue | Family | Role | Rationale |
|---|---|---|---|
| **lowest bucket pixel crosses the surface upward** | A | **clock** | The definition says *entire* bucket, so the lowest point is the correct feature, not the centroid |
| elevation rate positive, sustained | A | corroborate | **Not independent** — it is the derivative of the clock cue. Records persistence, adds no new evidence |
| mask area recovers | B | corroborate | The bucket emerges from the soil that was occluding it |

Verdict: **adequate, with a caveat.** Only two independent families, and both
T1 and T2 depend on the same estimated surface height — an error in that scalar
moves them in *opposite* directions, lengthening one phase and shortening the
other. That sensitivity must be measured, not assumed.

## T3 — hauling → dumping

*"begins when the bucket reaches the dumping location and starts tipping or
uncurling"*, and material spilling during transport **remains hauling**.

| Cue | Family | Role | Rationale |
|---|---|---|---|
| **bucket rotation begins (curl rate)** | A | **clock** *(provisional)* | The definitional event. Currently unusable at 480x272 — see below |
| mask area drops as material leaves | B | corroborate | Independent of shape; the silhouette shrinks when the load goes |
| downward optical flow below the bucket | **C** | corroborate | **The only cue that touches no mask.** Genuinely independent evidence |
| in the dump zone, bearing flat | D | **gate** | Required by the spec: spillage in transit is not dumping |

Verdict: **the weakest link, and the best evidenced.** Three families available —
the most of any transition — but the clock cue is in doubt. Two things must
happen before this transition is settled:

1. **Fix the curl measurement.** The current noise is partly a bug: an axis angle
   repeats every 180° and was unwrapped as if it repeated every 360°; the measured
   region includes the stick as well as the bucket; and the angle was taken
   absolutely rather than relative to the forearm. Until those are corrected, the
   cue has not been fairly tested.
2. **If it is still too noisy**, promote a corroborating cue to clock. Optical
   flow is the better candidate, being independent of the mask — but its timing
   relative to the definitional event must be measured against labels, because
   material takes time to fall and would bias the boundary late.

## T4 — dumping → swinging

*"begins when the excavator starts rotating the emptied bucket back toward the
digging location"*

| Cue | Family | Role | Rationale |
|---|---|---|---|
| **slew rate crosses zero toward the return direction** | A | **clock** | An extremum of a smooth signal; the most robust measurement we have |
| bearing extremum | A | — | **The same event.** Not corroboration, the same measurement stated differently |
| bucket leaves the dump zone | D | corroborate | Coarse, and later than the definitional event |
| bucket re-curls | A | corroborate | Weak, and shares the curl cue's problems |

Verdict: **strong clock, thin corroboration.** Acceptable, because the clock cue
is the cleanest signal in the whole feature set — but it should be recorded that
T4 rests on a single family rather than pretending otherwise.

---

## Combination rule

```
for each transition:
    1. every cue emits candidate events with a strength
    2. a transition is ACCEPTED when
           the clock cue fires
       AND at least one corroborating cue fires within a short window
       AND every gate holds
    3. the boundary time comes from the CLOCK CUE ALONE
    4. disagreements are logged, never averaged away
```

Two deliberate consequences:

* **A clock cue firing alone is suspicious, not fatal.** It is accepted with
  reduced confidence and flagged, because the pipeline must still produce an
  answer.
* **Corroboration without the clock cue is not a transition.** Material visibly
  falling while the bucket is in the dig zone is spillage, which the spec says
  is still hauling.

## How each cue gets validated

**Without labels** — every cue must satisfy checks that follow from the physics:

| Check | Why it catches a bad cue |
|---|---|
| fires exactly once per cycle | a noise-driven cue fires many times or never |
| evenly spaced firings | erratic spacing means it is not tracking the cycle |
| consistent order across cycles | the physical order is fixed, so violations mean noise |
| repeatable position within the cycle | a real event happens at a repeatable point |
| peak stands clear of its noise floor | measures the margin, not just the firing |
| agrees with a cue from a **different family** | correlated cues agreeing proves nothing |

**With labels** — the measurement that decides the grade:

```
per transition:  mean signed error   (bias — fatal, and invisible above)
                 scatter             (noise — averages down across cycles)
```

A cue can pass every label-free check and still be 0.7 s early in every cycle.
Only labels expose that, and a pure offset is the easiest failure to fix: it
means the onset *definition* is wrong, not the detection.

## Known limitation on the current clip

The 29.6 s video appears to contain about one complete cycle. "Fires once per
cycle" and "evenly spaced" need several cycles to carry weight, so the label-free
tier will be weak until a longer clip is used. Worth stating plainly rather than
reporting checks that cannot fail.
