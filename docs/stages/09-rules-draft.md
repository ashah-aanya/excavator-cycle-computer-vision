# 09 — Hard and soft rules

**Date:** 2026-09-27 · **Status:** classification approved by Aanya ("do the hard and the soft") ·
**Measured on the 29.6 s dev clip only** (`tests/fixtures/dev_clip`, `eval/labels.json`).
The 83 s clip is not used for this work until permission is given.

## Where these rules sit

```
candidate moment (after the previous phase started)
   → 1. HARD RULES: veto. Could this phase even be starting here?
   → 2. TREND EVIDENCE: knee, big triangle, long look-back. Does it look like the moment?
   → 3. WEIGHTING (later): how much to trust each trend, from its signal's noise
   → the FIRST moment passing 1 with enough of 2 is the onset
```

A hard rule does not find the moment. It removes moments where the phase cannot be
starting. So a rule has to be **true at every real onset** (never vetoes the right
answer), and is only **useful** if it is **false during the previous phase**, where
wrong matches happen.

## What makes a rule safe to use as a veto

1. **Physically necessary**, not merely true on the clips we have.
2. **Relative, never an absolute number**: "higher than *this cycle's* dig height",
   not "height > 0.3". Needs the state machine to remember values from earlier in
   the cycle.
3. **True at the transition itself**, not only across the phase. A rule can be
   borderline in the first fraction of a second of a phase.
4. **Orientation-free**: "pile side" and "truck side" use the truck side measured
   from the video, never "left" or "right".

## The draft rules

Sources: Aanya's notes and the earlier level analysis. Every number here is from
the dev clip, which has **one cycle**. "Holds" is the share of samples where the rule
is true. "Rules out" is the share of the previous phase where it is false, meaning
the share of wrong moments it would block.

| id | during… | rule | holds: phase / first 0.6 s | rules out: prev. phase / its last 1.5 s | proposed | why |
|---|---|---|---|---|---|---|
| D1 | dig | no overlap with the truck box | 100% / 100% | 17% / 0% | **hard** (Aanya) | can't dig in the truck. Caveat: overlap is 2-D, so a pile drawn in front of the truck could overlap |
| D2 | dig | pile side of the truck | 100% / 100% | 49% / 0% | **hard** | digging happens at the pile, away from the truck |
| D3 | dig | pile side of the cabin | 100% / 100% | 60% / 0% | **soft** | true when pile and truck are on opposite sides of the machine, which is typical but not guaranteed |
| D4 | dig | below the cabin | 100% / 100% | 87% / 47% | soft | depends on camera height and the cabin estimate |
| D5 | dig | below the slew centre | 100% / 100% | 89% / 53% | soft | depends on the estimated slew centre |
| H1 | haul | higher than this cycle's dig height | 100% / 100% | 50% / 27% | **hard** (Aanya) | a loaded bucket must be lifted clear of the material |
| H2 | haul | higher in the image than this cycle's dig y | 100% / 100% | 50% / 27% | **drop** (duplicate of H1) | height IS the slew centre's y minus the bucket's y, scaled: H1 and H2 are true at exactly the same samples. H1 kept because up = bigger |
| H3 | haul | further toward the truck than during the dig | 100% / 100% | 50% / 0% | soft | assumes the swing shows up as sideways motion, which fails if the camera looks along the swing |
| H4 | haul | pile side of the truck centre | 100% / 100% | 0% / 0% | **drop** | true everywhere; blocks nothing |
| P1 | dump | truck side of the cabin | 100% / 100% | 28% / 0% | **hard** | the bucket must be over the truck; uses the measured truck side |
| P2 | dump | higher than this cycle's dig height | 100% / 100% | 0% / 0% | **hard** (Aanya) | necessary, but blocks nothing on its own after a haul; its value is stopping a haul lift being read as a dump |
| P3 | dump | above the cabin | 100% / 100% | 9% / 0% | soft | camera-dependent |
| P4 | dump | above the slew centre | 100% / 100% | 5% / 0% | soft | camera-dependent |
| P5 | dump | above the truck centre | 100% / 100% | 10% / 0% | soft | camera- and truck-box-dependent |
| P6 | dump | overlapping the truck box | 93% / 100% | 33% / 0% | soft | not true for the whole dump even here; the truck level was already too weak on another clip |
| P7 | dump | further toward the truck than the haul's median x | 100% / 100% | 51% / 13% | soft | same camera caveat as H3 |
| S1 | swing | higher than this cycle's dig height | 100% / 100% | 0% / 0% | **drop** | blocks nothing |

Every rule holds at every onset on the dev clip, including the first 0.6 s. With
one cycle that shows only that none is broken here, not that any is always true.

## What rules can and cannot do

- **They stop large errors**, meaning firings seconds away from the true onset and
  cascades like an early haul's lift being read as a dump.
- **They barely sharpen timing.** In the last 1.5 s before each onset the bucket
  already satisfies the next phase's rules (0% ruled out for most). The timing
  misses live exactly there, so fixing timing is the trend patterns' job (knee,
  big triangle, long look-back).
- Only D4/D5 (below the cabin / slew centre) bite near a boundary, ruling out about
  half of the swing's last 1.5 s before the dig. They are also the most
  camera-dependent, hence soft.

## What the pipeline needs for these

- **Memory within a cycle**: this cycle's dig height, dig y and dig x (medians from
  the detected dig start to the current sample), and the haul's median x. `walk`
  would carry them; the cues are stateless today.
- **Truck side**: already measured (`dump_side`).

## Decisions (2026-09-27)

- **Hard (veto):** D1, D2, H1, P1, P2.
- **Soft (evidence for the weighting layer, never a veto):** D3, D4, D5, H3, P3, P4,
  P5, P6, P7.
- **Dropped:** H2 (duplicate of H1), H4 and S1 (block nothing).
- D3 is soft because pile and truck are not guaranteed to be on opposite sides of
  the machine in the hidden videos. Revisit if Aanya wants it hard.
- Return swing has no hard rule yet.

## Decision: what "dump starts" means (2026-09-27)

**The dump starts when the bucket starts tipping, read from the bucket box's aspect
ratio: the start of its big drop, as the box goes from wide (curled bucket) to tall
(open bucket pointing down).** Aanya: "ignore the labels, follow the aspect ratio".

Why: the task defines dumping as beginning when the bucket "starts tipping or
uncurling to release its load". On the dev clip, the frames show the bucket still
curled at the labelled dump start (18.65 s), opening at 21.0 s and fully open at 21.7
s. The aspect ratio peaks at 19.2 s and drops fastest at 20.9 s. The labelled time
marks the bucket ARRIVING over the truck, about 2 s before it tips.

Consequence: the labelled dump times in `eval/labels.json` are not the target for
dump cue work. `eval/score.py` still grades against them, so its dumping and
hauling fields do not measure the aspect-ratio definition until the labels are
revisited.
