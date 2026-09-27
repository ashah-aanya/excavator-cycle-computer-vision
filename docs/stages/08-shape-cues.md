# 08 — Onset cues from shapes, not levels

**Date:** 2026-09-26 · **Branch:** `cues/shape-cues`

## The problem

Every onset cue was a *level*: "the bucket is down and still" (dig), "above the
material and rising" (haul), "over the bed, past the cabin" (dump), "traversing
and descending" (swing). A level is true for a whole phase, so it is poor at saying
when a phase *starts*. Measured against hand labels, the level cues matched **0 of
13** onsets on the 83 s clip and **0 of 5** on the dev clip. On the 83 s clip they
never found a dump or a swing, because the truck level scored 0.78 against the
0.80 bar and switched dump detection off.

## The idea

A phase starts at a moment, and a moment shows up as a change of **trend**. So
each onset cue reads the *shape* of a signal around a sample: fit a line to the 2 s
before and the 2 s after, and name the pair (rising, falling, flat). "Speed was
falling and has gone flat" happens where the return swing brakes onto the pile,
not for the whole dig. See `src/excavator_cycles/shapes.py`.

A shape that also happens elsewhere in the cycle is harmless, because the state
machine only asks for a phase after the previous phase has started. What matters is
whether it is the **first** such shape after the previous onset. That is the
question the evaluation checker scores (`eval/check_cues.py`).

## The four cues

| Phase | Onset cue | 83 s clip (3 cycles) | Dev clip (1) |
|---|---|---|---|
| dig | 2-D speed: drop ends → flat | 3/3 | 1/1, via the edge fallback |
| haul | height: keeps rising | 2/3, ~1.5 s early in the misses | 1/1 |
| dump | height: flat → starts rising, on the truck's side of the cabin | 3/3 end to end | ✗ 1.8 s late |
| swing | horizontal acceleration toward the truck: peak | 3/3 | 1/1 |

End to end (`eval/check_onsets.py`): **10/13** onsets within 0.6 s on the 83 s
clip, **4/5** on the dev clip. The CLI's `answer.json` on the dev clip passes **4 of
6** graded fields (`eval/score.py`), up from 1: haul and dump each miss by 1.5 s,
both caused by the late dump.

## Decisions, and why

1. **State conditions and onset cues are separate.** The old level conditions are
   kept, renamed `in_digging` etc. (`STATES`), for the two jobs that ask *whether* a
   phase happened: the out-of-sequence dig alarm and the evidence `cycles.py`
   counts cycles on. The dig onset's shape happens several times a cycle; used as
   the alarm, each would abandon the cycle in progress.

2. **Dump is gated on the truck side of the cabin.** Without it, an early haul
   cascaded: the haul's real lift off the pile is also "flat → starts rising", so it
   was taken for a dump and the rest of the cycle was wrong (cycle 3 of the 83 s
   clip). A dump can only happen over the truck. `dump_side` is read from the
   video, from the truck level *as measured*, so a weak truck level no longer
   disables dumping.

3. **The swing cue is signed toward the truck.** Raw d²x/dt² flips sign on mirrored
   footage, which turns the swing's peak into a dip and loses every swing. Multiplied
   by `dump_side`, the cue is the same event either way. Tested: the mirrored 83 s
   clip gives identical onsets.

4. **Near the clip edges, dig falls back to the digging state.** The dev clip ends
   while the bucket is still braking, so "drop ends → flat" never happens on camera.
   Without a fallback its one complete cycle was lost (`cycle_count` 1 → 0). Where
   a shape cannot be read, the dig cue uses the old level condition.

5. **The unreadable edge is a duration (0.4 s), not a sample count.** With "three
   samples", the edge was 0.6 s wide at 5 Hz and 0.12 s at 25 Hz, and the synthetic
   clip gave 3 cycles at 10 Hz but 2 at 25 Hz. As a duration, 5–30 Hz all agree.

6. **Pass 2 keeps the shape's start time for dig, haul and swing.** A shape cue
   fires where its shape starts, which is the onset itself; the old refinements
   were built for level cues and mostly found nothing. Dump keeps its refinement
   (the bucket box at its most stretched), which moved the first dump on the 83 s
   clip from +0.61 s to −0.19 s.

7. **"Above the dig level" did not fix haul.** It was tried as a gate, and made haul
   late in every cycle: at the labelled haul onset the bucket is still below the
   level `calibrate` calls "in the material".

## What is still wrong

- **Dump on the dev clip is 1.8 s late**, which costs the dev answer both the haul
  and the dump fields. This is the one window that still misses its onset
  (`test_every_window_contains_its_onset`, strict xfail).
- **Haul fires ~1.5 s early** in two of the 83 s clip's three cycles.
- **The cues were chosen by looking at the 83 s clip.** They will do worse on
  footage they have not seen. The dev clip is the only unseen check so far, and it
  holds one cycle.
- **The first dig of a clip** uses the edge fallback, so on the 83 s clip it lands
  at 0.0 s (label 0.53 s).

## Tools

- `eval/check_cues.py` — score one cue, or search every feature × shape, by "first
  match after the previous phase". Evaluation only; imports nothing from the
  pipeline.
- `eval/check_onsets.py` — run the real state machine on a clip's features and
  compare every onset with the labels.
- `eval/labels_long_clip.json` + `tests/fixtures/long_clip/` — the 83 s clip's
  labels and features.
- Gates: `tests/test_long_clip_accuracy.py` and `tests/test_onset_accuracy.py`
  ratchet every onset error so no change can make one worse unnoticed.
