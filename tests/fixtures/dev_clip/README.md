# `dev_clip` — a real feature table, committed on purpose

`features.npz` and `scene.json` are stage 3's output for the task video
(`construction_excavator_cycle_duration_1.mp4`, 296 samples at ~9.99 Hz,
spanning 0–29.53 s). They are copied verbatim from a `track` + `features` run.

## Why these are in git when `outputs/` is ignored

Every test of the state machine was synthetic. `_scripted(schedule)` hands
`walk` a table built to trigger on cue, which verifies *sequencing* — that
phases advance in order, that windows do not overlap — and cannot verify
*accuracy*, because a hand-built schedule has no truth to be wrong about.

The cost of that gap was concrete: all five pass-1 windows excluded the
labelled onset they were meant to bracket, by −1.9 s to +4.8 s against a
±0.6 s grading tolerance, and the suite could not fail on it. A diagnostic
script printed `in window? NO` five times and the branch was committed anyway.

36 KB in git buys an accuracy gate that runs in CI on every change instead of
skipping whenever a local run happens to be absent. That is the trade.

## What may read this

Tests. This is pipeline *output*, not ground truth — the labels it gets
compared against live in `eval/labels.json`, and `src/excavator_cycles/` may
reference neither (enforced by `test_pipeline_never_references_the_labels`).

## Regenerating

    run.py track <video> --out outputs/track/dual
    run.py features outputs/track/dual
    cp outputs/track/dual/{features.npz,scene.json} tests/fixtures/dev_clip/

Regenerate only alongside a deliberate change to tracking or features, and
expect the measured errors in `test_onset_accuracy.py` to move when you do.
