# `long_clip`: the 83 s clip's feature table

`features.npz` is stage 3's output for the **83 s** video that shares its filename
with the dev clip (`construction_excavator_cycle_duration_1.mp4`, sha256
`e80d63cd…b077`). It has 416 samples at 5 Hz, spanning 0–83.03 s. It was produced by a
`track` + `features` run on the cluster and copied verbatim.

## Why it is here

It holds three complete cycles, against the dev clip's one. `eval/check_cues.py`
scores candidate cues against it and against `eval/labels_long_clip.json`, and
`tests/test_check_cues.py` pins the checker's results on it.

## What may read this

Tests and `eval/`. Like `dev_clip`, this is pipeline *output*, not ground truth,
and `src/excavator_cycles/` may reference neither it nor the labels.

## Regenerating

On a GPU machine (never the laptop):

    run.py track <83 s video> --out outputs/track/long_clip
    run.py features outputs/track/long_clip
