# `eval/` — ground truth, and the scorer that uses it

**Evaluation only. The pipeline must never read anything in this directory.**

The task spec forbids hardcoding video-specific values and forbids the pipeline
reading `answer.json`. These labels are the
answer for one video. They live outside `src/` so that the separation is
structural rather than a promise: nothing under `src/excavator_cycles/` imports
`eval/`, `eval/score.py` imports nothing from `excavator_cycles`, and `eval/` is
not on the package path. If you ever find yourself wanting to import from here
inside the pipeline, the answer is no.

## What the labels are

`labels.json` holds the phase boundaries for
`construction_excavator_cycle_duration_1.mp4`, **hand-decided by Aanya from
frame-by-frame inspection of the video**. They are not model output and were not
produced by this pipeline.

Two facts about the file that the labels depend on:

- **fps is 29.97396912419384**, not 30. Rounding to 30 costs ~0.1% on every
  duration, against a ±0.6 s grading tolerance.
- **886 frames decode** (indices 0–885). The container header claims 1102, so
  `cv2.CAP_PROP_FRAME_COUNT` is wrong about this file by ~20%.

## Convention

Boundaries are **instants between frames**, not frame indices: boundary `f` means
"immediately before frame `f`". A boundary may legitimately be ≥ 886, meaning
"after the final frame".

The four phase definitions come from the task spec. The phases **tile the cycle
with no gaps** — each phase ends immediately before the next begins — so only the
four onsets are labelled, and the four phase durations sum to the cycle duration
by construction.

| boundary | frame |
|---|---|
| digging begins | 125 |
| hauling begins | 322 |
| dumping begins | 559 |
| swinging begins | 691 |
| swinging ends / cycle ends | 880 |

`cycle_count = 1`. There is exactly one complete cycle; anything before f125 or
after f880 is an incomplete partial and is ignored.

| field | frames | seconds |
|---|---|---|
| digging | 197 | 6.572 |
| hauling | 237 | 7.907 |
| dumping | 132 | 4.404 |
| swinging | 189 | 6.305 |
| **cycle** | **755** | **25.188** |

(The cycle's exact value is 25.1885 s, reported as 25.188 so the four rounded
phases sum exactly to the rounded cycle.)

**Digging onset convention:** marked **+9 frames after the bucket's teeth first
touch the material**. First touch was observed at ~f116, hence f125.

## Known caveat — the onset offset is asymmetric

The cycle-closing onset at f880 uses a *larger* offset: ~+22 frames after its own
first touch at ~f858. The two ends of the cycle therefore do not use the same
convention. That ~13-frame difference (**~0.43 s** at 29.974 fps) is added to the
labelled `swinging` duration and to the labelled `average_cycle_duration_seconds`,
relative to a symmetric convention. `digging`, `hauling` and `dumping` are
unaffected, because an offset applied to both ends of a phase cancels.

Practically: ~0.43 s of the ±0.6 s budget is already spent on two of the five
fields. A pipeline that disagrees with `swinging` by ~0.4 s may be using a *more*
consistent convention than these labels do. Read a sub-0.5 s disagreement as
labelling noise, not as a defect.

## Running the scorer

```bash
uv run python eval/score.py outputs/answer.json
```

Options: `--labels PATH` (default `eval/labels.json`), `--tolerance SECONDS`
(default read from the labels, 0.6).

It prints a per-field table with expected, actual, signed error and PASS/FAIL,
then an overall verdict. **Exit code is 0 only if every field passes**, and 1
otherwise — including when `answer.json` is missing or malformed, which is
reported as a one-line message rather than a traceback.

`cycle_count` must match **exactly**; the ±0.6 s tolerance applies to the cycle
average and to each of the four phase averages.

```
field                                    expected     actual     error  result
------------------------------------------------------------------------------
cycle_count                                     1          1     exact  PASS
average_cycle_duration_seconds             25.188     25.400    +0.212  PASS
average_phase_duration_seconds.digging      6.572      6.100    -0.472  PASS
...
```

Standard library only, so it runs without the pipeline's dependencies installed.

Tests live in `tests/test_eval_labels.py` (the labels' arithmetic is
self-consistent) and `tests/test_eval_score.py` (the scorer's verdicts and exit
codes).

## Checking a cue: `check_cues.py`

`score.py` grades a finished `answer.json`. `check_cues.py` works one level lower,
on a single candidate cue, before anything goes into the pipeline.

A cue is a **shape in one feature**, like "2D speed: drop ends -> flat". The state
machine only looks for a phase once the previous phase has started, so a shape
that also appears earlier in the cycle does no harm. The checker asks the question
that matters: *scanning forward from the previous phase's labelled start, is the
first stretch with this shape the true start?*

```bash
# one cue, on the 83 s clip (three cycles)
uv run python eval/check_cues.py --phase dig --feature speed_2d --shape "drop ends -> flat"

# every feature x shape for a phase, ranked
uv run python eval/check_cues.py --phase dumping --search

# the same cue on the 29 s dev clip -- the one the cues were NOT studied on
uv run python eval/check_cues.py --clip dev --phase dig --feature speed_2d --shape "drop ends -> flat"

uv run python eval/check_cues.py --list-features
```

For each labelled onset it prints the first matching stretch, whether that
stretch contains the true start (within ±0.6 s), and how far the stretch's start
and centre are from it. **Exit code 0** when every cycle's first match contains
the start, **1** when one does not, **2** when the inputs cannot be read.

Shapes: at each sample, a line is fitted to the 2 s before and the 2 s after.
Each side is rising, falling or flat (it moves less than 8% of the feature's
p95–p5 range over 2 s). The pair names the shape: `peak`, `dip`,
`drop ends -> flat`, `flat -> starts rising`, `keeps rising` and so on, plus
`--detail "speeds up"` / `"slows down"` when the slope changes by 1.5×.

Two limits to keep in mind:

- The scan starts from the previous phase's **labelled** start. In the pipeline
  it starts from the **detected** one, so an error there carries forward.
  `--anchor-back 2` scans from two phases back, to see what happens when the
  previous phase is missed.
- A cue found by studying the 83 s clip will look better there than on a video
  it has not seen. Check it with `--clip dev` too.

It uses `labels_long_clip.json` (the 83 s clip, which shares its filename with
the dev clip but is a different video) and `tests/fixtures/long_clip/features.npz`.
Like `score.py`, it imports nothing from `excavator_cycles`; its derivative
re-implements the pipeline's (`rates.derivative`), and `tests/test_check_cues.py` holds the two equal.

## Checking the whole pipeline

`check_cues.py` scores one cue on paper. To check what the pipeline itself finds, run its
phase search on a clip's `features.npz` and compare with the labels:

```bash
uv run pytest tests/test_label_accuracy.py -v
```

That test finds the phase starts on both labelled clips through `excavator_cycles.starts`,
exactly as `run.py` does, and holds each labelled start's error to its recorded value (plus
0.05 s of slack) and its window to containing the label. It reads the labels; the pipeline
still may not name anything in `eval/`.

`tests/test_phase_starts.py` is the label-free companion: it holds the search to what the
original scripts returned before the method moved into `src/`.
