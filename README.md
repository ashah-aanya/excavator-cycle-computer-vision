# excavator-cycle-computer-vision

Measure excavator work-cycle phase durations from video, with a rule-based
computer-vision pipeline.

Given a video of an excavator loading material, the pipeline finds every
complete work cycle — digging, hauling, dumping, swinging — and reports the
average duration of each phase and of a full cycle, along with an annotated
video showing how it reached that answer.

**Status: end to end, with the cues known wrong.** Every stage is built and the
pipeline writes `answer.json` from a video in one command. `cycle_count` is correct
on the development clip; the four phase averages are **not**, because all four
coarse triggers fire outside the window that contains the transition they are
looking for. That gap is measured by a test rather than estimated -- see
`tests/test_onset_accuracy.py`, which carries the five onset errors and one strict
`xfail` that turns green when the cues are fixed.

See [`docs/where-things-live.md`](docs/where-things-live.md) for what is where and
why.

## Design

- [`docs/where-things-live.md`](docs/where-things-live.md) — **read first**: the
  branches, the three generations of measurement, what is live and what is not
- [`docs/pipeline-design.md`](docs/pipeline-design.md) — the original architecture
  document. Parts of it describe approaches since superseded; `where-things-live`
  says which.
- [`docs/state-detection-inventory.md`](docs/state-detection-inventory.md) — every
  phase-boundary cue tried, and what survived

Three decisions shape everything else:

1. **Detect as little as possible, derive as much as possible.** Only the
   excavator and the truck are detected. The bucket is segmented from a
   geometrically-derived seed, and the slew centre, the reach `L` and the cabin
   reference are all computed from the machine's own outline over time.
2. **Measure from bounding boxes.** Two earlier attempts recovered the arm's
   geometry directly — a fitted three-link chain, then dense optical flow — and
   both failed for the same reason: they tried to read a 3-D articulated motion
   out of a 2-D projection. The chain's "rigid" links varied 5.2–8.2× in length;
   the flow-derived house rotation correlated with the real slew at r = 0.025,
   because a machine slewing about a *vertical* axis translates sideways in
   projection rather than rotating in the image. A box is a weaker measurement
   that is strong enough.
3. **Nothing video-specific.** Times in seconds, distances as fractions of the
   machine's own reach, and brackets from Otsu splits of that video's own
   distributions. The frame rate, the truck box, the slew centre and `L` are all
   derived per video. The same code has to work on videos we have never seen.

## Setup

```bash
uv sync --group dev          # core pipeline + test tools
uv sync --extra models       # adds torch/transformers for the perception stage
```

## Usage

```bash
# THE DELIVERABLE: video in, answer.json out. Runs every stage in order.
uv run run.py run path/to/video.mp4

# Everything again except the GPU pass, reusing the cached masks. Seconds, not minutes.
uv run run.py run path/to/video.mp4 --reuse

# ...plus the diagnostic video and a per-cycle breakdown.
uv run run.py run path/to/video.mp4 --reuse --test
```

The stages exist separately because only one of them needs a GPU, and the other
three are worth re-running on a cached result in under a second:

```bash
# What are we working with?
uv run run.py probe path/to/video.mp4

# Stage 2: detect, segment the excavator and the bucket, cache the masks.
# The only stage that wants a GPU.
uv run run.py track path/to/video.mp4 --out outputs/track/mine

# Stage 3: derive the scene and every kinematic feature, and plot them.
uv run run.py features outputs/track/mine

# Stage 4: find the cycles and write answer.json.
uv run run.py cycles outputs/track/mine

# The annotated video.
uv run run.py render outputs/track/mine --scale 2
```

`track` and everything after it write to the directory you name:

| File | What it is |
|---|---|
| `answer.json` | the four phase averages, the cycle average and the cycle count |
| `features.png` | 13 feature panels against time — **look at this first** |
| `annotated.mp4` | the video with the masks, the boxes and the live values drawn on |
| `features.npz` | one row per sample; lengths in `L`, rates in `L`/second |
| `scene.json` | slew centre, reach `L`, truck box — all derived from this video |
| `masks.npz` | the excavator and bucket masks, run-length encoded |
| `track.json`, `run.json` | per-frame detections, plus config, git revision, versions, device, timing |

`run.py cycles <dir> --test` additionally writes `cycles.mp4`, which shades the
window pass 1 searched and marks where each cue fired, and prints the per-cycle
breakdown with the reason any cycle was excluded from the averages.

## Development

```bash
uv run pytest -q     # tests run on synthetic video; no weights, no GPU
uv run ruff check .
uv run ruff format .
```

Every tunable value lives in [`configs/default.yaml`](configs/default.yaml),
mirrored by `src/excavator_cycles/config.py`. `null` in that file means *derive
this from the data* — a recorded decision, not an unfilled blank.

`eval/` is firewalled from the pipeline: `eval/labels.json` holds the hand-made
ground truth and `eval/score.py` grades an `answer.json` against it. No module under
`src/excavator_cycles/` may reference either, and
`tests/test_eval_labels.py::test_pipeline_never_references_the_labels` enforces that
rather than trusting it.
