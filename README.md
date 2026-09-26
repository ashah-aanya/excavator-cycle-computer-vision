# excavator-cycle-computer-vision

Measure excavator work-cycle phase durations from video, with a rule-based
computer-vision pipeline.

Given a video of an excavator loading material, the pipeline finds every
complete work cycle — digging, hauling, dumping, swinging — and reports the
average duration of each phase and of a full cycle, along with an annotated
video showing how it reached that answer.

**Status: in development.** Perception and the kinematic features are built and
verified; the state machine and the answer are not. See
[`docs/where-things-live.md`](docs/where-things-live.md) for what is where and
why, and [`docs/open-questions-for-the-state-machine.md`](docs/open-questions-for-the-state-machine.md)
for what remains to be decided.

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
# What are we working with?
uv run run.py probe path/to/video.mp4

# Stage 1+2: detect, segment the excavator and the bucket, cache the masks.
# The only stage that wants a GPU.
uv run run.py track path/to/video.mp4 --out outputs/track/mine

# Stage 3: derive the scene and every kinematic feature, and plot them.
uv run run.py features outputs/track/mine

# The annotated video.
uv run run.py render outputs/track/mine --scale 2
```

`track` and everything after it write to the directory you name:

| File | What it is |
|---|---|
| `features.png` | 13 feature panels against time — **look at this first** |
| `annotated.mp4` | the video with the masks, the boxes and the live values drawn on |
| `features.npz` | one row per sample; lengths in `L`, rates in `L`/second |
| `scene.json` | slew centre, reach `L`, truck box — all derived from this video |
| `masks.npz` | the excavator and bucket masks, run-length encoded |
| `track.json`, `run.json` | per-frame detections, plus config, git revision, versions, device, timing |

Not built yet: the state machine that turns these features into phases, and the
`answer.json` the task asks for.

## Development

```bash
uv run pytest -q     # tests run on synthetic video; no weights, no GPU
uv run ruff check .
uv run ruff format .
```

Every tunable value lives in [`configs/default.yaml`](configs/default.yaml),
mirrored by `src/excavator_cycles/config.py`. `null` in that file means *derive
this from the data* — a recorded decision, not an unfilled blank.

`reference/box-physics/` holds recovered exploratory code that detected all four
phase transitions inside tolerance. It is kept verbatim as the record of what was
verified, is excluded from lint and tests, and must not be imported: it hardcodes
one video's frame rate, truck box and sample count, and reads the ground truth to
score itself. Its README says what to port and what not to.
