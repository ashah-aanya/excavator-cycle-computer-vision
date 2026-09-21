# excavator-cycle-computer-vision

Measure excavator work-cycle phase durations from video, with a rule-based
computer-vision pipeline.

Given a video of an excavator loading material, the pipeline finds every
complete work cycle — digging, hauling, dumping, swinging — and reports the
average duration of each phase and of a full cycle, along with an annotated
video showing how it reached that answer.

**Status: in development.** The perception stage is being validated; the design
is settled.

## Design

- [`docs/pipeline-design.md`](docs/pipeline-design.md) — architecture, model
  choice and the reasoning behind it, boundary definitions, failure handling
- [`docs/report-draft.md`](docs/report-draft.md) — the deliverable report

Two decisions shape everything else:

1. **Detect as little as possible, derive as much as possible.** Only the
   excavator is detected. The arm, the bucket, the dig and dump locations and
   the material surface are all computed from the machine's outline over time,
   because detectors are unreliable on object *parts* and on amorphous terrain.
2. **Nothing video-specific.** Times are in seconds, distances are fractions of
   the machine's own reach, and signal thresholds are multipliers of a
   percentile of that video's own signals. The same code has to work on videos
   we have never seen.

## Setup

```bash
uv sync --group dev          # core pipeline + test tools
uv sync --extra models       # adds torch/transformers for the perception stage
```

## Usage

```bash
# What are we working with?
uv run run.py probe path/to/video.mp4

# Stage 1: does the detector actually find the excavator in this footage?
uv run run.py spike path/to/video.mp4 --detector grounding_dino --detector owlv2
```

The spike writes to `outputs/spike/<video>/`:

| File | What it is |
|---|---|
| `contact_sheet.jpg` | every sampled frame with its boxes — **look at this first** |
| `frames/*.jpg` | the same frames individually |
| `metrics.json` | detection rate, scores, box stability, cross-model agreement, gate verdict |
| `run.json` | config, git revision, package versions, device, timing |

## Development

```bash
uv run pytest -q     # tests run on synthetic video; no weights, no GPU
uv run ruff check .
uv run ruff format .
```

Every tunable value lives in [`configs/default.yaml`](configs/default.yaml),
mirrored by `src/excavator_cycles/config.py`. A test asserts the two agree, so
they cannot drift apart. `null` in that file means *derive this from the data* —
a recorded decision, not an unfilled blank.
