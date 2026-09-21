# Stage 1 findings — detector spike

**Date:** 2026-09-21
**Question:** does an open-vocabulary detector reliably find the excavator in this footage?
**Verdict: yes.** Stage 2 (tracking) proceeds as designed.

## What was tested

The target video was not yet available, so the spike ran on **8 real frames from the
same footage**, extracted from the figures in the task PDF and in Cheng et al. (2023):
the same Volvo excavator, same site, same camera setup. Same machine, same
conditions — the only thing missing is motion.

- Model: `IDEA-Research/grounding-dino-base`, transformers 5.17, torch 2.14, CPU/MPS
- Prompts: `"excavator."` and `"digger."`
- Evidence: [`../evidence/spike-real-frames.jpg`](../evidence/spike-real-frames.jpg)

## Results

| prompt | found | median score | box area | centre jump | gate |
|---|---|---|---|---|---|
| `excavator.` | **100%** | **0.75** | 0.24 of frame | 0.13 | pass |
| `digger.` | 100% | 0.64 | 0.24 of frame | 0.13 | pass |

Comfortably above every gate threshold: detection rate 100% against a 95% floor,
median score 0.75 against 0.40, and a box covering a quarter of the frame — the
right order of magnitude for a machine at this distance.

## What the pictures showed that the numbers did not

**1. The box includes the arm and the bucket.** This is the finding that matters
most, because the whole geometric derivation rests on it: the bucket tip is the
farthest point of the machine's region from its rotation centre. When the arm is
extended over the truck, the box stretches with it rather than clipping to the
machine's body. Had it clipped, the tip would have been wrong in exactly the frames
that decide the dumping boundary.

**2. `"digger."` also fires on the dump truck**, scoring 0.35–0.45 — always below
its score on the excavator, so picking the highest-scoring box still lands on the
right machine. But it is unnecessary risk on unseen footage where the truck might
be nearer the camera or better lit.

**Change made:** the default prompt is now `"excavator."` alone. `"digger."` remains
available via config for a video where the primary prompt underperforms.

## A bug this caught, which is the point of running it early

Converting BGR to RGB with a reversed NumPy slice (`image[:, :, ::-1]`) produces an
array with negative strides, which torch refuses to convert to a tensor. It raised
from deep inside the processor, and it would have passed every stub test. Fixed by
using an explicit colour conversion.

This is the failure class integration tests exist for: on the real video it would
have been indistinguishable from "the model doesn't work on this footage".

## Caveats — what this does *not* establish

- **Stills, not video.** Nothing here tests temporal stability, motion blur, or the
  frames where the arm is hidden behind the cab mid-swing. The centre-jump figure is
  meaningless for unrelated stills and is reported only for completeness.
- **Four distinct moments, not a full cycle.** The frames happen to span digging,
  hauling, dumping and swinging, which is fortunate, but 8 frames is not 600.
- **One site.** The hidden videos may differ in camera angle, machine colour, or
  scale.

The spike should be re-run on the real video when it arrives; the numbers here are
the prior, not the answer.

## Reproducing

```bash
uv sync --extra models
uv run run.py spike data/real_frames/tiles --detector grounding_dino --frames 8
uv run pytest -m integration          # 5 tests against the real model
```
