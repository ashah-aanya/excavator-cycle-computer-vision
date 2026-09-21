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
- Evidence, all committed under [`../evidence/`](../evidence/):

| File | What it shows |
|---|---|
| `spike-real-frames.jpg` | contact sheet: all 8 frames with boxes |
| `frame-digging.jpg` | bucket in the pile; box tight on the machine |
| `frame-arm-extended.jpg` | arm reaching over the truck; **box stretches to include the bucket** |
| `frame-dumping.jpg` | material released over the bed |
| `spike-metrics.json` | the numbers behind the table below |

Every box in those images was drawn from the pipeline's own `Detection` output by
`draw_detections()`; none is hand-placed. The same is required of the final
annotated video, so the drawing code is shared rather than written twice.

### Per-frame detail

| frame | score | box (x1,y1,x2,y2) | area |
|---|---|---|---|
| paper_p17_0_00 | 0.801 | (109, 0, 304, 265) | 20.3% |
| paper_p17_0_01 | 0.657 | (137, 1, 447, 184) | 22.1% |
| paper_p17_0_10 | 0.725 | (126, 1, 513, 189) | 28.2% |
| paper_p17_0_11 | 0.821 | (135, 0, 597, 191) | 34.1% |
| task_p1_0_00 | 0.786 | (85, 6, 212, 192) | 19.1% |
| task_p1_0_01 | 0.659 | (86, 6, 295, 134) | 21.7% |
| task_p1_0_10 | 0.693 | (98, 0, 356, 124) | 26.2% |
| task_p1_0_11 | 0.769 | (84, 0, 397, 125) | 32.0% |

Exactly one box per frame, every frame. Note the right edge (`x2`) growing from
304 to 597 while the left edge stays near 85-137: the box follows the arm as it
extends, anchored by the machine's body. That is the numeric version of the
claim the pictures make.

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

---

## Second run: harder footage, different site

Five frames from a **different site with a different machine** — a CAT excavator
and an articulated dump truck, camera further back, noticeable motion blur, tracks
partly buried. The closest proxy available for the two hidden videos, and a much
harder test than the Volvo frames above.

Each prompt was run separately, and the boxes below were drawn from the
coordinates the detector returned.

| | `excavator.` | `digger.` |
|---|---|---|
| Boxes per frame | **exactly 1** | 1–2 |
| Score range | **0.48 – 0.76** | 0.36 – 0.53 |
| Spurious box on the truck | never | 4 of 5 frames |

Evidence: [`../evidence/hard-frames-excavator.jpg`](../evidence/hard-frames-excavator.jpg)
and [`../evidence/hard-frames-digger.jpg`](../evidence/hard-frames-digger.jpg).

The `digger.` extra boxes — `(1138,640,1818,1012)`, `(1117,628,1838,1008)` — are the
dump truck every time. This confirms the prompt decision on footage that was never
used to make it.

### New risk found: the box is loose when the arm extends

See [`../evidence/hard-frame-loose-box.jpg`](../evidence/hard-frame-loose-box.jpg).
The detection is correct — `(539,177,1818,1010)`, score 0.74 — but its right edge
sits well past the bucket, over the truck's cab. 36% of the frame.

This matters because the box becomes SAM 2's prompt in stage 2. If the resulting
mask absorbs part of the truck, the bucket tip — defined as the farthest point of
the machine's region from its rotation centre — could land on the truck. Every
downstream measurement would then be wrong while still looking plausible.

Note *when* it happens: while the arm is extended over the truck, which is the
dumping phase. The most fragile boundary, again.

Three existing safeguards should catch it, and stage 2 must verify them
explicitly rather than assume them:

1. mask area stability — a mask that swallows a truck jumps in area
2. anchor-versus-mask agreement — the independent 1 Hz detection disagreeing with
   the propagated mask
3. the tip-jump filter — a tip teleporting onto a truck is not physically possible

**Stage 2 acceptance now includes:** on these frames, the mask must cover the
excavator and not the truck. If SAM 2 cannot separate them from a loose box, the
fallback is to shrink the prompt box toward the machine's core before prompting,
or to prompt with a point rather than a box.
